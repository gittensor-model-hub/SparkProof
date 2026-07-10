import ast
import json
from pathlib import Path
import subprocess

import pytest

from sparkproof.triton_dataset.build_prompts import iter_all_prompts, write_prompts_jsonl
from sparkproof.triton_dataset.doc_api_pages import api_page_html_url
from sparkproof.triton_dataset.decontaminate import TritonDecontaminator, get_canonical_structure
from sparkproof.triton_dataset.eval_problems import iter_eval_problem_prompts
from sparkproof.triton_dataset.eval_harness import TritonBenchHarness
from sparkproof.triton_dataset.failure_miner import mine_failure_to_tasks
from sparkproof.triton_dataset.mutator import build_mutation_prompt, strip_boundary_mask
from sparkproof.triton_dataset.multi_candidate import acceptance_score, assign_tier
from sparkproof.triton_dataset.orchestrate import run_dataset_generation_step
from sparkproof.triton_dataset.reference_kernels import REFERENCE_KERNELS, VECTOR_ADD_VALID
from sparkproof.triton_dataset.release_gate import check_trajectory_row
from sparkproof.triton_dataset.schema import PromptValidationError, validate_prompt_record
from sparkproof.triton_dataset.self_evolve import evolve_parent


def _prompt(task_id: str = "task") -> dict:
    return {
        "task_id": task_id,
        "prompt": "Write a kernel",
        "system": "Be correct",
        "source": "torch_op",
        "origin": "torch_op",
        "split": "train",
        "category": "translation",
    }


def _write_eval_corpus(root: Path) -> None:
    for level in range(1, 5):
        folder = root / f"level{level}_{['basic', 'intermediate', 'advanced', 'expert'][level - 1]}"
        folder.mkdir(parents=True)
        (folder / f"problem_{level}.yaml").write_text(
            f"id: eval_{level}\nprompt: Eval prompt {level}\ncategory: level_{level}\n",
            encoding="utf-8",
        )


def test_eval_loader_covers_all_levels(tmp_path: Path):
    _write_eval_corpus(tmp_path)
    records = iter_eval_problem_prompts(problems_dir=tmp_path)
    assert {record["task_id"] for record in records} == {"eval_1", "eval_2", "eval_3", "eval_4"}
    assert all(record["origin"] == "tritonbench" and record["split"] == "eval" for record in records)


def test_strict_decontaminator_rejects_missing_eval_corpus(tmp_path: Path):
    with pytest.raises(RuntimeError, match="requires a TritonBench problem corpus"):
        TritonDecontaminator(problems_dir=tmp_path / "missing", require_eval_corpus=True)


def test_strict_doc_build_rejects_registry_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="incomplete pinned Triton documentation corpus"):
        list(
            iter_all_prompts(
                include_sources=frozenset({"api_doc"}),
                auto_fetch_docs=False,
                strict_docs=True,
            )
        )


def test_remote_api_enrichment_requires_pinned_mirror(monkeypatch):
    monkeypatch.delenv("SPARKPROOF_TRITON_API_PAGE_BASE", raising=False)
    with pytest.raises(ValueError, match="version-pinned mirror"):
        api_page_html_url("dot")


def test_release_gate_requires_positive_validation():
    decon = TritonDecontaminator()
    trajectory = {
        "response": "print('valid nonempty code')",
        "metadata": {"prompt_meta": _prompt()},
    }
    assert "missing or failed sparkproof validation" in check_trajectory_row(trajectory, decon)


def test_ast_fingerprint_preserves_kernel_logic():
    add = "@triton.jit\ndef k(x, y):\n    return tl.load(x) + tl.load(y)\n"
    multiply = "@triton.jit\ndef q(a, b):\n    return tl.load(a) * tl.load(b)\n"
    renamed_add = "@triton.jit\ndef q(a, b):\n    return tl.load(a) + tl.load(b)\n"
    assert get_canonical_structure(add) == get_canonical_structure(renamed_add)
    assert get_canonical_structure(add) != get_canonical_structure(multiply)


def test_mutation_is_deterministic_and_changes_source():
    first = build_mutation_prompt(task_id="vector_add", valid_kernel=VECTOR_ADD_VALID)
    second = build_mutation_prompt(task_id="vector_add", valid_kernel=VECTOR_ADD_VALID)
    assert first["broken_code"] == second["broken_code"]
    assert first["broken_code"] != VECTOR_ADD_VALID


@pytest.mark.parametrize("name,code", REFERENCE_KERNELS.items())
def test_mask_mutation_is_syntax_safe(name: str, code: str):
    mutated, _ = strip_boundary_mask(code)
    ast.parse(mutated)
    assert mutated != code, name
    assert "mask=" not in mutated


def test_explicit_noop_mutation_is_rejected():
    def noop(code: str) -> tuple[str, str]:
        return code, "did nothing"

    with pytest.raises(ValueError, match="no applicable"):
        build_mutation_prompt(task_id="noop", valid_kernel=VECTOR_ADD_VALID, mutator=noop)


def test_self_evolution_is_reproducible_without_oracle():
    parent = _prompt()
    first = evolve_parent(parent, depth=3)
    second = evolve_parent(parent, depth=3)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]
    assert first


def test_failure_mining_uses_task_family_and_distinct_variants():
    tasks = mine_failure_to_tasks(
        {
            "task_id": "failed",
            "task_origin": "torch_op",
            "split": "train",
            "failure_class": "tail_mask_failure",
            "task_family": "softmax",
            "tags": ["translation", "softmax"],
        }
    )
    assert len(tasks) == 2
    assert all("softmax" in task["prompt"] for task in tasks)
    assert tasks[0]["prompt"] != tasks[1]["prompt"]


def test_candidate_scoring_and_tiers():
    passing = {"passed": True}
    assert acceptance_score(passing, output_tokens=100) == 109.0
    assert assign_tier(passing) == "silver"
    assert assign_tier(passing, repairs_used=1) == "repair"
    assert assign_tier({"passed": False}) == "reject"


def test_eval_harness_timeout_is_reported(tmp_path: Path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="tritonbench", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    report = TritonBenchHarness(bench_root=tmp_path).run_eval_cycle(
        endpoint="http://localhost/v1",
        model_name="model",
        out_path=tmp_path / "report.json",
        timeout_seconds=1,
    )
    assert report == {"status": "error", "message": "TritonBench timed out after 1s"}


def test_prompt_schema_and_atomic_duplicate_rejection(tmp_path: Path):
    with pytest.raises(PromptValidationError):
        validate_prompt_record({"prompt": "missing fields"})

    out = tmp_path / "prompts.jsonl"
    out.write_text("original\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task_id"):
        write_prompts_jsonl(out, [_prompt("same"), _prompt("same")])
    assert out.read_text(encoding="utf-8") == "original\n"

    assert json.loads(json.dumps(validate_prompt_record(_prompt())))["task_id"] == "task"


def test_orchestration_runs_base_and_children_and_returns_raw_trajectories(monkeypatch):
    child = {**_prompt("child"), "parent_id": "task"}
    monkeypatch.setattr(
        "sparkproof.triton_dataset.orchestrate.evolve_parent",
        lambda task, depth: [child],
    )

    def fake_generate(task, **kwargs):
        trajectory = {
            "provider": "openai",
            "prompt": task["prompt"],
            "response": "print('SPARKPROOF_TRITON_PASS')",
            "metadata": {"prompt_meta": task},
            "sparkproof_validation": {"passed": True},
        }
        return {
            "passed": True,
            "response": trajectory["response"],
            "trajectory": trajectory,
            "tier": "silver",
        }

    monkeypatch.setattr(
        "sparkproof.triton_dataset.orchestrate.generate_best_candidate",
        fake_generate,
    )
    result = run_dataset_generation_step(
        _prompt(),
        client={},
        validator=None,
        decontaminator=TritonDecontaminator(),
        evolve_depth=1,
    )
    assert [row["task_id"] for row in result["results"]] == ["task", "child"]
    assert all("trajectory" in row and "response" in row["trajectory"] for row in result["results"])
