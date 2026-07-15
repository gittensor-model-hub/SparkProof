import json
from pathlib import Path

import pytest

from sparkproof.bundle import write_bundle
from sparkproof.hashing import sha256_file
from sparkproof.manifest import build_manifest_v2
from sparkproof.openrouter_request import build_chat_body, request_sha256
from sparkproof.policy import REQUIRED_REASONING_EFFORT
from sparkproof.triton_dataset.release_gate import run_release_gate
from tests.conftest_helpers import TEST_GEN_CONFIG


def _write_eval_corpus(root: Path) -> None:
    folder = root / "level1_basic"
    folder.mkdir(parents=True)
    (folder / "problem_1.yaml").write_text(
        "id: eval_1\nprompt: Eval prompt 1\ncategory: level_1\n", encoding="utf-8"
    )


def _verified_traj(task_id: str, *, prompt: str | None = None, response: str | None = None) -> dict:
    prompt = prompt or f"Write {task_id} Triton kernel"
    body = build_chat_body(
        provider="anthropic",
        prompt=prompt,
        system=None,
        max_tokens=TEST_GEN_CONFIG["max_tokens"],
        temperature=TEST_GEN_CONFIG["temperature"],
    )
    return {
        "prompt": prompt,
        "response": response or f"```python\nimport triton\nTASK_MARKER = {task_id!r}\n```",
        "provider": "anthropic",
        "model": "claude-fable-5",
        "gateway": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "request_url": "https://openrouter.ai/api/v1/chat/completions",
        "gateway_model": "anthropic/claude-fable-5",
        "openrouter_model": "anthropic/claude-fable-5",
        "request_sha256": request_sha256(body),
        "response_sha256": "b" * 64,
        "metadata": {
            "gateway_reasoning_effort": REQUIRED_REASONING_EFFORT,
            "openrouter_reasoning_effort": REQUIRED_REASONING_EFFORT,
            "openrouter_max_tokens": TEST_GEN_CONFIG["max_tokens"],
            "openrouter_temperature": TEST_GEN_CONFIG["temperature"],
            "prompt_meta": {"task_id": task_id, "origin": "torch_op", "split": "train", "category": task_id},
        },
        "sparkproof_validation": {
            "passed": True,
            "triton_version": "3.7.1",
            "code_sha256": "c" * 64,
            "stages": {"syntax": {"passed": True}, "compile_execute": {"passed": True}},
        },
    }


def _build_bundle(tmp_path: Path, trajectories: list[dict]) -> Path:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    gpu_profile = {
        "family": "blackwell",
        "profile": "workstation",
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "capability": [12, 0],
        "device_index": 0,
    }
    manifest = build_manifest_v2(
        trajectories,
        prompts_sha256=sha256_file(str(prompts)),
        gpu_profile=gpu_profile,
        raw_sample_count=len(trajectories),
        benchmark_enabled=False,
        openrouter_generation_config=TEST_GEN_CONFIG,
        attestation_hash="e" * 64,
    ).to_dict()
    bundle = tmp_path / "bundle"
    write_bundle(out_dir=bundle, trajectories=trajectories, manifest=manifest, prompts_path=prompts)
    (bundle / "trajectories_raw.jsonl").write_text((bundle / "trajectories.jsonl").read_text())
    (bundle / "validation_report.jsonl").write_text(
        "\n".join(
            json.dumps({"index": i, "validation": t["sparkproof_validation"]}) for i, t in enumerate(trajectories)
        )
        + "\n"
    )
    (bundle / "gpu_attestation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "environment": "REMOTE",
                "token": "tok",
                "claims": {},
                "gpu_profile": gpu_profile,
                "token_sha256": "e" * 64,
            }
        )
    )
    return bundle


def test_run_release_gate_reports_all_novel_with_no_registry(tmp_path: Path):
    _write_eval_corpus(tmp_path / "eval")
    bundle = _build_bundle(tmp_path, [_verified_traj("api_tl_load"), _verified_traj("api_tl_store")])

    manifest = run_release_gate(bundle, problems_dir=tmp_path / "eval")

    assert manifest["passed"] is True
    assert manifest["novelty"] == {
        "verified_rows": 2,
        "exact_duplicate_rows": 0,
        "near_duplicate_rows": 0,
        "novel_verified_rows": 2,
        "duplicate_task_ids": [],
    }
    report_path = bundle / "novelty_report.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == manifest["novelty"]


def test_run_release_gate_catches_within_bundle_duplicate(tmp_path: Path):
    _write_eval_corpus(tmp_path / "eval")
    same_prompt = "Write vector_add Triton kernel"
    same_response = "```python\nimport triton\n# vector_add\n```"
    bundle = _build_bundle(
        tmp_path,
        [
            _verified_traj("vector_add_a", prompt=same_prompt, response=same_response),
            _verified_traj("vector_add_b", prompt=same_prompt, response=same_response),
        ],
    )

    manifest = run_release_gate(bundle, problems_dir=tmp_path / "eval")

    # Duplicates are reported for reward accounting, never block the gate.
    assert manifest["passed"] is True
    assert manifest["novelty"]["verified_rows"] == 2
    assert manifest["novelty"]["exact_duplicate_rows"] == 1
    assert manifest["novelty"]["novel_verified_rows"] == 1
    assert manifest["novelty"]["duplicate_task_ids"] == ["vector_add_b"]


def test_run_release_gate_checks_registry_snapshot(tmp_path: Path):
    _write_eval_corpus(tmp_path / "eval")
    accepted_prompt = "Write vector_add Triton kernel"
    accepted_response = "```python\nimport triton\n# vector_add\n```"
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text(
        json.dumps(_verified_traj("vector_add_prior", prompt=accepted_prompt, response=accepted_response)) + "\n"
    )
    bundle = _build_bundle(
        tmp_path,
        [_verified_traj("vector_add_new", prompt=accepted_prompt, response=accepted_response)],
    )

    manifest = run_release_gate(bundle, problems_dir=tmp_path / "eval", registry_snapshot_path=registry_path)

    assert manifest["novelty"]["exact_duplicate_rows"] == 1
    assert manifest["novelty"]["novel_verified_rows"] == 0
    assert manifest["novelty"]["duplicate_task_ids"] == ["vector_add_new"]


def test_run_release_gate_excludes_blocked_rows_from_novelty(tmp_path: Path):
    _write_eval_corpus(tmp_path / "eval")
    blocked_row = _verified_traj("bad_row")
    blocked_row["metadata"]["prompt_meta"]["split"] = "eval"
    bundle = _build_bundle(tmp_path, [_verified_traj("good_row"), blocked_row])

    with pytest.raises(ValueError, match="release gate failed"):
        run_release_gate(bundle, problems_dir=tmp_path / "eval")

    manifest = json.loads((bundle / "dataset_manifest.json").read_text())
    assert manifest["novelty"]["verified_rows"] == 1


def test_run_release_gate_carries_hopper_gpu_architecture(tmp_path: Path):
    _write_eval_corpus(tmp_path / "eval")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    trajectories = [_verified_traj("api_tl_load")]
    gpu_profile = {
        "family": "hopper",
        "gpu_architecture": "hopper-h100",
        "name": "NVIDIA H100 80GB HBM3",
        "capability": [9, 0],
        "device_index": 0,
    }
    manifest = build_manifest_v2(
        trajectories,
        prompts_sha256=sha256_file(str(prompts)),
        gpu_profile=gpu_profile,
        raw_sample_count=len(trajectories),
        benchmark_enabled=False,
        openrouter_generation_config=TEST_GEN_CONFIG,
        attestation_hash="e" * 64,
    ).to_dict()
    bundle = tmp_path / "bundle"
    write_bundle(out_dir=bundle, trajectories=trajectories, manifest=manifest, prompts_path=prompts)
    (bundle / "trajectories_raw.jsonl").write_text((bundle / "trajectories.jsonl").read_text())
    (bundle / "validation_report.jsonl").write_text(
        json.dumps({"index": 0, "validation": trajectories[0]["sparkproof_validation"]}) + "\n"
    )
    (bundle / "gpu_attestation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "environment": "REMOTE",
                "token": "tok",
                "claims": {},
                "gpu_profile": gpu_profile,
                "token_sha256": "e" * 64,
            }
        )
    )

    dataset_manifest = run_release_gate(bundle, problems_dir=tmp_path / "eval")
    assert dataset_manifest["gpu_architecture"] == "hopper-h100"
    assert dataset_manifest["gpu_targets"] == ["hopper-h100"]
