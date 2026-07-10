import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparkproof.triton.validator import TritonKernelValidator
from sparkproof.triton_dataset.adversarial_harness import (
    SEED_PASS_MARKER,
    build_adversarial_wrapper,
    run_adversarial_execution,
)
from sparkproof.triton_dataset.anti_cheat import analyze_anti_cheat, detect_torch_fallbacks
from sparkproof.triton_dataset.benchmark_pairs import build_preference_pair, preference_pairs_from_adjudication
from sparkproof.triton_dataset.build_prompts import build_prompts_file
from sparkproof.triton_dataset.dataset_split import assign_splits, split_group_key, summarize_splits
from sparkproof.triton_dataset.dpo_export import export_dpo_jsonl
from sparkproof.triton_dataset.error_capture import capture_execution_error
from sparkproof.triton_dataset.ir_artifacts import capture_ir_artifacts
from sparkproof.triton_dataset.multi_candidate import _client_value
from sparkproof.triton_dataset.prompt_templates import apply_prompt_template, wrap_prompt
from sparkproof.triton_dataset.torch_ops import iter_torch_translation_prompts


VALID_KERNEL = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def launch_add(x, y):
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=128)
    return out

x = torch.randn(1003, device="cuda")
y = torch.randn(1003, device="cuda")
out = launch_add(x, y)
torch.testing.assert_close(out, x + y)
print("SPARKPROOF_TRITON_PASS")
"""


CHEATING_KERNEL = """
import torch
import triton
import triton.language as tl

@triton.jit
def fake_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

def launch(x):
    n = x.numel()
    out = torch.empty_like(x)
    fake_kernel[(1,)](x, out, n, BLOCK=128)
    return torch.softmax(out, dim=-1)

x = torch.randn(128, device="cuda")
out = launch(x)
print("SPARKPROOF_TRITON_PASS")
"""


def test_anti_cheat_accepts_valid_launcher():
    result = analyze_anti_cheat(VALID_KERNEL)
    assert result["passed"] is True
    assert result["issues"] == []


def test_anti_cheat_rejects_torch_softmax_fallback():
    issues = detect_torch_fallbacks(CHEATING_KERNEL)
    assert any("torch.softmax" in issue for issue in issues)
    result = analyze_anti_cheat(CHEATING_KERNEL)
    assert result["passed"] is False


def test_anti_cheat_allows_torch_reference_in_top_level_test():
    code = VALID_KERNEL.replace(
        "torch.testing.assert_close(out, x + y)",
        "reference = torch.softmax(x + y, dim=-1)\ntorch.testing.assert_close(torch.softmax(out, dim=-1), reference)",
    )
    assert analyze_anti_cheat(code)["passed"] is True


def test_adversarial_wrapper_keeps_candidate_at_module_scope():
    wrapped = build_adversarial_wrapper(
        "torch.manual_seed(42)\nx = 1\nprint('SPARKPROOF_TRITON_PASS')"
    )
    assert "\nx = 1\n" in wrapped
    assert "manual_seed(42)" not in wrapped
    assert "SPARKPROOF_ADVERSARIAL_SEED_PASS" in wrapped
    ast.parse(wrapped)


def test_adversarial_execution_tracks_each_seed_without_marker_double_count(monkeypatch):
    def fake_run(*args, **kwargs):
        seed = kwargs["env"]["SPARKPROOF_ADVERSARIAL_SEED"]
        return SimpleNamespace(
            returncode=0,
            stdout=f"SPARKPROOF_TRITON_PASS\n{SEED_PASS_MARKER}:{seed}\n",
            stderr="",
        )

    monkeypatch.setattr("sparkproof.triton_dataset.adversarial_harness.subprocess.run", fake_run)
    result = run_adversarial_execution("print('SPARKPROOF_TRITON_PASS')", seeds=(0, 7, 42))
    assert result["passed"] is True
    assert result["seed_passes"] == 3


def test_strict_validator_accepts_reference_ops_outside_launcher(monkeypatch):
    validator = TritonKernelValidator()
    monkeypatch.setattr(
        validator,
        "compile_and_execute",
        lambda code: (True, "SPARKPROOF_TRITON_PASS"),
    )
    monkeypatch.setattr(
        "sparkproof.triton.validator.run_adversarial_execution",
        lambda code, gpu_index: {"passed": True, "seed_passes": 3, "seed_total": 3},
    )
    code = VALID_KERNEL.replace(
        "torch.testing.assert_close(out, x + y)",
        "reference = torch.softmax(x + y, dim=-1)\ntorch.testing.assert_close(torch.softmax(out, dim=-1), reference)",
    )
    assert validator.validate_response(code, strict=True)["passed"] is True


def test_preference_pair_requires_measurable_speedup():
    winner = {
        "response": "fast",
        "validation": {"benchmark": {"timing_ms": 8.0, "timing_method": "candidate_triton_do_bench"}},
    }
    loser = {
        "response": "slow",
        "validation": {"benchmark": {"timing_ms": 10.0, "timing_method": "candidate_triton_do_bench"}},
    }
    pair = build_preference_pair(
        task_id="matmul",
        prompt="Optimize matmul",
        winner=winner,
        loser=loser,
        min_speedup=0.03,
    )
    assert pair is not None
    assert pair["speedup"] == pytest.approx(0.2)

    noisy = build_preference_pair(
        task_id="matmul",
        prompt="Optimize matmul",
        winner=winner,
        loser=loser,
        min_speedup=0.5,
    )
    assert noisy is None


def test_preference_pairs_from_adjudication():
    rows = [
        {
            "task_id": "translate_matmul",
            "prompt": "Optimize matmul",
            "candidates": [
                {
                    "passed": True,
                    "response": "winner",
                    "validation": {
                        "benchmark": {"timing_ms": 5.0, "timing_method": "candidate_triton_do_bench"}
                    },
                },
                {
                    "passed": True,
                    "response": "loser",
                    "validation": {
                        "benchmark": {"timing_ms": 7.0, "timing_method": "candidate_triton_do_bench"}
                    },
                },
            ],
        }
    ]
    pairs = preference_pairs_from_adjudication(rows, min_speedup=0.01)
    assert len(pairs) == 1
    exported = export_dpo_jsonl(rows, min_speedup=0.01)
    assert exported[0]["prompt"] == "Optimize matmul"
    assert exported[0]["chosen"] == "winner"
    assert exported[0]["rejected"] == "loser"


def test_dataset_split_keeps_ancestry_groups_together():
    records = [
        {"task_id": "m1", "source": "mutation", "task_family": "softmax", "mutation_reason": "mask", "split": "train", "prompt": "p", "origin": "mutation", "category": "debugging"},
        {"task_id": "m2", "source": "mutation", "task_family": "softmax", "mutation_reason": "mask", "split": "train", "prompt": "p", "origin": "mutation", "category": "debugging"},
        {"task_id": "t1", "source": "torch_op", "task_family": "gelu", "torch_reference": "gelu", "split": "train", "prompt": "p", "origin": "torch_op", "category": "translation"},
    ]
    assert split_group_key(records[0]) == split_group_key(records[1])
    split_records = assign_splits(records, dev_fraction=0.5)
    splits = {record["task_id"]: record["split"] for record in split_records}
    assert splits["m1"] == splits["m2"]
    assert sum(summarize_splits(split_records).values()) == 3


def test_dataset_split_keeps_parent_and_descendant_together():
    records = [
        {
            "task_id": "parent",
            "source": "doc_tutorial",
            "origin": "doc_tutorial",
            "split": "train",
            "category": "kernel_generation",
            "prompt": "parent",
        },
        {
            "task_id": "child",
            "parent_id": "parent",
            "source": "self_evolution",
            "origin": "self_evolution",
            "split": "train",
            "category": "kernel_generation",
            "prompt": "child",
        },
    ]
    split_records = assign_splits(records, dev_fraction=0.5)
    assert len({record["split"] for record in split_records}) == 1
    assert len({record["split_group"] for record in split_records}) == 1


def test_benchmark_parses_kernel_timing_marker():
    result = TritonKernelValidator._benchmark_score(
        "@triton.jit\ndef kernel():\n    pass\ngrid = (1,)",
        "SPARKPROOF_TRITON_PASS\nSPARKPROOF_TRITON_TIMING_MS: 0.125\n",
    )
    assert result["timing_ms"] == pytest.approx(0.125)
    assert result["timing_method"] == "candidate_triton_do_bench"


def test_ir_capture_uses_triton_dump_directory(monkeypatch):
    def fake_run(*args, **kwargs):
        dump_dir = Path(kwargs["env"]["TRITON_DUMP_DIR"])
        (dump_dir / "kernel.ttir").write_text("module { tt.func @kernel }", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="SPARKPROOF_TRITON_PASS\n", stderr="")

    monkeypatch.setattr("sparkproof.triton_dataset.ir_artifacts.subprocess.run", fake_run)
    result = capture_ir_artifacts("print('SPARKPROOF_TRITON_PASS')")
    assert result["available"] is True
    assert "tt.func @kernel" in result["artifacts"]["ttir"]


def test_error_capture_fails_before_subprocess_without_blackwell(monkeypatch):
    def unavailable(_gpu_index):
        raise RuntimeError("Blackwell GPU required")

    monkeypatch.setattr("sparkproof.triton_dataset.error_capture.require_blackwell_gpu", unavailable)
    with pytest.raises(RuntimeError, match="Blackwell GPU required"):
        capture_execution_error("print('SPARKPROOF_TRITON_PASS')")


def test_client_value_supports_dicts_and_objects():
    assert _client_value({"gpu_index": 3}, "gpu_index", 0) == 3
    assert _client_value(SimpleNamespace(gpu_index=4), "gpu_index", 0) == 4


def test_prompt_templates_add_sections():
    record = apply_prompt_template(
        {
            "task_id": "translate_relu",
            "source": "torch_op",
            "origin": "torch_op",
            "split": "train",
            "category": "translation",
            "prompt": "Implement ReLU.",
        }
    )
    assert "## Design" in record["prompt"]
    assert "## Implementation" in record["prompt"]
    assert "SPARKPROOF_TRITON_PASS" in record["prompt"]
    assert "SPARKPROOF_TRITON_TIMING_MS" in record["prompt"]
    assert record["prompt_template"] == "translation:structured-v1"
    assert "Implement ReLU." in record["prompt"]
    assert wrap_prompt("body only", include_sections=False) == "body only"


def test_torch_shape_variants_are_optional():
    base = iter_torch_translation_prompts(include_shape_variants=False)
    expanded = iter_torch_translation_prompts(include_shape_variants=True)
    assert len(base) == 17
    assert len(expanded) == 17 + 17 * 2
    assert any("127" in record["prompt"] for record in expanded)


def test_enriched_prompt_build_has_unique_ids_and_split_metadata(tmp_path):
    out = tmp_path / "prompts.jsonl"
    count = build_prompts_file(
        out,
        sources=frozenset({"mutation", "torch_op"}),
        apply_templates=True,
        torch_shape_variants=True,
        assign_dev_splits=True,
        dev_fraction=0.2,
    )
    records = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert count == len(records) == 66
    assert len({record["task_id"] for record in records}) == count
    assert all(record.get("split_group") for record in records)
