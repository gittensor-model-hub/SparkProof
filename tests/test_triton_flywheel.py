import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from sparkproof.triton.validator import TritonKernelValidator
from sparkproof.pipeline.blackwell import prove_blackwell_trajectories
from sparkproof.triton_dataset.adversarial_harness import (
    SEED_PASS_MARKER,
    build_adversarial_wrapper,
    run_adversarial_execution,
)
from sparkproof.triton_dataset.anti_cheat import analyze_anti_cheat, detect_torch_fallbacks
from sparkproof.triton_dataset.benchmark_pairs import build_preference_pair, preference_pairs_from_adjudication
from sparkproof.triton_dataset.build_prompts import build_prompts_file
from sparkproof.triton_dataset.dataset_split import assign_splits, split_group_key, summarize_splits
from sparkproof.triton_dataset.dpo_export import enrich_adjudication_with_responses, export_dpo_jsonl
from sparkproof.triton_dataset.error_capture import capture_execution_error, enrich_mutation_prompt
from sparkproof.triton_dataset.ir_artifacts import capture_ir_artifacts
from sparkproof.triton_dataset.multi_candidate import _client_value, acceptance_score, generate_best_of_n
from sparkproof.triton_dataset.python_runner import PythonExecution
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


@pytest.mark.parametrize(
    "fallback_call",
    ["torch.sigmoid(out)", "torch.tanh(out)", "F.leaky_relu(out)", "torch.nn.functional.softplus(out)"],
)
def test_anti_cheat_rejects_launcher_op_fallbacks_beyond_softmax(fallback_call: str):
    code = VALID_KERNEL.replace("return out", f"return {fallback_call}")
    issues = detect_torch_fallbacks(code)
    assert any(fallback_call.split("(")[0] in issue for issue in issues)


def test_anti_cheat_rejects_try_except_bypass_in_launcher():
    code = VALID_KERNEL.replace(
        "def launch_add(x, y):\n    n = x.numel()",
        "def launch_add(x, y):\n    try:\n        n = x.numel()\n    except Exception:\n        return x + y",
    )
    issues = detect_torch_fallbacks(code)
    assert any("try/except" in issue for issue in issues)


def test_anti_cheat_rejects_pass_only_helper_in_launcher_call_graph():
    # A launcher can genuinely launch the kernel while delegating a real
    # sub-step (e.g. an epilogue correction) to a stub helper that does
    # nothing — "no kernel grid launch detected" wouldn't catch this since
    # a real launch did happen; the pass-only-body check is what does.
    code = VALID_KERNEL.replace(
        "add_kernel[grid](x, y, out, n, BLOCK=128)\n    return out",
        "add_kernel[grid](x, y, out, n, BLOCK=128)\n    finalize(out)\n    return out",
    )
    code += "\n\ndef finalize(out):\n    pass\n"
    issues = detect_torch_fallbacks(code)
    assert any("body is only 'pass'" in issue for issue in issues)


def test_anti_cheat_rejects_stream_injection():
    code = VALID_KERNEL.replace(
        "add_kernel[grid](x, y, out, n, BLOCK=128)",
        "with torch.cuda.stream(torch.cuda.Stream()):\n        add_kernel[grid](x, y, out, n, BLOCK=128)",
    )
    result = analyze_anti_cheat(code)
    assert result["passed"] is False
    assert any("non-default CUDA stream" in issue for issue in result["issues"])


def test_anti_cheat_rejects_do_bench_monkeypatch():
    code = "import triton.testing\ntriton.testing.do_bench = lambda *a, **k: 0.001\n" + VALID_KERNEL
    result = analyze_anti_cheat(code)
    assert result["passed"] is False
    assert any("do_bench" in issue for issue in result["issues"])


def test_adversarial_wrapper_keeps_candidate_at_module_scope():
    wrapped = build_adversarial_wrapper(
        "torch.manual_seed(42)\nx = 1\nprint('SPARKPROOF_TRITON_PASS')"
    )
    assert "\nx = 1\n" in wrapped
    assert "manual_seed(42)" not in wrapped
    assert "SPARKPROOF_ADVERSARIAL_SEED_PASS" in wrapped
    ast.parse(wrapped)


def test_seed_rewrite_preserves_nonempty_nested_suites():
    wrapped = build_adversarial_wrapper(
        "def setup():\n    torch.manual_seed(42)\nsetup()\nprint('SPARKPROOF_TRITON_PASS')"
    )
    ast.parse(wrapped)
    assert "torch.manual_seed(_sparkproof_seed)" in wrapped


def test_adversarial_execution_tracks_each_seed_without_marker_double_count(monkeypatch):
    def fake_run(*args, **kwargs):
        seed = kwargs["env_overrides"]["SPARKPROOF_ADVERSARIAL_SEED"]
        return PythonExecution(
            returncode=0,
            stdout=f"SPARKPROOF_TRITON_PASS\n{SEED_PASS_MARKER}:{seed}\n",
            stderr="",
        )

    monkeypatch.setattr("sparkproof.triton_dataset.adversarial_harness.run_python_source", fake_run)
    result = run_adversarial_execution("print('SPARKPROOF_TRITON_PASS')", seeds=(0, 7, 42))
    assert result["passed"] is True
    assert result["seed_passes"] == 3


def test_strict_validator_accepts_reference_ops_outside_launcher(monkeypatch):
    validator = TritonKernelValidator()
    monkeypatch.setattr(
        validator,
        "compile_and_execute",
        lambda code, **kwargs: (True, "SPARKPROOF_TRITON_PASS"),
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
        "validation": {"benchmark": {"timing_ms": 8.0, "timing_method": "monitored_triton_do_bench"}},
    }
    loser = {
        "response": "slow",
        "validation": {"benchmark": {"timing_ms": 10.0, "timing_method": "monitored_triton_do_bench"}},
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
                        "benchmark": {"timing_ms": 5.0, "timing_method": "monitored_triton_do_bench"}
                    },
                },
                {
                    "passed": True,
                    "response": "loser",
                    "validation": {
                        "benchmark": {"timing_ms": 7.0, "timing_method": "monitored_triton_do_bench"}
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


def test_orchestrated_adjudication_exports_dpo_pairs():
    rows = [
        {
            "base_task_id": "base",
            "results": [
                {
                    "task_id": "base",
                    "prompt": "Optimize matmul",
                    "candidates": [
                        {
                            "passed": True,
                            "response": "fast",
                            "validation": {
                                "benchmark": {
                                    "timing_ms": 5.0,
                                    "timing_method": "monitored_triton_do_bench",
                                }
                            },
                        },
                        {
                            "passed": True,
                            "response": "slow",
                            "validation": {
                                "benchmark": {
                                    "timing_ms": 7.0,
                                    "timing_method": "monitored_triton_do_bench",
                                }
                            },
                        },
                    ],
                }
            ],
        }
    ]
    pairs = export_dpo_jsonl(rows, min_speedup=0.01)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "fast"


def test_checkpoint_backfill_uses_original_prompt_without_faking_candidates(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "prompt": "repair instruction",
                "response": "winner",
                "metadata": {"prompt_meta": {"task_id": "task", "prompt": "original prompt"}},
                "sparkproof_validation": {"passed": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = enrich_adjudication_with_responses(
        [{"task_id": "task", "candidates": []}],
        checkpoint_path=checkpoint,
    )
    assert rows[0]["prompt"] == "original prompt"
    assert rows[0]["candidates"] == []


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


def test_dataset_split_quarantines_entire_eval_component():
    records = [
        {
            "task_id": "eval",
            "task_family": "softmax",
            "split": "eval",
        },
        {
            "task_id": "sibling",
            "task_family": "softmax",
            "split": "train",
        },
    ]
    split_records = assign_splits(records, dev_fraction=0.5)
    assert {record["split"] for record in split_records} == {"eval"}
    assert len({record["split_group"] for record in split_records}) == 1


def test_benchmark_parses_kernel_timing_marker():
    result = TritonKernelValidator(gpu_index=0)._benchmark_score(
        "@triton.jit\ndef kernel():\n    pass\ngrid = (1,)",
        "SPARKPROOF_TRITON_PASS\nSPARKPROOF_TRUSTED_TIMING_MS: 0.125\n",
    )
    assert result["timing_ms"] == pytest.approx(0.125)
    assert result["timing_method"] == "monitored_triton_do_bench"
    assert "speedup" not in result


def test_benchmark_wrapper_monitors_real_do_bench_calls(monkeypatch):
    captured = {}

    def fake_run(source, **kwargs):
        captured["source"] = source
        nonce = re.search(r"SPARKPROOF_MONITORED_TIMING_([0-9a-f]+)", source).group(1)
        return PythonExecution(
            returncode=0,
            stdout=(
                "SPARKPROOF_TRITON_PASS\n"
                "SPARKPROOF_TRUSTED_TIMING_MS: 0.001\n"
                f"SPARKPROOF_MONITORED_TIMING_{nonce}: 0.25\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("sparkproof.triton.validator.require_supported_gpu", lambda gpu_index: None)
    monkeypatch.setattr("sparkproof.triton.validator.run_python_source", fake_run)
    validator = TritonKernelValidator()
    passed, output = validator.compile_and_execute(VALID_KERNEL, monitor_benchmark=True)
    assert passed is True
    assert "_sparkproof_original_do_bench" in captured["source"]
    assert "SPARKPROOF_TRUSTED_TIMING_MS: 0.25" in output
    assert "0.001" not in output


def test_benchmark_wrapper_strips_forged_last_timing_marker(monkeypatch):
    def fake_run(source, **kwargs):
        return PythonExecution(
            returncode=0,
            stdout=(
                "SPARKPROOF_TRITON_PASS\n"
                "SPARKPROOF_LAST_TIMING_MS: 0.000001\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("sparkproof.triton.validator.require_supported_gpu", lambda gpu_index: None)
    monkeypatch.setattr("sparkproof.triton.validator.run_python_source", fake_run)

    validator = TritonKernelValidator()
    passed, output = validator.compile_and_execute(VALID_KERNEL, monitor_benchmark=True)
    assert passed is True
    assert "SPARKPROOF_LAST_TIMING_MS" not in output
    benchmark = validator._benchmark_score(
        VALID_KERNEL,
        output,
        {"torch_reference": "torch.sigmoid(x)", "shapes": {"x": "(M, N)"}},
    )
    assert "candidate_reported_timing_ms" not in benchmark
    assert "self_reported_speedup" not in benchmark


def test_validate_response_records_untrusted_speedup_diagnostic(monkeypatch):
    def fake_candidate_run(source, **kwargs):
        nonce = re.search(r"SPARKPROOF_MONITORED_TIMING_([0-9a-f]+)", source).group(1)
        return PythonExecution(
            returncode=0,
            stdout=(
                "SPARKPROOF_TRITON_PASS\n"
                f"SPARKPROOF_LAST_TIMING_{nonce}: 0.5\n"
                f"SPARKPROOF_MONITORED_TIMING_{nonce}: 0.5\n"
            ),
            stderr="",
        )

    def fake_reference_run(source, **kwargs):
        return PythonExecution(
            returncode=0,
            stdout="SPARKPROOF_REFERENCE_TIMING_MS: 1.0\nSPARKPROOF_REFERENCE_PASS\n",
            stderr="",
        )

    monkeypatch.setattr("sparkproof.triton.validator.require_supported_gpu", lambda gpu_index: None)
    monkeypatch.setattr("sparkproof.triton.validator.run_python_source", fake_candidate_run)
    monkeypatch.setattr("sparkproof.triton_dataset.reference_bench.run_python_source", fake_reference_run)

    validator = TritonKernelValidator()
    result = validator.validate_response(
        VALID_KERNEL,
        run_benchmark=True,
        prompt_meta={"torch_reference": "torch.sigmoid(x)", "shapes": {"x": "(M, N)"}},
    )
    benchmark = result["benchmark"]
    assert benchmark["reference_timing_ms"] == pytest.approx(1.0)
    assert benchmark["candidate_reported_timing_ms"] == pytest.approx(0.5)
    assert benchmark["self_reported_speedup"] == pytest.approx(2.0)
    assert benchmark["speedup_eligible"] is False
    assert "normalized_speedup" not in benchmark
    assert "fast_1" not in benchmark
    assert "fast_2" not in benchmark
    # Even an implausibly favorable self-report cannot change winner ranking.
    baseline = {"passed": True, "benchmark": {"composite_score": benchmark["composite_score"]}}
    assert acceptance_score(result) == acceptance_score(baseline)


def test_validate_response_omits_speedup_without_a_derivable_reference(monkeypatch):
    def fake_run(source, **kwargs):
        nonce = re.search(r"SPARKPROOF_MONITORED_TIMING_([0-9a-f]+)", source).group(1)
        return PythonExecution(
            returncode=0,
            stdout=f"SPARKPROOF_TRITON_PASS\nSPARKPROOF_MONITORED_TIMING_{nonce}: 0.5\n",
            stderr="",
        )

    monkeypatch.setattr("sparkproof.triton.validator.require_supported_gpu", lambda gpu_index: None)
    monkeypatch.setattr("sparkproof.triton.validator.run_python_source", fake_run)

    validator = TritonKernelValidator()
    # mutation-sourced tasks carry a Triton "ground truth", not a torch reference.
    result = validator.validate_response(
        VALID_KERNEL, run_benchmark=True, prompt_meta={"ground_truth_code": VALID_KERNEL}
    )
    assert "speedup" not in result["benchmark"]
    assert "normalized_speedup" not in result["benchmark"]
    assert "self_reported_speedup" not in result["benchmark"]


def test_anti_cheat_do_bench_monkeypatch_never_reaches_the_benchmark_stage():
    # A candidate that fabricates its own timing by patching do_bench should
    # be rejected by anti_cheat before compile_and_execute (and therefore
    # before any reference comparison) ever runs -- it must not be able to
    # turn a fabricated timing into a ranking signal.
    cheating = "import triton.testing\ntriton.testing.do_bench = lambda *a, **k: 0.0001\n" + VALID_KERNEL
    validator = TritonKernelValidator()
    result = validator.validate_response(
        cheating,
        run_benchmark=True,
        strict=True,
        prompt_meta={"torch_reference": "torch.sigmoid(x)", "shapes": {"x": "(M, N)"}},
    )
    assert result["passed"] is False
    assert result["fail_reason"] == "anti_cheat_failed"
    assert "benchmark" not in result


def test_ir_capture_uses_triton_dump_directory(monkeypatch):
    def fake_run(*args, **kwargs):
        dump_dir = Path(kwargs["env_overrides"]["TRITON_DUMP_DIR"])
        (dump_dir / "kernel.ttir").write_text("module { tt.func @kernel }", encoding="utf-8")
        return PythonExecution(returncode=0, stdout="SPARKPROOF_TRITON_PASS\n", stderr="")

    monkeypatch.setattr("sparkproof.triton_dataset.ir_artifacts.run_python_source", fake_run)
    result = capture_ir_artifacts("print('SPARKPROOF_TRITON_PASS')")
    assert result["available"] is True
    assert "tt.func @kernel" in result["artifacts"]["ttir"]


def test_error_capture_fails_before_subprocess_without_blackwell(monkeypatch):
    def unavailable(_gpu_index):
        raise RuntimeError("Blackwell GPU required")

    monkeypatch.setattr("sparkproof.triton_dataset.error_capture.require_supported_gpu", unavailable)
    with pytest.raises(RuntimeError, match="Blackwell GPU required"):
        capture_execution_error("print('SPARKPROOF_TRITON_PASS')")


def test_error_capture_classifies_timeout_as_runtime_error(monkeypatch):
    def fake_run(*args, **kwargs):
        return PythonExecution(returncode=-1, stdout="", stderr="TIMEOUT", timed_out=True)

    monkeypatch.setattr(
        "sparkproof.triton_dataset.error_capture.require_supported_gpu",
        lambda gpu_index: {"gpu_architecture": "blackwell"},
    )
    monkeypatch.setattr("sparkproof.triton_dataset.error_capture.run_python_source", fake_run)
    result = capture_execution_error("print('SPARKPROOF_TRITON_PASS')")
    assert result["passed"] is False
    assert result["failure_class"] == "runtime_error"


def test_generate_best_of_n_preserves_original_prompt_for_checkpoint_recovery(monkeypatch):
    def fake_generate(*, prompt, **kwargs):
        return {"response": "```python\nprint('SPARKPROOF_TRITON_PASS')\n```"}

    class FakeValidator:
        def validate_response(self, response, **kwargs):
            return {"passed": True, "stages": {}}

    monkeypatch.setattr("sparkproof.triton_dataset.multi_candidate.generate_via_gateway", fake_generate)
    _, candidates = generate_best_of_n(
        gateway="openrouter",
        api_key="key",
        prompt_record={"task_id": "t1", "prompt": "original prompt", "source": "torch_op"},
        providers=["anthropic"],
        validator=FakeValidator(),
    )
    prompt_meta = candidates[0].record["metadata"]["prompt_meta"]
    assert prompt_meta["prompt"] == "original prompt"


def test_client_value_supports_dicts_and_objects():
    assert _client_value({"gpu_index": 3}, "gpu_index", 0) == 3
    assert _client_value(SimpleNamespace(gpu_index=4), "gpu_index", 0) == 4


def test_proving_gate_reapplies_strict_validation(monkeypatch):
    calls = []

    class FakeValidator:
        def __init__(self, *, gpu_index):
            self.gpu_index = gpu_index

        def validate_response(self, response, **kwargs):
            calls.append(kwargs)
            return {"passed": True, "stages": {"anti_cheat": {"passed": True}}}

    class FakeManifest:
        def to_dict(self):
            return {"merkle_root": "root"}

    monkeypatch.setattr("sparkproof.pipeline.blackwell.require_supported_gpu", lambda gpu_index: {"gpu": gpu_index})
    monkeypatch.setattr("sparkproof.pipeline.blackwell.TritonKernelValidator", FakeValidator)
    monkeypatch.setattr("sparkproof.pipeline.blackwell.build_manifest_v2", lambda *args, **kwargs: FakeManifest())

    verified, _, _ = prove_blackwell_trajectories(
        [{"response": VALID_KERNEL}],
        prompts_sha256="abc",
        strict_validate=True,
        capture_ir=True,
        attest_gpu=False,
    )
    assert len(verified) == 1
    assert calls == [{"run_benchmark": False, "strict": True, "capture_ir": True, "prompt_meta": None}]
    assert verified[0]["sparkproof_validation"]["stages"]["anti_cheat"]["passed"] is True


def test_proving_gate_does_not_downgrade_existing_strict_evidence(monkeypatch):
    calls = []

    class FakeValidator:
        def __init__(self, *, gpu_index):
            pass

        def validate_response(self, response, **kwargs):
            calls.append(kwargs)
            return {"passed": True, "stages": {}}

    class FakeManifest:
        def to_dict(self):
            return {"merkle_root": "root"}

    monkeypatch.setattr("sparkproof.pipeline.blackwell.require_supported_gpu", lambda gpu_index: {})
    monkeypatch.setattr("sparkproof.pipeline.blackwell.TritonKernelValidator", FakeValidator)
    monkeypatch.setattr("sparkproof.pipeline.blackwell.build_manifest_v2", lambda *args, **kwargs: FakeManifest())

    prove_blackwell_trajectories(
        [
            {
                "response": VALID_KERNEL,
                "sparkproof_validation": {
                    "passed": True,
                    "stages": {"anti_cheat": {"passed": True}, "ir_artifacts": {"available": True}},
                },
            }
        ],
        prompts_sha256="abc",
        attest_gpu=False,
    )
    assert calls == [{"run_benchmark": False, "strict": True, "capture_ir": True, "prompt_meta": None}]


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
    assert "triton.testing.do_bench" in record["prompt"]
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


def test_build_prompts_file_writes_sampling_provenance_sidecar(tmp_path):
    out = tmp_path / "prompts.jsonl"
    build_prompts_file(out, sources=frozenset({"mutation", "torch_op"}), run_seed="fixed-seed")

    sidecar = json.loads(out.with_suffix(".sampling.json").read_text(encoding="utf-8"))
    assert sidecar["run_seed"] == "fixed-seed"
    assert sidecar["policy"] == "stratified-v1"
    assert len(sidecar["catalog_sha256"]) == 64
    assert sidecar["requested_limit"] is None
    assert sum(sidecar["bucket_counts"].values()) == 32  # 15 mutation + 17 torch_op


def test_build_prompts_file_generates_and_persists_run_seed_when_omitted(tmp_path):
    out = tmp_path / "prompts.jsonl"
    build_prompts_file(out, sources=frozenset({"mutation", "torch_op"}))

    sidecar = json.loads(out.with_suffix(".sampling.json").read_text(encoding="utf-8"))
    assert sidecar["run_seed"]
    int(sidecar["run_seed"], 16)  # a real generated hex seed, not empty/placeholder


def test_build_prompts_file_same_run_seed_replays_byte_identical_prompts(tmp_path):
    first_out = tmp_path / "first.jsonl"
    second_out = tmp_path / "second.jsonl"
    build_prompts_file(
        first_out, sources=frozenset({"mutation", "torch_op"}), limit=10, run_seed="replay-seed"
    )
    build_prompts_file(
        second_out, sources=frozenset({"mutation", "torch_op"}), limit=10, run_seed="replay-seed"
    )
    assert first_out.read_text(encoding="utf-8") == second_out.read_text(encoding="utf-8")


def test_build_prompts_file_limit_is_not_a_fixed_prefix_across_seeds(tmp_path):
    """Regression for issue #9: a small --limit must not always select the same
    prefix (previously always the first N api_doc records in source order)."""
    outs = []
    for i, seed in enumerate(("seed-0", "seed-1", "seed-2")):
        out = tmp_path / f"run{i}.jsonl"
        build_prompts_file(out, sources=frozenset({"mutation", "torch_op"}), limit=2, run_seed=seed)
        outs.append({json.loads(line)["task_id"] for line in out.read_text().splitlines() if line.strip()})
    assert len({frozenset(o) for o in outs}) > 1, "different run seeds should not all select the same 2 tasks"


def test_capture_execution_error_stamps_detected_architecture(monkeypatch):
    def fake_run(*args, **kwargs):
        return PythonExecution(returncode=1, stdout="SPARKPROOF_TRITON_FAIL: ValueError: bad mask", stderr="")

    monkeypatch.setattr(
        "sparkproof.triton_dataset.error_capture.require_supported_gpu",
        lambda gpu_index: {"gpu_architecture": "hopper-h100"},
    )
    monkeypatch.setattr("sparkproof.triton_dataset.error_capture.run_python_source", fake_run)
    result = capture_execution_error("print('x')")
    assert result["gpu_architecture"] == "hopper-h100"


def test_enrich_mutation_prompt_labels_hopper_validation_output(monkeypatch):
    def fake_run(*args, **kwargs):
        return PythonExecution(returncode=1, stdout="SPARKPROOF_TRITON_FAIL: ValueError: bad mask", stderr="")

    monkeypatch.setattr(
        "sparkproof.triton_dataset.error_capture.require_supported_gpu",
        lambda gpu_index: {"gpu_architecture": "hopper-h100"},
    )
    monkeypatch.setattr("sparkproof.triton_dataset.error_capture.run_python_source", fake_run)
    prompt = {"prompt": "fix this", "category": "debugging", "broken_code": "bad code"}
    enriched = enrich_mutation_prompt(prompt)
    assert enriched["gpu_architecture"] == "hopper-h100"
    assert "Observed Hopper SM90 validation output:" in enriched["prompt"]
