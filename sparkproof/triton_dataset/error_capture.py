"""Capture real compiler/runtime errors from broken kernels."""

from __future__ import annotations

from typing import Any

from sparkproof.gpu.architecture import require_supported_gpu, sm_label
from sparkproof.triton_dataset.failure_miner import classify_failure
from sparkproof.triton_dataset.python_runner import run_python_source


def capture_execution_error(
    code: str,
    *,
    gpu_index: int = 0,
    timeout: int = 120,
) -> dict[str, Any]:
    # Fail the build clearly instead of recording "CUDA required" as if it were
    # a compiler/runtime defect in the mutated kernel.
    gpu_profile = require_supported_gpu(gpu_index)
    wrapped = f"""
import torch
import triton
import triton.language as tl
import sys

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

try:
{chr(10).join("    " + line for line in code.splitlines())}
    print("SPARKPROOF_TRITON_PASS")
except Exception as e:
    print(f"SPARKPROOF_TRITON_FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
    execution = run_python_source(wrapped, gpu_index=gpu_index, timeout=timeout)
    if execution.timed_out:
        return {
            "passed": False,
            "output_tail": "TIMEOUT",
            "failure_class": "runtime_error",
            "returncode": execution.returncode,
            "gpu_architecture": gpu_profile["gpu_architecture"],
        }
    output = execution.output
    passed = "SPARKPROOF_TRITON_PASS" in execution.stdout and execution.returncode == 0
    validation = {
        "passed": passed,
        "fail_reason": None if passed else "compile_execute_failed",
        "stages": {"compile_execute": {"output_tail": output[-2500:]}},
    }
    return {
        "passed": passed,
        "output_tail": output[-2500:],
        "failure_class": classify_failure(validation) if not passed else "pass",
        "returncode": execution.returncode,
        "gpu_architecture": gpu_profile["gpu_architecture"],
    }


def enrich_mutation_prompt(prompt: dict[str, Any], *, gpu_index: int = 0) -> dict[str, Any]:
    """Attach a real execution error to a mutation debugging prompt when possible."""
    broken = prompt.get("broken_code")
    if not broken or prompt.get("category") != "debugging":
        return prompt

    capture = capture_execution_error(broken, gpu_index=gpu_index)
    if capture["passed"]:
        return prompt

    out = dict(prompt)
    out["captured_error"] = capture["output_tail"]
    out["captured_failure_class"] = capture["failure_class"]
    out["gpu_architecture"] = capture["gpu_architecture"]
    out["prompt"] = (
        f"{prompt['prompt']}\n\n"
        f"Observed {sm_label(capture['gpu_architecture'])} validation output:\n"
        f"```text\n{capture['output_tail'][-1500:]}\n```"
    )
    return out
