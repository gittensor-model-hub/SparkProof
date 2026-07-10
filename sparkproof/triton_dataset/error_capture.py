"""Capture real compiler/runtime errors from broken kernels."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from sparkproof.blackwell.gpu import require_blackwell_gpu
from sparkproof.triton_dataset.failure_miner import classify_failure


def capture_execution_error(
    code: str,
    *,
    gpu_index: int = 0,
    timeout: int = 120,
) -> dict[str, Any]:
    # Fail the build clearly instead of recording "CUDA required" as if it were
    # a compiler/runtime defect in the mutated kernel.
    require_blackwell_gpu(gpu_index)
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapped)
        tmpfile = f.name
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        proc = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = proc.stdout + proc.stderr
        passed = "SPARKPROOF_TRITON_PASS" in proc.stdout and proc.returncode == 0
        validation = {
            "passed": passed,
            "fail_reason": None if passed else "compile_execute_failed",
            "stages": {"compile_execute": {"output_tail": output[-2500:]}},
        }
        return {
            "passed": passed,
            "output_tail": output[-2500:],
            "failure_class": classify_failure(validation) if not passed else "pass",
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output_tail": "TIMEOUT",
            "failure_class": "runtime_error",
            "returncode": -1,
        }
    finally:
        Path(tmpfile).unlink(missing_ok=True)


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
    out["prompt"] = (
        f"{prompt['prompt']}\n\n"
        "Observed Blackwell validation output:\n"
        f"```text\n{capture['output_tail'][-1500:]}\n```"
    )
    return out
