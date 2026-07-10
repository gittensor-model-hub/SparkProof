"""Validate teacher Triton kernel responses on Blackwell (Triton 3.7.1)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from sparkproof.blackwell.gpu import require_blackwell_gpu

PASS_MARKER = "SPARKPROOF_TRITON_PASS"
TRITON_VERSION = "3.7.1"


class TritonKernelValidator:
    def __init__(self, *, gpu_index: int = 0, triton_version: str = TRITON_VERSION):
        self.gpu_index = gpu_index
        self.version = triton_version
        self.deprecated_apis = ["tl.make_block_ptr", "tl.advance"]

    def extract_code(self, response: str) -> str:
        for pattern in (r"```python\n(.*?)```", r"```\n(.*?)```"):
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return "\n\n".join(matches)
        return response

    def check_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def check_triton_api(self, code: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "modern": True,
            "issues": [],
            "features_used": [],
            "deprecated_used": [],
        }
        for dep in self.deprecated_apis:
            if dep in code:
                result["deprecated_used"].append(dep)
                result["issues"].append(f"Uses deprecated API: {dep}")
        for pattern, msg in {
            "@triton.jit": "Missing @triton.jit decorator",
            "tl.program_id": "Missing tl.program_id",
        }.items():
            if pattern not in code:
                result["issues"].append(msg)
                result["modern"] = False
        if result["deprecated_used"]:
            result["modern"] = False
        return result

    def compile_and_execute(self, code: str, timeout: int = 120) -> tuple[bool, str]:
        require_blackwell_gpu(self.gpu_index)
        wrapped = f"""
import torch
import triton
import triton.language as tl
import sys

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

try:
{self._indent(code, 4)}
    print("{PASS_MARKER}")
except Exception as e:
    print(f"SPARKPROOF_TRITON_FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped)
            tmpfile = f.name
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
            env["TRITON_PRINT_AUTOTUNING"] = "0"
            proc = subprocess.run(
                [sys.executable, tmpfile],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            output = proc.stdout + proc.stderr
            return PASS_MARKER in proc.stdout, output
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        finally:
            Path(tmpfile).unlink(missing_ok=True)

    def validate_response(
        self,
        response: str,
        *,
        run_benchmark: bool = False,
    ) -> dict[str, Any]:
        code = self.extract_code(response)
        stages: dict[str, Any] = {}

        syntax_ok = self.check_syntax(code)
        stages["syntax"] = {"passed": syntax_ok}
        if not syntax_ok:
            return self._result(False, code, stages, "syntax_error")

        api = self.check_triton_api(code)
        stages["triton_api"] = {"passed": api["modern"], "details": api}
        if not api["modern"]:
            return self._result(False, code, stages, "triton_api")

        exec_ok, exec_output = self.compile_and_execute(code)
        stages["compile_execute"] = {"passed": exec_ok, "output_tail": exec_output[-2000:]}
        if not exec_ok:
            return self._result(False, code, stages, "compile_execute_failed")

        benchmark: dict[str, Any] | None = None
        if run_benchmark:
            benchmark = self._benchmark_score(code, exec_output)
            stages["benchmark"] = benchmark
            if benchmark["composite_score"] < 0.25:
                return self._result(False, code, stages, "benchmark_below_floor")

        return self._result(True, code, stages, None, benchmark=benchmark)

    @staticmethod
    def _result(
        passed: bool,
        code: str,
        stages: dict[str, Any],
        fail_reason: str | None,
        *,
        benchmark: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "passed": passed,
            "triton_version": TRITON_VERSION,
            "stages": stages,
            "code_sha256": __import__("hashlib").sha256(code.encode()).hexdigest(),
        }
        if fail_reason:
            out["fail_reason"] = fail_reason
        if benchmark is not None:
            out["benchmark"] = benchmark
        return out

    @staticmethod
    def _benchmark_score(code: str, exec_output: str) -> dict[str, Any]:
        """Lightweight static+binary pass score (not full TritonBench perf harness)."""
        checks = {
            "correctness": 1.0 if PASS_MARKER in exec_output or "allclose" in exec_output.lower() else 0.5,
            "autotune": 1.0 if "@triton.autotune" in code else 0.0,
            "block_tiling": 1.0 if re.search(r"BLOCK_[MNK]\s*[=:]\s*\d+", code) else 0.0,
            "launch_grid": 1.0 if "grid" in code else 0.0,
            "blackwell_hint": 1.0
            if any(h in code.lower() for h in ("blackwell", "sm_120", "sm120", "rtx pro"))
            else 0.0,
        }
        composite = sum(checks.values()) / len(checks)
        return {"checks": checks, "composite_score": composite}

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n"))
