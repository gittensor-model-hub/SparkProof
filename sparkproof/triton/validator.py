"""Validate teacher Triton kernel responses on Blackwell (Triton 3.7.1)."""

from __future__ import annotations

import ast
import re
import secrets
from typing import Any

from sparkproof.blackwell.gpu import require_blackwell_gpu
from sparkproof.triton_dataset.adversarial_harness import run_adversarial_execution
from sparkproof.triton_dataset.anti_cheat import analyze_anti_cheat
from sparkproof.triton_dataset.ir_artifacts import capture_ir_artifacts
from sparkproof.triton_dataset.python_runner import run_python_source
from sparkproof.triton_dataset.reference_bench import benchmark_reference

PASS_MARKER = "SPARKPROOF_TRITON_PASS"
TIMING_MARKER_RE = re.compile(r"SPARKPROOF_TRUSTED_TIMING_MS\s*[:=]\s*(\d+(?:\.\d+)?)")
LAST_TIMING_MARKER_RE = re.compile(r"SPARKPROOF_LAST_TIMING_MS\s*[:=]\s*(\d+(?:\.\d+)?)")
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

    def compile_and_execute(
        self,
        code: str,
        timeout: int = 120,
        *,
        monitor_benchmark: bool = False,
    ) -> tuple[bool, str]:
        require_blackwell_gpu(self.gpu_index)
        benchmark_setup = ""
        benchmark_report = ""
        timing_nonce = secrets.token_hex(16)
        if monitor_benchmark:
            benchmark_setup = """
import numbers
import triton.testing
_sparkproof_timings = []
_sparkproof_original_do_bench = triton.testing.do_bench
def _sparkproof_monitored_do_bench(*args, **kwargs):
    result = _sparkproof_original_do_bench(*args, **kwargs)
    if isinstance(result, numbers.Real):
        _sparkproof_timings.append(float(result))
    return result
triton.testing.do_bench = _sparkproof_monitored_do_bench
"""
            benchmark_report = f"""
    if _sparkproof_timings:
        print(f"SPARKPROOF_LAST_TIMING_{timing_nonce}: {{_sparkproof_timings[-1]}}")
        _sparkproof_timings.sort()
        _sparkproof_median = _sparkproof_timings[len(_sparkproof_timings) // 2]
        print(f"SPARKPROOF_MONITORED_TIMING_{timing_nonce}: {{_sparkproof_median}}")
"""
        wrapped = f"""
import torch
import triton
import triton.language as tl
import sys
{benchmark_setup}

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

try:
{self._indent(code, 4)}
{benchmark_report}
    print("{PASS_MARKER}")
except Exception as e:
    print(f"SPARKPROOF_TRITON_FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
        execution = run_python_source(
            wrapped,
            gpu_index=self.gpu_index,
            timeout=timeout,
            env_overrides={"TRITON_PRINT_AUTOTUNING": "0"},
        )
        # Candidate stdout is untrusted. Remove both public marker forms before
        # appending values recovered from nonce-bound wrapper markers below.
        # Otherwise a candidate can print a forged generic marker without ever
        # calling do_bench.
        output = LAST_TIMING_MARKER_RE.sub("", TIMING_MARKER_RE.sub("", execution.output))
        if monitor_benchmark:
            monitored_pattern = re.compile(
                rf"SPARKPROOF_MONITORED_TIMING_{timing_nonce}\s*:\s*(\d+(?:\.\d+)?)"
            )
            monitored = monitored_pattern.findall(execution.output)
            if monitored:
                output += f"\nSPARKPROOF_TRUSTED_TIMING_MS: {monitored[-1]}\n"
            last_pattern = re.compile(rf"SPARKPROOF_LAST_TIMING_{timing_nonce}\s*:\s*(\d+(?:\.\d+)?)")
            last = last_pattern.findall(execution.output)
            if last:
                output += f"\nSPARKPROOF_LAST_TIMING_MS: {last[-1]}\n"
        return (
            execution.returncode == 0 and PASS_MARKER in execution.stdout,
            output,
        )

    def validate_response(
        self,
        response: str,
        *,
        run_benchmark: bool = False,
        strict: bool = False,
        capture_ir: bool = False,
        prompt_meta: dict[str, Any] | None = None,
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

        if strict:
            anti_cheat = analyze_anti_cheat(code)
            stages["anti_cheat"] = anti_cheat
            if not anti_cheat["passed"]:
                return self._result(False, code, stages, "anti_cheat_failed")

        exec_ok, exec_output = self.compile_and_execute(
            code,
            monitor_benchmark=run_benchmark,
        )
        stages["compile_execute"] = {"passed": exec_ok, "output_tail": exec_output[-2000:]}
        if not exec_ok:
            return self._result(False, code, stages, "compile_execute_failed")

        if strict:
            adversarial = run_adversarial_execution(code, gpu_index=self.gpu_index)
            stages["adversarial"] = adversarial
            if not adversarial["passed"]:
                return self._result(False, code, stages, "adversarial_failed")

        benchmark: dict[str, Any] | None = None
        if run_benchmark:
            benchmark = self._benchmark_score(code, exec_output, prompt_meta)
            stages["benchmark"] = benchmark
            if benchmark["composite_score"] < 0.25:
                return self._result(False, code, stages, "benchmark_below_floor")

        if capture_ir:
            ir = capture_ir_artifacts(code, gpu_index=self.gpu_index)
            stages["ir_artifacts"] = ir

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

    def _benchmark_score(
        self, code: str, exec_output: str, prompt_meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Score structure and record diagnostic ``do_bench`` timings.

        The candidate controls the callable passed to do_bench, its inputs, and
        its dtype. Consequently this is not a trusted KernelBench ``fast_p``
        measurement and must not affect candidate ranking. A future trusted
        speedup metric must have the harness invoke the launcher, verify its
        output, and time it on harness-owned canonical inputs.
        """
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
        out: dict[str, Any] = {"checks": checks, "composite_score": composite}
        timings = [float(value) for value in TIMING_MARKER_RE.findall(exec_output)]
        if timings:
            timings.sort()
            out["timing_ms"] = timings[len(timings) // 2]
            out["timing_samples"] = len(timings)
            out["timing_method"] = "monitored_triton_do_bench"

        # The last monitored call is only a self-reported diagnostic. The
        # nonce-bound wrapper proves that do_bench returned this value, but it
        # cannot prove which callable, shape, dtype, or output was measured.
        last_timings = [float(value) for value in LAST_TIMING_MARKER_RE.findall(exec_output)]
        candidate_ms = last_timings[-1] if last_timings else None
        if candidate_ms:
            out["candidate_reported_timing_ms"] = candidate_ms
            reference_ms = benchmark_reference(prompt_meta, gpu_index=self.gpu_index)
            if reference_ms is not None:
                out["reference_timing_ms"] = reference_ms
                out["self_reported_speedup"] = reference_ms / candidate_ms
                out["speedup_eligible"] = False
                out["speedup_ineligible_reason"] = (
                    "candidate controls benchmark callable/inputs/dtype; "
                    "canonical-shape correctness is not harness-verified"
                )
        return out

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n"))
