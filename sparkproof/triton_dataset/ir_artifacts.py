"""Optional Triton IR artifact capture (TTIR/TTGIR) for analysis."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from sparkproof.triton_dataset.python_runner import run_python_source


def capture_ir_artifacts(
    code: str,
    *,
    gpu_index: int = 0,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute candidate code and collect files emitted by Triton's dump hook."""
    dump_dir = Path(tempfile.mkdtemp(prefix="sparkproof-triton-ir-"))
    try:
        execution = run_python_source(
            code,
            gpu_index=gpu_index,
            timeout=timeout,
            env_overrides={
                "TRITON_KERNEL_DUMP": "1",
                "TRITON_ALWAYS_COMPILE": "1",
                "TRITON_DUMP_DIR": str(dump_dir),
                "TRITON_PRINT_AUTOTUNING": "0",
            },
        )
        artifacts: dict[str, str] = {}
        for suffix in ("ttir", "ttgir", "ptx"):
            matches = sorted(dump_dir.rglob(f"*.{suffix}"))
            if matches:
                # Keep bundle metadata bounded while retaining the generated IR.
                artifacts[suffix] = "\n\n".join(
                    path.read_text(encoding="utf-8", errors="replace") for path in matches
                )[-250_000:]
        return {
            "available": bool(artifacts),
            "artifacts": artifacts,
            "output_tail": execution.output[-1500:],
            "returncode": execution.returncode,
        }
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def ncu_available() -> bool:
    return shutil.which("ncu") is not None
