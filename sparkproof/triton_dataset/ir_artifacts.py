"""Optional Triton IR artifact capture (TTIR/TTGIR) for analysis."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def capture_ir_artifacts(
    code: str,
    *,
    gpu_index: int = 0,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute candidate code and collect files emitted by Triton's dump hook."""
    dump_dir = Path(tempfile.mkdtemp(prefix="sparkproof-triton-ir-"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["TRITON_KERNEL_DUMP"] = "1"
        env["TRITON_ALWAYS_COMPILE"] = "1"
        env["TRITON_DUMP_DIR"] = str(dump_dir)
        env["TRITON_PRINT_AUTOTUNING"] = "0"
        proc = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = proc.stdout + proc.stderr
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
            "output_tail": output[-1500:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"available": False, "artifacts": {}, "output_tail": "TIMEOUT", "returncode": -1}
    finally:
        Path(tmpfile).unlink(missing_ok=True)
        shutil.rmtree(dump_dir, ignore_errors=True)


def ncu_available() -> bool:
    return shutil.which("ncu") is not None
