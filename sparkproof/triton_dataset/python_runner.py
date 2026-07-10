"""Shared isolated Python subprocess runner for GPU validation harnesses."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping


@dataclass(frozen=True)
class PythonExecution:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def run_python_source(
    source: str,
    *,
    gpu_index: int,
    timeout: int,
    env_overrides: Mapping[str, str] | None = None,
) -> PythonExecution:
    """Run source in a temporary file with consistent CUDA isolation and cleanup."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write(source)
        tmpfile = Path(handle.name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if env_overrides:
        env.update(env_overrides)

    try:
        process = subprocess.run(
            [sys.executable, str(tmpfile)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return PythonExecution(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return PythonExecution(returncode=-1, stdout=stdout, stderr=stderr or "TIMEOUT", timed_out=True)
    finally:
        tmpfile.unlink(missing_ok=True)
