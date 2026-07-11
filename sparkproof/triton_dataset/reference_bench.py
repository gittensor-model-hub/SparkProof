"""Reference PyTorch benchmark for diagnostic candidate/reference comparisons.

KernelBench's core metric compares a generated kernel against the PyTorch
reference it replaces: ``speedup = reference_wall_clock / kernel_wall_clock``.
SparkProof carries a per-task PyTorch reference forward from `torch_ops.py`
(source C) through self-evolution (source D) as `torch_reference`/`reference_expr`
plus `shapes`, but nothing ever benchmarked it — this module does. The candidate
side is still self-reported, so the resulting ratio is not a trusted fast_p
metric and must not affect candidate ranking.

Reference availability is per-task, not universal: mutation-sourced tasks
(source B) carry a Triton kernel as "ground truth", not a PyTorch expression,
so there's nothing to benchmark against there, and that's fine — a missing
reference just means no speedup fields, not an error.

The reference always runs in its own clean subprocess, entirely separate from
the candidate's execution. The candidate's code never runs here, so a
candidate cannot influence how its own baseline gets measured, however it
tries (see `anti_cheat.py`'s timing-manipulation checks for how it might try).
"""

from __future__ import annotations

import re
from typing import Any

from sparkproof.triton_dataset.python_runner import run_python_source

REFERENCE_PASS_MARKER = "SPARKPROOF_REFERENCE_PASS"
REFERENCE_TIMING_RE = re.compile(r"SPARKPROOF_REFERENCE_TIMING_MS\s*[:=]\s*(\d+(?:\.\d+)?)")

# Sized for a stable, overhead-amortized timing signal. Deliberately distinct
# from torch_ops.py's ADVERSARIAL_SHAPE_PRESETS, which use small non-power-of-two
# sizes to exercise boundary-mask correctness, not to produce comparable timings.
# torch_ops.py asks the candidate to benchmark these sizes for diagnostics.
# The harness does not yet control or verify that candidate invocation, so the
# comparison must not be treated as apples-to-apples or reward-bearing.
DEFAULT_BENCHMARK_SIZES: dict[str, int] = {"M": 4096, "N": 4096, "K": 4096, "B": 32, "D": 4096, "L": 2048}

# The only scalar (non-tensor, non-dimension) free variable any current
# torch_ops.py reference expression relies on (RMSNorm's `eps`). Extending
# this catalog is safe: an unresolved free variable just makes the reference
# fail to execute, which benchmark_reference() treats as "unavailable", not
# a crash.
DEFAULT_SCALARS: dict[str, float] = {"eps": 1e-5}

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")


def _dims_in_shape(shape_expr: str) -> set[str]:
    return set(_IDENTIFIER_RE.findall(shape_expr))


def reference_runnable(
    prompt_meta: dict[str, Any] | None,
    *,
    sizes: dict[str, int] | None = None,
) -> str | None:
    """Build standalone Python source that times a task's PyTorch reference.

    Returns None when `prompt_meta` doesn't carry a usable reference — no
    `torch_reference`/`reference_expr`, no `shapes`, or a dimension letter in
    `shapes` this call doesn't have a concrete size for. Callers must treat
    None as "no reference available for this task", not an error.
    """
    if not prompt_meta:
        return None
    expr = prompt_meta.get("torch_reference") or prompt_meta.get("reference_expr")
    shapes: dict[str, str] = prompt_meta.get("shapes") or {}
    if not expr or not shapes:
        return None

    resolved_sizes = {**DEFAULT_BENCHMARK_SIZES, **(sizes or {})}
    dims: set[str] = set()
    for shape_expr in shapes.values():
        dims.update(_dims_in_shape(shape_expr))
    if dims - resolved_sizes.keys():
        return None

    setup_lines = [f"{name} = {resolved_sizes[name]}" for name in sorted(dims)]
    for tensor_name, shape_expr in shapes.items():
        setup_lines.append(f"{tensor_name} = torch.randn({shape_expr}, device='cuda', dtype=torch.float32)")
    for scalar_name, value in DEFAULT_SCALARS.items():
        setup_lines.append(f"{scalar_name} = {value}")
    setup = "\n".join(setup_lines)

    return f"""
import torch
import triton.testing
import sys

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

torch.manual_seed(42)
{setup}

try:
    def _sparkproof_reference():
        return {expr}

    _sparkproof_reference()
    torch.cuda.synchronize()
    _sparkproof_ms = triton.testing.do_bench(_sparkproof_reference)
    print(f"SPARKPROOF_REFERENCE_TIMING_MS: {{_sparkproof_ms}}")
    print("{REFERENCE_PASS_MARKER}")
except Exception as e:
    print(f"SPARKPROOF_REFERENCE_FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""


def benchmark_reference(
    prompt_meta: dict[str, Any] | None,
    *,
    gpu_index: int = 0,
    timeout: int = 60,
    sizes: dict[str, int] | None = None,
) -> float | None:
    """Time the PyTorch reference for `prompt_meta` in its own subprocess.

    Returns None if no reference is available, or if it fails to run —
    both are "no fast_p signal for this task", never an error the caller
    needs to handle specially.
    """
    source = reference_runnable(prompt_meta, sizes=sizes)
    if source is None:
        return None
    execution = run_python_source(source, gpu_index=gpu_index, timeout=timeout)
    if execution.returncode != 0 or REFERENCE_PASS_MARKER not in execution.stdout:
        return None
    matches = REFERENCE_TIMING_RE.findall(execution.output)
    if not matches:
        return None
    return float(matches[-1])
