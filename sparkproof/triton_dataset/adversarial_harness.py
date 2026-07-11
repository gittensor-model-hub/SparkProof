"""External adversarial validation beyond teacher-written tests."""

from __future__ import annotations

import ast
from typing import Any

from sparkproof.triton_dataset.python_runner import run_python_source

PASS_MARKER = "SPARKPROOF_TRITON_PASS"
SEED_PASS_MARKER = "SPARKPROOF_ADVERSARIAL_SEED_PASS"

ADVERSARIAL_SEEDS = (0, 7, 42)


def rewrite_seed_overrides(code: str) -> str:
    """Force candidate seed calls to use the externally selected seed.

    Rewriting the argument preserves non-empty Python suites; deleting a seed
    call could leave a function or conditional body syntactically invalid.
    """

    class SeedCallRewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            node = self.generic_visit(node)
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in {"manual_seed", "manual_seed_all"}:
                node.args = [ast.Name(id="_sparkproof_seed", ctx=ast.Load())]
                node.keywords = []
            return node

    tree = ast.parse(code)
    transformed = SeedCallRewriter().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed) + "\n"


def strip_seed_overrides(code: str) -> str:
    """Backward-compatible alias for the seed-rewrite behavior."""
    return rewrite_seed_overrides(code)


def build_adversarial_wrapper(code: str) -> str:
    """Build one top-level execution; the parent process supplies the seed."""
    code = rewrite_seed_overrides(code)
    return f"""
import os
import torch
import triton
import triton.language as tl
import sys

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

_sparkproof_seed = int(os.environ["SPARKPROOF_ADVERSARIAL_SEED"])
torch.manual_seed(_sparkproof_seed)
torch.cuda.manual_seed_all(_sparkproof_seed)

{code}

print(f"{SEED_PASS_MARKER}:{{_sparkproof_seed}}")
"""


def run_adversarial_execution(
    code: str,
    *,
    gpu_index: int = 0,
    timeout: int = 180,
    seeds: tuple[int, ...] = ADVERSARIAL_SEEDS,
) -> dict[str, Any]:
    wrapped = build_adversarial_wrapper(code)
    outputs: list[str] = []
    passed_seeds: list[int] = []
    for seed in seeds:
        execution = run_python_source(
            wrapped,
            gpu_index=gpu_index,
            timeout=timeout,
            env_overrides={"SPARKPROOF_ADVERSARIAL_SEED": str(seed)},
        )
        outputs.append(f"seed={seed}:\n{execution.output}")
        marker = f"{SEED_PASS_MARKER}:{seed}"
        if execution.returncode == 0 and marker in execution.stdout and PASS_MARKER in execution.stdout:
            passed_seeds.append(seed)
    combined_output = "\n".join(outputs)
    return {
        "passed": len(passed_seeds) == len(seeds),
        "seed_passes": len(passed_seeds),
        "seed_total": len(seeds),
        "passed_seeds": passed_seeds,
        "output_tail": combined_output[-2500:],
    }
