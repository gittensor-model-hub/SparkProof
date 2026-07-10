"""Source B: structural mutations on valid kernels (free ground truth)."""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Callable


MutationFn = Callable[[str], tuple[str, str]]


def strip_boundary_mask(code: str) -> tuple[str, str]:
    class MaskRemover(ast.NodeTransformer):
        changed = False

        def visit_Call(self, node: ast.Call) -> ast.Call:
            node = self.generic_visit(node)
            is_memory_op = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
                and node.func.attr in {"load", "store"}
            )
            if not is_memory_op:
                return node
            filtered = [kw for kw in node.keywords if kw.arg not in {"mask", "other"}]
            if len(filtered) != len(node.keywords):
                self.changed = True
                node.keywords = filtered
            return node

    tree = ast.parse(code)
    remover = MaskRemover()
    mutated_tree = remover.visit(tree)
    if not remover.changed:
        return code, "No boundary masks were present."
    ast.fix_missing_locations(mutated_tree)
    mutated = ast.unparse(mutated_tree) + "\n"
    return mutated, "Removed boundary masks from tl.load/tl.store."


def downgrade_accumulator_precision(code: str) -> tuple[str, str]:
    class PrecisionDowngrader(ast.NodeTransformer):
        changed = False

        @staticmethod
        def _is_tl_float32(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "tl"
                and node.attr == "float32"
            )

        @staticmethod
        def _float16() -> ast.Attribute:
            return ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="float16", ctx=ast.Load())

        def visit_Call(self, node: ast.Call) -> ast.Call:
            node = self.generic_visit(node)
            is_tl_zeros = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
                and node.func.attr == "zeros"
            )
            if is_tl_zeros:
                for keyword in node.keywords:
                    if keyword.arg == "dtype" and self._is_tl_float32(keyword.value):
                        keyword.value = self._float16()
                        self.changed = True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to" and node.args:
                if self._is_tl_float32(node.args[0]):
                    node.args[0] = self._float16()
                    self.changed = True
            return node

    tree = ast.parse(code)
    downgrader = PrecisionDowngrader()
    mutated_tree = downgrader.visit(tree)
    if not downgrader.changed:
        return code, "No fp32 accumulation path was present."
    ast.fix_missing_locations(mutated_tree)
    return (
        ast.unparse(mutated_tree) + "\n",
        "Downgraded accumulators to float16, causing numerical instability.",
    )


def strip_autotune_and_stages(code: str) -> tuple[str, str]:
    mutated = re.sub(r"@triton\.autotune\(.*?\)\s*(@triton\.jit)", r"\1", code, flags=re.DOTALL)
    mutated = re.sub(r",\s*num_stages\s*=\s*\d+", "", mutated)
    mutated = re.sub(r",\s*num_warps\s*=\s*\d+", "", mutated)
    return mutated, "Removed @triton.autotune, num_stages, and num_warps."


MUTATIONS: list[MutationFn] = [
    strip_boundary_mask,
    downgrade_accumulator_precision,
    strip_autotune_and_stages,
]


def build_mutation_prompt(
    *,
    task_id: str,
    valid_kernel: str,
    mutator: MutationFn | None = None,
) -> dict:
    if mutator is not None:
        candidates = [mutator]
    else:
        start = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % len(MUTATIONS)
        candidates = [MUTATIONS[(start + offset) % len(MUTATIONS)] for offset in range(len(MUTATIONS))]

    selected: tuple[MutationFn, str, str] | None = None
    for fn in candidates:
        broken, reason = fn(valid_kernel)
        if broken == valid_kernel:
            continue
        try:
            ast.parse(broken)
        except SyntaxError:
            continue
        selected = (fn, broken, reason)
        break
    if selected is None:
        raise ValueError(f"no applicable, syntax-preserving mutation for task {task_id!r}")

    fn, broken, reason = selected
    is_opt = fn is strip_autotune_and_stages

    if is_opt:
        user = (
            "Optimize the following Triton 3.7.1 kernel for Blackwell SM12x. "
            "Add @triton.autotune, tune num_warps/num_stages, and ensure boundary masks.\n\n"
            f"```python\n{broken}\n```"
        )
        category = "optimization"
    else:
        user = (
            "The following Triton 3.7.1 kernel fails or is numerically wrong. "
            "Diagnose the bug and return corrected, runnable code with a torch.allclose test. "
            "End with: print(\"SPARKPROOF_TRITON_PASS\"). "
            "Do NOT pass BLOCK= to the kernel launch when @triton.autotune is used.\n\n"
            f"```python\n{broken}\n```"
        )
        category = "debugging"

    return {
        "task_id": f"mutate_{task_id}",
        "source": "mutation",
        "category": category,
        "prompt": user,
        "ground_truth_code": valid_kernel,
        "mutation_reason": reason,
        "broken_code": broken,
    }


def iter_mutation_prompts(*, task_id: str, valid_kernel: str) -> list[dict]:
    """Produce every applicable syntax-preserving mutation with stable task IDs."""
    prompts: list[dict] = []
    for fn in MUTATIONS:
        variant_id = f"{task_id}_{fn.__name__}"
        try:
            prompt = build_mutation_prompt(
                task_id=variant_id,
                valid_kernel=valid_kernel,
                mutator=fn,
            )
            prompt["task_family"] = task_id
            prompts.append(prompt)
        except ValueError:
            continue
    if not prompts:
        raise ValueError(f"reference kernel {task_id!r} has no applicable mutations")
    return prompts
