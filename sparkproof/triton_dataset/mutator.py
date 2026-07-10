"""Source B: structural mutations on valid kernels (free ground truth)."""

from __future__ import annotations

import random
import re
from typing import Callable


MutationFn = Callable[[str], tuple[str, str]]


def strip_boundary_mask(code: str) -> tuple[str, str]:
    pattern = r"tl\.(load|store)\((.*?),\s*mask\s*=\s*[^,)]+(?:,\s*other\s*=\s*[^)]+)?\)"
    mutated = re.sub(pattern, r"tl.\1(\2)", code, flags=re.DOTALL)
    return mutated, "Removed boundary masks from tl.load/tl.store."


def downgrade_accumulator_precision(code: str) -> tuple[str, str]:
    mutated = re.sub(
        r"acc\s*=\s*tl\.zeros\(([^)]+),\s*dtype\s*=\s*tl\.float32\)",
        r"acc = tl.zeros(\1, dtype=tl.float16)",
        code,
    )
    mutated = re.sub(r"\.to\(tl\.float32\)", ".to(tl.float16)", mutated)
    return mutated, "Downgraded accumulators to float16, causing numerical instability."


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
    fn = mutator or random.choice(MUTATIONS)
    broken, reason = fn(valid_kernel)
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
