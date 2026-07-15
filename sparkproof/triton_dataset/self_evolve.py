"""Source D: deterministic self-evolution over validated train/dev parents."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any

from sparkproof.gpu.architecture import ARCH_BLACKWELL, sm_label
from sparkproof.triton_dataset.run_seed import evolution_seed
from sparkproof.triton_dataset.task_policy import is_evolution_parent_allowed, normalize_train_task

EVOLUTION_OPS = (
    "tail_dimension",
    "non_contiguous_stride",
    "bf16",
    "add_fusion",
    "inject_bug",
    "optimization_target",
    "blackwell_autotune",
)

_OP_PROMPTS: dict[str, str] = {
    "tail_dimension": (
        "Use non-power-of-two tail dimensions (e.g. N=6143, M=1025) with explicit boundary masks on every load/store."
    ),
    "non_contiguous_stride": (
        "Inputs must use non-contiguous strided tensors (non-unit stride on the reduction or batch dimension)."
    ),
    "bf16": ("Compute in bf16 inputs with fp32 accumulators for reductions and tl.dot."),
    "add_fusion": ("Fuse a residual add into the same kernel pass before normalization or activation."),
    "inject_bug": (
        "DEBUGGING TASK: the reference kernel below has a deliberate mask/stride bug — fix it.\n\n"
        "```python\n{ground_truth}\n```"
    ),
    "optimization_target": (
        "Add @triton.autotune with at least 3 configs and tune num_warps/num_stages for {gpu_label}."
    ),
    "blackwell_autotune": (
        "Target {gpu_label} explicitly; prefer tl.make_tensor_descriptor where applicable over deprecated block_ptr."
    ),
}


def apply_evolution(parent: dict[str, Any], operation: str) -> dict[str, Any] | None:
    if not is_evolution_parent_allowed(parent):
        return None

    child = copy.deepcopy(parent)
    child["parent_id"] = parent.get("task_id")
    child["origin"] = "self_evolution"
    child["source"] = "self_evolution"
    child["split"] = parent.get("split", "train")
    child["evolution_ops"] = list(parent.get("evolution_ops", [])) + [operation]
    child["difficulty"] = min(5, int(parent.get("difficulty", 1)) + 1)

    gpu_architecture = parent.get("gpu_architecture", ARCH_BLACKWELL)
    gpu_label = sm_label(gpu_architecture)
    base_prompt = parent.get("prompt", "")

    if operation == "inject_bug":
        gt = parent.get("ground_truth_code") or ""
        if not gt:
            return None
        from sparkproof.triton_dataset.mutator import strip_boundary_mask

        broken, reason = strip_boundary_mask(gt)
        child["prompt"] = (
            f"Fix the Triton 3.7.1 kernel bug on {gpu_label}.\n\n```python\n{broken}\n```\n\nHint: {reason}"
        )
        child["ground_truth_code"] = gt
        child["category"] = "debugging"
    else:
        op_text = _OP_PROMPTS.get(operation, operation).format(gpu_label=gpu_label)
        child["prompt"] = f"{base_prompt}\n\nAdditional requirement: {op_text}"
        child["category"] = parent.get("category", "kernel_generation")

    child["gpu_architecture"] = gpu_architecture
    child["task_id"] = f"evolve_{parent.get('task_id', 'parent')}_{operation}"
    return normalize_train_task(child)


def evolve_parent(
    parent: dict[str, Any],
    *,
    depth: int = 1,
    rng: random.Random | None = None,
    run_seed: str | None = None,
) -> list[dict[str, Any]]:
    """Apply up to `depth` distinct evolution ops (deterministic sampling).

    Without `run_seed`, op selection is stable per parent task_id alone (every
    run picks the same children of a given parent). With `run_seed`, selection
    is scoped to `H(run_seed || parent_task_id || depth)` — different run seeds
    can explore different children of the same parent, while any individual
    run remains exactly reproducible from its own seed.
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    parent_id = str(parent.get("task_id", "parent"))
    if run_seed:
        stable_seed = evolution_seed(run_seed, parent_id, depth)
    else:
        stable_seed = int(hashlib.sha256(parent_id.encode()).hexdigest(), 16)
    r = rng or random.Random(stable_seed)
    ops = r.sample(list(EVOLUTION_OPS), k=min(depth, len(EVOLUTION_OPS)))
    children: list[dict[str, Any]] = []
    for op in ops:
        child = apply_evolution(parent, op)
        if child is not None:
            children.append(child)
    return children


def evolve_verified_trajectory(
    trajectory: dict[str, Any], *, depth: int = 1, run_seed: str | None = None
) -> list[dict[str, Any]]:
    """Build evolution parents from a passing trajectory row."""
    validation = trajectory.get("sparkproof_validation") or {}
    if validation and not validation.get("passed"):
        return []

    meta = (trajectory.get("metadata") or {}).get("prompt_meta") or {}
    parent = {
        "task_id": meta.get("task_id") or trajectory.get("task_id", "traj"),
        "prompt": trajectory["prompt"],
        "system": trajectory.get("system"),
        "origin": meta.get("origin") or meta.get("source", "torch_op"),
        "split": meta.get("split", "train"),
        "category": meta.get("category"),
        "torch_reference": meta.get("torch_reference"),
        "ground_truth_code": meta.get("ground_truth_code"),
        "reference_expr": meta.get("torch_reference"),
        "shapes": meta.get("shapes"),
        "task_family": meta.get("category"),
    }
    return evolve_parent(parent, depth=depth, run_seed=run_seed)
