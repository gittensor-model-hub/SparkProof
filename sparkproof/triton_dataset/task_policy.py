"""Train/eval split guards — TritonBench must never enter training generation."""

from __future__ import annotations

from typing import Any

FORBIDDEN_TRAINING_ORIGINS = frozenset(
    {
        "tritonbench",
        "kernelbench_eval",
        "private_eval",
        "yaml",
    }
)

FORBIDDEN_TRAINING_SPLITS = frozenset({"test", "eval"})

TRAINABLE_SPLITS = frozenset({"train", "dev"})


def assert_trainable_task(task: dict[str, Any]) -> None:
    origin = task.get("origin") or task.get("source")
    if origin in FORBIDDEN_TRAINING_ORIGINS:
        raise ValueError(f"Eval-origin task {task.get('task_id', '?')!r} cannot enter training (origin={origin!r})")
    split = task.get("split")
    if split in FORBIDDEN_TRAINING_SPLITS:
        raise ValueError(f"Task {task.get('task_id', '?')!r} belongs to split={split!r}")


def normalize_train_task(task: dict[str, Any], *, default_split: str = "train") -> dict[str, Any]:
    """Ensure trainable metadata on a prompt record."""
    out = dict(task)
    if "origin" not in out and "source" in out:
        out["origin"] = out["source"]
    out.setdefault("split", default_split)
    assert_trainable_task(out)
    return out


def is_evolution_parent_allowed(parent: dict[str, Any]) -> bool:
    try:
        assert_trainable_task(parent)
    except ValueError:
        return False
    if not parent.get("reference_expr") and not parent.get("torch_reference") and not parent.get("ground_truth_code"):
        return False
    return True
