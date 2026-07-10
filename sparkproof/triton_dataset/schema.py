"""Runtime validation for Triton prompt JSONL records."""

from __future__ import annotations

from typing import Any

from sparkproof.triton_dataset.task_policy import TRAINABLE_SPLITS, assert_trainable_task


class PromptValidationError(ValueError):
    """A prompt record does not satisfy the production JSONL contract."""


def validate_prompt_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise PromptValidationError("prompt record must be a JSON object")

    out = dict(record)
    for field in ("task_id", "prompt", "source", "origin", "split", "category"):
        value = out.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PromptValidationError(f"{field} must be a non-empty string")

    system = out.get("system")
    if system is not None and not isinstance(system, str):
        raise PromptValidationError("system must be a string when present")
    if out["split"] not in TRAINABLE_SPLITS:
        raise PromptValidationError(f"training prompt split must be one of {sorted(TRAINABLE_SPLITS)}")

    assert_trainable_task(out)
    return out
