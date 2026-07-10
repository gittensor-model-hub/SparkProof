"""Filter prompt records by source and task_id."""

from __future__ import annotations

from typing import Any


def parse_filter_set(values: list[str] | None) -> frozenset[str] | None:
    if not values:
        return None
    out = {v.strip() for v in values if v and v.strip()}
    return frozenset(out) if out else None


def prompt_matches_filters(
    record: dict[str, Any],
    *,
    sources: frozenset[str] | None = None,
    task_ids: frozenset[str] | None = None,
) -> bool:
    if sources is not None and record.get("source") not in sources:
        return False
    if task_ids is not None and record.get("task_id") not in task_ids:
        return False
    return True
