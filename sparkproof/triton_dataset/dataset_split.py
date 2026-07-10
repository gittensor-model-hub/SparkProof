"""Ancestry-aware train/dev splits (never random row sampling)."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable


def split_group_key(record: dict[str, Any]) -> str:
    """Return the strongest leakage identity available for one record."""
    if record.get("task_family"):
        return f"family:{record['task_family']}"
    if record.get("ground_truth_code"):
        digest = hashlib.sha256(str(record["ground_truth_code"]).encode()).hexdigest()
        return f"reference:{digest}"
    if record.get("torch_reference"):
        digest = hashlib.sha256(str(record["torch_reference"]).encode()).hexdigest()
        return f"torch_reference:{digest}"
    if record.get("parent_id"):
        return f"ancestry:{record['parent_id']}"
    if record.get("target_api"):
        return f"api:{record['target_api']}"
    if record.get("prompt_template"):
        return f"template:{record['prompt_template']}"
    return f"task:{record.get('task_id', '')}"


def _stable_bucket(key: str, *, dev_fraction: float) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()
    threshold = int(dev_fraction * 256)
    return "dev" if int(digest[:2], 16) < threshold else "train"


def assign_splits(
    records: Iterable[dict[str, Any]],
    *,
    dev_fraction: float = 0.1,
    respect_existing: bool = True,
) -> list[dict[str, Any]]:
    """Assign split labels while keeping ancestry groups intact."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")

    rows = [dict(record) for record in records]
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    task_indexes = {
        str(record["task_id"]): index
        for index, record in enumerate(rows)
        if record.get("task_id")
    }
    identity_indexes: dict[str, int] = {}
    for index, record in enumerate(rows):
        parent_id = record.get("parent_id")
        if parent_id is not None and str(parent_id) in task_indexes:
            union(index, task_indexes[str(parent_id)])

        identity = split_group_key(record)
        if identity in identity_indexes:
            union(index, identity_indexes[identity])
        else:
            identity_indexes[identity] = index

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(rows):
        grouped[find(index)].append(record)

    out: list[dict[str, Any]] = []
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: min(str(record.get("task_id", "")) for record in group),
    )
    for group in ordered_groups:
        identities = sorted({split_group_key(record) for record in group})
        group_key = "component:" + hashlib.sha256("\n".join(identities).encode()).hexdigest()
        split = _stable_bucket(group_key, dev_fraction=dev_fraction)
        for record in group:
            if respect_existing and record.get("split") in {"eval", "held_out"}:
                out.append(record)
                continue
            record["split"] = split
            record["split_group"] = group_key
            out.append(record)
    return out


def summarize_splits(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("split", "unknown"))] += 1
    return dict(sorted(counts.items()))
