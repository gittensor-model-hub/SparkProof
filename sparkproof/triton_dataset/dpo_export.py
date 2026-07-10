"""Export optimization preference pairs for SparkDistill DPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from sparkproof.triton_dataset.benchmark_pairs import preference_pairs_from_adjudication


def load_adjudication(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid adjudication JSON at {path}:{line_number}") from exc
    return rows


def enrich_adjudication_with_responses(
    adjudication: list[dict[str, Any]],
    *,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Backfill original prompts from winner checkpoints.

    Checkpoints intentionally contain winners only, so they cannot reconstruct
    missing loser candidates for preference pairs.
    """
    if checkpoint_path is None or not checkpoint_path.is_file():
        return adjudication

    by_task: dict[str, list[dict[str, Any]]] = {}
    with checkpoint_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = (row.get("metadata") or {}).get("prompt_meta") or {}
            task_id = meta.get("task_id") or row.get("task_id")
            if task_id:
                by_task.setdefault(task_id, []).append(row)

    enriched: list[dict[str, Any]] = []
    for row in adjudication:
        out = dict(row)
        task_id = row.get("task_id")
        trajectories = by_task.get(str(task_id), [])
        if not out.get("prompt") and trajectories:
            first = trajectories[0]
            prompt_meta = (first.get("metadata") or {}).get("prompt_meta") or {}
            out["prompt"] = prompt_meta.get("prompt") or ""
        enriched.append(out)
    return enriched


def export_dpo_jsonl(
    adjudication_rows: Iterable[dict[str, Any]],
    *,
    min_speedup: float = 0.03,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in adjudication_rows:
        if isinstance(row.get("results"), list):
            normalized.extend(result for result in row["results"] if isinstance(result, dict))
        else:
            normalized.append(row)
    pairs = preference_pairs_from_adjudication(normalized, min_speedup=min_speedup)
    return [
        {
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "metadata": {
                "task_id": pair["task_id"],
                "pair_type": pair["pair_type"],
                "winner_ms": pair["winner_ms"],
                "loser_ms": pair["loser_ms"],
                "speedup": pair["speedup"],
            },
        }
        for pair in pairs
    ]


def write_dpo_jsonl(path: Path, pairs: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
    return count
