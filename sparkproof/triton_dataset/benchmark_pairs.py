"""Build optimization preference pairs from measured candidate timings."""

from __future__ import annotations

import statistics
from typing import Any

DEFAULT_MIN_SPEEDUP = 0.03


def _latency_ms(validation: dict[str, Any]) -> float | None:
    bench = validation.get("benchmark") or {}
    if bench.get("timing_method") != "candidate_triton_do_bench":
        return None
    timing = bench.get("timing_ms")
    if isinstance(timing, (int, float)):
        return float(timing)
    return None


def candidate_timing(candidate: dict[str, Any]) -> float | None:
    validation = candidate.get("validation") or candidate.get("sparkproof_validation") or {}
    return _latency_ms(validation)


def build_preference_pair(
    *,
    task_id: str,
    prompt: str,
    winner: dict[str, Any],
    loser: dict[str, Any],
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
) -> dict[str, Any] | None:
    winner_ms = candidate_timing(winner)
    loser_ms = candidate_timing(loser)
    chosen = winner.get("response", "")
    rejected = loser.get("response", "")
    if (
        winner_ms is None
        or loser_ms is None
        or loser_ms <= 0
        or not prompt.strip()
        or not chosen.strip()
        or not rejected.strip()
    ):
        return None
    speedup = (loser_ms - winner_ms) / loser_ms
    if speedup < min_speedup:
        return None
    return {
        "task_id": task_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "winner_ms": winner_ms,
        "loser_ms": loser_ms,
        "speedup": speedup,
        "pair_type": "optimization",
    }


def preference_pairs_from_adjudication(
    adjudication_rows: list[dict[str, Any]],
    *,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in adjudication_rows:
        candidates = row.get("candidates") or []
        passing = [c for c in candidates if c.get("passed")]
        if len(passing) < 2:
            continue
        passing.sort(key=lambda c: candidate_timing(c) or float("inf"))
        winner = passing[0]
        for loser in passing[1:]:
            pair = build_preference_pair(
                task_id=str(row.get("task_id", "task")),
                prompt=str(row.get("prompt", "")),
                winner=winner,
                loser=loser,
                min_speedup=min_speedup,
            )
            if pair is not None:
                pairs.append(pair)
    return pairs


def summarize_timings(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {}
    ordered = sorted(samples_ms)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }
