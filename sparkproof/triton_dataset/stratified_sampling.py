"""Stratified, seeded sampling over the full prompt catalog.

Replaces prefix truncation (`--limit N` cutting off source iteration order)
with deterministic, coverage-first sampling: every available source gets one
record before any source gets a second, and no single source can claim more
than `max_share` of the limit unless supply elsewhere is exhausted.
Which specific records get picked (not just which bucket) is seeded, so
different run seeds explore different subsets while a fixed seed replays
byte-identically.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from sparkproof.hashing import canonical_json_bytes, sha256_hex


def catalog_sha256(records: list[dict[str, Any]]) -> str:
    """Content hash of the full eligible catalog, independent of iteration order."""
    fingerprint = sorted((str(r.get("task_id", "")), str(r.get("prompt", ""))) for r in records)
    return sha256_hex(canonical_json_bytes(fingerprint))


def bucket_key(record: dict[str, Any]) -> tuple[str, str]:
    source = record.get("source") or record.get("origin") or "unknown"
    family = (
        record.get("task_family")
        or record.get("category")
        or record.get("target_api")
        or "unknown"
    )
    return (str(source), str(family))


def _source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[bucket_key(record)[0]] += 1
    return dict(counts)


def stratified_sample(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
    max_share: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample up to `limit` records; returns (sampled_records, source_counts).

    Coverage-first: sources round-robin (each source's own families round-robin
    within its turn), so every source gets a record before any source gets a
    second. A source capped at `max_share * limit` is skipped in later passes
    unless every other source is exhausted or also capped, in which case the
    cap is lifted rather than under-filling `limit`.
    """
    if limit is None or limit >= len(records):
        return list(records), _source_counts(records)
    if limit <= 0:
        return [], {}

    rng = random.Random(seed)
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        source, family = bucket_key(record)
        by_source[source][family].append(record)
    for families in by_source.values():
        for members in families.values():
            rng.shuffle(members)

    source_order = list(by_source.keys())
    rng.shuffle(source_order)
    family_order: dict[str, list[str]] = {}
    for source in source_order:
        families = list(by_source[source].keys())
        rng.shuffle(families)
        family_order[source] = families

    cap = max(1, math.ceil(limit * max_share))
    taken_by_bucket: dict[tuple[str, str], int] = defaultdict(int)
    taken_by_source: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []

    def take_one(source: str) -> bool:
        """Pick the next unclaimed record from one of `source`'s families,
        rotating which family goes first each call. Returns False if `source`
        has nothing left across any of its families."""
        families = family_order[source]
        for _ in range(len(families)):
            family = families[0]
            families.append(families.pop(0))
            members = by_source[source][family]
            idx = taken_by_bucket[(source, family)]
            if idx < len(members):
                taken_by_bucket[(source, family)] += 1
                taken_by_source[source] += 1
                selected.append(members[idx])
                return True
        return False

    enforce_cap = True
    while len(selected) < limit:
        progressed = False
        for source in source_order:
            if len(selected) >= limit:
                break
            if enforce_cap and taken_by_source[source] >= cap:
                continue
            if take_one(source):
                progressed = True
        if progressed:
            continue
        if enforce_cap:
            enforce_cap = False  # supply exists only in capped buckets — relax rather than under-fill
            continue
        break  # genuinely exhausted every bucket

    return selected, dict(taken_by_source)
