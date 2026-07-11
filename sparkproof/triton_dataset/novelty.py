"""Global novelty accounting against a pinned accepted-dataset registry.

Stratified sampling and run-scoped evolution diversify what a single run
generates, but diversity cannot rely on random seeds alone — a miner can
still regenerate the same prompt set. The authoritative check is comparing a
submitted bundle's rows against previously-accepted fingerprints, using the
same unsalted fingerprints `decontaminate.py` already uses for eval-leakage
detection:

  - normalized prompt text hash        (`text_fingerprint`)
  - semantic task fingerprint          (`semantic_task_fingerprint`)
  - canonical reference-kernel AST hash (`get_canonical_structure` on ground truth/broken code)
  - canonical assistant-code AST hash  (`get_canonical_structure` on the extracted response)

Never salt these fingerprints with `run_seed`, `run_id`, or contributor
identity — that would disguise duplicates instead of creating diversity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from sparkproof.triton_dataset.decontaminate import (
    extract_python_from_response,
    get_canonical_structure,
    semantic_task_fingerprint,
    text_fingerprint,
)

Verdict = Literal["exact", "near", "novel"]


@dataclass(frozen=True)
class RowFingerprint:
    task_id: str
    prompt_hash: str
    semantic_hash: str
    reference_ast_hash: str | None
    assistant_ast_hash: str | None
    parent_id: str | None


def fingerprint_row(row: dict[str, Any]) -> RowFingerprint:
    """Fingerprint a prompt/trajectory row. Inputs only — never run/identity metadata."""
    meta = (row.get("metadata") or {}).get("prompt_meta") or row
    prompt = row.get("prompt") or meta.get("prompt") or ""
    reference_code = meta.get("ground_truth_code") or meta.get("broken_code") or ""
    assistant_code = extract_python_from_response(row["response"]) if row.get("response") else ""

    return RowFingerprint(
        task_id=str(row.get("task_id") or meta.get("task_id") or ""),
        prompt_hash=text_fingerprint(prompt) if prompt.strip() else "",
        semantic_hash=semantic_task_fingerprint(meta),
        reference_ast_hash=get_canonical_structure(reference_code) if reference_code.strip() else None,
        assistant_ast_hash=get_canonical_structure(assistant_code) if assistant_code.strip() else None,
        parent_id=meta.get("parent_id"),
    )


class NoveltyRegistry:
    """A pinned snapshot of previously-accepted fingerprints to compare new rows against.

    Two independent key sets: `prompt_hash`/`assistant_ast_hash` matches are
    exact duplicates (the same question, or byte-for-byte-equivalent code,
    was already accepted); `semantic_hash`/`reference_ast_hash` matches with
    no exact match are near-duplicates (same operator family/reference
    kernel, different surface code).
    """

    def __init__(self, fingerprints: Iterable[RowFingerprint] = ()) -> None:
        self._exact_keys: set[str] = set()
        self._near_keys: set[str] = set()
        for fp in fingerprints:
            self.add(fp)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> "NoveltyRegistry":
        return cls(fingerprint_row(row) for row in rows)

    def add(self, fingerprint: RowFingerprint) -> None:
        for value in (fingerprint.prompt_hash, fingerprint.assistant_ast_hash):
            if value:
                self._exact_keys.add(value)
        for value in (fingerprint.semantic_hash, fingerprint.reference_ast_hash):
            if value:
                self._near_keys.add(value)

    def classify(self, fingerprint: RowFingerprint) -> Verdict:
        if (fingerprint.prompt_hash and fingerprint.prompt_hash in self._exact_keys) or (
            fingerprint.assistant_ast_hash and fingerprint.assistant_ast_hash in self._exact_keys
        ):
            return "exact"
        if (fingerprint.semantic_hash and fingerprint.semantic_hash in self._near_keys) or (
            fingerprint.reference_ast_hash and fingerprint.reference_ast_hash in self._near_keys
        ):
            return "near"
        return "novel"

    def copy(self) -> "NoveltyRegistry":
        clone = NoveltyRegistry()
        clone._exact_keys = set(self._exact_keys)
        clone._near_keys = set(self._near_keys)
        return clone


@dataclass
class NoveltyReport:
    verified_rows: int
    exact_duplicate_rows: int
    near_duplicate_rows: int
    novel_verified_rows: int
    duplicate_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_rows": self.verified_rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "near_duplicate_rows": self.near_duplicate_rows,
            "novel_verified_rows": self.novel_verified_rows,
            "duplicate_task_ids": self.duplicate_task_ids,
        }


def compute_novelty_report(
    verified_rows: Iterable[dict[str, Any]],
    registry: NoveltyRegistry,
) -> NoveltyReport:
    """Classify `verified_rows` against `registry` without mutating it.

    Also catches duplicates *within this bundle* (retried rows submitted
    twice in the same run) by growing a working copy of the registry as rows
    are classified — so `total - dup_count` never over-counts intra-bundle
    repeats as separately novel. `verified_rows` must already be
    validation-passed; this only accounts novelty, not correctness.
    """
    rows = list(verified_rows)
    working = registry.copy()
    exact = 0
    near = 0
    duplicate_task_ids: list[str] = []

    for row in rows:
        fp = fingerprint_row(row)
        verdict = working.classify(fp)
        if verdict == "exact":
            exact += 1
            duplicate_task_ids.append(fp.task_id)
        elif verdict == "near":
            near += 1
            duplicate_task_ids.append(fp.task_id)
        working.add(fp)

    novel = len(rows) - exact - near
    return NoveltyReport(
        verified_rows=len(rows),
        exact_duplicate_rows=exact,
        near_duplicate_rows=near,
        novel_verified_rows=novel,
        duplicate_task_ids=duplicate_task_ids,
    )
