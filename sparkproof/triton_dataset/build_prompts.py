"""Merge prompt sources A–E into SparkProof-compatible jsonl."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator

from sparkproof.triton_dataset.doc_chunks import (
    CHUNK_API_SYMBOL,
    api_unit_chunks_from_registry,
    doc_kinds_for_sources,
    load_doc_chunks,
    prompt_from_doc_chunk,
)
from sparkproof.triton_dataset.mutator import iter_mutation_prompts
from sparkproof.triton_dataset.prompt_filters import prompt_matches_filters
from sparkproof.triton_dataset.reference_kernels import REFERENCE_KERNELS
from sparkproof.triton_dataset.schema import validate_prompt_record
from sparkproof.triton_dataset.task_policy import normalize_train_task
from sparkproof.triton_dataset.torch_ops import iter_torch_translation_prompts

DEFAULT_SYSTEM = (
    "You are a Triton 3.7.1 GPU kernel expert for Blackwell SM12x. "
    "Write complete runnable Python with @triton.jit, launcher, and torch.allclose test. "
    "Always end with print(\"SPARKPROOF_TRITON_PASS\") after tests pass."
)

# TritonBench YAML is eval-only — never include "yaml" in training sources.

DEFAULT_TRAIN_SOURCES = frozenset(
    {
        "api_doc",
        "doc_semantics",
        "doc_tutorial",
        "mutation",
        "torch_op",
    }
)
DEFAULT_TRAIN_SOURCES_STR = "api_doc,doc_semantics,doc_tutorial,mutation,torch_op"


def _finalize(record: dict[str, Any]) -> dict[str, Any]:
    out = {
        "prompt": record["prompt"],
        "system": record.get("system", DEFAULT_SYSTEM),
    }
    for key in (
        "task_id",
        "source",
        "origin",
        "split",
        "category",
        "target_api",
        "ground_truth_code",
        "ground_truth_hint",
        "broken_code",
        "mutation_reason",
        "torch_reference",
        "reference_expr",
        "doc_chunk_id",
        "level",
        "title",
        "task_family",
        "parent_id",
        "evolution_ops",
    ):
        if key in record and record[key] is not None:
            out[key] = record[key]
    out.setdefault("origin", out.get("source", "unknown"))
    out.setdefault("split", "train")
    if out.get("torch_reference") and not out.get("reference_expr"):
        out["reference_expr"] = out["torch_reference"]
    return normalize_train_task(out)


def _load_jsonl_prompts(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _yield_if_matches(
    record: dict[str, Any],
    *,
    filter_sources: frozenset[str] | None,
    filter_task_ids: frozenset[str] | None,
) -> dict[str, Any] | None:
    finalized = _finalize(record)
    if not prompt_matches_filters(finalized, sources=filter_sources, task_ids=filter_task_ids):
        return None
    return finalized


def iter_all_prompts(
    *,
    doc_dir: Path | None = None,
    mined_prompts_path: Path | None = None,
    evolved_prompts_path: Path | None = None,
    include_sources: frozenset[str] | None = None,
    auto_fetch_docs: bool = True,
    enrich_api_pages: bool | None = None,
    strict_docs: bool = False,
    filter_sources: frozenset[str] | None = None,
    filter_task_ids: frozenset[str] | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    sources = include_sources or DEFAULT_TRAIN_SOURCES
    emitted = 0

    def emit(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
        nonlocal emitted
        if limit is not None and emitted >= limit:
            return
        out = _yield_if_matches(record, filter_sources=filter_sources, filter_task_ids=filter_task_ids)
        if out is None:
            return
        emitted += 1
        yield out

    def limit_reached() -> bool:
        return limit is not None and emitted >= limit

    doc_kinds = doc_kinds_for_sources(sources)
    if doc_kinds:
        chunks = load_doc_chunks(
            doc_dir,
            auto_fetch=auto_fetch_docs,
            kinds=doc_kinds,
            enrich_api_pages=enrich_api_pages,
        )
        if strict_docs:
            counts = {
                kind: sum(1 for chunk in chunks if chunk.get("chunk_kind") == kind)
                for kind in doc_kinds
            }
            minimums = {
                "api_symbol": 50,
                "semantics": 1,
                "tutorial": 1,
            }
            missing = {
                kind: (counts.get(kind, 0), minimums[kind])
                for kind in doc_kinds
                if counts.get(kind, 0) < minimums[kind]
            }
            if missing:
                details = ", ".join(
                    f"{kind}={actual} (minimum {minimum})"
                    for kind, (actual, minimum) in sorted(missing.items())
                )
                raise RuntimeError(
                    f"incomplete pinned Triton documentation corpus: {details}; "
                    "provide --doc-dir/cache or explicitly allow partial docs"
                )
        if not chunks and CHUNK_API_SYMBOL in doc_kinds:
            chunks = api_unit_chunks_from_registry()
        for chunk in chunks:
            if limit_reached():
                break
            rec = prompt_from_doc_chunk(chunk)
            rec.setdefault("origin", rec.get("source", "api_doc"))
            yield from emit(rec)

    if limit_reached():
        return

    if "mutation" in sources:
        for name, code in REFERENCE_KERNELS.items():
            if limit_reached():
                break
            for rec in iter_mutation_prompts(task_id=name, valid_kernel=code):
                if limit_reached():
                    break
                rec.setdefault("origin", "mutation")
                rec.setdefault("ground_truth_code", code)
                yield from emit(rec)

    if limit_reached():
        return

    if "torch_op" in sources:
        for rec in iter_torch_translation_prompts():
            if limit_reached():
                break
            rec.setdefault("origin", "torch_op")
            yield from emit(rec)

    if limit_reached():
        return

    if "failure_mining" in sources and mined_prompts_path and mined_prompts_path.exists():
        for rec in _load_jsonl_prompts(mined_prompts_path):
            if limit_reached():
                break
            rec.setdefault("origin", "failure_mining")
            yield from emit(rec)

    if limit_reached():
        return

    if "self_evolution" in sources and evolved_prompts_path and evolved_prompts_path.exists():
        for rec in _load_jsonl_prompts(evolved_prompts_path):
            if limit_reached():
                break
            rec.setdefault("origin", "self_evolution")
            yield from emit(rec)


def write_prompts_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Atomically stream validated records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_task_ids: set[str] = set()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            for rec in records:
                validated = validate_prompt_record(rec)
                task_id = validated["task_id"]
                if task_id in seen_task_ids:
                    raise ValueError(f"duplicate task_id {task_id!r}")
                seen_task_ids.add(task_id)
                f.write(json.dumps(validated, ensure_ascii=False) + "\n")
                count += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return count


def build_prompts_file(
    out_path: Path,
    *,
    doc_dir: Path | None = None,
    mined_prompts_path: Path | None = None,
    evolved_prompts_path: Path | None = None,
    limit: int | None = None,
    sources: frozenset[str] | None = None,
    auto_fetch_docs: bool = True,
    enrich_api_pages: bool | None = None,
    strict_docs: bool = False,
    filter_sources: frozenset[str] | None = None,
    filter_task_ids: frozenset[str] | None = None,
) -> int:
    records = iter_all_prompts(
        doc_dir=doc_dir,
        mined_prompts_path=mined_prompts_path,
        evolved_prompts_path=evolved_prompts_path,
        include_sources=sources,
        auto_fetch_docs=auto_fetch_docs,
        enrich_api_pages=enrich_api_pages,
        strict_docs=strict_docs,
        filter_sources=filter_sources,
        filter_task_ids=filter_task_ids,
        limit=limit,
    )
    return write_prompts_jsonl(out_path, records)
