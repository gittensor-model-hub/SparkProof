"""Merge prompt sources A–E into SparkProof-compatible jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from sparkproof.triton_dataset.doc_chunks import api_unit_chunks_from_registry, load_doc_chunks, prompt_from_api_chunk
from sparkproof.triton_dataset.mutator import build_mutation_prompt
from sparkproof.triton_dataset.reference_kernels import REFERENCE_KERNELS
from sparkproof.triton_dataset.task_policy import normalize_train_task
from sparkproof.triton_dataset.torch_ops import iter_torch_translation_prompts

DEFAULT_SYSTEM = (
    "You are a Triton 3.7.1 GPU kernel expert for Blackwell SM12x. "
    "Write complete runnable Python with @triton.jit, launcher, and torch.allclose test. "
    "Always end with print(\"SPARKPROOF_TRITON_PASS\") after tests pass."
)

# TritonBench YAML is eval-only — never include "yaml" in training sources.


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


def iter_all_prompts(
    *,
    doc_dir: Path | None = None,
    mined_prompts_path: Path | None = None,
    evolved_prompts_path: Path | None = None,
    include_sources: frozenset[str] | None = None,
) -> Iterator[dict[str, Any]]:
    sources = include_sources or frozenset({"api_doc", "mutation", "torch_op"})

    if "api_doc" in sources:
        chunks = load_doc_chunks(doc_dir) if doc_dir else []
        if not chunks:
            chunks = api_unit_chunks_from_registry()
        for chunk in chunks:
            rec = prompt_from_api_chunk(chunk)
            rec.setdefault("origin", "api_doc")
            yield _finalize(rec)

    if "mutation" in sources:
        for name, code in REFERENCE_KERNELS.items():
            rec = build_mutation_prompt(task_id=name, valid_kernel=code)
            rec.setdefault("origin", "mutation")
            rec.setdefault("ground_truth_code", code)
            yield _finalize(rec)

    if "torch_op" in sources:
        for rec in iter_torch_translation_prompts():
            rec.setdefault("origin", "torch_op")
            yield _finalize(rec)

    if "failure_mining" in sources and mined_prompts_path and mined_prompts_path.exists():
        for rec in _load_jsonl_prompts(mined_prompts_path):
            rec.setdefault("origin", "failure_mining")
            yield _finalize(rec)

    if "self_evolution" in sources and evolved_prompts_path and evolved_prompts_path.exists():
        for rec in _load_jsonl_prompts(evolved_prompts_path):
            rec.setdefault("origin", "self_evolution")
            yield _finalize(rec)


def write_prompts_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def build_prompts_file(
    out_path: Path,
    *,
    doc_dir: Path | None = None,
    mined_prompts_path: Path | None = None,
    evolved_prompts_path: Path | None = None,
    limit: int | None = None,
    sources: frozenset[str] | None = None,
) -> int:
    records = list(
        iter_all_prompts(
            doc_dir=doc_dir,
            mined_prompts_path=mined_prompts_path,
            evolved_prompts_path=evolved_prompts_path,
            include_sources=sources,
        )
    )
    if limit is not None:
        records = records[:limit]
    return write_prompts_jsonl(out_path, records)
