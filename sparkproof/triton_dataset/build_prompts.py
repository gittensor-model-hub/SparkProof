"""Merge prompt sources A–E into SparkProof-compatible jsonl."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator

from sparkproof.gpu.architecture import ARCH_BLACKWELL, sm_label
from sparkproof.triton_dataset.doc_chunks import (
    CHUNK_API_SYMBOL,
    api_unit_chunks_from_registry,
    doc_kinds_for_sources,
    load_doc_chunks,
    prompt_from_doc_chunk,
)
from sparkproof.triton_dataset.error_capture import enrich_mutation_prompt
from sparkproof.triton_dataset.prompt_templates import apply_prompt_template
from sparkproof.triton_dataset.prompt_filters import prompt_matches_filters
from sparkproof.triton_dataset.reference_kernels import REFERENCE_KERNELS
from sparkproof.triton_dataset.run_seed import SAMPLING_POLICY_VERSION, generate_run_seed, sampling_seed
from sparkproof.triton_dataset.schema import validate_prompt_record
from sparkproof.triton_dataset.stratified_sampling import catalog_sha256, stratified_sample
from sparkproof.triton_dataset.task_policy import normalize_train_task
from sparkproof.triton_dataset.mutator import iter_mutation_prompts
from sparkproof.triton_dataset.torch_ops import iter_torch_translation_prompts

def default_system(gpu_architecture: str = ARCH_BLACKWELL) -> str:
    return (
        f"You are a Triton 3.7.1 GPU kernel expert for {sm_label(gpu_architecture)}. "
        "Write complete runnable Python with @triton.jit, launcher, and torch.allclose test. "
        "Always end with print(\"SPARKPROOF_TRITON_PASS\") after tests pass."
    )


DEFAULT_SYSTEM = default_system(ARCH_BLACKWELL)

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


def _finalize(record: dict[str, Any], *, gpu_architecture: str = ARCH_BLACKWELL) -> dict[str, Any]:
    out = {
        "prompt": record["prompt"],
        "system": record.get("system", default_system(gpu_architecture)),
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
        "shape_class",
        "doc_chunk_id",
        "level",
        "title",
        "task_family",
        "prompt_template",
        "captured_error",
        "captured_failure_class",
        "parent_id",
        "evolution_ops",
        "gpu_architecture",
        "source_dataset",
        "source_uuid",
        "entry_point",
        "licenses",
        "repo_name",
        "repo_link",
        "repair_hint_kernel",
    ):
        if key in record and record[key] is not None:
            out[key] = record[key]
    out.setdefault("origin", out.get("source", "unknown"))
    out.setdefault("split", "train")
    out.setdefault("gpu_architecture", gpu_architecture)
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
    gpu_architecture: str = ARCH_BLACKWELL,
) -> dict[str, Any] | None:
    finalized = _finalize(record, gpu_architecture=gpu_architecture)
    if not prompt_matches_filters(finalized, sources=filter_sources, task_ids=filter_task_ids):
        return None
    return finalized


def iter_all_prompts(
    *,
    doc_dir: Path | None = None,
    mined_prompts_path: Path | None = None,
    evolved_prompts_path: Path | None = None,
    seed_prompts_path: Path | None = None,
    include_sources: frozenset[str] | None = None,
    auto_fetch_docs: bool = True,
    enrich_api_pages: bool | None = None,
    strict_docs: bool = False,
    capture_mutation_errors: bool = False,
    apply_templates: bool = False,
    torch_shape_variants: bool = False,
    gpu_index: int = 0,
    gpu_architecture: str = ARCH_BLACKWELL,
    filter_sources: frozenset[str] | None = None,
    filter_task_ids: frozenset[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the full eligible prompt catalog (no truncation).

    Selecting a subset (e.g. for `--limit`) is `build_prompts_file`'s job via
    stratified sampling — this function only decides *eligibility* (sources,
    filters), never *how many*, so callers always see the complete catalog to
    sample from. `gpu_architecture` decides the SM/GPU label baked into prompt
    text (see `sparkproof.gpu.architecture`); it does not need to match
    `gpu_index`'s actual hardware except where `capture_mutation_errors=True`
    actually executes kernels, which stamps the real detected architecture.
    """
    sources = include_sources or DEFAULT_TRAIN_SOURCES

    def emit(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
        out = _yield_if_matches(
            record, filter_sources=filter_sources, filter_task_ids=filter_task_ids, gpu_architecture=gpu_architecture
        )
        if out is not None:
            yield out

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
            chunks = api_unit_chunks_from_registry(gpu_architecture=gpu_architecture)
        for chunk in chunks:
            rec = prompt_from_doc_chunk(chunk, gpu_architecture=gpu_architecture)
            rec.setdefault("origin", rec.get("source", "api_doc"))
            if apply_templates:
                rec = apply_prompt_template(rec, gpu_architecture=gpu_architecture)
            yield from emit(rec)

    if "mutation" in sources:
        for name, code in REFERENCE_KERNELS.items():
            for rec in iter_mutation_prompts(task_id=name, valid_kernel=code, gpu_architecture=gpu_architecture):
                rec.setdefault("origin", "mutation")
                rec.setdefault("ground_truth_code", code)
                if capture_mutation_errors:
                    rec = enrich_mutation_prompt(rec, gpu_index=gpu_index)
                if apply_templates:
                    rec = apply_prompt_template(rec, gpu_architecture=gpu_architecture)
                yield from emit(rec)

    if "torch_op" in sources:
        for rec in iter_torch_translation_prompts(
            include_shape_variants=torch_shape_variants, gpu_architecture=gpu_architecture
        ):
            rec.setdefault("origin", "torch_op")
            if apply_templates:
                rec = apply_prompt_template(rec, gpu_architecture=gpu_architecture)
            yield from emit(rec)

    if "failure_mining" in sources and mined_prompts_path and mined_prompts_path.exists():
        for rec in _load_jsonl_prompts(mined_prompts_path):
            rec.setdefault("origin", "failure_mining")
            yield from emit(rec)

    if "self_evolution" in sources and evolved_prompts_path and evolved_prompts_path.exists():
        for rec in _load_jsonl_prompts(evolved_prompts_path):
            rec.setdefault("origin", "self_evolution")
            yield from emit(rec)

    if "kernelbook_seed" in sources and seed_prompts_path and seed_prompts_path.exists():
        for rec in _load_jsonl_prompts(seed_prompts_path):
            rec.setdefault("source", "kernelbook_seed")
            rec.setdefault("origin", "kernelbook_seed")
            if apply_templates:
                rec = apply_prompt_template(rec, gpu_architecture=gpu_architecture)
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
    seed_prompts_path: Path | None = None,
    limit: int | None = None,
    sources: frozenset[str] | None = None,
    auto_fetch_docs: bool = True,
    enrich_api_pages: bool | None = None,
    strict_docs: bool = False,
    capture_mutation_errors: bool = False,
    apply_templates: bool = False,
    torch_shape_variants: bool = False,
    assign_dev_splits: bool = False,
    dev_fraction: float = 0.1,
    gpu_index: int = 0,
    gpu_architecture: str = ARCH_BLACKWELL,
    filter_sources: frozenset[str] | None = None,
    filter_task_ids: frozenset[str] | None = None,
    run_seed: str | None = None,
    sampling_policy: str = SAMPLING_POLICY_VERSION,
    max_bucket_share: float = 0.25,
) -> int:
    """Build the eligible catalog, sample up to `limit` deterministically from
    `run_seed`, and write it out. Sampling provenance (policy, run_seed,
    catalog_sha256, bucket_counts) is written alongside as
    `<out_path>.sampling.json` — auto-generating and persisting `run_seed`
    when omitted so the exact run can be replayed later. `gpu_architecture`
    (default Blackwell) selects the SM/GPU label baked into prompt text; pass
    the value from `require_supported_gpu`'s detected profile to match
    prompts to whatever hardware will validate them.
    """
    from sparkproof.triton_dataset.dataset_split import assign_splits

    catalog = list(
        iter_all_prompts(
            doc_dir=doc_dir,
            mined_prompts_path=mined_prompts_path,
            evolved_prompts_path=evolved_prompts_path,
            seed_prompts_path=seed_prompts_path,
            include_sources=sources,
            auto_fetch_docs=auto_fetch_docs,
            enrich_api_pages=enrich_api_pages,
            strict_docs=strict_docs,
            capture_mutation_errors=capture_mutation_errors,
            apply_templates=apply_templates,
            torch_shape_variants=torch_shape_variants,
            gpu_index=gpu_index,
            gpu_architecture=gpu_architecture,
            filter_sources=filter_sources,
            filter_task_ids=filter_task_ids,
        )
    )
    catalog_hash = catalog_sha256(catalog)
    resolved_run_seed = run_seed or generate_run_seed()
    seed = sampling_seed(catalog_hash, resolved_run_seed, sampling_policy)
    sampled, bucket_counts = stratified_sample(catalog, limit=limit, seed=seed, max_share=max_bucket_share)

    records: Iterable[dict[str, Any]] = sampled
    if assign_dev_splits:
        records = assign_splits(records, dev_fraction=dev_fraction)
    count = write_prompts_jsonl(out_path, records)

    sampling_report_path = out_path.with_suffix(".sampling.json")
    sampling_report_path.write_text(
        json.dumps(
            {
                "policy": sampling_policy,
                "run_seed": resolved_run_seed,
                "catalog_sha256": catalog_hash,
                "catalog_size": len(catalog),
                "requested_limit": limit,
                "max_bucket_share": max_bucket_share,
                "bucket_counts": bucket_counts,
                "gpu_architecture": gpu_architecture,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return count
