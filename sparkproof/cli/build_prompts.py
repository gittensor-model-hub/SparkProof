"""Build Triton prompt jsonl from train/dev sources (never TritonBench yaml)."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
import sys
from pathlib import Path

from sparkproof.triton_dataset.build_prompts import (
    DEFAULT_TRAIN_SOURCES_STR,
    build_prompts_file,
)
from sparkproof.triton_dataset.prompt_filters import parse_filter_set
from sparkproof.hashing import sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output prompts.jsonl")
    parser.add_argument("--report", type=Path, default=None, help="build report JSON path")
    parser.add_argument(
        "--doc-dir",
        type=Path,
        default=None,
        help="optional local Triton docs clone; otherwise auto-fetch triton.language.rst",
    )
    parser.add_argument(
        "--no-fetch-docs",
        action="store_true",
        help="do not download triton.language.rst (use --doc-dir, cache, or registry fallback)",
    )
    parser.add_argument(
        "--no-enrich-api-pages",
        action="store_true",
        help="skip Sphinx API page enrichment for api_doc symbols (Option B)",
    )
    parser.add_argument(
        "--allow-partial-docs",
        action="store_true",
        help="development only: permit missing/truncated pinned documentation sources",
    )
    parser.add_argument(
        "--capture-mutation-errors",
        action="store_true",
        help="run broken kernels on GPU and attach real compiler/runtime tails to mutation prompts",
    )
    parser.add_argument(
        "--apply-templates",
        action="store_true",
        help="wrap prompts with structured design/implementation/validation sections",
    )
    parser.add_argument(
        "--torch-shape-variants",
        action="store_true",
        help="emit adversarial shape presets for each torch_op translation prompt",
    )
    parser.add_argument(
        "--assign-dev-splits",
        action="store_true",
        help="assign ancestry-aware train/dev splits (default keeps all train)",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for mutation error capture")
    parser.add_argument("--mined-prompts", type=Path, default=None, help="failure-mined tasks jsonl")
    parser.add_argument("--evolved-prompts", type=Path, default=None, help="self-evolved tasks jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source",
        dest="filter_sources",
        action="append",
        help="only include prompts with this source (repeatable)",
    )
    parser.add_argument(
        "--task-id",
        dest="filter_task_ids",
        action="append",
        help="only include prompts with this task_id (repeatable)",
    )
    parser.add_argument(
        "--sources",
        default=DEFAULT_TRAIN_SOURCES_STR,
        help="comma-separated: api_doc,doc_semantics,doc_tutorial,mutation,torch_op,failure_mining,self_evolution",
    )
    args = parser.parse_args(argv)

    sources = frozenset(s.strip() for s in args.sources.split(",") if s.strip())
    count = build_prompts_file(
        args.out,
        doc_dir=args.doc_dir,
        mined_prompts_path=args.mined_prompts,
        evolved_prompts_path=args.evolved_prompts,
        limit=args.limit,
        sources=sources,
        auto_fetch_docs=not args.no_fetch_docs,
        enrich_api_pages=False if args.no_enrich_api_pages else None,
        strict_docs=not args.allow_partial_docs,
        capture_mutation_errors=args.capture_mutation_errors,
        apply_templates=args.apply_templates,
        torch_shape_variants=args.torch_shape_variants,
        assign_dev_splits=args.assign_dev_splits,
        dev_fraction=args.dev_fraction,
        gpu_index=args.gpu,
        filter_sources=parse_filter_set(args.filter_sources),
        filter_task_ids=parse_filter_set(args.filter_task_ids),
    )
    report_path = args.report or args.out.with_suffix(".report.json")
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    with args.out.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source_counts[record["source"]] += 1
            category_counts[record["category"]] += 1
    report = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "prompts_path": str(args.out),
        "prompts_sha256": sha256_file(str(args.out)),
        "total": count,
        "sources": dict(sorted(source_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "strict_docs": not args.allow_partial_docs,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {count} Triton prompts to {args.out}", file=sys.stderr)
    print(f"build report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
