"""Build Triton prompt jsonl from train/dev sources (never TritonBench yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparkproof.triton_dataset.build_prompts import build_prompts_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output prompts.jsonl")
    parser.add_argument("--doc-dir", type=Path, default=None, help="optional Triton markdown docs tree")
    parser.add_argument("--mined-prompts", type=Path, default=None, help="failure-mined tasks jsonl")
    parser.add_argument("--evolved-prompts", type=Path, default=None, help="self-evolved tasks jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sources",
        default="api_doc,mutation,torch_op",
        help="comma-separated: api_doc,mutation,torch_op,failure_mining,self_evolution",
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
    )
    print(f"wrote {count} Triton prompts to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
