"""Assign ancestry-aware train/dev splits to a prompts jsonl file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkproof.triton_dataset.dataset_split import assign_splits, summarize_splits
from sparkproof.triton_dataset.build_prompts import write_prompts_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="input prompts jsonl")
    parser.add_argument("--out", type=Path, required=True, help="output prompts jsonl with split labels")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    split_records = assign_splits(records, dev_fraction=args.dev_fraction)
    count = write_prompts_jsonl(args.out, split_records)
    summary = summarize_splits(split_records)
    print(f"wrote {count} prompts with splits {summary} to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
