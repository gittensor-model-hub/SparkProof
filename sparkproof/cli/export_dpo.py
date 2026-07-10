"""Export DPO preference pairs from a generation bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparkproof.triton_dataset.dpo_export import (
    enrich_adjudication_with_responses,
    export_dpo_jsonl,
    load_adjudication,
    write_dpo_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="generation bundle directory")
    parser.add_argument("--out", type=Path, required=True, help="output DPO jsonl path")
    parser.add_argument("--min-speedup", type=float, default=0.03)
    args = parser.parse_args(argv)

    adjudication_path = args.bundle / "adjudication.jsonl"
    if not adjudication_path.is_file():
        adjudication_path = args.bundle / "generation_adjudication.jsonl"
    if not adjudication_path.is_file():
        print(f"error: no adjudication jsonl in {args.bundle}", file=sys.stderr)
        return 2

    rows = enrich_adjudication_with_responses(
        load_adjudication(adjudication_path),
        checkpoint_path=args.bundle / "generation_checkpoint.jsonl",
    )
    pairs = export_dpo_jsonl(rows, min_speedup=args.min_speedup)
    count = write_dpo_jsonl(args.out, pairs)
    print(f"exported {count} DPO pairs to {args.out}", file=sys.stderr)
    if count == 0:
        print(
            "warning: no task had two passing, monitored benchmarks above the speedup threshold",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
