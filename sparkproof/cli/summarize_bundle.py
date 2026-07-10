"""Print pass-rate summary for a SparkProof bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparkproof.bundle_summary import format_summary, summarize_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="bundle directory")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    report = summarize_bundle(args.bundle)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
