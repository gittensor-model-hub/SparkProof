"""Run TritonBench eval (isolated from training dataset)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkproof.triton_dataset.eval_harness import TritonBenchHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name")
    parser.add_argument("--endpoint", required=True, help="API base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path (outside training dirs)")
    parser.add_argument("--config", default="configs/eval_quick.yaml")
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--bench-root", type=Path, default=None, help="SparkDistill/tritonbench root")
    parser.add_argument("--timeout", type=int, default=3600, help="evaluation timeout in seconds")
    args = parser.parse_args(argv)

    harness = TritonBenchHarness(bench_root=args.bench_root)
    report = harness.run_eval_cycle(
        endpoint=args.endpoint,
        model_name=args.model,
        out_path=args.out,
        config=args.config,
        levels=args.levels,
        timeout_seconds=args.timeout,
    )
    if report.get("status") == "error":
        print(report.get("message", "eval failed"), file=sys.stderr)
        return 1

    print(json.dumps(report.get("metrics", report), indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
