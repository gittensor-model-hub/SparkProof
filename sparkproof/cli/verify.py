"""Verify a SparkProof dataset bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkproof.verify import verify_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="bundle directory")
    parser.add_argument(
        "--allow-no-gpu-attest",
        action="store_true",
        help="accept sparkproof-2 without gpu_attestation.json (dev only)",
    )
    args = parser.parse_args(argv)

    report = verify_bundle(args.bundle, require_gpu_attestation=not args.allow_no_gpu_attest)
    print(json.dumps(report, indent=2))
    if report["verified"]:
        print("VERIFIED", file=sys.stderr)
        return 0
    print("REJECTED", file=sys.stderr)
    for issue in report["issues"]:
        print(f"  - {issue}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
