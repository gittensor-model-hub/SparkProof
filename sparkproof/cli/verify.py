"""Verify a SparkProof dataset bundle."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--dev",
        action="store_true",
        help="skip production integrity checks (pinned generator, raw/verified consistency)",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="also verify the NVIDIA NRAS token signature against NVIDIA's JWKS "
        "(and the OpenRouter generation ledger when OPENROUTER_API_KEY is set)",
    )
    args = parser.parse_args(argv)

    report = verify_bundle(
        args.bundle,
        require_gpu_attestation=not args.allow_no_gpu_attest,
        production=not args.dev,
    )
    if args.online:
        from sparkproof.verify_online import verify_bundle_online

        online = verify_bundle_online(
            args.bundle,
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        )
        report["online"] = online
        if not online["verified"]:
            report["verified"] = False
            report["issues"] = list(report.get("issues") or []) + online["issues"]
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
