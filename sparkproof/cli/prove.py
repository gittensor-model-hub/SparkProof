"""Prove an existing SparkProof bundle on Blackwell or Hopper H100/H200 (Triton validate + GPU CC)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkproof.pipeline.blackwell import prove_blackwell_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="bundle directory with trajectories.jsonl")
    parser.add_argument(
        "--gpu", type=int, default=0, help="CUDA device index (must be Blackwell or Hopper H100/H200)"
    )
    parser.add_argument("--benchmark", action="store_true", help="require lightweight benchmark score floor")
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help="run anti-cheat and adversarial multi-seed validation at the proving gate",
    )
    parser.add_argument(
        "--capture-ir",
        action="store_true",
        help="capture TTIR/TTGIR/PTX artifacts during proving",
    )
    parser.add_argument(
        "--no-gpu-attest",
        action="store_true",
        help="skip NVIDIA GPU CC attestation (Triton validation only — not valid for production PRs)",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="minimum fraction of raw trajectories that must pass validation (0.0–1.0)",
    )
    args = parser.parse_args(argv)

    report = prove_blackwell_bundle(
        args.bundle,
        gpu_index=args.gpu,
        benchmark=args.benchmark,
        strict_validate=args.strict_validate,
        capture_ir=args.capture_ir,
        attest_gpu=not args.no_gpu_attest,
        min_pass_rate=args.min_pass_rate,
    )
    print(json.dumps(report, indent=2), file=sys.stderr)
    gpu_architecture = report["gpu_profile"].get("gpu_architecture", "blackwell")
    print(f"verified {report['verified_count']}/{report['raw_count']} on {gpu_architecture}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
