"""Publish a verified SparkProof bundle to Hugging Face datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparkproof.publish.hf_dataset import publish_bundle_to_hf
from sparkproof.triton_dataset.release_gate import run_release_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="HF datasets repo id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-release-gate",
        action="store_true",
        help="development only: publish without decontamination and provenance checks",
    )
    parser.add_argument("--release-gate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--problems-dir",
        type=Path,
        default=None,
        help="TritonBench problems directory used by the release gate",
    )
    parser.add_argument(
        "--benchmark-py-dir",
        type=Path,
        default=None,
        help="optional held-out benchmark Python tree for structural fingerprints",
    )
    parser.add_argument("--dataset-version", default="triton-distill-v0.2")
    parser.add_argument(
        "--registry-snapshot",
        type=Path,
        default=None,
        help="optional JSONL of previously-accepted rows to check novelty against",
    )
    args = parser.parse_args(argv)

    try:
        if not args.skip_release_gate:
            run_release_gate(
                args.bundle,
                dataset_version=args.dataset_version,
                problems_dir=args.problems_dir,
                benchmark_py_dir=args.benchmark_py_dir,
                registry_snapshot_path=args.registry_snapshot,
            )

        url = publish_bundle_to_hf(bundle_dir=args.bundle, repo_id=args.repo_id, private=args.private)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
