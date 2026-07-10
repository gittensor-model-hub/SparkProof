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
    parser.add_argument("--release-gate", action="store_true", help="run decontamination + provenance gate before publish")
    parser.add_argument("--dataset-version", default="triton-distill-v0.2")
    args = parser.parse_args(argv)

    if args.release_gate:
        run_release_gate(args.bundle, dataset_version=args.dataset_version)

    url = publish_bundle_to_hf(bundle_dir=args.bundle, repo_id=args.repo_id, private=args.private)
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
