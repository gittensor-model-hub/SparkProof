"""Download SparkDistill's pinned accepted-registry snapshot for miner novelty checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sparkproof.triton_dataset.registry_snapshot import (
    DEFAULT_MINING_DATASET_REPO,
    download_registry_snapshot,
    verify_snapshot_against_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_MINING_DATASET_REPO,
        help=f"canonical mining HF dataset repo (default: {DEFAULT_MINING_DATASET_REPO})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="directory to copy snapshot artifacts into (default: cwd)",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="verify an existing snapshot JSONL against live mix_manifest pins (no download)",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        if args.verify_only is not None:
            from sparkproof.triton_dataset.registry_snapshot import download_mining_manifest

            manifest = download_mining_manifest(args.repo_id, hf_token=token)
            issues = verify_snapshot_against_manifest(args.verify_only, manifest)
            if issues:
                print(json.dumps({"verified": False, "issues": issues}, indent=2))
                return 1
            print(
                json.dumps(
                    {
                        "verified": True,
                        "snapshot_path": str(args.verify_only),
                        "rows_total": sum(
                            1
                            for line in args.verify_only.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ),
                        "sha256": manifest.get("accepted_registry_snapshot_sha256"),
                    },
                    indent=2,
                )
            )
            return 0

        report = download_registry_snapshot(
            repo_id=args.repo_id,
            out_dir=args.out_dir,
            hf_token=token,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"download registry snapshot failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "repo_id": report.repo_id,
                "snapshot_path": str(report.snapshot_path),
                "task_ids_path": str(report.task_ids_path) if report.task_ids_path else None,
                "rows_total": report.rows_total,
                "sha256": report.sha256,
                "verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
