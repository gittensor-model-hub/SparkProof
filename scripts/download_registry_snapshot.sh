#!/usr/bin/env bash
# Download SparkDistill's pinned accepted-registry snapshot for SparkProof novelty checks.
#   scripts/download_registry_snapshot.sh
#   scripts/download_registry_snapshot.sh --out-dir ./snapshots
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra publish --frozen
exec uv run sparkproof-download-registry-snapshot "$@"
