#!/usr/bin/env bash
# Summarize teacher + Blackwell pass rates for a bundle.
#
#   scripts/summarize_bundle.sh --bundle bundles/doc-full-001
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run sparkproof-summarize-bundle "$@"
