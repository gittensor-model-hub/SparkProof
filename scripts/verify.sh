#!/usr/bin/env bash
# Verify a SparkProof bundle (OpenRouter xhigh + Blackwell sparkproof-2).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run sparkproof-verify "$@"
