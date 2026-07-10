#!/usr/bin/env bash
# Build Triton prompt jsonl from train/dev sources (never TritonBench yaml).
#
#   scripts/build_triton_prompts.sh --out prompts/triton-phase1.jsonl
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run sparkproof-build-prompts "$@"
