#!/usr/bin/env bash
# Generate + prove a SparkProof bundle on Blackwell CC (OpenRouter xhigh → Triton validate → GPU attest).
#
#   scripts/generate.sh --prompts ../SparkDistill/data/prompts/phase1.jsonl --out bundles/run-001
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
export SPARKPROOF_BLACKWELL_PROFILE="${SPARKPROOF_BLACKWELL_PROFILE:-workstation}"
export SPARKPROOF_GATEWAY="${SPARKPROOF_GATEWAY:-openrouter}"
exec uv run sparkproof-generate "$@"
