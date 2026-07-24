#!/usr/bin/env bash
# Import KernelBook / external traces as SparkProof *task seeds*, then optionally
# generate a verified multi-turn bundle on a CC VM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-prompts/kernelbook_seed.jsonl}"
LIMIT="${LIMIT:-}"
EXTRA=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi

uv run sparkproof-import-external-tasks \
  --out "${OUT}" \
  --opus-traces "${OPUS_TRACES:-ppbhatt500/kernelbook-opus4.8-multiturn-traces}" \
  --gptoss-traces "${GPTOSS_TRACES:-ppbhatt500/kernelbook-triton-reasoning-traces}" \
  "${EXTRA[@]}" \
  "$@"

echo "Next (CC VM): sparkproof-triton-generate --prompts ${OUT} --out bundles/kb-seed-\$(date +%Y%m%d) --decontaminate --orchestrate --benchmark"
