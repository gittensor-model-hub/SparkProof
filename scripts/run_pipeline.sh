#!/usr/bin/env bash
# Generate → verify → optional SFT. Prefer scripts/miner_run.sh for one-shot miner defaults.
#
#   scripts/run_pipeline.sh \
#     --prompts ../SparkDistill/data/prompts/phase1.jsonl \
#     --bundle bundles/phase1-cc-001 \
#     --sft-out ../SparkDistill/data/processed/phase1_sft.jsonl
#
# Requires: OPENROUTER_API_KEY in .env, uv, torch+triton (--extra blackwell), nv-attestation (--extra gpu)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prompts=""
bundle=""
sft_out=""
limit=""
skip_sft=false
extra_gen=()

while [ $# -gt 0 ]; do
  case "$1" in
    --prompts) prompts="$2"; shift 2 ;;
    --bundle) bundle="$2"; shift 2 ;;
    --sft-out) sft_out="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --skip-sft) skip_sft=true; shift ;;
    *) extra_gen+=("$1"); shift ;;
  esac
done

if [ -z "$prompts" ] || [ -z "$bundle" ]; then
  echo "usage: scripts/run_pipeline.sh --prompts <jsonl> --bundle <dir> [--sft-out <jsonl>] [--limit N] [--skip-sft]" >&2
  exit 1
fi

gen_args=(--prompts "$prompts" --out "$bundle" "${extra_gen[@]:-}")
if [ -n "$limit" ]; then gen_args+=(--limit "$limit"); fi

scripts/generate.sh "${gen_args[@]}"
scripts/verify.sh --bundle "$bundle"

if [ "$skip_sft" = false ] && [ -n "$sft_out" ]; then
  sparkdistill_root="${SPARKDISTILL_ROOT:-$(cd .. && pwd)/SparkDistill}"
  if [ -d "$sparkdistill_root/teacher" ]; then
    (cd "$sparkdistill_root" && uv run python -m teacher.format \
      --in "$(cd "$(dirname "$bundle")" && pwd)/$(basename "$bundle")/trajectories.jsonl" \
      --out "$sft_out" \
      --format messages)
    echo "wrote SFT messages to $sft_out"
  else
    echo "SparkDistill not found beside prompts — skip SFT conversion" >&2
  fi
fi

echo "pipeline complete: $bundle"
