#!/usr/bin/env bash
# Self-generating Triton dataset pipeline (prompts → multi-candidate → prove → verify → SFT → [HF]).
#
#   scripts/run_triton_pipeline.sh --limit 2
#   scripts/run_triton_pipeline.sh --run-id triton-cc-001 --publish your-org/sparkproof-triton-v1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_id=""
prompts=""
bundle=""
limit=""
publish_repo=""
sft_out=""
sparkdistill_root=""
allow_no_gpu_attest=false
release_gate=false
extra=()

usage() {
  cat <<'EOF'
usage: scripts/run_triton_pipeline.sh [options]

  1. build_triton_prompts.sh  (api_doc + doc_semantics + doc_tutorial + mutation + torch — no TritonBench yaml)
  2. sparkproof-triton-generate (best-of-N + repair + Blackwell prove)
  3. sparkproof-verify
  4. teacher.format → SFT messages
  5. optional HF publish

Options:
  --run-id ID
  --prompts PATH          (default: prompts/triton-<run-id>.jsonl — built if missing)
  --bundle PATH           (default: bundles/<run-id>)
  --limit N
  --publish REPO_ID       HF datasets repo (needs HF_TOKEN in .env + uv sync --extra publish)
  --release-gate          run provenance/decontamination gate before HF publish
  --sft-out PATH
  --sparkdistill PATH
  --benchmark
  --no-gpu-attest
  --allow-no-gpu-attest
  -h, --help
EOF
}

resolve_sparkdistill() {
  if [ -n "$sparkdistill_root" ] && [ -d "$sparkdistill_root/teacher" ]; then
    printf '%s\n' "$sparkdistill_root"
    return 0
  fi
  for candidate in "${SPARKDISTILL_ROOT:-}" "$(cd .. && pwd)/SparkDistill" "$HOME/SparkDistill"; do
    if [ -n "$candidate" ] && [ -d "$candidate/teacher" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --prompts) prompts="$2"; shift 2 ;;
    --bundle) bundle="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --publish) publish_repo="$2"; shift 2 ;;
    --sft-out) sft_out="$2"; shift 2 ;;
    --sparkdistill) sparkdistill_root="$2"; shift 2 ;;
    --benchmark) extra+=(--benchmark); shift ;;
    --no-gpu-attest) extra+=(--no-gpu-attest); shift ;;
    --allow-no-gpu-attest) allow_no_gpu_attest=true; shift ;;
    --release-gate) release_gate=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SD="$(resolve_sparkdistill)" || { echo "error: SparkDistill not found" >&2; exit 1; }
run_id="${run_id:-triton-cc-$(date +%Y%m%d-%H%M%S)}"
prompts="${prompts:-prompts/${run_id}.jsonl}"
bundle="${bundle:-bundles/$run_id}"
sft_out="${sft_out:-$SD/data/processed/${run_id}_sft.jsonl}"
allow_no_gpu_attest="${allow_no_gpu_attest:-false}"

if [ -f .env ]; then set -a; source .env; set +a; fi
export SPARKPROOF_BLACKWELL_PROFILE="${SPARKPROOF_BLACKWELL_PROFILE:-workstation}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${YUNWU_API_KEY:-}" ]; then
  echo "error: set OPENROUTER_API_KEY or YUNWU_API_KEY in .env" >&2
  exit 1
fi

if [ ! -f "$prompts" ]; then
  echo ">>> building Triton prompts: $prompts"
  build_args=(--out "$prompts")
  if [ -n "$limit" ]; then build_args+=(--limit "$limit"); fi
  scripts/build_triton_prompts.sh "${build_args[@]}"
fi

gen_args=(--prompts "$prompts" --out "$bundle" --decontaminate --run-id "$run_id")
if [ -n "$limit" ]; then gen_args+=(--limit "$limit"); fi
if [ "${#extra[@]}" -gt 0 ]; then gen_args+=("${extra[@]}"); fi

echo ">>> multi-candidate generate + prove"
uv run sparkproof-triton-generate "${gen_args[@]}"

verify_args=(--bundle "$bundle")
if [ "$allow_no_gpu_attest" = true ]; then verify_args+=(--allow-no-gpu-attest); fi
scripts/verify.sh "${verify_args[@]}"

bundle_abs="$(cd "$(dirname "$bundle")" && pwd)/$(basename "$bundle")"
mkdir -p "$(dirname "$sft_out")"
(cd "$SD" && uv run python -m teacher.format \
  --in "$bundle_abs/trajectories.jsonl" \
  --out "$sft_out" \
  --format messages)
echo "wrote SFT: $sft_out"

if [ -n "$publish_repo" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: HF_TOKEN is required for --publish" >&2
    exit 1
  fi
  uv sync --extra publish --frozen
  echo ">>> publishing to HF: $publish_repo"
  pub_args=(--bundle "$bundle" --repo-id "$publish_repo")
  if [ "$release_gate" = true ]; then pub_args+=(--release-gate); fi
  uv run sparkproof-publish-dataset "${pub_args[@]}"
fi

echo "triton pipeline complete: $bundle"
