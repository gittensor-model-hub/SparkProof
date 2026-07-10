#!/usr/bin/env bash
# One-shot miner pipeline on Blackwell CC VM (no Polaris).
#
#   cd SparkProof
#   cp .env.example .env    # OPENROUTER_API_KEY
#   scripts/install.sh
#   scripts/miner_run.sh --limit 2
#   scripts/miner_run.sh --run-id my-run-001 --train
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_id=""
prompts=""
bundle=""
sft_out=""
sparkdistill_root=""
limit=""
do_setup=false
do_train=false
dry_run_train=false
skip_sft=false
allow_no_gpu_attest=false
gateway=""
gen_extra=()

usage() {
  cat <<'EOF'
usage: scripts/miner_run.sh [options]

Miner one-shot on Blackwell CC: [setup] → generate → prove → verify → SFT → [train]

Options:
  --setup                  Run scripts/install.sh (uv + all deps)
  --run-id ID              Bundle name under bundles/ (default: phase1-cc-YYYYMMDD-HHMMSS)
  --prompts PATH           Prompt jsonl (default: <SparkDistill>/data/prompts/phase1.jsonl)
  --bundle PATH            Bundle dir (default: bundles/<run-id>)
  --sft-out PATH           SFT messages jsonl (default: <SparkDistill>/data/processed/<run-id>_sft.jsonl)
  --sparkdistill PATH      SparkDistill repo root (auto-detected if omitted)
  --limit N                Cap prompt count (smoke tests)
  --train                  Run Axolotl SFT after formatting (qwen3.5-4b-phase1 recipe)
  --dry-run-train          Print train command only (--train)
  --skip-sft               Stop after bundle verify (no teacher.format step)
  --skip-blackwell         Dev: skip GPU Triton validation (not for production PRs)
  --no-gpu-attest          Dev: skip GPU CC attestation during prove
  --allow-no-gpu-attest    Dev: verify accepts bundle without gpu_attestation.json
  --gateway GATEWAY        openrouter | yunwu (default: SPARKPROOF_GATEWAY or openrouter)
  -h, --help

Environment:
  OPENROUTER_API_KEY              Required — set in SparkProof/.env
  SPARKPROOF_BLACKWELL_PROFILE    Default: workstation
  SPARKDISTILL_ROOT               Override SparkDistill path
EOF
}

resolve_sparkdistill() {
  if [ -n "$sparkdistill_root" ] && [ -d "$sparkdistill_root/teacher" ]; then
    printf '%s\n' "$sparkdistill_root"
    return 0
  fi
  local candidate
  for candidate in \
    "${SPARKDISTILL_ROOT:-}" \
    "$(cd "$ROOT/.." && pwd)/SparkDistill" \
    "$HOME/SparkDistill"; do
  if [ -n "$candidate" ] && [ -d "$candidate/teacher" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  done
  return 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --setup) do_setup=true; shift ;;
    --run-id) run_id="$2"; shift 2 ;;
    --prompts) prompts="$2"; shift 2 ;;
    --bundle) bundle="$2"; shift 2 ;;
    --sft-out) sft_out="$2"; shift 2 ;;
    --sparkdistill) sparkdistill_root="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --train) do_train=true; shift ;;
    --dry-run-train) dry_run_train=true; do_train=true; shift ;;
    --skip-sft) skip_sft=true; shift ;;
    --skip-blackwell) gen_extra+=(--skip-blackwell); shift ;;
    --no-gpu-attest) gen_extra+=(--no-gpu-attest); shift ;;
    --allow-no-gpu-attest) allow_no_gpu_attest=true; shift ;;
    --benchmark) gen_extra+=(--benchmark); shift ;;
    --gateway) gateway="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SD="$(resolve_sparkdistill)" || {
  echo "error: SparkDistill not found — clone beside SparkProof or pass --sparkdistill" >&2
  exit 1
}

if [ "$do_setup" = true ]; then
  scripts/install.sh --sparkdistill "$SD"
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export SPARKPROOF_BLACKWELL_PROFILE="${SPARKPROOF_BLACKWELL_PROFILE:-workstation}"
if [ -n "$gateway" ]; then
  export SPARKPROOF_GATEWAY="$gateway"
fi
export SPARKPROOF_GATEWAY="${SPARKPROOF_GATEWAY:-openrouter}"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${YUNWU_API_KEY:-}" ]; then
  echo "error: set OPENROUTER_API_KEY or YUNWU_API_KEY in .env (see .env.example)" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — re-run with --setup" >&2
  exit 1
fi

run_id="${run_id:-phase1-cc-$(date +%Y%m%d-%H%M%S)}"
prompts="${prompts:-$SD/data/prompts/phase1.jsonl}"
bundle="${bundle:-bundles/$run_id}"
sft_out="${sft_out:-$SD/data/processed/${run_id}_sft.jsonl}"

if [ ! -f "$prompts" ]; then
  echo "error: prompts not found: $prompts" >&2
  exit 1
fi

mkdir -p "$(dirname "$bundle")"
if [ "$skip_sft" = false ]; then
  mkdir -p "$(dirname "$sft_out")"
fi

echo "=== SparkProof miner run ==="
echo "  sparkdistill: $SD"
echo "  prompts:      $prompts"
echo "  bundle:       $bundle"
if [ "$skip_sft" = false ]; then
  echo "  sft:          $sft_out"
fi
if [ -n "$limit" ]; then
  echo "  limit:        $limit"
fi

gen_args=(--prompts "$prompts" --out "$bundle")
if [ -n "$limit" ]; then
  gen_args+=(--limit "$limit")
fi
if [ "${#gen_extra[@]}" -gt 0 ]; then
  gen_args+=("${gen_extra[@]}")
fi

scripts/generate.sh "${gen_args[@]}"

verify_args=(--bundle "$bundle")
if [ "$allow_no_gpu_attest" = true ]; then
  verify_args+=(--allow-no-gpu-attest)
fi
scripts/verify.sh "${verify_args[@]}"

if [ "$skip_sft" = false ]; then
  bundle_abs="$(cd "$(dirname "$bundle")" && pwd)/$(basename "$bundle")"
  (cd "$SD" && uv run python -m teacher.format \
    --in "$bundle_abs/trajectories.jsonl" \
    --out "$sft_out" \
    --format messages)
  echo "wrote SFT messages: $sft_out"

  recipe_sft="$SD/data/processed/phase1_sft.jsonl"
  if [ "$sft_out" != "$recipe_sft" ]; then
    cp -f "$sft_out" "$recipe_sft"
    echo "copied to recipe default: $recipe_sft"
  fi
fi

if [ "$do_train" = true ]; then
  train_args=(recipes/qwen3.5-4b-phase1/sft.yaml)
  if [ "$dry_run_train" = true ]; then
    train_args+=(--dry-run)
  fi
  (cd "$SD" && scripts/train.sh "${train_args[@]}")
fi

echo ""
echo "miner run complete."
echo "  bundle:  $bundle"
echo "  verify:  scripts/verify.sh --bundle $bundle"
if [ "$skip_sft" = false ]; then
  echo "  sft:     $sft_out"
  if [ "$do_train" = false ]; then
    echo "  train:   cd $SD && scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml"
  fi
fi
