#!/usr/bin/env bash
# Full diverse dataset: all doc sources + mutation + torch_op → prove → verify → SFT → [train].
#
#   scripts/run_full_diverse.sh --limit 2
#   scripts/run_full_diverse.sh --run-id diverse-001 --train
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_id=""
prompts=""
bundle=""
limit=""
filter_source_args=()
filter_task_id_args=()
no_enrich_api_pages=false
apply_templates=false
assign_dev_splits=false
torch_shape_variants=false
strict_validate=false
capture_ir=false
export_dpo=""
do_train=false
dry_run_train=false
allow_no_gpu_attest=false
gateway=""
extra=()

usage() {
  cat <<'EOF'
usage: scripts/run_full_diverse.sh [options]

All train sources (api_doc + doc_semantics + doc_tutorial + mutation + torch_op)
→ sparkproof-triton-generate → verify → summarize → SFT → [train]

Options:
  --run-id ID              default: diverse-YYYYMMDD-HHMMSS
  --prompts PATH           default: prompts/<run-id>.jsonl
  --bundle PATH            default: bundles/<run-id>
  --limit N                cap seeds (smoke tests)
  --source SOURCE          only this prompt source at generate time (repeatable)
  --task-id ID             only this task_id at generate time (repeatable)
  --no-enrich-api-pages    skip Sphinx API page enrichment (Option B)
  --apply-templates        structured design/implementation/validation prompt sections
  --assign-dev-splits      ancestry-aware train/dev split at prompt build
  --torch-shape-variants   adversarial shape presets for torch_op prompts
  --strict-validate        anti-cheat + multi-seed adversarial GPU validation
  --capture-ir             attach TTIR/TTGIR artifacts when available
  --export-dpo PATH        write optimization DPO pairs from adjudication
  --train                  Axolotl SFT via SparkDistill (qwen3.5-4b-phase1)
  --dry-run-train          print train command only
  --benchmark
  --no-gpu-attest
  --allow-no-gpu-attest
  --gateway GATEWAY        openrouter | yunwu
  -h, --help
EOF
}

resolve_sparkdistill() {
  for candidate in "${SPARKDISTILL_ROOT:-}" "$(cd "$ROOT/.." && pwd)/SparkDistill" "$HOME/SparkDistill"; do
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
    --source) filter_source_args+=(--source "$2"); shift 2 ;;
    --task-id) filter_task_id_args+=(--task-id "$2"); shift 2 ;;
    --no-enrich-api-pages) no_enrich_api_pages=true; shift ;;
    --apply-templates) apply_templates=true; shift ;;
    --assign-dev-splits) assign_dev_splits=true; shift ;;
    --torch-shape-variants) torch_shape_variants=true; shift ;;
    --strict-validate) strict_validate=true; shift ;;
    --capture-ir) capture_ir=true; shift ;;
    --export-dpo) export_dpo="$2"; extra+=(--benchmark); shift 2 ;;
    --train) do_train=true; shift ;;
    --dry-run-train) dry_run_train=true; do_train=true; shift ;;
    --benchmark) extra+=(--benchmark); shift ;;
    --no-gpu-attest) extra+=(--no-gpu-attest); shift ;;
    --allow-no-gpu-attest) allow_no_gpu_attest=true; shift ;;
    --gateway) gateway="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SD="$(resolve_sparkdistill)" || { echo "error: SparkDistill not found (clone sibling repo)" >&2; exit 1; }

if [ -f .env ]; then set -a; source .env; set +a; fi
export SPARKPROOF_BLACKWELL_PROFILE="${SPARKPROOF_BLACKWELL_PROFILE:-workstation}"
if [ -n "$gateway" ]; then export SPARKPROOF_GATEWAY="$gateway"; fi
export SPARKPROOF_GATEWAY="${SPARKPROOF_GATEWAY:-openrouter}"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${YUNWU_API_KEY:-}" ]; then
  echo "error: set OPENROUTER_API_KEY or YUNWU_API_KEY in .env" >&2
  exit 1
fi

run_id="${run_id:-diverse-$(date +%Y%m%d-%H%M%S)}"
prompts="${prompts:-prompts/${run_id}.jsonl}"
bundle="${bundle:-bundles/$run_id}"
sft_out="$SD/data/processed/${run_id}_sft.jsonl"

echo ">>> building full diverse prompts (doc + mutation + torch_op)"
build_args=(--out "$prompts")
if [ -n "$limit" ]; then build_args+=(--limit "$limit"); fi
if [ "$no_enrich_api_pages" = true ]; then build_args+=(--no-enrich-api-pages); fi
if [ "$apply_templates" = true ]; then build_args+=(--apply-templates); fi
if [ "$assign_dev_splits" = true ]; then build_args+=(--assign-dev-splits); fi
if [ "$torch_shape_variants" = true ]; then build_args+=(--torch-shape-variants); fi
if [ "${#filter_source_args[@]}" -gt 0 ]; then build_args+=("${filter_source_args[@]}"); fi
if [ "${#filter_task_id_args[@]}" -gt 0 ]; then build_args+=("${filter_task_id_args[@]}"); fi
uv run sparkproof-build-prompts "${build_args[@]}"

gen_args=(--prompts "$prompts" --out "$bundle" --decontaminate)
if [ -n "$limit" ]; then gen_args+=(--limit "$limit"); fi
if [ "${#filter_source_args[@]}" -gt 0 ]; then gen_args+=("${filter_source_args[@]}"); fi
if [ "${#filter_task_id_args[@]}" -gt 0 ]; then gen_args+=("${filter_task_id_args[@]}"); fi
if [ "${#extra[@]}" -gt 0 ]; then gen_args+=("${extra[@]}"); fi
if [ "$strict_validate" = true ]; then gen_args+=(--strict-validate); fi
if [ "$capture_ir" = true ]; then gen_args+=(--capture-ir); fi
if [ -n "$export_dpo" ]; then gen_args+=(--export-dpo "$export_dpo"); fi

echo ">>> teacher generate + Blackwell prove"
uv run sparkproof-triton-generate "${gen_args[@]}"

verify_args=(--bundle "$bundle")
if [ "$allow_no_gpu_attest" = true ]; then verify_args+=(--allow-no-gpu-attest); fi
scripts/verify.sh "${verify_args[@]}"

uv run sparkproof-summarize-bundle --bundle "$bundle"

bundle_abs="$(cd "$(dirname "$bundle")" && pwd)/$(basename "$bundle")"
mkdir -p "$(dirname "$sft_out")"
echo ">>> SFT messages for Qwen"
(cd "$SD" && uv run python -m teacher.format \
  --in "$bundle_abs/trajectories.jsonl" \
  --out "$sft_out" \
  --format messages)
echo "wrote SFT: $sft_out"

if [ "$do_train" = true ]; then
  train_args=(recipes/qwen3.5-4b-phase1/sft.yaml)
  if [ "$dry_run_train" = true ]; then train_args+=(--dry-run); fi
  echo ">>> Axolotl SFT"
  (cd "$SD" && scripts/train.sh "${train_args[@]}")
fi

echo "full diverse pipeline complete"
echo "  prompts: $prompts"
echo "  bundle:  $bundle"
echo "  sft:     $sft_out"
