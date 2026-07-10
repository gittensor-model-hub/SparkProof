#!/usr/bin/env bash
# Build the next deterministic prompt round from a prior generation bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bundle=""
out=""
extra=()

usage() {
  echo "usage: scripts/build_next_round.sh --bundle BUNDLE --out PROMPTS_JSONL [build options]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) bundle="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) extra+=("$1"); shift ;;
  esac
done

if [ -z "$bundle" ] || [ -z "$out" ]; then
  usage
  exit 2
fi

mined="$bundle/mined_tasks.jsonl"
evolved="$bundle/evolved_tasks.jsonl"
if [ ! -s "$mined" ] && [ ! -s "$evolved" ]; then
  echo "error: bundle contains no mined_tasks.jsonl or evolved_tasks.jsonl rows" >&2
  exit 1
fi

args=(
  --out "$out"
  --sources api_doc,doc_semantics,doc_tutorial,mutation,torch_op,failure_mining,self_evolution
)
if [ -s "$mined" ]; then args+=(--mined-prompts "$mined"); fi
if [ -s "$evolved" ]; then args+=(--evolved-prompts "$evolved"); fi
if [ "${#extra[@]}" -gt 0 ]; then args+=("${extra[@]}"); fi

uv run sparkproof-build-prompts "${args[@]}"
