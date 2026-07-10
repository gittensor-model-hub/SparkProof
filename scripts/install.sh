#!/usr/bin/env bash
# Install full miner environment on Blackwell CC VM.
#
#   cd SparkProof
#   scripts/install.sh
#   cp .env.example .env    # add OPENROUTER_API_KEY
#   scripts/miner_run.sh --limit 2
#
# Options:
#   --sparkdistill PATH   SparkDistill root (default: sibling ../SparkDistill)
#   --minimal             SparkProof blackwell+gpu only (no dev extras)
#   --skip-distill        Do not sync SparkDistill
#   --skip-check          Skip post-install cc_check.sh
#   --with-axolotl        Print Axolotl install hint (not auto-installed)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sparkdistill_root=""
minimal=false
skip_distill=false
skip_check=false
with_axolotl=false

usage() {
  cat <<'EOF'
usage: scripts/install.sh [options]

Install uv + Python deps for SparkProof (Blackwell CC) and SparkDistill.

Options:
  --sparkdistill PATH   SparkDistill repo root (auto-detect sibling if omitted)
  --minimal             SparkProof runtime only (blackwell + gpu extras)
  --skip-distill        Skip SparkDistill sync
  --skip-check          Skip scripts/cc_check.sh at the end
  --with-axolotl        Show Axolotl training install instructions
  -h, --help

After install:
  cp .env.example .env          # OPENROUTER_API_KEY
  scripts/cc_check.sh           # verify Blackwell + GPU CC
  scripts/miner_run.sh --limit 2
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
    --sparkdistill) sparkdistill_root="$2"; shift 2 ;;
    --minimal) minimal=true; shift ;;
    --skip-distill) skip_distill=true; shift ;;
    --skip-check) skip_check=true; shift ;;
    --with-axolotl) with_axolotl=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

echo "=== SparkProof environment install ==="
echo "  root: $ROOT"

# --- uv ---
if ! command -v uv >/dev/null 2>&1; then
  echo ""
  echo ">>> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not on PATH after install — add ~/.local/bin to your shell profile" >&2
  exit 1
fi
echo "  uv: $(command -v uv) ($(uv --version))"

# --- python (managed uv CPython includes Python.h for Triton JIT) ---
py_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
py_major="${py_version%%.*}"
py_minor="${py_version#*.}"
if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 12 ]; }; then
  echo "error: Python 3.12+ required (found $py_version)" >&2
  exit 1
fi
echo "  python (system): $(python3 --version)"
echo ">>> pinning managed Python 3.12 (includes Python.h for Triton JIT)"
uv python install 3.12
uv python pin 3.12

# --- GPU probe (non-fatal) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  echo "  gpu: ${gpu_line:-unknown}"
else
  echo "  gpu: nvidia-smi not found (install NVIDIA driver for Blackwell validation)" >&2
fi

export SPARKPROOF_BLACKWELL_PROFILE="${SPARKPROOF_BLACKWELL_PROFILE:-workstation}"
export TRITONBENCH_BLACKWELL_PROFILE="${TRITONBENCH_BLACKWELL_PROFILE:-workstation}"
echo "  SPARKPROOF_BLACKWELL_PROFILE=$SPARKPROOF_BLACKWELL_PROFILE"

# --- SparkProof deps ---
echo ""
echo ">>> syncing SparkProof"
if [ "$minimal" = true ]; then
  uv sync --extra blackwell --extra gpu
else
  uv sync --extra dev --extra blackwell --extra gpu
fi

# numpy silences torch warnings; not pinned in pyproject
uv pip install -q numpy

# --- SparkDistill deps ---
SD=""
if [ "$skip_distill" = false ]; then
  if SD="$(resolve_sparkdistill)"; then
    echo ""
    echo ">>> syncing SparkDistill ($SD)"
    if [ "$minimal" = true ]; then
      (cd "$SD" && uv sync --extra proof)
    else
      (cd "$SD" && uv sync --extra dev --extra proof)
    fi
  else
    echo ""
    echo "warn: SparkDistill not found — skipped (clone beside SparkProof for SFT/train)" >&2
  fi
fi

# --- .env ---
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo ""
  echo ">>> created .env from .env.example — set OPENROUTER_API_KEY before miner_run"
fi

# --- Axolotl note ---
if [ "$with_axolotl" = true ]; then
  cat <<'EOF'

>>> Axolotl (training — install separately)
  Axolotl is not bundled here. On this Blackwell CC VM, after install:
    pip install axolotl
  or follow: https://github.com/axolotl-ai-cloud/axolotl
  Then: scripts/miner_run.sh --train   (from SparkProof, with SparkDistill sibling)

EOF
fi

# --- verify ---
if [ "$skip_check" = false ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo ">>> running cc_check.sh"
    scripts/cc_check.sh
  else
    echo ""
    echo ">>> skipped cc_check.sh (no GPU)"
  fi
fi

echo ""
echo "=== install complete ==="
echo "  SparkProof:  $ROOT"
[ -n "$SD" ] && echo "  SparkDistill: $SD"
echo ""
echo "Next:"
echo "  1. Edit $ROOT/.env — set OPENROUTER_API_KEY and/or YUNWU_API_KEY"
echo "     SPARKPROOF_GATEWAY=openrouter|yunwu (default: openrouter)"
echo "  2. scripts/miner_run.sh --limit 2"
[ -n "$SD" ] && echo "  3. scripts/miner_run.sh --run-id my-run --train   # needs Axolotl"
