#!/usr/bin/env bash
# Auto-configure yunwu model slugs then run limit-1 Triton smoke on CC.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="${HOME}/.local/bin:${PATH}"
if [ -f .env ]; then set -a; source .env; set +a; fi
export SPARKPROOF_GATEWAY=yunwu
uv run sparkproof-yunwu-probe --auto --write-env .env
uv run sparkproof-yunwu-probe --test
exec scripts/run_triton_pipeline.sh --limit 1 --run-id "yunwu-smoke-$(date +%Y%m%d-%H%M%S)" --allow-no-gpu-attest "$@"
