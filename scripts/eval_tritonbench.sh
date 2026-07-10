#!/usr/bin/env bash
# Run TritonBench eval (eval-only — never writes to training dataset dirs).
#
#   scripts/eval_tritonbench.sh \
#     --endpoint http://localhost:8000/v1 \
#     --model sparkdistill-triton-qwen-4b \
#     --out results/tritonbench_round1.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run sparkproof-eval-tritonbench "$@"
