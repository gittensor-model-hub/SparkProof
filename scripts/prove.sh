#!/usr/bin/env bash
# Prove trajectories on Blackwell: Triton compile/execute (+ optional benchmark) + GPU CC.
#
#   scripts/prove.sh --bundle bundles/my-run-001
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec uv run sparkproof-prove "$@"
