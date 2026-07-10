#!/usr/bin/env bash
# Probe Blackwell GPU + GPU CC attestation on this host (CC VM).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export SPARKPROOF_BLACKWELL_PROFILE="${SPARKPROOF_BLACKWELL_PROFILE:-workstation}"

echo "=== Triton build deps ==="
uv run python -c "
import os, sysconfig
p = os.path.join(sysconfig.get_path('include'), 'Python.h')
assert os.path.exists(p), f'missing {p} — run scripts/install.sh (uv managed Python)'
print('  Python.h:', p)
"

echo ""
echo "=== Blackwell GPU gate ==="
uv run python -c "
from sparkproof.blackwell.gpu import require_blackwell_gpu
import json
print(json.dumps(require_blackwell_gpu(0), indent=2))
"

echo ""
echo "=== GPU CC attestation (NRAS) ==="
uv run python -c "
from sparkproof.blackwell.gpu import require_blackwell_gpu
from sparkproof.gpu.attestation import attest_blackwell_gpu
import importlib.metadata as m

profile = require_blackwell_gpu(0)
result = attest_blackwell_gpu(gpu_profile=profile)
print('sdk_version:', m.version('nv-attestation-sdk'))
print('passed:', result.passed)
print('environment:', result.environment)
if result.token_sha256():
    print('token_sha256:', result.token_sha256())
"
