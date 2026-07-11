import base64
import json

import pytest

pytest.importorskip("jwt")

from sparkproof.gpu.attestation import _decode_gpu_claims
from sparkproof.hashing import dataset_attestation_nonce


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unverified_jwt(claims: dict) -> str:
    """Build a JWT-shaped string with a real base64url payload; signature is
    irrelevant since _decode_gpu_claims decodes with verify_signature=False,
    matching how it treats NRAS's actually-signed tokens."""
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps(claims).encode())
    return f"{header}.{payload}.unsigned"


def test_decode_gpu_claims_extracts_real_nras_jwt_not_wrapper():
    """The SDK's outer token is [["JWT", wrapper], {"REMOTE_GPU_CLAIMS": [["JWT", real]]}] —
    the wrapper carries no hardware claims; the real NRAS-signed evidence (with the
    echoed nonce) is nested under REMOTE_GPU_CLAIMS."""
    wrapper = _unverified_jwt({"iss": "NV-Attestation-SDK", "iat": 1, "exp": 2})
    real = _unverified_jwt({"iss": "https://nras.attestation.nvidia.com", "eat_nonce": "abc123", "sub": "NVIDIA-PLATFORM-ATTESTATION"})
    token = json.dumps([["JWT", wrapper], {"REMOTE_GPU_CLAIMS": [["JWT", real]]}])

    claims = _decode_gpu_claims(token)
    assert claims["eat_nonce"] == "abc123"
    assert claims["iss"] == "https://nras.attestation.nvidia.com"


def test_decode_gpu_claims_returns_empty_on_malformed_token():
    assert _decode_gpu_claims("not json") == {}
    assert _decode_gpu_claims(json.dumps([["JWT", "x"]])) == {}


def test_dataset_attestation_nonce_is_deterministic_and_content_sensitive():
    first = dataset_attestation_nonce("prompts-hash", "trajectories-hash")
    second = dataset_attestation_nonce("prompts-hash", "trajectories-hash")
    assert first == second

    different = dataset_attestation_nonce("prompts-hash", "different-trajectories-hash")
    assert different != first
