"""NRAS token signature verification against a local key pair (no network)."""

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from sparkproof.gpu.token_verify import extract_detached_gpu_jwt, verify_nras_token
from sparkproof.verify_online import verify_openrouter_generations


class _StaticJwkClient:
    """Stand-in for jwt.PyJWKClient bound to one local public key."""

    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, raw_jwt):
        class _Key:
            key = self._public_key

        return _Key()


@pytest.fixture(scope="module")
def keypair():
    private = ec.generate_private_key(ec.SECP384R1())
    return private, private.public_key()


def _make_token(private_key, claims: dict) -> str:
    raw_jwt = jwt.encode(claims, private_key, algorithm="ES384", headers={"kid": "nras-test-key"})
    composite = [
        ["JWT", "outer-wrapper-ignored"],
        {"REMOTE_GPU_CLAIMS": [["JWT", raw_jwt]]},
    ]
    return json.dumps(composite)


def _claims(nonce: str = "n" * 64) -> dict:
    return {
        "iss": "https://nras.attestation.nvidia.com",
        "iat": int(time.time()) - 3600,
        "exp": int(time.time()) - 1800,  # expired — must still verify by signature
        "eat_nonce": nonce,
        "measres": "comparison-successful",
    }


def test_extract_detached_gpu_jwt(keypair):
    private, _ = keypair
    token = _make_token(private, _claims())
    assert extract_detached_gpu_jwt(token) is not None
    assert extract_detached_gpu_jwt("not json") is None
    assert extract_detached_gpu_jwt(json.dumps([{"OTHER": []}])) is None


def test_valid_signature_and_nonce_verifies(keypair):
    private, public = keypair
    token = _make_token(private, _claims(nonce="a" * 64))
    result = verify_nras_token(token, expected_nonce="a" * 64, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is True
    assert result["issues"] == []
    assert result["claims"]["eat_nonce"] == "a" * 64


def test_expired_token_still_verifies_by_signature(keypair):
    private, public = keypair
    token = _make_token(private, _claims())
    result = verify_nras_token(token, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is True


def test_forged_signature_is_rejected(keypair):
    _, public = keypair
    attacker = ec.generate_private_key(ec.SECP384R1())
    token = _make_token(attacker, _claims())
    result = verify_nras_token(token, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is False
    assert any("INVALID" in issue for issue in result["issues"])


def test_wrong_nonce_is_rejected(keypair):
    private, public = keypair
    token = _make_token(private, _claims(nonce="a" * 64))
    result = verify_nras_token(token, expected_nonce="b" * 64, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is False
    assert any("eat_nonce" in issue for issue in result["issues"])


def test_wrong_issuer_is_rejected(keypair):
    private, public = keypair
    claims = _claims()
    claims["iss"] = "https://evil.example.com"
    token = _make_token(private, claims)
    result = verify_nras_token(token, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is False
    assert any("issuer" in issue for issue in result["issues"])


def test_failed_measurement_is_rejected(keypair):
    private, public = keypair
    claims = _claims()
    claims["measres"] = "comparison-failed"
    token = _make_token(private, claims)
    result = verify_nras_token(token, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is False
    assert any("measurement" in issue for issue in result["issues"])


def _openrouter_record(generation_id="gen-abc", model="anthropic/claude-fable-5"):
    return {
        "gateway": "openrouter",
        "gateway_model": model,
        "openrouter_model": model,
        "metadata": {"openrouter_generation_id": generation_id},
    }


def test_openrouter_ledger_model_mismatch_is_flagged():
    def fetch(generation_id, api_key):
        return {"data": {"id": generation_id, "model": "openai/gpt-4o-mini"}}

    issues = verify_openrouter_generations([_openrouter_record()], api_key="k", fetch=fetch)
    assert any("ledger says model" in issue for issue in issues)


def test_openrouter_ledger_match_passes():
    def fetch(generation_id, api_key):
        return {"data": {"id": generation_id, "model": "anthropic/claude-fable-5"}}

    issues = verify_openrouter_generations([_openrouter_record()], api_key="k", fetch=fetch)
    assert issues == []


def test_missing_generation_id_is_flagged():
    record = _openrouter_record()
    record["metadata"] = {}
    issues = verify_openrouter_generations([record], api_key="k", fetch=lambda g, k: {})
    assert any("no openrouter_generation_id" in issue for issue in issues)


def _make_token_with_device(private_key, platform_claims: dict, device_claims: dict) -> str:
    encode = lambda payload: jwt.encode(  # noqa: E731
        payload, private_key, algorithm="ES384", headers={"kid": "nras-test-key"}
    )
    composite = [
        ["JWT", "outer-wrapper-ignored"],
        {"REMOTE_GPU_CLAIMS": [["JWT", encode(platform_claims)], {"GPU-0": encode(device_claims)}]},
    ]
    return json.dumps(composite)


def test_device_tokens_verified_and_exposed(keypair):
    private, public = keypair
    token = _make_token_with_device(
        private,
        {"iss": "https://nras.attestation.nvidia.com", "iat": int(time.time()), "eat_nonce": "aa" * 32},
        {"iss": "https://nras.attestation.nvidia.com", "hwmodel": "GB20X"},
    )
    result = verify_nras_token(token, expected_nonce="aa" * 32, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is True
    assert result["claims"]["devices"]["GPU-0"]["hwmodel"] == "GB20X"


def test_forged_device_token_fails(keypair):
    private, public = keypair
    other = ec.generate_private_key(ec.SECP384R1())
    forged_device = jwt.encode({"hwmodel": "GB20X"}, other, algorithm="ES384", headers={"kid": "nras-test-key"})
    composite = json.dumps(
        [
            ["JWT", "outer-wrapper-ignored"],
            {
                "REMOTE_GPU_CLAIMS": [
                    ["JWT", jwt.encode({"iss": "https://nras.attestation.nvidia.com", "iat": int(time.time())}, private, algorithm="ES384", headers={"kid": "nras-test-key"})],
                    {"GPU-0": forged_device},
                ]
            },
        ]
    )
    result = verify_nras_token(composite, jwk_client=_StaticJwkClient(public))
    assert result["verified"] is False
    assert any("GPU-0" in issue for issue in result["issues"])
