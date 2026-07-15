"""GPU CC attestation — binds validation to a Blackwell confidential-computing GPU."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_NRAS_GPU_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
DEFAULT_RIM_URL = "https://rim.attestation.nvidia.com/v1/rim/"
DEFAULT_OCSP_URL = "https://ocsp.ndis.nvidia.com/"

# Key the SDK nests the real NRAS-signed per-GPU JWT under (see
# nv_attestation_sdk.attestation._create_eat's claims_key_mapping for
# (Devices.GPU, Environment.REMOTE)). The outer token returned by
# client.get_token() is a client-side wrapper JWT (iss=NV-Attestation-SDK)
# carrying no hardware claims — the real, NVIDIA-signed evidence (including
# the echoed nonce) lives one level down, under this key.
_REMOTE_GPU_CLAIMS_KEY = "REMOTE_GPU_CLAIMS"


@dataclass(frozen=True)
class GpuAttestationResult:
    passed: bool
    environment: str
    token: str
    claims: dict[str, Any]
    gpu_profile: dict[str, Any]
    nonce: str = ""
    nonce_verified: bool = False
    tdx: dict[str, Any] | None = None

    def token_sha256(self) -> str:
        return hashlib.sha256(self.token.encode()).hexdigest() if self.token else ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "passed": self.passed,
            "environment": self.environment,
            "token": self.token,
            "claims": self.claims,
            "gpu_profile": self.gpu_profile,
            "token_sha256": self.token_sha256(),
            "nonce": self.nonce,
            "nonce_verified": self.nonce_verified,
        }
        if self.tdx is not None or self.nonce:
            payload["tdx"] = self.tdx
        return payload


def _add_gpu_verifier(
    client: Any,
    *,
    env: Any,
    nras_gpu_url: str,
    ocsp_url: str,
    rim_url: str,
) -> None:
    """Support nv-attestation-sdk 1.x (4-arg) and 2.x (ocsp/rim kwargs)."""
    from nv_attestation_sdk import attestation as nv_attestation

    params = inspect.signature(client.add_verifier).parameters
    if "ocsp_url" in params:
        client.add_verifier(
            nv_attestation.Devices.GPU,
            env,
            nras_gpu_url,
            "",
            ocsp_url=ocsp_url,
            rim_url=rim_url,
        )
        return
    client.add_verifier(nv_attestation.Devices.GPU, env, nras_gpu_url, "")


def _run_attest(client: Any) -> bool:
    if hasattr(client, "get_evidence"):
        evidence_list = client.get_evidence()
        return bool(client.attest(evidence_list))
    return bool(client.attest())


def _reset_client(nv_attestation: Any) -> None:
    reset = getattr(nv_attestation.Attestation, "reset", None)
    if callable(reset):
        reset()
        return
    # nv-attestation-sdk 1.x singleton — clear configured verifiers between runs.
    nv_attestation.Attestation._verifiers = []
    nv_attestation.Attestation._tokens = {}


def attest_blackwell_gpu(
    *,
    gpu_profile: dict[str, Any],
    environment: str = "REMOTE",
    policy_path: Path | None = None,
    service_key: str | None = None,
    nras_gpu_url: str = DEFAULT_NRAS_GPU_URL,
    rim_url: str = DEFAULT_RIM_URL,
    ocsp_url: str = DEFAULT_OCSP_URL,
    nonce: str | None = None,
) -> GpuAttestationResult:
    """Collect GPU evidence and verify against NVIDIA NRAS (requires `uv sync --extra gpu`).

    When `nonce` is supplied, it's sent to NRAS as the attestation challenge and
    echoed back signed in the returned claims as `eat_nonce` — binding this specific
    attested GPU session to whatever the caller derived the nonce from (e.g. a
    dataset's content hash), rather than attesting the hardware in isolation.
    """
    from nv_attestation_sdk import attestation as nv_attestation

    default_policy = Path(__file__).resolve().parents[2] / "policies" / "gpu_remote_v3.json"
    policy = policy_path or default_policy
    env = getattr(nv_attestation.Environment, environment.upper())

    client = nv_attestation.Attestation()
    client.set_name("sparkproof-blackwell")
    set_service_key = getattr(client, "set_service_key", None)
    if service_key and callable(set_service_key):
        set_service_key(service_key)
    if nonce:
        client.set_nonce(nonce)

    _add_gpu_verifier(
        client,
        env=env,
        nras_gpu_url=nras_gpu_url,
        ocsp_url=ocsp_url,
        rim_url=rim_url,
    )

    passed = _run_attest(client)
    token = client.get_token() if passed else ""
    policy_text = policy.read_text()
    validated = bool(client.validate_token(policy_text)) if passed else False
    claims = _decode_gpu_claims(token) if token else {}
    used_nonce = nonce or str(claims.get("eat_nonce") or "")
    nonce_verified = bool(nonce) and claims.get("eat_nonce") == nonce
    _reset_client(nv_attestation)

    tdx: dict[str, Any] | None = None
    if nonce:
        from sparkproof.gpu.tdx import tdx_quote

        tdx = tdx_quote(nonce)

    gpu_passed = passed and validated and (not nonce or nonce_verified)
    tdx_required = bool(nonce)
    tdx_passed = not tdx_required or tdx is not None

    return GpuAttestationResult(
        passed=gpu_passed and tdx_passed,
        environment=environment.upper(),
        token=token,
        claims=claims,
        gpu_profile=gpu_profile,
        nonce=used_nonce,
        nonce_verified=nonce_verified,
        tdx=tdx,
    )


def _decode_gpu_claims(token: str) -> dict[str, Any]:
    """Decode the real, NVIDIA NRAS-signed per-GPU claims (not the SDK's outer wrapper JWT)."""
    import jwt

    try:
        parsed = json.loads(token)
        detached = next(entry for entry in parsed if isinstance(entry, dict) and _REMOTE_GPU_CLAIMS_KEY in entry)
        raw_jwt = detached[_REMOTE_GPU_CLAIMS_KEY][0][1]
        return jwt.decode(raw_jwt, options={"verify_signature": False})
    except Exception:
        return {}
