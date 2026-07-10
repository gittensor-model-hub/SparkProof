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


@dataclass(frozen=True)
class GpuAttestationResult:
    passed: bool
    environment: str
    token: str
    claims: dict[str, Any]
    gpu_profile: dict[str, Any]

    def token_sha256(self) -> str:
        return hashlib.sha256(self.token.encode()).hexdigest() if self.token else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "environment": self.environment,
            "token": self.token,
            "claims": self.claims,
            "gpu_profile": self.gpu_profile,
            "token_sha256": self.token_sha256(),
        }


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
) -> GpuAttestationResult:
    """Collect GPU evidence and verify against NVIDIA NRAS (requires `uv sync --extra gpu`)."""
    from nv_attestation_sdk import attestation as nv_attestation

    default_policy = Path(__file__).resolve().parents[2] / "policies" / "gpu_remote_v3.json"
    policy = policy_path or default_policy
    env = getattr(nv_attestation.Environment, environment.upper())

    client = nv_attestation.Attestation()
    client.set_name("sparkproof-blackwell")
    set_service_key = getattr(client, "set_service_key", None)
    if service_key and callable(set_service_key):
        set_service_key(service_key)

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
    claims = _decode_overall_claims(token) if token else {}
    _reset_client(nv_attestation)

    return GpuAttestationResult(
        passed=passed and validated,
        environment=environment.upper(),
        token=token,
        claims=claims,
        gpu_profile=gpu_profile,
    )


def _decode_overall_claims(token: str) -> dict[str, Any]:
    import jwt

    try:
        overall_jwt = json.loads(token)[0][1]
        return jwt.decode(overall_jwt, options={"verify_signature": False})
    except Exception:
        return {}
