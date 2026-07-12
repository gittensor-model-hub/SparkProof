"""Cryptographic verification of NVIDIA NRAS attestation tokens.

A stored ``gpu_attestation.json`` is only trustworthy if the embedded NRAS JWT
actually carries a valid NVIDIA signature — otherwise a miner could fabricate
the whole file. This module verifies the detached per-GPU claims JWT against
NVIDIA's published JWKS (https://nras.attestation.nvidia.com/.well-known/jwks.json),
turning "trust the JSON blob" into "verify NVIDIA's signature".

Expiry note: NRAS tokens are short-lived, but a validator re-checks bundles
hours or days after attestation. Signature validity is what proves the token
came from NVIDIA at issuance; ``exp`` being in the past is expected and is
reported, not treated as tampering.
"""

from __future__ import annotations

import json
from typing import Any

DEFAULT_NRAS_JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
NRAS_ISSUER_SUBSTRING = "nras.attestation.nvidia.com"
_REMOTE_GPU_CLAIMS_KEY = "REMOTE_GPU_CLAIMS"
_ACCEPTED_ALGORITHMS = ["ES384", "ES256", "RS256", "RS384", "PS384"]


def extract_detached_gpu_jwt(token: str) -> str | None:
    """Pull the NVIDIA-signed per-GPU claims JWT out of the SDK's composite token."""
    try:
        parsed = json.loads(token)
        detached = next(
            entry for entry in parsed if isinstance(entry, dict) and _REMOTE_GPU_CLAIMS_KEY in entry
        )
        return detached[_REMOTE_GPU_CLAIMS_KEY][0][1]
    except Exception:
        return None


def extract_device_jwts(token: str) -> dict[str, str]:
    """Pull the NVIDIA-signed per-device JWTs (e.g. GPU-0) out of the composite token.

    These carry the hardware identity claims (``hwmodel``, driver/vbios versions)
    that hardware corroboration should read from *signed* material rather than
    the unsigned attestation JSON.
    """
    devices: dict[str, str] = {}
    try:
        parsed = json.loads(token)
        detached = next(
            entry for entry in parsed if isinstance(entry, dict) and _REMOTE_GPU_CLAIMS_KEY in entry
        )
        for entry in detached[_REMOTE_GPU_CLAIMS_KEY]:
            if isinstance(entry, dict):
                devices.update({str(k): str(v) for k, v in entry.items()})
    except Exception:
        pass
    return devices


def verify_nras_token(
    token: str,
    *,
    expected_nonce: str | None = None,
    jwks_url: str = DEFAULT_NRAS_JWKS_URL,
    jwk_client: Any | None = None,
) -> dict[str, Any]:
    """Verify the NRAS signature and claims of a stored attestation token.

    Returns {"verified": bool, "issues": [...], "claims": {...}}. Pass a
    ``jwk_client`` (anything with ``get_signing_key_from_jwt``) to avoid the
    network JWKS fetch in tests or air-gapped re-verification with a pinned key set.
    """
    import jwt

    issues: list[str] = []
    raw_jwt = extract_detached_gpu_jwt(token)
    if raw_jwt is None:
        return {
            "verified": False,
            "issues": ["token does not contain a detached REMOTE_GPU_CLAIMS JWT"],
            "claims": {},
        }

    client = jwk_client or jwt.PyJWKClient(jwks_url)
    try:
        signing_key = client.get_signing_key_from_jwt(raw_jwt)
    except Exception as exc:
        return {
            "verified": False,
            "issues": [f"could not resolve NRAS signing key: {exc}"],
            "claims": {},
        }

    key = getattr(signing_key, "key", signing_key)
    try:
        claims = jwt.decode(
            raw_jwt,
            key,
            algorithms=_ACCEPTED_ALGORITHMS,
            options={
                "verify_aud": False,
                # Tokens are short-lived; validators verify long after issuance.
                # The signature (not freshness) is the trust anchor here.
                "verify_exp": False,
            },
        )
    except jwt.InvalidSignatureError:
        return {
            "verified": False,
            "issues": ["NRAS JWT signature is INVALID — attestation token was forged or corrupted"],
            "claims": {},
        }
    except Exception as exc:
        return {"verified": False, "issues": [f"NRAS JWT could not be verified: {exc}"], "claims": {}}

    issuer = str(claims.get("iss") or "")
    if NRAS_ISSUER_SUBSTRING not in issuer:
        issues.append(f"unexpected token issuer {issuer!r} (expected NVIDIA NRAS)")
    if not claims.get("iat"):
        issues.append("token missing iat (issuance time)")

    measurement = claims.get("measres") or claims.get("x-nvidia-overall-att-result")
    if measurement is not None and str(measurement).lower() not in {"success", "comparison-successful", "true"}:
        issues.append(f"attestation measurement result is {measurement!r}, not success")

    if expected_nonce is not None and claims.get("eat_nonce") != expected_nonce:
        issues.append(
            "signed eat_nonce does not match the expected dataset-bound nonce — "
            "token was not produced for this bundle's content"
        )

    # Per-device tokens carry the hardware identity (hwmodel etc.); verify each
    # signature so corroboration can trust those claims, not the unsigned JSON.
    devices: dict[str, dict[str, Any]] = {}
    for device, device_jwt in extract_device_jwts(token).items():
        try:
            device_key = client.get_signing_key_from_jwt(device_jwt)
            devices[device] = jwt.decode(
                device_jwt,
                getattr(device_key, "key", device_key),
                algorithms=_ACCEPTED_ALGORITHMS,
                options={"verify_aud": False, "verify_exp": False},
            )
        except Exception as exc:
            issues.append(f"device token {device}: signature could not be verified: {exc}")
    if devices:
        claims = dict(claims)
        claims["devices"] = devices

    return {"verified": not issues, "issues": issues, "claims": claims}
