"""Online trust-anchor checks that upgrade a stored bundle to cryptographic proof.

Offline `verify_bundle` proves internal consistency (hashes, merkle, policy).
These checks anchor the bundle to external roots of trust:

1. `verify_attestation_signature` — verifies the NVIDIA NRAS signature on the
   stored GPU attestation token against NVIDIA's published JWKS. A forged or
   hand-written gpu_attestation.json fails here, because only NVIDIA can sign
   the detached per-GPU claims JWT (which also carries the dataset-bound nonce).
2. `verify_openrouter_generations` — re-queries OpenRouter's generation
   endpoint for recorded generation ids and cross-checks the routed model.
   OpenRouter scopes this endpoint to the key that created the generation, so
   this is a miner-side self-audit / key-escrow check, not something an
   arbitrary validator can run for someone else's bundle.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from sparkproof.gpu.token_verify import verify_nras_token
from sparkproof.hashing import canonical_json_bytes, dataset_attestation_nonce, sha256_hex

OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation"


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_attestation_signature(
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    jwk_client: Any | None = None,
) -> list[str]:
    """Cryptographically verify the stored NRAS token signature + dataset nonce."""
    path = bundle_dir / "gpu_attestation.json"
    if not path.exists():
        return ["missing gpu_attestation.json — cannot verify NRAS signature"]
    att = json.loads(path.read_text())
    token = att.get("token") or ""
    if not token:
        return ["gpu_attestation.json has no token — cannot verify NRAS signature"]

    expected_nonce: str | None = None
    raw_path = bundle_dir / "trajectories_raw.jsonl"
    if raw_path.exists():
        raw = _load_trajectories(raw_path)
        expected_nonce = dataset_attestation_nonce(
            manifest.get("prompts_sha256") or "", sha256_hex(canonical_json_bytes(raw))
        )

    result = verify_nras_token(token, expected_nonce=expected_nonce, jwk_client=jwk_client)
    return [f"nras: {issue}" for issue in result["issues"]]


def _default_fetch_generation(generation_id: str, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{OPENROUTER_GENERATION_URL}?id={generation_id}",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verify_openrouter_generations(
    trajectories: list[dict[str, Any]],
    *,
    api_key: str,
    fetch: Callable[[str, str], dict[str, Any]] | None = None,
    max_checks: int | None = None,
) -> list[str]:
    """Cross-check recorded OpenRouter generation ids against OpenRouter's ledger.

    Confirms each recorded generation id exists and was routed to the pinned
    teacher model. Requires the API key that created the generations (OpenRouter
    scopes the endpoint per key) — run as miner self-audit before submission, or
    by a validator holding an escrowed key.
    """
    fetch = fetch or _default_fetch_generation
    issues: list[str] = []
    checked = 0
    for i, record in enumerate(trajectories):
        if record.get("gateway") != "openrouter":
            continue
        meta = record.get("metadata") or {}
        generation_id = meta.get("openrouter_generation_id") or meta.get("gateway_generation_id")
        if not generation_id:
            issues.append(f"trajectory[{i}]: no openrouter_generation_id recorded")
            continue
        if max_checks is not None and checked >= max_checks:
            break
        checked += 1
        try:
            payload = fetch(str(generation_id), api_key)
        except urllib.error.HTTPError as exc:
            issues.append(f"trajectory[{i}]: generation {generation_id!r} not found on OpenRouter (HTTP {exc.code})")
            continue
        except Exception as exc:
            issues.append(f"trajectory[{i}]: generation lookup failed: {exc}")
            continue

        data = payload.get("data") or payload
        ledger_model = data.get("model") or ""
        recorded_model = record.get("gateway_model") or record.get("openrouter_model") or ""
        if ledger_model and ledger_model != recorded_model:
            issues.append(
                f"trajectory[{i}]: OpenRouter ledger says model {ledger_model!r} "
                f"but bundle records {recorded_model!r}"
            )
    return issues


def verify_bundle_online(
    bundle_dir: Path,
    *,
    openrouter_api_key: str | None = None,
    jwk_client: Any | None = None,
    max_generation_checks: int | None = None,
) -> dict[str, Any]:
    """Run all available online trust-anchor checks for a bundle."""
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    issues = verify_attestation_signature(bundle_dir, manifest, jwk_client=jwk_client)

    generation_checked = False
    if openrouter_api_key:
        trajectories = _load_trajectories(bundle_dir / "trajectories.jsonl")
        issues.extend(
            verify_openrouter_generations(
                trajectories,
                api_key=openrouter_api_key,
                max_checks=max_generation_checks,
            )
        )
        generation_checked = True

    return {
        "verified": not issues,
        "issues": issues,
        "nras_signature_checked": True,
        "openrouter_ledger_checked": generation_checked,
    }
