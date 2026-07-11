"""Identity-free run entropy for reproducible, diversified sampling.

A `run_seed` is random entropy committed to the run's provenance before any
teacher requests are made — it is not a miner/contributor identity. It only
makes an otherwise-random sampling/evolution choice reproducible: the same
catalog + run_seed + policy version always replays the same prompt set.
"""

from __future__ import annotations

import hashlib
import secrets

SAMPLING_POLICY_VERSION = "stratified-v1"


def generate_run_seed() -> str:
    """256-bit random run entropy, hex-encoded."""
    return secrets.token_hex(32)


def _derive(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def sampling_seed(catalog_sha256: str, run_seed: str, policy_version: str = SAMPLING_POLICY_VERSION) -> int:
    """sampling_seed = H(catalog_sha256 || run_seed || sampling_policy_version)."""
    return int(_derive("sampling", catalog_sha256, run_seed, policy_version), 16)


def evolution_seed(
    run_seed: str,
    parent_task_id: str,
    depth: int,
    policy_version: str = SAMPLING_POLICY_VERSION,
) -> int:
    """evolution_seed = H(run_seed || parent_task_id || depth || policy_version)."""
    return int(_derive("evolution", run_seed, parent_task_id, str(depth), policy_version), 16)
