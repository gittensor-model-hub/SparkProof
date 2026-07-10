"""Trajectory generation via pinned teacher gateways."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from sparkproof.generate.gateway_client import generate_via_gateway, generation_config_for_run
from sparkproof.gateways import default_gateway, resolve_api_key
from sparkproof.hashing import sha256_file
from sparkproof.policy import SUPPORTED_PROVIDERS


def iter_prompts(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def generate_trajectories(
    *,
    prompts_path: Path,
    providers: list[str],
    gateway: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], str]:
    gateway_name = gateway or default_gateway()
    key = api_key or resolve_api_key(gateway_name)

    unknown = set(providers) - SUPPORTED_PROVIDERS
    if unknown:
        raise ValueError(f"unsupported providers {sorted(unknown)}")

    trajectories: list[dict[str, Any]] = []
    for record in iter_prompts(prompts_path, limit):
        prompt = record["prompt"]
        system = record.get("system")
        for provider in providers:
            trajectories.append(
                generate_via_gateway(
                    gateway=gateway_name,
                    api_key=key,
                    provider=provider,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )

    return (
        trajectories,
        sha256_file(str(prompts_path)),
        generation_config_for_run(max_tokens=max_tokens, temperature=temperature),
        gateway_name,
    )
