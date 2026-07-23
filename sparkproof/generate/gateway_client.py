"""Teacher clients for pinned gateways (OpenRouter, yunwu.ai)."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from sparkproof.gateways import (
    GATEWAY_OPENROUTER,
    GATEWAY_YUNWU,
    gateway_max_retries,
    gateway_retry_backoff_seconds,
    gateway_timeout_seconds,
    get_gateway,
)
from sparkproof.hashing import canonical_json_bytes, sha256_hex
from sparkproof.policy import REQUIRED_REASONING_EFFORT, normalize_upstream_model, validate_gateway_trajectory
from sparkproof.teacher_request import build_chat_body, generation_config, request_sha256


class GatewayTransientError(RuntimeError):
    """Teacher gateway failed after retries (timeout, rate limit, upstream 5xx)."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {429, 502, 503, 504}
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        # Connection reset / transient network blips on long yunwu runs.
        if isinstance(reason, OSError):
            return True
    return False


def _post_chat(
    *,
    gateway: str,
    api_key: str,
    body: dict[str, Any],
    timeout: int | None = None,
    max_retries: int | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    policy = get_gateway(gateway)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if gateway == GATEWAY_OPENROUTER:
        headers["HTTP-Referer"] = "https://github.com/gittensor-model-hub/SparkProof"
        headers["X-OpenRouter-Title"] = "SparkProof"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(policy.chat_url, data=data, headers=headers, method="POST")
    read_timeout = timeout if timeout is not None else gateway_timeout_seconds(gateway)
    retries = gateway_max_retries() if max_retries is None else max(0, max_retries)
    backoff = gateway_retry_backoff_seconds()

    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=read_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                response_headers = {k.lower(): v for k, v in resp.headers.items()}
                return payload, response_headers
        except urllib.error.HTTPError as e:
            last_exc = e
            detail = e.read().decode("utf-8", errors="replace")
            if not _is_retryable(e) or attempt >= retries:
                hint = ""
                if gateway == GATEWAY_YUNWU and e.code == 503:
                    hint = (
                        " — yunwu has no channel for this model slug. "
                        "Run: sparkproof-yunwu-probe --auto --write-env .env"
                    )
                raise RuntimeError(f"HTTP {e.code} from {gateway}: {detail}{hint}") from e
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt >= retries:
                if _is_retryable(e):
                    raise GatewayTransientError(
                        f"{gateway} request failed after {retries + 1} attempt(s) "
                        f"(timeout={read_timeout}s): {e}"
                    ) from e
                raise

        sleep_s = backoff * (2**attempt)
        if sleep_s > 0:
            time.sleep(sleep_s)

    raise GatewayTransientError(f"{gateway} request failed: {last_exc}")


def generate_via_gateway(
    *,
    gateway: str,
    api_key: str,
    provider: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float = 0.7,
    reasoning_effort: str = REQUIRED_REASONING_EFFORT,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    policy = get_gateway(gateway)
    body = build_chat_body(
        gateway=gateway,
        provider=provider,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    requested_model = body["model"]
    request_sha = request_sha256(body)
    payload, response_headers = _post_chat(
        gateway=gateway,
        api_key=api_key,
        body=body,
        timeout=timeout,
        max_retries=max_retries,
    )
    response_sha = sha256_hex(canonical_json_bytes(payload))

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    upstream_model = payload.get("model", requested_model)
    logical_model = normalize_upstream_model(
        provider,
        upstream_model.split("/")[-1] if "/" in upstream_model else upstream_model,
        gateway=gateway,
    )

    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if not reasoning and message.get("reasoning_details"):
        # Prefer plaintext reasoning.text / summary; never persist encrypted-only
        # blobs as training CoT (GPT 5.6 Sol often returns encrypted details).
        from sparkproof.triton_dataset.training_cot import extract_plaintext_reasoning_details

        reasoning = extract_plaintext_reasoning_details(message["reasoning_details"])
        if reasoning is None:
            # Keep a compact marker for debugging; SFT export skips non-usable CoT.
            reasoning = json.dumps(message["reasoning_details"], ensure_ascii=False)

    metadata: dict[str, Any] = {
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "usage": payload.get("usage", {}),
        "gateway_generation_id": payload.get("id"),
        "gateway_requested_model": requested_model,
        "gateway_response_model": upstream_model,
        "gateway_reasoning_effort": reasoning_effort,
        "gateway_max_tokens": max_tokens,
        "gateway_temperature": temperature,
    }
    if gateway == GATEWAY_OPENROUTER:
        metadata.update(
            {
                "openrouter_generation_id": payload.get("id"),
                "openrouter_requested_model": requested_model,
                "openrouter_response_model": upstream_model,
                "openrouter_reasoning_effort": reasoning_effort,
                "openrouter_max_tokens": max_tokens,
                "openrouter_temperature": temperature,
                "openrouter_response_headers": {
                    k: response_headers[k]
                    for k in ("x-openrouter-request-id", "x-openrouter-generation-id", "server")
                    if k in response_headers
                },
            }
        )

    record: dict[str, Any] = {
        "prompt": prompt,
        "response": message.get("content") or "",
        "provider": provider,
        "model": logical_model,
        "system": system,
        "reasoning": reasoning,
        "gateway": gateway,
        "api_base": policy.api_base,
        "request_url": policy.chat_url,
        "gateway_model": upstream_model,
        "openrouter_model": upstream_model,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "metadata": metadata,
    }
    validate_gateway_trajectory(record)
    return record


def generation_config_for_run(*, max_tokens: int, temperature: float) -> dict[str, Any]:
    return generation_config(max_tokens=max_tokens, temperature=temperature)
