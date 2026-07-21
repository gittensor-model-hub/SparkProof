#!/usr/bin/env python3
"""Probe yunwu gpt-5.6-sol reasoning shapes (SparkProof request format)."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

# Load SparkProof/.env if YUNWU_API_KEY not already exported
if not os.environ.get("YUNWU_API_KEY"):
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkproof.generate.gateway_client import _post_chat  # noqa: E402
from sparkproof.gateways import GATEWAY_YUNWU  # noqa: E402
from sparkproof.teacher_request import build_chat_body  # noqa: E402


def classify(msg: dict) -> dict:
    rd = msg.get("reasoning_details") or []
    types = [x.get("type") for x in rd if isinstance(x, dict)]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    return {
        "message_keys": sorted(msg.keys()),
        "plaintext_reasoning_field": bool(isinstance(reasoning, str) and reasoning.strip()),
        "reasoning_text": "reasoning.text" in types,
        "reasoning_summary": "reasoning.summary" in types,
        "reasoning_encrypted": any("encrypted" in (t or "") for t in types),
        "reasoning_detail_types": types,
    }


def probe(name: str, prompt: str, *, max_tokens: int) -> None:
    print(f"\n{'=' * 60}\n=== {name} ===\n{'=' * 60}")
    api_key = os.environ.get("YUNWU_API_KEY", "").strip()
    if not api_key:
        print("ERROR: YUNWU_API_KEY not set")
        return

    body = build_chat_body(
        gateway=GATEWAY_YUNWU,
        provider="openai",
        prompt=prompt,
        system=None,
        max_tokens=max_tokens,
        temperature=0.7,
        reasoning_effort="xhigh",
    )
    print("request model:", body["model"])
    print("request keys:", sorted(body.keys()))

    try:
        payload, _headers = _post_chat(gateway=GATEWAY_YUNWU, api_key=api_key, body=body, timeout=180)
    except Exception as exc:
        print("REQUEST_ERROR:", type(exc).__name__, exc)
        return

    if "error" in payload:
        print("API_ERROR:", json.dumps(payload["error"], indent=2))
        return

    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = payload.get("usage") or {}
    ctd = usage.get("completion_tokens_details") or {}

    print("response model:", payload.get("model"))
    print("finish_reason:", choice.get("finish_reason"))
    print("content_len:", len(msg.get("content") or ""))
    print("content_preview:", repr((msg.get("content") or "")[:180]))
    print("usage:", usage)
    print("reasoning_tokens:", ctd.get("reasoning_tokens"))

    for field in ("reasoning", "reasoning_content"):
        if field in msg:
            v = msg[field]
            print(f"{field}: type={type(v).__name__}")
            if isinstance(v, str):
                print(f"  len={len(v)} preview={v[:250]!r}")

    rd = msg.get("reasoning_details")
    if rd:
        print(f"reasoning_details: {len(rd)} item(s)")
        for i, item in enumerate(rd):
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            print(f"  [{i}] type={t!r}")
            if t == "reasoning.text":
                text = item.get("text") or ""
                print(f"    text_len={len(text)} preview={text[:200]!r}")
            elif t == "reasoning.summary":
                summary = item.get("summary") or ""
                print(f"    summary_len={len(summary)} preview={summary[:200]!r}")
            elif "encrypted" in (t or ""):
                data = item.get("data") or ""
                print(f"    encrypted_data_len={len(data)} preview={data[:80]!r}...")
                try:
                    dec = base64.b64decode(data + "=" * (-len(data) % 4))
                    print(f"    decoded_preview={dec[:120]!r}")
                except Exception as exc:
                    print(f"    decode_err={exc}")
    else:
        print("reasoning_details: none")

    print("CLASSIFICATION:", json.dumps(classify(msg), indent=2))


def main() -> None:
    probe("trivial math (17+25)", "What is 17+25? Reply with only the number.", max_tokens=512)
    probe(
        "hard Triton prompt",
        (
            "Write a Triton kernel that fuses layer norm and GELU for a 2D tensor. "
            "Include only Python code using triton.language as tl."
        ),
        max_tokens=2048,
    )


if __name__ == "__main__":
    main()
