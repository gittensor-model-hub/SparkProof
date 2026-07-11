"""Probe yunwu.ai token: list /v1/models and smoke-test configured teacher slugs."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sparkproof.env_loader import load_dotenv
from sparkproof.gateways import (
    GATEWAY_YUNWU,
    YUNWU_DEFAULT_ANTHROPIC,
    YUNWU_DEFAULT_OPENAI,
    YUNWU_PINNED_SLUGS,
    get_gateway,
    resolve_api_key,
    yunwu_models_url,
)
from sparkproof.policy import REQUIRED_REASONING_EFFORT

_ANTHROPIC_PREFS = (
    "claude-fable-5",
)
_OPENAI_PREFS = (
    "gpt-5.6-sol",
    "gpt-5.6",
)


def _pick_model(ids: list[str], prefs: tuple[str, ...]) -> str | None:
    lower_map = {i.lower(): i for i in ids}
    for pref in prefs:
        if pref in lower_map:
            return lower_map[pref]
    for pref in prefs:
        for i in ids:
            if pref in i.lower():
                return i
    return None


def _get_json(url: str, api_key: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_chat(url: str, api_key: str, body: dict) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8")[:500]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:500]


def _test_model(chat_url: str, api_key: str, model: str) -> tuple[bool, int, str]:
    for effort in (REQUIRED_REASONING_EFFORT, "high", "medium"):
        body = {
            "model": model,
            "max_tokens": 32,
            "temperature": 0.7,
            "reasoning_effort": effort,
            "messages": [{"role": "user", "content": "Reply with exactly: YUNWU_OK"}],
        }
        code, text = _post_chat(chat_url, api_key, body)
        if code == 200:
            return True, code, text
    return False, code, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print models visible to this token")
    parser.add_argument("--test", action="store_true", help="smoke-test configured YUNWU_MODEL_* slugs")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="pick working anthropic/openai slugs from token model list and print export lines",
    )
    parser.add_argument("--write-env", type=Path, default=None, help="merge auto-picked models into .env")
    args = parser.parse_args(argv)

    load_dotenv()
    api_key = resolve_api_key(GATEWAY_YUNWU)
    policy = get_gateway(GATEWAY_YUNWU)

    payload = _get_json(yunwu_models_url(), api_key)
    models = payload.get("data", payload if isinstance(payload, list) else [])
    ids = sorted(m.get("id", m) if isinstance(m, dict) else str(m) for m in models)

    if args.list or (not args.test and not args.auto):
        print(f"yunwu base: {policy.api_base}")
        print(f"models visible to token: {len(ids)}")
        for needle in ("claude", "gpt-5", "fable", "sonnet", "gpt-4o"):
            hits = [i for i in ids if needle.lower() in i.lower()]
            if hits:
                print(f"\n-- {needle} --")
                for h in hits[:20]:
                    print(h)

    picked_anthropic = _pick_model(ids, _ANTHROPIC_PREFS) or YUNWU_DEFAULT_ANTHROPIC
    picked_openai = _pick_model(ids, _OPENAI_PREFS) or YUNWU_DEFAULT_OPENAI

    if args.auto or args.write_env:
        missing = [slug for slug in sorted(YUNWU_PINNED_SLUGS) if slug not in ids]
        if missing:
            print(
                f"error: token cannot see pinned production slugs {missing!r} — "
                "yunwu must expose claude-fable-5 and gpt-5.6-sol for SparkProof production",
                file=sys.stderr,
            )
            return 2
        ok_a, _, _ = _test_model(policy.chat_url, api_key, picked_anthropic)
        ok_o, _, _ = _test_model(policy.chat_url, api_key, picked_openai)
        if not ok_a or not ok_o:
            print(
                "error: smoke test failed for pinned production teachers — "
                "confirm claude-fable-5 and gpt-5.6-sol are enabled on your yunwu token",
                file=sys.stderr,
            )
            return 1
        print("\n# add to .env:")
        print(f"YUNWU_MODEL_ANTHROPIC={picked_anthropic}")
        print(f"YUNWU_MODEL_OPENAI={picked_openai}")
        print("SPARKPROOF_GATEWAY=yunwu")
        if args.write_env:
            path = args.write_env
            lines = path.read_text().splitlines() if path.exists() else []
            updates = {
                "YUNWU_MODEL_ANTHROPIC": picked_anthropic,
                "YUNWU_MODEL_OPENAI": picked_openai,
                "SPARKPROOF_GATEWAY": "yunwu",
            }
            out: list[str] = []
            seen: set[str] = set()
            for line in lines:
                key = line.split("=", 1)[0].strip() if "=" in line else ""
                if key in updates:
                    out.append(f"{key}={updates[key]}")
                    seen.add(key)
                else:
                    out.append(line)
            for key, val in updates.items():
                if key not in seen:
                    out.append(f"{key}={val}")
            path.write_text("\n".join(out) + "\n")
            print(f"wrote {path}", file=sys.stderr)

    if args.test:
        print("\n=== chat smoke tests ===")
        for provider, model in policy.models_by_provider.items():
            ok, code, text = _test_model(policy.chat_url, api_key, model)
            status = "OK" if ok else "FAIL"
            print(f"{provider} {model}: {status} HTTP {code}")
            if not ok:
                print(f"  {text[:300]}")
                print("  hint: run sparkproof-yunwu-probe --auto --write-env .env", file=sys.stderr)
            else:
                print(f"  {text[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
