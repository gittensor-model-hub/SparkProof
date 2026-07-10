"""Generate a SparkProof dataset on Blackwell CC (teacher gateway → prove → sparkproof-2)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sparkproof.bundle import write_bundle
from sparkproof.gateways import ALLOWED_GATEWAYS, default_gateway, resolve_api_key
from sparkproof.generate.runner import generate_trajectories
from sparkproof.manifest import build_manifest
from sparkproof.pipeline.blackwell import prove_blackwell_bundle


def _load_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="jsonl of {prompt, system?} records")
    parser.add_argument("--out", type=Path, required=True, help="output bundle directory")
    parser.add_argument(
        "--gateway",
        choices=sorted(ALLOWED_GATEWAYS),
        default=None,
        help="teacher API gateway (default: SPARKPROOF_GATEWAY or openrouter)",
    )
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        choices=["anthropic", "openai"],
        default=None,
        help="teacher providers (default: both Fable 5 xhigh + GPT 5.6 Sol xhigh)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--skip-blackwell",
        action="store_true",
        help="skip Blackwell Triton validation (dev only — not valid for production)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Blackwell CUDA device for validation")
    parser.add_argument("--benchmark", action="store_true", help="require benchmark score floor on Blackwell")
    parser.add_argument(
        "--no-gpu-attest",
        action="store_true",
        help="skip GPU CC attestation during Blackwell prove (dev only)",
    )
    args = parser.parse_args(argv)

    _load_env()
    gateway = args.gateway or default_gateway()
    api_key = resolve_api_key(gateway)
    providers = args.providers or ["anthropic", "openai"]

    trajectories, prompts_sha256, gen_config, gateway_name = generate_trajectories(
        prompts_path=args.prompts,
        providers=providers,
        gateway=gateway,
        api_key=api_key,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )
    manifest = build_manifest(
        trajectories,
        prompts_sha256=prompts_sha256,
        openrouter_generation_config=gen_config,
        gateway=gateway_name,
    ).to_dict()
    write_bundle(
        out_dir=args.out,
        trajectories=trajectories,
        manifest=manifest,
        prompts_path=args.prompts,
    )
    print(f"wrote raw teacher bundle to {args.out} via {gateway_name}", file=sys.stderr)

    if not args.skip_blackwell:
        report = prove_blackwell_bundle(
            args.out,
            gpu_index=args.gpu,
            benchmark=args.benchmark,
            attest_gpu=not args.no_gpu_attest,
        )
        print(
            f"proved on Blackwell: {report['verified_count']}/{report['raw_count']} verified",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
