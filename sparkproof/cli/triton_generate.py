"""Multi-candidate Triton dataset generation → Blackwell prove → sparkproof-2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sparkproof.bundle import write_bundle
from sparkproof.env_loader import load_dotenv
from sparkproof.gateways import ALLOWED_GATEWAYS, default_gateway, resolve_api_key
from sparkproof.generate.runner import iter_prompts
from sparkproof.hashing import sha256_file
from sparkproof.manifest import build_manifest
from sparkproof.pipeline.blackwell import prove_blackwell_bundle
from sparkproof.triton_dataset.decontaminate import TritonDecontaminator, filter_decontaminated
from sparkproof.triton_dataset.failure_miner import record_failure
from sparkproof.triton_dataset.multi_candidate import generate_best_of_n
from sparkproof.triton_dataset.orchestrate import run_dataset_generation_step
from sparkproof.triton_dataset.self_evolve import evolve_verified_trajectory
from sparkproof.triton_dataset.task_policy import assert_trainable_task


def _load_env() -> None:
    load_dotenv()


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gateway", choices=sorted(ALLOWED_GATEWAYS), default=None)
    parser.add_argument("--provider", dest="providers", action="append", choices=["anthropic", "openai"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-gpu-attest", action="store_true")
    parser.add_argument("--skip-blackwell", action="store_true")
    parser.add_argument("--decontaminate", action="store_true", help="drop structurally contaminated samples")
    parser.add_argument("--orchestrate", action="store_true", help="evolve tasks + mine failures per prompt")
    parser.add_argument("--evolve-depth", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    _load_env()
    gateway = args.gateway or default_gateway()
    api_key = resolve_api_key(gateway)
    providers = args.providers or ["anthropic", "openai"]

    trajectories: list[dict] = []
    adjudication: list[dict] = []
    evolved_tasks: list[dict] = []
    run_id = args.run_id or args.out.name
    decontaminator = TritonDecontaminator()
    client = {
        "gateway": gateway,
        "api_key": api_key,
        "providers": providers,
        "max_tokens": args.max_tokens,
        "max_repairs": args.max_repairs,
        "gpu_index": args.gpu,
    }

    for prompt_record in iter_prompts(args.prompts, args.limit):
        try:
            assert_trainable_task(prompt_record)
        except ValueError as exc:
            print(f"skip non-trainable task: {exc}", file=sys.stderr)
            continue

        if args.orchestrate:
            step = run_dataset_generation_step(
                prompt_record,
                client=client,
                validator=None,
                decontaminator=decontaminator,
                evolve_depth=args.evolve_depth,
                run_id=run_id,
                run_benchmark=args.benchmark,
                debug_split=args.out / "failure_records.jsonl",
                mined_split=args.out / "mined_tasks.jsonl",
            )
            adjudication.append(step)
            for item in step["results"]:
                if item.get("status") == "accepted":
                    trajectories.append(item["trajectory"])
            continue

        winner, all_candidates = generate_best_of_n(
            gateway=gateway,
            api_key=api_key,
            prompt_record=prompt_record,
            providers=providers,
            max_tokens=args.max_tokens,
            max_repairs=args.max_repairs,
            gpu_index=args.gpu,
            run_benchmark=args.benchmark,
        )
        adjudication.append(
            {
                "task_id": prompt_record.get("task_id"),
                "winner_provider": winner.provider if winner else None,
                "candidates": [
                    {
                        "provider": c.provider,
                        "passed": c.validation.get("passed"),
                        "score": c.score,
                        "repairs_used": c.repairs_used,
                    }
                    for c in all_candidates
                ],
            }
        )
        if winner is not None:
            trajectories.append(winner.record)
            if args.evolve_depth > 0:
                evolved_tasks.extend(evolve_verified_trajectory(winner.record, depth=args.evolve_depth))
        else:
            failed = next((c for c in reversed(all_candidates)), None)
            if failed is not None:
                failure = record_failure(
                    run_id=run_id,
                    task=prompt_record,
                    model=providers[0],
                    validation=failed.validation,
                    response=failed.record.get("response", ""),
                )
                _append_jsonl(args.out / "failure_records.jsonl", failure)

    if evolved_tasks:
        evolved_path = args.out / "evolved_tasks.jsonl"
        evolved_path.parent.mkdir(parents=True, exist_ok=True)
        evolved_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in evolved_tasks))

    if args.decontaminate:
        trajectories = filter_decontaminated(trajectories)

    gen_config = {
        "reasoning_effort": "xhigh",
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "multi_candidate": True,
        "max_repairs": args.max_repairs,
    }
    manifest = build_manifest(
        trajectories,
        prompts_sha256=sha256_file(str(args.prompts)),
        openrouter_generation_config=gen_config,
        gateway=gateway,
    ).to_dict()

    write_bundle(
        out_dir=args.out,
        trajectories=trajectories,
        manifest=manifest,
        prompts_path=args.prompts,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adjudication.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in adjudication)
    )

    print(
        f"multi-candidate: {len(trajectories)} winners from {len(adjudication)} prompts",
        file=sys.stderr,
    )

    if not args.skip_blackwell and trajectories:
        report = prove_blackwell_bundle(
            args.out,
            gpu_index=args.gpu,
            benchmark=args.benchmark,
            attest_gpu=not args.no_gpu_attest,
        )
        print(f"proved: {report['verified_count']}/{report['raw_count']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
