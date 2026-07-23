"""Multi-candidate Triton dataset generation → Blackwell prove → sparkproof-2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkproof.bundle import write_bundle
from sparkproof.env_loader import load_dotenv
from sparkproof.gateways import ALLOWED_GATEWAYS, default_gateway, resolve_api_key
from sparkproof.generate.gateway_client import GatewayTransientError
from sparkproof.generate.runner import iter_prompts
from sparkproof.hashing import sha256_file
from sparkproof.manifest import build_manifest
from sparkproof.pipeline.blackwell import prove_blackwell_bundle
from sparkproof.triton_dataset.decontaminate import TritonDecontaminator
from sparkproof.triton_dataset.dpo_export import enrich_adjudication_with_responses, write_dpo_jsonl, export_dpo_jsonl
from sparkproof.triton_dataset.failure_miner import mine_failure_to_tasks, record_failure
from sparkproof.triton_dataset.multi_candidate import generate_best_of_n
from sparkproof.triton_dataset.orchestrate import run_dataset_generation_step
from sparkproof.triton_dataset.run_seed import generate_run_seed
from sparkproof.triton_dataset.self_evolve import evolve_verified_trajectory
from sparkproof.triton_dataset.prompt_filters import parse_filter_set
from sparkproof.triton_dataset.task_policy import assert_trainable_task


def _load_env() -> None:
    load_dotenv()


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at {path}:{line_number}") from exc
    return rows


def _gateway_error_row(prompt_record: dict, exc: Exception) -> dict:
    return {
        "task_id": prompt_record.get("task_id"),
        "prompt": prompt_record.get("prompt", ""),
        "system": prompt_record.get("system", ""),
        "status": "gateway_error",
        "error": str(exc),
        "winner_provider": None,
        "candidates": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gateway", choices=sorted(ALLOWED_GATEWAYS), default=None)
    parser.add_argument("--provider", dest="providers", action="append", choices=["anthropic", "openai"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source",
        dest="filter_sources",
        action="append",
        help="only run prompts with this source (repeatable), e.g. api_doc, doc_semantics",
    )
    parser.add_argument(
        "--task-id",
        dest="filter_task_ids",
        action="append",
        help="only run prompts with this task_id (repeatable), e.g. api_tl_dot",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument(
        "--no-episodes",
        action="store_true",
        help="disable multi-turn episode recording (legacy single-turn trajectories only)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="skip the post-correct measured optimization teacher pass (requires --benchmark)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help="run anti-cheat AST checks and multi-seed adversarial execution",
    )
    parser.add_argument(
        "--capture-ir",
        action="store_true",
        help="attach TTIR/TTGIR artifacts to validation when available",
    )
    parser.add_argument(
        "--export-dpo",
        type=Path,
        default=None,
        help="write optimization preference pairs from adjudication to this jsonl path",
    )
    parser.add_argument("--dpo-min-speedup", type=float, default=0.03)
    parser.add_argument("--no-gpu-attest", action="store_true")
    parser.add_argument("--skip-blackwell", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="development only: allow a run with zero winners")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.5,
        help="minimum Blackwell revalidation pass rate (default: 0.5)",
    )
    parser.add_argument("--decontaminate", action="store_true", help="drop structurally contaminated samples")
    parser.add_argument(
        "--problems-dir",
        type=Path,
        default=None,
        help="TritonBench problems directory used for strict decontamination",
    )
    parser.add_argument(
        "--benchmark-py-dir",
        type=Path,
        default=None,
        help="optional held-out benchmark Python tree for structural fingerprints",
    )
    parser.add_argument("--orchestrate", action="store_true", help="evolve tasks + mine failures per prompt")
    parser.add_argument("--evolve-depth", type=int, default=1)
    parser.add_argument(
        "--run-seed",
        default=None,
        help="hex entropy scoping self-evolution op selection to this run (identity-free; "
        "auto-generated and printed if omitted, not required to match the seed used at "
        "build-prompts time)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="resume from generation checkpoints in --out")
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")
    if not 0.0 <= args.dpo_min_speedup < 1.0:
        parser.error("--dpo-min-speedup must be between 0 and 1")
    if args.export_dpo and not args.benchmark:
        parser.error("--export-dpo requires --benchmark")

    _load_env()
    run_seed = args.run_seed or generate_run_seed()
    if not args.run_seed:
        print(f"generated run seed (persist to replay this run): {run_seed}", file=sys.stderr)
    gateway = args.gateway or default_gateway()
    api_key = resolve_api_key(gateway)
    providers = args.providers or ["anthropic", "openai"]

    trajectory_checkpoint = args.out / "generation_checkpoint.jsonl"
    adjudication_checkpoint = args.out / "generation_adjudication.jsonl"
    if not args.resume:
        trajectory_checkpoint.unlink(missing_ok=True)
        adjudication_checkpoint.unlink(missing_ok=True)
    trajectories: list[dict] = _load_jsonl(trajectory_checkpoint) if args.resume else []
    adjudication: list[dict] = _load_jsonl(adjudication_checkpoint) if args.resume else []
    completed_task_ids = {
        row.get("base_task_id") or row.get("task_id")
        for row in adjudication
        if row.get("base_task_id") or row.get("task_id")
    }
    evolved_tasks: list[dict] = []
    run_id = args.run_id or args.out.name
    decontaminator = TritonDecontaminator(
        problems_dir=args.problems_dir,
        benchmark_py_dir=args.benchmark_py_dir,
        require_eval_corpus=args.decontaminate or args.orchestrate,
    )
    if args.decontaminate or args.orchestrate:
        counts = decontaminator.fingerprint_counts
        print(
            f"decontamination fingerprints: {counts['prompts']} prompts, "
            f"{counts['semantics']} semantic, {counts['structures']} code",
            file=sys.stderr,
        )
    client = {
        "gateway": gateway,
        "api_key": api_key,
        "providers": providers,
        "max_tokens": args.max_tokens,
        "max_repairs": args.max_repairs,
        "gpu_index": args.gpu,
        "strict_validate": args.strict_validate,
        "capture_ir": args.capture_ir,
    }

    filter_sources = parse_filter_set(args.filter_sources)
    filter_task_ids = parse_filter_set(args.filter_task_ids)

    for prompt_record in iter_prompts(
        args.prompts,
        args.limit,
        sources=filter_sources,
        task_ids=filter_task_ids,
    ):
        if prompt_record.get("task_id") in completed_task_ids:
            continue
        try:
            assert_trainable_task(prompt_record)
        except ValueError as exc:
            print(f"skip non-trainable task: {exc}", file=sys.stderr)
            continue

        if args.orchestrate:
            try:
                step = run_dataset_generation_step(
                    prompt_record,
                    client=client,
                    validator=None,
                    decontaminator=decontaminator,
                    evolve_depth=args.evolve_depth,
                    run_id=run_id,
                    run_benchmark=args.benchmark,
                    run_seed=run_seed,
                    debug_split=args.out / "failure_records.jsonl",
                    mined_split=args.out / "mined_tasks.jsonl",
                )
            except GatewayTransientError as exc:
                print(
                    f"gateway error for {prompt_record.get('task_id')}: {exc}",
                    file=sys.stderr,
                )
                step = _gateway_error_row(prompt_record, exc)
            adjudication.append(step)
            _append_jsonl(adjudication_checkpoint, step)
            for item in step.get("results", []):
                if item.get("status") == "accepted":
                    trajectories.append(item["trajectory"])
                    _append_jsonl(trajectory_checkpoint, item["trajectory"])
            continue

        try:
            winner, all_candidates = generate_best_of_n(
                gateway=gateway,
                api_key=api_key,
                prompt_record=prompt_record,
                providers=providers,
                max_tokens=args.max_tokens,
                max_repairs=args.max_repairs,
                gpu_index=args.gpu,
                run_benchmark=args.benchmark,
                strict_validate=args.strict_validate,
                capture_ir=args.capture_ir,
                record_episode=not args.no_episodes,
                enable_optimize=args.benchmark and not args.no_optimize,
            )
        except GatewayTransientError as exc:
            print(
                f"gateway error for {prompt_record.get('task_id')}: {exc}",
                file=sys.stderr,
            )
            adjudication_row = _gateway_error_row(prompt_record, exc)
            adjudication.append(adjudication_row)
            _append_jsonl(adjudication_checkpoint, adjudication_row)
            continue

        adjudication_row = {
            "task_id": prompt_record.get("task_id"),
            "prompt": prompt_record.get("prompt", ""),
            "system": prompt_record.get("system", ""),
            "winner_provider": winner.provider if winner else None,
            "candidates": [
                {
                    "provider": c.provider,
                    "passed": c.validation.get("passed"),
                    "score": c.score,
                    "repairs_used": c.repairs_used,
                    "validation": c.validation,
                    "response": c.record.get("response", ""),
                }
                for c in all_candidates
            ],
        }
        adjudication.append(adjudication_row)
        _append_jsonl(adjudication_checkpoint, adjudication_row)
        if winner is not None:
            trajectories.append(winner.record)
            _append_jsonl(trajectory_checkpoint, winner.record)
            if args.evolve_depth > 0:
                evolved_tasks.extend(
                    evolve_verified_trajectory(winner.record, depth=args.evolve_depth, run_seed=run_seed)
                )
        else:
            failed = next((c for c in reversed(all_candidates)), None)
            if failed is not None:
                failure = record_failure(
                    run_id=run_id,
                    task=prompt_record,
                    model=failed.provider,
                    validation=failed.validation,
                    response=failed.record.get("response", ""),
                )
                _append_jsonl(args.out / "failure_records.jsonl", failure)
                for mined in mine_failure_to_tasks(failure):
                    _append_jsonl(args.out / "mined_tasks.jsonl", mined)

    if evolved_tasks:
        evolved_path = args.out / "evolved_tasks.jsonl"
        evolved_path.parent.mkdir(parents=True, exist_ok=True)
        evolved_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in evolved_tasks))

    if args.decontaminate:
        trajectories = decontaminator.filter_trajectories(trajectories)

    gen_config = {
        "reasoning_effort": "xhigh",
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "multi_candidate": True,
        "max_repairs": args.max_repairs,
        "record_episodes": not args.no_episodes,
        "enable_optimize": bool(args.benchmark and not args.no_optimize),
    }
    sampling_report_path = args.prompts.with_suffix(".sampling.json")
    sampling_provenance = None
    if sampling_report_path.is_file():
        sampling_provenance = json.loads(sampling_report_path.read_text(encoding="utf-8"))
        sampling_provenance["generation_run_seed"] = run_seed
    manifest = build_manifest(
        trajectories,
        prompts_sha256=sha256_file(str(args.prompts)),
        openrouter_generation_config=gen_config,
        gateway=gateway,
        sampling=sampling_provenance,
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

    if args.export_dpo:
        enriched = enrich_adjudication_with_responses(
            adjudication,
            checkpoint_path=trajectory_checkpoint,
        )
        pairs = export_dpo_jsonl(enriched, min_speedup=args.dpo_min_speedup)
        count = write_dpo_jsonl(args.export_dpo, pairs)
        print(f"exported {count} DPO preference pairs to {args.export_dpo}", file=sys.stderr)
        if count == 0:
            print(
                "warning: no task had two passing, monitored benchmarks above the speedup threshold",
                file=sys.stderr,
            )

    print(
        f"multi-candidate: {len(trajectories)} winners from {len(adjudication)} prompts",
        file=sys.stderr,
    )

    if not trajectories and not args.allow_empty:
        print("error: generation produced zero accepted trajectories", file=sys.stderr)
        return 2

    if not args.skip_blackwell and trajectories:
        report = prove_blackwell_bundle(
            args.out,
            gpu_index=args.gpu,
            benchmark=args.benchmark,
            strict_validate=args.strict_validate,
            capture_ir=args.capture_ir,
            attest_gpu=not args.no_gpu_attest,
            min_pass_rate=args.min_pass_rate,
        )
        print(f"proved: {report['verified_count']}/{report['raw_count']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
