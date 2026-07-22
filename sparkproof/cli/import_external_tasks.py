"""Import KernelBook / external traces as SparkProof task seeds (not verified rows).

Extracts PyTorch modules only, decontaminates against TritonBench + KernelBench,
and writes prompts.jsonl for ``sparkproof-triton-generate`` (pinned teachers +
CC attestation). Never copies external CoT, messages, or Inductor kernels.

Examples::

    # Local fixtures (tests / offline)
    sparkproof-import-external-tasks \\
      --kernelbook tests/fixtures/external_seeds/kernelbook.jsonl \\
      --kernelbench tests/fixtures/external_seeds/kernelbench.jsonl \\
      --out prompts/kernelbook_seed.jsonl --no-require-eval-corpus

    # Hugging Face corpora (needs: uv sync --extra publish)
    sparkproof-import-external-tasks \\
      --opus-traces ppbhatt500/kernelbook-opus4.8-multiturn-traces \\
      --gptoss-traces ppbhatt500/kernelbook-triton-reasoning-traces \\
      --out prompts/kernelbook_seed.jsonl --limit 200

    # Then prove on a CC VM (multi-turn episodes on by default)
    sparkproof-triton-generate --prompts prompts/kernelbook_seed.jsonl \\
      --out bundles/kb-seed-001 --decontaminate --orchestrate --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkproof.gpu.architecture import ARCH_BLACKWELL, SUPPORTED_ARCHITECTURES, require_supported_gpu
from sparkproof.hashing import sha256_file
from sparkproof.triton_dataset.external_seeds import (
    DEFAULT_GPTOSS_TRACES,
    DEFAULT_KERNELBENCH,
    DEFAULT_KERNELBOOK,
    DEFAULT_OPUS_TRACES,
    import_seed_rows,
    write_seed_prompts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output prompts.jsonl")
    parser.add_argument("--report", type=Path, default=None, help="import report JSON path")
    parser.add_argument(
        "--kernelbook",
        default=DEFAULT_KERNELBOOK,
        help=f"HF id or local path for KernelBook (default: {DEFAULT_KERNELBOOK})",
    )
    parser.add_argument(
        "--no-kernelbook",
        action="store_true",
        help="skip KernelBook",
    )
    parser.add_argument(
        "--opus-traces",
        default=None,
        help=f"HF id or local path for opus multi-turn traces (e.g. {DEFAULT_OPUS_TRACES})",
    )
    parser.add_argument(
        "--gptoss-traces",
        default=None,
        help=f"HF id or local path for gpt-oss reasoning traces (e.g. {DEFAULT_GPTOSS_TRACES})",
    )
    parser.add_argument(
        "--kernelbench",
        default=DEFAULT_KERNELBENCH,
        help=f"HF id or local path for KernelBench eval fingerprints (default: {DEFAULT_KERNELBENCH})",
    )
    parser.add_argument(
        "--no-kernelbench",
        action="store_true",
        help="skip KernelBench fingerprint load (not recommended for production)",
    )
    parser.add_argument(
        "--problems-dir",
        type=Path,
        default=None,
        help="TritonBench problems directory for decontamination",
    )
    parser.add_argument(
        "--require-eval-corpus",
        action="store_true",
        help="fail if TritonBench problem fingerprints are empty",
    )
    parser.add_argument(
        "--no-require-eval-corpus",
        action="store_true",
        help="allow empty TritonBench corpus (local fixtures / CI without vendored problems)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--gpu-architecture",
        choices=sorted(SUPPORTED_ARCHITECTURES),
        default=None,
        help="GPU/SM label baked into prompt text (default: auto-detect, else blackwell)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--allow-nonpermissive-license",
        action="store_true",
        help="do not filter on permissive licenses (default: MIT/Apache/BSD only)",
    )
    parser.add_argument(
        "--no-repair-hints",
        action="store_true",
        help="do not attach code-only prior failed kernels from opus turns",
    )
    args = parser.parse_args(argv)

    gpu_architecture = args.gpu_architecture
    if gpu_architecture is None:
        try:
            gpu_architecture = require_supported_gpu(args.gpu)["gpu_architecture"]
        except RuntimeError:
            gpu_architecture = ARCH_BLACKWELL

    require_eval = args.require_eval_corpus and not args.no_require_eval_corpus

    records, stats = import_seed_rows(
        kernelbook=None if args.no_kernelbook else args.kernelbook,
        opus_traces=args.opus_traces,
        gptoss_traces=args.gptoss_traces,
        kernelbench=None if args.no_kernelbench else args.kernelbench,
        problems_dir=args.problems_dir,
        gpu_architecture=gpu_architecture,
        limit=args.limit,
        require_permissive_license=not args.allow_nonpermissive_license,
        include_repair_hints=not args.no_repair_hints,
        require_eval_corpus=require_eval,
    )
    count = write_seed_prompts(args.out, records)
    report_path = args.report or args.out.with_suffix(".import.json")
    report = {
        "prompts_path": str(args.out),
        "prompts_sha256": sha256_file(str(args.out)) if count else None,
        "gpu_architecture": gpu_architecture,
        "stats": stats,
        "note": (
            "Task seeds only — regenerate with sparkproof-triton-generate on a CC VM; "
            "do not publish external CoT as SparkProof trajectories."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {count} seed prompts to {args.out}", file=sys.stderr)
    print(f"import report: {report_path}", file=sys.stderr)
    print(json.dumps(stats, indent=2), file=sys.stderr)
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
