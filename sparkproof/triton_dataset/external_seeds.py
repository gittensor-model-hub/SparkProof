"""Import external PyTorch→Triton corpora as *task seeds only*.

Sources (KernelBook, opus multi-turn traces, gpt-oss reasoning traces) contribute
PyTorch problems — never their teacher CoT / messages / Inductor kernels — into
SparkProof ``prompts.jsonl``. Rows are re-solved by pinned teachers (Fable/Sol)
and GPU-attested on a CC VM.

KernelBench is eval-only: its problems are loaded only as decontamination
fingerprints and any seed whose ``source`` is kernelbench is dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from sparkproof.gpu.architecture import ARCH_BLACKWELL, sm_label
from sparkproof.triton_dataset.decontaminate import TritonDecontaminator, text_fingerprint
from sparkproof.triton_dataset.schema import validate_prompt_record
from sparkproof.triton_dataset.task_policy import normalize_train_task

SEED_SOURCE = "kernelbook_seed"
SEED_ORIGIN = "kernelbook_seed"
SEED_CATEGORY = "translation"

# Permissive license tokens (case-insensitive, normalized).
PERMISSIVE_LICENSE_TOKENS = frozenset(
    {
        "mit",
        "apache-2.0",
        "apache2.0",
        "apache-2",
        "apache 2.0",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-2",
        "bsd-3",
        "isc",
        "unlicense",
        "cc0-1.0",
        "cc0",
        "0bsd",
    }
)

# Never train on KernelBench / TritonBench problems.
_BLOCKED_SOURCE_TOKENS = frozenset(
    {
        "kernelbench",
        "kernelbench_eval",
        "tritonbench",
        "scalingintelligence/kernelbench",
    }
)

DEFAULT_KERNELBOOK = "GPUMODE/KernelBook"
DEFAULT_OPUS_TRACES = "ppbhatt500/kernelbook-opus4.8-multiturn-traces"
DEFAULT_GPTOSS_TRACES = "ppbhatt500/kernelbook-triton-reasoning-traces"
DEFAULT_KERNELBENCH = "ScalingIntelligence/KernelBench"


def _normalize_license_token(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", "-"))


def licenses_permissive(licenses: Any) -> bool:
    """True when every listed license is permissive (empty list → reject)."""
    if licenses is None:
        return False
    if isinstance(licenses, str):
        licenses = [licenses]
    if not isinstance(licenses, (list, tuple)) or not licenses:
        return False
    for item in licenses:
        token = _normalize_license_token(str(item))
        if token not in PERMISSIVE_LICENSE_TOKENS:
            return False
    return True


def _blocked_source_label(value: Any) -> bool:
    if value is None:
        return False
    token = str(value).strip().lower().replace(" ", "")
    return token in _BLOCKED_SOURCE_TOKENS or "kernelbench" in token


def _slug(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or "task")[:max_len]


def build_seed_prompt(
    *,
    pytorch_code: str,
    entry_point: str,
    task_id: str,
    gpu_architecture: str = ARCH_BLACKWELL,
    source_dataset: str,
    source_uuid: str | int | None = None,
    licenses: list[str] | None = None,
    repo_name: str | None = None,
    repo_link: str | None = None,
    repair_hint_kernel: str | None = None,
) -> dict[str, Any]:
    """Build a SparkProof training prompt from a PyTorch module only."""
    code = pytorch_code.strip()
    if not code:
        raise ValueError("pytorch_code must be non-empty")
    gpu_label = sm_label(gpu_architecture)
    prompt = f"""Write a Triton 3.7.1 kernel replicating this PyTorch module on {gpu_label}.

Entry point / module name: `{entry_point}`

```python
{code}
```

Requirements:
1. @triton.jit kernel + host launcher with tl.cdiv grid
2. Boundary masks on tl.load/tl.store where needed
3. fp32 accumulator for reductions
4. Self-contained test matching the module's get_inputs()/get_init_inputs() when present, otherwise a minimal torch.allclose check
5. Print SPARKPROOF_TRITON_PASS after successful test
6. Invoke triton.testing.do_bench(lambda: launcher(...)) on your correctness-test inputs
7. Do not call the PyTorch module from the kernel path — implement the op in Triton"""
    if repair_hint_kernel and repair_hint_kernel.strip():
        # Code-only prior attempt for curriculum; teacher still regenerates the trajectory.
        prompt += (
            "\n\nA previous attempt failed validation. You may use it only as a starting "
            "point — fix correctness, do not copy broken patterns:\n"
            f"```python\n{repair_hint_kernel.strip()}\n```"
        )

    record: dict[str, Any] = {
        "task_id": task_id,
        "source": SEED_SOURCE,
        "origin": SEED_ORIGIN,
        "split": "train",
        "category": SEED_CATEGORY,
        "task_family": _slug(entry_point),
        "prompt": prompt,
        "torch_reference": code,
        "reference_expr": code,
        "gpu_architecture": gpu_architecture,
        "source_dataset": source_dataset,
        "entry_point": entry_point,
    }
    if source_uuid is not None:
        record["source_uuid"] = source_uuid
    if licenses:
        record["licenses"] = list(licenses)
    if repo_name:
        record["repo_name"] = repo_name
    if repo_link:
        record["repo_link"] = repo_link
    if repair_hint_kernel and repair_hint_kernel.strip():
        record["repair_hint_kernel"] = repair_hint_kernel.strip()
    return normalize_train_task(record)


def extract_repair_hint_from_opus_turns(turns: Any) -> str | None:
    """Return the last failed kernel code before a later success (code only)."""
    if not isinstance(turns, list) or not turns:
        return None
    failed_kernels: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        kernel = turn.get("kernel")
        if not isinstance(kernel, str) or not kernel.strip():
            continue
        correct = turn.get("correct")
        status = str(turn.get("status") or "").lower()
        if correct is False or status in {"fail", "failed", "error"}:
            failed_kernels.append(kernel.strip())
        elif correct is True or status in {"ok", "pass", "passed"}:
            # Prefer the attempt immediately before success.
            if failed_kernels:
                return failed_kernels[-1]
    return failed_kernels[-1] if failed_kernels else None


def iter_records(spec: str, *, splits: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """Yield row dicts from a local path or Hugging Face dataset id."""
    path = Path(spec).expanduser()
    if path.exists():
        yield from _iter_local(path)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "loading Hugging Face datasets requires the publish extra: "
            "uv sync --extra publish"
        ) from exc

    if splits:
        for split in splits:
            ds = load_dataset(spec, split=split)
            for row in ds:
                yield dict(row)
        return
    ds = load_dataset(spec, split="train")
    for row in ds:
        yield dict(row)


def _iter_local(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix in {".jsonl", ".json", ".parquet"}:
                yield from _iter_local(child)
        return
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            for row in payload["data"]:
                if isinstance(row, dict):
                    yield row
        return
    if path.suffix == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("reading parquet requires: uv sync --extra publish") from exc
        ds = Dataset.from_parquet(str(path))
        for row in ds:
            yield dict(row)
        return
    raise ValueError(f"unsupported local seed file: {path}")


def load_kernelbench_into_decontaminator(
    decontaminator: TritonDecontaminator,
    kernelbench_spec: str | None,
) -> int:
    """Fingerprint KernelBench PyTorch problems so they never enter training seeds."""
    if not kernelbench_spec:
        return 0
    # KernelBench publishes level_1..level_4 splits.
    splits = ["level_1", "level_2", "level_3", "level_4"]
    count = 0
    try:
        rows: Iterable[dict[str, Any]] = iter_records(kernelbench_spec, splits=splits)
    except Exception:
        # Local fixture may be a single jsonl without HF splits.
        rows = iter_records(kernelbench_spec)
    for row in rows:
        code = str(row.get("code") or row.get("pytorch_code") or "").strip()
        if not code:
            continue
        decontaminator.add_eval_pytorch_code(code)
        name = str(row.get("name") or row.get("problem_id") or "")
        if name:
            decontaminator.prompt_hashes.add(text_fingerprint(name))
        count += 1
    return count


def _row_pytorch_fields(row: dict[str, Any]) -> tuple[str, str, Any] | None:
    """Return (pytorch_code, entry_point, uuid) or None if not a usable seed row."""
    if _blocked_source_label(row.get("source")) or _blocked_source_label(row.get("origin")):
        return None
    code = (
        row.get("pytorch_problem")
        or row.get("python_code")
        or row.get("pytorch_code")
        or ""
    )
    code = str(code).strip()
    if not code:
        return None
    entry = str(row.get("entry_point") or row.get("name") or row.get("module_name") or "Module").strip()
    uid = row.get("kernelbook_uuid", row.get("uuid", row.get("problem_id", row.get("sample_key"))))
    return code, entry, uid


def import_seed_rows(
    *,
    kernelbook: str | None = DEFAULT_KERNELBOOK,
    opus_traces: str | None = None,
    gptoss_traces: str | None = None,
    kernelbench: str | None = DEFAULT_KERNELBENCH,
    problems_dir: Path | None = None,
    gpu_architecture: str = ARCH_BLACKWELL,
    limit: int | None = None,
    require_permissive_license: bool = True,
    include_repair_hints: bool = True,
    require_eval_corpus: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Import external corpora into validated SparkProof prompt records."""
    decontaminator = TritonDecontaminator(
        problems_dir=problems_dir,
        require_eval_corpus=require_eval_corpus,
    )
    kb_fps = load_kernelbench_into_decontaminator(decontaminator, kernelbench)

    seen_code_fps: set[str] = set()
    kept: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "scanned": 0,
        "kept": 0,
        "skipped_license": 0,
        "skipped_blocked_source": 0,
        "skipped_empty": 0,
        "skipped_duplicate": 0,
        "skipped_decontam": 0,
        "with_repair_hint": 0,
        "kernelbench_fingerprints": kb_fps,
        "decontam_fingerprints": decontaminator.fingerprint_counts,
        "by_dataset": {},
    }

    sources: list[tuple[str, str | None]] = [
        ("kernelbook", kernelbook),
        ("opus_traces", opus_traces),
        ("gptoss_traces", gptoss_traces),
    ]

    for dataset_label, spec in sources:
        if not spec:
            continue
        dataset_kept = 0
        for row in iter_records(spec):
            stats["scanned"] += 1
            if _blocked_source_label(row.get("source")) or _blocked_source_label(row.get("origin")):
                stats["skipped_blocked_source"] += 1
                continue
            parsed = _row_pytorch_fields(row)
            if parsed is None:
                stats["skipped_empty"] += 1
                continue
            code, entry, uid = parsed
            licenses = row.get("licenses")
            if require_permissive_license:
                if licenses is not None:
                    if not licenses_permissive(licenses):
                        stats["skipped_license"] += 1
                        continue
                elif dataset_label in {"kernelbook", "opus_traces"}:
                    # KernelBook / opus rows always carry licenses in the public corpora.
                    stats["skipped_license"] += 1
                    continue
                # gpt-oss traces often omit licenses; keep only non-KernelBench rows (already filtered).

            code_fp = text_fingerprint(code)
            if code_fp in seen_code_fps:
                stats["skipped_duplicate"] += 1
                continue

            repair_hint = None
            if include_repair_hints and dataset_label == "opus_traces":
                repair_hint = extract_repair_hint_from_opus_turns(row.get("turns"))

            task_id = f"kb_{_slug(entry)}_{uid if uid is not None else code_fp[:10]}"
            try:
                record = build_seed_prompt(
                    pytorch_code=code,
                    entry_point=entry,
                    task_id=task_id,
                    gpu_architecture=gpu_architecture,
                    source_dataset=spec,
                    source_uuid=uid,
                    licenses=[str(x) for x in licenses] if isinstance(licenses, list) else None,
                    repo_name=str(row["repo_name"]) if row.get("repo_name") else None,
                    repo_link=str(row["repo_link"]) if row.get("repo_link") else None,
                    repair_hint_kernel=repair_hint,
                )
            except ValueError:
                stats["skipped_empty"] += 1
                continue

            issues = decontaminator.check_task(record)
            if issues or decontaminator.is_contaminated_code(code):
                stats["skipped_decontam"] += 1
                continue

            record = validate_prompt_record(record)
            seen_code_fps.add(code_fp)
            kept.append(record)
            dataset_kept += 1
            if repair_hint:
                stats["with_repair_hint"] += 1
            if limit is not None and len(kept) >= limit:
                stats["by_dataset"][dataset_label] = dataset_kept
                stats["kept"] = len(kept)
                return kept, stats
        stats["by_dataset"][dataset_label] = dataset_kept

    stats["kept"] = len(kept)
    return kept, stats


def write_seed_prompts(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            validated = validate_prompt_record(rec)
            f.write(json.dumps(validated, ensure_ascii=False) + "\n")
    return len(records)
