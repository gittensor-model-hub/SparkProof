# External corpora → SparkProof task seeds

This guide explains how to turn public Triton / KernelBook-style datasets into
**SN74-verifiable** SparkProof training data — without trusting external teachers or
pasting unverified CoT into the registry.

**Canonical flow:** external PyTorch problems → `prompts.jsonl` (`origin: kernelbook_seed`)
→ pinned teachers (Fable 5 / GPT 5.6 Sol) + multi-turn repair on a CC VM → release gate →
HF `proof/` → SparkDistill registry.

Full miner publish path: [`MINER_GUIDE.md`](MINER_GUIDE.md).

---

## Why not publish the HF datasets as-is?

| External field | In SparkProof `proof/`? |
|---|---|
| PyTorch module / problem | **Yes** — as the **task prompt** only |
| Inductor / author Triton kernel | **No** — not pinned-teacher provenance |
| gpt-oss / Opus CoT, `messages`, tool traces | **No** — wrong teachers, no CC attestation |
| KernelBench problems | **Never** — held-out eval (same class as TritonBench) |

SparkProof rewards require:

1. Pinned teachers + `request_sha256` (Fable / Sol @ `xhigh`)
2. Kernels that **compile and run** on an attested Blackwell or Hopper CC GPU
3. Merkle + release gate (decontam, novelty)
4. GPU CC + TDX attestation bound to the bundle

None of the public trace datasets satisfy (1)–(4). They are **curriculum and problem
factories**, not verified trajectories.

---

## Supported inputs

| Dataset | Default HF id | Role |
|---|---|---|
| [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | `GPUMODE/KernelBook` | Primary PyTorch modules (`python_code`) |
| [opus multi-turn traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-opus4.8-multiturn-traces) | `ppbhatt500/kernelbook-opus4.8-multiturn-traces` | Extra tasks (`pytorch_problem`) + optional **code-only** repair hints from failed turns |
| [gpt-oss reasoning traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) | `ppbhatt500/kernelbook-triton-reasoning-traces` | Extra tasks (`pytorch_code`) when `source` is not KernelBench |
| [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) | `ScalingIntelligence/KernelBench` | **Fingerprints only** — blocks eval leakage into seeds |

Local `.jsonl` / `.parquet` paths work the same as HF ids (useful for fixtures and air-gapped boxes).

---

## What the importer keeps and drops

Implemented in `sparkproof/triton_dataset/external_seeds.py` (`sparkproof-import-external-tasks`).

### Kept (per row)

- PyTorch source (`python_code` / `pytorch_problem` / `pytorch_code`)
- Entry point / module name
- Permissive licenses when present
- Optional `repair_hint_kernel`: last **failed** kernel **code** from opus `turns` before a later success (curriculum only — the teacher still regenerates the full trajectory)
- Metadata: `source_dataset`, `source_uuid`, `repo_name`, `repo_link`

### Dropped

| Reason | Stat key in `*.import.json` |
|---|---|
| Non-permissive or missing license (KernelBook / opus) | `skipped_license` |
| `source` / `origin` contains `kernelbench` | `skipped_blocked_source` |
| Empty PyTorch code | `skipped_empty` |
| Duplicate `torch_reference` fingerprint within the import | `skipped_duplicate` |
| Prompt / `torch_reference` / AST structure matches TritonBench or KernelBench | `skipped_decontam` |

Output rows always have:

```text
source = origin = "kernelbook_seed"
split  = "train"
category = "translation"
```

`FORBIDDEN_TRAINING_ORIGINS` includes `kernelbench` and `tritonbench` — release gate rejects any row that tries to smuggle eval origins.

---

## License policy

Default: **permissive only** (case-insensitive):

`MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `Unlicense`, `CC0-1.0`, `0BSD`

- KernelBook / opus rows **must** list only permissive licenses (missing license → skip).
- gpt-oss traces often omit licenses; rows are kept only when they are not KernelBench-sourced (already filtered) — prefer KernelBook-backed problems when mixing corpora.
- Escape hatch (not for production registry work): `--allow-nonpermissive-license`.

Respect upstream license metadata on the published SparkProof dataset card when you mix seeds.

---

## Decontamination

Before a seed is written, `TritonDecontaminator`:

1. Loads **TritonBench** problems (`--problems-dir` or `SPARKPROOF_TRITONBENCH_PROBLEMS`)
2. Loads **KernelBench** PyTorch `code` fields (all `level_*` splits on HF, or a local jsonl)
3. Rejects if:
   - full prompt text fingerprint matches eval
   - `torch_reference` fingerprint or **canonical AST structure** matches eval (class names are stripped so `ReLUTiny` vs `Model` with the same body still collide)

Production imports should pass a real TritonBench problems tree and **not** use `--no-kernelbench`.  
CI / offline fixtures may use `--no-require-eval-corpus` when TritonBench YAML is absent.

---

## End-to-end commands

### 1. Install publish deps (HF loaders)

```bash
cd SparkProof
uv sync --extra publish --frozen
```

### 2. Import seeds

```bash
# Convenience wrapper (KernelBook + opus + gpt-oss defaults)
scripts/import_external_tasks.sh --limit 50

# Explicit CLI
uv run sparkproof-import-external-tasks \
  --kernelbook GPUMODE/KernelBook \
  --opus-traces ppbhatt500/kernelbook-opus4.8-multiturn-traces \
  --gptoss-traces ppbhatt500/kernelbook-triton-reasoning-traces \
  --kernelbench ScalingIntelligence/KernelBench \
  --out prompts/kernelbook_seed.jsonl \
  --limit 50 \
  --gpu-architecture blackwell
```

Artifacts:

| File | Contents |
|---|---|
| `prompts/kernelbook_seed.jsonl` | Validated prompt records |
| `prompts/kernelbook_seed.import.json` | Scan stats (`kept`, `skipped_*`, fingerprint counts) |

Useful flags:

| Flag | Meaning |
|---|---|
| `--limit N` | Cap kept prompts |
| `--no-kernelbook` / omit opus/gptoss | Subset corpora |
| `--no-repair-hints` | Do not attach prior failed kernel code |
| `--no-kernelbench` | Skip KernelBench fingerprints (**not recommended**) |
| `--require-eval-corpus` | Fail if TritonBench fingerprints are empty |
| `--no-require-eval-corpus` | Allow empty TritonBench (local fixtures) |
| `--gpu-architecture` | `blackwell` / `hopper-h100` / `hopper-h200` baked into prompt text |

### 3. Mix with other prompt sources (optional)

```bash
uv run sparkproof-build-prompts \
  --out prompts/mixed.jsonl \
  --sources kernelbook_seed,torch_op,api_doc \
  --seed-prompts prompts/kernelbook_seed.jsonl \
  --limit 100
```

### 4. Re-prove on a CC VM (required for registry)

Pinned teachers + GPU validation + multi-turn episodes (default):

```bash
uv run sparkproof-triton-generate \
  --prompts prompts/kernelbook_seed.jsonl \
  --out bundles/kb-seed-001 \
  --decontaminate \
  --orchestrate \
  --benchmark
```

- Episodes record `task → attempt → validator fail → repair → optional optimize`.
- Do **not** pass external CoT as `response`; generation must come from Fable/Sol.
- `--no-episodes` / `--no-optimize` only for debugging.

### 5. Release gate, publish, registry PR

Same as any dataset-track run — see [`MINER_GUIDE.md`](MINER_GUIDE.md):

```bash
uv run sparkproof-publish-dataset \
  --bundle bundles/kb-seed-001 \
  --repo-id YOU/sparkproof-kb-seed-v1 \
  --release-gate --mining-repo
```

Then open a **text-only** SparkDistill PR appending one `datasets/registry.jsonl` line.

Fair labels use **novel rows after registry mix dedupe**, not your bundle size. Preview
`novelty_report.json` (`novel_verified_rows` ≥ 25 for `dataset:xs`).

---

## Failure modes / FAQ

**Importer exits 1 with 0 rows**  
Check `*.import.json`: often license filter, KernelBench decontam, or missing `datasets`
extra (`uv sync --extra publish`).

**“I already have correct Triton in opus traces — why regenerate?”**  
Correctness on someone else’s GPU is not SparkProof provenance. Regenerating binds
pinned teachers + your CC attestation. External gold is only a curriculum hint.

**Can repair hints leak KernelBench?**  
Hints are taken only from opus turns on rows that already passed decontam on the
PyTorch problem. The hint kernel itself is not validated as gold and is not exported
as the teacher response.

**Can I train a private student on raw opus/gpt-oss messages?**  
Yes for research, **outside** the attested registry / `sparkproof-mining` pin. Do not
claim SparkProof verification for that mix.

**Does this raise the training frontier automatically?**  
No. Dataset track only adds verified rows. Training-track `eval:*` still needs a recipe
PR that beats `runs/frontiers.json`.

---

## Implementation map

| Piece | Path |
|---|---|
| Import logic | `sparkproof/triton_dataset/external_seeds.py` |
| CLI | `sparkproof/cli/import_external_tasks.py` |
| Shell wrapper | `scripts/import_external_tasks.sh` |
| Decontam + KernelBench codes | `sparkproof/triton_dataset/decontaminate.py` |
| Origin policy | `sparkproof/triton_dataset/task_policy.py` |
| Prompt mix-in | `sparkproof-build-prompts --seed-prompts` / source `kernelbook_seed` |
| Tests / fixtures | `tests/test_external_seeds.py`, `tests/fixtures/external_seeds/` |
