# _SP⚡RKPROOF_

**Blackwell GPU–verified Triton dataset generation for [SPARKDISTILL](https://github.com/gittensor-model-hub/SparkDistill).**

**SPARKPROOF** is the dataset-provenance companion to
[`SparkDistill`](https://github.com/gittensor-model-hub/SparkDistill): it generates the
teacher trajectories SPARKDISTILL trains on, then proves — with GPU confidential-computing
attestation and a Merkle root over verified samples — that every kept sample actually
compiled and executed on an attested Blackwell GPU, not just that a teacher model emitted
plausible-looking text.

## Why SPARKPROOF

A distillation dataset is only as trustworthy as its provenance. SPARKPROOF's goal is
**verifiable data provenance**: prove a training sample was produced by a pinned teacher
model, at a pinned reasoning effort, and — for code — actually validated by running it, not
just accepted on the teacher's word. Run entirely on your **RTX PRO 6000 Blackwell CC VM** —
no Polaris, no CPU TDX:

1. Calls teachers via **OpenRouter** (`reasoning.effort: xhigh`)
2. **Compiles and executes** Triton 3.7.1 kernels on Blackwell
3. Seals **`sparkproof-2`** with **GPU CC attestation** + Merkle root over verified samples

| Gateway | Base URL | Env key | Model slugs |
|---|---|---|---|
| **openrouter** (default) | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `anthropic/claude-fable-5`, `openai/gpt-5.6-sol` |
| **yunwu** | `https://yunwu.ai/v1` | `YUNWU_API_KEY` | Native slugs from [yunwu docs](https://yunwu.apifox.cn/) — default `claude-sonnet-5`, `gpt-5-mini` (override via `YUNWU_MODEL_*`) |

Set `SPARKPROOF_GATEWAY=yunwu` or pass `--gateway yunwu` to `sparkproof-generate` / `miner_run.sh`.

## Layout

| Path | What |
|---|---|
| [`sparkproof/`](sparkproof) | gateway clients, GPU attestation, manifest/Merkle verification, Triton dataset pipeline |
| [`scripts/`](scripts) | one-command install/generate/verify/pipeline entry points |
| [`policies/`](policies) | pinned teacher + GPU policy (`gpu_remote_v3.json`) |
| [`tests/`](tests) | manifest, Merkle, policy, and gateway unit tests |

## CC VM quickstart (one command)

```bash
ssh -p 20002 ubuntu@<cc-host>

git clone https://github.com/gittensor-model-hub/SparkProof.git SparkProof
git clone https://github.com/gittensor-model-hub/SparkDistill.git SparkDistill   # sibling directory

cd SparkProof
cp .env.example .env   # OPENROUTER_API_KEY only

# First boot: install uv + deps on SparkProof and SparkDistill
scripts/install.sh

# Smoke test (2 prompts) → bundle → verify → SFT messages
scripts/miner_run.sh --limit 2

# Full run + train Qwen3.5-4B Phase 1
scripts/miner_run.sh --run-id phase1-cc-001 --train
```

`scripts/miner_run.sh` defaults: `phase1.jsonl` prompts, `bundles/<run-id>/`, SFT to
`SparkDistill/data/processed/<run-id>_sft.jsonl` (also copied to `phase1_sft.jsonl` for the recipe).

Dev flags: `--skip-blackwell`, `--no-gpu-attest`, `--allow-no-gpu-attest` (not for production PRs).

### Step-by-step (optional)

```bash
scripts/generate.sh --prompts ... --out bundles/run-001
scripts/verify.sh --bundle bundles/run-001
```

## Triton self-generating pipeline

Five prompt sources — **TritonBench YAML is eval-only** (never in training prompts):

| Source | Module | Status |
|--------|--------|--------|
| A API docs | `triton_dataset/doc_chunks.py` | Registry fallback + optional `--doc-dir` markdown |
| B Mutation | `triton_dataset/mutator.py` | vector_add, softmax + ground-truth kernels |
| C Torch ops | `triton_dataset/torch_ops.py` | LayerNorm, GELU, RMSNorm, Softmax, Matmul, SiLU, LogSoftmax |
| D Self-evolution | `triton_dataset/self_evolve.py` | Deterministic ops over oracle-backed parents |
| E Failure-mining | `triton_dataset/failure_miner.py` | Dev failures → new private tasks (never eval) |
| Eval only | `eval_problems.py` + `eval_harness.py` | `sparkproof-eval-tritonbench` — isolated from dataset |

Guards: `task_policy.assert_trainable_task()` blocks `tritonbench` / `eval` split from generation.
Decontamination: AST structure + prompt hash + semantic fingerprint (`decontaminate.py`).
Release gate: `--release-gate` on `sparkproof-publish-dataset`.

```bash
# Full Triton pipeline (prompts → best-of-N + repair → prove → verify → SFT → optional HF)
scripts/run_triton_pipeline.sh --limit 2
scripts/run_triton_pipeline.sh --run-id triton-cc-001 --publish your-org/sparkproof-triton-v1 --release-gate

# TritonBench eval (held-out — results go outside training dirs)
uv run sparkproof-eval-tritonbench \
  --endpoint http://localhost:8000/v1 \
  --model triton-qwen-9b \
  --out results/tritonbench_round1.json

# Step by step
scripts/build_triton_prompts.sh --out prompts/triton.jsonl
uv run sparkproof-triton-generate --prompts prompts/triton.jsonl --out bundles/run-001 --decontaminate --orchestrate
uv run sparkproof-publish-dataset --bundle bundles/run-001 --repo-id your-org/dataset --release-gate
```

Multi-candidate uses **yunwu/openrouter** gateways (Fable 5 + GPT 5.6 xhigh), not raw OpenAI/Anthropic SDKs.

What a verified sample proves:

- OpenRouter calls with pinned slugs + **`reasoning.effort: xhigh`** (`request_sha256` replay)
- Each kept sample passed Triton validation **on the attested Blackwell GPU**
- `gpu_attestation.json` from NVIDIA CC (NRAS)
- `trajectories.jsonl` = verified-only; `trajectories_raw.jsonl` = all teacher outputs

## Bundle layout

```
bundles/<run-id>/
  trajectories.jsonl
  trajectories_raw.jsonl
  validation_report.jsonl
  manifest.json              # sparkproof-2
  prompts.jsonl
  gpu_attestation.json
```

## Dev flags (not for production PRs)

| Flag | Effect |
|---|---|
| `--skip-blackwell` | Skip GPU validation |
| `--no-gpu-attest` | Validate on GPU but skip CC attestation |
| `verify --allow-no-gpu-attest` | Accept bundle without `gpu_attestation.json` |

## Requirements

- **Hardware:** Blackwell SM12x (RTX PRO 6000 Server Edition CC)
- **Software:** `torch>=2.6`, `triton==3.7.1`, `nv-attestation-sdk` for GPU CC
- **Secrets:** `OPENROUTER_API_KEY` and/or `YUNWU_API_KEY` (see `SPARKPROOF_GATEWAY`)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the pinned-teacher/gateway policy and the
legal and terms-of-service gate that applies to every published bundle.

## License

MIT, see [`LICENSE`](LICENSE).
