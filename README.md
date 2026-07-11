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
just accepted on the teacher's word. Run entirely on your **RTX PRO 6000 Blackwell CC VM**:

1. Calls teachers via **OpenRouter** (`reasoning.effort: xhigh`)
2. **Compiles and executes** Triton 3.7.1 kernels on Blackwell
3. Seals **`sparkproof-2`** with **GPU CC attestation** + Merkle root over verified samples

| Gateway | Base URL | Env key | Model slugs |
|---|---|---|---|
| **openrouter** (default) | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `anthropic/claude-fable-5`, `openai/gpt-5.6-sol` |
| **yunwu** | `https://yunwu.ai/v1` | `YUNWU_API_KEY` | `claude-fable-5`, `gpt-5.6-sol` (same teachers as OpenRouter; see [yunwu docs](https://yunwu.apifox.cn/)) |

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
| A API docs | `triton_dataset/doc_chunks.py` | Auto-fetch Triton docs → ~129 prompts — see [`docs/DOC_CHUNK_PROMPTS.md`](docs/DOC_CHUNK_PROMPTS.md) |
| B Mutation | `triton_dataset/mutator.py` | 15 deterministic, syntax-safe variants across 6 reference kernels |
| C Torch ops | `triton_dataset/torch_ops.py` | 17 PyTorch → Triton translation tasks |
| D Self-evolution | `triton_dataset/self_evolve.py` | Deterministic ops over oracle-backed parents |
| E Failure-mining | `triton_dataset/failure_miner.py` | Dev failures → new private tasks (never eval) |
| Eval only | `eval_problems.py` + `eval_harness.py` | `sparkproof-eval-tritonbench` — isolated from dataset |

Guards: `task_policy.assert_trainable_task()` blocks `tritonbench` / `eval` split from generation.
Decontamination: AST structure + prompt hash + semantic fingerprint (`decontaminate.py`).
Release gate: `--release-gate` on `sparkproof-publish-dataset`.

```bash
# Full Triton pipeline (prompts → best-of-N + repair → prove → verify → SFT → optional HF)
scripts/run_triton_pipeline.sh --limit 2
scripts/run_full_diverse.sh --run-id diverse-001 --train   # all doc + mutation + torch_op
scripts/run_triton_pipeline.sh --run-id triton-cc-001 --publish your-org/sparkproof-triton-v1 --release-gate

# TritonBench eval (held-out — results go outside training dirs)
uv run sparkproof-eval-tritonbench \
  --endpoint http://localhost:8000/v1 \
  --model sparkdistill-triton-qwen-4b \
  --out results/tritonbench_round1.json

# Step by step
scripts/build_triton_prompts.sh --out prompts/triton.jsonl
scripts/run_doc_qwen.sh --run-id doc-full-001          # doc-only: api + semantics + tutorials
uv run sparkproof-triton-generate --prompts prompts/triton.jsonl --out bundles/run-001 --decontaminate --orchestrate
scripts/build_next_round.sh --bundle bundles/run-001 --out prompts/round-2.jsonl
uv run sparkproof-publish-dataset --bundle bundles/run-001 --repo-id your-org/dataset --release-gate
```

Multi-candidate uses **yunwu/openrouter** gateways (Fable 5 + GPT 5.6 xhigh), not raw OpenAI/Anthropic SDKs.

`sparkproof-publish-dataset` uploads the dataset rows **and** the bundle's proof
artifacts (`manifest.json`, `dataset_manifest.json`, `gpu_attestation.json`,
`trajectories.jsonl`, ...) under `proof/` in the same HF repo. That is what lets a
SparkDistill validator re-verify everything from the HF link alone. To get the dataset
rewarded (`dataset:s/m/l`), open a text-only PR appending your HF URL and
`trajectories_sha256` to SparkDistill's `datasets/registry.jsonl` — see
`SparkDistill/datasets/README.md`.

What a verified sample proves:

- OpenRouter calls with pinned slugs + **`reasoning.effort: xhigh`** (`request_sha256` replay)
- Each kept sample passed Triton validation **on the attested Blackwell GPU**
- `gpu_attestation.json` from NVIDIA CC (NRAS)
- `trajectories.jsonl` = verified-only; `trajectories_raw.jsonl` = all teacher outputs

## Verifying proofs (no CC VM required)

**Proving** a bundle requires a Blackwell CC VM — generation, Triton validation, and NRAS
attestation all happen on the GPU node. **Verifying** a bundle does not: any CPU host
(GitHub Actions, a laptop, this repo's `sparkproof-verify`) can re-check stored artifacts.

```bash
# Offline — hashes, policy, merkle, raw→verified consistency, attestation nonce
uv run sparkproof-verify --bundle bundles/run-001

# Online — above + NVIDIA NRAS JWT signature against NVIDIA JWKS
uv run sparkproof-verify --bundle bundles/run-001 --online
```

SparkDistill validators run the same checks from the HF `proof/` directory via
`python -m eval.dataset_verify --hf-repo <org>/<repo> --sparkproof-root ../SparkProof`.

### What offline verify enforces

For each trajectory row, production verification checks the **stored bundle** — not live
hardware or live teacher API calls:

| Check | What it proves |
|---|---|
| `provider` + `model` | Only `claude-fable-5` (Anthropic) and `gpt-5.6` / `gpt-5.6-sol` (OpenAI) |
| `gateway` + `gateway_model` | Call went through **OpenRouter** or **yunwu** with pinned slugs |
| `request_sha256` | The committed request body matches the pinned call: model slug + `reasoning.effort=xhigh` + prompt/settings |
| `metadata.gateway_response_model` (yunwu) | Response model slug is also pinned (`claude-fable-5`, `gpt-5.6-sol`, or `gpt-5.6`) |
| raw → verified consistency | Miner cannot swap `trajectories.jsonl` after GPU attestation / release gate |
| `gpu_attestation` nonce | Attestation is bound to `trajectories_raw.jsonl`, not a different dataset |

**Offline verify means:** the miner recorded the exact pinned teacher slugs
(`claude-fable-5` + `gpt-5.6-sol`) via an approved gateway at `xhigh` reasoning, and did
not tamper with the bundle after proving. It is **not** a live cryptographic proof that
OpenAI/Anthropic actually served those models on every call — only that the committed JSON,
request fingerprint, and attestation binding are internally consistent.

### Offline vs online

| Mode | Teacher model guarantee | GPU guarantee |
|---|---|---|
| **Offline** | Bundle claims + `request_sha256` + gateway slug metadata + tamper checks | Stored `gpu_attestation.json` fields + nonce binding |
| **Online (`--online`)** | Same as offline | Above **plus** NVIDIA NRAS JWT signature verified against NVIDIA JWKS |
| **Online + OpenRouter ledger** | Can re-query OpenRouter generation IDs to confirm routed model — only for `gateway=openrouter` and only with the creating API key | Same as online |

For **yunwu** bundles there is currently no external teacher ledger re-check; offline trust
rests on stored gateway metadata, `request_sha256`, and post-prove tamper detection. A
dishonest relay could theoretically echo `gpt-5.6-sol` while serving a cheaper model;
swapping rows to another model (e.g. `gpt-4o-mini`) is caught by policy + raw/verified
consistency checks.

## Training-data strategy for a Triton specialist

The target model must combine five capabilities: Python/PyTorch coding, Triton programming,
GPU optimization, parallel-algorithm reasoning, and debugging/profiling tool use. The current
161 deterministic seeds are the verified foundation, not the final training scale.

Recommended SFT mixture:

| Capability | Share | Dataset source |
|---|---:|---|
| PyTorch → Triton | 30% | Operator specifications and externally verified translations |
| Debugging | 20% | Mutated kernels plus real compiler/runtime errors and verified fixes |
| Optimization | 20% | Correct before/after kernels with statistically stable speedups |
| Triton semantics/docs | 15% | API, semantics, and official tutorial prompts |
| Python/PyTorch tooling | 10% | Licensed high-quality coding replay data |
| Profiling/IR analysis | 5% | Real NCU, TTIR, TTGIR, and profiler artifacts |

Use frontier teachers as hypothesis generators; the compiler, PyTorch oracle, profiler, and
Blackwell GPU are the source of truth:

```text
task specification
  → best-of-N frontier teachers
  → syntax and Triton API validation
  → compile and execute
  → external numerical tests
  → anti-cheating checks
  → benchmark/profile
  → decontaminate
  → Blackwell prove and attest
  → SFT / preference datasets
```

### Acceptance requirements

Do not accept a sample solely because its teacher-written `torch.allclose` test passes.
SparkProof should test generated kernels independently across:

- tiny, normal, and adversarial dimensions, including tails such as 127, 1003, and 6143;
- contiguous and non-contiguous layouts;
- FP32, FP16, and BF16 where the operation supports them;
- multiple random seeds and extreme values for reductions or exponentials;
- unseen shapes that were not supplied to the teacher.

Anti-cheating checks inspect the launcher AST and its local helper call graph, reject forbidden
PyTorch compute fallbacks there, and confirm that a custom Triton kernel uses grid-launch
syntax. PyTorch reference operations remain allowed in top-level correctness tests. Replacing
a JIT kernel body with `pass` is not a reliable general anti-cheating test.

### Reasoning and debugging records

Request inspectable engineering rationale rather than private chain-of-thought. A useful
teacher response explains decomposition/grid, tile selection, pointer and stride equations,
masking, accumulation precision, expected bottleneck, implementation, and validation.

Debugging prompts must contain the **actual** error produced by running the broken kernel:

```text
input:  broken kernel + compiler/runtime output + failing shape/dtype
target: concise root cause + complete corrected kernel + regression test
```

Useful bug families include masks, strides, grid under-coverage, reduction axes, accumulator
precision, races/atomics, `tl.dot` constraints, descriptor/layout misuse, autotune errors, and
numerical overflow.

### Optimization records

Label a kernel as optimized only when it remains correct and repeated measurements prove an
improvement above noise. Record the baseline and optimized code, GPU and software versions,
shape/dtype/layout, warmups, iterations, median/tail latency, variance, speedup, and profiler
metrics. Use NCU on representative bottlenecks rather than every candidate. Slower but valid
candidates belong in preference/DPO pairs, not positive SFT examples.

Split train/dev/eval by operator family, reference kernel, mutation ancestry, and prompt
template—not randomly by row—to prevent near-duplicate leakage.

Suggested scale:

1. Smoke: 161 deterministic seeds.
2. Phase 1 SFT: 5,000–10,000 verified trajectories.
3. Phase 2: 20,000–50,000 shape/dtype/layout variants.
4. Preference training: at least 5,000 measured winner/loser pairs.
5. Execution RL: correctness plus measured performance reward.

SparkProof now includes launcher-scoped AST fallback detection, multi-seed adversarial
execution, real broken-kernel error capture, monitored `triton.testing.do_bench` preference
pairs, optional TTIR/TTGIR/PTX capture, and component-aware dataset splitting. Strict and IR
validation are reapplied at the Blackwell proving boundary, so generation evidence cannot be
silently downgraded:

```bash
uv run sparkproof-triton-generate \
  --prompts prompts/full.jsonl \
  --out bundles/run-001 \
  --strict-validate --benchmark --capture-ir \
  --export-dpo bundles/run-001/dpo.jsonl

uv run sparkproof-prove \
  --bundle bundles/run-001 \
  --strict-validate --benchmark --capture-ir
```

The same flags are available through the full pipeline script:

```bash
scripts/run_full_diverse.sh --run-id diverse-001 \
  --apply-templates --assign-dev-splits --torch-shape-variants \
  --strict-validate --capture-ir --export-dpo bundles/diverse-001/dpo.jsonl
```

- `--apply-templates` wraps prompts in structured design/implementation/validation sections.
- `--assign-dev-splits` assigns component-aware train/dev splits at prompt-build time
  (equivalent to running `sparkproof-split-dataset` on an existing prompts file).
- `--torch-shape-variants` adds adversarial shape presets to torch-op translation prompts.

`sparkproof-export-dpo --bundle bundles/run-001 --out dpo.jsonl` recovers preference pairs
from an existing bundle's adjudication + generation checkpoint, for bundles produced without
`--export-dpo` at generation time. Checkpoints record winning candidates only, so recovery can
backfill the original prompt but cannot reconstruct discarded losing candidates.

Operation-specific external shape/layout harnesses and representative NCU metric collection
remain future work; the current generic adversarial gate varies random seeds and relies on
verified task tests for shape, dtype, and stride coverage.

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
