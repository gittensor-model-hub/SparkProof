# Contributing to SparkProof

SparkProof is the **dataset provenance** companion to [SparkDistill](https://github.com/gittensor-model-hub/SparkDistill). Contributions here improve attested trajectory generation, verification, and bundle tooling — not model weights.

## Principles

- **OpenRouter or yunwu.ai.** Production bundles must route teacher calls through an approved gateway (`gateway: openrouter` or `gateway: yunwu`). Direct Anthropic/OpenAI API calls are rejected at verification.
- **Pinned teachers.** Only Fable 5 (`anthropic/claude-fable-5`) and GPT 5.6 Sol (`openai/gpt-5.6-sol`), both with **`reasoning.effort: xhigh`** — see `sparkproof/policy.py`.
- **Blackwell CC only.** Production runs on RTX PRO 6000 Blackwell CC VM — OpenRouter generation + Triton prove + GPU attestation. No Polaris.
- **Policy stays in sync.** Teacher slugs and logical models must match `SparkDistill/teacher/providers.py`.

## Before you open a PR

```bash
cd SparkProof
scripts/install.sh              # first boot only
scripts/miner_run.sh --limit 2  # smoke test
scripts/run_triton_pipeline.sh --limit 2  # required for triton_dataset changes

uv sync --extra dev
ruff check .
pytest -q

# Blackwell runner: execute all reference-kernel integration tests
SPARKPROOF_RUN_GPU_TESTS=1 pytest -q tests/test_reference_kernels_gpu.py
```

If you change verification rules or the OpenRouter request policy, run the full test suite and update `README.md`.

## Legal and terms-of-service gate (required)

**By contributing to SparkProof, publishing a bundle, or using this tooling to generate training data, you accept sole responsibility for legal and contractual compliance.** Project maintainers do not review, approve, or warrant your use of teacher APIs, prompts, or published datasets.

### You must verify before collecting teacher outputs

1. **OpenRouter terms** — Your use of `OPENROUTER_API_KEY`, including storing and reusing model outputs, must comply with [OpenRouter's terms](https://openrouter.ai/terms) and acceptable-use policies as they apply to your account and jurisdiction.

2. **Upstream provider terms** — Routes through OpenRouter still invoke upstream models (Anthropic, OpenAI). Before using teacher outputs to train, fine-tune, or distill another model (including Qwen3.5-4B or any competing foundation model), confirm that the applicable provider terms permit that use. Restrictions vary by provider, product, and account type.

3. **Prompt and data rights** — Use only prompts and source material you own, created for this purpose, or are explicitly licensed to use for model training. Do not include:
   - private customer or user chats without documented consent;
   - copyrighted text beyond what your license allows for derivative ML training;
   - confidential or regulated data (PII, PHI, etc.) unless you have a lawful basis and appropriate safeguards.

4. **No hidden-reasoning extraction** — Train on normal, inspectable artifacts: final answers, permitted explanations, tool calls, test outputs, citations, and structured fields returned by the API. Do not attempt to recover, reconstruct, or publish provider-internal reasoning that the API did not intentionally expose in the response payload.

5. **Publication** — When you publish bundles to Hugging Face or link them in SparkDistill PRs, you represent that you have the right to distribute the prompts and teacher outputs included in that bundle.

### Maintainer disclaimer

**Maintainers and contributors to this repository provide software as-is.** They are **not** responsible for:

- your compliance with OpenRouter, Anthropic, OpenAI, or other third-party terms;
- whether your dataset is lawful to collect, train on, or publish;
- licensing disputes, copyright claims, or regulatory exposure arising from your bundles;
- downstream use of your data by miners, evaluators, or third parties.

SparkProof verification passing means the bundle matches the technical policy (OpenRouter xhigh, Blackwell validation, GPU CC). **It is not a legal clearance.**

## What to contribute

- Bug fixes and tests for `sparkproof-verify` / `sparkproof-generate`
- Clearer verification rules (with tests and migration notes for existing bundles)
- Documentation and reproducibility improvements
- Integration hooks for SparkDistill eval (discuss in an issue first)

## What needs discussion first

- Changing the pinned teacher list or OpenRouter slugs (must align with SparkDistill)
- Relaxing OpenRouter-only gateway requirements
- Breaking changes to bundle or manifest schema

Open an issue before large design changes.

## License

By contributing code, you agree that your contributions are licensed under the same license as the project (MIT). That license covers **source code only** — not datasets you generate or publish using this tooling.
