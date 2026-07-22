# Changelog

All notable changes to SparkProof are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
`generator_version` in published bundles tracks the SparkProof release that produced them (currently `0.3.0`).

---

## [Unreleased]

### Added
- **Multi-turn training episodes** (fail → critique → fix → optimize → accept):
  `generate_with_repair` records a `metadata.episode` (`sparkproof-episode-v1`) with
  real validator feedback turns and an optional measured optimization pass when
  `--benchmark` is set. HF/SFT export prefers multi-turn chat messages over
  single Prompt→Answer rows. Flags: `--no-episodes`, `--no-optimize`.

### Fixed
- **TDX binding reads REPORTDATA from ``quote_b64``, not JSON** : `verify_tdx_attestation`
  extracts the 64-byte REPORTDATA at the TDX v4 offset inside the quote and compares it
  to `tdx_report_data(nonce)`. Miner-editable `tdx.report_data` can no longer rebind a
  genuine Intel-signed quote to another dataset nonce; JSON/quote mismatches are rejected.

## [0.1.3] — 2026-07-21

Train-able CoT recovery when GPT 5.6 Sol wins with encrypted reasoning, plus repair-tier
novelty fingerprinting that matches the mining task (not the self-repair wrapper).

`generator_version` remains **0.3.0** (package release `0.1.3`).

### Added
- **Sol→Fable training-CoT recovery** ([#31]): when GPT 5.6 Sol wins best-of-N but only
  returns encrypted/empty reasoning, SparkProof calls Claude Fable 5 to (1) re-solve
  with Sol's verified kernel as a hint and re-validate, preferring Fable when it
  passes with plaintext CoT, else (2) explain Sol's gold answer and attach Fable's
  rationale under `reasoning` (`metadata.cot_recovery`: `fable_resolve` /
  `fable_explain`). SFT export skips encrypted reasoning JSON and prefers
  `prompt_meta.prompt` as the user turn.
- **`scripts/yunwu_reasoning_probe.py`**: probe yunwu `gpt-5.6-sol` reasoning shapes
  (plaintext vs encrypted `reasoning_details`) using the production request format.

### Fixed
- **Repair-tier novelty fingerprint** ([#29], [#30]): `fingerprint_row()` now prefers
  `metadata.prompt_meta.prompt` over top-level `prompt`, so repair rows fingerprint the
  mining task instead of the shared self-repair wrapper. Fixes false exact duplicates and
  pre-gen vs post-gen novelty mismatch on repair-heavy bundles.

## [0.1.1] — 2026-07-15

Hopper joins Blackwell for dataset generation, per-device NRAS tokens are
cryptographically verified, OpenRouter ledger checks accept dated build suffixes,
and Intel TDX closes the userland trust gap on the dataset track.

`generator_version` in new bundles remains **0.3.0** (package release `0.1.1`).

### Added

- **Per-device NRAS JWKS verification** ([#18]): `verify_nras_token` now
  signature-verifies every GPU device JWT in `REMOTE_GPU_CLAIMS` (not just the
  platform token). Signed `hwmodel` / driver claims are exposed under
  `claims["devices"]` — hardware corroboration no longer trusts unsigned JSON.
- **Hopper H100/H200 dataset generation** ([#20]): `sparkproof/gpu/architecture.py`
  accepts Blackwell (SM10x/12x) or Hopper H100/H200 (SM90, memory-size split).
  Prompt templates, doc chunks, mutation/failure-mining/self-evolution text, API-unit
  registry (Hopper excludes FP4), decontamination fingerprints, and
  `dataset_manifest.json` are architecture-aware.
- **Intel TDX measured-VM attestation** ([#22]): `sparkproof/gpu/tdx.py` captures
  Intel TDX quotes via configfs-tsm with `report_data` bound to the dataset
  attestation nonce. Production `sparkproof-verify` requires `gpu_attestation.tdx`
  on new bundles; `--online` DCAP-verifies via `dcap-qvl` (`tdx_signature_checked`).
  Legacy bundles without a `tdx` key are grandfathered.

### Changed

- **OpenRouter response model recording** ([#19]): trajectories store OpenRouter's
  actual `payload.model` in `gateway_model`; pinned request slug kept in
  `metadata.openrouter_requested_model`. Dated build suffixes (e.g.
  `gpt-5.6-sol-20260709`) accepted in policy and online ledger checks.
- **Legacy `gpu_architecture` fallback** ([#21]): pre-Hopper bundles missing
  `gpu_architecture` default to `"blackwell"` when `gpu_profile.family == "blackwell"`.

### Security

| Threat | Before | After |
|---|---|---|
| Unsigned per-device GPU claims in attestation JSON | Possible | **Blocked** — device JWTs JWKS-verified ([#18]) |
| Patched userland + valid NRAS GPU token | Possible | **Blocked** — TDX quote binds measured guest ([#22]) |
| OpenRouter dated slug vs pinned request model | `--online` false reject | **Accepted** when ledger matches ([#19]) |

### Miner setup (TDX guest, once per boot)

```bash
sudo chmod 0777 /sys/kernel/config/tsm/report
mkdir /sys/kernel/config/tsm/report/sparkproof
sudo chmod 0666 /sys/kernel/config/tsm/report/sparkproof/inblob
export SPARKPROOF_TSM_REPORT_PATH=/sys/kernel/config/tsm/report/sparkproof
```

Pairs with SparkDistill [#122](https://github.com/gittensor-model-hub/SparkDistill/pull/122).

---

## [0.1.2] — 2026-07-15

Miner-side accepted-registry snapshot download so novelty checks catch cross-registry
duplicates before publish — pairs with SparkDistill [#119](https://github.com/gittensor-model-hub/SparkDistill/pull/119).

`generator_version` remains **0.3.0** (package release `0.1.2`).

### Added

- **`sparkproof/triton_dataset/registry_snapshot.py`** — download
  `accepted_registry_snapshot.jsonl` + `mix_manifest.json` from the canonical mining HF
  repo; verify `accepted_registry_snapshot_sha256` and row-count pins.
- **`sparkproof-download-registry-snapshot`** CLI + `scripts/download_registry_snapshot.sh`
  — local download with optional `--verify-only` against live pins.
- **`sparkproof-publish-dataset --mining-repo`** — bare flag downloads from
  `gittensor-model-hub/sparkproof-mining` and passes the verified snapshot to the release
  gate (alternative to manual `--registry-snapshot`).
- **`docs/MINER_GUIDE.md`** — dataset-track workflow including registry dedupe prevention.

### Fixed

- **Architecture-scoped exact dedupe** ([#26]): exact prompt/code fingerprints are keyed as
  `{gpu_architecture}:{hash}`. The same prompt on Blackwell vs Hopper is a fresh row, not
  an exact duplicate. Near-semantic dedupe was already architecture-aware via
  `semantic_task_fingerprint`. Pairs with SparkDistill [#133](https://github.com/gittensor-model-hub/SparkDistill/pull/133)
  and the republished `sparkproof-mining` canonical mix (174→178 rows).

---

## [Unreleased] — online trust anchors (PR #17, shipped in v0.1.0)

Closes the remaining gap where a miner could fabricate `gpu_attestation.json` or replay an attestation token from another bundle. Offline verification proves internal consistency; online verification anchors the bundle to NVIDIA's and (optionally) OpenRouter's external roots of trust.

### Added

- **`sparkproof/gpu/token_verify.py`** — cryptographic NRAS JWT verification.
  - `extract_detached_gpu_jwt()` pulls the real NVIDIA-signed per-GPU claims JWT out of the SDK's composite token (the outer `client.get_token()` wrapper is *not* the hardware evidence).
  - `verify_nras_token()` validates signature against NVIDIA's published JWKS (`https://nras.attestation.nvidia.com/.well-known/jwks.json`), checks issuer (`nras.attestation.nvidia.com`), measurement result (`measres` / `x-nvidia-overall-att-result`), and optional dataset-bound `eat_nonce`.
  - Expired tokens (`exp` in the past) still verify by signature — NRAS tokens are short-lived but validators re-check bundles days later; signature validity is the trust anchor, not freshness.

- **`sparkproof/verify_online.py`** — orchestrates online trust-anchor checks.
  - `verify_attestation_signature()` re-derives the expected nonce from `manifest.prompts_sha256` + `trajectories_raw.jsonl` and verifies the stored token.
  - `verify_openrouter_generations()` re-queries `https://openrouter.ai/api/v1/generation` for recorded generation IDs and cross-checks the routed model against the bundle record. Requires the API key that created the generation (OpenRouter scopes the endpoint per key) — miner self-audit or validator key escrow.
  - `verify_bundle_online()` runs all available online checks and returns `{verified, issues, nras_signature_checked, openrouter_ledger_checked}`.

- **`sparkproof-verify --online`** — CLI flag to run online checks after offline `verify_bundle`. Sets `report["online"]` and fails the bundle if any online issue is found. When `OPENROUTER_API_KEY` is set, also runs the OpenRouter ledger cross-check.

- **`tests/test_token_verify.py`** — 10 unit tests with a local EC keypair (no network): valid signature, expired-but-signed token, forged signature, wrong nonce, wrong issuer, failed measurement, OpenRouter ledger match/mismatch, missing generation ID.

### Changed

- **`pyproject.toml`** — moved `pyjwt>=2.0.0` from the optional `gpu` extra to **core dependencies**, so validators without a GPU or `nv-attestation-sdk` can still verify NRAS signatures.

### Security

| Threat | Offline verify (PR #16) | Online verify (PR #17) |
|---|---|---|
| Fabricated `gpu_attestation.json` | Catches missing/malformed token, nonce mismatch vs manifest | **Catches invalid NRAS signature** — only NVIDIA can sign |
| Token stolen from another bundle | Nonce must match this bundle's content hash | **Signed `eat_nonce` must match** re-derived nonce |
| Post-prove trajectory swap | Raw/verified consistency + release-gate sha256 | Unchanged (already blocked) |
| Wrong teacher model (yunwu) | Pinned slug validation at generation + verify | Unchanged |
| Wrong teacher model (openrouter) | Manifest policy + request hashes | **Optional ledger re-query** confirms routed model |

### Verified — RTX PRO 6000 Blackwell CC VM (2026-07-11)

Environment: `ubuntu@157.254.50.65:20004`, NVIDIA RTX PRO 6000 Blackwell Server Edition (compute 12.0), confidential computing **ON**.

| Suite | Result |
|---|---|
| SparkProof unit tests | **247 passed**, 9 skipped (GPU off) |
| SparkProof GPU tests (`SPARKPROOF_RUN_GPU_TESTS=1`) | **9 passed** on real Blackwell hardware |
| `tests/test_token_verify.py` | **10 passed** |
| SparkDistill validator tests | **37 passed** |

**Live NRAS + JWKS:**
- Content-bound nonce attestation → `attest passed: True`, `nonce_verified: True`
- `verify_nras_token()` against NVIDIA's live JWKS → `verified: True`, `issues: []`
- Wrong expected nonce → rejected ("not produced for this bundle's content")
- Forged claims + original signature → rejected ("NRAS JWT signature is INVALID")

**Full pipeline e2e:**
```
build-prompts (4 prompts)
  → triton-generate (--limit 2, --benchmark, --strict-validate, --capture-ir, yunwu)
  → proved 2/2 with GPU attestation (nonce verified)
  → release gate passed (2 rows, 0 blocked)
  → sparkproof-verify --online → VERIFIED
  → SparkDistill eval.dataset_verify → verified=true
```

Teachers: `claude-fable-5` (anthropic), `gpt-5.6` (openai, from `gpt-5.6-sol` gateway slug).

**Tamper test:** append text to a trajectory row after proving → `dataset:REJECT` ("rows changed after release gate").

---

## [0.3.0] — trustless production verification (PR #16, 2026-07-11)

Production bundles must pass stricter checks than dev bundles. Validators can reject miner-side tampering without trusting the miner's word.

### Added

- **`verify_pinned_generator()`** — `manifest.generator_version` must equal the validator's `GENERATOR_VERSION` (`0.3.0`). Bundles from older or forked SparkProof builds are rejected.

- **`verify_production_artifacts()`** — requires the full proof directory:
  - `manifest.json`
  - `prompts.jsonl`
  - `trajectories.jsonl`
  - `trajectories_raw.jsonl`
  - `validation_report.jsonl`
  - `gpu_attestation.json`

- **`verify_raw_to_verified_consistency()`** — ensures `trajectories.jsonl` (verified rows) is an unmodified subset of `trajectories_raw.jsonl` (pre-prove archive):
  - `validation_report.jsonl` row count must match raw archive
  - Every passing raw index must have a matching verified fingerprint `(prompt, response, provider, model, request_sha256, gateway, gateway_model)`
  - No verified row may exist that doesn't correspond to a passing raw row — blocks injection or alteration after validation

- **`sparkproof-verify --dev`** — skips production integrity checks (pinned generator, artifact set, raw/verified consistency) for local development.

- **`production` parameter on `verify_bundle()`** — defaults to `require_gpu_attestation`; when `True`, runs all production checks.

### Changed

- **Yunwu gateway model pinning** — production teachers are now strictly:
  - Anthropic: `claude-fable-5` (was `claude-sonnet-5`)
  - OpenAI: `gpt-5.6-sol` (was `gpt-5-mini`)
  - `YUNWU_PINNED_SLUGS` enforced at env load — misconfigured `YUNWU_MODEL_*` vars raise `ValueError` before any generation starts
  - `YUNWU_ACCEPTED_RESPONSE_SLUGS` allows `gpt-5.6` as a response alias when gateway echoes without `-sol` suffix
  - `normalize_upstream_model()` and `validate_gateway_trajectory()` reject non-pinned yunwu response slugs

- **`sparkproof/cli/yunwu_probe.py`** — `--auto` now requires pinned production slugs (`claude-fable-5`, `gpt-5.6-sol`) to pass smoke tests before writing `.env`.

- **`.env.example`** — defaults updated to `claude-fable-5` / `gpt-5.6-sol`.

- **`README.md`** — gateway table updated to reflect pinned yunwu slugs.

### Tests

- `tests/test_yunwu_gateway.py` — added `test_yunwu_rejects_non_pinned_gateway_slug`.

---

## [0.3.0] — release gate & sampling (PR #15, 2026-07-11)

### Fixed

- **Release gate secret scan false positives** — bare `"sk-"` substring matched benign hyphenated words like `mask-based`. Replaced with shape-aware regex `\bsk-[A-Za-z0-9_\-]{16,}` plus targeted patterns for home paths and env key names.

- **`held_out` split leakage** — trajectories with `split: held_out` were not blocked from publish. Now treated as a reserved split alongside `test` and `eval`.

### Changed

- **Stratified sampling documentation** — `max_share` cap applies per **source**, not per `(source, family)` bucket. Docstrings in `stratified_sampling.py` and `--max-bucket-share` help in `build_prompts.py` corrected to match code.

### Tests

- `test_release_gate_secret_scan_ignores_benign_hyphenated_words`
- `test_release_gate_secret_scan_still_catches_real_api_keys_and_paths`
- `test_release_gate_blocks_held_out_split`

---

## [0.3.0] — benchmark integrity & anti-cheat (PR #14, #12)

### Added

- **Reference-vs-kernel speedup metric** — `KernelBench fast_p` timing for candidate ranking; self-reported `do_bench` timings no longer influence winner selection (closes #13).

### Changed

- **Anti-cheat static checks expanded** — AST inspection now covers:
  - Operation coverage (required Triton ops present in kernel body)
  - PyTorch fallback bypass patterns
  - Timing manipulation (CUDA stream injection, clock patching)
  - Correct `kernel[grid](...)` launch syntax

### Fixed

- Candidate ranking could be gamed by inflated self-reported benchmark numbers.

---

## [0.3.0] — dataset flywheel (PR #10, #11)

### Added

- **Identity-free diverse sampling** — `run_seed` entropy scopes prompt selection; stratified round-robin across sources with per-source `max_share` cap; sampling provenance recorded in `prompts.sampling.json`.

- **Novelty gate** — fingerprint-based duplicate detection within a bundle; wired into the release gate with `novelty_report.json` artifact.

### Changed

- Release gate now requires `novelty_report.json` and blocks rows that fail novelty or decontamination checks.

---

## [0.3.0] — HF publish & attestation binding (PR #6–#8)

### Added

- **HF proof artifact upload** — `sparkproof-publish-dataset` uploads bundle proof artifacts (`manifest.json`, `gpu_attestation.json`, `trajectories*.jsonl`, etc.) alongside dataset rows.

- **Content-bound GPU attestation** — NRAS `eat_nonce` derived from `sha256(prompts_sha256 + trajectories_raw_sha256)`; binds attestation to this specific bundle's content (fixes #5).

### Fixed

- Proof artifacts are uploaded **before** dataset rows go public on Hugging Face, so the first snapshot already contains verifiable proof.

---

## [0.3.0] — validation pipeline hardening (PR #2–#4)

### Fixed

- Adversarial validation runs each seed in an isolated subprocess (not shared interpreter state).
- `strip_seed_overrides` → `rewrite_seed_overrides` preserves valid Python syntax when rewriting `torch.manual_seed` calls.
- `prove_blackwell_bundle` respects explicit `--strict-validate` / `--capture-ir` flags (no silent downgrade).
- IR capture uses Triton's `TRITON_DUMP_DIR` instead of fragile inline execution.
- Benchmark timing wraps `triton.testing.do_bench` to capture kernel-only timings.
- DPO export backfills prompt context; empty-prompt pairs rejected.
- Ancestry splitting quarantines existing `eval`/`held_out` splits in merged components.
- Checkpoint prompt backfill and timeout misclassification from flywheel review.

### Added

- Test coverage for self-evolution and failure-mining at parity with mutation source.

---

## Validator integration (SparkDistill)

These SparkProof changes are consumed by SparkDistill's dataset track:

| SparkDistill module | What it checks |
|---|---|
| `eval/dataset_verify.py` | Required proof artifacts, release gate pass, sha256 consistency, re-runs `sparkproof-verify` in production mode |
| `eval/registry_gate.py` | Schema validation, duplicate detection, end-to-end HF download + `dataset_verify` for registry PRs |
| `.github/workflows/dataset_registry.yml` | Auto-verify and merge registry PRs on pass |

Production verification command chain:

```bash
# Miner (Blackwell CC VM)
sparkproof-triton-generate ... --strict-validate --benchmark
sparkproof-verify --bundle bundles/run-001 --online

# Validator (no GPU required for signature check)
sparkproof-verify --bundle proof/ --online
python -m eval.dataset_verify --hf-repo user/sparkproof-triton-v0 --sparkproof-root ../SparkProof
```

---

## Upgrade notes

### Yunwu users

Update `.env` before running generation:

```bash
YUNWU_MODEL_ANTHROPIC=claude-fable-5
YUNWU_MODEL_OPENAI=gpt-5.6-sol
```

Old slugs (`claude-sonnet-5`, `gpt-5-mini`) are rejected at startup.

### Validators

- Install SparkProof with core deps only (`uv sync`) — `pyjwt` is now required, not optional.
- Run `sparkproof-verify --online` for full trustless verification including NRAS signature.
- `OPENROUTER_API_KEY` is only needed for OpenRouter ledger cross-check (yunwu bundles skip this).
- Bundles without `trajectories_raw.jsonl` or with `generator_version != 0.3.0` fail production verify.

### GPU test runners

Set `SPARKPROOF_RUN_GPU_TESTS=1` on a Blackwell or Hopper runner. Triton JIT requires `Python.h` — use uv-managed CPython (`uv python install 3.12`) if system `python3-dev` is unavailable.

---

[Unreleased]: https://github.com/gittensor-model-hub/SparkProof/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/gittensor-model-hub/SparkProof/releases/tag/v0.1.3
[0.1.2]: https://github.com/gittensor-model-hub/SparkProof/releases/tag/v0.1.2
[0.1.1]: https://github.com/gittensor-model-hub/SparkProof/releases/tag/v0.1.1
[0.1.0]: https://github.com/gittensor-model-hub/SparkProof/releases/tag/v0.1.0
