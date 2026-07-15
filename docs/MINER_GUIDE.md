# SparkProof Miner Guide

This guide covers the **dataset track** workflow: generate verified Triton training data on
a Blackwell or Hopper H100/H200 CC VM, publish to Hugging Face, and open a registry PR in
[SparkDistill](https://github.com/gittensor-model-hub/SparkDistill).

## Prerequisites

- **GPU:** NVIDIA RTX PRO 6000 Blackwell **or** Hopper H100/H200 on an Intel TDX CC guest
  (e.g. [Targon](https://targon.com/) SN4)
- **Teachers:** OpenRouter or yunwu with pinned **Fable 5** + **GPT 5.6 Sol** at `xhigh`
- **Sibling repo:** clone SparkDistill beside SparkProof for training recipes (optional for
  dataset-only miners)

```bash
git clone https://github.com/gittensor-model-hub/SparkProof.git SparkProof
git clone https://github.com/gittensor-model-hub/SparkDistill.git SparkDistill
cd SparkProof && cp .env.example .env   # OPENROUTER_API_KEY or YUNWU_API_KEY
scripts/install.sh
```

On TDX guests, provision configfs-tsm once per boot (see README) before `sparkproof-prove`.

## End-to-end pipeline

```bash
# 1. Generate + prove on CC VM
scripts/run_triton_pipeline.sh --run-id my-run-001 --release-gate

# 2. Publish to your HF dataset repo (with registry dedupe — see below)
uv sync --extra publish --frozen
sparkproof-publish-dataset --bundle bundles/my-run-001 --repo-id YOU/sparkproof-triton-v1 \
  --release-gate --mining-repo

# 3. Open a text-only SparkDistill registry PR (HF URL + trajectories_sha256)
```

`--release-gate` runs decontamination, novelty accounting, and production verify before upload.

## Avoid registry dedupe surprises

SparkDistill merges every accepted dataset into one canonical mining mix
(`gittensor-model-hub/sparkproof-mining`) with **`exact` dedupe** — identical prompts
drop at mix time. **Fair reward labels** count `rows_selected` after dedupe, not your raw
bundle row count. A 159-row bundle that only adds 25 novel mix rows earns `dataset:xs`, not
`dataset:xl`.

After each registry merge, SparkDistill publishes:

| HF artifact | Purpose |
|---|---|
| `accepted_registry_snapshot.jsonl` | Full trajectory rows occupying the accepted mix state |
| `accepted_task_ids.json` | Lightweight task-id index for pre-generation filtering |
| `mix_manifest.json` | Pins `accepted_registry_snapshot_sha256` + row count |

**Check novelty before you burn GPU:** pass that snapshot to SparkProof's release gate so
`novelty_report.json` includes **cross-registry** duplicates (not just within your bundle).

### Option A — download snapshot, then publish

```bash
uv sync --extra publish --frozen
scripts/download_registry_snapshot.sh --out-dir ./snapshots

# Confirm pins match live mix_manifest (optional)
sparkproof-download-registry-snapshot --verify-only ./snapshots/accepted_registry_snapshot.jsonl

sparkproof-publish-dataset --bundle bundles/my-run-001 --repo-id YOU/sparkproof-triton-v1 \
  --release-gate \
  --registry-snapshot ./snapshots/accepted_registry_snapshot.jsonl
```

### Option B — one-step publish (recommended)

```bash
uv sync --extra publish --frozen
sparkproof-publish-dataset --bundle bundles/my-run-001 --repo-id YOU/sparkproof-triton-v1 \
  --release-gate --mining-repo
```

`--mining-repo` (bare flag) downloads from `gittensor-model-hub/sparkproof-mining`, verifies
the `mix_manifest` sha256 pin, and passes the snapshot to the release gate automatically.

### Reading `novelty_report.json`

After release gate / publish:

```json
{
  "verified_rows": 50,
  "exact_duplicate_rows": 25,
  "near_duplicate_rows": 0,
  "novel_verified_rows": 25,
  "duplicate_task_ids": ["..."]
}
```

Target **`novel_verified_rows` ≥ 25** (`dataset:xs` threshold) before opening a registry PR.
SparkDistill labels from canonical-mix contribution, but this preview avoids wasted GPU on
rows that will not earn credit.

**Architecture-aware dedupe:** the same prompt on **Blackwell vs Hopper** counts as a
**fresh** row, not a duplicate — exact dedupe keys include `gpu_architecture`.

**Repair-tier novelty:** post-generation dedupe and `novelty_report.json` fingerprint
`metadata.prompt_meta.prompt` (the mining task), not the shared self-repair wrapper stored
in top-level `prompt`. Pre-generation filters already use task prompts — counts should
match after a successful run.

### Pre-generation filtering (optional)

Download `accepted_task_ids.json` from the mining HF repo and filter prompts before
generation so you do not retry tasks already in the accepted registry.

## Production checklist

| Step | Command / artifact |
|---|---|
| GPU prove + attest | `sparkproof-prove` / `scripts/run_triton_pipeline.sh` |
| Offline verify | `sparkproof-verify --bundle bundles/...` |
| Online verify (NRAS + TDX) | `sparkproof-verify --bundle bundles/... --online` |
| Registry dedupe preview | `--mining-repo` or `--registry-snapshot` on publish |
| HF publish | `sparkproof-publish-dataset` |
| SparkDistill registry PR | append line to `datasets/registry.jsonl` |

## Further reading

- SparkDistill dataset track: [`SparkDistill/datasets/README.md`](https://github.com/gittensor-model-hub/SparkDistill/blob/main/datasets/README.md)
- SparkDistill miner guide (rewards, tiers): [`SparkDistill/docs/miner-guide.md`](https://github.com/gittensor-model-hub/SparkDistill/blob/main/docs/miner-guide.md)
- Doc-chunk prompt sources: [`docs/DOC_CHUNK_PROMPTS.md`](DOC_CHUNK_PROMPTS.md)
