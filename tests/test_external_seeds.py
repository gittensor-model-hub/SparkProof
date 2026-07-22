"""External corpus → SparkProof task seeds (not verified trajectories)."""

from __future__ import annotations

import json
from pathlib import Path

from sparkproof.triton_dataset.external_seeds import (
    SEED_ORIGIN,
    build_seed_prompt,
    extract_repair_hint_from_opus_turns,
    import_seed_rows,
    licenses_permissive,
    write_seed_prompts,
)
from sparkproof.triton_dataset.schema import validate_prompt_record

FIXTURES = Path(__file__).parent / "fixtures" / "external_seeds"


def test_licenses_permissive():
    assert licenses_permissive(["MIT"])
    assert licenses_permissive(["Apache-2.0", "BSD-3-Clause"])
    assert not licenses_permissive(["Proprietary"])
    assert not licenses_permissive([])


def test_repair_hint_from_opus_turns():
    turns = [
        {"attempt": 1, "kernel": "broken_a", "correct": False},
        {"attempt": 2, "kernel": "broken_b", "correct": False},
        {"attempt": 3, "kernel": "good", "correct": True},
    ]
    assert extract_repair_hint_from_opus_turns(turns) == "broken_b"


def test_build_seed_prompt_is_trainable():
    rec = build_seed_prompt(
        pytorch_code="import torch\nclass M(torch.nn.Module):\n    def forward(self, x):\n        return x\n",
        entry_point="M",
        task_id="kb_m_1",
        source_dataset="local",
        licenses=["MIT"],
    )
    validated = validate_prompt_record(rec)
    assert validated["origin"] == SEED_ORIGIN
    assert "torch_reference" in validated
    assert "SPARKPROOF_TRITON_PASS" in validated["prompt"]


def test_import_filters_license_kernelbench_and_dedupes(tmp_path: Path):
    records, stats = import_seed_rows(
        kernelbook=str(FIXTURES / "kernelbook.jsonl"),
        opus_traces=str(FIXTURES / "opus_traces.jsonl"),
        gptoss_traces=str(FIXTURES / "gptoss_traces.jsonl"),
        kernelbench=str(FIXTURES / "kernelbench.jsonl"),
        require_eval_corpus=False,
        require_permissive_license=True,
        include_repair_hints=True,
    )
    # kernelbook: SumAggregator + ReLUTiny (ProprietaryOp dropped)
    # opus: BehlerAngular kept, BenchLeak blocked source
    # gptoss: LayerNormLite kept, kernelbench row blocked
    # ReLUTiny / BenchLeak / gptoss relu overlap KernelBench → decontam drop for relu family
    task_ids = {r["task_id"] for r in records}
    entry_points = {r["entry_point"] for r in records}
    assert "SumAggregator" in entry_points
    assert "BehlerAngular" in entry_points
    assert "LayerNormLite" in entry_points
    assert "ProprietaryOp" not in entry_points
    assert "ReLUTiny" not in entry_points  # KernelBench structure overlap
    assert "BenchLeak" not in entry_points
    assert stats["skipped_license"] >= 1
    assert stats["skipped_blocked_source"] >= 1
    assert stats["skipped_decontam"] >= 1
    assert stats["with_repair_hint"] >= 1
    assert all(r["origin"] == SEED_ORIGIN for r in records)

    out = tmp_path / "seeds.jsonl"
    write_seed_prompts(out, records)
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == len(records)
    assert lines[0]["task_id"] in task_ids


def test_cli_main(tmp_path: Path):
    from sparkproof.cli import import_external_tasks as cli

    out = tmp_path / "out.jsonl"
    rc = cli.main(
        [
            "--out",
            str(out),
            "--kernelbook",
            str(FIXTURES / "kernelbook.jsonl"),
            "--opus-traces",
            str(FIXTURES / "opus_traces.jsonl"),
            "--kernelbench",
            str(FIXTURES / "kernelbench.jsonl"),
            "--no-require-eval-corpus",
            "--gpu-architecture",
            "blackwell",
        ]
    )
    assert rc == 0
    assert out.is_file()
    assert out.with_suffix(".import.json").is_file()
