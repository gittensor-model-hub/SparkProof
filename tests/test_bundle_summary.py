import json
from pathlib import Path

from sparkproof.bundle_summary import format_summary, summarize_bundle


def test_summarize_bundle_with_adjudication_and_validation(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "prompts.jsonl").write_text(
        json.dumps({"task_id": "api_tl_dot", "source": "api_doc", "category": "kernel_write"})
        + "\n"
        + json.dumps({"task_id": "sem_x", "source": "doc_semantics", "category": "doc_explain_implement"})
        + "\n",
        encoding="utf-8",
    )
    (bundle / "adjudication.jsonl").write_text(
        json.dumps({"task_id": "api_tl_dot", "winner_provider": "anthropic"})
        + "\n"
        + json.dumps({"task_id": "sem_x", "winner_provider": None})
        + "\n",
        encoding="utf-8",
    )
    raw = [
        {
            "provider": "anthropic",
            "metadata": {"prompt_meta": {"task_id": "api_tl_dot", "source": "api_doc", "category": "kernel_write"}},
        },
        {
            "provider": "openai",
            "metadata": {"prompt_meta": {"task_id": "sem_x", "source": "doc_semantics", "category": "doc_explain_implement"}},
        },
    ]
    (bundle / "trajectories_raw.jsonl").write_text(
        "\n".join(json.dumps(r) for r in raw) + "\n", encoding="utf-8"
    )
    (bundle / "validation_report.jsonl").write_text(
        json.dumps({"validation": {"passed": True}})
        + "\n"
        + json.dumps({"validation": {"passed": False}})
        + "\n",
        encoding="utf-8",
    )
    (bundle / "trajectories.jsonl").write_text(json.dumps(raw[0]) + "\n", encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({"version": "sparkproof-2", "sample_count": 1}), encoding="utf-8")

    report = summarize_bundle(bundle)
    assert report["teacher"]["teacher_winners"] == 1
    assert report["prove"]["proved_passed"] == 1
    text = format_summary(report)
    assert "api_doc" in text
    assert "doc_semantics" in text
