from pathlib import Path

import pytest

from sparkproof.generate.runner import iter_prompts
from sparkproof.triton_dataset.build_prompts import build_prompts_file
from sparkproof.triton_dataset.doc_api_pages import (
    enrich_api_symbol_chunk,
    html_to_text,
    parse_api_page_rst,
)
from sparkproof.triton_dataset.prompt_filters import prompt_matches_filters

SAMPLE_API_RST = """\
triton.language.dot
=================

.. autofunction:: triton.language.dot

Returns the matrix product of two blocks.

Parameters
----------

input
    The first tensor to be multiplied.

other
    The second tensor to be multiplied.
"""

SAMPLE_HTML = """\
<html><body><h1>triton.language.dot</h1>
<p>Returns the matrix product of two blocks.</p>
<h2>Parameters</h2>
<p>input – first tensor</p>
</body></html>
"""


def test_parse_api_page_rst_strips_directives():
    text = parse_api_page_rst(SAMPLE_API_RST)
    assert "Returns the matrix product" in text
    assert ".. autofunction::" not in text


def test_html_to_text_extracts_content():
    text = html_to_text(SAMPLE_HTML)
    assert "triton.language.dot" in text
    assert "matrix product" in text


def test_enrich_api_symbol_chunk_uses_local_rst(tmp_path: Path):
    generated = tmp_path / "python-api" / "generated"
    generated.mkdir(parents=True)
    (generated / "triton.language.dot.rst").write_text(SAMPLE_API_RST, encoding="utf-8")

    chunk = {
        "chunk_kind": "api_symbol",
        "symbol": "dot",
        "section_id": "api_tl_dot",
        "target_api": "tl.dot",
        "doc_category": "Linear Algebra Ops",
        "content": "short",
    }
    enriched = enrich_api_symbol_chunk(chunk, doc_dir=tmp_path, fetch=False)
    assert enriched.get("api_page_enriched") is True
    assert "matrix product" in enriched["content"]


def test_prompt_matches_filters():
    rec = {"source": "api_doc", "task_id": "api_tl_dot"}
    assert prompt_matches_filters(rec, sources=frozenset({"api_doc"}), task_ids=None)
    assert not prompt_matches_filters(rec, sources=frozenset({"doc_semantics"}), task_ids=None)
    assert prompt_matches_filters(rec, sources=None, task_ids=frozenset({"api_tl_dot"}))


def test_iter_prompts_filters_jsonl(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"task_id":"a","source":"api_doc","origin":"api_doc","split":"train","category":"kernel_write","prompt":"p1"}',
                '{"task_id":"b","source":"doc_semantics","origin":"doc_semantics","split":"train","category":"doc_explain_implement","prompt":"p2"}',
                '{"task_id":"c","source":"api_doc","origin":"api_doc","split":"train","category":"kernel_write","prompt":"p3"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(iter_prompts(path, sources=frozenset({"api_doc"})))
    assert [r["task_id"] for r in rows] == ["a", "c"]

    rows2 = list(iter_prompts(path, task_ids=frozenset({"b"})))
    assert [r["task_id"] for r in rows2] == ["b"]


def test_build_prompts_filter_task_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "v3.7.1"
    cache.mkdir(parents=True)
    (cache / "triton.language.rst").write_text(
        "triton.language\n===============\n\nProgramming Model\n-----------------\n\n"
        ".. autosummary::\n   :toctree: generated\n\n   dot\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_CACHE", str(tmp_path))
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_REF", "v3.7.1")
    monkeypatch.setenv("SPARKPROOF_TRITON_API_PAGES", "0")

    out = tmp_path / "out.jsonl"
    build_prompts_file(
        out,
        sources=frozenset({"api_doc"}),
        filter_task_ids=frozenset({"api_tl_dot"}),
        enrich_api_pages=False,
    )
    import json

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["task_id"] == "api_tl_dot"
