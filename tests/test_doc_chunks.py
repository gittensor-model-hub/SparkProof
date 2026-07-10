from pathlib import Path

import pytest

from sparkproof.triton_dataset.build_prompts import iter_all_prompts
from sparkproof.triton_dataset.doc_chunks import (
    CHUNK_API_SYMBOL,
    CHUNK_SEMANTICS,
    find_triton_language_rst,
    load_doc_chunks,
    load_doc_chunks_from_triton_rst,
    parse_triton_language_rst,
    parse_triton_semantics_rst,
    parse_tutorial_py,
    prompt_from_doc_chunk,
    prompt_from_explain_chunk,
    resolve_triton_language_rst,
    triton_docs_cache_path,
)

SAMPLE_RST = """\
triton.language
===============

Programming Model
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   tensor
   program_id

Memory/Pointer Ops
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   store
   make_tensor_descriptor
   make_block_ptr

Linear Algebra Ops
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dot
   dot_scaled
"""

SAMPLE_SEMANTICS = """\
Triton Semantics
================

Type Promotion
--------------

Type Promotion occurs when tensors of different data types are used in an operation.
For binary operations, Triton converts inputs to a common dtype following a hierarchy.

Broadcasting
------------

Broadcasting allows operations on tensors of different shapes by expanding dimensions.
Two dimensions are compatible if they are equal, or if one of them is 1.
"""

SAMPLE_TUTORIAL = '''\
"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.
You will learn about the basic programming model and triton.jit.
"""

# %%
# Compute Kernel
# --------------
# In this section we implement a blocked vector add kernel with tl.load and tl.store.
# Each program handles a contiguous block of elements using tl.program_id and tl.arange.
# Masking is required when the vector length is not divisible by the block size.

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
'''


def test_parse_triton_language_rst_extracts_symbols_and_categories():
    chunks = parse_triton_language_rst(SAMPLE_RST, source="triton.language.rst")
    assert len(chunks) == 8
    by_symbol = {c["symbol"]: c for c in chunks}
    assert by_symbol["dot"]["target_api"] == "tl.dot"
    assert by_symbol["dot"]["doc_category"] == "Linear Algebra Ops"


def test_parse_triton_semantics_rst_extracts_sections():
    chunks = parse_triton_semantics_rst(SAMPLE_SEMANTICS, source="triton-semantics.rst")
    assert len(chunks) == 2
    assert chunks[0]["title"] == "Type Promotion"
    assert chunks[0]["chunk_kind"] == CHUNK_SEMANTICS


def test_parse_tutorial_py_extracts_intro_and_cells():
    chunks = parse_tutorial_py(SAMPLE_TUTORIAL, source="01-vector-add.py")
    assert len(chunks) >= 2
    assert chunks[0]["section_id"].endswith("_intro")
    assert any(c["section_id"].endswith("_sec_1") for c in chunks)


def test_prompt_from_explain_chunk_requests_explain_and_code():
    chunk = {
        "chunk_kind": CHUNK_SEMANTICS,
        "section_id": "sem_type_promotion",
        "title": "Type Promotion",
        "content": "Type Promotion occurs when tensors of different dtypes mix.",
    }
    rec = prompt_from_explain_chunk(chunk)
    assert rec["category"] == "doc_explain_implement"
    assert rec["source"] == "doc_semantics"
    assert "Explain the key rules" in rec["prompt"]
    assert "Type Promotion" in rec["prompt"]


def test_load_doc_chunks_prefers_rst_over_markdown(tmp_path: Path):
    docs = tmp_path / "docs"
    api_dir = docs / "python-api"
    api_dir.mkdir(parents=True)
    (api_dir / "triton.language.rst").write_text(SAMPLE_RST, encoding="utf-8")
    (docs / "notes.md").write_text("## ignored\n" + "x" * 200, encoding="utf-8")

    chunks = load_doc_chunks(docs, auto_fetch=False, kinds=frozenset({CHUNK_API_SYMBOL}))
    assert len(chunks) == 8


def test_find_triton_language_rst_accepts_docs_root_or_python_api(tmp_path: Path):
    docs = tmp_path / "docs"
    api_dir = docs / "python-api"
    api_dir.mkdir(parents=True)
    rst = api_dir / "triton.language.rst"
    rst.write_text(SAMPLE_RST, encoding="utf-8")

    assert find_triton_language_rst(docs) == rst
    assert find_triton_language_rst(api_dir) == rst


def test_prompt_from_doc_chunk_api_without_double_prefix():
    chunk = {
        "chunk_kind": CHUNK_API_SYMBOL,
        "section_id": "api_tl_dot",
        "content": "Category: Linear Algebra Ops\nAPI: tl.dot",
        "target_api": "tl.dot",
    }
    prompt = prompt_from_doc_chunk(chunk)
    assert prompt["task_id"] == "api_tl_dot"
    assert prompt["target_api"] == "tl.dot"


def test_iter_all_prompts_doc_sources(tmp_path: Path):
    docs = tmp_path / "docs"
    api_dir = docs / "python-api"
    api_dir.mkdir(parents=True)
    (api_dir / "triton.language.rst").write_text(SAMPLE_RST, encoding="utf-8")
    (api_dir / "triton-semantics.rst").write_text(SAMPLE_SEMANTICS, encoding="utf-8")
    tutorials = tmp_path / "python" / "tutorials"
    tutorials.mkdir(parents=True)
    (tutorials / "01-vector-add.py").write_text(SAMPLE_TUTORIAL, encoding="utf-8")

    records = list(
        iter_all_prompts(
            doc_dir=docs,
            include_sources=frozenset({"api_doc", "doc_semantics", "doc_tutorial"}),
            auto_fetch_docs=False,
        )
    )
    sources = {r["source"] for r in records}
    assert "api_doc" in sources
    assert "doc_semantics" in sources
    assert "doc_tutorial" in sources
    assert any(r["category"] == "doc_explain_implement" for r in records)


def test_load_doc_chunks_auto_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "v3.7.1"
    cache.mkdir(parents=True)
    (cache / "triton.language.rst").write_text(SAMPLE_RST, encoding="utf-8")
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_CACHE", str(tmp_path))
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_REF", "v3.7.1")

    chunks = load_doc_chunks(None, kinds=frozenset({CHUNK_API_SYMBOL}))
    assert len(chunks) == 8


def test_offline_without_cache_falls_back_to_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_OFFLINE", "1")
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_CACHE", "/nonexistent/sparkproof-cache")

    records = list(iter_all_prompts(include_sources=frozenset({"api_doc"}), auto_fetch_docs=True))
    assert len(records) == 5


def test_resolve_prefers_explicit_doc_dir_over_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs = tmp_path / "docs"
    api_dir = docs / "python-api"
    api_dir.mkdir(parents=True)
    local_rst = api_dir / "triton.language.rst"
    local_rst.write_text(SAMPLE_RST, encoding="utf-8")

    cache = tmp_path / "cache" / "v3.7.1"
    cache.mkdir(parents=True)
    (cache / "triton.language.rst").write_text("triton.language\n===============\n", encoding="utf-8")
    monkeypatch.setenv("SPARKPROOF_TRITON_DOCS_CACHE", str(tmp_path / "cache"))

    assert resolve_triton_language_rst(doc_dir=docs, fetch=False) == local_rst
    assert triton_docs_cache_path("triton.language.rst", ref="v3.7.1") == cache / "triton.language.rst"


def test_load_doc_chunks_from_triton_rst_roundtrip(tmp_path: Path):
    rst = tmp_path / "triton.language.rst"
    rst.write_text(SAMPLE_RST, encoding="utf-8")
    chunks = load_doc_chunks_from_triton_rst(rst)
    assert chunks[0]["symbol"] == "tensor"
