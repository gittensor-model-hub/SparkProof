"""Source A: API-unit doc chunks (Triton 3.7.1 feature registry)."""

from __future__ import annotations

from pathlib import Path
import re


def _chunk_markdown_by_header(content: str, source: str) -> list[dict]:
    sections = re.split(r"\n(?=##\s|###\s)", content)
    chunks: list[dict] = []
    for idx, sec in enumerate(sections):
        text = sec.strip()
        if len(text) < 100:
            continue
        chunks.append({"source": source, "section_id": f"{Path(source).stem}_sec_{idx}", "content": text})
    return chunks


def load_doc_chunks(doc_dir: Path | None) -> list[dict]:
    if doc_dir is None or not doc_dir.exists():
        return []
    chunks: list[dict] = []
    for path in doc_dir.glob("**/*.md"):
        chunks.extend(_chunk_markdown_by_header(path.read_text(), str(path.name)))
    return chunks


def api_unit_chunks_from_registry() -> list[dict]:
    """Fallback when no doc tree is present — one chunk per pinned Triton 3.7.1 API unit."""
    units = [
        ("tl.make_tensor_descriptor", "Tensor descriptors: desc.load/store offsets (replaces block_ptr in 3.7)."),
        ("tl.dot", "tl.dot with fp32 acc and input_precision tf32/ieee on Blackwell."),
        ("tl.associative_scan", "tl.associative_scan / tl.reduce / tl.cumsum patterns."),
        ("@triton.autotune", "@triton.autotune configs with num_warps, num_stages, BLOCK_* keys."),
        ("fp8_dtypes", "tl.float8e4nv / tl.float8e5m2 and fp4 tl.float4e2m1 where applicable."),
    ]
    return [
        {
            "source": "triton_371_registry",
            "section_id": f"api_{name.replace('.', '_')}",
            "content": f"API unit: {name}\n{desc}",
            "target_api": name,
        }
        for name, desc in units
    ]


def prompt_from_api_chunk(chunk: dict) -> dict:
    target = chunk.get("target_api", "tl.dot")
    return {
        "task_id": f"api_{chunk['section_id']}",
        "source": "api_doc",
        "category": "kernel_write",
        "target_api": target,
        "prompt": (
            f"Write a complete Triton 3.7.1 kernel on Blackwell SM12x that **must use** `{target}`.\n"
            f"Define exact input shapes/dtypes, include @triton.jit, launcher, masks, and torch.allclose test.\n\n"
            f"Context:\n{chunk['content'][:1500]}"
        ),
        "doc_chunk_id": chunk["section_id"],
    }
