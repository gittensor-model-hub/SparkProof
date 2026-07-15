"""Source A: Triton doc chunks — API symbols, semantics prose, and tutorials."""

from __future__ import annotations

import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from sparkproof.gpu.architecture import ARCH_BLACKWELL, fp4_supported, sm_label
from sparkproof.policy import TRITON_VERSION

_SYMBOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RST_UNDERLINE_RE = re.compile(r"^[-=]{3,}$")
_TUTORIAL_CELL_RE = re.compile(r"^# %%\s*$", re.MULTILINE)

DEFAULT_TRITON_DOCS_REF = f"v{TRITON_VERSION}"
TRITON_LANGUAGE_RST_PATH = Path("docs/python-api/triton.language.rst")
TRITON_SEMANTICS_RST_PATH = Path("docs/python-api/triton-semantics.rst")
TUTORIAL_FILES = (
    "01-vector-add.py",
    "02-fused-softmax.py",
    "03-matrix-multiplication.py",
    "04-low-memory-dropout.py",
    "05-layer-norm.py",
    "06-fused-attention.py",
)

CHUNK_API_SYMBOL = "api_symbol"
CHUNK_SEMANTICS = "semantics"
CHUNK_TUTORIAL = "tutorial"


def triton_docs_cache_dir() -> Path:
    override = os.environ.get("SPARKPROOF_TRITON_DOCS_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "sparkproof" / "triton-docs"
    return Path.home() / ".cache" / "sparkproof" / "triton-docs"


def _docs_ref() -> str:
    return os.environ.get("SPARKPROOF_TRITON_DOCS_REF", DEFAULT_TRITON_DOCS_REF)


def triton_docs_cache_path(name: str, *, ref: str | None = None) -> Path:
    ref = ref or _docs_ref()
    return triton_docs_cache_dir() / ref / name


def triton_language_rst_url(ref: str | None = None) -> str:
    ref = ref or _docs_ref()
    return (
        f"https://raw.githubusercontent.com/triton-lang/triton/{ref}/"
        f"{TRITON_LANGUAGE_RST_PATH.as_posix()}"
    )


def triton_semantics_rst_url(ref: str | None = None) -> str:
    ref = ref or _docs_ref()
    return (
        f"https://raw.githubusercontent.com/triton-lang/triton/{ref}/"
        f"{TRITON_SEMANTICS_RST_PATH.as_posix()}"
    )


def triton_tutorial_url(filename: str, ref: str | None = None) -> str:
    ref = ref or _docs_ref()
    return f"https://raw.githubusercontent.com/triton-lang/triton/{ref}/python/tutorials/{filename}"


def _is_offline() -> bool:
    return os.environ.get("SPARKPROOF_TRITON_DOCS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:64] or "section"


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _write_cache(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def fetch_triton_language_rst(cache_path: Path, *, ref: str | None = None) -> Path:
    content = _fetch_text(triton_language_rst_url(ref))
    if "triton.language" not in content or ".. autosummary::" not in content:
        raise ValueError("downloaded file does not look like triton.language.rst")
    return _write_cache(cache_path, content)


def fetch_triton_semantics_rst(cache_path: Path, *, ref: str | None = None) -> Path:
    content = _fetch_text(triton_semantics_rst_url(ref))
    if "Triton Semantics" not in content:
        raise ValueError("downloaded file does not look like triton-semantics.rst")
    return _write_cache(cache_path, content)


def fetch_triton_tutorial(cache_path: Path, filename: str, *, ref: str | None = None) -> Path:
    content = _fetch_text(triton_tutorial_url(filename, ref))
    if "@triton.jit" not in content:
        raise ValueError(f"downloaded file does not look like a Triton tutorial: {filename}")
    return _write_cache(cache_path, content)


def _chunk_markdown_by_header(content: str, source: str) -> list[dict]:
    sections = re.split(r"\n(?=##\s|###\s)", content)
    chunks: list[dict] = []
    for idx, sec in enumerate(sections):
        text = sec.strip()
        if len(text) < 100:
            continue
        chunks.append(
            {
                "chunk_kind": CHUNK_SEMANTICS,
                "source": source,
                "section_id": f"md_{Path(source).stem}_sec_{idx}",
                "title": text.splitlines()[0][:120],
                "content": text,
            }
        )
    return chunks


def find_triton_language_rst(doc_dir: Path) -> Path | None:
    for candidate in (
        doc_dir / "python-api" / "triton.language.rst",
        doc_dir / "triton.language.rst",
    ):
        if candidate.is_file():
            return candidate
    return None


def find_triton_semantics_rst(doc_dir: Path) -> Path | None:
    for candidate in (
        doc_dir / "python-api" / "triton-semantics.rst",
        doc_dir / "triton-semantics.rst",
    ):
        if candidate.is_file():
            return candidate
    return None


def find_tutorial_files(doc_dir: Path) -> list[Path]:
    tutorials_dir = doc_dir / ".." / "python" / "tutorials"
    tutorials_dir = tutorials_dir.resolve()
    if not tutorials_dir.is_dir():
        tutorials_dir = doc_dir / "tutorials"
    if not tutorials_dir.is_dir():
        return []
    return sorted(p for p in tutorials_dir.glob("*.py") if p.name[0].isdigit())


def parse_triton_language_rst(content: str, *, source: str) -> list[dict]:
    lines = content.splitlines()
    chunks: list[dict] = []
    seen: set[str] = set()
    category = "triton.language"
    i = 0

    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _RST_UNDERLINE_RE.match(lines[i + 1]) and line.strip():
            category = line.strip()
            i += 2
            continue

        if line.strip() == ".. autosummary::":
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
                stripped = lines[i].strip()
                if not stripped:
                    i += 1
                    continue
                if stripped.startswith(":"):
                    i += 1
                    continue
                if _SYMBOL_RE.match(stripped) and stripped not in seen:
                    seen.add(stripped)
                    target_api = f"tl.{stripped}"
                    chunks.append(
                        {
                            "chunk_kind": CHUNK_API_SYMBOL,
                            "source": source,
                            "section_id": f"api_tl_{stripped}",
                            "title": stripped,
                            "content": (
                                f"Category: {category}\n"
                                f"API: {target_api}\n"
                                f"Symbol: {stripped} (triton.language)"
                            ),
                            "target_api": target_api,
                            "doc_category": category,
                            "symbol": stripped,
                        }
                    )
                i += 1
            continue

        i += 1

    return chunks


def parse_triton_semantics_rst(content: str, *, source: str) -> list[dict]:
    lines = content.splitlines()
    chunks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _RST_UNDERLINE_RE.match(lines[i + 1]) and line.strip():
            title = line.strip()
            i += 2
            body: list[str] = []
            while i < len(lines):
                if i + 1 < len(lines) and _RST_UNDERLINE_RE.match(lines[i + 1]) and lines[i].strip():
                    break
                body.append(lines[i])
                i += 1
            text = "\n".join(body).strip()
            if len(text) < 80:
                continue
            section_id = f"sem_{_slug(title)}"
            chunks.append(
                {
                    "chunk_kind": CHUNK_SEMANTICS,
                    "source": source,
                    "section_id": section_id,
                    "title": title,
                    "content": f"{title}\n\n{text}",
                }
            )
            continue
        i += 1
    return chunks


def _comment_prose_from_cell(cell: str) -> tuple[str, str]:
    title = ""
    prose: list[str] = []
    for line in cell.splitlines():
        if line.startswith("# ") and not line.startswith("# %%"):
            prose.append(line[2:].strip())
        elif line.strip() and not line.startswith("#"):
            break
    text = "\n".join(prose).strip()
    if prose:
        title = prose[0].strip("=- ").strip() or title
    return title, text


def parse_tutorial_py(content: str, *, source: str) -> list[dict]:
    chunks: list[dict] = []
    stem = Path(source).stem

    docstring = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if docstring:
        intro = docstring.group(1).strip()
        if len(intro) >= 80:
            title = intro.splitlines()[0].strip("=- ").strip() or stem
            chunks.append(
                {
                    "chunk_kind": CHUNK_TUTORIAL,
                    "source": source,
                    "section_id": f"tut_{_slug(stem)}_intro",
                    "title": title,
                    "content": intro,
                    "tutorial_file": source,
                }
            )

    cells = _TUTORIAL_CELL_RE.split(content)
    for idx, cell in enumerate(cells[1:], start=1):
        title, prose = _comment_prose_from_cell(cell)
        if len(prose) < 80:
            continue
        section_title = title or f"{stem} section {idx}"
        chunks.append(
            {
                "chunk_kind": CHUNK_TUTORIAL,
                "source": source,
                "section_id": f"tut_{_slug(stem)}_sec_{idx}",
                "title": section_title,
                "content": prose,
                "tutorial_file": source,
            }
        )
    return chunks


def resolve_triton_language_rst(*, doc_dir: Path | None = None, fetch: bool = True) -> Path | None:
    if doc_dir and doc_dir.exists():
        found = find_triton_language_rst(doc_dir)
        if found is not None:
            return found

    env_dir = os.environ.get("SPARKPROOF_TRITON_DOCS_DIR")
    if env_dir:
        found = find_triton_language_rst(Path(env_dir))
        if found is not None:
            return found

    cached = triton_docs_cache_path("triton.language.rst")
    if cached.is_file():
        return cached

    if fetch and not _is_offline():
        try:
            return fetch_triton_language_rst(cached)
        except (OSError, urllib.error.URLError, ValueError):
            return None
    return None


def resolve_triton_semantics_rst(*, doc_dir: Path | None = None, fetch: bool = True) -> Path | None:
    if doc_dir and doc_dir.exists():
        found = find_triton_semantics_rst(doc_dir)
        if found is not None:
            return found

    env_dir = os.environ.get("SPARKPROOF_TRITON_DOCS_DIR")
    if env_dir:
        found = find_triton_semantics_rst(Path(env_dir))
        if found is not None:
            return found

    cached = triton_docs_cache_path("triton-semantics.rst")
    if cached.is_file():
        return cached

    if fetch and not _is_offline():
        try:
            return fetch_triton_semantics_rst(cached)
        except (OSError, urllib.error.URLError, ValueError):
            return None
    return None


def resolve_tutorial_files(*, doc_dir: Path | None = None, fetch: bool = True) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []

    if doc_dir and doc_dir.exists():
        for path in find_tutorial_files(doc_dir):
            resolved.append((path.name, path))

    if resolved:
        return resolved

    env_dir = os.environ.get("SPARKPROOF_TRITON_DOCS_DIR")
    if env_dir:
        for path in find_tutorial_files(Path(env_dir)):
            resolved.append((path.name, path))
    if resolved:
        return resolved

    ref = _docs_ref()
    cache_tutorials = triton_docs_cache_dir() / ref / "tutorials"
    for filename in TUTORIAL_FILES:
        cached = cache_tutorials / filename
        if cached.is_file():
            resolved.append((filename, cached))
            continue
        if fetch and not _is_offline():
            try:
                fetch_triton_tutorial(cached, filename, ref=ref)
                resolved.append((filename, cached))
            except (OSError, urllib.error.URLError, ValueError):
                continue
    return resolved


def load_all_doc_chunks(
    doc_dir: Path | None = None,
    *,
    auto_fetch: bool = True,
    kinds: frozenset[str] | None = None,
    enrich_api_pages: bool | None = None,
) -> list[dict]:
    allowed = kinds or frozenset({CHUNK_API_SYMBOL, CHUNK_SEMANTICS, CHUNK_TUTORIAL})
    chunks: list[dict] = []

    if CHUNK_API_SYMBOL in allowed:
        rst_path = resolve_triton_language_rst(doc_dir=doc_dir, fetch=auto_fetch)
        if rst_path is not None:
            api_chunks = parse_triton_language_rst(
                rst_path.read_text(encoding="utf-8"), source=rst_path.name
            )
            from sparkproof.triton_dataset.doc_api_pages import enrich_api_symbol_chunks

            chunks.extend(
                enrich_api_symbol_chunks(
                    api_chunks,
                    doc_dir=doc_dir,
                    fetch=auto_fetch,
                    enabled=enrich_api_pages,
                )
            )

    if CHUNK_SEMANTICS in allowed:
        sem_path = resolve_triton_semantics_rst(doc_dir=doc_dir, fetch=auto_fetch)
        if sem_path is not None:
            chunks.extend(
                parse_triton_semantics_rst(sem_path.read_text(encoding="utf-8"), source=sem_path.name)
            )

    if CHUNK_TUTORIAL in allowed:
        for filename, path in resolve_tutorial_files(doc_dir=doc_dir, fetch=auto_fetch):
            chunks.extend(parse_tutorial_py(path.read_text(encoding="utf-8"), source=filename))

    if not chunks and doc_dir and doc_dir.exists():
        for path in doc_dir.glob("**/*.md"):
            chunks.extend(_chunk_markdown_by_header(path.read_text(encoding="utf-8"), str(path.name)))

    return chunks


def load_doc_chunks(
    doc_dir: Path | None = None,
    *,
    auto_fetch: bool = True,
    kinds: frozenset[str] | None = None,
    enrich_api_pages: bool | None = None,
) -> list[dict]:
    return load_all_doc_chunks(
        doc_dir,
        auto_fetch=auto_fetch,
        kinds=kinds,
        enrich_api_pages=enrich_api_pages,
    )


def load_doc_chunks_from_triton_rst(rst_path: Path) -> list[dict]:
    return parse_triton_language_rst(rst_path.read_text(encoding="utf-8"), source=rst_path.name)


def doc_kinds_for_sources(sources: frozenset[str]) -> frozenset[str]:
    kinds: set[str] = set()
    if "api_doc" in sources:
        kinds.add(CHUNK_API_SYMBOL)
    if "doc_semantics" in sources:
        kinds.add(CHUNK_SEMANTICS)
    if "doc_tutorial" in sources:
        kinds.add(CHUNK_TUTORIAL)
    return frozenset(kinds)


def api_unit_chunks_from_registry(gpu_architecture: str = ARCH_BLACKWELL) -> list[dict]:
    gpu_label = sm_label(gpu_architecture)
    fp8_desc = (
        "tl.float8e4nv / tl.float8e5m2 and fp4 tl.float4e2m1 where applicable."
        if fp4_supported(gpu_architecture)
        else "tl.float8e4nv / tl.float8e5m2 where applicable (no native fp4 tensor cores on this target)."
    )
    units = [
        ("tl.make_tensor_descriptor", "Tensor descriptors: desc.load/store offsets (replaces block_ptr in 3.7)."),
        ("tl.dot", f"tl.dot with fp32 acc and input_precision tf32/ieee on {gpu_label}."),
        ("tl.associative_scan", "tl.associative_scan / tl.reduce / tl.cumsum patterns."),
        ("@triton.autotune", "@triton.autotune configs with num_warps, num_stages, BLOCK_* keys."),
        ("fp8_dtypes", fp8_desc),
    ]
    return [
        {
            "chunk_kind": CHUNK_API_SYMBOL,
            "source": "triton_371_registry",
            "section_id": f"api_{name.replace('.', '_').lstrip('@')}",
            "title": name,
            "content": f"API unit: {name}\n{desc}",
            "target_api": name,
        }
        for name, desc in units
    ]


def _task_id_from_section(section_id: str, *, prefix: str = "doc") -> str:
    if section_id.startswith(("api_", "sem_", "tut_", "doc_")):
        return section_id
    return f"{prefix}_{section_id}"


def prompt_from_api_chunk(chunk: dict, gpu_architecture: str = ARCH_BLACKWELL) -> dict:
    target = chunk.get("target_api", "tl.dot")
    section_id = chunk["section_id"]
    ctx_limit = 4000 if chunk.get("api_page_enriched") else 1500
    gpu_label = sm_label(gpu_architecture)
    return {
        "task_id": _task_id_from_section(section_id),
        "source": "api_doc",
        "category": "kernel_write",
        "target_api": target,
        "title": chunk.get("title", target),
        "prompt": (
            f"Write a complete Triton 3.7.1 kernel on {gpu_label} that **must use** `{target}`.\n"
            f"Define exact input shapes/dtypes, include @triton.jit, launcher, masks, and torch.allclose test.\n\n"
            f"Context:\n{chunk['content'][:ctx_limit]}"
        ),
        "doc_chunk_id": section_id,
        "gpu_architecture": gpu_architecture,
    }


def prompt_from_explain_chunk(chunk: dict, gpu_architecture: str = ARCH_BLACKWELL) -> dict:
    title = chunk.get("title", "Triton concept")
    section_id = chunk["section_id"]
    source = "doc_tutorial" if chunk.get("chunk_kind") == CHUNK_TUTORIAL else "doc_semantics"
    gpu_label = sm_label(gpu_architecture)
    return {
        "task_id": _task_id_from_section(section_id),
        "source": source,
        "category": "doc_explain_implement",
        "title": title,
        "prompt": (
            f"Read this Triton 3.7.1 documentation about **{title}**.\n\n"
            "1. Explain the key rules or ideas in 3-6 sentences.\n"
            f"2. Write a complete runnable Triton 3.7.1 kernel on {gpu_label} that demonstrates the concept.\n"
            "Include @triton.jit, launcher, masks where needed, torch.allclose test, and "
            'print("SPARKPROOF_TRITON_PASS") after tests pass.\n\n'
            f"Documentation:\n{chunk['content'][:2500]}"
        ),
        "doc_chunk_id": section_id,
        "gpu_architecture": gpu_architecture,
    }


def prompt_from_doc_chunk(chunk: dict, gpu_architecture: str = ARCH_BLACKWELL) -> dict:
    kind = chunk.get("chunk_kind", CHUNK_API_SYMBOL)
    if kind == CHUNK_API_SYMBOL:
        return prompt_from_api_chunk(chunk, gpu_architecture=gpu_architecture)
    return prompt_from_explain_chunk(chunk, gpu_architecture=gpu_architecture)
