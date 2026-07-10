"""Option B: enrich API symbol chunks from Sphinx-generated triton.language pages."""

from __future__ import annotations

import os
from html import unescape
from pathlib import Path
import re
import urllib.error
import urllib.request

_API_PAGE_RST_RE = re.compile(r"^triton\.language\.([a-z0-9_]+)\.rst$", re.IGNORECASE)
_DEFAULT_TRITON_DOCS_REF = "v3.7.1"


def _docs_ref() -> str:
    return os.environ.get("SPARKPROOF_TRITON_DOCS_REF", _DEFAULT_TRITON_DOCS_REF)


def _is_offline() -> bool:
    return os.environ.get("SPARKPROOF_TRITON_DOCS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def triton_docs_cache_dir() -> Path:
    override = os.environ.get("SPARKPROOF_TRITON_DOCS_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "sparkproof" / "triton-docs"
    return Path.home() / ".cache" / "sparkproof" / "triton-docs"


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _write_cache(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

def api_page_base_url() -> str | None:
    """Return an operator-pinned API HTML mirror; moving ``main`` is never implicit."""
    configured = os.environ.get("SPARKPROOF_TRITON_API_PAGE_BASE", "").strip()
    return configured.rstrip("/") if configured else None


def api_pages_enabled() -> bool:
    return os.environ.get("SPARKPROOF_TRITON_API_PAGES", "1").strip().lower() not in {"0", "false", "no"}


def api_page_cache_path(symbol: str, *, ref: str | None = None) -> Path:
    ref = ref or _docs_ref()
    return triton_docs_cache_dir() / ref / "api-pages-pinned" / f"{symbol}.txt"


def api_page_html_url(symbol: str) -> str:
    base = api_page_base_url()
    if base is None:
        raise ValueError(
            "remote API enrichment requires SPARKPROOF_TRITON_API_PAGE_BASE "
            "pointing to a version-pinned mirror"
        )
    return f"{base}/triton.language.{symbol}.html"


def find_local_api_page_rst(doc_dir: Path, symbol: str) -> Path | None:
    for base in (doc_dir, doc_dir / "python-api"):
        generated = base / "generated" / f"triton.language.{symbol}.rst"
        if generated.is_file():
            return generated
        direct = base / f"triton.language.{symbol}.rst"
        if direct.is_file():
            return direct
    return None


def list_local_api_page_rsts(doc_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for base in (doc_dir / "python-api" / "generated", doc_dir / "generated", doc_dir):
        if not base.is_dir():
            continue
        for path in base.glob("triton.language.*.rst"):
            match = _API_PAGE_RST_RE.match(path.name)
            if match:
                found[match.group(1)] = path
    return found


def parse_api_page_rst(content: str) -> str:
    lines: list[str] = []
    skip_directive_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(".. ") and stripped.endswith("::"):
            skip_directive_block = True
            continue
        if skip_directive_block:
            if not stripped or stripped.startswith(":"):
                continue
            skip_directive_block = False
        if stripped.startswith(".."):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|nav|header|footer).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"<(br|hr)\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</(p|div|li|h\d|tr|pre)>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_api_page_html(symbol: str) -> str:
    return _fetch_text(api_page_html_url(symbol))


def resolve_api_page_text(
    symbol: str,
    *,
    doc_dir: Path | None = None,
    fetch: bool = True,
) -> str | None:
    if doc_dir and doc_dir.exists():
        local = find_local_api_page_rst(doc_dir, symbol)
        if local is not None:
            parsed = parse_api_page_rst(local.read_text(encoding="utf-8"))
            if len(parsed) >= 80:
                return parsed

    env_dir = os.environ.get("SPARKPROOF_TRITON_DOCS_DIR")
    if env_dir:
        local = find_local_api_page_rst(Path(env_dir), symbol)
        if local is not None:
            parsed = parse_api_page_rst(local.read_text(encoding="utf-8"))
            if len(parsed) >= 80:
                return parsed

    cached = api_page_cache_path(symbol)
    if cached.is_file():
        return cached.read_text(encoding="utf-8")

    if fetch and not _is_offline() and api_page_base_url() is not None:
        try:
            html = fetch_api_page_html(symbol)
            text = html_to_text(html)
            if len(text) < 80 or "triton.language" not in text:
                return None
            _write_cache(cached, text)
            return text
        except (OSError, urllib.error.URLError, ValueError):
            return None
    return None


def enrich_api_symbol_chunk(
    chunk: dict,
    *,
    doc_dir: Path | None = None,
    fetch: bool = True,
) -> dict:
    symbol = chunk.get("symbol")
    if not symbol:
        return chunk
    page = resolve_api_page_text(symbol, doc_dir=doc_dir, fetch=fetch)
    if not page:
        return chunk
    category = chunk.get("doc_category", "triton.language")
    target_api = chunk.get("target_api", f"tl.{symbol}")
    enriched = dict(chunk)
    enriched["content"] = (
        f"Category: {category}\n"
        f"API: {target_api}\n"
        f"Symbol: {symbol} (triton.language)\n\n"
        f"Official API documentation:\n{page[:4000]}"
    )
    enriched["api_page_enriched"] = True
    return enriched


def enrich_api_symbol_chunks(
    chunks: list[dict],
    *,
    doc_dir: Path | None = None,
    fetch: bool = True,
    enabled: bool | None = None,
) -> list[dict]:
    if enabled is None:
        enabled = api_pages_enabled()
    if not enabled:
        return chunks
    return [
        enrich_api_symbol_chunk(chunk, doc_dir=doc_dir, fetch=fetch)
        if chunk.get("chunk_kind") == "api_symbol" and chunk.get("symbol")
        else chunk
        for chunk in chunks
    ]
