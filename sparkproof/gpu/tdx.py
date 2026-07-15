"""Intel TDX measured-VM attestation for dataset-generation bundles.

Binds the SparkProof validation pipeline (userland + kernel) to the same
dataset-bound nonce as GPU CC attestation, so validators can trust the
measured guest — not just the GPU — ran the release gate.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

# TDX quote v4 layout: 48-byte header + 584-byte TD report body; REPORTDATA is
# the final 64 bytes of the body; MRTD follows earlier fixed fields.
_TDX_QUOTE_HEADER_LEN = 48
_TDX_BODY_LEN = 584
_TDX_REPORT_DATA_OFFSET = _TDX_QUOTE_HEADER_LEN + _TDX_BODY_LEN - 64
_TDX_MRTD_OFFSET = _TDX_QUOTE_HEADER_LEN + 16 + 48 + 48 + 8 + 8 + 8

DEFAULT_TSM_REPORT_PATH = Path("/sys/kernel/config/tsm/report/sparkproof")


def tdx_report_data(nonce_hex: str) -> bytes:
    """64-byte TDX REPORTDATA for a dataset nonce: sha256 bytes + zero padding."""
    digest = bytes.fromhex(nonce_hex)
    if len(digest) > 64:
        raise ValueError("nonce too long for TDX report data (max 64 bytes)")
    return digest.ljust(64, b"\x00")


def tdx_quote(nonce_hex: str, report_path: Path | None = None) -> dict | None:
    """Capture an Intel TDX quote binding `nonce_hex`, or None without TDX/TSM."""
    node = report_path or Path(os.environ.get("SPARKPROOF_TSM_REPORT_PATH") or DEFAULT_TSM_REPORT_PATH)
    created = False
    try:
        if not node.is_dir():
            node.mkdir()
            created = True
        (node / "inblob").write_bytes(tdx_report_data(nonce_hex))
        quote = (node / "outblob").read_bytes()
        provider = (node / "provider").read_text().strip() if (node / "provider").exists() else ""
    except OSError:
        return None
    finally:
        if created:
            try:
                node.rmdir()
            except OSError:
                pass
    if len(quote) < _TDX_REPORT_DATA_OFFSET + 64:
        return None
    return {
        "provider": provider,
        "quote_b64": base64.b64encode(quote).decode(),
        "report_data": quote[_TDX_REPORT_DATA_OFFSET : _TDX_REPORT_DATA_OFFSET + 64].hex(),
        "mrtd": quote[_TDX_MRTD_OFFSET : _TDX_MRTD_OFFSET + 48].hex(),
    }


def verify_tdx_quote(quote_b64: str, pccs_url: str | None = None) -> dict:
    """DCAP-verify a TDX quote via Intel PCS (validator-side)."""
    try:
        import dcap_qvl
    except ImportError:
        return {"verified": False, "status": "dcap-qvl not installed", "advisory_ids": []}

    import asyncio

    try:
        quote = base64.b64decode(quote_b64)
        report = asyncio.run(dcap_qvl.get_collateral_and_verify(quote, pccs_url))
    except Exception as exc:
        return {"verified": False, "status": f"verification error: {exc}", "advisory_ids": []}
    return {
        "verified": report.status == "UpToDate" and not report.advisory_ids,
        "status": report.status,
        "advisory_ids": list(report.advisory_ids),
    }
