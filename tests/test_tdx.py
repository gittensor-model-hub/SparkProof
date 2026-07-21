"""Tests for Intel TDX helpers in SparkProof dataset attestation."""

from sparkproof.gpu.tdx import tdx_quote, tdx_report_data


def test_tdx_report_data_pads_nonce():
    nonce = "ab" * 32
    data = tdx_report_data(nonce)
    assert len(data) == 64
    assert data[:32] == bytes.fromhex(nonce)
    assert data[32:] == b"\x00" * 32


def test_extract_report_data_from_quote():
    import base64

    from sparkproof.gpu.tdx import _TDX_REPORT_DATA_OFFSET, extract_report_data_from_quote, tdx_report_data

    nonce = "cd" * 32
    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(nonce) + b"\x00" * 32
    assert extract_report_data_from_quote(base64.b64encode(quote).decode()) == tdx_report_data(nonce).hex()
    assert extract_report_data_from_quote("AAAA") is None


def test_tdx_quote_absent_without_tsm_node(tmp_path):
    assert tdx_quote("ab" * 32, report_path=tmp_path / "missing" / "tsm") is None
