"""Tests for Intel TDX helpers in SparkProof dataset attestation."""

from sparkproof.gpu.tdx import tdx_quote, tdx_report_data


def test_tdx_report_data_pads_nonce():
    nonce = "ab" * 32
    data = tdx_report_data(nonce)
    assert len(data) == 64
    assert data[:32] == bytes.fromhex(nonce)
    assert data[32:] == b"\x00" * 32


def test_tdx_quote_absent_without_tsm_node(tmp_path):
    assert tdx_quote("ab" * 32, report_path=tmp_path / "missing" / "tsm") is None
