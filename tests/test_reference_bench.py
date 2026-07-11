import pytest

from sparkproof.triton_dataset.reference_bench import (
    DEFAULT_BENCHMARK_SIZES,
    benchmark_reference,
    reference_runnable,
)
from sparkproof.triton_dataset.torch_ops import TORCH_OPS, build_torch_translation_prompt


def test_reference_runnable_none_without_prompt_meta():
    assert reference_runnable(None) is None
    assert reference_runnable({}) is None


def test_reference_runnable_none_without_reference_or_shapes():
    assert reference_runnable({"torch_reference": "torch.sigmoid(x)"}) is None
    assert reference_runnable({"shapes": {"x": "(M, N)"}}) is None


def test_reference_runnable_none_for_unresolvable_dimension():
    meta = {"torch_reference": "torch.sigmoid(x)", "shapes": {"x": "(WEIRD_DIM,)"}}
    assert reference_runnable(meta) is None


@pytest.mark.parametrize("op", TORCH_OPS, ids=[op["name"] for op in TORCH_OPS])
def test_reference_runnable_covers_every_torch_op(op: dict):
    prompt = build_torch_translation_prompt(op)
    source = reference_runnable(prompt)
    assert source is not None, f"{op['name']} has an unresolvable dimension in {op['shapes']}"
    assert op["code"] in source
    assert "triton.testing.do_bench" in source
    assert "SPARKPROOF_REFERENCE_PASS" in source


def test_reference_runnable_binds_dimension_letters_and_tensors():
    meta = {"torch_reference": "torch.matmul(a, b)", "shapes": {"a": "(M, K)", "b": "(K, N)"}}
    source = reference_runnable(meta)
    assert f"M = {DEFAULT_BENCHMARK_SIZES['M']}" in source
    assert f"K = {DEFAULT_BENCHMARK_SIZES['K']}" in source
    assert f"N = {DEFAULT_BENCHMARK_SIZES['N']}" in source
    assert "a = torch.randn((M, K), device='cuda', dtype=torch.float32)" in source
    assert "b = torch.randn((K, N), device='cuda', dtype=torch.float32)" in source


def test_reference_runnable_binds_orphan_scalar_default():
    meta = {
        "torch_reference": "x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight",
        "shapes": {"x": "(B, L, D)", "weight": "(D,)"},
    }
    source = reference_runnable(meta)
    assert "eps = 1e-05" in source


def test_benchmark_reference_none_without_gpu_available_reference():
    assert benchmark_reference(None) is None
    assert benchmark_reference({"torch_reference": "torch.sigmoid(x)"}) is None
