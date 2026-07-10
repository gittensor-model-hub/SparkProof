"""Reference kernels with known-good behavior (Source B: mutation seeds)."""

VECTOR_ADD_VALID = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def launch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    n = x.numel()
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    add_kernel[grid](x, y, out, n)
    return out


def test_kernel():
    x = torch.randn(1003, device="cuda", dtype=torch.float32)
    y = torch.randn(1003, device="cuda", dtype=torch.float32)
    out = launch_add(x, y)
    assert torch.allclose(out, x + y)


test_kernel()
print("SPARKPROOF_TRITON_PASS")
'''

SOFTMAX_ROW_VALID = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    ],
    key=["ncols"],
)
@triton.jit
def softmax_kernel(out_ptr, in_ptr, stride_row, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_in = in_ptr + row * stride_row
    cols = tl.arange(0, BLOCK)
    mask = cols < ncols
    vals = tl.load(row_in + cols, mask=mask, other=-float("inf"))
    vals = vals.to(tl.float32)
    row_max = tl.max(vals, axis=0)
    vals = vals - row_max
    num = tl.exp(vals)
    den = tl.sum(num, axis=0)
    out = num / den
    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


def launch_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    grid = (rows,)
    softmax_kernel[grid](out, x, x.stride(0), cols)
    return out


def test_kernel():
    x = torch.randn(8, 1003, device="cuda", dtype=torch.float32)
    out = launch_softmax(x)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)


test_kernel()
print("SPARKPROOF_TRITON_PASS")
'''

REFERENCE_KERNELS: dict[str, str] = {
    "vector_add": VECTOR_ADD_VALID,
    "softmax_row": SOFTMAX_ROW_VALID,
}
