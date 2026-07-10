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
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
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
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
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

ELEMENTWISE_MUL_VALID = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * y, mask=mask)


def launch_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    mul_kernel[grid](x, y, out, n)
    return out


def test_kernel():
    x = torch.randn(1003, device="cuda", dtype=torch.float32)
    y = torch.randn(1003, device="cuda", dtype=torch.float32)
    out = launch_mul(x, y)
    assert torch.allclose(out, x * y)


test_kernel()
print("SPARKPROOF_TRITON_PASS")
'''

RELU_VALID = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def launch_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    relu_kernel[grid](x, out, n)
    return out


def test_kernel():
    x = torch.randn(1003, device="cuda", dtype=torch.float32)
    out = launch_relu(x)
    assert torch.allclose(out, torch.relu(x))


test_kernel()
print("SPARKPROOF_TRITON_PASS")
'''

ROW_SUM_VALID = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["ncols"],
)
@triton.jit
def row_sum_kernel(out_ptr, in_ptr, stride_row, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_in = in_ptr + row * stride_row
    cols = tl.arange(0, BLOCK)
    mask = cols < ncols
    vals = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(out_ptr + row, acc)


def launch_row_sum(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    grid = (rows,)
    row_sum_kernel[grid](out, x, x.stride(0), cols)
    return out


def test_kernel():
    x = torch.randn(8, 1003, device="cuda", dtype=torch.float32)
    out = launch_row_sum(x)
    ref = x.sum(dim=-1)
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)


test_kernel()
print("SPARKPROOF_TRITON_PASS")
'''

MATMUL_2D_VALID = '''
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    c = acc.to(tl.float16)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def launch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c


def test_kernel():
    a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.float16)
    out = launch_matmul(a, b)
    ref = torch.matmul(a, b)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


test_kernel()
print("SPARKPROOF_TRITON_PASS")
'''

REFERENCE_KERNELS: dict[str, str] = {
    "vector_add": VECTOR_ADD_VALID,
    "softmax_row": SOFTMAX_ROW_VALID,
    "elementwise_mul": ELEMENTWISE_MUL_VALID,
    "relu": RELU_VALID,
    "row_sum": ROW_SUM_VALID,
    "matmul_2d": MATMUL_2D_VALID,
}
