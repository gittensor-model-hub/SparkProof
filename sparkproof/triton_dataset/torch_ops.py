"""Source C: torch-op → Triton translation prompts."""

from __future__ import annotations

import json

from sparkproof.gpu.architecture import ARCH_BLACKWELL, sm_label
from sparkproof.triton_dataset.reference_bench import DEFAULT_BENCHMARK_SIZES

TORCH_OPS = [
    {
        "name": "LayerNorm",
        "code": "torch.nn.functional.layer_norm(x, (D,), weight=w, bias=b, eps=1e-5)",
        "shapes": {"x": "(B, D)", "w": "(D,)", "b": "(D,)"},
        "task_family": "layernorm",
    },
    {
        "name": "GELU",
        "code": "torch.nn.functional.gelu(x, approximate='tanh')",
        "shapes": {"x": "(M, N)"},
        "task_family": "gelu",
    },
    {
        "name": "RMSNorm",
        "code": "x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight",
        "shapes": {"x": "(B, L, D)", "weight": "(D,)"},
        "task_family": "rmsnorm",
    },
    {
        "name": "Softmax",
        "code": "torch.softmax(x, dim=-1)",
        "shapes": {"x": "(M, N)"},
        "task_family": "softmax",
        "shape_class": "M_x_tail_N",
    },
    {
        "name": "LogSoftmax",
        "code": "torch.log_softmax(x, dim=-1)",
        "shapes": {"x": "(M, N)"},
        "task_family": "log_softmax",
    },
    {
        "name": "Matmul",
        "code": "torch.matmul(a, b)",
        "shapes": {"a": "(M, K)", "b": "(K, N)"},
        "task_family": "matmul",
    },
    {
        "name": "SiLU",
        "code": "torch.nn.functional.silu(x)",
        "shapes": {"x": "(M, N)"},
        "task_family": "silu",
    },
    {
        "name": "ReLU",
        "code": "torch.nn.functional.relu(x)",
        "shapes": {"x": "(M, N)"},
        "task_family": "relu",
    },
    {
        "name": "LeakyReLU",
        "code": "torch.nn.functional.leaky_relu(x, negative_slope=0.01)",
        "shapes": {"x": "(M, N)"},
        "task_family": "leaky_relu",
    },
    {
        "name": "Softplus",
        "code": "torch.nn.functional.softplus(x)",
        "shapes": {"x": "(M, N)"},
        "task_family": "softplus",
    },
    {
        "name": "Sigmoid",
        "code": "torch.sigmoid(x)",
        "shapes": {"x": "(M, N)"},
        "task_family": "sigmoid",
    },
    {
        "name": "Tanh",
        "code": "torch.tanh(x)",
        "shapes": {"x": "(M, N)"},
        "task_family": "tanh",
    },
    {
        "name": "ReduceSum",
        "code": "x.sum(dim=-1)",
        "shapes": {"x": "(M, N)"},
        "task_family": "reduce_sum",
        "shape_class": "M_x_tail_N",
    },
    {
        "name": "ReduceMean",
        "code": "x.mean(dim=-1)",
        "shapes": {"x": "(M, N)"},
        "task_family": "reduce_mean",
        "shape_class": "M_x_tail_N",
    },
    {
        "name": "Clamp",
        "code": "torch.clamp(x, min=-1.0, max=1.0)",
        "shapes": {"x": "(M, N)"},
        "task_family": "clamp",
    },
    {
        "name": "Abs",
        "code": "torch.abs(x)",
        "shapes": {"x": "(M, N)"},
        "task_family": "abs",
    },
    {
        "name": "SquaredDiff",
        "code": "(a - b).pow(2)",
        "shapes": {"a": "(M, N)", "b": "(M, N)"},
        "task_family": "squared_diff",
    },
]


ADVERSARIAL_SHAPE_PRESETS = [
    {"M": 127, "N": 1003, "K": 6143, "B": 3, "D": 1003, "L": 127},
    {"M": 255, "N": 511, "K": 1023, "B": 5, "D": 511, "L": 255},
]


def _benchmark_size_hint(shapes: dict[str, str]) -> str:
    dims = sorted({name for shape_expr in shapes.values() for name in shape_expr.strip("()").replace(",", " ").split()})
    sizes = {dim: DEFAULT_BENCHMARK_SIZES[dim] for dim in dims if dim in DEFAULT_BENCHMARK_SIZES}
    return ", ".join(f"{key}={value}" for key, value in sizes.items())


def build_torch_translation_prompt(
    op: dict, *, shape_preset: dict[str, int] | None = None, gpu_architecture: str = ARCH_BLACKWELL
) -> dict:
    benchmark_sizes = _benchmark_size_hint(op["shapes"])
    gpu_label = sm_label(gpu_architecture)
    prompt = f"""Write a Triton 3.7.1 kernel replicating this PyTorch operation on {gpu_label}:

Operation: `{op['code']}`
Shapes: {json.dumps(op['shapes'])}

Requirements:
1. @triton.jit kernel + host launcher with tl.cdiv grid
2. Boundary masks on tl.load/tl.store where needed
3. fp32 accumulator for reductions
4. Self-contained test with torch.allclose at the end
5. Instantiate symbolic dimensions with concrete adversarial sizes; use a non-power-of-two tail such as 1003
6. Test float32 and float16 inputs (and state justified tolerances)
7. Print SPARKPROOF_TRITON_PASS after successful test
8. Invoke triton.testing.do_bench(lambda: launcher(...)) on your correctness-test
   inputs; SparkProof records the returned timing independently
9. Additionally invoke triton.testing.do_bench once more with float32 inputs at
   these larger sizes:
   {benchmark_sizes}
   This last timing is diagnostic only. Candidate-controlled benchmark calls are
   not eligible for speed ranking or KernelBench fast_p credit."""
    if shape_preset:
        shape_hint = ", ".join(f"{key}={value}" for key, value in sorted(shape_preset.items()))
        prompt += f"\n\nUse these concrete dimensions in your self-test: {shape_hint}"
    task_suffix = ""
    if shape_preset:
        task_suffix = "_" + "_".join(str(shape_preset.get(k, "")) for k in ("M", "N", "K") if k in shape_preset)
    return {
        "task_id": f"translate_{op['name'].lower()}{task_suffix}",
        "source": "torch_op",
        "origin": "torch_op",
        "split": "train",
        "category": "translation",
        "task_family": op.get("task_family", op["name"].lower()),
        "shape_class": op.get("shape_class"),
        "prompt": prompt,
        "torch_reference": op["code"],
        "reference_expr": op["code"],
        "shapes": op["shapes"],
        "gpu_architecture": gpu_architecture,
    }


def iter_torch_translation_prompts(
    *, include_shape_variants: bool = False, gpu_architecture: str = ARCH_BLACKWELL
) -> list[dict]:
    prompts = [build_torch_translation_prompt(op, gpu_architecture=gpu_architecture) for op in TORCH_OPS]
    if not include_shape_variants:
        return prompts
    for op in TORCH_OPS:
        for preset in ADVERSARIAL_SHAPE_PRESETS:
            prompts.append(build_torch_translation_prompt(op, shape_preset=preset, gpu_architecture=gpu_architecture))
    return prompts
