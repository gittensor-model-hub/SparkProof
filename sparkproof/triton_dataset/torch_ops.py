"""Source C: torch-op → Triton translation prompts."""

from __future__ import annotations

import json

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
]


def build_torch_translation_prompt(op: dict) -> dict:
    prompt = f"""Write a Triton 3.7.1 kernel replicating this PyTorch operation on Blackwell SM12x:

Operation: `{op['code']}`
Shapes: {json.dumps(op['shapes'])}

Requirements:
1. @triton.jit kernel + host launcher with tl.cdiv grid
2. Boundary masks on tl.load/tl.store where needed
3. fp32 accumulator for reductions
4. Self-contained test with torch.allclose at the end
5. Print SPARKPROOF_TRITON_PASS after successful test"""
    return {
        "task_id": f"translate_{op['name'].lower()}",
        "source": "torch_op",
        "origin": "torch_op",
        "split": "train",
        "category": "translation",
        "task_family": op.get("task_family", op["name"].lower()),
        "shape_class": op.get("shape_class"),
        "prompt": prompt,
        "torch_reference": op["code"],
        "reference_expr": op["code"],
    }


def iter_torch_translation_prompts() -> list[dict]:
    return [build_torch_translation_prompt(op) for op in TORCH_OPS]
