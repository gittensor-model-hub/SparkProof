import os

import pytest

from sparkproof.triton_dataset.reference_bench import benchmark_reference
from sparkproof.triton_dataset.torch_ops import TORCH_OPS, build_torch_translation_prompt

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("SPARKPROOF_RUN_GPU_TESTS") != "1",
    reason="set SPARKPROOF_RUN_GPU_TESTS=1 on a Blackwell runner",
)
@pytest.mark.parametrize(
    "op",
    [op for op in TORCH_OPS if op["name"] in {"Sigmoid", "Matmul", "LayerNorm"}],
    ids=lambda op: op["name"],
)
def test_reference_bench_executes_and_times_on_blackwell(op: dict):
    prompt_meta = build_torch_translation_prompt(op)
    timing_ms = benchmark_reference(prompt_meta, gpu_index=0)
    assert timing_ms is not None, f"{op['name']} reference failed to execute"
    assert timing_ms > 0.0
