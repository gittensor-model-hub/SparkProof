import os

import pytest

from sparkproof.triton_dataset.reference_kernels import REFERENCE_KERNELS


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("SPARKPROOF_RUN_GPU_TESTS") != "1",
    reason="set SPARKPROOF_RUN_GPU_TESTS=1 on a Blackwell runner",
)
@pytest.mark.parametrize("name,code", REFERENCE_KERNELS.items())
def test_reference_kernel_executes_on_blackwell(name: str, code: str):
    from sparkproof.triton.validator import TritonKernelValidator

    validation = TritonKernelValidator(gpu_index=0).validate_response(code)
    assert validation["passed"], f"{name}: {validation}"
