"""Structured teacher prompt templates for Triton specialist training."""

from __future__ import annotations

from typing import Any


DESIGN_SECTION = """## Design
- Target GPU: NVIDIA Blackwell SM12x with Triton {triton_version}
- Use @triton.jit and explicit tl.cdiv grid sizing
- Include boundary masks on tl.load/tl.store where shapes are not tile-aligned
- Use fp32 accumulators for reductions unless the reference op is explicitly fp16-safe"""


IMPLEMENTATION_SECTION = """## Implementation
- Provide a complete runnable Python file: imports, kernel(s), launcher, and self-test
- Avoid PyTorch compute fallbacks (no torch.matmul/softmax/etc. in the launcher path)
- Instantiate adversarial test shapes (e.g. M=127, N=1003, K=6143) and both fp32/fp16 inputs
- End the self-test with print("SPARKPROOF_TRITON_PASS") after torch.allclose passes"""


VALIDATION_SECTION = """## Validation expectations
- Kernel must launch with explicit grid syntax: `kernel[grid](...)`
- Tests must cover non-power-of-two tails and multiple random seeds
- Document rtol/atol choices when comparing fp16 outputs
- Invoke `triton.testing.do_bench(lambda: launcher(...))`; SparkProof records
  the returned timing independently"""


def wrap_prompt(
    body: str,
    *,
    triton_version: str = "3.7.1",
    include_sections: bool = True,
) -> str:
    if not include_sections:
        return body.strip()
    sections = "\n\n".join(
        section.format(triton_version=triton_version)
        for section in (DESIGN_SECTION, IMPLEMENTATION_SECTION, VALIDATION_SECTION)
    )
    return f"{body.strip()}\n\n{sections}"


def apply_prompt_template(record: dict[str, Any], *, triton_version: str = "3.7.1") -> dict[str, Any]:
    out = dict(record)
    category = out.get("category", "translation")
    base = out.get("prompt", "").strip()
    if category == "debugging":
        header = "Debug the broken Triton kernel below. Fix correctness on Blackwell without changing the intended algorithm."
    elif category == "translation":
        header = "Translate the reference PyTorch operation into an optimized Triton 3.7.1 kernel."
    elif category == "api_usage":
        header = "Demonstrate correct usage of the target Triton API on Blackwell."
    else:
        header = "Write a production-quality Triton 3.7.1 kernel for Blackwell."
    out["prompt"] = wrap_prompt(f"{header}\n\n{base}", triton_version=triton_version)
    out["prompt_template"] = f"{category}:structured-v1"
    return out
