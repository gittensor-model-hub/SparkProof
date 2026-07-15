"""Prompt-generation text and provenance correctly track gpu_architecture."""

from sparkproof.gpu.architecture import ARCH_BLACKWELL, ARCH_HOPPER_H100
from sparkproof.triton_dataset.doc_chunks import (
    api_unit_chunks_from_registry,
    prompt_from_api_chunk,
    prompt_from_doc_chunk,
    prompt_from_explain_chunk,
)
from sparkproof.triton_dataset.failure_miner import failure_template, mine_failure_to_tasks
from sparkproof.triton_dataset.mutator import build_mutation_prompt, iter_mutation_prompts
from sparkproof.triton_dataset.prompt_templates import apply_prompt_template, wrap_prompt
from sparkproof.triton_dataset.self_evolve import apply_evolution
from sparkproof.triton_dataset.torch_ops import TORCH_OPS, build_torch_translation_prompt


def test_default_gpu_architecture_preserves_legacy_blackwell_text():
    r = apply_prompt_template({"category": "translation", "prompt": "x"})
    assert "Blackwell SM12x" in r["prompt"]
    assert r["gpu_architecture"] == ARCH_BLACKWELL


def test_hopper_gpu_architecture_swaps_prompt_wording():
    r = apply_prompt_template({"category": "api_usage", "prompt": "x"}, gpu_architecture=ARCH_HOPPER_H100)
    assert "Hopper SM90" in r["prompt"]
    assert "Blackwell" not in r["prompt"]
    assert r["gpu_architecture"] == ARCH_HOPPER_H100


def test_wrap_prompt_design_section_matches_architecture():
    body = wrap_prompt("do the thing", gpu_architecture=ARCH_HOPPER_H100)
    assert "Target GPU: NVIDIA Hopper SM90" in body


def test_doc_chunk_prompts_swap_wording_and_stamp_architecture():
    chunk = {"section_id": "api_tl_dot", "target_api": "tl.dot", "content": "docs", "chunk_kind": "api_symbol"}
    rec = prompt_from_api_chunk(chunk, gpu_architecture=ARCH_HOPPER_H100)
    assert "Hopper SM90" in rec["prompt"]
    assert rec["gpu_architecture"] == ARCH_HOPPER_H100

    explain_chunk = {"section_id": "sem_1", "title": "Masking", "content": "docs", "chunk_kind": "semantics"}
    rec2 = prompt_from_explain_chunk(explain_chunk, gpu_architecture=ARCH_HOPPER_H100)
    assert "Hopper SM90" in rec2["prompt"]

    dispatched = prompt_from_doc_chunk(chunk, gpu_architecture=ARCH_HOPPER_H100)
    assert dispatched["gpu_architecture"] == ARCH_HOPPER_H100


def test_api_unit_registry_excludes_fp4_for_hopper():
    blackwell_units = api_unit_chunks_from_registry(ARCH_BLACKWELL)
    hopper_units = api_unit_chunks_from_registry(ARCH_HOPPER_H100)
    blackwell_fp8 = next(u for u in blackwell_units if u["target_api"] == "fp8_dtypes")
    hopper_fp8 = next(u for u in hopper_units if u["target_api"] == "fp8_dtypes")
    assert "tl.float4e2m1" in blackwell_fp8["content"]
    assert "tl.float4e2m1" not in hopper_fp8["content"]


def test_mutation_prompts_swap_wording_and_stamp_architecture():
    prompt = build_mutation_prompt(
        task_id="t1", valid_kernel="@triton.autotune(x)\n@triton.jit\ndef f(): pass\n", gpu_architecture=ARCH_HOPPER_H100
    )
    # optimization-branch mutation triggers on autotune-stripping kernels
    all_prompts = iter_mutation_prompts(
        task_id="t1", valid_kernel="@triton.autotune(x)\n@triton.jit\ndef f(): pass\n", gpu_architecture=ARCH_HOPPER_H100
    )
    assert all(p["gpu_architecture"] == ARCH_HOPPER_H100 for p in all_prompts)
    opt_prompt = next(p for p in all_prompts if p["category"] == "optimization")
    assert "Hopper SM90" in opt_prompt["prompt"]
    assert prompt["gpu_architecture"] == ARCH_HOPPER_H100


def test_torch_translation_prompt_swaps_wording():
    rec = build_torch_translation_prompt(TORCH_OPS[0], gpu_architecture=ARCH_HOPPER_H100)
    assert "Hopper SM90" in rec["prompt"]
    assert rec["gpu_architecture"] == ARCH_HOPPER_H100


def test_failure_template_resolves_gpu_label_leaving_op_placeholder():
    template = failure_template("wrong_api_version", ARCH_HOPPER_H100)
    assert "Hopper SM90" in template
    assert "{op}" in template
    assert template.format(op="softmax") == template.replace("{op}", "softmax")


def test_mine_failure_to_tasks_uses_failure_gpu_arch_and_stamps_children():
    tasks = mine_failure_to_tasks(
        {
            "task_id": "f1",
            "task_origin": "torch_op",
            "split": "train",
            "failure_class": "wrong_api_version",
            "task_family": "matmul",
            "gpu_arch": ARCH_HOPPER_H100,
        }
    )
    assert tasks
    assert "Hopper SM90" in tasks[0]["prompt"]
    assert tasks[0]["gpu_architecture"] == ARCH_HOPPER_H100


def test_apply_evolution_inherits_parent_gpu_architecture():
    parent = {
        "task_id": "p1",
        "prompt": "base prompt",
        "category": "kernel_generation",
        "split": "train",
        "source": "torch_op",
        "origin": "torch_op",
        "gpu_architecture": ARCH_HOPPER_H100,
    }
    child = apply_evolution(parent, "optimization_target")
    assert child is not None
    assert "Hopper SM90" in child["prompt"]
    assert child["gpu_architecture"] == ARCH_HOPPER_H100


def test_apply_evolution_inject_bug_uses_parent_architecture():
    parent = {
        "task_id": "p1",
        "prompt": "base",
        "category": "kernel_generation",
        "split": "train",
        "source": "torch_op",
        "origin": "torch_op",
        "ground_truth_code": "@triton.jit\ndef k(x_ptr, n, BLOCK: tl.constexpr):\n    tl.load(x_ptr, mask=n)\n",
        "gpu_architecture": ARCH_HOPPER_H100,
    }
    child = apply_evolution(parent, "inject_bug")
    assert child is not None
    assert "Hopper SM90" in child["prompt"]
