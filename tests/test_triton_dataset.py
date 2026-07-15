from sparkproof.triton_dataset.build_prompts import iter_all_prompts
from sparkproof.triton_dataset.decontaminate import get_canonical_structure, semantic_task_fingerprint
from sparkproof.triton_dataset.eval_problems import iter_eval_problem_prompts
from sparkproof.triton_dataset.failure_miner import classify_failure, mine_failure_to_tasks
from sparkproof.triton_dataset.mutator import strip_boundary_mask
from sparkproof.triton_dataset.self_evolve import evolve_parent
from sparkproof.triton_dataset.task_policy import assert_trainable_task


def test_build_prompts_excludes_yaml_by_default():
    records = list(iter_all_prompts(include_sources=frozenset({"mutation", "torch_op"})))
    sources = {r["source"] for r in records}
    assert "yaml" not in sources
    assert "mutation" in sources
    assert "torch_op" in sources
    assert len([r for r in records if r["source"] == "mutation"]) == 15
    assert len([r for r in records if r["source"] == "torch_op"]) == 17
    assert all("prompt" in r and "system" in r for r in records)
    assert all(r.get("split") == "train" for r in records)


def test_default_train_sources_include_all_doc_kinds():
    from sparkproof.triton_dataset.build_prompts import DEFAULT_TRAIN_SOURCES

    assert DEFAULT_TRAIN_SOURCES == frozenset(
        {"api_doc", "doc_semantics", "doc_tutorial", "mutation", "torch_op"}
    )


def test_mutator_strips_mask():
    code = 'x = tl.load(ptr + offs, mask=offs < n, other=0.0)'
    mutated, reason = strip_boundary_mask(code)
    assert "mask=" not in mutated
    assert reason


def test_assert_trainable_task_blocks_tritonbench():
    import pytest

    with pytest.raises(ValueError, match="cannot enter training"):
        assert_trainable_task({"task_id": "tb_1", "origin": "tritonbench", "split": "eval"})


def test_eval_problems_marked_eval_only():
    prompts = iter_eval_problem_prompts()
    if not prompts:
        return
    assert prompts[0]["origin"] == "tritonbench"
    assert prompts[0]["split"] == "eval"


def test_self_evolution_requires_oracle():
    parent = {
        "task_id": "torch_rmsnorm_1",
        "prompt": "Write RMSNorm",
        "origin": "torch_op",
        "split": "train",
        "ground_truth_code": "@triton.jit\ndef k(): pass",
    }
    children = evolve_parent(parent, depth=1)
    assert children
    assert children[0]["origin"] == "self_evolution"
    assert children[0]["parent_id"] == "torch_rmsnorm_1"


def test_failure_miner_skips_eval_origin():
    failure = {
        "task_origin": "tritonbench",
        "failure_class": "compile_error",
        "tags": ["softmax"],
        "task_id": "tb_x",
        "split": "eval",
        "run_id": "r1",
    }
    assert mine_failure_to_tasks(failure) == []


def test_failure_classify_compile():
    validation = {"passed": False, "fail_reason": "compile_execute_failed", "stages": {"compile_execute": {"output_tail": "mask out of bounds"}}}
    assert classify_failure(validation) == "tail_mask_failure"


def test_ast_structure_stable():
    a = get_canonical_structure("import triton\nimport triton.language as tl\n@triton.jit\ndef k(x_ptr, n):\n    offs = tl.arange(0, 128)\n    tl.load(x_ptr + offs, mask=offs < n)")
    b = get_canonical_structure("import triton\nimport triton.language as tl\n@triton.jit\ndef q(y_ptr, m):\n    idx = tl.arange(0, 128)\n    tl.load(y_ptr + idx, mask=idx < m)")
    assert a == b


def test_semantic_fingerprint():
    fp = semantic_task_fingerprint({"task_family": "softmax", "dtype": "fp16", "shape_class": "M_x_tail_N"})
    assert len(fp) == 64


def test_semantic_fingerprint_distinguishes_architectures():
    task = {"task_family": "softmax", "dtype": "fp16", "shape_class": "M_x_tail_N"}
    blackwell_fp = semantic_task_fingerprint(task)
    hopper_fp = semantic_task_fingerprint({**task, "gpu_architecture": "hopper-h100"})
    default_fp = semantic_task_fingerprint({**task, "gpu_architecture": "blackwell"})
    assert blackwell_fp == default_fp  # unset defaults identically to explicit "blackwell"
    assert blackwell_fp != hopper_fp
