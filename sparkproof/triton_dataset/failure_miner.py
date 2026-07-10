"""Source E: mine dev failures into new private training tasks (never eval tasks)."""

from __future__ import annotations

from typing import Any

from sparkproof.triton_dataset.multi_candidate import extract_code
from sparkproof.triton_dataset.task_policy import FORBIDDEN_TRAINING_ORIGINS, normalize_train_task

FAILURE_TEMPLATES: dict[str, str] = {
    "compile_error": (
        "Write a Triton 3.7.1 kernel for Blackwell SM12x: row-wise {op} with tail dimension N=8191, "
        "explicit masks, fp32 accumulator, and torch.allclose test."
    ),
    "tail_mask_failure": (
        "Write a row {op} kernel where N is not divisible by BLOCK_SIZE (use N=6143). "
        "Mask all loads/stores. Include torch.allclose vs PyTorch."
    ),
    "stride_error": (
        "Write a Triton kernel for {op} on a **non-contiguous** strided tensor. "
        "Validate against PyTorch on the same strided layout."
    ),
    "dtype_error": (
        "Write a bf16 {op} kernel with fp32 accumulation. Compare to PyTorch bf16 reference with tolerance."
    ),
    "wrong_api_version": (
        "Write Triton 3.7.1 kernel using tl.make_tensor_descriptor (not block_ptr) for {op} on Blackwell."
    ),
    "runtime_error": (
        "Write a robust {op} kernel with grid computed via tl.cdiv and boundary checks for arbitrary M,N."
    ),
}


def classify_failure(validation: dict[str, Any]) -> str:
    if validation.get("passed"):
        return "pass"
    reason = validation.get("fail_reason") or ""
    stages = validation.get("stages") or {}
    if reason == "syntax_error":
        return "parse_error"
    if reason == "triton_api":
        return "wrong_api_version"
    if reason == "compile_execute_failed":
        tail = (stages.get("compile_execute") or {}).get("output_tail", "").lower()
        if "mask" in tail or "bound" in tail:
            return "tail_mask_failure"
        if "stride" in tail:
            return "stride_error"
        if "dtype" in tail or "float" in tail:
            return "dtype_error"
        return "compile_error"
    if reason == "benchmark_below_floor":
        return "performance_regression"
    return "runtime_error"


def record_failure(
    *,
    run_id: str,
    task: dict[str, Any],
    model: str,
    validation: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    origin = task.get("origin") or task.get("source")
    return {
        "run_id": run_id,
        "task_id": task.get("task_id"),
        "task_origin": origin,
        "split": task.get("split", "dev"),
        "model": model,
        "failure_stage": validation.get("fail_reason"),
        "failure_class": classify_failure(validation),
        "tags": [task.get("category") or "triton", task.get("task_family") or "kernel"],
        "gpu_arch": "blackwell",
        "triton_version": "3.7.1",
        "broken_code": extract_code(response),
    }


def mine_failure_to_tasks(failure: dict[str, Any], *, n: int = 2) -> list[dict[str, Any]]:
    if failure.get("task_origin") in FORBIDDEN_TRAINING_ORIGINS:
        return []
    if failure.get("split") in {"test", "eval"}:
        return []

    failure_class = failure.get("failure_class", "compile_error")
    template = FAILURE_TEMPLATES.get(failure_class, FAILURE_TEMPLATES["compile_error"])
    op = (failure.get("tags") or ["softmax"])[0]

    tasks: list[dict[str, Any]] = []
    for i in range(n):
        prompt = template.format(op=op)
        task = normalize_train_task(
            {
                "task_id": f"mined_{failure.get('task_id', 'x')}_{failure_class}_{i}",
                "origin": "failure_mining",
                "source": "failure_mining",
                "split": "train",
                "category": "failure_mined",
                "prompt": prompt,
                "parent_failure_class": failure_class,
                "parent_run_id": failure.get("run_id"),
            }
        )
        tasks.append(task)
    return tasks
