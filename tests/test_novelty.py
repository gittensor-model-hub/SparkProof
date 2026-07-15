from sparkproof.triton_dataset.novelty import NoveltyRegistry, compute_novelty_report, fingerprint_row


def _row(task_id: str, prompt: str, response: str, category: str, **meta) -> dict:
    return {
        "task_id": task_id,
        "prompt": prompt,
        "response": f"```python\n{response}\n```",
        "metadata": {"prompt_meta": {"task_id": task_id, "category": category, **meta}},
    }


def test_identical_prompt_is_an_exact_duplicate():
    accepted = [_row("a1", "Write a relu kernel", "print(1)", "relu")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_row("b1", "Write a relu kernel", "print(2)", "relu")], registry
    )
    assert report.exact_duplicate_rows == 1
    assert report.novel_verified_rows == 0


def test_identical_code_is_an_exact_duplicate_even_with_different_prompt_text():
    accepted = [_row("a1", "Write a relu kernel please", "print(1)", "relu")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_row("b1", "Please write a kernel for relu", "print(1)", "relu")], registry
    )
    assert report.exact_duplicate_rows == 1


def test_same_family_different_code_is_a_near_duplicate():
    accepted = [_row("a1", "Write a relu kernel", "print(1)", "relu")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_row("b1", "Implement ReLU activation in Triton", "print(999)", "relu")], registry
    )
    assert report.exact_duplicate_rows == 0
    assert report.near_duplicate_rows == 1


def test_unrelated_row_is_novel():
    accepted = [_row("a1", "Write a relu kernel", "print(1)", "relu")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_row("b1", "Write a flash attention kernel", "print(999)", "attention")], registry
    )
    assert report.novel_verified_rows == 1
    assert report.exact_duplicate_rows == 0
    assert report.near_duplicate_rows == 0


def test_within_bundle_duplicates_are_caught_against_empty_registry():
    registry = NoveltyRegistry()
    report = compute_novelty_report(
        [
            _row("b1", "Write a matmul kernel", "print(1)", "matmul"),
            _row("b2", "Write a matmul kernel", "print(2)", "matmul"),
        ],
        registry,
    )
    assert report.verified_rows == 2
    assert report.exact_duplicate_rows == 1  # b2 duplicates b1's prompt
    assert report.novel_verified_rows == 1


def test_dedup_evasion_by_changing_task_id_alone_does_not_evade_detection():
    """Changing only the task_id (or, by extension, a run_seed/run_id) must not
    disguise a duplicate — fingerprints are content-based, not identity-based."""
    accepted = [_row("original_task_id", "Write a relu kernel", "print(1)", "relu")]
    registry = NoveltyRegistry.from_rows(accepted)

    evasion_attempt = _row("totally_different_task_id_12345", "Write a relu kernel", "print(1)", "relu")
    report = compute_novelty_report([evasion_attempt], registry)
    assert report.exact_duplicate_rows == 1
    assert report.novel_verified_rows == 0


def test_compute_novelty_report_does_not_mutate_the_input_registry():
    registry = NoveltyRegistry()
    compute_novelty_report([_row("b1", "Write a matmul kernel", "print(1)", "matmul")], registry)
    # a fresh call against the same (unmutated) registry sees no prior state
    report = compute_novelty_report([_row("b2", "Write a matmul kernel", "print(2)", "matmul")], registry)
    assert report.novel_verified_rows == 1
    assert report.exact_duplicate_rows == 0


def test_same_prompt_different_gpu_architecture_is_novel():
    accepted = [_row("a1", "Write a relu kernel", "print(1)", "relu", gpu_architecture="blackwell")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_row("b1", "Write a relu kernel", "print(1)", "relu", gpu_architecture="hopper-h100")],
        registry,
    )
    assert report.novel_verified_rows == 1
    assert report.exact_duplicate_rows == 0
    assert report.near_duplicate_rows == 0


def test_same_prompt_same_gpu_architecture_remains_exact_duplicate():
    accepted = [_row("a1", "Write a relu kernel", "print(1)", "relu", gpu_architecture="blackwell")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_row("b1", "Write a relu kernel", "print(2)", "relu", gpu_architecture="blackwell")],
        registry,
    )
    assert report.exact_duplicate_rows == 1
    assert report.novel_verified_rows == 0


def test_fingerprint_row_ignores_absent_response():
    fp = fingerprint_row({"task_id": "p1", "prompt": "Write a kernel", "metadata": {"prompt_meta": {"category": "relu"}}})
    assert fp.assistant_ast_hash is None
    assert fp.prompt_hash


_REPAIR_WRAPPER = (
    "Your prior Triton 3.7.1 answer failed hardware validation.\n"
    "Failure: triton_api\nTrace tail:\n\n"
    "Return corrected **complete runnable Python** (kernel + launcher + torch.allclose test).\n\n"
    "```python\npass\n```"
)


def _repair_row(task_id: str, task_prompt: str, response: str, category: str, **meta) -> dict:
    return {
        "task_id": task_id,
        "prompt": _REPAIR_WRAPPER,
        "response": f"```python\n{response}\n```",
        "metadata": {
            "tier": "repair",
            "prompt_meta": {"task_id": task_id, "prompt": task_prompt, "category": category, **meta},
        },
    }


def test_repair_tier_fingerprints_task_prompt_not_shared_wrapper():
    accepted = [_repair_row("a1", "Implement translate_matmul for Hopper SM90", "print(1)", "matmul")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_repair_row("b1", "Implement translate_relu for Hopper SM90", "print(2)", "relu")],
        registry,
    )
    assert report.novel_verified_rows == 1
    assert report.exact_duplicate_rows == 0


def test_repair_tier_same_task_prompt_remains_exact_duplicate():
    task_prompt = "Implement translate_matmul for Hopper SM90 with TMA descriptors"
    accepted = [_repair_row("a1", task_prompt, "print(1)", "matmul")]
    registry = NoveltyRegistry.from_rows(accepted)

    report = compute_novelty_report(
        [_repair_row("b1", task_prompt, "print(2)", "matmul")],
        registry,
    )
    assert report.exact_duplicate_rows == 1
    assert report.novel_verified_rows == 0


def test_fingerprint_row_prefers_prompt_meta_over_repair_wrapper():
    task_prompt = "Write a flash attention kernel for hopper"
    repair_fp = fingerprint_row(_repair_row("t1", task_prompt, "print(1)", "attention"))
    silver_fp = fingerprint_row(_row("t2", task_prompt, "print(2)", "attention"))
    assert repair_fp.prompt_hash == silver_fp.prompt_hash
