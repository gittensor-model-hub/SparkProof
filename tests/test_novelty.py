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


def test_fingerprint_row_ignores_absent_response():
    fp = fingerprint_row({"task_id": "p1", "prompt": "Write a kernel", "metadata": {"prompt_meta": {"category": "relu"}}})
    assert fp.assistant_ast_hash is None
    assert fp.prompt_hash
