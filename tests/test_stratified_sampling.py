import math

from sparkproof.triton_dataset.stratified_sampling import bucket_key, catalog_sha256, stratified_sample


def _catalog() -> list[dict]:
    records = []
    for i in range(20):
        records.append({"task_id": f"doc_{i}", "prompt": f"doc {i}", "source": "api_doc", "task_family": f"fam{i % 4}"})
    for i in range(15):
        records.append({"task_id": f"mut_{i}", "prompt": f"mutation {i}", "source": "mutation", "task_family": f"kernel{i % 6}"})
    for i in range(17):
        records.append({"task_id": f"top_{i}", "prompt": f"torch {i}", "source": "torch_op", "task_family": f"op{i}"})
    return records


def test_no_limit_returns_everything_unchanged():
    records = _catalog()
    sampled, counts = stratified_sample(records, limit=None, seed=1)
    assert sampled == records
    assert sum(counts.values()) == len(records)


def test_limit_zero_returns_empty():
    sampled, counts = stratified_sample(_catalog(), limit=0, seed=1)
    assert sampled == []
    assert counts == {}


def test_small_limit_covers_multiple_sources_not_a_fixed_prefix():
    """Regression for issue #9: --limit 2 must not always be the same api_doc prefix."""
    records = _catalog()
    sampled, counts = stratified_sample(records, limit=2, seed=42)
    assert len(sampled) == 2
    assert len(counts) == 2  # two distinct sources covered, not one source twice
    task_ids = {r["task_id"] for r in sampled}
    # must not be the deterministic first-two api_doc records every time
    assert task_ids != {"doc_0", "doc_1"}


def test_limit_covers_every_source_when_limit_allows():
    records = _catalog()
    sampled, counts = stratified_sample(records, limit=9, seed=7)
    assert len(sampled) == 9
    assert set(counts.keys()) == {"api_doc", "mutation", "torch_op"}
    assert all(count >= 1 for count in counts.values())


def test_same_seed_is_byte_identical_reproducible():
    records = _catalog()
    first, _ = stratified_sample(records, limit=12, seed=123)
    second, _ = stratified_sample(records, limit=12, seed=123)
    assert [r["task_id"] for r in first] == [r["task_id"] for r in second]


def test_different_seeds_produce_different_subsets():
    records = _catalog()
    a, _ = stratified_sample(records, limit=12, seed=1)
    b, _ = stratified_sample(records, limit=12, seed=2)
    assert [r["task_id"] for r in a] != [r["task_id"] for r in b]


def test_max_share_cap_is_respected_when_it_is_mathematically_satisfiable():
    # 3 sources: a cap must be >= 1/3 of the total to ever hold without relaxation.
    records = _catalog()
    sampled, counts = stratified_sample(records, limit=15, seed=5, max_share=0.4)
    assert len(sampled) == 15
    cap = math.ceil(15 * 0.4)
    assert all(count <= cap for count in counts.values())


def test_tight_cap_is_relaxed_when_too_few_buckets_can_satisfy_it():
    # 3 sources at a 25% cap can supply at most 3 * ceil(limit * 0.25) < limit for
    # most limits — relaxation must still reach `limit` rather than under-fill.
    records = _catalog()
    sampled, _ = stratified_sample(records, limit=16, seed=5, max_share=0.25)
    assert len(sampled) == 16


def test_cap_is_relaxed_rather_than_under_filling_limit():
    # Only one source available — the cap must not prevent reaching `limit`.
    records = [{"task_id": f"m_{i}", "prompt": f"p{i}", "source": "mutation", "task_family": "k"} for i in range(15)]
    sampled, counts = stratified_sample(records, limit=10, seed=1, max_share=0.1)
    assert len(sampled) == 10
    assert counts["mutation"] == 10


def test_stratified_sample_never_duplicates_or_invents_records():
    records = _catalog()
    sampled, _ = stratified_sample(records, limit=30, seed=9)
    task_ids = [r["task_id"] for r in sampled]
    assert len(task_ids) == len(set(task_ids))
    assert set(task_ids) <= {r["task_id"] for r in records}


def test_bucket_key_falls_back_through_family_fields():
    assert bucket_key({"source": "torch_op", "task_family": "relu"}) == ("torch_op", "relu")
    assert bucket_key({"source": "torch_op", "category": "translation"}) == ("torch_op", "translation")
    assert bucket_key({"origin": "mutation"}) == ("mutation", "unknown")


def test_catalog_sha256_is_order_independent_and_content_sensitive():
    records = _catalog()
    assert catalog_sha256(records) == catalog_sha256(list(reversed(records)))
    mutated = list(records)
    mutated[0] = {**mutated[0], "prompt": "changed"}
    assert catalog_sha256(mutated) != catalog_sha256(records)
