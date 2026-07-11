from sparkproof.triton_dataset.run_seed import evolution_seed, generate_run_seed, sampling_seed


def test_generate_run_seed_is_random_and_hex():
    a = generate_run_seed()
    b = generate_run_seed()
    assert a != b
    assert len(a) == 64  # 256-bit, hex-encoded
    int(a, 16)  # valid hex


def test_sampling_seed_is_deterministic():
    first = sampling_seed("catalog-hash", "run-seed")
    second = sampling_seed("catalog-hash", "run-seed")
    assert first == second


def test_sampling_seed_varies_with_each_input():
    base = sampling_seed("catalog-hash", "run-seed")
    assert sampling_seed("different-catalog", "run-seed") != base
    assert sampling_seed("catalog-hash", "different-run-seed") != base
    assert sampling_seed("catalog-hash", "run-seed", "policy-v2") != base


def test_evolution_seed_is_deterministic_and_scoped_per_parent_and_depth():
    first = evolution_seed("run-seed", "parent-1", 3)
    second = evolution_seed("run-seed", "parent-1", 3)
    assert first == second
    assert evolution_seed("run-seed", "parent-2", 3) != first
    assert evolution_seed("run-seed", "parent-1", 4) != first
    assert evolution_seed("different-run-seed", "parent-1", 3) != first
