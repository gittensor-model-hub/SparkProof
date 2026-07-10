from sparkproof.merkle import merkle_root


def test_merkle_root_is_stable():
    leaves = ["aa" * 32, "bb" * 32, "cc" * 32]
    assert merkle_root(leaves) == merkle_root(leaves)


def test_empty_merkle_root():
    assert len(merkle_root([])) == 64
