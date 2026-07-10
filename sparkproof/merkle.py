"""Simple binary Merkle tree over hex leaf digests."""

from __future__ import annotations

import hashlib


def _pair_hash(left: str, right: str) -> str:
    data = bytes.fromhex(left) + bytes.fromhex(right)
    return hashlib.sha256(data).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_pair_hash(level[i], level[i + 1]))
            else:
                nxt.append(_pair_hash(level[i], level[i]))
        level = nxt
    return level[0]
