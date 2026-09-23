"""Regression: `crowd()`'s per-k loop
used to `break` on the first invalid (too-large) k instead of `continue`-ing past it, silently
assuming `ks` is passed pre-sorted ascending. Both real call sites in arena_crowding.py pass hardcoded
ascending literals, so the bug was inert on every real run -- but undocumented, and a future
reordering of `ks` would have silently dropped valid pool sizes with no error.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import arena_crowding as AC


def test_a_larger_invalid_k_followed_by_a_smaller_valid_k_still_computes_the_valid_one():
    """THE CORE REGRESSION: crowd(L, rng, [8, 2]) on a 4-model array used to return {} -- the loop
    broke on k=8 (> len(L)=4) before ever reaching the valid k=2."""
    rng = np.random.default_rng(0)
    L = np.random.default_rng(1).normal(size=(4, 3))
    out = AC.crowd(L, rng, [8, 2])
    assert 2 in out, "valid k=2 was dropped because it came after an invalid, larger k"
    assert 8 not in out, "k=8 exceeds the pool size and must still be skipped"


def test_ascending_sorted_ks_still_works_exactly_as_before():
    """Both real call sites pass ascending-sorted ks -- confirms the fix is a no-op there."""
    rng = np.random.default_rng(0)
    L = np.random.default_rng(1).normal(size=(6, 3))
    out = AC.crowd(L, rng, [2, 4, 6])
    assert set(out.keys()) == {2, 4, 6}


def test_every_k_larger_than_the_pool_is_skipped_not_just_the_first():
    rng = np.random.default_rng(0)
    L = np.random.default_rng(1).normal(size=(3, 3))
    out = AC.crowd(L, rng, [10, 20, 2])
    assert set(out.keys()) == {2}
