"""Shared synthetic-panel builder for tests: test_permutation_null.py and
test_election_null_ar1.py each hand-copied this identical
`_panel(T, K, seed)` body (only the default T differed, and every real call site at both files
passes T= explicitly, so the differing default was never actually load-bearing). Leading
underscore keeps pytest from collecting this as a test module.
"""
import numpy as np


def synthetic_panel(T, K, seed):
    rng = np.random.default_rng(seed)
    models = [f"m{i}" for i in range(K)]
    base = rng.normal(1.0, 0.1, T)
    L = {m: base + rng.normal(0, 0.02, T) for m in models}
    return L, np.ones(T)
