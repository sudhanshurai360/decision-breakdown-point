"""Regression test: mcs_test.py::mcs() used to fall through T=0 (a per-period loss matrix with zero evaluation periods)
to an opaque `IndexError: arrays used as indices must be of integer (or boolean) type` deep inside
the bootstrap loop -- `_block_idx(T=0, block, rng)` returns an empty FLOAT64 array (there are no
elements to infer an int dtype from), which then fails numpy fancy-indexing at `sub[idx]`.

T=1 and T=2 are already handled gracefully via the existing MIN_T_FOR_ZERO_SE degenerate-SE-floor
machinery (both return the conservative "everyone tied" verdict, no error); T=0 alone was missed.
Confirmed genuinely latent against every real caller today -- this test locks in the fix (a named
ValueError instead of an opaque crash), it does not indicate T=0 is actually reachable on real data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import mcs_test as MT  # noqa: E402


def test_zero_periods_raises_a_named_error_not_an_opaque_indexerror():
    with pytest.raises(ValueError, match="T=0"):
        MT.mcs(np.empty((0, 3)), alpha=0.10, seed=0)


def test_one_period_still_works_unaffected_by_the_t0_guard():
    """Sanity check the fix didn't accidentally raise the floor above T=0 -- T=1 (already
    guarded by MIN_T_FOR_ZERO_SE machinery, tested in test_mcs_test_degenerate_se.py) must still
    return normally, not raise."""
    L = np.array([[1.0, 2.0, 3.0]])
    surv, p = MT.mcs(L, alpha=0.10, B=50, seed=0)
    assert isinstance(surv, list) and len(surv) >= 1
