"""null_budget.py calibration sanity: the seed-spacing guard, the Student-t fallback table's key
lookup, and verdict()'s undetermined-on-overlap behavior -- the defect class (RSAFS: a hard verdict
printed from a gap smaller than the noise) this module exists to prevent.
"""
import sys

import numpy as np
import pytest

import null_budget as NB


def test_seeds_are_twenty_and_widely_spaced():
    assert len(NB.SEEDS) == 20
    NB._check_spacing(NB.SEEDS, NB.N_REP)  # must not raise


def test_check_spacing_raises_on_close_seeds():
    with pytest.raises(ValueError):
        NB._check_spacing((0, 1, 2, 3), NB.N_REP)


@pytest.mark.parametrize("n_seeds,expected_q", [
    (2, 12.706), (3, 4.303), (5, 2.776), (10, 2.262), (20, 2.093), (21, 2.086),
])
def test_band_student_t_fallback_table(n_seeds, expected_q, monkeypatch):
    # band() imports scipy.stats INSIDE the function and falls back to a hardcoded Student-t table
    # on any exception. scipy is pinned in this environment, so the fallback is dead code unless it
    # is forced: poisoning sys.modules["scipy.stats"] with None makes `from scipy.stats import t`
    # raise ImportError, which is what band()'s `except Exception` is there to catch. Without this
    # the parametrized expectations below never touch the table at all.
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    sd = 1.0
    got = NB.band(sd, n_seeds)
    q_table = got / (sd / (n_seeds ** 0.5))
    assert q_table == pytest.approx(expected_q, abs=0.001)

    # and the table entry must agree with the scipy quantile it stands in for, so the fallback is
    # not silently a different (looser or tighter) band than the primary path.
    monkeypatch.undo()
    q_scipy = NB.band(sd, n_seeds) / (sd / (n_seeds ** 0.5))
    assert q_table == pytest.approx(q_scipy, abs=0.001)


def test_band_scales_with_sd_and_shrinks_with_more_seeds():
    b_few = NB.band(1.0, 3)
    b_many = NB.band(1.0, 20)
    assert b_many < b_few
    assert NB.band(2.0, 10) == pytest.approx(2 * NB.band(1.0, 10))


def test_verdict_fragile_when_obs_clearly_below_null():
    assert NB.verdict(obs=0.3, null=0.9, obs_band=0.02, null_band=0.02) == "fragile"


def test_verdict_not_fragile_when_obs_clearly_above_null():
    assert NB.verdict(obs=0.95, null=0.5, obs_band=0.02, null_band=0.02) == "not_fragile"


def test_verdict_undetermined_when_bands_touch():
    # this is the exact RSAFS defect class: a small gap (0.0024) inside band-noise must NOT resolve
    # to a hard verdict.
    assert NB.verdict(obs=0.5275, null=0.5299, obs_band=0.01, null_band=0.01) == "undetermined"


def test_stable_benchmark_reports_n_eff_and_spread():
    # small n_rep/n_boot for test speed; only checks the bookkeeping, not statistical precision.
    seeds = (0, 20_000_000, 40_000_000)
    out = NB.stable_benchmark(K=3, T=10, n_rep=20, n_boot=20, seeds=seeds)
    assert out["n_eff"] == 3 * 20
    assert out["n_seeds"] == 3
    assert out["p05_lo"] <= out["p05"] <= out["p05_hi"]


def test_stable_stability_reports_n_seeds_and_n_boot():
    L = {"a": np.array([1.0, 2.0, 1.5, 1.2, 1.8]), "b": np.array([2.0, 1.0, 1.6, 1.9, 1.1])}
    w = np.ones(5)
    out = NB.stable_stability(L, w, n_boot=50, seeds=(0, 1))
    assert out["n_seeds"] == 2
    assert out["n_boot"] == 50
    assert 0.0 <= out["value"] <= 1.0
