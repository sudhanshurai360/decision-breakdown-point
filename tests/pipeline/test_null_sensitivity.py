"""null_sensitivity.py's own logic (`panel`'s rho/ar mixing formula, plus the `_ar1` recursion it
imports from election_null_ar1,
`null_dist`'s NaN filtering). The cross-check gate only re-derives canonical.json from this file's
own CSV output; nothing else exercises the mixing/filtering math itself against a known-correct
answer.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import null_sensitivity as NS  # noqa: E402
import fragility as F  # noqa: E402


def test_ar1_with_ar_zero_returns_input_unchanged():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 40)
    assert np.array_equal(NS._ar1(x, 0.0), x)


def test_panel_at_rho_zero_ar_zero_is_iid_mean1_sd_module_constant():
    """panel(rho=0, ar=0) reduces to K independent N(1, SD) draws (idio only, no shared/serial
    component) -- pins the documented base case directly, at a large T so Monte Carlo noise is
    small."""
    g = np.random.default_rng(1)
    T_saved = NS.T
    NS.T = 200_000
    try:
        L = NS.panel(g, rho=0.0, ar=0.0)
        assert set(L.keys()) == {f"m{i}" for i in range(NS.K)}
        for m, series in L.items():
            assert abs(series.mean() - 1.0) < 0.01, f"{m}: mean={series.mean():.4f}"
            assert abs(series.std() - NS.SD) < 0.01, f"{m}: std={series.std():.4f}, expected ~{NS.SD}"
    finally:
        NS.T = T_saved


def test_panel_models_are_more_correlated_at_higher_rho():
    """Sanity check on the mixing formula's actual behavior: pairwise correlation between two
    models' series should increase with rho (more shared common component), confirming `rho`
    actually controls what its name says it does."""
    T_saved = NS.T
    NS.T = 50_000
    try:
        corrs = []
        for rho in (0.0, 0.3, 0.6, 0.9):
            g = np.random.default_rng(2)
            L = NS.panel(g, rho=rho, ar=0.0)
            c = np.corrcoef(L["m0"], L["m1"])[0, 1]
            corrs.append(c)
            assert abs(c - rho) < 0.03, f"rho={rho}: empirical pairwise corr={c:.4f}"
        assert corrs == sorted(corrs)
    finally:
        NS.T = T_saved


def test_panel_argmin_structure_is_invariant_to_rho_at_a_fixed_seed():
    """THE CORE CLAIM this whole module exists to demonstrate, checked at the mechanism level
    rather than only empirically on the aggregate p05 statistic: a per-period shock common to
    every model (`sqrt(rho)*common`) shifts every model's loss by the SAME amount in that period,
    so it cannot change which model has the smallest loss in any period, nor which model has the
    smallest POOLED mean loss (the common term sums to an identical constant added to every
    model's total) -- regardless of rho. At a FIXED seed (so `common`/`idio_i` draws are identical
    across calls), per-period argmin and the pooled winner must therefore be EXACTLY the same
    across every rho, even though the raw loss values themselves differ completely."""
    T_saved = NS.T
    NS.T = 30
    try:
        w = np.ones(NS.T)
        pooled_winners = set()
        argmin_sequences = set()
        for rho in (0.0, 0.3, 0.5, 0.8, 0.99):
            g = np.random.default_rng(777)  # fresh, identically-seeded generator each call
            L = NS.panel(g, rho=rho, ar=0.0)
            pooled_winners.add(F.pooled_winner(L, w))
            A = np.column_stack([L[m] for m in sorted(L)])
            argmin_sequences.add(tuple(np.argmin(A, axis=1).tolist()))
        assert len(pooled_winners) == 1, f"pooled winner changed across rho: {pooled_winners}"
        assert len(argmin_sequences) == 1, "per-period argmin sequence changed across rho"
    finally:
        NS.T = T_saved


def test_panel_winner_stability_is_bit_identical_across_rho_at_a_fixed_seed():
    """The strongest form of the invariance claim: not just the pooled winner and per-period
    argmin, but winner_stability()'s full bootstrap output must be bit-identical across rho at a
    fixed seed -- since winner_stability only ever resamples periods and recomputes pooled argmin
    on the resampled set, and that computation is itself rho-invariant (see the test above), the
    entire bootstrap distribution must be too. This is a genuine mathematical identity -- argmin is
    invariant to a shift applied to every model in a period, so the statistic cannot see the common
    component at all (main()'s own emit spec for nullsens.p05_rho08 makes the same point) -- not an
    empirical coincidence -- checked directly here rather than only via the aggregate p05 figure
    null_dist() reports over 400 replicates."""
    T_saved = NS.T
    NS.T = 30
    try:
        w = np.ones(NS.T)
        results = []
        for rho in (0.0, 0.5, 0.8):
            g = np.random.default_rng(555)
            L = NS.panel(g, rho=rho, ar=0.0)
            v, freq = F.winner_stability(L, w, n_boot=300, seed=999)
            results.append(v)
        assert len(set(results)) == 1, f"winner_stability differed across rho: {results}"
    finally:
        NS.T = T_saved


def test_null_dist_returns_valid_count_and_dist_when_no_degeneracy():
    """On a normal (non-degenerate) configuration, null_dist must return exactly N_REP valid
    replicates -- pins the happy path's count explicitly, so a future change to the NaN-filtering
    logic that silently dropped valid replicates would be caught. Reps/boot reduced (module
    constants, not params -- monkeypatched) purely to keep this test fast; the counting logic being
    pinned does not depend on their magnitude."""
    n_rep_saved, n_boot_saved = NS.N_REP, NS.N_BOOT
    NS.N_REP, NS.N_BOOT = 30, 100
    try:
        med, p05, p95, n = NS.null_dist(rho=0.0, ar=0.0)
        assert n == NS.N_REP
        assert 0.0 < p05 < med < p95 <= 1.0
    finally:
        NS.N_REP, NS.N_BOOT = n_rep_saved, n_boot_saved


def test_null_dist_filters_out_nan_replicates():
    """null_dist must EXCLUDE non-finite winner_stability draws from its reported count and
    statistics, not let a NaN propagate into median/percentile (which would silently poison every
    downstream figure). Monkeypatches fragility.winner_stability to return NaN on a fixed fraction
    of calls, independent of any real degenerate-panel trigger condition."""
    n_rep_saved, n_boot_saved = NS.N_REP, NS.N_BOOT
    NS.N_REP, NS.N_BOOT = 30, 100
    calls = {"n": 0}
    real_winner_stability = F.winner_stability

    def flaky(L, w, n_boot=None, seed=None):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            return float("nan"), {}
        return real_winner_stability(L, w, n_boot=n_boot, seed=seed)

    import unittest.mock as mock
    try:
        with mock.patch.object(F, "winner_stability", side_effect=flaky):
            med, p05, p95, n = NS.null_dist(rho=0.0, ar=0.0)
        assert n < NS.N_REP, "expected some replicates to be dropped for returning NaN"
        assert np.isfinite(med) and np.isfinite(p05) and np.isfinite(p95)
    finally:
        NS.N_REP, NS.N_BOOT = n_rep_saved, n_boot_saved
