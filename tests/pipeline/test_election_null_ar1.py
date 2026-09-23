"""election_null_ar1.py's AR(1)-calibrated null machinery (`_ar1`, `estimate_ar1_coefficients`,
`panel_ar1`, `stable_benchmark_ar1`) and social_choice.py's `ar1_null()`. This is the paper's
more-trusted calibration for Section 5 -- the within-period null "has no power to detect anything"
while "the AR(1)-calibrated null uses fresh synthetic draws instead, and does have real power"
(draft_ssrn.md §5.0 "Calibration: what an arbitrary winner produces", the location claims.py
indexes as S5.0 -- not Section 5's untitled preamble, and not the later S5.4 subsection) -- and it is more
complex than `permutation_null()`/`_cell_stats()` (covered
in tests/pipeline/test_permutation_null.py). These tests exercise the four functions directly on
small or hand-constructed inputs with known-in-advance answers, independent of the real curated
data.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import pandas as pd
import election_null_ar1 as EAR1  # noqa: E402
import social_choice as SC  # noqa: E402
# Shared with test_permutation_null.py -- one implementation, imported by both.
from _synthetic_panel import synthetic_panel as _panel  # noqa: E402


def test_ar1_with_ar_zero_returns_input_unchanged():
    """`_ar1(x, ar=0)` (and any ar<=0) is documented as a no-op passthrough -- pins that branch
    directly rather than trusting it never regresses."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    out = EAR1._ar1(x, 0.0)
    assert np.array_equal(out, x)
    out_neg = EAR1._ar1(x, -0.3)
    assert np.array_equal(out_neg, x)


def test_ar1_preserves_unit_variance_across_coefficients():
    """The recursion y[t] = ar*y[t-1] + sqrt(1-ar^2)*x[t] is an EXACT unit-variance-preserving
    transform by induction (Var(y[t]) = ar^2*Var(y[t-1]) + (1-ar^2)*Var(x[t]) = 1 whenever both
    inputs have unit variance) -- not just asymptotically. A single long series' time-average
    variance should converge close to 1 for every ar tested, confirming the recursion itself (not
    just its stationary distribution in theory) behaves as documented."""
    rng = np.random.default_rng(1)
    T = 200_000
    for ar in (0.0, 0.3, 0.6, 0.9):
        x = rng.normal(0, 1, T)
        y = EAR1._ar1(x, ar)
        assert abs(y.mean()) < 0.02, f"ar={ar}: mean drifted to {y.mean():.4f}"
        assert abs(y.std() - 1.0) < 0.02, f"ar={ar}: std drifted to {y.std():.4f}, expected ~1.0"


def test_ar1_higher_coefficient_gives_higher_lag1_autocorrelation():
    """Sanity check on the recursion's actual dependence structure: empirical lag-1 autocorrelation
    of the OUTPUT series should increase monotonically with the `ar` parameter -- otherwise the
    whole calibration exercise (estimate a coefficient, feed it back into this same generator) is
    measuring something that doesn't track what it's supposed to control."""
    rng = np.random.default_rng(2)
    T = 50_000
    acfs = []
    for ar in (0.0, 0.3, 0.6, 0.9):
        y = EAR1._ar1(rng.normal(0, 1, T), ar)
        acf = np.corrcoef(y[:-1], y[1:])[0, 1]
        acfs.append(acf)
        assert abs(acf - ar) < 0.03, f"ar={ar}: empirical lag-1 acf={acf:.4f}, expected ~{ar}"
    assert acfs == sorted(acfs)


def _synthetic_ar1_frame(true_ar, T=5000, seed=3, series="S", model="M"):
    """A `d`-shaped DataFrame (series/model/year/sae/msse) matching what estimate_ar1_coefficients
    expects, built from EAR1._ar1 itself at a KNOWN coefficient -- so the OLS estimator's recovered
    value can be checked against a value fixed by construction, not derived from the function under
    test. Mean shifted well above EAR1.SD's scale (real loss values -- sae/msse -- are non-negative
    by construction; panel_ar1's own mean=1.0 convention would occasionally dip negative at this
    SD and T, feeding rmsse's sqrt() a negative mean for a rare year -- harmless to the estimate
    itself (that year is dropped via dropna()) but an avoidable artifact in a purpose-built panel)."""
    rng = np.random.default_rng(seed)
    y = 10.0 + EAR1.SD * EAR1._ar1(rng.normal(0, 1, T), true_ar)
    return pd.DataFrame({
        "series": series, "model": model, "year": np.arange(T),
        "sae": y, "msse": y,
    })


def test_estimate_ar1_coefficients_levels_recovers_a_known_coefficient():
    """The demeaned lag-1 OLS estimator (per-model LEVEL series -- kept as
    estimate_ar1_coefficients_levels for disclosure) must recover a coefficient
    fixed by construction, within sampling noise, at a large T where small-T bias is negligible."""
    for true_ar in (0.2, 0.5, 0.8):
        d = _synthetic_ar1_frame(true_ar, T=5000, seed=hash(true_ar) % 1000)
        coeffs = EAR1.estimate_ar1_coefficients_levels(d)
        assert len(coeffs) == 2, "expected one row per metric (mase, rmsse) for this single series/model"
        for phi in coeffs["ar1_estimate"]:
            assert abs(phi - true_ar) < 0.03, f"true_ar={true_ar}: estimated {phi:.4f}"


def test_estimate_ar1_coefficients_levels_near_zero_for_white_noise():
    """Negative control: i.i.d. (ar=0) input should recover an estimate close to 0, not spuriously
    positive or negative."""
    d = _synthetic_ar1_frame(0.0, T=5000, seed=42)
    coeffs = EAR1.estimate_ar1_coefficients_levels(d)
    assert (coeffs["ar1_estimate"].abs() < 0.03).all()


def _two_model_frame(true_ar, T=5000, seed=3, series="S"):
    """Two INDEPENDENT AR(1)(true_ar) models in one cell -- their differential is AR(1)(true_ar) too."""
    rng = np.random.default_rng(seed)
    parts = []
    for m in ("A", "B"):
        y = 10.0 + EAR1.SD * EAR1._ar1(rng.normal(0, 1, T), true_ar)
        parts.append(pd.DataFrame({"series": series, "model": m, "year": np.arange(T), "sae": y, "msse": y}))
    return pd.concat(parts, ignore_index=True)


def test_estimate_ar1_coefficients_uses_pairwise_differentials_and_recovers_the_coefficient():
    """The calibration estimator runs on
    pairwise loss DIFFERENTIALS. For two independent AR(1)(phi) models the differential is itself
    AR(1)(phi), so the estimator must recover phi -- and must return one row per (metric, pair)
    with a `pair` label, not a `model` label."""
    for true_ar in (0.2, 0.5, 0.8):
        d = _two_model_frame(true_ar, T=5000, seed=hash(true_ar) % 1000 + 7)
        coeffs = EAR1.estimate_ar1_coefficients(d)
        assert len(coeffs) == 2 and set(coeffs["pair"]) == {"A-B"}
        for phi in coeffs["ar1_estimate"]:
            assert abs(phi - true_ar) < 0.03, f"true_ar={true_ar}: estimated {phi:.4f}"


def test_estimate_ar1_coefficients_differentials_ignore_a_common_component():
    """THE POINT OF THE CHANGE: a strongly persistent series added to EVERY model (a common
    per-period component, exactly what dominates the real panel's loss levels) must not move the
    differential estimate at all, while it drives the level estimate up."""
    d = _two_model_frame(0.0, T=5000, seed=11)
    rng = np.random.default_rng(99)
    common = 3.0 * EAR1._ar1(rng.normal(0, 1, 5000), 0.9)
    d2 = d.copy()
    for m in ("A", "B"):
        idx = d2.model == m
        d2.loc[idx, "sae"] = d2.loc[idx, "sae"].to_numpy() + common
        d2.loc[idx, "msse"] = d2.loc[idx, "msse"].to_numpy() + common
    # exact cancellation holds on the per-year LOSS MATRIX; the mase metric's per-year aggregate is
    # a mean (linear), so adding the component at the row level cancels exactly there. The rmsse
    # metric aggregates through a sqrt (nonlinear), so row-level addition is only approximately
    # common at the per-year level -- the invariance argument is about the loss matrix itself.
    def _mase_median(df):
        c = EAR1.estimate_ar1_coefficients(df)
        return c.loc[c.metric == "mase", "ar1_estimate"].median()
    assert abs(_mase_median(d2) - _mase_median(d)) < 1e-9, "a common component must cancel exactly in the differential"
    lev = EAR1.estimate_ar1_coefficients_levels(d2)
    assert lev.loc[lev.metric == "mase", "ar1_estimate"].median() > 0.8, "the same common component dominates the level estimate"


def test_panel_ar1_output_matches_documented_mean_and_dispersion():
    """panel_ar1's own docstring promises "K independent, mean-1.0 loss series" with dispersion
    SD (the module constant EAR1.SD=0.30) -- pins both directly, at a large T so Monte Carlo noise
    is small, since _ar1's unit-variance guarantee (tested above) is what this claim rests on."""
    g = np.random.default_rng(4)
    T = 200_000
    L = EAR1.panel_ar1(g, K=3, T=T, ar=0.5)
    assert set(L.keys()) == {"m0", "m1", "m2"}
    for m, series in L.items():
        assert len(series) == T
        assert abs(series.mean() - 1.0) < 0.01, f"{m}: mean={series.mean():.4f}, expected ~1.0"
        assert abs(series.std() - EAR1.SD) < 0.01, f"{m}: std={series.std():.4f}, expected ~{EAR1.SD}"


def test_stable_benchmark_ar1_is_deterministic():
    """Same (K, T, ar) + same seeds must reproduce byte-identical results -- this is what makes the
    published AR(1)-adjusted threshold reproducible at all. Uses small n_rep/n_boot and widely-spaced
    seeds (per null_budget._check_spacing's own overlap requirement) to keep the test fast."""
    seeds = (0, 5_000_000)
    r1 = EAR1.stable_benchmark_ar1(K=4, T=20, ar=0.4, n_rep=5, n_boot=100, seeds=seeds)
    r2 = EAR1.stable_benchmark_ar1(K=4, T=20, ar=0.4, n_rep=5, n_boot=100, seeds=seeds)
    assert r1 == r2


def test_stable_benchmark_ar1_null_is_non_degenerate():
    """The null's p05 must sit strictly below its median (i.e. the winner_stability distribution
    under this null has real spread, not a degenerate point mass). A degenerate null would make
    every "below the null's fifth percentile" verdict in Section 5 vacuous while still running
    clean, so the spread is asserted here rather than left to inspection."""
    r = EAR1.stable_benchmark_ar1(K=4, T=20, ar=0.4, n_rep=5, n_boot=100, seeds=(0, 5_000_000))
    assert 0.0 < r["p05"] < r["median"] <= 1.0
    assert r["p05_sd"] >= 0.0


def test_stable_benchmark_ar1_higher_ar_raises_the_null_p05():
    """Higher serial dependence should make the exchangeable null's winner MORE stable by chance
    (less period-to-period churn to overturn it), raising p05 -- this monotonic direction is
    exactly the S5.4 claim that "serial dependence... does move the threshold: from 0.297 under
    i.i.d. to 0.351 at AR(1)=0.6". Same seeds/n_rep/n_boot across both calls isolates the `ar`
    effect."""
    seeds = (0, 5_000_000)
    low = EAR1.stable_benchmark_ar1(K=4, T=20, ar=0.1, n_rep=8, n_boot=200, seeds=seeds)
    high = EAR1.stable_benchmark_ar1(K=4, T=20, ar=0.8, n_rep=8, n_boot=200, seeds=seeds)
    assert high["p05"] > low["p05"], (
        f"expected higher AR(1) to raise the null's p05: ar=0.1 -> {low['p05']:.4f}, "
        f"ar=0.8 -> {high['p05']:.4f}"
    )


def test_ar1_null_is_deterministic():
    """Same cells + same (ar_hat, seed) must reproduce byte-identical results, mirroring
    test_permutation_null.py's identical determinism check for the sibling null."""
    cells = [_panel(T=6, K=3, seed=i) for i in range(3)]
    obs1, null1 = SC.ar1_null(cells, ar_hat=0.4, n_perm=20, seed=777)
    obs2, null2 = SC.ar1_null(cells, ar_hat=0.4, n_perm=20, seed=777)
    assert np.array_equal(obs1, obs2)
    assert np.array_equal(null1, null2)


def test_ar1_null_observed_matches_direct_cell_stats_sum():
    """obs must equal independently summing _cell_stats() across the same cells directly -- pins
    that ar1_null(), like permutation_null(), does not mutate `cells` or double-count."""
    cells = [_panel(T=7, K=4, seed=i) for i in range(4)]
    obs, _ = SC.ar1_null(cells, ar_hat=0.4, n_perm=10, seed=1)
    direct = np.sum([SC._cell_stats(L, w) for L, w in cells], axis=0)
    assert np.array_equal(obs, direct.astype(int))


def test_ar1_null_shape_and_valid_range():
    """null array must be (n_perm, 5) with every entry a valid count in [0, n_cells], matching
    permutation_null()'s own shape contract exactly (same interface, different null distribution)."""
    n_cells = 5
    cells = [_panel(T=6, K=3, seed=i) for i in range(n_cells)]
    obs, null = SC.ar1_null(cells, ar_hat=0.5, n_perm=15, seed=3)
    assert obs.shape == (5,)
    assert null.shape == (15, 5)
    assert (null >= 0).all() and (null <= n_cells).all()
    assert (obs >= 0).all() and (obs <= n_cells).all()


def test_ar1_null_uses_each_cells_own_k_and_t():
    """ar1_null draws synthetic panels at each cell's OWN (K, T) (per its docstring), not a single
    shared shape -- construct cells with deliberately different (K, T) and confirm it runs without
    shape errors and the null retains one row of counts per permutation regardless."""
    cells = [_panel(T=6, K=2, seed=0), _panel(T=10, K=5, seed=1), _panel(T=25, K=3, seed=2)]
    obs, null = SC.ar1_null(cells, ar_hat=0.3, n_perm=8, seed=9)
    assert obs.shape == (5,)
    assert null.shape == (8, 5)
