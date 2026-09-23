"""model_set_audit.py -- how similar are the six candidate models, really. This producer emits 14
canonical `modelset.*` values cited directly in S3.1's "how similar are the candidate models"
discussion; without these tests its only coverage is indirect, via `canonical.py --check`
re-running the whole module.

These tests independently RECOMPUTE a subset of the module's own statistics directly from the real
`breadth_distinct_predictions.csv` (the same source file the module reads), using the module's own
constants (ALPHA/B/BLOCK/SEED/BENCHMARK) but a fresh, from-scratch computation -- not by re-running
`main()` and diffing its output -- and cross-check the result against the committed
`multiseries/results/model_set_audit.csv`. Skips gracefully if the real data isn't available (a
clean checkout without the tracked prediction cache, or a CI environment).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_set_audit as MSA
import mcs_test as MT

ROOT = Path(__file__).resolve().parent.parent.parent
PRED = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
AUDIT_CSV = ROOT / "multiseries" / "results" / "model_set_audit.csv"


@pytest.fixture(scope="module")
def real_predictions():
    try:
        d = pd.read_csv(PRED)
    except FileNotFoundError:
        pytest.skip("real breadth_distinct_predictions.csv not available")
    d["year"] = pd.to_datetime(d.target).dt.year
    return d


@pytest.fixture(scope="module")
def real_audit_csv():
    try:
        return pd.read_csv(AUDIT_CSV).set_index("series")
    except FileNotFoundError:
        pytest.skip("real model_set_audit.csv not available")


def test_gap_pct_recomputed_independently_matches_committed_csv(real_predictions, real_audit_csv):
    """Recompute the best-vs-runner-up gap for one series directly (equal-year weighting, matching
    S5/Table 1's convention) and confirm it matches the committed CSV -- not via re-running main(),
    a genuinely independent recomputation of the same formula."""
    d = real_predictions
    for sid in ("BOPGEXP", "RSAFS", "PI"):
        g = d[d.series == sid]
        yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
        means = yl.mean(0).sort_values()
        best, runner = means.index[0], means.index[1]
        gap_pct = float((means.iloc[1] - means.iloc[0]) / means.iloc[0] * 100)
        row = real_audit_csv.loc[sid]
        assert best == row["best"], f"{sid}: best model mismatch"
        assert runner == row["runner_up"], f"{sid}: runner-up mismatch"
        assert gap_pct == pytest.approx(row["gap_pct"], abs=5e-3), f"{sid}: gap_pct mismatch"


def test_benchmark_is_last_in_eleven_of_twelve_series(real_predictions, real_audit_csv):
    """The module's own headline claim (S3.1): seasonal_naive is last (rank 6 of 6) in 11 of the 12
    curated series, RSAFS being the one exception. Recomputed independently from real predictions,
    not read off the committed CSV alone."""
    d = real_predictions
    n_last, series_count = 0, 0
    for sid, g in d.groupby("series"):
        series_count += 1
        yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
        means = yl.mean(0).sort_values()
        rank = list(means.index).index(MSA.BENCHMARK) + 1
        n_last += (rank == len(means))
        assert rank == real_audit_csv.loc[sid, "benchmark_rank"], f"{sid}: benchmark_rank mismatch"
    assert series_count == 12
    assert n_last == 11
    assert real_audit_csv.loc["RSAFS", "benchmark_rank"] == 1   # the one series where it's NOT last


def test_pairwise_loss_correlation_among_non_benchmark_models_is_high(real_predictions):
    """S3.1's near-collinearity claim: the five non-benchmark models' annual loss series correlate
    at a median near 0.99. Recomputed directly (not read off the CSV) with the module's own
    BENCHMARK exclusion."""
    import itertools
    d = real_predictions
    corrs = []
    for sid, g in d.groupby("series"):
        yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
        others = [m for m in yl.columns if m != MSA.BENCHMARK]
        corrs += [float(yl[a].corr(yl[b])) for a, b in itertools.combinations(sorted(others), 2)]
    assert np.median(corrs) > 0.95
    assert min(corrs) > 0.5   # even the least-correlated real pair is still substantially correlated


def test_elimination_rule_gives_identical_surviving_sets_on_a_real_series(real_predictions):
    """S4.1's tested claim: raw-mean vs studentized elimination give the IDENTICAL surviving MCS on
    every curated series, not merely the same cardinality. Directly exercises mcs_test.MT.mcs (real
    code under test, not reimplemented) on one real series rather than trusting the aggregate count
    in the committed CSV."""
    d = real_predictions
    g = d[d.series == "BOPGEXP"]
    yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
    models, X = list(yl.columns), yl.to_numpy(float)
    a, _ = MT.mcs(X, alpha=MSA.ALPHA, B=MSA.B, block=MSA.BLOCK, seed=MSA.SEED, elimination="raw")
    b, _ = MT.mcs(X, alpha=MSA.ALPHA, B=MSA.B, block=MSA.BLOCK, seed=MSA.SEED, elimination="studentized")
    assert sorted(models[i] for i in a) == sorted(models[i] for i in b)


def test_e_R_rule_differs_from_raw_on_bopgexp_and_reproduces_archs_set(real_predictions):
    """The test above compares raw against e_max -- HLN's rule
    for T_max, which neither this file's mcs() (T_R) nor arch uses. Under HLN's T_R-paired rule e_R
    (what arch's MCS(method='R') eliminates) BOPGEXP's surviving set is {ar1, ets, sarima, theta},
    one model wider than raw's {ar1, sarima, theta} -- and identical to arch's set for this series.
    This is the series on which the paper's old 'rules out the elimination rule' claim was wrong."""
    d = real_predictions
    g = d[d.series == "BOPGEXP"]
    yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
    models, X = list(yl.columns), yl.to_numpy(float)
    raw, _ = MT.mcs(X, alpha=MSA.ALPHA, B=MSA.B, block=MSA.BLOCK, seed=MSA.SEED, elimination="raw")
    e_R, _ = MT.mcs(X, alpha=MSA.ALPHA, B=MSA.B, block=MSA.BLOCK, seed=MSA.SEED, elimination="e_R")
    assert sorted(models[i] for i in raw) == ["ar1", "sarima", "theta"]
    assert sorted(models[i] for i in e_R) == ["ar1", "ets", "sarima", "theta"]


def test_e_R_rule_eliminates_the_losing_side_of_the_pair_attaining_T_R():
    """Synthetic: model 2 is dramatically worse than model 0 in every period (the pair attaining
    T_R), while model 1 has the largest MEAN loss only because of one outlier period. Raw drops
    model 1 first (largest mean); e_R must drop model 2 (the loser of the T_R pair)."""
    rng = np.random.default_rng(0)
    T = 40
    m0 = rng.normal(1.0, 0.05, T)
    m2 = m0 + 0.5 + rng.normal(0, 0.02, T)          # consistently worse -> huge t against m0
    m1 = m0 + rng.normal(0, 0.05, T); m1[0] += 60.0   # largest mean, but from one outlier
    L = np.column_stack([m0, m1, m2])
    assert L.mean(0).argmax() == 1
    surv_raw, _ = MT.mcs(L, alpha=0.10, B=500, block=3, seed=0, elimination="raw")
    surv_eR, _ = MT.mcs(L, alpha=0.10, B=500, block=3, seed=0, elimination="e_R")
    # raw drops model 1 first (largest mean), then model 2 -> only model 0 survives; e_R drops model 2
    # first (the T_R pair's loser) and then cannot separate 0 from 1 (model 1's single outlier gives
    # the pair a huge bootstrap se) -> {0, 1} survive. The two rules must give DIFFERENT sets here,
    # or the test is not exercising the rule at all.
    assert surv_raw == [0], f"raw: expected only model 0 to survive, got {surv_raw}"
    assert surv_eR == [0, 1], f"e_R: expected models 0 and 1 to survive, got {surv_eR}"


def test_module_constants_match_the_manuscripts_stated_mcs_settings():
    """Lock in ALPHA/B/BLOCK/SEED against silent drift -- every modelset.* emit()'s spec string
    cites these by value, so a change here would desync the canonical spec text from the real
    computation without any test catching it otherwise."""
    assert MSA.ALPHA == 0.10
    assert MSA.B == 2000
    assert MSA.BLOCK == 3
    assert MSA.SEED == 0
    assert MSA.BENCHMARK == "seasonal_naive"
