"""mcs_circular_check.py's own
purpose is to re-run the headline S4.1/S4.2 MCS cross-checks with `circular=True` instead of the
default moving-block bootstrap, so a bug here would silently corrupt a diagnostic the paper cites
for whether its own stated "bootstrap scheme explains the arch discrepancy" hypothesis holds.

Three pure/near-pure functions are unit-tested directly, with small synthetic panels chosen so the
expected answer is either exactly hand-derivable (constructed ties / strict dominance, which force
`mcs_test.mcs()`'s degenerate-se certain-rejection/genuine-tie branches deterministically,
independent of bootstrap randomness -- verified by first reasoning through mcs_test.py's own
elimination logic, then confirmed by direct execution) or checked via a monkeypatched spy on
`mcs_test.mcs` that never re-implements or re-uses the function under test:

  - size_rate(circular, n_rep, seed): the SE-formula/determinism checks mirror
    tests/pipeline/test_mcs_size_level.py's already-established pattern for the sibling
    mcs_size.py::size_row. The parity test (moving-block and circular calls draw the IDENTICAL
    sequence of null panels for the same seed, and the `circular` flag is correctly threaded
    through to mcs_test.mcs) is this file's own novel contract, tested via a spy stub -- not by
    calling the real (expensive) mcs() twice and hoping the two sequences happen to match.
  - curated_band(losses, circular, seeds): ground-truthed with an exactly-tied synthetic panel
    (must NEVER point-identify, any seed, either bootstrap scheme) and a strictly-ordered panel
    (must ALWAYS point-identify) -- plus a separate aggregation-only spy test that stubs mcs_test.mcs
    to return known survivor-set sizes, to check curated_band's own mean/min/max/n arithmetic in
    isolation from mcs_test.py's real statistical behavior.
  - three_series_cross_check(losses): exercises the REAL `arch` third-party library (matching this
    project's convention elsewhere, e.g. mcs_size.py's own arch_cross_check) on a synthetic 3-series
    panel with one model given an overwhelming, low-noise advantage -- all three arms (moving-block,
    circular, arch) should agree on the singleton winner.

None of these tests call main() (avoids real file I/O, real emit()/canonical registry writes, and
the real, much larger N_SIZE=2000/B=2000 pipeline-scale replication counts).
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest

import mcs_circular_check as MCC
import mcs_test as MT


# --- size_rate --------------------------------------------------------------------------------

def test_size_rate_returns_valid_rate_and_se_matching_binomial_formula():
    rate, se = MCC.size_rate(circular=False, n_rep=30, seed=12345)
    assert 0.0 <= rate <= 1.0
    assert se == pytest.approx(float(np.sqrt(rate * (1 - rate) / 30)))


def test_size_rate_deterministic_given_seed():
    rate1, se1 = MCC.size_rate(circular=True, n_rep=25, seed=999)
    rate2, se2 = MCC.size_rate(circular=True, n_rep=25, seed=999)
    assert rate1 == rate2
    assert se1 == se2


def test_size_rate_edge_case_single_model_never_rejects(monkeypatch):
    """Documented edge case: with a single model (K=1), 'at least one model
    eliminated' (len(surv) < K) can never fire -- len(surv) is always exactly 1, and 1 < 1 is
    False -- so the rate must be EXACTLY 0.0 (and se exactly 0.0), for every seed, regardless of
    the bootstrap scheme. A hand-derivable edge case, not a statistical approximation."""
    monkeypatch.setattr(MCC, "K_PAPER", 1)
    monkeypatch.setattr(MCC, "T_PAPER", 10)
    monkeypatch.setattr(MCC, "BLOCK_PAPER", 1)
    for circular in (False, True):
        rate, se = MCC.size_rate(circular=circular, n_rep=8, seed=0)
        assert rate == 0.0
        assert se == 0.0


def test_size_rate_moving_block_and_circular_see_identical_null_panel_draws_and_flag_is_wired_through(monkeypatch):
    """size_rate's fairness contract: the SAME sequence of null panels must be drawn regardless of
    `circular`, so any difference between the two arms' rates is attributable to the resampling
    scheme alone, not to different data. Also confirms the `circular` kwarg this whole file exists
    to test is actually threaded through to mcs_test.mcs (not silently dropped or hardcoded), via a
    stub that never re-implements mcs()'s own logic -- it only records what it was called with."""
    recorded = []

    def stub(L, **kwargs):
        recorded.append((np.array(L, copy=True), kwargs.get("circular")))
        return list(range(L.shape[1])), 1.0

    monkeypatch.setattr(MT, "mcs", stub)

    recorded.clear()
    MCC.size_rate(circular=False, n_rep=5, seed=123)
    panels_false = [r[0] for r in recorded]
    flags_false = [r[1] for r in recorded]

    recorded.clear()
    MCC.size_rate(circular=True, n_rep=5, seed=123)
    panels_true = [r[0] for r in recorded]
    flags_true = [r[1] for r in recorded]

    assert flags_false == [False] * 5
    assert flags_true == [True] * 5
    assert len(panels_false) == len(panels_true) == 5
    for a, b in zip(panels_false, panels_true):
        assert np.array_equal(a, b), "moving-block and circular arms must see identical panel draws"


# --- curated_band -----------------------------------------------------------------------------

def _tied_panel(T=10, K=3, value=1.0):
    """All K models get the IDENTICAL constant loss every period: a genuine exact tie. mcs_test.mcs
    skips every pair (obs_diff == 0 exactly) and never eliminates anyone -- deterministically, for
    ANY seed and EITHER bootstrap scheme, since no bootstrap variability is ever consulted."""
    return np.full((T, K), value)


def _strictly_ordered_panel(T=10, K=3):
    """K models with strictly distinct, constant-per-model losses: a deterministic total order.
    Every pair has a real, non-zero, zero-variance deterministic gap -> mcs_test.mcs's degenerate-se
    certain-rejection branch fires every round -> the elimination proceeds all the way to the single
    best (lowest-loss) model, deterministically, for ANY seed and EITHER bootstrap scheme."""
    cols = [np.full(T, float(v)) for v in range(1, K + 1)]
    return np.column_stack(cols)


def test_curated_band_all_series_non_identified_when_models_are_exactly_tied():
    tied = _tied_panel()
    losses = {"A": (tied, None), "B": (tied, None), "C": (tied, None)}
    for circular in (False, True):
        mean, lo, hi, n = MCC.curated_band(losses, circular=circular, seeds=range(3))
        assert n == 3
        assert mean == 3.0
        assert lo == 3
        assert hi == 3


def test_curated_band_all_series_point_identified_when_models_are_strictly_ordered():
    strict = _strictly_ordered_panel()
    losses = {"A": (strict, None), "B": (strict, None), "C": (strict, None)}
    for circular in (False, True):
        mean, lo, hi, n = MCC.curated_band(losses, circular=circular, seeds=range(3))
        assert n == 3
        assert mean == 0.0
        assert lo == 0
        assert hi == 0


def test_curated_band_aggregation_arithmetic_in_isolation(monkeypatch):
    """Stubs mcs_test.mcs to return a KNOWN, fixed survivor-set size per series (identified by the
    constant value baked into each series' loss matrix, not by seed), decoupling curated_band's own
    mean/min/max/n bookkeeping from mcs_test.py's real statistical behavior. Series 'A' (value 0)
    and 'B' (value 1) are stubbed non-identified (survivor-set size 2); series 'C' (value 2) is
    stubbed identified (size 1) -- so each of the 3 seeds in `seeds` must independently compute
    exactly 2 non-identified series, giving mean=2.0, min=2, max=2 by hand."""

    def stub(L, **kwargs):
        val = float(L[0, 0])
        return (list(range(2)) if val < 1.5 else [0]), 1.0

    monkeypatch.setattr(MT, "mcs", stub)
    losses = {
        "A": (np.zeros((5, 3)), None),
        "B": (np.ones((5, 3)), None),
        "C": (np.full((5, 3), 2.0), None),
    }
    mean, lo, hi, n = MCC.curated_band(losses, circular=False, seeds=range(3))
    assert n == 3
    assert mean == 2.0
    assert lo == 2
    assert hi == 2


# --- three_series_cross_check (real arch dependency, small synthetic panels) ------------------

def _dominant_panel(T=15, K=3, seed=0):
    """Model 0 has an overwhelming, low-noise advantage every period (loss ~1.0 vs ~5.0 for the
    others, sd=0.05) -- not a deterministic exact tie/strict-order construction like the
    curated_band cases above, but a real, near-certain-to-resolve statistical signal, chosen so all
    three independent methods (our moving-block, our circular, third-party arch) should agree on
    the singleton winner with overwhelming probability."""
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 0.05, size=(T, K))
    base[:, 0] = rng.normal(1.0, 0.05, size=T)
    return base


def test_three_series_cross_check_all_three_methods_agree_on_the_dominant_model():
    losses = {
        "S1": (_dominant_panel(seed=1), None),
        "S2": (_dominant_panel(seed=2), None),
        "S3": (_dominant_panel(seed=3), None),
    }
    cross = MCC.three_series_cross_check(losses)
    assert len(cross) == 3
    assert set(cross.columns) == {
        "series", "ours_moving_block", "ours_circular", "arch",
        "exact_match_mb", "exact_match_circ",
    }
    assert (cross.ours_moving_block == 1).all()
    assert (cross.ours_circular == 1).all()
    assert (cross.arch == 1).all()
    assert cross.exact_match_mb.all()
    assert cross.exact_match_circ.all()
