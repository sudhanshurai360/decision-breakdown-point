"""Value-pinning tests for code/pipeline/mcs_test.py and mcs_size.py.

Of 13 mutations planted in a mutation scan of mcs_test.py, the number-moving ones (block start
range off by one, block length +1, default block=2, transposed L) were caught by exactly ONE test
-- test_regime_robustness's golden
comparison against the committed mcs_test.csv, which is skipped whenever the real data files are
absent -- and two mutations survived everything:

  R2   se centred on the BOOTSTRAP MEAN (np.std(diff)) instead of on the sample differential
       (HLN's var-hat; the manuscript's package paragraph says the "paper's pipeline centers on
       the sample differential with moving blocks")  -> no curated cell moves today, so nothing
       failed
  R2'  fresh bootstrap indices at every elimination step instead of HLN's single shared draw

This file makes the bootstrap arithmetic testable WITHOUT real data by (a) driving mcs() with a
tiny FIXED set of resample index vectors (monkeypatched _block_idx) so every bootstrap quantity
is hand-computable, (b) pinning _block_idx's own start-range/block-length behaviour to
hand-computed appearance probabilities, (c) pinning the defaults and the call-count wiring, and
(d) ONE golden pin on the frozen breadth_distinct_predictions.csv that recomputes Table 1's
|MCS| column and the 11-of-12 headline (canonical key mcs.curated_nonidentified) through the
real code path. """
import inspect
from pathlib import Path

import numpy as np
import pytest

import mcs_size as MS
import mcs_test as MT


# ---------------------------------------------------------------------------------------------
# (a) fixed-resample harness: B = 4 explicit index vectors, T = 4, K = 3
# ---------------------------------------------------------------------------------------------
def _fixed_resamples(monkeypatch, idx_list):
    """Replace the block-index draw with a cycle through `idx_list` (B must equal len)."""
    it = iter([np.asarray(i) for i in idx_list])
    calls = []

    def fake_block_idx(T, block, rng):
        calls.append((T, block))
        return next(it)

    monkeypatch.setattr(MT, "_block_idx", fake_block_idx)
    return calls


# Fixture A. L rows (periods) x columns (models 0,1,2):
LA = np.array([[4.0, 5.0, 5.0],
               [3.0, 0.0, 4.0],
               [0.0, 5.0, 4.0],
               [5.0, 5.0, 3.0]])
IDX_A = [[1, 1, 2, 2], [1, 2, 3, 2], [0, 1, 2, 2], [1, 1, 2, 0]]


def test_se_is_centred_on_the_sample_differential_not_the_bootstrap_mean(monkeypatch):
    """kills: R2 (se = np.std(diff), bootstrap-mean centring).
    dbar = [3, 3.75, 4]. Resample means (rows = the 4 index vectors):
      [1.5, 2.5, 4], [2, 3.75, 3.75], [1.75, 3.75, 4.25], [2.5, 2.5, 4.25]
    Pair (0,2): obs = -1; diffs = [-2.5, -1.75, -2.5, -1.75]; diff-obs = [-1.5, -.75, -1.5, -.75]
      sample-centred se = sqrt(5.625/4) = 1.1859 -> t = 0.8433   <- the range statistic T_R
      (bootstrap-mean-centred: mean diff = -2.125, deviations +/-0.375, se = 0.375, t = 2.667)
    Pair (0,1): obs = -.75; diff-obs = [-.25, -1, -1.25, .75]; se = sqrt(3.1875/4) = .8927; t = .840
    Pair (1,2): obs = -.25; diff-obs = [-1.25, .25, -.25, -1.5]; se = sqrt(3.9375/4) = .9922; t = .252
    Per-resample T*_b = max over pairs of |diff-obs|/se = [1.265, 1.120, 1.400, 1.512], ALL >= T_R
    = 0.8433, so p = (4+1)/(4+1) = 1.0 exactly and nothing is eliminated at any alpha.
    Under bootstrap-mean centring T_R = 2.667 and only resamples 1 and 3 reach it (T*_b =
    [4, 2, 4, 2.097]): p = 3/5 = 0.6, so at alpha = 0.8 model 2 (argmax dbar) is eliminated."""
    _fixed_resamples(monkeypatch, IDX_A)
    surv, p = MT.mcs(LA, alpha=0.8, B=4, block=3, seed=0)
    assert surv == [0, 1, 2]
    assert p == pytest.approx(1.0)


# Fixture B. Chosen so the first-step p is EXACTLY (2+1)/(4+1) = 0.6:
LB = np.array([[1.0, 0.0, 0.0],
               [5.0, 0.0, 2.0],
               [4.0, 4.0, 5.0],
               [0.0, 2.0, 0.0]])
IDX_B = [[0, 3, 0, 2], [0, 1, 3, 1], [1, 0, 3, 0], [1, 0, 0, 3]]


def test_p_value_uses_plus_one_over_B_plus_one_and_stops_at_p_equal_alpha(monkeypatch):
    """kills: `p > alpha` (strict stop) and the dropped +1/(B+1) correction -- both individually
    harmless on the curated panel (no step p sits at alpha) but each is a silent convention change.
    dbar = [2.5, 1.5, 1.75]; resample means [1.5,1.5,1.25], [2.75,.5,1], [1.75,.5,.5], [1.75,.5,.5].
    Pair (0,1): obs = 1; diff-obs = [-1, 1.25, .25, .25]; se = sqrt(2.6875/4) = .8197; t = 1.220 = T_R
    Pair (0,2): obs = .75; diff-obs = [-.5, 1, .5, .5]; se = .6614; t = 1.134
    Pair (1,2): obs = -.25; diff-obs = [.5, -.25, .25, .25]; se = .3307; t = .756
    T*_b = [1.512, 1.525, 0.756, 0.756] -> 2 of 4 reach T_R -> p = (2+1)/(4+1) = 0.6 EXACTLY.
    With alpha = 0.6 the rule `p >= alpha` stops with all three models; `p > alpha` would
    eliminate model 0, and #/B = 0.5 would eliminate it too."""
    _fixed_resamples(monkeypatch, IDX_B)
    surv, p = MT.mcs(LB, alpha=0.6, B=4, block=3, seed=0)
    assert p == pytest.approx(0.6)
    assert surv == [0, 1, 2]

    # ... and one notch stricter the same fixture DOES eliminate argmax(dbar) = model 0 (raw rule);
    # the same 4 index vectors serve every step (HLN's single shared draw), so no more are needed.
    _fixed_resamples(monkeypatch, IDX_B)
    surv2, _ = MT.mcs(LB, alpha=0.61, B=4, block=3, seed=0)
    assert 0 not in surv2 and len(surv2) <= 2


def test_bootstrap_indices_are_drawn_once_and_shared_across_elimination_steps(monkeypatch):
    """kills: fresh bootstrap per elimination step.
    HLN's procedure fixes ONE set of B resamples and re-uses it at every step. On a panel where
    two eliminations happen (a hopeless model and a clearly-worse one), _block_idx must be called
    exactly B times in total, not B x (number of steps)."""
    rng = np.random.default_rng(0)
    T = 20
    good = rng.normal(1.0, 0.05, T)
    L = np.column_stack([good, good + 1.0 + rng.normal(0, 0.05, T), good + 3.0])
    calls = []
    real = MT._block_idx

    def counting(T_, block, rng_):
        calls.append((T_, block))
        return real(T_, block, rng_)

    monkeypatch.setattr(MT, "_block_idx", counting)
    surv, _ = MT.mcs(L, alpha=0.10, B=50, block=3, seed=0)
    assert surv == [0], "fixture must eliminate both dominated models (two steps)"
    assert len(calls) == 50
    assert set(calls) == {(20, 3)}


# ---------------------------------------------------------------------------------------------
# (b) _block_idx: start range [0, T-block] INCLUSIVE, fixed block length, moving (no wrap)
# ---------------------------------------------------------------------------------------------
def test_block_idx_start_range_and_block_length_hand_computed_frequencies():
    """kills: start range `[0, T-block)` (off by one) and block length +1.
    T = 5, block = 3: starts are uniform on {0, 1, 2}; each draw is block {s1, s1+1, s1+2} followed
    by {s2, s2+1} (second block truncated to fill T = 5). Hand-computed:
      P(period 4 present) = P(s1 = 2) = 1/3           (the truncated 2nd block never reaches 4)
      P(period 0 present) = 1 - P(s1 != 0) P(s2 != 0) = 1 - (2/3)^2 = 5/9
    Off-by-one starts {0, 1}: P(4 present) = 0, P(0 present) = 3/4. Block length 4 (starts {0, 1}):
    P(4 present) = 1/2. 6000 draws -> SE <= 0.0065; bands are +/- 0.03."""
    rng = np.random.default_rng(123)
    T, block, n = 5, 3, 6000
    has4 = has0 = 0
    starts = set()
    for _ in range(n):
        idx = MT._block_idx(T, block, rng)
        assert idx.shape == (T,) and idx.min() >= 0 and idx.max() < T
        s1 = int(idx[0])
        starts.add(s1)
        assert list(idx[:3]) == [s1, s1 + 1, s1 + 2], "first block must be 3 consecutive periods"
        assert idx[4] == idx[3] + 1, "second block must be consecutive too"
        has4 += int(4 in idx)
        has0 += int(0 in idx)
    assert starts == {0, 1, 2}, starts
    assert abs(has4 / n - 1 / 3) < 0.03, has4 / n
    assert abs(has0 / n - 5 / 9) < 0.03, has0 / n


# ---------------------------------------------------------------------------------------------
# (c) defaults and wiring the paper states: block 3, B 2000, seed 0, alpha 0.10, raw rule,
#     MOVING block (not circular)
# ---------------------------------------------------------------------------------------------
def test_mcs_defaults_match_the_manuscript_and_moving_block_is_the_default(monkeypatch):
    """kills: default block=2 (and any silent change of B / seed / alpha / rule / circular).
    S4 body: 'moving-block bootstrap, block length 3, B=2,000 resamples, seed 0'. Also confirms mcs() with
    defaults really routes through _block_idx (moving) with block 3, not _circular_block_idx."""
    sig = inspect.signature(MT.mcs).parameters
    assert sig["alpha"].default == 0.10
    assert sig["B"].default == 2000
    assert sig["block"].default == 3
    assert sig["seed"].default == 0
    assert sig["elimination"].default == "raw"
    assert sig["circular"].default is False and MT.CIRCULAR_BOOTSTRAP is False

    seen = []
    real = MT._block_idx

    def recording(T, block, rng):
        seen.append(block)
        return real(T, block, rng)

    def circular_must_not_run(*a, **k):
        raise AssertionError("the circular index path was used by default")

    monkeypatch.setattr(MT, "_block_idx", recording)
    monkeypatch.setattr(MT, "_circular_block_idx", circular_must_not_run)
    L = np.random.default_rng(0).normal(1.0, 0.1, (12, 3))
    MT.mcs(L, B=5)
    assert seen == [3] * 5


def test_size_row_calls_mcs_with_the_documented_bootstrap_settings_and_counts_rejections(monkeypatch):
    """mcs_size.size_row: rejection = at least one model eliminated; the per-panel call uses
    B=800 and bootstrap seed 0, plus the requested block and alpha.
    Stub MCS.mcs to eliminate on every other panel -> rate 0.5, se = sqrt(.25/n)."""
    seen = []
    state = {"n": 0}

    def fake_mcs(L, alpha, B, block, seed):
        seen.append((L.shape, alpha, B, block, seed))
        state["n"] += 1
        K = L.shape[1]
        return (list(range(K - 1)) if state["n"] % 2 == 0 else list(range(K))), 0.5

    monkeypatch.setattr(MS.MCS, "mcs", fake_mcs)
    rate, se = MS.size_row(K=6, T=30, block=3, alpha=0.10, n_rep=10, seed=1)
    assert rate == pytest.approx(0.5) and se == pytest.approx(np.sqrt(0.25 / 10))
    assert len(seen) == 10 and set(seen) == {((30, 6), 0.10, 800, 3, 0)}


# ---------------------------------------------------------------------------------------------
# (d) GOLDEN PIN: Table 1's |MCS| column and the 11-of-12 headline from the frozen predictions
# ---------------------------------------------------------------------------------------------
TABLE1_MCS_ABS_10 = {"PI": 1, "RSAFS": 2, "BOPGEXP": 3, "BOPGIMP": 3, "PCE": 4, "UNEMPLOY": 4,
                     "BUSINV": 5, "DGORDER": 5, "HOUST": 5, "M2SL": 5, "PAYEMS": 5, "TCU": 5}


@pytest.mark.skipif(not Path(MT.PRED).exists(),
                    reason="frozen breadth_distinct_predictions.csv not present in this checkout")
def test_curated_mcs_headline_pin_from_frozen_predictions():
    """GOLDEN PIN (kills every block-index / transpose / centring mutant that moves a Table 1
    cell, independently of mcs_test.csv): recompute |MCS| at alpha=0.10, B=2000, block 3, seed 0
    for the 12 curated series from the frozen input via mcs_size.curated_panels() (verified
    elsewhere to build the same per-year sae matrices as mcs_test.main()). Protects canonical key
    mcs.curated_nonidentified = 11 of 12 (Abstract, Introduction, S4.1) and the per-series |MCS|
    values Table 1 prints. The block-length mutants move BUSINV 5->2 and BOPGIMP 3->4."""
    panels = MS.curated_panels()
    assert set(panels) == set(TABLE1_MCS_ABS_10)
    sizes = {sid: len(MT.mcs(L, alpha=0.10, seed=0)[0]) for sid, L in panels.items()}
    assert sizes == TABLE1_MCS_ABS_10
    assert sum(v > 1 for v in sizes.values()) == 11
