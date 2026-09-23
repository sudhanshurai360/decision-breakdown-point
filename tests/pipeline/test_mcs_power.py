"""mcs_power.py is a power-simulation study -- a place where an overclaim is easy to make and hard
to notice (see sweep()'s recenter=True arm, marked superseded below). This file
unit-tests the module's independently-testable pure/near-pure functions
directly, using small synthetic panels with either exactly hand-derivable expected answers
(constructed ties/strict dominance, which force mcs_test.mcs()'s deterministic certain-
rejection/genuine-tie branches independent of bootstrap randomness -- reasoned through by hand,
then confirmed by direct execution) or documented edge-case guards:

  - sweep(mats, delta, recenter): the module's core statistic. Ground-truthed on a 4-model
    constant-panel construction where one model starts far worse and a small additive edge is
    injected on each model in turn -- every resulting share (unconditional power, conditional
    power, n_qualifying, median survivor-set size) was hand-derived by tracing mcs_test.mcs()'s
    studentized-elimination branch step by step, then confirmed by direct execution (not the other
    way around: the expected values are not read off a run and pasted in blind).
  - positive_control()/negative_control()/negative_control_raw(): this file's own validity-gate and
    specificity-figure functions. CONTROL_N and _CONTROL_SEEDS are monkeypatched down for test
    speed (both are deterministic given their fixed internal seeds -- 7 for positive_control, the
    _CONTROL_SEEDS tuple for the negative controls -- so shrinking N changes precision, not
    reproducibility). Matches this project's convention of testing a statistical gate with a
    reduced-but-still-decisive replication count (cf. tests/pipeline/test_mcs_size_level.py).
  - load_panels(): the ALL-OR-NOTHING completeness guard -- a single missing (model, year) cell
    anywhere silently drops the WHOLE series, not just that row (see the module's own
    AssertionError message). Tested against synthetic CSVs for the pass case (exactly 12
    rectangular series), the fail case
    (wrong series count -> AssertionError), and the documented-but-unfixed whole-series-drop
    behavior itself (13 series, 1 incomplete -> exactly 12 survive).
  - main()'s positive-control ABORT gate: the contract mcspower.positive_control's own spec= string
    states ("A VALIDITY GATE, not a result: the stage raises if this falls below 0.9",
    mcs_power.py:121-123), mirroring tests/pipeline/test_mcs_size_level.py's analogous SystemExit
    gate test for mcs_size.py::main().

No test calls the real load_panels() against production data or runs main() end-to-end (both would
pull in real file I/O, real emit()/canonical registry writes, and the real N=2000 replication
counts) -- these tests only exercise the module's own pure computation via synthetic inputs.
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

import mcs_power as MP


# --- sweep(): hand-derived exact ground truth ---------------------------------------------------

def _hand_derived_panel(T=30):
    """4 models, T periods, all constant (zero within-column variance): model 0 loss=2.0 (already
    worse than the rest), models 1-3 loss=1.0 (exactly tied with each other). scale = mean of the
    whole matrix = (2+1+1+1)/4 = 1.25.

    Traced by hand for delta=0.05, recenter=False (reduction = 0.05*1.25 = 0.0625 subtracted from
    the focal column only):
      f=0: col0 -> 1.9375, still the worst of the four -> eliminated first, remaining {1,2,3} stay
           an exact 3-way tie -> survivors=[1,2,3] (size 3) -> NOT a hit; pooled-best is col1
           (first of the tied 1.0s), which != f=0 -> does NOT qualify for the conditional figure.
      f=1: col1 -> 0.9375, uniquely lowest -> deterministic certain-rejection cascades all the way
           down -> survivors=[1] alone (size 1) -> a hit; col1 is also the pooled best -> qualifies.
      f=2, f=3: symmetric to f=1 by construction -> hit, qualifies.
    So of 4 trials: 3 hits (f=1,2,3), 1 miss (f=0) -> power = 3/4 = 0.75. Of those, f=0 alone fails
    to qualify (1 of 4) -> n_qualifying = 3, and all 3 qualifying trials are hits -> conditional
    power = 1.0. Sizes = [3, 1, 1, 1] -> median = 1.0.
    """
    return np.column_stack([
        np.full(T, 2.0), np.full(T, 1.0), np.full(T, 1.0), np.full(T, 1.0),
    ])


def test_sweep_hand_derived_exact_case_with_edge():
    pw, med, n_trials, cond, n_qual = MP.sweep([_hand_derived_panel()], delta=0.05, recenter=False)
    assert pw == pytest.approx(0.75)
    assert med == pytest.approx(1.0)
    assert n_trials == 4
    assert cond == pytest.approx(1.0)
    assert n_qual == 3


def test_sweep_hand_derived_zero_edge_case():
    """At delta=0.0, `L[:, f] -= 0*scale` is a no-op for every f -> every trial sees the IDENTICAL
    unmodified base panel, so all 4 trials produce the identical outcome: eliminate col0 (the worst,
    2.0), leaving the exact 3-way tie {1,2,3} (size 3, never a singleton) -> power = 0.0 for all 4
    trials, sizes = [3,3,3,3] -> median = 3.0. The pooled-best index (argmin, tie-broken to the
    FIRST index among the tied 1.0 columns = col1) equals f only for f=1 -> exactly 1 of 4 trials
    qualifies, and that one trial is a miss (survivors=[1,2,3], not a singleton) -> conditional
    power = 0.0."""
    pw, med, n_trials, cond, n_qual = MP.sweep([_hand_derived_panel()], delta=0.0, recenter=False)
    assert pw == 0.0
    assert med == pytest.approx(3.0)
    assert n_trials == 4
    assert cond == 0.0
    assert n_qual == 1


def test_sweep_aggregates_correctly_across_multiple_panels():
    """Two IDENTICAL copies of the hand-derived panel must simply double every count while leaving
    every share (power, median, conditional power) unchanged -- a direct check that sweep()'s
    aggregation loops over `mats` and does not, e.g., only look at the first panel or double-count
    incorrectly."""
    panel = _hand_derived_panel()
    pw, med, n_trials, cond, n_qual = MP.sweep([panel, panel], delta=0.05, recenter=False)
    assert pw == pytest.approx(0.75)
    assert med == pytest.approx(1.0)
    assert n_trials == 8
    assert cond == pytest.approx(1.0)
    assert n_qual == 6


def test_sweep_recentred_design_runs_and_returns_valid_ranges():
    """The re-centred (recenter=True) arm is EXPLICITLY documented as superseded and prone to a
    known "structurally uneliminable model" defect that makes its power reading not simply
    predictable from the panel construction -- so this is a
    shape/range sanity check (there is no clean hand-derivable answer for this arm), not a
    ground-truth value assertion. Uses a panel with genuine variation (not the
    exact-tie construction above, which happens to make recenter a no-op)."""
    rng = np.random.default_rng(0)
    M = rng.normal(1.0, 0.1, size=(30, 4))
    pw, med, n_trials, cond, n_qual = MP.sweep([M], delta=0.3, recenter=True)
    assert 0.0 <= pw <= 1.0
    assert 1.0 <= med <= 4.0
    assert n_trials == 4
    assert 0 <= n_qual <= 4
    assert np.isnan(cond) or 0.0 <= cond <= 1.0


def test_sweep_power_is_monotonic_in_delta_on_a_noisy_synthetic_panel():
    """A weaker, more realistic invariant than the exact-tie case above: on a panel with genuine
    (small) random dispersion but no engineered ties, detection power must not DECREASE as the
    planted additive edge grows from 0 to a large value. sweep() is fully deterministic given a
    fixed `mats` (mcs_test.mcs is always called with seed=0 internally), so this is not a flaky
    statistical check -- it reproduces identically on every run."""
    rng = np.random.default_rng(7)
    mats = [rng.normal(1.0, 0.08, size=(30, 5)) for _ in range(6)]
    pw0, *_ = MP.sweep(mats, delta=0.0, recenter=False)
    pw_small, *_ = MP.sweep(mats, delta=0.05, recenter=False)
    pw_large, *_ = MP.sweep(mats, delta=0.5, recenter=False)
    assert pw0 <= pw_small <= pw_large
    assert pw_large > 0.8  # a 50% additive edge against sd=0.08 dispersion should be reliably caught


# --- positive_control / negative_control (deterministic given fixed internal seeds) -----------

def test_positive_control_detects_a_planted_edge_with_reduced_replication(monkeypatch):
    """positive_control() is this file's own validity gate: a 30% additive edge on an i.i.d. K=6
    T=30 panel must be detected reliably. CONTROL_N is monkeypatched down from the production 2000
    to 200 for test speed -- both values are deterministic given the fixed seed=7 inside
    positive_control(), so this is not a flaky reduced-precision measurement, just a cheaper one.
    Threshold (0.85) is looser than the production CONTROL_MIN (0.90) to leave headroom for the
    lower replication count without becoming a tautology."""
    monkeypatch.setattr(MP, "CONTROL_N", 200)
    ctrl = MP.positive_control()
    assert ctrl > 0.85


def test_negative_control_has_low_false_discovery_with_reduced_replication(monkeypatch):
    """The specificity figure: i.i.d. panels with NO planted edge should almost never collapse to a
    singleton. _CONTROL_SEEDS shrunk to 2 seeds and CONTROL_N to 200 for speed; both negative
    controls (studentized and raw elimination) are checked."""
    monkeypatch.setattr(MP, "CONTROL_N", 200)
    monkeypatch.setattr(MP, "_CONTROL_SEEDS", (11, 10_000_030))
    neg = MP.negative_control()
    neg_raw = MP.negative_control_raw()
    for result in (neg, neg_raw):
        assert result["n_seeds"] == 2
        assert 0.0 <= result["mean"] < 0.05
        assert 0.0 <= result["min"] <= result["max"] < 0.05


# --- load_panels(): the documented ALL-OR-NOTHING completeness guard --------------------------

def _build_pred_csv(path, n_series, models=("m1", "m2", "m3"), n_years=20, drop_cell=None):
    rows = []
    for si in range(n_series):
        sid = f"S{si}"
        for m in models:
            for yi in range(n_years):
                year = 2000 + yi
                if drop_cell == (sid, m, year):
                    continue
                rows.append({"series": sid, "model": m, "year": year,
                             "sae": float(si + len(m) + yi) * 0.01 + 1.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_panels_returns_12_rectangular_matrices_when_data_is_complete(monkeypatch, tmp_path):
    csv_path = tmp_path / "pred_12.csv"
    _build_pred_csv(csv_path, n_series=12)
    monkeypatch.setattr(MP, "PRED", csv_path)
    mats, models = MP.load_panels()
    assert len(mats) == 12
    assert models == ["m1", "m2", "m3"]
    for M in mats:
        assert M.shape == (20, 3)
        assert np.all(np.isfinite(M))


def test_load_panels_raises_assertion_error_when_series_count_is_wrong(monkeypatch, tmp_path):
    csv_path = tmp_path / "pred_5.csv"
    _build_pred_csv(csv_path, n_series=5)
    monkeypatch.setattr(MP, "PRED", csv_path)
    with pytest.raises(AssertionError, match="expected 12 curated series"):
        MP.load_panels()


def test_load_panels_drops_the_whole_series_on_a_single_missing_cell(monkeypatch, tmp_path):
    """Documents CURRENT behavior: a missing (model, year) cell silently drops the whole series (see
    load_panels()'s own AssertionError message) -- NOT asserting this is desirable, only pinning what
    the code does today rather than quietly changing suspected production behavior as a side effect
    of a test. 13 series total, one has exactly one missing (model, year) cell -> that ENTIRE series
    is dropped (not just the one row), leaving exactly 12."""
    csv_path = tmp_path / "pred_13_incomplete.csv"
    _build_pred_csv(csv_path, n_series=13, drop_cell=("S12", "m1", 2000))
    monkeypatch.setattr(MP, "PRED", csv_path)
    mats, models = MP.load_panels()
    assert len(mats) == 12  # S12 (the incomplete one) dropped entirely, not just its one bad row


# --- main()'s positive-control ABORT gate -------------------------------------------------------

def test_main_aborts_loudly_when_positive_control_falls_below_the_floor(monkeypatch):
    """The contract mcs_power.py:114-116 encodes: a harness that cannot detect an edge it was
    handed must raise SystemExit ("every power number below is meaningless"), not print a warning
    nobody reads.
    load_panels() and positive_control() are stubbed so this test never touches real data or the
    real (expensive) simulation -- it only checks the gate's own control flow."""
    monkeypatch.setattr(MP, "load_panels", lambda: ([np.ones((20, 2))], ["m1", "m2"]))
    monkeypatch.setattr(MP, "positive_control", lambda: 0.5)  # below CONTROL_MIN=0.90
    with pytest.raises(SystemExit, match="ABORT"):
        MP.main()
