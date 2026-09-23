"""Mutation-killing tests for mcs_power.py and mcs_robustness.py.

A blind scan planted number-changing mutations that the existing suite did not catch. Each test
below names the mutation it kills in its docstring ("kills: ...") and was verified to FAIL on a
copy of the module carrying that mutation and PASS on the real module.

Design rules followed: synthetic fixtures with hand-derived answers wherever the answer can be
derived by hand (constant-column panels drive mcs_test.mcs() down its deterministic certain-
rejection / exact-tie branches, so no bootstrap arithmetic is needed); one golden pin on the frozen
breadth_distinct_predictions.csv for the paper's own 5%-edge power figure, because that number is
the elimination rule acting on real dispersion and no synthetic fixture can pin it; never a test
that reads a results CSV and compares it to itself.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

import mcs_power as MP
import mcs_robustness as MR
import mcs_test as MCS


# ============================================================================================
# mcs_power.sweep(): the edge unit and the elimination rule
# ============================================================================================

def _constant_panel(cols, T=30):
    return np.column_stack([np.full(T, float(c)) for c in cols])


def test_sweep_edge_is_denominated_in_the_grand_mean_loss_hand_derived():
    """kills: M1 (`scale = np.mean(base)` -> best-model column mean).

    Constant panel [2.0, 1.0, 1.0, 1.0] (T=30 >= MIN_T_FOR_ZERO_SE, so every non-zero constant
    differential is a certain rejection and every exact tie is skipped -- no bootstrap arithmetic).
    Grand mean = 1.25; best model's own mean = 1.0. At delta = 0.85 the subtracted edge is
    0.85 * 1.25 = 1.0625 under the real code, so the focal model 0 drops from 2.0 to 0.9375 and
    becomes the UNIQUE minimum -> singled out -> hit. Under the best-model denominator the edge is
    only 0.85, model 0 lands at 1.15, stays the worst, is eliminated first and the exact 3-way tie
    {1,2,3} survives (size 3) -> miss. Models 1-3 are hits under either unit (their column falls
    strictly below the rest). Hand-derived expectation, real code: power 4/4, all four trials
    qualify (the focal is the pooled best in each), conditional 1.0, median |MCS| 1. The mutant
    gives 3/4 and n_qualifying 3."""
    pw, med, n_trials, cond, n_qual = MP.sweep([_constant_panel([2.0, 1.0, 1.0, 1.0])],
                                               delta=0.85, recenter=False)
    assert n_trials == 4
    assert pw == pytest.approx(1.0)
    assert n_qual == 4
    assert cond == pytest.approx(1.0)
    assert med == pytest.approx(1.0)


def test_sweep_hands_mcs_the_studentized_rule_seed_zero_and_module_constants(monkeypatch):
    """kills: M14 (`elimination="studentized"` -> "raw"), M3 (`seed=0` -> `seed=f`).

    The paper commits to HLN's studentized elimination rule for the power simulation (S4.2: "uses
    HLN's studentized elimination rule throughout") and to a common-random-numbers design (seed 0
    on every trial). Spies on the one mcs_test.mcs() call site inside sweep() and asserts every
    trial passes exactly those arguments plus the module's own B/ALPHA."""
    calls = []
    real_mcs = MCS.mcs

    def spy(L, **kw):
        calls.append(dict(kw))
        return real_mcs(L, **kw)

    monkeypatch.setattr(MP.MCS, "mcs", spy)
    MP.sweep([_constant_panel([2.0, 1.0, 1.0, 1.0])], delta=0.05, recenter=False)
    assert len(calls) == 4
    for kw in calls:
        assert kw["elimination"] == "studentized"
        assert kw["seed"] == 0
        assert kw["alpha"] == MP.ALPHA
        assert kw["B"] == MP.B


# A strictly positive 30 x 4 loss panel on which the studentized and raw elimination rules reach
# DIFFERENT survivor sets for the same planted edge. Generated once (numpy default_rng(102):
# means 1+U(0,0.3), sd in {0.03, 0.1, 0.4}, rounded to 4 dp) and frozen here as literals so the
# test does not depend on the RNG stream. Column 1 is the high-variance model (sd ~0.4).
_RULE_SENSITIVE_PANEL = np.array([
    [1.0467, 1.3011, 1.2936, 0.8667], [1.0838, 1.3008, 1.3516, 1.128], [0.9897, 1.2, 1.2314, 0.9412],
    [1.1114, 0.1036, 1.2065, 1.0736], [1.1045, 1.3089, 1.0282, 1.1197], [1.0046, 1.3844, 1.3017, 1.0677],
    [1.0391, 0.8842, 1.0843, 1.0048], [1.0299, 1.3697, 1.192, 1.0426], [1.0369, 1.5245, 1.2935, 1.1669],
    [1.0779, 0.2998, 1.2492, 0.9424], [1.0334, 1.5578, 1.2319, 0.9898], [1.0213, 0.9681, 1.2878, 1.1155],
    [1.0707, 0.6946, 1.0578, 1.1652], [1.0902, 1.2284, 1.2463, 1.081], [1.0364, 0.8335, 1.3716, 1.0278],
    [1.0588, 0.9598, 1.361, 1.1555], [1.0063, 0.7343, 1.086, 1.0424], [1.0804, 0.8715, 1.3389, 1.2952],
    [1.1197, 0.648, 1.2861, 1.2209], [1.0686, 0.9052, 1.1389, 1.2785], [1.0485, 0.6988, 1.1, 0.9844],
    [0.988, 0.5924, 1.2255, 1.0338], [1.0407, 1.5946, 1.3843, 1.0553], [0.9873, 1.1672, 1.2905, 0.9546],
    [1.0743, 1.3332, 1.1427, 1.0291], [1.1053, 1.8742, 1.2516, 1.1822], [1.0933, 1.4377, 1.3494, 0.9791],
    [1.0555, 1.1687, 1.2315, 1.0536], [1.0656, 0.7097, 1.1928, 1.2393], [1.0295, 1.4929, 1.2886, 1.1162],
])


def _sweep_one_panel_under_rule(M, delta, rule):
    """Independent re-statement of sweep()'s per-trial loop with the elimination rule as a
    parameter -- used only to document that the fixture discriminates the two rules."""
    hits = []
    scale = float(np.mean(M))
    for f in range(M.shape[1]):
        L = M.copy()
        L[:, f] = L[:, f] - delta * scale
        s, _ = MCS.mcs(L, alpha=MP.ALPHA, B=MP.B, seed=0, elimination=rule)
        hits.append(len(s) == 1 and s[0] == f)
    return hits


def test_sweep_power_on_a_rule_sensitive_panel_is_the_studentized_value():
    """kills: M14 (`elimination="studentized"` -> "raw").

    On the frozen panel above at a 10% edge the studentized rule singles out the focal model in
    exactly 1 of 4 trials (survivor sizes [1, 3, 3, 2]; 3 trials qualify as pooled-best, 1 of
    those is a hit), whereas the raw-mean rule singles it out in 2 of 4 (sizes [1, 3, 3, 1]).
    sweep() must report the studentized figures. The two direct-loop calls at the top are a
    fixture-validity guard (they prove the panel discriminates the rules), not the assertion."""
    assert _sweep_one_panel_under_rule(_RULE_SENSITIVE_PANEL, 0.10, "studentized") == [True, False, False, False]
    assert _sweep_one_panel_under_rule(_RULE_SENSITIVE_PANEL, 0.10, "raw") == [True, False, False, True]

    pw, med, n_trials, cond, n_qual = MP.sweep([_RULE_SENSITIVE_PANEL], delta=0.10, recenter=False)
    assert n_trials == 4
    assert pw == pytest.approx(0.25)
    assert med == pytest.approx(2.5)
    assert n_qual == 3
    assert cond == pytest.approx(1.0 / 3.0)


@pytest.mark.skipif(not MP.PRED.exists(), reason="frozen breadth_distinct_predictions.csv absent")
def test_golden_pin_studentized_power_at_5pct_edge_on_the_frozen_panel():
    """GOLDEN PIN on the frozen input (breadth_distinct_predictions.csv), recomputed through the
    real code path. Protects canonical keys mcspower.studentized_power_5pct (= 31/72 = 0.43056),
    mcspower.conditional_power_5pct (= 31/49 = 0.6327, n_qualifying 49) and mcspower.n_trials (72).
    kills: M14 (raw rule gives 34/72 = 0.472 here), M1 (best-model denominator gives 0.306), and
    any silent change to the panel loader. Runs in about one second (72 mcs() calls at B=800)."""
    mats, models = MP.load_panels()
    assert len(mats) == 12 and len(models) == 6
    pw, med, n_trials, cond, n_qual = MP.sweep(mats, delta=0.05, recenter=False)
    assert n_trials == 72
    assert pw == pytest.approx(31 / 72, abs=1e-12)
    assert n_qual == 49
    assert cond == pytest.approx(31 / 49, abs=1e-12)


# ============================================================================================
# mcs_robustness.monthly_resolution_robustness(): the `> 1` non-identification count
# ============================================================================================

def _recorder(monkeypatch, module):
    store = {}

    def rec(key, value, **kw):
        store[key] = (value, kw)

    monkeypatch.setattr(module, "emit", rec)
    monkeypatch.setattr(module, "flush", lambda: None)
    return store


def _write_monthly_pred_csv(path, series_models, n_years=10, start_year=2000):
    """breadth_distinct_predictions.csv-shaped rows: one row per (series, model, target month)
    with a constant per-model loss, so the per-year means and the native-monthly matrix are both
    constant columns (certain rejection when the constants differ; exact tie when they agree)."""
    rows = []
    for sid, per_model in series_models.items():
        for m, loss in per_model.items():
            for yi in range(n_years):
                for mo in range(1, 13):
                    rows.append({"series": sid, "model": m, "year": start_year + yi,
                                 "target": f"{start_year + yi}-{mo:02d}-01",
                                 "sae": float(loss), "msse": float(loss) ** 2})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_monthly_resolution_robustness_counts_only_strictly_non_identified_series(monkeypatch, tmp_path):
    """kills: M5 (`(R.MCS_annual > 1).sum()` -> `>= 1`).

    Three synthetic series, two models each, 10 years x 12 months of CONSTANT loss:
      IDENT_A  m1=1.0, m2=5.0  -> certain rejection at every resolution, |MCS| = 1 (identified)
      TIED     m1=m2=2.0       -> exact tie, obs_diff == 0 is skipped, |MCS| = 2 (non-identified)
      IDENT_B  m1=3.0, m2=1.0  -> identified
    Hand-derived emits: mcs.annual_nonidentified = 1 of 3 (the mutant reports 3 of 3, since every
    |MCS| is >= 1); mcs.monthly_nonidentified_range = [1, 1] over every block length; the native
    monthly matrix has T = 120 rows for every series so data.curated_monthly_T_min = _max = 120."""
    csv_path = tmp_path / "pred.csv"
    _write_monthly_pred_csv(csv_path, {
        "IDENT_A": {"m1": 1.0, "m2": 5.0},
        "TIED": {"m1": 2.0, "m2": 2.0},
        "IDENT_B": {"m1": 3.0, "m2": 1.0},
    })
    monkeypatch.setattr(MR.MT, "PRED", csv_path)
    monkeypatch.setattr(MR, "RES", tmp_path)
    store = _recorder(monkeypatch, MR)

    MR.monthly_resolution_robustness()

    assert store["mcs.annual_nonidentified"][0] == 1
    assert store["mcs.annual_nonidentified"][1]["of"] == 3
    assert store["mcs.monthly_nonidentified_range"][0] == [1, 1]
    assert store["data.curated_monthly_T_min"][0] == 120
    assert store["data.curated_monthly_T_max"][0] == 120
    out = pd.read_csv(tmp_path / "mcs_monthly_resolution.csv").set_index("series")
    assert out.loc["TIED", "MCS_annual"] == 2
    assert out.loc["IDENT_A", "MCS_annual"] == 1 and out.loc["IDENT_B", "MCS_annual"] == 1
    for blk in (1, 2, 3, 6, 12, 24):
        assert out.loc["TIED", f"MCS_mon_b{blk}"] == 2
        assert out.loc["IDENT_A", f"MCS_mon_b{blk}"] == 1


# ============================================================================================
# mcs_robustness.fredmd_robustness(): the alpha == 0.10 band filter
# ============================================================================================

def _install_fake_fredmd(monkeypatch, thresholds):
    """A stand-in `fred_md_sweep` module whose series_losses() returns, per column, a loss matrix
    that ENCODES the configuration under which a stub mcs() will call that series non-identified:
    L[0,0] = largest alpha at which it is non-identified, L[0,1] = smallest block length at which
    it is non-identified, L[1,0] = largest seed at which it is non-identified. The stub mcs()
    (installed on MR.MT) decodes those and returns a 2-model survivor set (non-identified) or a
    singleton. This isolates fredmd_robustness()'s own grid/filter/aggregation logic -- the thing
    the mutation targets -- from mcs()'s bootstrap, which has its own hand-verified tests."""
    cols = list(thresholds)
    fake = types.ModuleType("fred_md_sweep")
    # series_losses() receives the numpy values, not the column name, so each fake column carries
    # its own index as a constant value for the fake to look up.
    fake.load_fredmd = lambda: pd.DataFrame({c: np.full(10, float(i)) for i, c in enumerate(cols)},
                                            index=pd.date_range("2000-01-01", periods=10, freq="MS"))
    fake.load_series_gap_safe = lambda df, col: df[col]

    def series_losses(arr, index):
        thr_alpha, min_block, max_seed = thresholds[cols[int(arr[0])]]
        La = np.zeros((10, 2))
        La[0, 0], La[0, 1], La[1, 0] = thr_alpha, min_block, max_seed
        return (La, None, None)

    fake.series_losses = series_losses
    monkeypatch.setitem(sys.modules, "fred_md_sweep", fake)

    def stub_mcs(L, alpha=0.10, B=2000, block=3, seed=0, **kw):
        thr_alpha, min_block, max_seed = L[0, 0], L[0, 1], L[1, 0]
        non_id = (alpha <= thr_alpha + 1e-12) and (block >= min_block) and (seed <= max_seed)
        return ([0, 1] if non_id else [0]), 1.0

    monkeypatch.setattr(MR.MT, "mcs", stub_mcs)


def test_fredmd_robustness_band_is_taken_over_alpha_010_rows_only(monkeypatch, tmp_path):
    """kills: M8 (fredmd.band_alpha10 computed over every alpha instead of `R[R.alpha == 0.10]`).

    Seven stub series (see _install_fake_fredmd for the encoding: max alpha, min block, max seed):
      s1 (0.05, 1, 9)  non-identified at alpha=0.05 only
      s2 (0.10, 1, 9)  at alpha <= 0.10, any block
      s3 (0.10, 1, 9)  same
      s4 (0.10, 3, 9)  at alpha <= 0.10 and block >= 3
      s5 (0.25, 1, 9)  at every alpha
      s6 (0.25, 6, 9)  at every alpha but only block >= 6
      s7 (0.25, 1, 0)  at every alpha, seed 0 only (so it adds 1 to one of the 3 seeds)
    Hand-derived per-seed counts over the module's GRID [(block, B, alpha)]:
      (3, 2000, 0.10): s2 s3 s4 s5 (+s7) -> 5, 4, 4
      (1, 2000, 0.10): s2 s3 s5    (+s7) -> 4, 3, 3
      (6, 2000, 0.10): s2 s3 s4 s5 s6 (+s7) -> 6, 5, 5
      (3, 5000, 0.10): as the first row  -> 5, 4, 4
      (3, 2000, 0.05): s1 s2 s3 s4 s5 (+s7) -> 6, 5, 5   mean 5.33
      (3, 2000, 0.25): s5 (+s7)        -> 2, 1, 1   mean 1.33, min 1, max 2
    So fredmd.band_alpha10 = [3, 6] of 7 (the mutant, spanning every alpha, reports [1, 6]);
    fredmd.count_alpha05 = 5.33; fredmd.count_alpha25 = 1.33 with p25_min 1, p25_max 2."""
    _install_fake_fredmd(monkeypatch, {
        "s1": (0.05, 1, 9), "s2": (0.10, 1, 9), "s3": (0.10, 1, 9), "s4": (0.10, 3, 9),
        "s5": (0.25, 1, 9), "s6": (0.25, 6, 9), "s7": (0.25, 1, 0),
    })
    monkeypatch.setattr(MR, "RES", tmp_path)
    store = _recorder(monkeypatch, MR)

    MR.fredmd_robustness(seeds=3)

    assert store["fredmd.band_alpha10"][0] == [3, 6]
    assert store["fredmd.band_alpha10"][1]["of"] == 7
    assert store["fredmd.count_alpha05"][0] == pytest.approx(5.33)
    assert store["fredmd.count_alpha25"][0] == pytest.approx(1.33)
    assert store["fredmd.count_alpha25"][1]["p25_min"] == 1
    assert store["fredmd.count_alpha25"][1]["p25_max"] == 2
    band = pd.read_csv(tmp_path / "fredmd_robustness_band.csv")
    assert len(band) == 6 and (band.n == 7).all() and (band.seeds == 3).all()
    row = band[(band.block == 6) & (band.alpha == 0.10)].iloc[0]
    assert (row["min"], row["max"]) == (5, 6) and row["mean"] == pytest.approx(16 / 3)
