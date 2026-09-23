"""combination_benchmark.py has one subtle correctness trap that would silently flip its headline
finding ("equal-weight combination ranks median 6th of 7"): building the combination from the six
models' POINT FORECASTS and then scoring it (`y_pred_comb = mean(y_pred)`, then loss of that) vs.
incorrectly averaging the six models' LOSSES directly -- a different (and meaningless) object by
Jensen's inequality, since loss functions here are convex. `main()`/`main5()` do file I/O against the real
~26,545-line, ~80-minute production panel and are exercised by a full producer rerun; this file
instead unit-tests the one pure, independently-testable computation -- `per_series_panels()`, which
builds the per-series (per-year loss dict, n_years) the rest of main()/main5() consume -- against a
tiny synthetic prediction panel (via monkeypatching the module's PRED path, never touching real
data), constructed so that combine-predictions-then-score PROVABLY differs from average-of-losses
(Jensen's inequality on the convex absolute-value scoring function, with an asymmetric error
pattern by construction).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "instrument"))

import numpy as np
import pandas as pd

import combination_benchmark as CB


def _write_pred_csv(path, rows):
    """rows: list of dicts with keys series, model, year, target, y_pred, y_true, scale_abs, sae."""
    pd.DataFrame(rows, columns=["series", "model", "year", "target", "y_pred", "y_true",
                                 "scale_abs", "sae"]).to_csv(path, index=False)


def _five_year_rows(series, model, y_pred, y_true, scale_abs=1.0):
    """One row per year 2015..2019, one target date per year (2015-06-15 .. 2019-06-15), constant
    y_pred/y_true/scale_abs across years for that model unless the caller overrides per row."""
    years = [2015, 2016, 2017, 2018, 2019]
    rows = []
    for yr in years:
        sae = abs(y_true - y_pred) / scale_abs
        rows.append(dict(series=series, model=model, year=yr, target=f"{yr}-06-15",
                          y_pred=y_pred, y_true=y_true, scale_abs=scale_abs, sae=sae))
    return rows


# --- THE core correctness trap: combine the forecasts, then score ----------------------------

def test_per_series_panels_combines_predictions_then_scores_not_average_of_losses(monkeypatch, tmp_path):
    """5 years, 2 models, y_true=100, scale_abs=1 every year. Model A over-predicts by +10 every
    year (y_pred=110); model B under-predicts by -2 every year (y_pred=98) -- deliberately
    ASYMMETRIC errors so the two candidate combination objects provably differ:
      combine-then-score:  y_comb = mean(110, 98) = 104  -> sae_comb = |100-104|/1 = 4.0
      average-of-losses:   mean(|100-110|, |100-98|) = mean(10, 2) = 6.0
    4.0 != 6.0 (Jensen's inequality on the convex abs() scoring function with asymmetric errors).
    Combine-then-score is the intended approach: per_series_panels() averages the POINT FORECASTS
    at each target date and scores the average (combination_benchmark.py:32-39), rather than
    averaging the models' losses. Scoring it the wrong way would manufacture the result -- so
    assert the actual code produces 4.0, not 6.0."""
    rows = _five_year_rows("S1", "A", y_pred=110.0, y_true=100.0) + \
           _five_year_rows("S1", "B", y_pred=98.0, y_true=100.0)
    pred_path = tmp_path / "pred.csv"
    _write_pred_csv(pred_path, rows)
    monkeypatch.setattr(CB, "PRED", pred_path)

    panels = CB.per_series_panels()
    assert "S1" in panels
    L, T = panels["S1"]
    assert T == 5
    np.testing.assert_allclose(L[CB.COMB], np.full(5, 4.0))
    # explicitly confirm the naive-but-wrong alternative would have given a DIFFERENT answer,
    # so this test is actually discriminating between the two candidate implementations.
    assert not np.allclose(L[CB.COMB], np.full(5, 6.0))
    # the singles themselves (read straight from the precomputed 'sae' column) are untouched by
    # the combination logic and match the hand-computed per-model losses.
    np.testing.assert_allclose(L["A"], np.full(5, 10.0))
    np.testing.assert_allclose(L["B"], np.full(5, 2.0))


# --- exclude=() parameter: used by main5() to drop seasonal_naive from the combination only --

def test_per_series_panels_exclude_changes_combination_but_not_singles(monkeypatch, tmp_path):
    """3 models A, B, C. Excluding A from the combination should make the combination identical to
    the plain mean of B and C's predictions (not A, B, C) -- while A still appears in the returned
    per-model loss dict as an ordinary single. That is the contract the code encodes: `exclude`
    filters only the frame the combination is built from (`g_comb`,
    combination_benchmark.py:33-34), while the singles are taken from the unfiltered `g`
    (combination_benchmark.py:47)."""
    rows = (_five_year_rows("S1", "A", y_pred=1000.0, y_true=100.0)  # wildly wrong if ever included
            + _five_year_rows("S1", "B", y_pred=110.0, y_true=100.0)
            + _five_year_rows("S1", "C", y_pred=90.0, y_true=100.0))
    pred_path = tmp_path / "pred.csv"
    _write_pred_csv(pred_path, rows)
    monkeypatch.setattr(CB, "PRED", pred_path)

    panels_excl = CB.per_series_panels(exclude=("A",))
    L, T = panels_excl["S1"]
    assert T == 5
    # combination excluding A: mean(110, 90) = 100 exactly -> sae_comb = 0.0
    np.testing.assert_allclose(L[CB.COMB], np.full(5, 0.0), atol=1e-9)
    # A is still present as a single, with its own (huge) loss, unaffected by exclusion.
    np.testing.assert_allclose(L["A"], np.full(5, 900.0))
    np.testing.assert_allclose(L["B"], np.full(5, 10.0))
    np.testing.assert_allclose(L["C"], np.full(5, 10.0))

    # sanity: WITHOUT exclusion, the combination differs (A drags it far off) -- confirms the
    # exclude parameter is actually doing something, not silently ignored.
    panels_incl = CB.per_series_panels()
    L2, _ = panels_incl["S1"]
    assert not np.allclose(L2[CB.COMB], L[CB.COMB])


# --- "only dates every (non-excluded) model forecasts" -- the pivot_table dropna() behavior -

def test_per_series_panels_drops_dates_where_any_model_is_missing(monkeypatch, tmp_path):
    """Model B has no row at all for 2015 (not a NaN value -- an absent row, as would occur for a
    model that failed to produce a forecast on one target date). The pivot-table dropna() in
    per_series_panels() must drop that WHOLE target date from the combination (not just impute B's
    missing value), per the module's own comment at combination_benchmark.py:36 ("keep only target
    dates every (non-excluded) model forecast"). 6 candidate years in, 1 dropped -> exactly 5
    survive (the minimum for inclusion)."""
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    rows = []
    for yr in years:
        rows += _five_year_rows_single("S1", "A", yr, y_pred=110.0, y_true=100.0)
    for yr in years:
        if yr == 2015:
            continue  # B has no forecast at all for 2015
        rows += _five_year_rows_single("S1", "B", yr, y_pred=98.0, y_true=100.0)
    pred_path = tmp_path / "pred.csv"
    _write_pred_csv(pred_path, rows)
    monkeypatch.setattr(CB, "PRED", pred_path)

    panels = CB.per_series_panels()
    assert "S1" in panels
    L, T = panels["S1"]
    assert T == 5  # 2015 dropped, 2016-2020 survive
    np.testing.assert_allclose(L[CB.COMB], np.full(5, 4.0))  # same combine-then-score arithmetic as above


def _five_year_rows_single(series, model, year, y_pred, y_true, scale_abs=1.0):
    sae = abs(y_true - y_pred) / scale_abs
    return [dict(series=series, model=model, year=year, target=f"{year}-06-15",
                  y_pred=y_pred, y_true=y_true, scale_abs=scale_abs, sae=sae)]


# --- minimum-length filter: series with < 5 usable years are dropped entirely ----------------

def test_per_series_panels_drops_series_below_minimum_length(monkeypatch, tmp_path, capsys):
    """4 years only (per_series_panels requires len(panel) >= 5) -- the series must not appear in
    the returned dict at all, and the drop must now be disclosed via a [skip] print (this
    `continue` used to be silent)."""
    years = [2015, 2016, 2017, 2018]
    rows = []
    for yr in years:
        rows += _five_year_rows_single("S_SHORT", "A", yr, y_pred=110.0, y_true=100.0)
        rows += _five_year_rows_single("S_SHORT", "B", yr, y_pred=98.0, y_true=100.0)
    pred_path = tmp_path / "pred.csv"
    _write_pred_csv(pred_path, rows)
    monkeypatch.setattr(CB, "PRED", pred_path)

    panels = CB.per_series_panels()
    assert "S_SHORT" not in panels
    out = capsys.readouterr().out
    assert "[skip] S_SHORT: only 4 year(s)" in out


# --- sae_comb entirely empty (every date's y_true is NaN): must be disclosed, not silent -----

def test_per_series_panels_discloses_when_combination_has_no_finite_value(monkeypatch, tmp_path, capsys):
    """Every candidate date has y_true=NaN -- sae_comb is empty after the isfinite() filter, and
    the series must be dropped entirely with a [skip] print (this `continue`
    used to be silent)."""
    years = [2015, 2016, 2017, 2018, 2019]
    rows = []
    for yr in years:
        rows += _five_year_rows_single("S_ALLNAN", "A", yr, y_pred=110.0, y_true=float("nan"))
        rows += _five_year_rows_single("S_ALLNAN", "B", yr, y_pred=98.0, y_true=float("nan"))
    pred_path = tmp_path / "pred.csv"
    _write_pred_csv(pred_path, rows)
    monkeypatch.setattr(CB, "PRED", pred_path)

    panels = CB.per_series_panels()
    assert "S_ALLNAN" not in panels
    out = capsys.readouterr().out
    assert "[skip] S_ALLNAN: combination has no finite sae value" in out


# --- NaN y_true on one target date: must be dropped (isfinite filter), not propagate as NaN --

def test_per_series_panels_drops_dates_with_nan_y_true(monkeypatch, tmp_path):
    """One of 6 candidate dates has y_true=NaN -- a property of the evaluation panel rather than of
    any model, e.g. a date with no evaluation target. sae_comb is NaN there and must be filtered by
    the `np.isfinite(sae_comb)` guard (combination_benchmark.py:40), not carried into the per-year
    mean as a silent NaN-poisoned aggregate. 6 in, 1 dropped -> 5 survive."""
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    rows = []
    for yr in years:
        y_true = float("nan") if yr == 2015 else 100.0
        rows += _five_year_rows_single("S1", "A", yr, y_pred=110.0, y_true=y_true)
        rows += _five_year_rows_single("S1", "B", yr, y_pred=98.0, y_true=y_true)
    pred_path = tmp_path / "pred.csv"
    _write_pred_csv(pred_path, rows)
    monkeypatch.setattr(CB, "PRED", pred_path)

    panels = CB.per_series_panels()
    assert "S1" in panels
    L, T = panels["S1"]
    assert T == 5
    assert np.all(np.isfinite(L[CB.COMB]))
    np.testing.assert_allclose(L[CB.COMB], np.full(5, 4.0))
