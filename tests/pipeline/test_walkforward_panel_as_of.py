"""walkforward.py's real-time-integrity properties: panel_as_of's realtime_start <= as_of boundary,
year_scales' leak-free prior-year window and COVID exclusion (year_scales' own docstring,
walkforward.py:76: "scale_Y over realized months with year < Y (leak-free, COVID excluded)"), and
run()'s dedup origin selection (its inline comment at walkforward.py:127: "dedupe (model, target):
keep the latest origin's forecast (most information)"). The sibling
tests/pipeline/test_walkforward_rmsse_not_mislabeled.py covers agg_metric()'s RMSSE handling only;
this file covers the no-leakage properties.

The SCOPE NOTE in that sibling file establishes why this matters downstream: run_multiseries.py
--combine's output is byte-identical to breadth_distinct_predictions.csv, which IS read by
canonical.py/metric_consistency.py -- so a bug here has live downstream consequences, not just
cosmetic ones.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import walkforward as WF  # noqa: E402


def _vintages(rows):
    """rows: list of (date, realtime_start, value) -- builds the [date, realtime_start, value]
    DataFrame panel_as_of()/realized() expect, with proper datetime dtypes."""
    df = pd.DataFrame(rows, columns=["date", "realtime_start", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df


def test_panel_as_of_boundary_is_inclusive():
    """panel_as_of's own docstring: "the value with the latest realtime_start <= as_of" -- a
    vintage published EXACTLY on the as_of date must be included, not excluded (a `<` vs `<=`
    off-by-one would silently look one publication behind on every real cutoff)."""
    v = _vintages([("2020-01-01", "2020-02-01", 100.0)])
    p = WF.panel_as_of(v, "2020-02-01")
    assert len(p) == 1 and p["value"].iloc[0] == 100.0


def test_panel_as_of_excludes_a_vintage_published_after_as_of():
    """Negative control: a vintage published one day after as_of must not leak into the panel --
    the exact real-time-leakage boundary this function exists to enforce."""
    v = _vintages([("2020-01-01", "2020-02-02", 100.0)])
    p = WF.panel_as_of(v, "2020-02-01")
    assert p.empty


def test_panel_as_of_picks_the_latest_known_vintage_per_month():
    """When a month has multiple vintages published by as_of, panel_as_of must return the value
    from the LATEST realtime_start among them -- the most-recently-known revision, not the first
    or an arbitrary one."""
    v = _vintages([
        ("2020-01-01", "2020-02-01", 100.0),   # first (preliminary) release
        ("2020-01-01", "2020-03-01", 101.0),   # first revision
        ("2020-01-01", "2020-04-01", 102.0),   # second revision, published AFTER as_of below
    ])
    p = WF.panel_as_of(v, "2020-03-15")
    assert len(p) == 1 and p["value"].iloc[0] == 101.0, (
        "expected the 2020-03-01 revision (latest one known by 2020-03-15), not the preliminary "
        "or the not-yet-published second revision"
    )


def test_panel_as_of_returns_one_row_per_month_across_a_real_time_history():
    """A short, realistic multi-month real-time history: panel_as_of at a late as_of date must
    return exactly one row per calendar month, each holding that month's latest-known-by-then
    value, sorted by date."""
    v = _vintages([
        ("2020-01-01", "2020-02-01", 10.0),
        ("2020-02-01", "2020-03-01", 20.0),
        ("2020-03-01", "2020-04-01", 30.0),
    ])
    p = WF.panel_as_of(v, "2020-12-01")
    assert list(p["date"]) == list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    assert list(p["value"]) == [10.0, 20.0, 30.0]


def test_year_scales_excludes_a_diff_touching_covid_on_either_endpoint():
    """year_scales' COVID exclusion masks a seasonal difference if EITHER endpoint (t or t-m) falls
    inside the COVID window. The fixture is built so the excluded and retained pairs have WILDLY
    different magnitudes, making the filter numerically detectable: with the exclusion the answer is
    12.0, without it 752.0. A flat or linear fixture would give the same number either way and the
    test would pass even with the filter stubbed out."""
    # 25 months: Jan-2019 .. Jan-2021. For years=[2021] only 2019-2020 are in scope, and m=12
    # pairs each 2020 month with the same month of 2019.
    dates = pd.date_range("2019-01-01", periods=25, freq="MS")
    values = np.full(25, 100.0)
    values[12:14] = 112.0     # 2020-01, 2020-02: outside COVID, seasonal diff 12.0
    values[14:24] = 1000.0    # 2020-03 .. 2020-12: inside COVID, seasonal diff 900.0
    truth = pd.DataFrame({"date": dates, "value": values})

    keep = WF._not_covid(truth["date"])
    assert not keep.all(), "test setup error: expected some months to fall inside the COVID window"

    out = WF.year_scales(truth, years=[2021], m=12, squared=False)
    # Only the two non-COVID pairs survive -> mean(|12.0|, |12.0|).
    assert out[2021] == pytest.approx(12.0), out[2021]
    # Without the exclusion all 12 pairs would count: (2*12 + 10*900) / 12 = 752.0.
    assert out[2021] != pytest.approx(752.0)


def test_year_scales_uses_only_strictly_prior_years_no_leakage():
    """year_scales(years=[Y]) must compute its scale from data strictly BEFORE year Y -- the
    module's own "leak-free" claim. Appending future (year Y and later) data with a wildly
    different magnitude must not move the scale for year Y at all."""
    dates_hist = pd.date_range("2015-01-01", periods=60, freq="MS")  # 2015-01 .. 2019-12
    values_hist = 100.0 + np.sin(np.arange(60) / 3.0)
    truth_hist = pd.DataFrame({"date": dates_hist, "value": values_hist})
    scale_hist = WF.year_scales(truth_hist, years=[2020], m=12, squared=False)[2020]

    dates_future = pd.date_range("2020-01-01", periods=12, freq="MS")
    values_future = np.full(12, 1e9)  # absurd future magnitude
    truth_full = pd.concat([truth_hist, pd.DataFrame({"date": dates_future, "value": values_future})],
                            ignore_index=True)
    scale_with_future = WF.year_scales(truth_full, years=[2020], m=12, squared=False)[2020]

    assert scale_hist == scale_with_future, (
        f"year_scales for 2020 changed after appending 2020 data ({scale_hist} -> "
        f"{scale_with_future}) -- future data leaked into a supposedly leak-free scale"
    )


def test_year_scales_returns_nan_for_a_year_with_insufficient_prior_history():
    """A year with fewer than `m` strictly-prior observations must return NaN, not a garbage
    scale from an underfull window."""
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    truth = pd.DataFrame({"date": dates, "value": np.arange(6, dtype=float) + 50.0})
    out = WF.year_scales(truth, years=[2020], m=12, squared=False)
    assert np.isnan(out[2020])


def test_year_scales_floors_a_perfectly_flat_window_instead_of_dividing_by_zero():
    """year_scales() floors its scale at 1e-9 via `max(np.mean(diffs[dkeep]), 1e-9)` -- run()'s
    sae/msse divide by this value with no guard of their own. (m_audit.py::scores() guards the
    analogous MASE-style denominator too, but by a DIFFERENT mechanism: an additive
    `np.abs(y_train).mean() + 1e-9` fallback taken only when the mean diff is non-positive, not a
    max()-floor -- same 1e-9 magnitude, not the same convention.) Real data never hits exactly zero (min scale_abs 2.497 across 26,544
    rows), but a perfectly flat m-month window (constructed here, not real) must not silently
    produce a zero-valued scale that would later divide-by-zero into inf/nan uncaught."""
    dates = pd.date_range("2015-01-01", periods=36, freq="MS")
    truth = pd.DataFrame({"date": dates, "value": np.full(36, 100.0)})  # perfectly flat -> d == 0 everywhere
    out = WF.year_scales(truth, years=[2018], m=12, squared=False)
    assert out[2018] == pytest.approx(1e-9), out[2018]
    assert out[2018] > 0.0


def test_year_scales_floor_does_not_perturb_any_real_nonzero_scale():
    """THE NO-REGRESSION PATH: the floor must be a true max()-floor, not an additive one -- any
    real, already-nonzero scale (every one ever observed in this codebase's real data is >> 1e-9)
    must come back bit-for-bit unperturbed, so the floor cannot move any reported number."""
    dates = pd.date_range("2015-01-01", periods=36, freq="MS")
    values = 100.0 + np.sin(np.arange(36) / 3.0) * 5.0  # genuinely varying, real-scale-like values
    truth = pd.DataFrame({"date": dates, "value": values})
    out = WF.year_scales(truth, years=[2018], m=12, squared=False)
    assert out[2018] > 1e-6  # comfortably real-valued, not floor-dominated
    # recompute the unfloored formula directly and confirm an exact match
    t = truth.sort_values("date").reset_index(drop=True)
    val = t["value"].to_numpy(float)
    idx = np.where(t["date"].dt.year.to_numpy() < 2018)[0]
    d = val[idx][12:] - val[idx][:-12]
    expected_unfloored = float(np.mean(np.abs(d)))
    assert out[2018] == pytest.approx(expected_unfloored, abs=0, rel=1e-15)


class _LastValueModel:
    """Trivial deterministic forecaster matching models_freq.py's own convention (a plain scalar,
    not an array -- see SeasonalNaive.forecast): predicts exp(last observed y) unchanged (log-space
    persistence), enough to make run()'s per-origin forecast values hand-predictable."""
    name = "last"

    def forecast(self, y, h=1):
        return float(y[-1])


def test_run_dedup_keeps_the_later_origins_forecast_not_the_earlier(monkeypatch):
    """run()'s own comment: "dedupe (model, target): keep the latest origin's forecast (most
    information)" -- and specifically LATEST, not first. Constructs a scenario where the SAME
    target month is reachable from two INFORMATIONALLY DIFFERENT origins: 2020-02's release is
    delayed (so both the 2020-02-01 and 2020-03-01 cutoffs see 2020-01 as their latest month, both
    computing target=2020-02-01), and 2020-01's own value is first published preliminarily as
    999.0, then REVISED to 888.0 before the second of those two origins. A model that echoes back
    the last training value therefore forecasts 999.0 from the earlier origin and 888.0 from the
    later one -- a real, keep="last"-vs-keep="first" discriminating case, unlike a plateau where
    every colliding origin sees identical data (see the structural dedup test below)."""
    dates = pd.date_range("2005-01-01", "2021-06-01", freq="MS")  # long enough to clear run()'s
    values = np.linspace(100.0, 200.0, len(dates))                 # own len(p) < 2*m+8 gate
    v = pd.DataFrame({
        "date": dates, "realtime_start": dates + pd.offsets.MonthBegin(1), "value": values,
    })
    v.loc[v.date == "2020-02-01", "realtime_start"] = pd.Timestamp("2020-05-01")  # delayed release
    prelim = v[v.date == "2020-01-01"].copy()
    prelim["value"] = 999.0
    prelim["realtime_start"] = pd.Timestamp("2020-02-01")            # preliminary, seen by Feb-01 origin
    v.loc[v.date == "2020-01-01", ["realtime_start", "value"]] = [pd.Timestamp("2020-03-01"), 888.0]
    v = pd.concat([v, prelim], ignore_index=True)                    # revision, seen by Mar-01 origin

    monkeypatch.setattr(WF, "load_vintages", lambda sid: v)
    df = WF.run("FAKE", [_LastValueModel()], eval_start=2020, eval_end=2020, m=12)

    row = df[df["target"] == pd.Timestamp("2020-02-01")]
    assert len(row) == 1, "expected the two colliding origins to collapse to exactly one row"
    assert row["y_pred"].iloc[0] == pytest.approx(888.0), (
        f"expected the LATER origin's (revised, 888.0) forecast to survive dedup, got "
        f"{row['y_pred'].iloc[0]} -- looks like the EARLIER (preliminary, 999.0) origin won instead"
    )


def test_run_dedup_collapses_repeated_origins_for_the_same_target(monkeypatch):
    """Structural companion to the test above: a realistic publication-delay scenario -- two
    consecutive months (2020-02, 2020-03) both released late, together, in a single May batch --
    makes THREE different monthly origins (2020-02-01, 2020-03-01, 2020-04-01) all see the same
    latest-known month (2020-01) and therefore compute the same target (2020-02-01), independently
    confirmed below via panel_as_of before ever calling run(). Post-dedup, exactly ONE row for that
    target must survive (this scenario's 3 origins are informationally IDENTICAL, so it cannot
    discriminate keep="last" from keep="first" -- that's what the test above is for)."""
    dates = pd.date_range("2005-01-01", "2021-06-01", freq="MS")
    values = np.linspace(100.0, 200.0, len(dates))
    v = pd.DataFrame({
        "date": dates, "realtime_start": dates + pd.offsets.MonthBegin(1), "value": values,
    })
    for late_month in ("2020-02-01", "2020-03-01"):
        v.loc[v.date == late_month, "realtime_start"] = pd.Timestamp("2020-05-01")

    # Confirm the intended duplicate-origin scenario is real BEFORE dedup, independent of run().
    origins_hitting_target = [
        c for c in pd.date_range("2020-01-01", "2020-06-01", freq="MS")
        if (WF.panel_as_of(v, c)["date"].max() + pd.offsets.MonthBegin(1)).normalize()
        == pd.Timestamp("2020-02-01")
    ]
    assert len(origins_hitting_target) == 3, (
        f"test setup error: expected 3 origins to collide on target=2020-02-01, "
        f"got {len(origins_hitting_target)}"
    )

    monkeypatch.setattr(WF, "load_vintages", lambda sid: v)
    df = WF.run("FAKE", [_LastValueModel()], eval_start=2020, eval_end=2020, m=12)

    assert not df.empty
    assert not df.duplicated(["model", "target"]).any(), (
        "duplicate (model, target) rows survived -- the dedup step did not collapse them"
    )
    assert (df["target"] == pd.Timestamp("2020-02-01")).sum() == 1
