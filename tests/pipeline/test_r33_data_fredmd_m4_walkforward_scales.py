"""Mutation-verified tests for walkforward.py.

Number-changing mutations were planted in walkforward.py that every existing test
passed under:
  M1  year_scales' COVID mask made a no-op
  M6  COVID window end moved from 2021-12-31 to 2020-12-31
  M2  run()'s `sae` divided by the SQUARED scale
  M7  run()'s `sae` left signed (no abs)
Each test below carries a hand-computed answer on a synthetic fixture and is mutation-verified
(fails on the named mutant, passes on the real module).
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


# --------------------------------------------------------------------------------------------
# year_scales: the COVID mask must actually remove a shocked month, on BOTH window edges.
# Fixture: a linear series (constant lag-12 difference of exactly 12.0) with one +1000 spike.
# If the spike month is excluded from every lag-12 pair, the scale is exactly 12.0; if it leaks
# into either of the two pairs it touches (t, t-12) and (t+12, t), the mean moves to
# 12 + (|12+1000| + |12-1000| - 24)/N = 12 + 1976/N, far from 12.
# --------------------------------------------------------------------------------------------

def _linear_truth_with_spike(spike_month: str, start="2015-01-01", end="2023-12-01", spike=1000.0):
    dates = pd.date_range(start, end, freq="MS")
    values = 100.0 + np.arange(len(dates), dtype=float)          # lag-12 diff == 12.0 everywhere
    truth = pd.DataFrame({"date": dates, "value": values})
    truth.loc[truth["date"] == pd.Timestamp(spike_month), "value"] += spike
    return truth


@pytest.mark.parametrize("spike_month", ["2020-03-01", "2020-09-01", "2021-06-01", "2021-12-01"])
def test_year_scales_drops_a_spike_anywhere_inside_the_covid_window(spike_month):
    """kills: M1 (COVID mask no-op in year_scales) and M6 (window end 2020-12-31 -- the 2021-06 and
    2021-12 spikes then leak in). Scale for 2024 over 2015-2023: every pair touching the spike
    month is masked, all remaining pairs have |diff| == 12, so the answer is exactly 12.0."""
    truth = _linear_truth_with_spike(spike_month)
    out = WF.year_scales(truth, years=[2024], m=12, squared=False)
    assert out[2024] == pytest.approx(12.0, abs=1e-9), (
        f"a +1000 spike at {spike_month} (inside COVID={WF.COVID}) leaked into the 2024 scale: {out[2024]}")
    out_sq = WF.year_scales(truth, years=[2024], m=12, squared=True)
    assert out_sq[2024] == pytest.approx(144.0, abs=1e-9)


@pytest.mark.parametrize("spike_month", ["2020-02-01", "2022-01-01"])
def test_year_scales_keeps_a_spike_one_month_outside_the_covid_window(spike_month):
    """Companion boundary check (the window must not be WIDER than 2020-03..2021-12 either): a spike
    one month before the start or one month after the end is real data and must move the scale.
    Hand count for 2015-01..2023-12 (108 months, 96 lag-12 pairs): the spike's two pairs are
    (spike, spike-12) and (spike+12, spike); one of them has its other endpoint inside COVID and
    is masked anyway, the other survives with |diff| = 1012 or 988."""
    truth = _linear_truth_with_spike(spike_month)
    out = WF.year_scales(truth, years=[2024], m=12, squared=False)
    keep = WF._not_covid(truth["date"])
    d = np.abs(truth["value"].to_numpy()[12:] - truth["value"].to_numpy()[:-12])
    dkeep = keep[12:] & keep[:-12]
    expected = float(d[dkeep].mean())                     # the independent hand formula
    assert expected > 12.5, "fixture error: the outside-window spike should survive masking"
    assert out[2024] == pytest.approx(expected, rel=1e-12)


def test_covid_window_is_exactly_2020_03_through_2021_12():
    """kills: M6. Pins the constant the paper's S3.1/S3.2 scale-exclusion sentence rests on."""
    assert WF.COVID == ("2020-03-01", "2021-12-31")
    keep = WF._not_covid(pd.Series(pd.to_datetime(["2020-02-01", "2020-03-01", "2021-12-01", "2022-01-01"])))
    assert keep.tolist() == [True, False, False, True]


# --------------------------------------------------------------------------------------------
# run(): sae = |err| / scale_abs and msse = err^2 / scale_sq, hand-computed on a DECREASING
# linear series so the error is negative (-1.0) and the two scales differ (12 vs 144).
# --------------------------------------------------------------------------------------------

class _LastValueModel:
    """Log-space persistence: forecast = last observed log1p value, so y_pred = last level."""
    name = "last"

    def forecast(self, y, h=1):
        return float(y[-1])


def _decreasing_vintages():
    dates = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
    values = 5000.0 - np.arange(len(dates), dtype=float)          # v_t - v_{t-1} = -1, lag-12 = -12
    return pd.DataFrame({"date": dates, "realtime_start": dates + pd.offsets.MonthBegin(1),
                         "value": values})


def test_run_sae_is_abs_error_over_scale_abs_and_msse_is_sq_error_over_scale_sq(monkeypatch):
    """kills: M2 (sae / scale_sq) and M7 (signed sae). Persistence on a series falling by 1 per
    month gives err = y_true - y_pred = -1 at every target; scale_abs(2019) = mean |lag-12 diff|
    over 2006-2018 = 12, scale_sq = 144. So sae must be +1/12 (positive!) and msse 1/144."""
    monkeypatch.setattr(WF, "load_vintages", lambda sid: _decreasing_vintages())
    df = WF.run("SYN", [_LastValueModel()], eval_start=2019, eval_end=2019, m=12)
    assert len(df) == 12, df
    err = df["y_true"] - df["y_pred"]
    assert np.allclose(err, -1.0, atol=1e-9)
    assert np.allclose(df["scale_abs"], 12.0, atol=1e-9)
    assert np.allclose(df["scale_sq"], 144.0, atol=1e-9)
    assert np.allclose(df["sae"], 1.0 / 12.0, atol=1e-9), df["sae"].tolist()
    assert (df["sae"] > 0).all(), "sae must be an ABSOLUTE scaled error (MASE numerator), not signed"
    assert np.allclose(df["msse"], 1.0 / 144.0, atol=1e-12), df["msse"].tolist()


def test_agg_metric_on_the_same_fixture_gives_mase_1_12_and_rmsse_1_12(monkeypatch):
    """Downstream sanity on the same fixture: MASE = mean(sae) = 1/12 and RMSSE = sqrt(mean(msse))
    = sqrt(1/144) = 1/12 -- equal here only because the error is constant, which is exactly what
    makes the pair hand-checkable."""
    monkeypatch.setattr(WF, "load_vintages", lambda sid: _decreasing_vintages())
    df = WF.run("SYN", [_LastValueModel()], eval_start=2019, eval_end=2019, m=12)
    mase = WF.agg_metric(df, "mase")
    rmsse = WF.agg_metric(df, "rmsse")
    assert mase["loss"].iloc[0] == pytest.approx(1 / 12)
    assert rmsse["loss"].iloc[0] == pytest.approx(1 / 12)
    assert int(mase["n"].iloc[0]) == 12
