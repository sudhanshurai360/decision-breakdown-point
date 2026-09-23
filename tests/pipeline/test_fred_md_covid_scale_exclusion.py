"""fred_md_sweep.py::series_losses() must compute its MASE/RMSSE scale on data that EXCLUDES the
real COVID window (2020-03-01 to 2021-12-31), matching the sibling curated-panel pipeline
(walkforward.py::year_scales(), used by run_multiseries.py/score_multiseries.py, which excludes it
via WF._not_covid()). Including it is a real, measured (2%-22% on real FRED-MD series)
scale-inflation divergence for any eval year whose training window (`tr = y[yr < Y]`) covers
2020-2021.

A read-only sensitivity check (not part of this test) confirmed the exclusion does NOT threaten
fredmd.law_r2_mcs_abs (draft_ssrn.md S4.3's one cited-significant regression): R^2 rises from
0.2079 to 0.2308 under the mask, strengthening rather than undermining the result. Only 1/74 real
FRED-MD series' MCS_abs classification changed (M1SL, 2->1).

Test design: build two synthetic series, identical except one has an artificial, large,
COVID-window-only shock (+1000) and the other doesn't. With a trivial constant-forecast model
(ignores its training data entirely, so its forecast never changes), the ONLY way the shocked
series' loss could differ from the clean series' is via the scale computation. After the fix, the
two must score IDENTICALLY for any eval year whose training window spans COVID -- the shock is
invisible to a correctly COVID-masked scale. Before the fix they would not (the shock leaks into
the unmasked scale, in whichever direction) -- checked explicitly as the discriminating half.
"""
import numpy as np
import pandas as pd
import pytest

import fred_md_sweep as FS
import walkforward as WF


def _dates():
    return pd.date_range("2000-01-01", "2023-12-01", freq="MS")


def _series(with_shock):
    dates = _dates()
    y = np.full(len(dates), 100.0)
    if with_shock:
        covid_lo, covid_hi = pd.Timestamp(WF.COVID[0]), pd.Timestamp(WF.COVID[1])
        y = np.where((dates >= covid_lo) & (dates <= covid_hi), 1100.0, y)
    yr = dates.year
    y[yr == 2023] = 105.0   # a known, fixed +5 error vs. the constant-100 forecast, in the one
    return y, dates         # eval year (2023) whose training window spans the COVID shock


@pytest.fixture(autouse=True)
def _constant_model(monkeypatch):
    # Ignores its training data entirely -- forecast never changes regardless of the shock, so any
    # difference in scored loss between the shocked and clean series can only come from the scale.
    monkeypatch.setattr(FS, "MODELS", {"const": lambda hist, m, h: np.repeat(100.0, h)})


def test_a_covid_only_shock_is_invisible_to_the_fixed_scale_computation():
    y_shocked, dates = _series(with_shock=True)
    y_clean, _ = _series(with_shock=False)

    La_shocked, _, models, years_shocked = FS.series_losses(y_shocked, dates)
    La_clean, _, _, years_clean = FS.series_losses(y_clean, dates)
    assert 2023 in years_shocked and 2023 in years_clean

    mase_shocked = La_shocked[years_shocked.index(2023), models.index("const")]
    mase_clean = La_clean[years_clean.index(2023), models.index("const")]
    assert mase_shocked == pytest.approx(mase_clean, rel=1e-9), (
        f"2023's MASE differs between the shocked ({mase_shocked}) and COVID-free ({mase_clean}) "
        f"series -- the COVID shock is leaking into the scale computation despite the mask."
    )


def test_the_same_shock_is_NOT_invisible_under_the_old_unmasked_scale_logic():
    """Adversarial check: reproduce the OLD (pre-fix) unmasked scale computation directly and
    confirm the shocked/clean pair DOES score differently under it -- proving the test above is
    discriminating real behavior (the mask genuinely doing something), not a vacuous equality that
    would hold regardless."""
    y_shocked, dates = _series(with_shock=True)
    y_clean, _ = _series(with_shock=False)
    yr = dates.year
    M = FS.M

    def old_unmasked_mase(y, Y):
        tr = y[yr < Y]; te = y[yr == Y]
        d12 = np.abs(tr[M:] - tr[:-M])
        scale_a = d12.mean() if len(d12) and d12.mean() > 0 else (np.abs(tr).mean() + 1e-9)
        e = te - 100.0   # the constant model's forecast
        return np.abs(e).mean() / scale_a

    mase_shocked_old = old_unmasked_mase(y_shocked, 2023)
    mase_clean_old = old_unmasked_mase(y_clean, 2023)
    assert mase_shocked_old != pytest.approx(mase_clean_old, rel=1e-6), (
        "the synthetic shock should measurably change 2023's MASE under the OLD unmasked scale "
        "logic -- if this fails, the test fixture isn't exercising the bug this fix addresses."
    )


def test_a_year_whose_training_window_never_touches_covid_is_unaffected_by_the_fix():
    """Negative control: an eval year entirely before COVID (training window never includes it)
    must be unaffected by masking -- the fix must not change scale computation for years where
    there was never anything to mask."""
    dates = _dates()
    yr = dates.year
    y = np.full(len(dates), 100.0)
    y[yr == 2010] = 105.0
    La, _, models, years = FS.series_losses(y, dates)
    assert 2010 in years
    mase_2010 = La[years.index(2010), models.index("const")]
    assert mase_2010 == pytest.approx(0.05, rel=0.05)   # 5 / (abs(tr).mean()+eps) = 5/100
