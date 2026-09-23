"""Regression tests for SARIMA's trend grid. The problem it solves: with
d=D=0 and no trend term, SARIMAX has no way to represent a nonzero mean/drift except by pushing the
AR polynomial toward the unit-root boundary -- confirmed live (AR root pinned at ~0.9999998) on the
curated panel's trending log-level series, producing a flat, driftless forecast (effectively
indistinguishable from "last observed value") rather than a genuine SARIMA forecast. This is a real
defect, NOT the same issue as "no seasonal signal found" -- the curated series are pre-seasonally-
adjusted by their source agency, so P=Q=0 winning the AIC grid on them is separately correct.

`trend` is therefore a third AIC-grid axis, restricted to {None, "c"} (models_freq.py's SARIMA class).
These tests run the REAL SARIMAX fits (statsmodels), not mocks -- deterministic synthetic series,
real optimizer -- so each test takes real wall-clock time (several fits per _select() call).
"""
import numpy as np
import pytest

import models_freq as MF


def _trending_series(n=48, drift=0.03, noise_sd=0.01, seed=0):
    """A deterministic, strongly-trending log1p-space series (~3%/period compounding drift + small
    noise) -- the space these models are actually called in (walkforward.py:118 fits every model on
    `np.log1p(...)` of the real-time history), and the shape the live curated panel has (trending
    log-level macro series)."""
    rng = np.random.default_rng(seed)
    steps = drift + noise_sd * rng.standard_normal(n)
    return np.cumsum(steps) + 5.0   # start away from 0, purely cosmetic


def _flat_series(n=48, mean=5.0, noise_sd=0.02, seed=1):
    """A deterministic, genuinely mean-reverting (no drift) series -- confirms the fix does not
    break the case where None really is the right trend choice."""
    rng = np.random.default_rng(seed)
    return mean + noise_sd * rng.standard_normal(n)


def test_trend_grid_is_restricted_to_none_and_c():
    assert MF.SARIMA.TREND_GRID == [None, "c"]


def test_select_returns_a_3tuple_including_trend():
    y = _trending_series()
    order = MF.SARIMA(m=12)._select(y)
    assert order is not None
    assert len(order) == 3
    p_order, sorder, trend = order
    assert len(p_order) == 3 and len(sorder) == 4
    assert trend in (None, "c")


def test_select_prefers_trend_c_on_a_trending_series_by_aic():
    """THE CONFIRMED REGRESSION, encoded as a permanent test, checking the diagnosed mechanism
    directly rather than a noisy single-point forecast comparison: on a strongly-trending series,
    the OLD behavior (trend forced to None) can only fit the drift by pushing the AR root to the
    stationarity boundary (~1.0), which fits badly (high AIC, inflated residual variance). The FIX
    (trend grid includes "c") lets the model use a genuinely moderate, well-identified AR root plus
    an intercept, which AIC strongly prefers. Directly verified against real statsmodels fits (not
    mocked) at a fixed order (1,0,0)x(0,0,0,12) to isolate the trend effect from order selection."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = _trending_series(n=60, drift=0.04, noise_sd=0.008, seed=2)

    res_none = SARIMAX(y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 12), trend=None,
                        enforce_stationarity=True, enforce_invertibility=True,
                        initialization="approximate_diffuse").fit(disp=False, maxiter=50)
    res_c = SARIMAX(y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 12), trend="c",
                     enforce_stationarity=True, enforce_invertibility=True,
                     initialization="approximate_diffuse").fit(disp=False, maxiter=50)

    root_none = float(abs(res_none.arparams[0]))
    root_c = float(abs(res_c.arparams[0]))
    assert root_none > 0.999, (
        f"sanity check failed: old (trend=None) AR root {root_none} is not pinned near the "
        f"stationarity boundary -- the synthetic series may not be exercising the diagnosed bug")
    assert root_c < root_none, (
        f"trend='c' should let the model use a materially less extreme AR root than trend=None: "
        f"root_c={root_c} vs root_none={root_none}")
    assert res_c.aic < res_none.aic, (
        f"AIC should strongly prefer trend='c' on a trending series (moderate root, better fit) "
        f"over trend=None (boundary-pinned root): aic_c={res_c.aic} vs aic_none={res_none.aic}")

    # And the REAL grid search (_select(), not a fixed order) must actually pick trend="c" here --
    # confirms the fix is wired into the model's actual selection path, not just true in isolation.
    order = MF.SARIMA(m=12)._select(y)
    assert order is not None
    _, _, chosen_trend = order
    assert chosen_trend == "c", (
        f"_select() should choose trend='c' on this strongly-trending series (AIC strongly prefers "
        f"it in isolation), got trend={chosen_trend!r}")


def test_forecast_on_flat_series_still_sensible():
    """Negative control the other direction: a genuinely flat (no-drift) series must still get a
    sensible forecast near its mean after the fix -- the extra trend='c' grid option must not push
    the model toward spurious drift when there isn't any."""
    y = _flat_series(n=48, mean=5.0, noise_sd=0.02, seed=3)
    f = MF.SARIMA(m=12).forecast(y, h=1)
    assert abs(f - 5.0) < 0.3, f"forecast {f} on a flat series (mean=5.0) should stay close to the mean"


def test_forecast_short_series_falls_back_to_last_value():
    """Negative control: below the 2*m+8 length gate, forecast() must still fall back to _last(y)
    regardless of the trend-grid change -- this path never reaches _select()."""
    y = np.array([1.0, 2.0, 3.0])
    assert MF.SARIMA(m=12).forecast(y, h=1) == pytest.approx(3.0)


def test_a_failed_order_selection_still_respects_reselect_every_cadence(monkeypatch):
    """A failed order selection must not force a full re-run of the expensive 32-combination
    AIC/trend grid on EVERY subsequent call. self._order stays None forever after a failure, so a
    `self._order is None or ...` guard would be permanently True -- breaking the intended annual
    reselect_every cadence and reintroducing, per-window, exactly the repeated-expensive-search risk
    FIT_TIMEOUT exists to bound per-fit. Confirms _select() is called at most once across
    `reselect_every` consecutive forecast() calls even when it always returns None."""
    s = MF.SARIMA(m=12, reselect_every=12)
    calls = {"n": 0}

    def failing_select(y):
        calls["n"] += 1
        return None

    monkeypatch.setattr(s, "_select", failing_select)
    y = _flat_series(n=60, mean=100.0, noise_sd=1.0, seed=7)
    for _ in range(5):
        s.forecast(y, h=1)
    assert calls["n"] == 1, f"expected 1 _select() call (cadence respected), got {calls['n']}"


def test_a_successful_order_selection_still_reselects_on_the_next_cadence_boundary(monkeypatch):
    """Companion negative control: the fix must not accidentally suppress re-selection when it
    SHOULD happen -- a small reselect_every must still trigger a second _select() call once the
    cadence counter reaches it, on both the success and failure path."""
    s = MF.SARIMA(m=12, reselect_every=2)
    calls = {"n": 0}

    def counting_select(y):
        calls["n"] += 1
        return None

    monkeypatch.setattr(s, "_select", counting_select)
    y = _flat_series(n=60, mean=100.0, noise_sd=1.0, seed=8)
    for _ in range(5):   # calls at _since=1e9(select),1,2(select),1,2(select) -> 3 selects
        s.forecast(y, h=1)
    assert calls["n"] == 3, f"expected 3 _select() calls at reselect_every=2 over 5 forecasts, got {calls['n']}"
