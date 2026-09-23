"""Mutation-verified tests for models_freq.py.

Mutation survivors from a blind scan: N1 ETS seasonal branch damped, N7 ETS seasonal_periods = m-1,
N2 Theta period m//2, N8 Theta deseasonalize=False, N5 DHR K=1. None changes a fallback path or a
short-series guard, which is what the existing tests cover; each changes the FORECAST on an
ordinary long seasonal series. The fixtures below are deterministic trend + sine series whose
continuation IS the hand answer: the real classes reproduce it to ~1e-3 and
each mutant misses it by 1 to 20 units.
"""
import numpy as np
import pytest

import models_freq as MF

M = 12
N = 120


def _trend_sine(t):
    """100 + 0.5 t + 10 sin(2 pi t/12) + 4 sin(2 pi 3t/12): period-12 seasonal with a THIRD harmonic,
    so a K=1 Fourier basis cannot represent it, plus a linear trend."""
    t = np.asarray(t, float)
    return 100.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / M) + 4.0 * np.sin(2 * np.pi * 3 * t / M)


def _pure_sine(t):
    """100 + 10 sin(2 pi t/12): a stationary period-12 series whose multiplicative seasonal
    decomposition is exact, so the Theta method's deseasonalize/reseasonalize round-trip is exact."""
    t = np.asarray(t, float)
    return 100.0 + 10.0 * np.sin(2 * np.pi * t / M)


# --- ETS ---------------------------------------------------------------------------------------

def test_ets_seasonal_branch_continues_a_trend_plus_seasonal_exactly_at_h1_and_h24():
    """kills: N1 (damped seasonal branch: h=24 forecast of a pure linear trend falls short by
    ~1.5 units because statsmodels caps phi at 0.98) and N7 (seasonal_periods = m-1: the
    period-12 pattern is misaligned by a month, h=1 error ~9.6). The real additive-trend,
    additive-seasonal Holt-Winters fit reproduces the deterministic series to < 1e-3."""
    y = _trend_sine(np.arange(N))
    ets = MF.ETS(m=M)
    for h in (1, 24):
        f = ets.forecast(y, h=h)
        assert f == pytest.approx(_trend_sine(N - 1 + h), abs=5e-2), (
            f"ETS h={h}: got {f}, expected {_trend_sine(N - 1 + h)}")


def test_ets_seasonal_branch_is_built_undamped_with_seasonal_periods_m(monkeypatch):
    """kills: N1, N7 (specification capture). The seasonal branch must construct
    ExponentialSmoothing(trend='add', seasonal='add', seasonal_periods=m) with NO damping -- the
    'fixed additive trend' the paper's model description states."""
    captured = {}

    class _Rec:
        def __init__(self, y, **kw):
            captured.update(kw)

        def fit(self):
            class _R:
                def forecast(self, h):
                    return np.array([1.0] * h)
            return _R()

    monkeypatch.setattr(MF, "ExponentialSmoothing", _Rec)
    MF.ETS(m=M).forecast(_trend_sine(np.arange(N)), h=1)
    assert captured.get("trend") == "add" and captured.get("seasonal") == "add"
    assert captured.get("seasonal_periods") == M, captured
    assert not captured.get("damped_trend", False), captured


# --- Theta -------------------------------------------------------------------------------------

def test_theta_continues_a_pure_period_12_seasonal_exactly():
    """kills: N2 (period = m//2 = 6: a period-12 sine is not periodic at lag 6, h=3 error ~10)
    and N8 (deseasonalize=False: the SES-plus-drift line ignores the seasonal, h=3 error ~10).
    The real ThetaModel(period=12) deseasonalizes exactly and forecasts the sine's continuation."""
    y = _pure_sine(np.arange(N))
    th = MF.Theta(m=M)
    for h in (1, 3, 6):
        f = th.forecast(y, h=h)
        assert f == pytest.approx(_pure_sine(N - 1 + h), abs=5e-2), (
            f"Theta h={h}: got {f}, expected {_pure_sine(N - 1 + h)}")


def test_theta_is_built_with_period_m_and_default_deseasonalisation(monkeypatch):
    """kills: N2, N8 (specification capture)."""
    captured = {}

    class _Rec:
        def __init__(self, y, **kw):
            captured.update(kw)

        def fit(self):
            class _R:
                def forecast(self, h):
                    return np.array([1.0] * h)
            return _R()

    monkeypatch.setattr(MF, "ThetaModel", _Rec)
    MF.Theta(m=M).forecast(_pure_sine(np.arange(N)), h=1)
    assert captured.get("period") == M, captured
    assert captured.get("deseasonalize", True) is True, captured


# --- DHR ---------------------------------------------------------------------------------------

def test_dhr_default_harmonics_is_four_at_m12():
    """kills: N5 (K=1). max(1, min(4, 12 // 3)) == 4, the worked example in the inline comment at
    models_freq.py:204 ("for m=12 -> 4, for m=52 -> 4 (capped)")."""
    assert MF.DHR(m=12).K == 4
    assert MF.DHR(m=52).K == 4
    assert MF.DHR(m=4).K == 1
    assert MF.DHR(m=12, K=2).K == 2


def test_dhr_captures_a_third_harmonic_that_a_single_harmonic_cannot():
    """kills: N5 (K=1: the 3rd-harmonic component of the fixture is unmodelled, h=1 error ~1.25).
    With K=4 the Fourier basis spans harmonics 1..4 and the fit is exact to < 1e-2."""
    y = _trend_sine(np.arange(N))
    f = MF.DHR(m=M).forecast(y, h=1)
    assert f == pytest.approx(_trend_sine(N), abs=5e-2), f"DHR h=1: got {f}, expected {_trend_sine(N)}"
