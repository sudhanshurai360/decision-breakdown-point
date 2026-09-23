"""Regression tests for models_freq.py's SeasonalNaive/AR1/ETS/Theta classes. DHR and SARIMA
already have dedicated coverage (test_models_freq.py,
test_models_freq_sarima.py); this file closes the remaining 4 classes.
"""
import numpy as np
import pytest

import models_freq as MF


# --- SeasonalNaive -----------------------------------------------------------------------------

def test_seasonal_naive_h1_returns_value_m_steps_back():
    y = np.arange(24, dtype=float)   # 0..23, m=12 -> idx = 24-12+0 = 12
    m = MF.SeasonalNaive(m=12)
    assert m.forecast(y, h=1) == 12.0


def test_seasonal_naive_h_gt_m_cycles_not_falls_flat():
    """h > m must wrap cyclically back into the same seasonal phase as h=1, not fall through to
    flat last-value persistence."""
    y = np.arange(24, dtype=float)
    m = MF.SeasonalNaive(m=12)
    h1 = m.forecast(y, h=1)     # idx = 24-12+0 = 12
    h13 = m.forecast(y, h=13)   # (13-1) % 12 = 0 -> same idx as h=1
    assert h13 == h1 == 12.0
    h2 = m.forecast(y, h=2)     # idx = 24-12+1 = 13
    h14 = m.forecast(y, h=14)   # (14-1) % 12 = 1 -> same idx as h=2
    assert h14 == h2 == 13.0


def test_seasonal_naive_falls_back_to_last_value_when_history_too_short():
    y = np.array([5.0, 6.0, 7.0])   # len=3 < m=12 -> idx negative -> falls back
    m = MF.SeasonalNaive(m=12)
    assert m.forecast(y, h=1) == 7.0   # _last(y)


# --- AR1 -----------------------------------------------------------------------------------------

def test_ar1_falls_back_to_last_value_below_length_4():
    y = np.array([1.0, 2.0, 3.0])
    m = MF.AR1(m=12)
    assert m.forecast(y, h=1) == 3.0


def test_ar1_hand_computed_against_manual_ols():
    """Hand-derive the OLS AR(1)-on-differences coefficient and h-step iteration independently of
    the class, and assert the class matches exactly -- not just "runs without crashing"."""
    y = np.array([10.0, 11.0, 13.0, 14.0, 14.5, 16.0, 17.5])
    dy = np.diff(y)               # [1, 2, 1, 0.5, 1.5, 1.5]
    x, z = dy[:-1], dy[1:]
    xm, zm = x.mean(), z.mean()
    denom = float(((x - xm) ** 2).sum())
    phi = float(((x - xm) * (z - zm)).sum() / denom)
    c = zm - phi * xm
    d, last = dy[-1], y[-1]
    val = last
    for _ in range(3):
        d = c + phi * d
        val = val + d

    m = MF.AR1(m=12)
    assert m.forecast(y, h=3) == pytest.approx(val)


def test_ar1_clamps_phi_to_avoid_explosive_extrapolation():
    """A perfectly explosive difference sequence (each difference exactly double the last) makes the
    raw OLS phi exactly 2.0, well outside the stationary region -- the class clamps it to +-0.999
    (models_freq.py:89). The clamped h-step iteration is hand-derived here and asserted exactly:
    with the clamp this call returns 315.4896...; drop the clamp and the same call returns 2047.0,
    which is still finite, so merely asserting finiteness would not catch the loss of the clamp."""
    y = np.concatenate([[0.0], np.cumsum(np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))])  # 0,1,3,..,63
    dy = np.diff(y)
    x, z = dy[:-1], dy[1:]
    xm, zm = x.mean(), z.mean()
    raw_phi = float(((x - xm) * (z - zm)).sum() / ((x - xm) ** 2).sum())
    assert raw_phi == pytest.approx(2.0), "fixture no longer drives phi outside the clamp"

    phi = 0.999                      # the clamp the class applies
    c = zm - phi * xm
    d, val = dy[-1], y[-1]
    for _ in range(5):
        d = c + phi * d
        val = val + d
    assert val == pytest.approx(315.489608575961)

    out = MF.AR1(m=12).forecast(y, h=5)
    assert out == pytest.approx(val)
    assert out < 400, "unclamped, the same iteration reaches 2047.0"


# --- ETS -----------------------------------------------------------------------------------------

def test_ets_short_series_returns_last_value_without_fitting(monkeypatch):
    """n < 10 must hit the `else: return _last(y)` branch and never reach statsmodels at all."""
    called = {"n": 0}
    monkeypatch.setattr(MF, "ExponentialSmoothing",
                         lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    y = np.arange(9, dtype=float)
    m = MF.ETS(m=12)
    assert m.forecast(y, h=1) == 8.0
    assert called["n"] == 0


def test_ets_m_equals_1_forces_non_seasonal_branch_even_with_long_series(monkeypatch):
    """self.m > 1 is required for the seasonal branch even when n is otherwise long enough --
    m=1 must always take the damped-Holt (non-seasonal) path."""
    seen_seasonal = {}

    class _FakeFit:
        def forecast(self, h):
            return np.array([1.23] * h)

    def fake_es(y, trend=None, seasonal=None, seasonal_periods=None, damped_trend=None,
                initialization_method=None):
        seen_seasonal["seasonal"] = seasonal
        return type("R", (), {"fit": lambda self: _FakeFit()})()

    monkeypatch.setattr(MF, "ExponentialSmoothing", fake_es)
    y = np.arange(30, dtype=float)
    m = MF.ETS(m=1)
    out = m.forecast(y, h=1)
    assert seen_seasonal["seasonal"] is None   # non-seasonal (damped Holt) branch taken
    assert out == 1.23


def test_ets_exception_falls_back_to_last_value_and_warns_once(monkeypatch, capsys):
    monkeypatch.setattr(MF, "_FALLBACK_TALLY", {})   # isolate from other tests' fallback state

    def boom(*a, **k):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(MF, "ExponentialSmoothing", boom)
    y = np.arange(30, dtype=float)
    m = MF.ETS(m=12)
    out = m.forecast(y, h=1)
    assert out == y[-1]
    err = capsys.readouterr().err
    assert "[warn] ETS.forecast() raised ValueError" in err


def test_ets_real_seasonal_fit_runs_end_to_end_and_returns_finite():
    """No mocking: a real statsmodels fit on a synthetic seasonal series, sanity-checking the
    seasonal branch is actually reachable and produces a finite forecast."""
    rng = np.random.default_rng(0)
    t = np.arange(48)
    y = 5.0 + 0.02 * t + 0.3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.01, 48)
    m = MF.ETS(m=12)
    out = m.forecast(y, h=1)
    assert np.isfinite(out)


# --- Theta ---------------------------------------------------------------------------------------

def test_theta_short_series_returns_last_value_without_fitting(monkeypatch):
    """Theta's fit floor is n<4, not max(2*m, 6): the wider gate was never a statsmodels
    requirement and silently made Theta identical to naive on 65/359 M4 Weekly series. Below the
    floor: no fit, last value; at n=5 the model must actually be fitted."""
    called = {"n": 0}
    monkeypatch.setattr(MF, "ThetaModel",
                         lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    m = MF.Theta(m=12)
    assert m.forecast(np.arange(3, dtype=float), h=1) == 2.0   # len=3 < 4: persistence, no fit
    assert called["n"] == 0
    try:
        m.forecast(np.arange(5, dtype=float), h=1)   # len=5 >= 4: the (fake) model IS constructed
    except Exception:
        pass   # the fake returns None, so .fit() raises -> falls back; only the call count matters here
    assert called["n"] == 1


def test_theta_exception_falls_back_to_last_value_and_warns_once(monkeypatch, capsys):
    monkeypatch.setattr(MF, "_FALLBACK_TALLY", {})

    def boom(*a, **k):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(MF, "ThetaModel", boom)
    y = np.arange(30, dtype=float)
    m = MF.Theta(m=12)
    out = m.forecast(y, h=1)
    assert out == y[-1]
    err = capsys.readouterr().err
    assert "[warn] Theta.forecast() raised ValueError" in err


def test_theta_real_fit_runs_end_to_end_and_returns_finite():
    rng = np.random.default_rng(1)
    t = np.arange(48)
    y = 5.0 + 0.02 * t + 0.3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.01, 48)
    m = MF.Theta(m=12)
    out = m.forecast(y, h=1)
    assert np.isfinite(out)
