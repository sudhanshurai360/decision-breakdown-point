"""Regression tests for models_freq.py's DHR (dynamic harmonic regression) forecaster.

THE HAZARD: DHR.forecast() builds the Fourier exog for ALL h target steps via
`self._fourier(h, start=n)`. Slicing that to only the LAST row and `np.repeat()`-ing that single
row h times before handing it to ARIMA's `.forecast(h, exog=...)` makes every forecast step
silently use the seasonal phase of the FINAL target step instead of its own. Every real call site
in the pipeline hardcodes h=1 (walkforward.py:121 is the only live caller), where the
slice-and-repeat and the full matrix are identical (both reduce to the single row for step n), so
such a defect changes no published number -- but the public `forecast(y, h)` contract advertises
general h like every sibling model in this file, so it is a genuine latent correctness defect, not
a cosmetic one.
"""
import numpy as np

import models_freq as MF


def _fake_arima(monkeypatch, captured):
    """Stub statsmodels' ARIMA entirely: .fit() returns a fake result whose .forecast(steps, exog=)
    RECORDS the exog it was given and echoes back its first column as the "forecast". This isolates
    exactly the buggy plumbing (which Fourier rows reach .forecast()) from the real ARIMA numerics."""

    class _FakeResult:
        def forecast(self, steps, exog=None):
            captured["exog"] = np.asarray(exog, float)
            captured["steps"] = steps
            return captured["exog"][:, 0]

    class _FakeARIMA:
        def __init__(self, endog, exog=None, order=None):
            pass

        def fit(self):
            return _FakeResult()

    # The mock must target models_freq's OWN already-bound `ARIMA` name, not the statsmodels one.
    # models_freq.py imports ARIMA at module level (so a missing statsmodels fails loudly at import
    # time instead of being silently caught downstream), and `from X import Y` binds a NEW name in
    # the importing module's namespace at import time -- so patching `statsmodels...ARIMA` has no
    # effect here. (It would work if the import were inside DHR.forecast(), where it re-resolves on
    # every call.)
    monkeypatch.setattr(MF, "ARIMA", _FakeARIMA)


def test_dhr_forecast_h_gt1_uses_per_step_seasonal_phase_not_last_step_repeated(monkeypatch):
    """THE CONFIRMED BUG, encoded as a permanent regression test. Compares the exog rows DHR.forecast
    actually hands to ARIMA against a HAND-DERIVED (no call to _fourier here) sin/cos phase for each
    of h=3 forecast steps -- the old code would have passed h copies of only the LAST row."""
    m, K = 12, 2
    n = 40    # length of the fitted history; >= 2*m+8=32 so the short-series fallback doesn't trigger
    h = 3     # >1, with every step checked individually below (well beyond "at least 2")

    captured = {}
    _fake_arima(monkeypatch, captured)

    y = np.linspace(0.0, 1.0, n)   # arbitrary; only its length matters here, ARIMA itself is stubbed
    dhr = MF.DHR(m=m, K=K)
    dhr.forecast(y, h=h)

    assert captured["steps"] == h
    exog = captured["exog"]
    assert exog.shape == (h, 2 * K), (
        f"expected one Fourier row per forecast step ({h} rows), got shape {exog.shape} -- the old "
        f"bug passed a single row repeated h times")

    # Hand-derived (not via _fourier) expected phase for target steps t = n, n+1, ..., n+h-1.
    for step in range(h):
        t = n + step
        expected = []
        for k in range(1, K + 1):
            expected.append(np.sin(2 * np.pi * k * t / m))
            expected.append(np.cos(2 * np.pi * k * t / m))
        np.testing.assert_allclose(exog[step], expected, atol=1e-12,
                                    err_msg=f"forecast step {step} (t={t}) got the wrong Fourier phase")

    # The old buggy code would have passed h COPIES of the LAST row -- confirm the first and last
    # rows actually differ, i.e. this setup would have caught the original bug.
    assert not np.allclose(exog[0], exog[-1]), (
        "first and last forecast-step exog rows are identical -- this test's own setup does not "
        "distinguish steps, so it cannot actually catch the original last-row-repeated bug")


def test_dhr_forecast_h1_unaffected(monkeypatch):
    """Negative control: h=1 (the only value every real call site in the pipeline uses today) must
    still pass a single, correctly-phased row -- the fix must not change behavior at h=1."""
    m, K = 12, 2
    n = 40

    captured = {}
    _fake_arima(monkeypatch, captured)

    y = np.linspace(0.0, 1.0, n)
    dhr = MF.DHR(m=m, K=K)
    dhr.forecast(y, h=1)

    exog = captured["exog"]
    assert exog.shape == (1, 2 * K)
    expected = []
    for k in range(1, K + 1):
        expected.append(np.sin(2 * np.pi * k * n / m))
        expected.append(np.cos(2 * np.pi * k * n / m))
    np.testing.assert_allclose(exog[0], expected, atol=1e-12)


def test_dhr_forecast_short_series_falls_back_to_last_value():
    """Negative control: below the 2*m+8 length gate, forecast() must still fall back to _last(y)
    regardless of the h>1 fix -- this path never reaches ARIMA at all."""
    y = np.array([1.0, 2.0, 3.0])
    assert MF.DHR(m=12).forecast(y, h=3) == 3.0


def test_dhr_forecast_h_gt1_runs_end_to_end_with_real_arima():
    """Sanity check with the REAL statsmodels ARIMA (not mocked): h>1 must not raise and must return
    a finite float, on a genuinely seasonal synthetic series."""
    m, K = 12, 2
    n = 60
    rng = np.random.default_rng(0)
    t = np.arange(n)
    y = 5.0 + 0.3 * np.sin(2 * np.pi * t / m) + 0.05 * rng.standard_normal(n)
    f = MF.DHR(m=m, K=K).forecast(y, h=4)
    assert np.isfinite(f)
