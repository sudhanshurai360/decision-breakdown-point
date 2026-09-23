"""`models_freq.py`'s SIGALRM `_fit_timeout()` guard exists so that no single window's fit can hang
a run (a non-converging optimizer can otherwise stall indefinitely). Wrapping only SARIMA's two fit
calls leaves DHR's `ARIMA(...).fit()` unguarded -- the model structurally closest to SARIMAX (same
MLE machinery, plus exogenous Fourier regressors that make optimization harder).

The failure mode is invisible from outside: monkeypatching `_fit_timeout` to unconditionally raise
`_FitTimeout` changes SARIMA's forecast to the last-value fallback, but an unguarded DHR's forecast
comes back byte-identical to the unpatched run. These tests pin that DHR routes through
`_fit_timeout()` exactly like SARIMA does, so a genuinely hanging DHR fit is abandoned rather than
allowed to hang the run.
"""
import importlib

import numpy as np
import pytest

import models_freq as MF


@pytest.fixture
def trending_series():
    rng = np.random.default_rng(0)
    n = 60
    return 100 + 0.5 * np.arange(n) + rng.normal(0, 1, n)


def test_dhr_forecast_routes_through_fit_timeout(monkeypatch, trending_series):
    """THE CORE REGRESSION: DHR's fit must be wrapped by _fit_timeout(), not called bare. Monkeypatch
    _fit_timeout to unconditionally raise -- if DHR's fit is genuinely routed through it, forecast()
    must catch the resulting _FitTimeout (a subclass of Exception) and fall back to last-value,
    exactly like every other guarded fit in this file does on any fit exception."""
    def _always_timeout(fn, secs=None):
        raise MF._FitTimeout()

    monkeypatch.setattr(MF, "_fit_timeout", _always_timeout)
    m = MF.DHR(m=12)
    out = m.forecast(trending_series, h=1)
    assert out == trending_series[-1], (
        "DHR must fall back to the last observed value when _fit_timeout raises -- if this fails, "
        "DHR's fit is not actually routed through _fit_timeout()"
    )


def test_dhr_forecast_is_unaffected_when_fit_timeout_is_a_pure_passthrough(monkeypatch, trending_series):
    """Companion sanity check: with _fit_timeout patched to a transparent passthrough (calls fn()
    directly, no guard), DHR's forecast must be byte-identical to the real, unpatched _fit_timeout
    -- confirms the wrapping itself doesn't change DHR's real output, only adds the timeout guard."""
    real_output = MF.DHR(m=12).forecast(trending_series, h=1)

    def _passthrough(fn, secs=None):
        return fn()

    monkeypatch.setattr(MF, "_fit_timeout", _passthrough)
    patched_output = MF.DHR(m=12).forecast(trending_series, h=1)
    assert patched_output == real_output


def test_sarima_forecast_still_routes_through_fit_timeout(monkeypatch, trending_series):
    """Companion regression for the sibling that was ALREADY correctly guarded -- confirms this
    fix didn't accidentally change SARIMA's own (already-correct) behavior."""
    def _always_timeout(fn, secs=None):
        raise MF._FitTimeout()

    monkeypatch.setattr(MF, "_fit_timeout", _always_timeout)
    m = MF.SARIMA(m=12)
    out = m.forecast(trending_series, h=1)
    assert out == trending_series[-1]


def test_fit_timeout_env_var_accepts_fractional_values(monkeypatch):
    """MS_FIT_TIMEOUT must be parsed with float(), not int(): int() crashes at import time on a
    fractional string.

    Note the shape of this test. Asserting `float("25") == 25.0` would be a plain Python fact, true
    regardless of what models_freq.py uses to parse the env var, and would pass even with int()
    restored. So this reloads the real module with MS_FIT_TIMEOUT set to a fractional value and
    asserts on the module's own FIT_TIMEOUT attribute -- `int(os.environ.get(...))` makes it fail
    with a real ValueError at reload time."""
    monkeypatch.setenv("MS_FIT_TIMEOUT", "0.01")
    try:
        importlib.reload(MF)
        assert MF.FIT_TIMEOUT == pytest.approx(0.01)
    finally:
        monkeypatch.delenv("MS_FIT_TIMEOUT", raising=False)
        importlib.reload(MF)  # restore the module-level default for every other test in this file
    assert MF.FIT_TIMEOUT == 25.0, "module should reload back to its real default once the env var is gone"


def test_fit_timeout_env_var_malformed_value_raises_a_clear_error(monkeypatch):
    """Regression: a malformed
    MS_FIT_TIMEOUT value (empty string, non-numeric -- e.g. a stale export from a broken shell
    interpolation) used to raise a bare, undiagnostic ValueError at module-import time, with no
    indication of which variable or value caused it. MS_FIT_TIMEOUT is the ONE env var actually
    live during a normal reproduce.py run, so this is worth a clear error rather than a raw
    traceback. Must now raise RuntimeError naming both the variable and the bad value."""
    for bad_value in ("", "abc", "not-a-number"):
        monkeypatch.setenv("MS_FIT_TIMEOUT", bad_value)
        try:
            with pytest.raises(RuntimeError, match="MS_FIT_TIMEOUT"):
                importlib.reload(MF)
        finally:
            monkeypatch.delenv("MS_FIT_TIMEOUT", raising=False)
            importlib.reload(MF)  # restore the module-level default for every other test in this file
    assert MF.FIT_TIMEOUT == 25.0
