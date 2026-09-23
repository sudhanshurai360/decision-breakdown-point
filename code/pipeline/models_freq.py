"""
models_freq.py — the CLASSICAL TIER, frequency-parameterized for single national series.
"""
import os, signal, sys, warnings
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

_FALLBACK_TALLY: dict[tuple[str, str], int] = {}


def _warn_fallback(model_name, exc):
    key = (model_name, type(exc).__name__)
    _FALLBACK_TALLY[key] = _FALLBACK_TALLY.get(key, 0) + 1
    if _FALLBACK_TALLY[key] == 1:
        print(f"  [warn] {model_name}.forecast() raised {type(exc).__name__}: {exc} -- falling back "
              f"to last-value. Further {model_name}/{type(exc).__name__} occurrences this run will "
              f"be silently tallied, not re-printed.", file=sys.stderr)


_ms_fit_timeout_raw = os.environ.get("MS_FIT_TIMEOUT", "25")
try:
    FIT_TIMEOUT = float(_ms_fit_timeout_raw)
except ValueError:
    raise RuntimeError(
        f"MS_FIT_TIMEOUT={_ms_fit_timeout_raw!r} is not a valid number (expected e.g. \"25\" or "
        f"\"0.01\"; set MS_FIT_TIMEOUT=0 to disable the guard entirely) -- fix or unset this "
        f"environment variable."
    ) from None


class _FitTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _FitTimeout()


def _fit_timeout(fn, secs=FIT_TIMEOUT):
    """Run fn() under a SIGALRM wall-clock guard; raises _FitTimeout on overrun. No-op if secs<=0 or no SIGALRM."""
    if secs and secs > 0 and hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, secs)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    return fn()


def _last(y):
    return float(y[-1])


class SeasonalNaive:
    """Forecast = value m periods before the h-step target (the standard seasonal benchmark)."""
    name = "seasonal_naive"

    def __init__(self, m=12):
        self.m = m

    def forecast(self, y, h=1):
        idx = len(y) - self.m + ((h - 1) % self.m)  # the period m steps before target t+h, cyclic
        return float(y[idx]) if 0 <= idx < len(y) else _last(y)


class AR1:
    """AR(1) on first differences (drift + persistence), period-agnostic. OLS on (Δy_{t-1} -> Δy_t)."""
    name = "ar1"

    def __init__(self, m=12):
        self.m = m

    def forecast(self, y, h=1):
        if len(y) < 4:
            return _last(y)
        dy = np.diff(y)
        x, z = dy[:-1], dy[1:]
        xm, zm = x.mean(), z.mean()
        denom = float(((x - xm) ** 2).sum())
        phi = float(((x - xm) * (z - zm)).sum() / denom) if denom > 1e-12 else 0.0
        phi = max(min(phi, 0.999), -0.999)
        c = zm - phi * xm
        d, last = dy[-1], y[-1]
        val = last
        for _ in range(h):                          # iterate the AR(1)-in-differences h steps
            d = c + phi * d
            val = val + d
        return float(val)


class ETS:
    """Exponential smoothing (Holt-Winters): additive trend+seasonal when >=2 full seasons, else damped Holt."""
    name = "ets"

    def __init__(self, m=12):
        self.m = m

    def forecast(self, y, h=1):
        n = len(y)
        try:
            if n >= max(2 * self.m, 10 + 2 * (self.m // 2)) and self.m > 1:
                res = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=self.m,
                                           initialization_method="heuristic").fit()
            elif n >= 10:
                res = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                           initialization_method="heuristic").fit()
            else:
                return _last(y)
            f = float(np.asarray(res.forecast(h)).ravel()[-1])
            return f if np.isfinite(f) else _last(y)
        except Exception as e:
            _warn_fallback("ETS", e)
            return _last(y)


class Theta:
    """The Theta method (statsmodels ThetaModel), seasonal period m."""
    name = "theta"

    def __init__(self, m=12):
        self.m = m

    def forecast(self, y, h=1):
        if len(y) < 4:
            return _last(y)
        try:
            res = ThetaModel(np.asarray(y, float), period=self.m).fit()
            f = float(np.asarray(res.forecast(h)).ravel()[-1])
            return f if np.isfinite(f) else _last(y)
        except Exception as e:
            _warn_fallback("Theta", e)
            return _last(y)


class SARIMA:
    """Seasonal ARIMA via statsmodels SARIMAX, order chosen by a small AIC grid (NOT auto_arima — same policy as
    the UI tier)."""
    name = "sarima"
    P_GRID = [(p, 0, q) for p in (0, 1) for q in (0, 1)]
    S_GRID = [(P, 0, Q, None) for P in (0, 1) for Q in (0, 1)]   # period filled in at fit
    TREND_GRID = [None, "c"]
    GUARD = 3.0                                                  # |forecast| <= max(|y|)+GUARD in log space

    def __init__(self, m=12, reselect_every=12):
        self.m = m
        self.reselect_every = reselect_every
        self._order = None
        self._since = 10 ** 9

    def _select(self, y):
        best, best_aic = None, np.inf
        for order in self.P_GRID:
            for (P, D, Q, _) in self.S_GRID:
                sorder = (P, D, Q, self.m)
                for trend in self.TREND_GRID:
                    try:
                        res = _fit_timeout(lambda: SARIMAX(y, order=order, seasonal_order=sorder, trend=trend,
                                           enforce_stationarity=True, enforce_invertibility=True,
                                           initialization="approximate_diffuse").fit(disp=False, maxiter=50))
                        if np.isfinite(res.aic) and res.aic < best_aic:
                            best, best_aic = (order, sorder, trend), res.aic
                    except Exception:                # includes _FitTimeout -> just skip this order
                        continue
        return best

    def forecast(self, y, h=1):
        y = np.asarray(y, float)
        if len(y) < 2 * self.m + 8:
            return _last(y)
        if self._since >= self.reselect_every:   # (re)select order annually
            self._order = self._select(y); self._since = 0
        self._since += 1
        if self._order is None:
            _warn_fallback("SARIMA(order-selection)", RuntimeError("_select() found no fittable order"))
            return _last(y)
        try:                                                            # refit params on the current window every time
            res = _fit_timeout(lambda: SARIMAX(y, order=self._order[0], seasonal_order=self._order[1],
                               trend=self._order[2],
                               enforce_stationarity=True, enforce_invertibility=True,
                               initialization="approximate_diffuse").fit(disp=False, maxiter=50))
            f = float(np.asarray(res.forecast(h)).ravel()[-1])
            cap = float(np.max(np.abs(y))) + self.GUARD
            return f if (np.isfinite(f) and abs(f) <= cap) else _last(y)
        except Exception as e:
            _warn_fallback("SARIMA", e)
            return _last(y)


class DHR:
    """Dynamic harmonic regression: Fourier seasonal terms (K harmonics at period m) + ARIMA(p,d,q) errors,
    d chosen by a KPSS-style difference test."""
    name = "dhr"

    def __init__(self, m=12, K=None):
        self.m = m
        self.K = K or max(1, min(4, m // 3))        # # harmonics; for m=12 -> 4, for m=52 -> 4 (capped)

    def _fourier(self, n, start=0):
        t = np.arange(start, start + n)
        cols = []
        for k in range(1, self.K + 1):
            cols.append(np.sin(2 * np.pi * k * t / self.m))
            cols.append(np.cos(2 * np.pi * k * t / self.m))
        return np.column_stack(cols)

    def _ndiff(self, y):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = kpss(y, regression="c", nlags="auto")[1]
            return 1 if p < 0.05 else 0
        except Exception as e:
            _warn_fallback("DHR(kpss)", e)
            return 0

    def forecast(self, y, h=1):
        y = np.asarray(y, float)
        n = len(y)
        if n < 2 * self.m + 8:
            return _last(y)
        d = self._ndiff(y)
        X = self._fourier(n)
        Xf = self._fourier(h, start=n)              # exog: one correctly-phased row per target step
        try:
            res = _fit_timeout(lambda: ARIMA(y, exog=X, order=(2, d, 0)).fit())
            f = float(np.asarray(res.forecast(h, exog=Xf)).ravel()[-1])
            return f if np.isfinite(f) else _last(y)
        except Exception as e:
            _warn_fallback("DHR", e)
            return _last(y)


REGISTRY = {m.name: m for m in (SeasonalNaive, AR1, ETS, Theta, SARIMA, DHR)}


def build(names, m=12):
    """Instantiate the requested classical models at seasonal period m."""
    return [REGISTRY[n](m=m) for n in names]
