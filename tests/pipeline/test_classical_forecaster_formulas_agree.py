"""m_audit.py and models_freq.py (plus, for the length-gate half, density_audit.py) independently
implement the SAME classical forecaster formulas: (a) an AR(1)-on-first-differences closed-form
OLS fit (m_audit.py::f_ar1() vs models_freq.py::AR1.forecast()); (b) ETS/Theta/SES minimum-length
gate constants tied to statsmodels' real heuristic-init requirements (m_audit.py::f_ets()/
f_theta() vs models_freq.py::ETS.forecast()/Theta.forecast() vs density_audit.py::f_ses()/
f_damped()). The (b) half already silently drifted once before a validator caught it by chance --
the same risk shape seen elsewhere in this codebase (an AR(1) formula that already drifted once,
and a 3-way `_warn_fallback` fan-out). A full merge was judged more invasive than consolidating
the return shape (array-of-h vs scalar-at-h for (a); function vs class-method across 2-3 files
for (b)), so this is a PIN: these tests assert the two/three independent implementations
currently agree, so the next gate-formula edit fails a test here instead of requiring a second
validator to catch it by chance, the way the drift actually was caught.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from m_audit import f_ar1, f_ets, f_theta
from models_freq import AR1, ETS, Theta
from density_audit import f_ses, f_damped


def _trending_series(n, seed, level=200.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 5, size=n).cumsum() / 5 + level


# --- (a) AR(1) closed-form agreement -------------------------------------------------------

def test_ar1_closed_form_agrees_at_h1():
    y = _trending_series(40, seed=0)
    assert f_ar1(y, 12, 1)[-1] == pytest.approx(AR1(m=12).forecast(y, h=1))


def test_ar1_closed_form_agrees_across_multiple_horizons():
    y = _trending_series(40, seed=1)
    for h in (1, 3, 5, 8):
        a = f_ar1(y, 12, h)[-1]
        b = AR1(m=12).forecast(y, h=h)
        assert a == pytest.approx(b), f"h={h}: f_ar1={a!r} vs AR1.forecast={b!r}"


def test_ar1_closed_form_agrees_on_a_short_series_below_the_length_4_guard():
    """Both implementations fall back to flat persistence for len(y) < 4 -- confirm the guard
    threshold itself agrees, not just the fitted-branch formula."""
    y = np.array([10.0, 12.0, 11.0])
    assert f_ar1(y, 12, 3)[-1] == pytest.approx(AR1(m=12).forecast(y, h=3))
    assert f_ar1(y, 12, 3)[-1] == pytest.approx(y[-1])  # both should be exactly flat here


# --- (b) ETS/Theta/SES minimum-length gate agreement ----------------------------------------

def _is_flat_fallback(forecast, y):
    return abs(forecast - y[-1]) < 1e-9


@pytest.mark.parametrize("m", [12, 52])
def test_ets_seasonal_gate_boundary_agrees(m):
    """max(2*m, 10+2*(m//2)) is the seasonal-branch gate in BOTH m_audit.py::f_ets() and
    models_freq.py::ETS.forecast() -- this is exactly the m=52 (Weekly) case that silently
    drifted once before a validator caught it by chance. Confirm both
    implementations make the SAME go/no-go decision one step below and at the threshold."""
    thresh = max(2 * m, 10 + 2 * (m // 2))
    for n, label in ((thresh - 1, "below"), (thresh, "at")):
        y = _trending_series(n, seed=100 + n)
        a = f_ets(y, m, 1)[0]
        b = ETS(m=m).forecast(y, h=1)
        a_flat, b_flat = _is_flat_fallback(a, y), _is_flat_fallback(b, y)
        assert a_flat == b_flat, (
            f"m={m}, n={n} ({label} threshold {thresh}): f_ets flat-fallback={a_flat} but "
            f"ETS.forecast flat-fallback={b_flat} -- the seasonal gate has drifted between files"
        )


@pytest.mark.parametrize("m", [12, 52])
def test_theta_gate_boundary_agrees(m):
    """n<4 is the gate in both m_audit.py::f_theta() and models_freq.py::Theta.forecast(). A wider
    gate of max(2*m, 6) was never a statsmodels requirement and silently turned Theta into naive
    persistence on 65/359 M4 Weekly series, so this also asserts both files produce a NON-flat
    (genuine Theta) forecast at the m=52 boundary that wider gate would have caught."""
    thresh = 4
    for n in (thresh - 1, thresh):
        y = _trending_series(n, seed=200 + n)
        a = f_theta(y, m, 1)[0]
        b = Theta(m=m).forecast(y, h=1)
        a_flat, b_flat = _is_flat_fallback(a, y), _is_flat_fallback(b, y)
        assert a_flat == b_flat, (
            f"n={n} (threshold {thresh}): f_theta flat-fallback={a_flat} but Theta.forecast "
            f"flat-fallback={b_flat} -- the theta gate has drifted between files"
        )
    # THE CORE REGRESSION: one observation below the OLD gate (2*m - 1) must now be a genuine
    # Theta fit in both files, not persistence -- on a trending series Theta carries drift.
    n_old = 2 * m - 1
    y = _trending_series(n_old, seed=300 + n_old)
    a, b = f_theta(y, m, 1)[0], Theta(m=m).forecast(y, h=1)
    assert not _is_flat_fallback(a, y), f"m={m}, n={n_old}: f_theta still returns persistence"
    assert not _is_flat_fallback(b, y), f"m={m}, n={n_old}: Theta.forecast still returns persistence"
    assert abs(a - b) < 1e-9, "the two Theta implementations disagree on the same input"


def test_ses_damped_and_ets_nonseasonal_gate_agree_at_n10():
    """n>=10 is the shared non-seasonal-branch floor across density_audit.py::f_ses()/f_damped()
    and models_freq.py::ETS.forecast()'s own non-seasonal (damped-Holt) branch."""
    for n in (9, 10):
        y = _trending_series(n, seed=300 + n)
        ses = f_ses(y, 12, 1)[0]
        damped = f_damped(y, 12, 1)[0]
        ets_nonssn = ETS(m=1).forecast(y, h=1)  # m=1 forces ETS into its non-seasonal branch
        ses_flat = _is_flat_fallback(ses, y)
        damped_flat = _is_flat_fallback(damped, y)
        ets_flat = _is_flat_fallback(ets_nonssn, y)
        assert ses_flat == damped_flat == ets_flat, (
            f"n={n}: f_ses flat={ses_flat}, f_damped flat={damped_flat}, "
            f"ETS(non-seasonal) flat={ets_flat} -- the shared n>=10 floor has drifted"
        )
