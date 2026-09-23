"""decision_protocol.py — the ONE definition of what a decision-maker knows at month t.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _monthly_grid(s: pd.Series) -> pd.DatetimeIndex:
    if not (s.index.day == 1).all():
        raise ValueError(
            "_monthly_grid(): index is not day=1-normalized -- every downstream reindex()/shift()/"
            "rolling() in this module would silently misalign against the real data"
        )
    return pd.date_range(s.index.min(), s.index.max(), freq="MS")


def date_shift(s: pd.Series, periods: int) -> pd.Series:
    """Calendar-correct `s.shift(periods)`: reindex onto a day=1 monthly grid before shifting,
    so the shift is by calendar months, not by however many rows happen to be present."""
    return s.reindex(_monthly_grid(s)).shift(periods).reindex(s.index)


def date_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Calendar-correct `s.rolling(window).mean()` -- a window of `window` calendar months, not
    `window` present rows."""
    return s.reindex(_monthly_grid(s)).rolling(window).mean().reindex(s.index)


def date_rolling_min(s: pd.Series, window: int) -> pd.Series:
    """Calendar-correct `s.rolling(window).min()` -- see date_rolling_mean()."""
    return s.reindex(_monthly_grid(s)).rolling(window).min().reindex(s.index)


def reference_average(published: pd.Series) -> pd.Series:
    """A reference leg is entirely historical, so it is entirely published -- under EITHER protocol."""
    return date_rolling_mean(published, 3)


def three_month_average(fc: pd.Series, realized: pd.Series, protocol: str = "published") -> pd.Series:
    """The CURRENT-decision-point 3-month average: two known (published) months plus one
    model-produced month under the 'published' protocol; under 'all_forecast', three
    model-produced months."""
    if protocol == "published":
        return (fc + date_shift(realized, 1) + date_shift(realized, 2)) / 3.0
    if protocol == "all_forecast":
        return date_rolling_mean(fc, 3)
    raise ValueError(f"three_month_average: unknown protocol {protocol!r} (expected 'published' or 'all_forecast')")


def _round_half_away_from_zero(x, decimals: int):
    """Round-half-away-from-zero at `decimals` places -- the statutory convention (20 CFR
    615.12(e)); numpy's default is round-half-to-even, which would misround x.x5 trigger values."""
    factor = 10.0 ** decimals
    return np.sign(x) * np.floor(np.abs(x) * factor + 0.5) / factor


def statutory_round_tur(avg3_rate_pct):
    """20 CFR 615.12(e)(3): a 3-month-average total-unemployment-rate series, already expressed
    as a percentage, rounded to the tenths place (one decimal). Approximation: the input is the mean
    of three published tenths-rounded rates, not the regulation's ratio of averaged levels -- agrees
    with DOL's published 3-month value within 0.1pp in 98% of 2009-13 state-weeks."""
    return _round_half_away_from_zero(avg3_rate_pct, 1)


def statutory_lookback_ok(current_tv, prior_tv):
    """20 CFR 615.12(e)(1)(ii): ratio of two (e)(3)-rounded Trigger Values, itself rounded to the
    hundredths place, multiplied by 100, reported as an integer, compared against the 110
    threshold."""
    ratio = current_tv / prior_tv
    ratio_pct_int = _round_half_away_from_zero(_round_half_away_from_zero(ratio, 2) * 100, 0)
    return (ratio_pct_int >= 110).astype("boolean").mask(pd.isna(ratio_pct_int))


def eb_on_statutory_rounding(fc: pd.Series, realized: pd.Series | None = None) -> pd.Series:
    """Standalone reference implementation of the (e)(3)/(e)(1)(ii) rounding cascade above."""
    if realized is None:
        realized = fc
    a = statutory_round_tur(three_month_average(fc, realized, "published"))
    ar = statutory_round_tur(date_rolling_mean(realized, 3))
    ref1, ref2 = date_shift(ar, 12), date_shift(ar, 24)
    flat_ok = (a >= 6.5).astype("boolean").mask(a.isna())
    lookback_ok = statutory_lookback_ok(a, ref1) | statutory_lookback_ok(a, ref2)
    return flat_ok & lookback_ok

