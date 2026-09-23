"""Mutation-closing tests for two mutants, named F4 and F5 in this file only.

F4: state_eb_vintage.eb_on_asof reading the LATEST vintage instead of the vintage current at month
t's first release. It survives a fixture whose first-release vintage is also its last column, so
the fixtures here deliberately give it a third, later vintage. On real data F4 drops
vintage.episodes_initial by roughly two-thirds and collapses realized_trigger_changed to near-zero
(see canonical.json for current figures).

F5: an off-by-one vintage in sahm_evans_realtime.asof_realized_signals (col_at(v - 1 day)), which
silently zeroes all 12 real-time onsets.

Every test here was verified to FAIL on the named mutant and PASS on the real modules.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
for _sub in ("pipeline", "instrument"):
    _p = str(ROOT / "code" / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import state_eb_decision as SEB  # noqa: E402
import state_eb_vintage as V  # noqa: E402
import sahm_evans_realtime as SER  # noqa: E402


def _rows(state, triples):
    """triples: (date, realtime_start, value) -> archive-style DataFrame (NaN value = removed)."""
    return pd.DataFrame({"state": state,
                         "date": pd.to_datetime([t[0] for t in triples]),
                         "realtime_start": pd.to_datetime([t[1] for t in triples]),
                         "value": [t[2] for t in triples]})


# ---------------------------------------------------------------------------------------------
# F4: eb_on_asof reads the vintage current at t's FIRST RELEASE -- not the latest, not the prior
# ---------------------------------------------------------------------------------------------

def _three_vintage_archive():
    """One state, three vintages. Months 2018-01..2020-12 (t = 2020-12 is the determination
    month; its current leg is 2020-10..12 at 6.6, the reference legs are 2019-10..12 and
    2018-10..12).

      v1 (2020-12-15): history through 2020-11 with the prior years at 6.1   (t not yet published)
      v2 (2021-01-20): t = 6.6 first published; prior years benchmark-revised DOWN to 6.0
                        -> a = 6.6, R = 6.0, 6.6/6.0 = 1.10 -> 110 -> ON
      v3 (2021-02-20): prior years re-revised UP to 6.1 (and 2021-01 = 6.6 added)
                        -> a = 6.6, R = 6.1, 6.6/6.1 = 1.08 -> 108 -> OFF

    The statutory quantity is the v2 determination (ON); the latest vintage says OFF; v1 cannot
    determine t at all (t absent)."""
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    t = pd.Timestamp("2020-12-01")
    v1, v2, v3 = (pd.Timestamp(d) for d in ("2020-12-15", "2021-01-20", "2021-02-20"))

    def hist(prior, now):
        return {d: (now if d >= pd.Timestamp("2020-10-01") else prior) for d in dates}

    rows = []
    for d, x in hist(6.1, 6.6).items():
        if d < t:
            rows.append((d, v1, x))
    for d, x in hist(6.0, 6.6).items():
        rows.append((d, v2, x))
    for d, x in hist(6.1, 6.6).items():
        rows.append((d, v3, x))
    rows.append((pd.Timestamp("2021-01-01"), v3, 6.6))
    full_index = dates.append(pd.DatetimeIndex([pd.Timestamp("2021-01-01")]))
    return _rows("XX", rows), full_index, t, (v1, v2, v3)


def test_eb_on_asof_uses_the_first_release_vintage_not_the_latest_one():
    """kills: F4 `u_t = P[r]` -> `P[P.columns[-1]]` (latest vintage) in eb_on_asof. With a third
    vintage that re-revises the prior years back up, the latest column says OFF at t while the
    vintage current at t's first release says ON. Also pins that the vintage BEFORE the first
    release cannot be what is read (t is absent there, so the status would be NA)."""
    v, full_index, t, (v1, v2, v3) = _three_vintage_archive()
    P = V.asof_vintage_pivot(v)
    assert list(P.columns) == [v1, v2, v3]
    # the three vintages really do disagree, per the hand derivation in the fixture docstring
    assert bool(SEB.eb_on(P[v2].reindex(full_index), P[v2].reindex(full_index)).loc[t]) is True
    assert bool(SEB.eb_on(P[v3].reindex(full_index), P[v3].reindex(full_index)).loc[t]) is False
    assert pd.isna(SEB.eb_on(P[v1].reindex(full_index), P[v1].reindex(full_index)).loc[t])
    out = V.eb_on_asof(None, P, pd.Series({t: v2}), full_index, [t])
    assert out.dtype.name == "boolean"
    assert bool(out.loc[t]) is True
    assert out.drop(t).isna().all()


def test_eb_on_asof_each_month_reads_its_own_first_release_vintage():
    """kills: F4 (second determination month). The month after, t2 = 2021-01, is first released
    in v3, where the prior-year legs are 6.1 -> OFF; t = 2020-12 stays ON from v2. A latest-
    vintage reading would give OFF/OFF, a single-vintage-for-all reading ON/ON."""
    v, full_index, t, (v1, v2, v3) = _three_vintage_archive()
    P = V.asof_vintage_pivot(v)
    t2 = pd.Timestamp("2021-01-01")
    out = V.eb_on_asof(None, P, pd.Series({t: v2, t2: v3}), full_index, [t, t2])
    assert bool(out.loc[t]) is True
    assert bool(out.loc[t2]) is False


def test_eb_on_asof_with_a_model_forecast_still_takes_the_legs_from_the_first_release_vintage():
    """kills: F4 on the model arm (fc is not None): the forecast supplies month t only; the
    published legs t-1, t-2 and the reference years come from the vintage at t's first release.
    fc = 6.6 at t reproduces the realized case (ON via v2); fc = 6.3 gives a = round1((6.3 + 6.6
    + 6.6)/3) = 6.5, 6.5/6.0 = 1.0833 -> 108 -> OFF, so the forecast IS used for month t."""
    v, full_index, t, (v1, v2, v3) = _three_vintage_archive()
    P = V.asof_vintage_pivot(v)
    fc_on = pd.Series(6.6, index=full_index)
    fc_off = pd.Series(6.6, index=full_index); fc_off.loc[t] = 6.3
    assert bool(V.eb_on_asof(fc_on, P, pd.Series({t: v2}), full_index, [t]).loc[t]) is True
    assert bool(V.eb_on_asof(fc_off, P, pd.Series({t: v2}), full_index, [t]).loc[t]) is False


# ---------------------------------------------------------------------------------------------
# F5: asof_realized_signals recomputes the Sahm/Evans signal on the vintage current at t's first
# release and reads it AT t (so month t must be present in that vintage)
# ---------------------------------------------------------------------------------------------

def _unrate_archive():
    """Synthetic UNRATE archive: months 2019-01..2021-01, each month m first released on the 5th
    of the following month, every vintage carrying all months up to m. Values: 3.5 flat through
    2020-10, then 2020-11 = 3.7, 2020-12 = 4.3 (first release), 2021-01 = 4.5. The 2021-02-05
    vintage (which first releases 2021-01) REVISES 2020-11 -> 3.8 and 2020-12 -> 4.6.

    Hand-computed as-of signals (3-month mean minus the minimum of that mean over t-12..t-1):
      t = 2020-12, vintage 2021-01-05: avg3 = (3.5+3.7+4.3)/3 = 3.8333; the twelve prior avg3 are
                   3.5 except 2020-11 = 3.5667 -> min 3.5 -> SAHM = 0.3333; EVANS = 3.8333
      t = 2021-01, vintage 2021-02-05: avg3 = (3.8+4.6+4.5)/3 = 4.3; prior avg3: 3.5 ... 2020-11 =
                   (3.5+3.5+3.8)/3 = 3.6, 2020-12 = (3.5+3.8+4.6)/3 = 3.9667 -> min 3.5 -> SAHM 0.8
    A latest-vintage reading at 2020-12 would give (3.5+3.8+4.6)/3 - 3.5 = 0.4667; the vintage the
    day BEFORE 2020-12's first release has no 2020-12 value at all (NaN)."""
    months = pd.date_range("2019-01-01", "2021-01-01", freq="MS")
    base = {m: 3.5 for m in months}
    base[pd.Timestamp("2020-11-01")] = 3.7
    base[pd.Timestamp("2020-12-01")] = 4.3
    base[pd.Timestamp("2021-01-01")] = 4.5
    rows = []
    for m in months:
        rel = m + pd.DateOffset(months=1, days=4)          # 5th of the following month
        for k in months[months <= m]:
            val = base[k]
            if m == pd.Timestamp("2021-01-01"):            # the annual revision in that vintage
                val = {pd.Timestamp("2020-11-01"): 3.8, pd.Timestamp("2020-12-01"): 4.6}.get(k, val)
            rows.append((k, rel, val))
    d = _rows("US", rows).drop(columns="state")
    return d, months


def test_asof_realized_signals_match_hand_computed_values_on_the_first_release_vintage():
    """kills: F5 `col_at(v)` -> `col_at(v - 1 day)` in asof_realized_signals (month t absent from
    its own series -> NaN signal -> zero onsets), and a latest-vintage reading (0.4667 at 2020-12
    instead of 0.3333)."""
    d, _months = _unrate_archive()
    idx = pd.DatetimeIndex([pd.Timestamp("2020-12-01"), pd.Timestamp("2021-01-01")])
    first_release, _P, col_at = SER.asof_series(d, idx)
    assert first_release[pd.Timestamp("2020-12-01")] == pd.Timestamp("2021-01-05")
    sig = SER.asof_realized_signals(idx, first_release, col_at)
    assert sig["SAHM"].loc["2020-12-01"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert sig["EVANS"].loc["2020-12-01"] == pytest.approx(11.5 / 3.0, abs=1e-9)
    assert sig["SAHM"].loc["2021-01-01"] == pytest.approx(0.8, abs=1e-9)
    assert sig["EVANS"].loc["2021-01-01"] == pytest.approx(4.3, abs=1e-9)
    assert not sig["SAHM"].isna().any() and not sig["EVANS"].isna().any()


def test_asof_realized_signals_feed_onsets_the_way_main_reads_them():
    """kills: F5 through the consumer: the realized Sahm onset list computed from the as-of
    signals over 2020-06..2021-01 must be exactly [2021-01] (the first month at or above 0.50);
    under the off-by-one mutant every signal is NaN and the onset list is empty."""
    d, _months = _unrate_archive()
    idx = pd.date_range("2020-06-01", "2021-01-01", freq="MS")
    first_release, _P, col_at = SER.asof_series(d, idx)
    sig = SER.asof_realized_signals(idx, first_release, col_at)
    assert SER.onsets(sig["SAHM"], 0.50) == [pd.Timestamp("2021-01-01")]
    assert SER.onsets(sig["EVANS"], 6.5) == []
    assert sig["SAHM"].loc["2020-06-01":"2020-10-01"].abs().max() == pytest.approx(0.0)


def test_asof_series_col_at_returns_the_last_vintage_on_or_before_the_date():
    """Pins the col_at contract asof_realized_signals relies on: on a vintage date it returns that
    vintage (month t present); the day before it returns the previous vintage (month t absent)."""
    d, _months = _unrate_archive()
    t = pd.Timestamp("2020-12-01")
    first_release, _P, col_at = SER.asof_series(d, pd.DatetimeIndex([t]))
    v = first_release[t]
    assert col_at(v).loc[t] == 4.3
    assert np.isnan(col_at(v - pd.Timedelta(days=1)).get(t, np.nan))
    assert col_at(v + pd.Timedelta(days=10)).loc[t] == 4.3          # still the same vintage
    assert col_at(pd.Timestamp("2021-02-05")).loc[t] == 4.6         # the revising vintage
