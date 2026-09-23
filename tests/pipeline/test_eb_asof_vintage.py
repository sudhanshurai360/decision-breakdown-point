"""Synthetic tests for state_eb_vintage.py's AS-OF vintage construct, which produces
vintage.realized_trigger_changed (live value 271).

The construct: for a determination month t, every leg of the EB trigger is read from the ONE
vintage current when month t was first published -- asof_vintage_pivot() builds the dates x
vintages matrix (forward-filled across vintages, with a removed observation carried as NaN, never
resurrected), and eb_on_asof() evaluates eb_on() on that vintage's series and keeps only its value
at t. These tests pin the properties with hand-built vintage rows and hand-computed answers.
"""
import numpy as np
import pandas as pd

import state_eb_vintage as V
import state_eb_decision as SEB


def _rows(state, triples):
    """triples: (date, realtime_start, value) -> archive-style DataFrame (NaN value = removed)."""
    return pd.DataFrame({"state": state,
                         "date": pd.to_datetime([t[0] for t in triples]),
                         "realtime_start": pd.to_datetime([t[1] for t in triples]),
                         "value": [t[2] for t in triples]})


def test_pivot_forward_fills_across_vintages_and_keeps_a_removal_as_nan():
    v = _rows("XX", [
        ("2020-01-01", "2020-02-10", 5.0),
        ("2020-02-01", "2020-03-10", 5.5),
        ("2020-01-01", "2020-03-10", 5.2),   # January revised in the March vintage
        ("2020-03-01", "2020-04-10", 6.0),
        ("2020-02-01", "2020-04-10", np.nan),  # February REMOVED from the April vintage on
    ])
    P = V.asof_vintage_pivot(v)
    cols = list(P.columns)
    assert cols == sorted(pd.to_datetime(["2020-02-10", "2020-03-10", "2020-04-10"]))
    jan, feb, mar = [pd.Timestamp(d) for d in ("2020-01-01", "2020-02-01", "2020-03-01")]
    # January: 5.0 at the Feb vintage, 5.2 from the March vintage on (carried into April)
    assert P.loc[jan].tolist() == [5.0, 5.2, 5.2]
    # February: not yet published at the Feb vintage; 5.5 in March; removed (NaN, not 5.5) in April
    assert np.isnan(P.loc[feb, cols[0]]) and P.loc[feb, cols[1]] == 5.5 and np.isnan(P.loc[feb, cols[2]])
    # March: only in the April vintage
    assert np.isnan(P.loc[mar, cols[0]]) and np.isnan(P.loc[mar, cols[1]]) and P.loc[mar, cols[2]] == 6.0


def test_pivot_never_resurrects_a_removed_value_in_later_vintages():
    v = _rows("XX", [
        ("2020-01-01", "2020-02-10", 5.0),
        ("2020-01-01", "2020-03-10", np.nan),   # removed
        ("2020-02-01", "2020-04-10", 5.1),      # a later vintage that says nothing about January
    ])
    P = V.asof_vintage_pivot(v)
    assert np.isnan(P.loc[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-10")])


def _trigger_panel(level_now, level_prior):
    """36 months of a state UR: prior two years flat at `level_prior`, the last 3 months at
    `level_now` -- so eb_on(u, u) at the last month is ON iff level_now >= 6.5 and >= 1.10*prior."""
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    vals = [level_prior] * 33 + [level_now] * 3
    return pd.Series(vals, index=dates, dtype=float)


def test_eb_on_asof_reads_the_vintage_current_at_t_and_only_that_month():
    """Two vintages of the same state: the early vintage has the prior years at 6.0 (no 10 % rise),
    the later one benchmark-revises them down to 5.0 (a rise). Month t=2020-12's first release is
    the LATER vintage, so the as-of status must be ON; the earlier vintage would say OFF."""
    base = _trigger_panel(6.6, 6.0)                       # flat 6.0 -> 6.6: 6.6 < 1.10*6.0=6.6? 6.6/6.0 = 1.10 -> ON at the margin
    base_off = _trigger_panel(6.6, 6.1)                   # 6.6/6.1 = 1.08 -> OFF
    early = pd.Timestamp("2020-12-15"); late = pd.Timestamp("2021-01-20")
    rows = []
    for d, x in base_off.items():                         # early vintage: prior years at 6.1
        if d < pd.Timestamp("2020-12-01"):
            rows.append((d, early, x))
    for d, x in base.items():                             # late vintage: t published, history revised to 6.0
        rows.append((d, late, x))
    v = _rows("XX", rows)
    P = V.asof_vintage_pivot(v)
    t = pd.Timestamp("2020-12-01")
    first_release = pd.Series({t: late})
    out = V.eb_on_asof(None, P, first_release, base.index, [t])
    assert out.dtype.name == "boolean"
    assert bool(out.loc[t]) is True
    assert out.drop(t).isna().all()                       # one determination per month; nothing else filled
    # the superseded per-leg construct (month t's first release spliced onto the EARLY vintage's
    # unrevised history) would say OFF -- the vintage choice is load-bearing
    spliced = P[early].reindex(base.index).copy(); spliced.loc[t] = base.loc[t]
    assert bool(SEB.eb_on(spliced, spliced).loc[t]) is False


def test_eb_on_asof_skips_months_whose_first_release_is_not_a_vintage_column():
    base = _trigger_panel(7.0, 5.0)
    v = _rows("XX", [(d, pd.Timestamp("2021-01-20"), x) for d, x in base.items()])
    P = V.asof_vintage_pivot(v)
    t = pd.Timestamp("2020-12-01")
    out = V.eb_on_asof(None, P, pd.Series({t: pd.Timestamp("2021-02-20")}), base.index, [t])
    assert out.isna().all()


def test_true_first_releases_exclude_the_archive_start_snapshot(monkeypatch, tmp_path):
    """Months that exist only because the archive's first snapshot carried them have no first
    release; they must not be treated as published on the snapshot date."""
    snap = pd.Timestamp("2005-06-15")
    rows = [("2005-01-01", snap, 5.0), ("2005-02-01", snap, 5.1),        # snapshot months
            ("2005-06-01", pd.Timestamp("2005-07-20"), 5.2)]              # a genuine release
    v = _rows("XX", rows)
    p = tmp_path / "v.parquet"; v.to_parquet(p)
    monkeypatch.setattr(V, "VINT", p)
    fr = V.first_release_dates()
    assert list(fr.index) == [pd.Timestamp("2005-06-01")]
    assert fr.loc[pd.Timestamp("2005-06-01"), "XX"] == pd.Timestamp("2005-07-20")
