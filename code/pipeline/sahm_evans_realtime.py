"""
sahm_evans_realtime.py — the real-time-honest national decision instance (Sahm and Evans rules on ALFRED UNRATE vintages).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
import m_audit as MA
import mcs_test as MCS
import fetch_alfred as FA
import decision_protocol as DP
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
RES = MULTI / "results"; CACHE = MULTI / "data" / "unrate_vintages.parquet"
MODELS = MA.EB_MODELS
EVAL_START = 1965


def real_time_series(force=False):
    """first-release UNRATE per reference month (value as FIRST published) -> point-in-time monthly series."""
    def _fetch():
        from fredapi import Fred
        return FA.fetch_one("UNRATE", Fred(api_key=FA._fred_key()))
    d, fetched = FA.load_cached_or_fetch("UNRATE", CACHE, _fetch, force)
    if fetched:
        CACHE.parent.mkdir(parents=True, exist_ok=True); d.to_parquet(CACHE)
    d = d.dropna(subset=["value"]).copy()
    d["date"] = pd.to_datetime(d["date"]); d["realtime_start"] = pd.to_datetime(d["realtime_start"])
    fr = d.sort_values("realtime_start").groupby("date").first()["value"]   # first release per month
    fr.index = pd.to_datetime(fr.index)
    return fr.sort_index().astype(float)


def raw_vintages():
    """The cached UNRATE vintage archive as stored (one row per (date, realtime_start) revision,
    NaN rows kept -- a NaN row records the observation's ABSENCE from that vintage on)."""
    d = pd.read_parquet(CACHE)
    d["date"] = pd.to_datetime(d["date"]); d["realtime_start"] = pd.to_datetime(d["realtime_start"])
    return d.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def asof_series(d, idx):
    """(first_release, P, col_at) for the as-of construct: each decision month's first-release
    date, the dates x vintage-dates value matrix, and a lookup returning the vintage current at
    any calendar date -- the national twin of state_eb_vintage.py's as-of pivot."""
    import state_eb_vintage as SEV
    first_release = d.dropna(subset=["value"]).groupby("date")["realtime_start"].min()
    archive_start = d["realtime_start"].min()
    if (first_release.reindex(idx) <= archive_start).any():
        raise ValueError("asof_series(): a decision month's first release is the archive-start "
                         "snapshot -- move EVAL_START past the archive's first vintage")
    if first_release.reindex(idx).isna().any():
        raise ValueError("asof_series(): a decision month has no published first release")
    P = SEV.asof_vintage_pivot(d)
    cols = P.columns

    def col_at(v):
        """the vintage current at calendar date v: last realtime_start <= v"""
        pos = cols.searchsorted(pd.Timestamp(v), side="right") - 1
        if pos < 0:
            raise ValueError(f"asof_series(): no vintage on or before {v}")
        return P[cols[pos]]

    return first_release, P, col_at


def sahm(avg3):
    """avg3: an ALREADY-COMPUTED 3-month average (boundary-aware for a model path, or a plain
    rolling(3).mean() for the realized/ground-truth path -- see main())."""
    ref_low = DP.date_shift(DP.date_rolling_min(avg3, 12), 1)   # calendar-aware; Oct-2025 gap propagates as NaN
    return avg3 - ref_low


def evans(avg3):
    return avg3                                                # compare to 6.5 threshold


def onsets(sig, thr):
    """Onset dates: months where `sig` crosses from NOT-hot to hot."""
    hot = (sig.round(9) >= thr).astype("boolean").mask(sig.isna())
    out = []
    prev = False   # Python bool, not numpy/pandas: "before the series starts" = not hot, matching
                    # the original fill_value=False. `prev is False` below needs real identity to
                    # work, which numpy.bool_/pandas boolean scalars do not satisfy even when equal.
    for d, h in zip(sig.index, hot):
        if pd.isna(h):
            prev = None   # unknown, not False -- distinct from a determined not-hot
            continue
        h = bool(h)
        if h and prev is False:
            out.append(d)
        prev = h
    return out


def asof_realized_signals(idx, first_release, col_at):
    """The realized (no-model) Sahm/Evans signals on the as-of basis: for each decision month t,
    the whole signal is recomputed on the vintage current at t's first release and read at t."""
    sig = {"SAHM": pd.Series(index=idx, dtype=float), "EVANS": pd.Series(index=idx, dtype=float)}
    fr = first_release.reindex(idx)
    for v, months in fr.groupby(fr).groups.items():
        series = col_at(v).dropna()
        avg3 = DP.three_month_average(series, series, "published")   # == calendar 3-month mean
        sig["SAHM"].loc[months] = sahm(avg3).reindex(months).to_numpy()
        sig["EVANS"].loc[months] = evans(avg3).reindex(months).to_numpy()
    return sig


def _run_protocol(protocol, now, realized, tied, real_sig=None, asof_legs=None):
    """One full sahm/evans onset-divergence pass under one three_month_average() protocol."""
    ref_low = None
    if asof_legs is None:
        avg3_model = {m: DP.three_month_average(now[m], realized, protocol) for m in tied}
    elif protocol == "published":
        r1, r2, ref_low = asof_legs
        avg3_model = {m: (now[m] + r1 + r2) / 3.0 for m in tied}
    else:                                   # all_forecast: three consecutive nowcasts, no published leg
        avg3_model = {m: DP.three_month_average(now[m], realized, protocol) for m in tied}
    if real_sig is None:
        avg3_real = DP.three_month_average(realized, realized, protocol)
        real_sig = {"SAHM": sahm(avg3_real), "EVANS": evans(avg3_real)}

    summary_rows = []; detail = {}; onset_rows = []
    for rule, sig_fn, thr in [("SAHM (>=0.50)", sahm, 0.50), ("EVANS (UNRATE 3mo-avg >= 6.5)", evans, 6.5)]:
        real_on = onsets(real_sig[rule.split()[0]], thr)
        if ref_low is not None and rule.startswith("SAHM"):
            m_on = {m: onsets(avg3_model[m] - ref_low, thr) for m in tied}   # published reference low
        else:
            m_on = {m: onsets(sig_fn(avg3_model[m]), thr) for m in tied}
        if protocol == "published":
            print(f"\n=== {rule}: realized real-time onsets: {[d.strftime('%Y-%m') for d in real_on]} ===")
        n_div = 0; genuine = []; misses = 0
        for ro in real_on:
            def nearest(o):
                return min(o, key=lambda d: abs((d - ro).days)) if o else None
            calls = {m: nearest(m_on[m]) for m in tied}
            months = {m: (c.strftime('%Y-%m') if c is not None else "NONE") for m, c in calls.items()}
            vals = [c for c in calls.values() if c is not None]
            spread = (float("inf") if len(vals) < len(calls)
                      else (max(vals) - min(vals)).days / 30.0 if len(vals) > 1 else 0)
            distinct = len(set(months.values()))
            if distinct > 1 or spread == float("inf"):
                n_div += 1
                (genuine.append(spread) if spread <= 12 else None)   # >12mo = a MISSED onset, not a timing shift
                misses += int(spread > 12)
            onset_rows.append({"rule": rule.split()[0], "protocol": protocol,
                                "onset": ro.strftime("%Y-%m"), "n_distinct_calls": distinct,
                                "divergent": distinct > 1, "spread_mo": None if spread == float("inf") else round(spread, 1),
                                "missed": spread > 12})
            if protocol == "published":
                print(f"    {ro.strftime('%Y-%m')}: {months}  -> {distinct} distinct, spread {spread:.0f} mo")
        med = float(np.median(genuine)) if genuine else 0.0
        if protocol == "published":
            print(f"  -> tied models DIVERGE on {n_div}/{len(real_on)} onsets; typical timing spread {med:.0f} mo; "
                  f"{misses} onset(s) MISSED by one model (NOT a long timing gap). Real-time-honest (no revisions);")
            print("     agreement on the MAJOR recessions -> the real-time national cost is modest (state EB stays powered).")
        key = rule.split()[0]
        summary_rows.append({"rule": key, "protocol": protocol, "n_onsets": len(real_on), "n_divergent": n_div,
                             "n_missed": misses, "median_spread_mo": round(med, 1)})
        detail[key] = {"n_divergent": n_div, "n_onsets": len(real_on), "n_missed": misses,
                        "median_spread_mo": round(med, 1)}
    return summary_rows, detail, onset_rows


def _calendar_filled(u: pd.Series) -> pd.Series:
    """`u` reindexed onto its full monthly calendar grid, with any single-month gap causally
    (forward-)filled -- built for main()'s rolling nowcast training array ONLY, mirroring
    state_eb_decision.py::_calendar_filled() exactly (same rationale, same `ffill(limit=1)` choice,
    same fail-loud guard on anything wider)."""
    grid = DP._monthly_grid(u)
    filled = u.reindex(grid).ffill(limit=1)
    if filled.isna().any():
        bad = filled[filled.isna()].index[0]
        raise ValueError(
            f"_calendar_filled(): un-bridgeable gap at {bad.date()} (more than 1 consecutive "
            f"missing month) in the real-time UNRATE series -- ffill(limit=1) only bridges a "
            f"single-month gap; a wider gap needs an explicit, audited decision, not a silent fill."
        )
    return filled


def main():
    unknown = [a for a in sys.argv[1:] if a != "--force"]
    if unknown:
        print(f"  UNKNOWN ARGUMENT(S): {unknown} -- refusing to silently ignore a possible typo. "
              f"Recognized: --force.", file=sys.stderr)
        sys.exit(2)
    u = real_time_series(force="--force" in sys.argv)
    print(f"real-time (first-release) UNRATE: {u.index.min().date()}..{u.index.max().date()}, {len(u)} months")
    _raw_vintages = pd.read_parquet(CACHE) if CACHE.exists() else None
    if _raw_vintages is not None:
        emit("data.unrate_alfred_vintages", int(_raw_vintages["realtime_start"].nunique()),
             spec="distinct ALFRED realtime_start dates in the raw UNRATE vintage cache, i.e. how many "
                  "genuinely different point-in-time snapshots of the series exist over its history",
             cites=["S3.4 body: national UNRATE vintage depth", "S7 body: real-time-honest complement"],
             cross_check="unrate_vintages.parquet: realtime_start.nunique()")
    idx = u.index[u.index.year >= EVAL_START]
    realized = u.reindex(idx)          # month t's own value as first published -- the same under both constructs

    first_release, _, col_at = asof_series(raw_vintages(), idx)
    if not np.allclose(pd.Series({t: col_at(first_release[t])[t] for t in idx}), realized, equal_nan=True):
        raise RuntimeError("as-of construct: month t's value at its own first release must equal the first release")
    now = {m: [] for m in MODELS}; r1 = []; r2 = []; ref_low = []
    for t in idx:                                             # rolling 1-step real-time nowcast (train on the ORIGIN vintage)
        origin = col_at(first_release[t] - pd.Timedelta(days=1)).dropna()
        hist_s = origin[origin.index < t]
        r1.append(hist_s.get(t - pd.offsets.MonthBegin(1), np.nan))
        r2.append(hist_s.get(t - pd.offsets.MonthBegin(2), np.nan))
        a3 = DP.three_month_average(hist_s, hist_s, "published")
        win = a3[(a3.index >= t - pd.offsets.MonthBegin(12)) & (a3.index <= t - pd.offsets.MonthBegin(1))]
        ref_low.append(float(win.min()) if len(win) == 12 and win.notna().all() else np.nan)
        hist = _calendar_filled(hist_s).to_numpy(float)   # training array only -- see _calendar_filled()
        for m, fn in MODELS.items():
            now[m].append(float(MA.safe_forecast(fn, m, hist, 12, 1, context=f"t={t}")[0]))
    now = {m: pd.Series(v, index=idx) for m, v in now.items()}
    asof_legs = (pd.Series(r1, index=idx, dtype=float), pd.Series(r2, index=idx, dtype=float),
                 pd.Series(ref_low, index=idx, dtype=float))
    real_sig = asof_realized_signals(idx, first_release, col_at)

    err = pd.DataFrame({m: np.abs(now[m] - realized) for m in MODELS})
    surv, _ = MCS.mcs(err.groupby(err.index.year).mean().to_numpy(float), alpha=0.10, seed=0)
    tied = [list(MODELS)[i] for i in surv]
    print(f"\n  MCS (accuracy-tied on REAL-TIME UNRATE 1-step, as-of vintages): {tied}")

    rows_pub, detail_pub, onsets_pub = _run_protocol("published", now, realized, tied,
                                                     real_sig=real_sig, asof_legs=asof_legs)
    rows_all, detail_all, onsets_all = _run_protocol("all_forecast", now, realized, tied,
                                                     real_sig=real_sig, asof_legs=asof_legs)

    u_filled = _calendar_filled(u)
    now_fr = {m: [] for m in MODELS}
    for t in idx:
        hist = u_filled[u_filled.index < t].to_numpy(float)
        for m, fn in MODELS.items():
            now_fr[m].append(float(MA.safe_forecast(fn, m, hist, 12, 1, context=f"first-release t={t}")[0]))
    now_fr = {m: pd.Series(v, index=idx) for m, v in now_fr.items()}
    err_fr = pd.DataFrame({m: np.abs(now_fr[m] - realized) for m in MODELS})
    surv_fr, _ = MCS.mcs(err_fr.groupby(err_fr.index.year).mean().to_numpy(float), alpha=0.10, seed=0)
    tied_fr = [list(MODELS)[i] for i in surv_fr]
    print(f"  MCS (superseded first-release-per-leg construct): {tied_fr}")
    rows_pub_fr, detail_pub_fr, onsets_pub_fr = _run_protocol("published", now_fr, realized, tied_fr)
    rows_all_fr, _, onsets_all_fr = _run_protocol("all_forecast", now_fr, realized, tied_fr)
    for r in rows_pub_fr + rows_all_fr:
        r["protocol"] += "_first_release_each_leg"
    for r in onsets_pub_fr + onsets_all_fr:
        r["protocol"] += "_first_release_each_leg"
    rows_pub += rows_pub_fr; rows_all += rows_all_fr
    onsets_pub += onsets_pub_fr; onsets_all += onsets_all_fr
    RM._atomic_to_csv(pd.DataFrame(rows_pub + rows_all), RES / "sahm_evans_summary.csv")   # persist for the gates, not just stdout
    RM._atomic_to_csv(pd.DataFrame(onsets_pub + onsets_all), RES / "sahm_evans_onsets.csv")
    print("\n  VERDICT: the decision cost of non-identification is present on REAL-TIME point-in-time data (UNRATE vintages")
    print("  back to 1960), curing the state-EB revised-data caveat. National coverage; state EB remains the powered panel.")
    print(f"  PROTOCOL CONTRAST: published (2 realized + 1 forecast) sahm={detail_pub['SAHM']} evans={detail_pub['EVANS']} "
          f"vs superseded all_forecast sahm={detail_all['SAHM']} evans={detail_all['EVANS']}")

    _asof = ("AS-OF basis: the realized signal at each decision month t is recomputed "
             "on the single ALFRED vintage current when month t was first published; the model path "
             "trains on the vintage current the day before (the origin vintage) and takes its two "
             "published months and, for the Sahm rule, its previous-12-month reference low from that "
             "same origin vintage -- only month t itself is model-produced. The companion field "
             "each_leg_first_release is the same figure under an alternative construct that splices "
             "each month's own first release into one series (a mixed-vintage series no "
             "determination ever saw), kept for disclosure.")
    emit("realtime.sahm_divergent", int(detail_pub["SAHM"]["n_divergent"]), of=int(detail_pub["SAHM"]["n_onsets"]),
         each_leg_first_release=int(detail_pub_fr["SAHM"]["n_divergent"]),
         each_leg_first_release_of=int(detail_pub_fr["SAHM"]["n_onsets"]),
         spec="Sahm-rule (3mo-avg UNRATE rises >=0.50pp above the minimum of the 3mo-avg over the previous "
              "12 months, t-12..t-1, Sahm 2019) real-time onsets on which accuracy-tied models disagree, "
              "under the 'published' protocol (2 published + 1 forecast month). " + _asof,
         cites=["S7 body: Sahm real-time divergence"],
         cross_check="sahm_evans_summary.csv: n_divergent where rule=='SAHM' and protocol=='published'")
    emit("realtime.sahm_median_spread_mo", float(detail_pub["SAHM"]["median_spread_mo"]),
         spec="Sahm-rule real-time onset divergences (published protocol): median timing spread in "
              "months among the tied models' calls, across only the divergent onsets (excludes "
              "outright misses, defined as spread>12mo)",
         cites=["S7 body: real-time-honest complement"],
         cross_check="sahm_evans_summary.csv: median_spread_mo where rule=='SAHM' and protocol=='published'")
    emit("realtime.sahm_n_missed", int(detail_pub["SAHM"]["n_missed"]),
         spec="Sahm-rule real-time onset divergences (published protocol): of the n_divergent onsets, "
              "how many were an outright MISS by one tied model (timing spread >12mo) rather than a "
              "modest within-year timing shift",
         cites=["S7 body: real-time-honest complement"],
         cross_check="sahm_evans_summary.csv: n_missed where rule=='SAHM' and protocol=='published'")
    emit("realtime.evans_divergent", int(detail_pub["EVANS"]["n_divergent"]), of=int(detail_pub["EVANS"]["n_onsets"]),
         each_leg_first_release=int(detail_pub_fr["EVANS"]["n_divergent"]),
         each_leg_first_release_of=int(detail_pub_fr["EVANS"]["n_onsets"]),
         spec="the 2012-14 Federal Reserve forward-guidance threshold -- the FOMC's public commitment "
              "to hold its policy rate near zero at least as long as unemployment stayed above 6.5%, "
              "operationalised here as the 3mo-avg UNRATE rising to/above 6.5% (EVANS is this arm's "
              "internal code identifier only; the manuscript names the FOMC threshold, not a person) "
              "-- real-time onsets on which accuracy-tied models disagree, under the 'published' "
              "protocol (2 published + 1 forecast month). " + _asof,
         cites=["S7 body: forward-guidance-threshold real-time divergence"],
         cross_check="sahm_evans_summary.csv: n_divergent where rule=='EVANS' and protocol=='published'")
    emit("realtime.sahm_divergent_all_forecast", int(detail_all["SAHM"]["n_divergent"]), of=int(detail_all["SAHM"]["n_onsets"]),
         spec="SAME as realtime.sahm_divergent but under the retired 'all_forecast' protocol (3 consecutive "
              "model nowcasts) -- the before/after contrast, never a result on its own",
         cites=[],  # superseded before/after audit trail, same convention as mcspower.recentred_power_*
         cross_check="sahm_evans_summary.csv: n_divergent where rule=='SAHM' and protocol=='all_forecast'")
    emit("realtime.evans_divergent_all_forecast", int(detail_all["EVANS"]["n_divergent"]), of=int(detail_all["EVANS"]["n_onsets"]),
         spec="SAME as realtime.evans_divergent but under the retired 'all_forecast' protocol (3 consecutive "
              "model nowcasts) -- the before/after contrast, never a result on its own",
         cites=[],  # superseded before/after audit trail, same convention as mcspower.recentred_power_*
         cross_check="sahm_evans_summary.csv: n_divergent where rule=='EVANS' and protocol=='all_forecast'")
    flush()


if __name__ == "__main__":
    main()
