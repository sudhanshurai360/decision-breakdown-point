"""
walkforward.py — single-series, real-time, rolling-origin (walk-forward) evaluation for the
multi-series macro panel (the macro series beyond the flagship one).
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/

RAW = MULTI / "data" / "raw"

COVID = ("2020-03-01", "2021-12-31")


def _assert_no_calendar_gaps(dates, sid: str) -> None:
    d = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    if len(d) < 2:
        return
    gap_days = d.diff().dt.days.iloc[1:]
    bad = gap_days[gap_days > 35]
    if not bad.empty:
        i = bad.index[0]
        raise ValueError(
            f"walkforward.py: {sid}'s realized monthly series has a calendar gap "
            f"({d.iloc[i - 1].date()} -> {d.iloc[i].date()}, {int(gap_days.loc[i])} days). "
            f"year_scales() and SeasonalNaive index positionally and would silently misalign "
            f"every observation from here forward rather than fail. Fix the source data or "
            f"explicitly handle the gap (see decision_protocol.py's date_shift/date_rolling_mean "
            f"for the calendar-reindex pattern) before re-running."
        )


def load_vintages(sid: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{sid}_alfred_vintages.parquet")
    df["date"] = pd.to_datetime(df["date"]); df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df


def _point_in_time(s: pd.DataFrame) -> pd.DataFrame:
    """For each month, the row with the latest realtime_start in `s`, INCLUDING a row whose value
    is NaN (= the observation was removed from the series as of that vintage); such months are
    then dropped, so a removed observation stays removed instead of falling back to an older
    vintage's value."""
    s = s.sort_values(["date", "realtime_start"]).groupby("date", as_index=False).tail(1)
    s = s.dropna(subset=["value"])
    return s[["date", "value"]].sort_values("date").reset_index(drop=True)


def panel_as_of(v: pd.DataFrame, as_of) -> pd.DataFrame:
    """Series known as of `as_of`: for each month, the value with the latest realtime_start <= as_of."""
    as_of = pd.Timestamp(as_of)
    s = v[v["realtime_start"] <= as_of]
    if s.empty:
        return s[["date", "value"]]
    return _point_in_time(s)


def realized(v: pd.DataFrame, sid: str = "", guard_before=None) -> pd.DataFrame:
    """Latest-vintage (final/revised) value per month — the scoring truth."""
    s = _point_in_time(v)
    dates = s["date"] if guard_before is None else s.loc[s["date"] < pd.Timestamp(guard_before), "date"]
    _assert_no_calendar_gaps(dates, sid)
    return s


def _not_covid(dates: pd.Series) -> np.ndarray:
    lo, hi = pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1])
    d = pd.to_datetime(dates)
    return ((d < lo) | (d > hi)).to_numpy()


def year_scales(truth: pd.DataFrame, years, m=12, squared=False, trail=None) -> dict:
    """scale_Y over realized months with year < Y (leak-free, COVID excluded)."""
    if trail is not None and trail <= 0:
        raise ValueError(f"year_scales: trail must be a positive month count or None, got {trail!r}")
    t = truth.sort_values("date").reset_index(drop=True)
    val = t["value"].to_numpy(float); yr = t["date"].dt.year.to_numpy()
    keep = _not_covid(t["date"])
    out = {}
    for Y in years:
        mask = (yr < Y)
        idx = np.where(mask)[0]
        if trail is not None:
            idx = idx[-trail:]                           # recent-era window: last `trail` months before Y
        if len(idx) <= m:
            out[Y] = np.nan; continue
        sub = val[idx]; subkeep = keep[idx]
        d = sub[m:] - sub[:-m]
        diffs = d ** 2 if squared else np.abs(d)        # squared -> RMSSE denominator; abs -> MASE denominator
        dkeep = subkeep[m:] & subkeep[:-m]
        out[Y] = float(max(np.mean(diffs[dkeep]), 1e-9)) if dkeep.any() else np.nan
    return out


def run(sid: str, models, eval_start=1990, eval_end=2025, m=12) -> pd.DataFrame:
    """Real-time walk-forward for one series."""
    v = load_vintages(sid)
    truth = realized(v, sid, guard_before=f"{eval_end}-01-01")   # year_scales() reads only year < eval_end (see realized())
    tmap = dict(zip(truth["date"], truth["value"]))
    years = list(range(eval_start, eval_end + 1))
    sc_abs = year_scales(truth, years, m=m, squared=False)   # MASE denominator (mean |Δ_m y|)
    sc_sq = year_scales(truth, years, m=m, squared=True)     # RMSSE denominator (mean (Δ_m y)^2)

    # monthly origins across the eval span (forecast made at the 1st of each month)
    cutoffs = pd.date_range(f"{eval_start}-01-01", f"{eval_end}-12-01", freq="MS")
    rows = []
    for c in cutoffs:
        p = panel_as_of(v, c)
        if len(p) < 2 * m + 8:
            continue
        L = p["date"].max()
        target = (L + pd.offsets.MonthBegin(1)).normalize()
        if target.year not in years or target not in tmap:
            continue
        y = np.log1p(p["value"].to_numpy(float))
        ytrue = tmap[target]
        for mdl in models:
            yhat = float(np.expm1(mdl.forecast(y, h=1)))
            rows.append({"series": sid, "model": mdl.name, "target": target,
                         "year": target.year, "y_true": ytrue, "y_pred": yhat})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # dedupe (model, target): keep the latest origin's forecast (most information)
    df = df.drop_duplicates(["model", "target"], keep="last")
    df["scale_abs"] = df["year"].map(sc_abs)
    df["scale_sq"] = df["year"].map(sc_sq)
    df = df.dropna(subset=["scale_abs", "scale_sq"])
    err = df["y_true"] - df["y_pred"]
    df["sae"] = err.abs() / df["scale_abs"]                  # |scaled error|     -> MASE  = mean(sae)
    df["msse"] = (err ** 2) / df["scale_sq"]                 # squared scaled err -> RMSSE = sqrt(mean(msse))
    return df.reset_index(drop=True)


METRIC_COL = {"mase": "sae", "rmsse": "msse"}


def agg_metric(detail: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate prediction detail to per-(series,model,year) loss for one metric ->
    [series,model,year,loss,n]."""
    col = METRIC_COL[metric]
    out = (detail.groupby(["series", "model", "year"], as_index=False)
                 .agg(loss=(col, "mean"), n=(col, "size")))
    if metric == "rmsse":
        out["loss"] = np.sqrt(out["loss"])
    return out


def to_fragility_inputs(agg: pd.DataFrame):
    """Reshape per-year loss into fragility() inputs: L={model: array over years}, w=array(n per year)."""
    years = sorted(agg["year"].unique())
    common_n = agg.groupby("year")["n"].max().reindex(years).to_numpy(float)
    L = {}
    for model, g in agg.groupby("model"):
        L[model] = g.set_index("year")["loss"].reindex(years).to_numpy(float)
    return L, common_n, years
