"""
m_audit.py — the M-COMPETITION RE-AUDIT (generality gate for "the breakdown point of a leaderboard").
"""
import sys, warnings, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
from canonical_emit import emit, flush
from m4_submissions_audit import _leading_run

DATA = MULTI / "data" / "m_audit"
RESULTS = MULTI / "results"
BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
M = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Weekly": 52, "Daily": 7, "Hourly": 24}


def fetch(freq):
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("Train", "Test"):
        f = DATA / f"{freq}-{split.lower()}.csv"
        if not f.exists():
            url = f"{BASE}/{split}/{freq}-{split.lower()}.csv"
            print(f"  fetching {url} ...")
            urllib.request.urlretrieve(url, f)
        out[split] = f
    return out["Train"], out["Test"]


# ── lightweight h-step forecasters (full path) ──────────────────────────────


def f_snaive(y, m, h):
    if len(y) >= m and m > 1:
        return np.array([y[-m + (i % m)] for i in range(h)])
    return np.repeat(y[-1], h)


def f_naive(y, m, h):
    return np.repeat(y[-1], h)


def f_ar1(y, m, h):
    if len(y) < 4:
        return np.repeat(y[-1], h)
    dy = np.diff(y); x, z = dy[:-1], dy[1:]
    xm, zm = x.mean(), z.mean(); den = ((x - xm) ** 2).sum()
    phi = float(((x - xm) * (z - zm)).sum() / den) if den > 1e-12 else 0.0
    phi = max(min(phi, 0.999), -0.999); c = zm - phi * xm
    d, last, out = dy[-1], y[-1], []
    for _ in range(h):
        d = c + phi * d; last = last + d; out.append(last)
    return np.array(out)


def f_ets(y, m, h):
    try:
        if len(y) >= max(2 * m, 10 + 2 * (m // 2)) and m > 1:
            r = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=m,
                                     initialization_method="heuristic").fit()
        elif len(y) >= 10:
            r = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="heuristic").fit()
        else:
            return np.repeat(y[-1], h)
        f = np.asarray(r.forecast(h), float)
        return f if np.all(np.isfinite(f)) else np.repeat(y[-1], h)
    except Exception as e:
        _warn_model_fallback("ets", e)
        _FALLBACK_COUNTS["ets"] += 1
        return np.repeat(y[-1], h)


def f_theta(y, m, h):
    try:
        if len(y) < 4:
            _FALLBACK_COUNTS["theta"] += 1
            return np.repeat(y[-1], h)
        r = ThetaModel(np.asarray(y, float), period=m).fit()
        f = np.asarray(r.forecast(h), float)
        return f if np.all(np.isfinite(f)) else np.repeat(y[-1], h)
    except Exception as e:
        _warn_model_fallback("theta", e)
        _FALLBACK_COUNTS["theta"] += 1
        return np.repeat(y[-1], h)


MODELS = {"snaive": f_snaive, "naive": f_naive, "ar1": f_ar1, "ets": f_ets, "theta": f_theta}

EB_MODELS = {"snaive": f_snaive, "ar1": f_ar1, "ets": f_ets, "theta": f_theta}

_FALLBACK_COUNTS = {"ets": 0, "theta": 0}

_SEEN_FAILURE_TYPES = set()


def _warn_model_fallback(name, exc, context=""):
    """Shared first-occurrence-only stderr print, used by BOTH f_ets/f_theta's own internal
    except-blocks and safe_forecast() below, against the same _SEEN_FAILURE_TYPES set -- a
    (model, exception type) pair prints once per run regardless of which of the two call paths
    hits it first."""
    key = (name, type(exc).__name__)
    if key not in _SEEN_FAILURE_TYPES:
        _SEEN_FAILURE_TYPES.add(key)
        ctx = f" ({context})" if context else ""
        print(f"  [warn] {name} forecast raised {type(exc).__name__}: {exc}{ctx} -- falling back to "
              f"naive persistence. Further {name}/{type(exc).__name__} occurrences this run will be "
              f"tallied, not re-printed.", file=sys.stderr)


def safe_forecast(fn, name, y, m, h, counts=None, context=""):
    """Call a MODELS forecaster, printing (not silently swallowing) the FIRST occurrence of each
    distinct (model, exception type) failure this run -- so a systemic failure (every call raising
    the same way) is visible on its very first call, without flooding output on a large per-series/
    per-month loop."""
    try:
        return fn(y, m, h)
    except Exception as e:
        _warn_model_fallback(name, e, context)
        if counts is not None:
            counts[name] = counts.get(name, 0) + 1
        return np.repeat(y[-1], h)


# ── metrics (aggregate over the h-step path) ────────────────────────────────


def scores(y_train, y_true, y_pred, m):
    e = y_true - y_pred; ae = np.abs(e)
    # in-sample seasonal-naive scale for MASE/RMSSE (LAG-m seasonal diff, not m-th-ORDER diff)
    d = np.abs(y_train[m:] - y_train[:-m]) if (m > 1 and len(y_train) > m) else np.abs(np.diff(y_train))
    scale = d.mean() if len(d) and d.mean() > 0 else (np.abs(y_train).mean() + 1e-9)
    dsq = ((y_train[m:] - y_train[:-m]) ** 2) if (m > 1 and len(y_train) > m) else (np.diff(y_train) ** 2)
    scsq = dsq.mean() if len(dsq) and dsq.mean() > 0 else scale ** 2
    smape = np.mean(2 * ae / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
    return {"MASE": ae.mean() / scale, "RMSSE": np.sqrt((e ** 2).mean() / scsq),
            "sMAPE": smape, "RMSE": np.sqrt((e ** 2).mean()), "MAE": ae.mean()}


def run_one(freq, sample=0):
    """Score one M4 frequency and return its per-series winner table.

    Kept separate from main() so a SINGLE process can run several frequencies and emit() them
    together: canonical_emit.flush() OVERWRITES this stage's whole fragment file on every call (by
    design -- a stage owns its keys completely, see that function's own docstring), so two separate
    CLI invocations (`m_audit.py Weekly` then `m_audit.py Monthly`) would have the second silently
    wipe out the first's emitted values."""
    m = M[freq]
    tr_f, te_f = fetch(freq)
    tr = pd.read_csv(tr_f); te = pd.read_csv(te_f)
    ids = tr.iloc[:, 0].tolist()
    if sample and sample < len(ids):
        rng = np.random.default_rng(0); pick = set(rng.choice(len(ids), sample, replace=False))
        ids = [ids[i] for i in sorted(pick)]
    trv = tr.set_index(tr.columns[0]); tev = te.set_index(te.columns[0])
    metrics = ["MASE", "RMSSE", "sMAPE", "RMSE", "MAE"]
    rows = []
    _FALLBACK_COUNTS["ets"] = 0; _FALLBACK_COUNTS["theta"] = 0
    _n_filtered = 0
    for k, sid in enumerate(ids):
        y = _leading_run(trv.loc[sid]); yt = _leading_run(tev.loc[sid])
        if len(y) < 6 or len(yt) < 1:
            _n_filtered += 1
            continue
        h = len(yt)
        sc = {}
        for name, fn in MODELS.items():
            sc[name] = scores(y, yt, fn(y, m, h), m)
        win = {metric: min(sorted(sc), key=lambda mo: sc[mo][metric]) for metric in metrics}
        rows.append({"id": sid, **{f"win_{metric}": win[metric] for metric in metrics}})
        if (k + 1) % 500 == 0:
            print(f"  ...{k+1}/{len(ids)}")
    d = pd.DataFrame(rows)
    if d.empty:
        raise ValueError(f"m_audit.py::run_one({freq!r}): 0 of {len(ids)} series had >=6 train / "
                          f">=1 test observations -- every series was filtered out, nothing to "
                          f"score. Check the fetched {freq} data isn't empty or corrupted.")
    RESULTS.mkdir(exist_ok=True); d.to_csv(RESULTS / f"m_audit_{freq}.csv", index=False)
    if _FALLBACK_COUNTS["ets"] or _FALLBACK_COUNTS["theta"]:
        print(f"  [warn] {freq}: ets fell back to naive persistence on "
              f"{_FALLBACK_COUNTS['ets']}/{len(ids)} series, theta on "
              f"{_FALLBACK_COUNTS['theta']}/{len(ids)} -- see the [warn] lines above for why. A "
              f"high fraction here (not just a handful of genuinely too-short series) usually means "
              f"a broken/wrong environment, not real per-series fit failures.")
    if _n_filtered:
        print(f"  [warn] {freq}: {_n_filtered}/{len(ids)} series filtered out before scoring "
              f"(len(train)<6 or len(test)<1)")

    n = len(d)
    # headline: does the winner flip between MASE and RMSSE? and across ALL 5 metrics?
    flip_mase_rmsse = (d["win_MASE"] != d["win_RMSSE"]).mean()
    n_distinct = d[[f"win_{x}" for x in metrics]].nunique(axis=1)
    any_flip = (n_distinct > 1).mean()
    print(f"\n=== M4 {freq} RE-AUDIT: {n} series, 5 standard models, 5 metrics ===")
    print(f"  winner FLIPS MASE vs RMSSE:        {flip_mase_rmsse:.1%}")
    print(f"  winner NOT metric-invariant (>=2 distinct winners across the 5 metrics): {any_flip:.1%}")
    print(f"  mean # distinct winners across metrics: {n_distinct.mean():.2f}")
    print("\n  per-metric winner distribution:")
    for metric in metrics:
        print(f"    {metric:6s}: {dict(d['win_'+metric].value_counts())}")
    print(f"\nwrote {RESULTS/('m_audit_'+freq+'.csv')}")
    return {"n": n, "flip_mase_rmsse": float(flip_mase_rmsse), "any_flip": float(any_flip),
            "mean_distinct": float(n_distinct.mean())}


def main():
    if "--emit" in sys.argv:
        results = {freq: run_one(freq) for freq in ("Weekly", "Monthly")}
        for freq, r in results.items():
            fx = freq.lower()
            emit(f"maudit.flip_mase_rmsse_{fx}", r["flip_mase_rmsse"], n=r["n"],
                 spec=f"fraction of M4 {freq} series (5 standard forecasters: snaive/naive/ar1/ets/"
                      f"theta) whose argmin winner differs between MASE and RMSSE scoring",
                 cites=["S9 body: per-series winner-flip audit"],
                 cross_check=f"m_audit_{freq}.csv: (win_MASE != win_RMSSE).mean()")
            emit(f"maudit.any_flip_{fx}", r["any_flip"], n=r["n"],
                 spec=f"fraction of M4 {freq} series where the argmin winner is not the same "
                      f"model across all 5 scoring metrics (MASE/RMSSE/sMAPE/RMSE/MAE)",
                 cites=["S9 body: per-series winner-flip audit"],
                 cross_check=f"m_audit_{freq}.csv: nunique(win_* across 5 metrics) > 1, mean")
            emit(f"maudit.mean_distinct_{fx}", r["mean_distinct"], of=len(["MASE", "RMSSE", "sMAPE", "RMSE", "MAE"]),
                 spec=f"mean number of distinct argmin winners per M4 {freq} series across the 5 "
                      f"scoring metrics (MASE/RMSSE/sMAPE/RMSE/MAE), 5 standard forecasters "
                      f"(snaive/naive/ar1/ets/theta)",
                 cites=[],
                 cross_check=f"m_audit_{freq}.csv: nunique(win_* across 5 metrics), mean")
        flush()
        return
    freq = sys.argv[1] if len(sys.argv) > 1 else "Weekly"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_one(freq, sample)


if __name__ == "__main__":
    main()
