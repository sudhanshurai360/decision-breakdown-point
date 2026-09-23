"""
density_audit.py — the LEADERBOARD-DENSITY curve on M4-Monthly (needs only the M4 train/test CSVs, no submission archives).
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "instrument"))
import m_audit as MA
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
from canonical_emit import emit, flush
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES


def f_drift(y, m, h):
    if len(y) < 2: return np.repeat(y[-1], h)
    s = (y[-1] - y[0]) / (len(y) - 1); return y[-1] + s * np.arange(1, h + 1)


def f_mean(y, m, h): return np.repeat(np.mean(y[-min(len(y), 2 * m if m > 1 else 6):]), h)
_FALLBACK_TALLY: dict[tuple[str, str], int] = {}


def _warn_fallback(model_name, exc):
    key = (model_name, type(exc).__name__)
    _FALLBACK_TALLY[key] = _FALLBACK_TALLY.get(key, 0) + 1
    if _FALLBACK_TALLY[key] == 1:
        print(f"  [warn] {model_name} forecast raised {type(exc).__name__}: {exc} -- falling back "
              f"to naive persistence. Further {model_name}/{type(exc).__name__} occurrences this "
              f"run will be silently tallied, not re-printed.", file=sys.stderr)


def f_ses(y, m, h):
    try:
        if len(y) < 10:
            return np.repeat(y[-1], h)
        return np.asarray(ES(y, initialization_method="heuristic").fit().forecast(h), float)
    except Exception as e:
        _warn_fallback("ses", e)
        return np.repeat(y[-1], h)


def f_damped(y, m, h):
    try:
        if len(y) >= 10: return np.asarray(ES(y, trend="add", damped_trend=True, initialization_method="heuristic").fit().forecast(h), float)
    except Exception as e:
        _warn_fallback("damped", e)
    return np.repeat(y[-1], h)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    MODELS = {**MA.MODELS, "drift": f_drift, "mean": f_mean, "ses": f_ses, "damped": f_damped}  # 9
    tr_f, te_f = MA.fetch("Monthly")
    tr = pd.read_csv(tr_f); te = pd.read_csv(te_f)
    ids = tr.iloc[:, 0].tolist(); rng = np.random.default_rng(0)
    ids = [ids[i] for i in sorted(rng.choice(len(ids), min(n, len(ids)), replace=False))]
    trv = tr.set_index(tr.columns[0]); tev = te.set_index(te.columns[0]); m = 12
    mae, rmse = {}, {}
    for sid in ids:
        y = MA._leading_run(trv.loc[sid]); yt = MA._leading_run(tev.loc[sid])
        if len(y) < 8 or len(yt) < 1: continue
        h = len(yt); mae[sid], rmse[sid] = {}, {}
        for nm, fn in MODELS.items():
            e = yt - fn(y, m, h); mae[sid][nm] = np.abs(e).mean(); rmse[sid][nm] = np.sqrt((e ** 2).mean())
    sids = list(mae); names = list(MODELS)
    rows = []
    for k in range(2, len(names) + 1):
        flips = []
        for sid in sids:
            for _ in range(6):
                sub = list(rng.choice(names, k, replace=False))
                flips.append(min(sub, key=lambda x: mae[sid][x]) != min(sub, key=lambda x: rmse[sid][x]))
        rows.append({"n_models": k, "flip_rate": float(np.mean(flips))})
    d = pd.DataFrame(rows)
    _csv_out = MA.RESULTS / "m4_density_curve.csv"
    RM._atomic_to_csv(d, _csv_out)
    print(f"M4-Monthly density curve ({len(sids)} series, up to {len(names)} models):")
    print(d.to_string(index=False)); print(f"wrote {MA.RESULTS/'m4_density_curve.csv'}")

    if n != 1500:
        print(f"  (n={n} != the documented default 1500: skipping canonical emit -- this is a proof/test "
              f"run, not the run the manuscript cites)")
    else:
        emit("density.n_series", len(sids), spec="number of M4-Monthly series sampled for the leaderboard-density curve",
             cites=[], cross_check=None)  # m4_density_curve.csv has one row per pool size, not per series;
                                           # the per-series id list is never persisted, so this has no
                                           # independent second derivation
        emit("density.n_models_max", len(names), spec="largest leaderboard pool size swept (k=2..n_models)",
             cites=[], cross_check="m4_density_curve.csv: n_models.max()")
        emit("density.flip_rate_min_pool", float(d.flip_rate.iloc[0]), n_models=int(d.n_models.iloc[0]),
             spec="MAE-vs-RMSE winner flip rate at the smallest pool size (n_models=2)",
             cites=[], cross_check="m4_density_curve.csv: flip_rate at n_models==2")
        emit("density.flip_rate_max_pool", float(d.flip_rate.iloc[-1]), n_models=int(d.n_models.iloc[-1]),
             spec="MAE-vs-RMSE winner flip rate at the largest pool size swept",
             cites=[], cross_check="m4_density_curve.csv: flip_rate at n_models==n_models.max()")
        strict_increasing = bool(np.all(np.diff(d.flip_rate.to_numpy()) > 0))
        emit("density.flip_rate_monotonic_increasing", strict_increasing,
             spec="whether the raw flip-rate curve rises STRICTLY with pool size over the swept "
                  "range (n_models=2..9, every step a real increase, not just non-decreasing) -- if "
                  "true, the curve itself does not visibly plateau within this range. Section 9 "
                  "does not read an identification verdict off this curve: it says the M4 crowding "
                  "arm 'is not a reliable identification signal on its own: on a 61-row panel the "
                  "independence null is too unstable to read'. This producer computes NO "
                  "independence null for this curve at all, and none is available elsewhere: the "
                  "arena.m4_flip_null_* keys are a different measurement on a different panel (a "
                  "mean-loss-vs-Borda flip rate over the 61 M4 SUBMISSIONS x 4 metrics, pools up to "
                  "60), not a null for this MAE-vs-RMSE curve over ~1500 M4-Monthly SERIES x 9 "
                  "models",
             cites=[], cross_check="m4_density_curve.csv: all(diff(flip_rate) > 0)")
        flush()


if __name__ == "__main__":
    main()
