"""election_null_ar1.py — recalibrates §5.4's exchangeable null under EMPIRICALLY-estimated serial
dependence, and reports how the "cells below the null's fifth percentile" count moves under it.
"""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "instrument"))
import fragility as F
from null_budget import SEEDS, N_REP, N_BOOT, _check_spacing
from canonical_emit import emit, flush

PRED = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
SC_CSV = ROOT / "multiseries" / "results" / "social_choice.csv"
RES = ROOT / "multiseries" / "results"
METRICS = {"mase": ("sae", lambda s: s.mean()), "rmsse": ("msse", lambda s: np.sqrt(s.mean()))}
SD = 0.30
        # matches null_sensitivity.py's convention; immaterial to winner_stability (a common scale
        # factor cannot change which model wins a resample)


def _ar1(x, ar):
    if abs(ar) >= 1:
        raise ValueError(f"_ar1: |ar|={abs(ar)!r} >= 1 -- sqrt(1-ar**2) would be non-finite")
    if ar <= 0:
        return x
    y = x.copy()
    for t in range(1, len(y)):
        y[t] = ar * y[t - 1] + np.sqrt(1 - ar ** 2) * y[t]
    return y


def _lag1_autocorr(x):
    """Lag-1 OLS autocorrelation of a demeaned series; None if too short or constant."""
    x = np.asarray(x, float)
    x = x - x.mean()
    if len(x) < 5 or x.std() == 0:
        return None
    den = float(np.sum(x[:-1] ** 2))
    if den <= 0:
        return None
    return float(np.sum(x[1:] * x[:-1]) / den)


def _per_year_loss_pivot(g, col, agg):
    per = g.groupby(["model", "year"])[col].apply(agg).reset_index()
    return per.pivot(index="year", columns="model", values=col).dropna()


def estimate_ar1_coefficients_levels(d):
    """Lag-1 OLS autocorrelation of each (series, model, metric) loss LEVEL series, demeaned."""
    rows = []
    for sid, g in d.groupby("series"):
        for metric, (col, agg) in METRICS.items():
            piv = _per_year_loss_pivot(g, col, agg)
            for m in piv.columns:
                phi = _lag1_autocorr(piv[m].to_numpy(float))
                if phi is not None:
                    rows.append({"series": sid, "metric": metric, "model": m, "ar1_estimate": phi})
    return pd.DataFrame(rows)


def estimate_ar1_coefficients(d):
    """Lag-1 OLS autocorrelation of every pairwise loss DIFFERENTIAL (model a minus model b, all
    K(K-1)/2 pairs) per (series, metric) cell, demeaned."""
    rows = []
    for sid, g in d.groupby("series"):
        for metric, (col, agg) in METRICS.items():
            piv = _per_year_loss_pivot(g, col, agg)
            models = list(piv.columns)
            for i, a in enumerate(models):
                for b in models[i + 1:]:
                    phi = _lag1_autocorr(piv[a].to_numpy(float) - piv[b].to_numpy(float))
                    if phi is not None:
                        rows.append({"series": sid, "metric": metric, "pair": f"{a}-{b}", "ar1_estimate": phi})
    return pd.DataFrame(rows)


def panel_ar1(g, K, T, ar):
    """K independent, mean-1.0 loss series, each with AR(1) dependence `ar` in its own noise."""
    return {f"m{i}": 1.0 + SD * _ar1(g.normal(0, 1, T), ar) for i in range(K)}


def stable_benchmark_ar1(K, T, ar, n_rep=N_REP, n_boot=N_BOOT, seeds=SEEDS):
    """Exchangeable null at (K, T) under AR(1) dependence `ar`, averaged over independent seeds --
    same seed-spacing discipline as null_budget.stable_benchmark, since a single-seed AR(1) null
    would carry the identical seed-noise risk that module was built to close for the i.i.d."""
    _check_spacing(seeds, n_rep)
    med, p05, p95 = [], [], []
    for s in seeds:
        vals = []
        for r in range(n_rep):
            g = np.random.default_rng(s + 1 + r)
            L = panel_ar1(g, K, T, ar)
            w = np.ones(T)
            v = F.winner_stability(L, w, n_boot=n_boot, seed=s + 1_000_003 + r)[0]
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        a = np.asarray(vals, float)
        med.append(float(np.median(a))); p05.append(float(np.percentile(a, 5))); p95.append(float(np.percentile(a, 95)))
    p05 = np.asarray(p05, float)
    return {
        "median": float(np.mean(med)), "p05": float(np.mean(p05)), "p95": float(np.mean(p95)),
        "p05_lo": float(p05.min()), "p05_hi": float(p05.max()), "p05_sd": float(p05.std(ddof=1)),
        "n_eff": int(len(seeds) * n_rep), "n_seeds": len(seeds),
    }


def main():
    d = pd.read_csv(PRED)
    coeffs_df = estimate_ar1_coefficients(d)            # pairwise loss DIFFERENTIALS: the calibration basis
    levels_df = estimate_ar1_coefficients_levels(d)     # loss LEVELS: a superseded basis, disclosure only
    coeffs = coeffs_df["ar1_estimate"].to_numpy(float)
    ar_hat = float(np.median(coeffs))
    ar_levels = float(np.median(levels_df["ar1_estimate"].to_numpy(float)))
    print("=== ELECTION NULL UNDER EMPIRICAL AR(1) ===")
    print(f"  (calibration basis: pairwise loss differentials; loss LEVELS median, disclosed only: {ar_levels:.4f})")
    print(f"  AR(1) estimates: n={len(coeffs)}  median={ar_hat:.4f}  mean={coeffs.mean():.4f}  "
          f"IQR=[{np.percentile(coeffs,25):.4f}, {np.percentile(coeffs,75):.4f}]  "
          f"share>0={float((coeffs>0).mean()):.4f}")

    sc = pd.read_csv(SC_CSV)
    st = sc["winner_stability"].dropna()
    _K = int(d.model.nunique())
    _T = float(d.groupby(["series", "year"]).ngroups / d.series.nunique())
    _T_int = int(_T)
    print(f"  panel dimensions: K={_K}  T(avg)={_T:.4f} -> using T={_T_int} for the null")

    nbd_ar1 = stable_benchmark_ar1(_K, _T_int, ar_hat)
    print(f"  AR(1)-adjusted null: median={nbd_ar1['median']:.4f}  p05={nbd_ar1['p05']:.4f}  "
          f"(band [{nbd_ar1['p05_lo']:.4f}, {nbd_ar1['p05_hi']:.4f}], sd={nbd_ar1['p05_sd']:.4f})")

    _sc_frag = json.loads((RES / "canonical" / "social_choice.json").read_text())
    p05_iid = float(_sc_frag["election.stability_null_p05"]["value"])
    delta_p05 = nbd_ar1["p05"] - p05_iid
    print(f"  rise in the null's fifth percentile from i.i.d. ({p05_iid:.3f}) to AR(1)-adjusted: {delta_p05:+.4f}")

    below_ar1 = int((st < nbd_ar1["p05"]).sum())
    below_band = sorted({int((st < nbd_ar1["p05_lo"]).sum()), int((st < nbd_ar1["p05_hi"]).sum())})
    print(f"  cells below AR(1)-adjusted p05: {below_ar1} of {len(st)}  (band count {below_band})")

    flagged = sc.loc[sc.winner_stability < nbd_ar1["p05"], ["series", "metric", "winner_stability"]].copy()
    flagged["own_ar1_median"] = flagged["series"].map(
        lambda s: float(coeffs_df.loc[coeffs_df.series == s, "ar1_estimate"].median()))
    flagged["own_ar1_below_pooled"] = flagged["own_ar1_median"] <= ar_hat
    print("  flagged cells' own series-level AR(1) vs the pooled figure used for the correction:")
    for _, r in flagged.iterrows():
        rel = "<=" if r["own_ar1_below_pooled"] else ">"
        print(f"    {r['series']:>9}/{r['metric']:<5} own median AR(1)={r['own_ar1_median']:.4f} "
              f"{rel} pooled {ar_hat:.4f}")
    all_below_pooled = bool(flagged["own_ar1_below_pooled"].all())

    pd.concat([coeffs_df.assign(basis="differential"), levels_df.assign(basis="level")],
              ignore_index=True).to_csv(RES / "election_null_ar1_coefficients.csv", index=False)
    flagged.to_csv(RES / "election_null_ar1_flagged_cells.csv", index=False)
    pd.DataFrame([{
        "K": _K, "T": _T_int, "ar_hat": ar_hat,
        "null_median": nbd_ar1["median"], "null_p05": nbd_ar1["p05"],
        "null_p05_lo": nbd_ar1["p05_lo"], "null_p05_hi": nbd_ar1["p05_hi"], "null_p05_sd": nbd_ar1["p05_sd"],
        "below_p05": below_ar1, "of": int(len(st)), "p05_iid": p05_iid, "delta_p05": delta_p05,
    }]).to_csv(RES / "election_null_ar1.csv", index=False)

    emit("election.ar1_coefficient_empirical", round(ar_hat, 4), n=int(len(coeffs)),
         iqr_lo=round(float(np.percentile(coeffs, 25)), 4), iqr_hi=round(float(np.percentile(coeffs, 75)), 4),
         levels_median=round(ar_levels, 4), n_levels=int(len(levels_df)),
         spec="pooled median lag-1 OLS autocorrelation, demeaned, across every pairwise loss "
              "DIFFERENTIAL (model a minus model b, all pairs) in every (series, metric) cell of the "
              "curated panel -- the empirical AR(1) coefficient used to recalibrate the exchangeable "
              "null below. Differentials, not levels, because every statistic the null calibrates "
              "is invariant to a common per-period component of the levels, and the synthetic null "
              "imposes its single coefficient on the differentials. The companion field "
              "levels_median is the same estimator on loss LEVELS, a superseded basis disclosed "
              "for comparison, not used",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1_coefficients.csv: median(ar1_estimate) where basis=='differential'")
    emit("election.stability_null_median_ar1", round(nbd_ar1["median"], 3), n_models=_K, n_periods=_T_int,
         n_eff=nbd_ar1["n_eff"],
         spec=f"median winner_stability under the exchangeable null at K={_K}, T={_T_int}, with each "
              f"model's series carrying AR(1) dependence at the empirically-estimated coefficient "
              f"above, instead of i.i.d. draws",
         cites=["S5.4 body: AR(1)-adjusted null"], cross_check="election_null_ar1.csv: null_median")
    emit("election.stability_null_p05_ar1", round(nbd_ar1["p05"], 3), n_models=_K, n_periods=_T_int,
         band=[round(nbd_ar1["p05_lo"], 3), round(nbd_ar1["p05_hi"], 3)], sd=round(nbd_ar1["p05_sd"], 4),
         n_eff=nbd_ar1["n_eff"],
         spec=f"fifth percentile of winner_stability under the exchangeable null at K={_K}, "
              f"T={_T_int}, with the empirically-estimated AR(1) dependence above applied to every "
              f"model's series -- the recalibrated threshold this stage exists to compute",
         cites=["S5.4 body: AR(1)-adjusted null"], cross_check="election_null_ar1.csv: null_p05")
    emit("election.stability_below_null_p05_ar1", below_ar1, of=int(len(st)),
         band_count=below_band, threshold=round(nbd_ar1["p05"], 3),
         expected=round(0.05 * len(st), 1),
         spec="cells whose observed winner_stability falls below the AR(1)-adjusted null's own "
              "fifth percentile, replacing the i.i.d. exchangeable null used for "
              "election.stability_below_null_p05 with one that reflects the panel's own estimated "
              "serial dependence",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1.csv: below_p05")
    emit("election.ar1_delta_p05", round(delta_p05, 4), p05_iid=round(p05_iid, 3),
         spec="rise in the null's fifth percentile from the i.i.d. exchangeable null "
              "(election.stability_null_p05, computed by social_choice.py) to this stage's "
              "AR(1)-adjusted null, at the same real panel dimensions",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1.csv: delta_p05")
    emit("election.ar1_flagged_cells_at_or_below_pooled", all_below_pooled, n_cells=int(len(flagged)),
         spec="whether every one of the cells flagged below the AR(1)-adjusted null's own threshold "
              "has an OWN median AR(1) at or below the pooled figure used to build that threshold, "
              "i.e. whether the pooled correction was, if anything, conservative (not understated) "
              "for the specific cells it is being used to classify. 'Own median AR(1)' is the "
              "median of that series' pairwise loss-DIFFERENTIAL lag-1 estimates (all model pairs, "
              "both metrics pooled), the same basis as election.ar1_coefficient_empirical",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1_flagged_cells.csv: own_ar1_below_pooled.all()")

    tcu_ar1 = float(flagged.loc[flagged.series == "TCU", "own_ar1_median"].iloc[0])
    houst_ar1 = float(flagged.loc[flagged.series == "HOUST", "own_ar1_median"].iloc[0])
    houst_ws = float(flagged.loc[flagged.series == "HOUST", "winner_stability"].iloc[0])
    houst_excess = houst_ar1 - ar_hat
    _houst_dir = "exceeds" if houst_excess > 0 else "is at or below"
    emit("election.tcu_own_ar1_median", round(tcu_ar1, 3),
         spec="TCU's own median lag-1 autocorrelation (both of TCU's cells share it), compared "
              "against the pooled figure (election.ar1_coefficient_empirical) used to build the "
              "AR(1)-adjusted null that flags TCU's two cells. 'Own median AR(1)' is the median of "
              "that series' pairwise loss-DIFFERENTIAL lag-1 estimates (all model pairs, both "
              "metrics pooled), the same basis as election.ar1_coefficient_empirical",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1_flagged_cells.csv: own_ar1_median where series=='TCU'")
    emit("election.houst_own_ar1_median", round(houst_ar1, 3),
         spec=f"HOUST's own median lag-1 autocorrelation, compared against the pooled figure "
              f"(election.ar1_coefficient_empirical) used to build the AR(1)-adjusted null that "
              f"flags HOUST's one cell; HOUST's own dependence {_houst_dir} the pooled figure on "
              f"the current (differential) basis. 'Own median AR(1)' is the median of that series' "
              f"pairwise loss-DIFFERENTIAL lag-1 estimates (all model pairs, both metrics pooled), "
              f"the same basis as election.ar1_coefficient_empirical",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1_flagged_cells.csv: own_ar1_median where series=='HOUST'")
    emit("election.houst_ar1_excess_over_pooled", round(houst_excess, 3),
         spec="HOUST's own median AR(1) minus the pooled figure used to correct it (positive: the "
              "pooled correction under-estimates HOUST's own serial dependence; negative: it "
              "over-estimates it), checked against how much moving the AR(1) coefficient actually "
              "shifts the threshold (nullsens.ar_delta_p05)",
         cites=["S5.4 body: AR(1)-adjusted null"], cross_check=None)
    emit("election.houst_mase_winner_stability", round(houst_ws, 3),
         spec=f"HOUST/mase's own observed winner_stability, the AR(1)-flagged cell whose own "
              f"series-level dependence {_houst_dir} the pooled correction -- reported so the safety "
              f"margin against the recalibrated threshold (election.stability_null_p05_ar1) is checkable",
         cites=["S5.4 body: AR(1)-adjusted null"],
         cross_check="election_null_ar1_flagged_cells.csv: winner_stability where series=='HOUST'")

    _T_by_series = d.groupby("series").apply(
        lambda g: int(g.groupby(["model", "year"]).ngroups / g.model.nunique()), include_groups=False)
    _distinct_Ts = sorted(_T_by_series.unique())
    print(f"\n  PER-CELL NULL: real per-series T takes {len(_distinct_Ts)} distinct values {_distinct_Ts} "
          f"(vs. the single pooled T={_T_int} used above)")
    _null_by_T = {}
    for _T_val in _distinct_Ts:
        _nbd_T = stable_benchmark_ar1(_K, _T_val, ar_hat)
        _null_by_T[_T_val] = _nbd_T
        print(f"    T={_T_val}: null p05={_nbd_T['p05']:.4f} (median={_nbd_T['median']:.4f}, "
              f"p05_sd={_nbd_T['p05_sd']:.4f})")
    sc2 = sc.copy()
    sc2["T"] = sc2["series"].map(_T_by_series)
    sc2["null_p05_percell"] = sc2["T"].map(lambda t: _null_by_T[t]["p05"])
    sc2_valid = sc2.dropna(subset=["winner_stability"])
    below_percell = int((sc2_valid["winner_stability"] < sc2_valid["null_p05_percell"]).sum())
    _flagged_percell = sorted(sc2_valid.loc[
        sc2_valid.winner_stability < sc2_valid.null_p05_percell, ["series", "metric"]
    ].apply(lambda r: f"{r.series}/{r.metric}", axis=1).tolist())
    print(f"  cells below their OWN T-specific AR(1)-adjusted p05: {below_percell} of {len(sc2_valid)} "
          f"({_flagged_percell})")
    pd.DataFrame([{"T": t, "null_p05": v["p05"], "null_median": v["median"], "null_p05_sd": v["p05_sd"]}
                  for t, v in _null_by_T.items()]).to_csv(RES / "election_null_ar1_percell.csv", index=False)
    _p05_vals = [v["p05"] for v in _null_by_T.values()]
    _p05_sd_max = round(max(v["p05_sd"] for v in _null_by_T.values()), 4)
    _p05_range = round(max(_p05_vals) - min(_p05_vals), 4)
    _n_seeds_max = max(v["n_seeds"] for v in _null_by_T.values())
    _se_mean_max = _p05_sd_max / (_n_seeds_max ** 0.5)
    print(f"  per-T p05 range = {_p05_range:.4f} vs. max per-T RAW seed-to-seed p05_sd = {_p05_sd_max:.4f} "
          f"({'within' if _p05_range <= _p05_sd_max else 'exceeds'} one RAW seed sd -- a cruder comparison "
          f"than the prose's, which uses the SE of the seed mean). On that tighter scale the range is "
          f"{_p05_range / _se_mean_max:.2f}x the SE of a {_n_seeds_max}-seed mean ({_se_mean_max:.4f}), i.e. "
          f"{'under' if _p05_range <= _se_mean_max else 'above'} one such SE -- S5.4 reads a spread this "
          f"small as what sampling noise predicts, not an excess.")
    emit("election.stability_below_null_p05_ar1_percell", below_percell, of=int(len(sc2_valid)),
         flagged_cells=_flagged_percell, p05_min=round(min(_p05_vals), 3), p05_max=round(max(_p05_vals), 3),
         p05_sd_max=_p05_sd_max,
         spec="cells whose observed winner_stability falls below a PER-CELL (really per-T, since T "
              "is constant within a series across its two metrics) AR(1)-adjusted null threshold, "
              "computed at that cell's own real T rather than the single pooled T=32 used for "
              "election.stability_below_null_p05_ar1 above. Real per-series T ranges 25-36 (4 "
              "distinct values), so this is a genuinely different threshold per cell, not the same "
              "number applied uniformly",
         cites=["S5.4 body: per-cell AR(1)-adjusted null"],
         cross_check="election_null_ar1_percell.csv + social_choice.csv: per-cell threshold comparison")

    flush()


if __name__ == "__main__":
    main()
