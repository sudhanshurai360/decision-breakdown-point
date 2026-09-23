"""model_set_audit.py — how similar are the six competing models, really?
"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "instrument"))
from canonical_emit import emit, flush
import mcs_test as MT

ROOT = HERE.parent.parent
RES = ROOT / "multiseries" / "results"
PRED = RES / "breadth_distinct" / "breadth_distinct_predictions.csv"
MONTHLY = RES / "mcs_monthly_resolution.csv"

# The deliberately-weak benchmark. Named as a constant because every statistic below either excludes
# it or is about it; it is not one of the five substantive candidates.
BENCHMARK = "seasonal_naive"
ALPHA, B, BLOCK, SEED = 0.10, 2000, 3, 0


def main():
    d = pd.read_csv(PRED)
    d["year"] = pd.to_datetime(d.target).dt.year
    rows, corrs = [], []
    for sid, g in d.groupby("series"):
        # Equal-year weighting throughout, matching the convention S5 states and Table 1 uses.
        yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
        means = yl.mean(0).sort_values()
        best, runner = means.index[0], means.index[1]
        gap_pct = float((means.iloc[1] - means.iloc[0]) / means.iloc[0] * 100)
        others = [m for m in yl.columns if m != BENCHMARK]
        pair_r = [float(yl[a].corr(yl[b])) for a, b in itertools.combinations(sorted(others), 2)]
        corrs += pair_r
        models = list(yl.columns)
        surv, _ = MT.mcs(yl.to_numpy(float), alpha=ALPHA, B=B, block=BLOCK, seed=SEED)
        in_mcs = [models[i] for i in surv]
        row = {
            "series": sid, "best": best, "runner_up": runner,
            "gap_pct": round(gap_pct, 3),
            "benchmark_rank": int(list(means.index).index(BENCHMARK)) + 1,
            "n_models": len(yl.columns),
            "benchmark_ratio": round(float(means[BENCHMARK] / means.iloc[0]), 3),
            "benchmark_in_mcs": BENCHMARK in in_mcs,
            "mcs_size": len(in_mcs),
            "loss_corr_median": round(float(np.median(pair_r)), 4),
            "loss_corr_min": round(float(np.min(pair_r)), 4),
            "mcs_members": ";".join(in_mcs),
        }
        for _m in means.index:
            row[f"loss_{_m}"] = float(means[_m])
        rows.append(row)
    R = pd.DataFrame(rows).sort_values("series").reset_index(drop=True)
    R.to_csv(RES / "model_set_audit.csv", index=False)
    n = len(R)

    corrs = np.asarray(corrs, dtype=float)
    _n_nan = int(np.isnan(corrs).sum())
    if _n_nan:
        print(f"  WARNING: {_n_nan}/{len(corrs)} pairwise loss correlations are NaN (a constant-loss "
              f"column) -- min/median below correctly propagate NaN rather than silently dropping it")

    print(f"=== HOW SIMILAR ARE THE {len(d.model.unique())} MODELS? ({n} curated series) ===")
    print(R.drop(columns=["mcs_members"]).to_string(index=False))
    print(f"\n  median best-vs-runner-up pooled gap: {R.gap_pct.median():.2f}%  "
          f"(min {R.gap_pct.min():.2f}%, max {R.gap_pct.max():.2f}%)")
    n_sub = len(set(d.model.unique()) - {BENCHMARK})
    print(f"  pairwise annual-loss correlation among the {n_sub} non-benchmark models: "
          f"median {np.median(corrs):.3f}, min {np.min(corrs):.3f}")
    print(f"  {BENCHMARK} is LAST in {int((R.benchmark_rank == R.n_models).sum())}/{n} and in the MCS in "
          f"{int(R.benchmark_in_mcs.sum())}/{n}")

    emit("modelset.median_gap_pct", round(float(R.gap_pct.median()), 2), n_series=n,
         spec="median across the 12 curated series of the pooled (equal-year-weighted) mean-loss gap "
              "between the BEST and RUNNER-UP model, in percent of the best model's own mean loss. "
              "The scale of the difference the MCS is being asked to detect.",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: gap_pct.median()")
    emit("modelset.min_gap_pct", round(float(R.gap_pct.min()), 2), n_series=n,
         spec="smallest best-vs-runner-up pooled gap across the curated series (percent)",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: gap_pct.min()")
    emit("modelset.max_gap_pct", round(float(R.gap_pct.max()), 2), n_series=n,
         spec="largest best-vs-runner-up pooled gap across the curated series (percent)",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: gap_pct.max()")
    emit("modelset.gap_under_1pct", int((R.gap_pct < 1).sum()), of=n,
         spec="series whose best-vs-runner-up pooled gap is below 1 percent",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: (gap_pct < 1).sum()")
    emit("modelset.loss_corr_median", round(float(np.median(corrs)), 3), n_pairs=len(corrs),
         spec="median pairwise Pearson correlation of the ANNUAL LOSS series among the five "
              "non-benchmark models, pooled over all series and pairs. Loss rather than forecast "
              "level on purpose: two forecasts of the same trending series correlate near 1 "
              "mechanically, so a forecast-level correlation would overstate the similarity.",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: np.median(all pooled pairwise loss correlations)")
    emit("modelset.loss_corr_min", round(float(np.min(corrs)), 3), n_pairs=len(corrs),
         spec="smallest pairwise annual-loss correlation among the five non-benchmark models, over "
              "every (series, pair) -- the least similar any two substantive candidates ever are. "
              "Attained on BOPGEXP (see model_set_audit.csv's per-series loss_corr_min column: "
              "min(that column) reproduces this value, since both are min(pooled "
              "pair_r) -- the median column alone could not re-derive it).",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: loss_corr_min.min()")
    emit("modelset.benchmark_last", int((R.benchmark_rank == R.n_models).sum()), of=n,
         spec=f"series in which {BENCHMARK} has the WORST pooled mean loss of the models that "
              f"actually competed on that series (six today, but compared against each series' own "
              f"model count, not a fixed number). The curated series are pre-seasonally-adjusted by "
              f"their source agency, so a 12-month seasonal-naive forecast is a deliberately weak "
              f"reference, not a rival.",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: (benchmark_rank == n_models).sum()")
    emit("modelset.benchmark_ratio_max", round(float(R.benchmark_ratio.max()), 2), n_series=n,
         spec=f"largest ratio of {BENCHMARK}'s pooled mean loss to the best model's, across series",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: benchmark_ratio.max()")
    emit("modelset.benchmark_in_mcs", int(R.benchmark_in_mcs.sum()), of=n,
         spec=f"series whose Model Confidence Set CONTAINS {BENCHMARK} (alpha={ALPHA}, absolute "
              f"loss, block={BLOCK}, B={B}). The complement is the count in which the one "
              f"structurally different candidate is correctly eliminated -- so the tied sets are "
              f"almost entirely composed of the five near-collinear models.",
         cites=["S3.1 body: how similar are the candidate models"],
         cross_check="model_set_audit.csv: benchmark_in_mcs.sum()")

    same_eR = same_emax = 0
    rule_rows = []
    for sid, g in d.groupby("series"):
        yl = g.groupby(["model", "year"])["sae"].mean().unstack(0).dropna()
        mm, X = list(yl.columns), yl.to_numpy(float)
        a, _ = MT.mcs(X, alpha=ALPHA, B=B, block=BLOCK, seed=SEED, elimination="raw")
        e, _ = MT.mcs(X, alpha=ALPHA, B=B, block=BLOCK, seed=SEED, elimination="e_R")
        b, _ = MT.mcs(X, alpha=ALPHA, B=B, block=BLOCK, seed=SEED, elimination="studentized")
        s_raw, s_eR, s_emax = (sorted(mm[i] for i in x) for x in (a, e, b))
        same_eR += s_raw == s_eR
        same_emax += s_raw == s_emax
        rule_rows.append({"series": sid, "raw": ";".join(s_raw), "e_R": ";".join(s_eR),
                          "e_max": ";".join(s_emax), "raw_eq_eR": s_raw == s_eR,
                          "raw_eq_emax": s_raw == s_emax})
    RR = pd.DataFrame(rule_rows)
    RR.to_csv(RES / "mcs_elimination_rule_check.csv", index=False)
    differ_eR = sorted(RR.series[~RR.raw_eq_eR].tolist())
    print(f"  elimination rule: raw vs e_R identical surviving sets on {same_eR}/{n} series "
          f"(differ: {differ_eR}); raw vs e_max identical on {same_emax}/{n}")
    emit("modelset.elimination_identical_sets", int(same_eR), of=n, differ=differ_eR,
         spec="curated series on which the raw-mean elimination rule (this paper's default) and "
              "HLN's e_R rule -- the rule HLN pair with the T_R range statistic, and the rule arch's "
              "MCS(method='R') implements -- return the IDENTICAL surviving set (same models, not "
              f"merely the same cardinality), at alpha={ALPHA}, block={BLOCK}, B={B}. Bears directly "
              "on S4.1: on the series where they differ, e_R reproduces arch's set, so the "
              "elimination rule IS part of the residual arch-vs-ours disagreement (the rest is "
              "arch's stationary-bootstrap default plus, on the population sweep, a residual that "
              "survives matching both -- see mcsarch.*_bootstrap_matched / *_fully_matched).",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="mcs_elimination_rule_check.csv: raw_eq_eR.sum()")

    M = pd.read_csv(MONTHLY)
    emit("modelset.mean_mcs_annual", round(float(M.MCS_annual.mean()), 2), of=len(M),
         spec="mean |MCS| across the curated series at ANNUAL resolution (T ~ 25-36), the paper's "
              "headline resolution. Reported alongside the monthly means below because the binary "
              "non-identified COUNT is a lossy summary: it can hold at 11 of 12 while the average "
              "set size moves materially.",
         cites=["S13.2: set size versus sample size"],
         cross_check="mcs_monthly_resolution.csv: MCS_annual.mean()")
    emit("modelset.monthly_obs_ratio", round(float((M.T_monthly / M.T_annual).mean()), 1), of=len(M),
         spec="mean across series of (monthly evaluation observations / annual evaluation "
              "observations) -- how much more data the native-monthly re-run has to work with",
         cites=["S4.1 body: set size versus sample size"],
         cross_check="mcs_monthly_resolution.csv: round((T_monthly / T_annual).mean(), 1) -- the "
                     "unrounded mean is 11.42")
    emit("modelset.mean_mcs_monthly_b1", round(float(M.MCS_mon_b1.mean()), 2), of=len(M),
         spec="mean |MCS| on the NATIVE MONTHLY panels (T ~ 292-431, about 11.4x the annual "
              "observations) at bootstrap block=1 -- the short-block end, where the monthly data's "
              "effective sample is largest",
         cites=["S13.2: set size versus sample size"],
         cross_check="mcs_monthly_resolution.csv: MCS_mon_b1.mean()")
    emit("modelset.mean_mcs_monthly_b24", round(float(M.MCS_mon_b24.mean()), 2), of=len(M),
         spec="mean |MCS| on the native monthly panels at bootstrap block=24. Emitted BESIDE the "
              "block=1 value on purpose: mean |MCS| falls from 3.92 (annual) to 3.17 at short "
              "blocks but RISES to this value at long ones, because a long block cuts the number of "
              "independent blocks back down. Quoting only the falling end would overstate the "
              "sample-size reading.",
         cites=["S13.2: set size versus sample size"],
         cross_check="mcs_monthly_resolution.csv: MCS_mon_b24.mean()")
    print(f"\n  wrote {RES/'model_set_audit.csv'}")
    flush()


if __name__ == "__main__":
    main()
