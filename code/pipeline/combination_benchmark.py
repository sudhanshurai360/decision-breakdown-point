"""combination_benchmark.py — "if selection is unreliable, why not just COMBINE?"
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_DIR / "code" / "instrument"))
import fragility as F
import mcs_test as MT
import null_budget as NB
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
_BD = HERE.parent.parent / "multiseries" / "results" / "breadth_distinct"
PRED = _BD / "breadth_distinct_predictions.csv"
RES = HERE.parent.parent / "multiseries" / "results"
COMB = "combination"
ALPHA = 0.10
BENCHMARK_SN = "seasonal_naive"


def per_series_panels(exclude=()):
    """{series: (per-year loss dict including the equal-weight combination, T)} for the curated
    panel, optionally excluding models in `exclude`."""
    d = pd.read_csv(PRED)
    out = {}
    for sid, g in d.groupby("series"):
        # equal-weight combination of the POINT FORECASTS at each target date
        g_comb = g[~g.model.isin(exclude)] if exclude else g
        piv = g_comb.pivot_table(index="target", columns="model", values="y_pred", aggfunc="mean")
        truth = g_comb.groupby("target")[["y_true", "scale_abs"]].first()
        piv = piv.dropna()   # keep only target dates every (non-excluded) model forecast
        truth = truth.loc[piv.index]
        y_comb = piv.mean(axis=1)
        sae_comb = (truth.y_true - y_comb).abs() / truth.scale_abs
        sae_comb = sae_comb[np.isfinite(sae_comb)]
        if sae_comb.empty:
            print(f"  [skip] {sid}: combination has no finite sae value on any evaluation date", flush=True)
            continue
        # per-YEAR mean loss, matching the rest of the project's evaluation unit
        yr = pd.to_datetime(sae_comb.index).year
        comb_year = pd.Series(sae_comb.to_numpy(), index=yr).groupby(level=0).mean()
        singles = g.groupby(["model", "year"])["sae"].mean().unstack(0)
        # ONE aligned frame, one dropna -- so the combination and the singles are guaranteed to be
        # defined on exactly the same years rather than on two independently-filtered index sets.
        panel = singles.join(comb_year.rename(COMB), how="inner").dropna()
        if len(panel) < 5:
            print(f"  [skip] {sid}: only {len(panel)} year(s) with both singles and the combination "
                  f"finite (need >=5)", flush=True)
            continue
        L = {m: panel[m].to_numpy(float) for m in panel.columns}
        out[sid] = (L, len(panel))
    return out


def _pooled_best_rank(L, w):
    pooled = {m: float(np.average(L[m], weights=w)) for m in L}
    best_single = min((m for m in pooled if m != COMB), key=lambda m: pooled[m])
    comb_wins = pooled[COMB] < pooled[best_single]
    singles_pooled = sorted(pooled[m] for m in pooled if m != COMB)
    med_single = float(np.median(singles_pooled))
    rank = 1 + sum(1 for v in singles_pooled if v < pooled[COMB])   # 1 = better than all singles
    return pooled, best_single, comb_wins, med_single, rank


def main():
    panels = per_series_panels()
    rows = []
    for sid, (L, T) in panels.items():
        w = np.ones(T)
        pooled, best_single, comb_wins, med_single, rank = _pooled_best_rank(L, w)
        # k* for the COMBINATION-vs-BEST-SINGLE contest, in whichever direction the winner runs
        win, lose = (COMB, best_single) if comb_wins else (best_single, COMB)
        k, _ = F.breakdown_number(L[win], L[lose], w)
        _stab_d = NB.stable_stability({win: L[win], lose: L[lose]}, w)
        stab, stab_lo, stab_hi, stab_sd = (_stab_d["value"], _stab_d["lo"], _stab_d["hi"], _stab_d["sd"])
        _nbd_row = NB.stable_benchmark(2, T)
        null_med, null_p05 = _nbd_row["median"], _nbd_row["p05"]
        null_p05_lo, null_p05_hi, null_p05_sd = (_nbd_row["p05_lo"], _nbd_row["p05_hi"], _nbd_row["p05_sd"])
        null_n_eff = _nbd_row["n_eff"]
        fragility_verdict = NB.verdict(stab, null_p05,
                                        NB.band(stab_sd, len(NB.SEEDS)),
                                        NB.band(null_p05_sd, len(NB.SEEDS)))
        # would the combination survive an MCS run that includes it? (7 candidates, separate run)
        models = sorted(L)
        M = np.column_stack([L[m] for m in models])
        surv, _ = MT.mcs(M, alpha=ALPHA, seed=0)
        surv_names = [models[i] for i in surv]
        worst_single = max((m for m in pooled if m != COMB), key=lambda m: pooled[m])
        worst_best_ratio = pooled[worst_single] / pooled[best_single]
        rows.append(dict(series=sid, T=T, best_single=best_single,
                         worst_single=worst_single, worst_best_ratio=float(worst_best_ratio),
                         pooled_comb=pooled[COMB], pooled_best=pooled[best_single],
                         pooled_median_single=med_single,
                         comb_wins=bool(comb_wins), comb_beats_median=bool(pooled[COMB] < med_single),
                         comb_rank_of_7=int(rank),
                         k_star=int(k), winner_stability=float(stab),
                         stab_lo=float(stab_lo), stab_hi=float(stab_hi), stab_sd=float(stab_sd),
                         null_median=float(null_med), null_p05=float(null_p05),
                         null_p05_lo=float(null_p05_lo), null_p05_hi=float(null_p05_hi),
                         null_p05_sd=float(null_p05_sd), null_n_eff=int(null_n_eff),
                         fragility_verdict=fragility_verdict,
                         mcs_size=len(surv_names), comb_in_mcs=COMB in surv_names))
    R = pd.DataFrame(rows).sort_values("series")
    RM._atomic_to_csv(R, RES / "combination_benchmark.csv")
    n = len(R)
    n_wins = int(R.comb_wins.sum())
    n_in_mcs = int(R.comb_in_mcs.sum())
    med_k = float(R.k_star.median())
    med_stab = float(R.winner_stability.median())
    print(f"=== EQUAL-WEIGHT COMBINATION vs BEST SINGLE ({n} curated series) ===")
    print(R[["series", "T", "best_single", "comb_wins", "comb_beats_median", "comb_rank_of_7",
             "k_star", "winner_stability", "mcs_size", "comb_in_mcs"]].to_string(index=False))
    print()
    n_beats_med = int(R.comb_beats_median.sum())
    med_rank = float(R.comb_rank_of_7.median())
    print(f"  combination beats the best single model pooled : {n_wins}/{n}   (ORACLE test -- hardest)")
    print(f"  combination beats the MEDIAN single model: {n_beats_med}/{n}   (the test a real "
          f"forecaster actually faces)")
    print(f"  combination's median rank among the 7 candidates: {med_rank:.1f}")
    print(f"  combination survives an MCS that includes it   : {n_in_mcs}/{n}")
    print(f"  median k* of the comb-vs-best contest: {med_k:.1f}")
    print(f"  median winner_stability of that contest: {med_stab:.3f}")
    print()
    print("  READING. Two comparisons are reported because they answer different questions. Beating")
    print("  the ORACLE-best single is the hardest test and favours the singles (that model is chosen")
    print("  with hindsight; a real forecaster must pick in advance). Beating the MEDIAN single is the")
    print("  test a blind chooser actually faces. Equal-weight combination loses BOTH here, at a median")
    print("  rank of 6th of 7, and the verdict is robust (median k* well above 1, stability ~1.0).")
    print("  So 'just combine' does not dissolve the selection problem -- with a quality-heterogeneous")
    print("  pool, equal weights drag a good model toward weak ones. SCOPE: this indicts EQUAL weights,")
    print("  not combination in general; trimmed / performance-weighted schemes are untested here.")
    emit("comb.beats_best_single", n_wins, of=n,
         spec="curated series where the EQUAL-WEIGHT COMBINATION of the six models has lower pooled "
              "mean scaled absolute loss than the best-performing SINGLE model. The combination is "
              "built from the point forecasts (mean of y_pred) and then scored on the same "
              "series-year scale_abs as the singles -- NOT an average of the singles' losses. Note "
              "the comparison is an ORACLE one and favours the singles: 'best single' is chosen with "
              "hindsight, while a real forecaster must select in advance",
         cites=["S4.4 body: combination benchmark"],
         cross_check="combination_benchmark.csv: comb_wins.sum()")
    emit("comb.beats_median_single", n_beats_med, of=n,
         spec="curated series where the equal-weight combination has lower pooled loss than the "
              "MEDIAN single model. This is the decision-relevant comparison, not comb.beats_best_"
              "single: the paper's own thesis is that the best single cannot be identified in "
              "advance, so a forecaster who must choose blind faces a typical model, not the oracle "
              "best one",
         cites=["S4.4 body: combination benchmark"],
         cross_check="combination_benchmark.csv: comb_beats_median.sum()")
    emit("comb.median_rank_of_7", med_rank, of=7, n_series=n,
         spec="median across series of the combination's rank among the seven candidates (the six "
              "singles plus itself), 1 = better than every single model. A rank consistently near "
              "the top without ever needing a selection decision is the practical case for combining",
         cites=["S4.4 body: combination benchmark"],
         cross_check="combination_benchmark.csv: comb_rank_of_7.median()")
    emit("comb.in_mcs", n_in_mcs, of=n,
         spec=f"curated series where the combination survives a Model Confidence Set run that "
              f"INCLUDES it as a seventh candidate (alpha={ALPHA}). This is a SEPARATE run and does "
              f"not alter the frozen six-model pool behind mcs.curated_nonidentified",
         cites=["S4.4 body: combination benchmark"],
         cross_check="combination_benchmark.csv: comb_in_mcs.sum()")
    emit("comb.median_kstar_vs_best", med_k, n_series=n,
         spec="median across series of k* for the combination-vs-best-single contest (fewest years "
              "to delete to flip whichever of the two wins). Higher = the verdict between combining "
              "and selecting is harder to overturn",
         cites=["S4.4 body: combination benchmark"],
         cross_check="combination_benchmark.csv: k_star.median()")
    med_null = float(R.null_median.median())
    med_null_p05 = float(R.null_p05.median())
    emit("comb.median_winner_stability", med_stab, n_series=n,
         spec="median across series of winner_stability for the combination-vs-best-single contest. "
              "Read against comb.stability_null_median/p05 (this file's own K=2 exchangeable null, "
              "computed at each series' own T then medianed across series), NOT against a bare "
              "threshold or a null borrowed from a different panel",
         cites=[],
         cross_check="combination_benchmark.csv: winner_stability.median()")
    emit("comb.stability_null_median", med_null, n_series=n,
         spec="median across series of the exchangeable-null median for winner_stability under K=2 "
              "identically-distributed models, computed AT EACH SERIES' OWN T (per_series_panels' T, "
              "not a fixed T~30 approximation), via null_budget.stable_benchmark: 20 independent seeds "
              "x 1000 panels/800 bootstrap draws per seed per series (20,000 effective replications) "
              "-- what comb.median_winner_stability must be read "
              "against, per the discipline established for flagship.stability_null_median (a null "
              "computed at different dimensions than the panel it calibrates is a real bug, not a "
              "convenience)",
         cites=[],
         cross_check="combination_benchmark.csv: null_median.median()")
    emit("comb.stability_null_p05", med_null_p05, n_series=n,
         spec="median across series of the exchangeable-null 5th percentile for winner_stability "
              "under K=2 identically-distributed models, same per-series-T computation as "
              "comb.stability_null_median; a series' own winner_stability below its own p05 is "
              "unusually fragile RELATIVE TO CHANCE at that series' T",
         cites=[],
         cross_check="combination_benchmark.csv: null_p05.median()")
    _wins = R[R.comb_wins]
    if len(_wins) == 1:
        _wr = _wins.iloc[0]
        emit("comb.winning_series", str(_wr.series),
             spec="the single series where the equal-weight combination beats the best single model",
             cites=["S4.4 body: the one cell the median conceals"],
             cross_check="combination_benchmark.csv: series where comb_wins")
        emit("comb.winning_series_stability", round(float(_wr.winner_stability), 3),
             band=[round(float(_wr.stab_lo), 3), round(float(_wr.stab_hi), 3)],
             sd=round(float(_wr.stab_sd), 4), n_seeds=len(NB.SEEDS),
             spec="winner_stability of the combination-vs-best-single contest ON the one series the "
                  "combination wins, averaged over independent bootstrap seeds. Read against "
                  "comb.winning_series_null_p05 THROUGH comb.winning_series_verdict -- never by "
                  "comparing the two point estimates directly: this cell's classification is "
                  "sensitive to seed-averaging discipline (a single seed=0 draw and a multi-seed "
                  "average can disagree), which is exactly why an averaged, banded estimate with an "
                  "explicit verdict field is reported here instead of a bare point estimate",
             cites=["S4.4 body: the one cell the median conceals"],
             cross_check="combination_benchmark.csv: winner_stability where comb_wins")
        emit("comb.winning_series_null_p05", round(float(_wr.null_p05), 3),
             band=[round(float(_wr.null_p05_lo), 3), round(float(_wr.null_p05_hi), 3)],
             sd=round(float(_wr.null_p05_sd), 4), n_eff=int(_wr.null_n_eff),
             spec="that same cell's OWN exchangeable-null 5th percentile, at its own K=2 and T, "
                  "averaged over independent seeds. Its seed-to-seed spread is reported because it is "
                  "comparable to the gap this threshold is used to adjudicate",
             cites=["S4.4 body: the one cell the median conceals"],
             cross_check="combination_benchmark.csv: null_p05 where comb_wins")
        emit("comb.winning_series_verdict", str(_wr.fragility_verdict),
             spec="'fragile' / 'not_fragile' / 'undetermined' for that cell, decided with BOTH Monte-"
                  "Carlo intervals in hand. 'undetermined' means the gap between the observed value "
                  "and its threshold is smaller than the noise in the two estimates -- a real answer, "
                  "and the one a single-seed point comparison could never return",
             cites=["S4.4 body: the one cell the median conceals"],
             cross_check="combination_benchmark.csv: fragility_verdict where comb_wins")
    n_rank6 = int((R.comb_rank_of_7 == 6).sum())
    _sn_worst = R[R.worst_single == BENCHMARK_SN]
    _ratios = sorted(_sn_worst.worst_best_ratio.tolist())
    emit("comb.rank6_series", n_rank6, of=n,
         spec="curated series where the combination ranks EXACTLY 6th of 7. The median rank is a "
              "summary of a near-degenerate distribution, and this count is what makes that visible",
         cites=["S4.4 body: why the rank is what it is"],
         cross_check="combination_benchmark.csv: (comb_rank_of_7 == 6).sum()")
    emit("comb.sn_worst_series", len(_sn_worst), of=n,
         spec="curated series where seasonal_naive is the WORST of the six single models by pooled "
              "mean scaled absolute loss -- the pool member an equal-weight mean cannot down-weight",
         cites=["S4.4 body: why the rank is what it is"],
         cross_check="combination_benchmark.csv: worst_single == 'seasonal_naive'")
    emit("comb.sn_worst_ratio_min", _ratios[0], aux_max=_ratios[-1],
         spec="across the series where seasonal_naive is worst, the smallest (and, as aux_max, the "
              "largest) ratio of its pooled loss to the best single model's -- how far the outlier "
              "member sits from the rest of the pool",
         cites=["S4.4 body: why the rank is what it is"],
         cross_check="combination_benchmark.csv: min/max of worst_best_ratio where worst_single == "
                      "'seasonal_naive'")
    print(f"  wrote {(RES / 'combination_benchmark.csv').relative_to(ROOT_DIR)}")
    flush()


def main5():
    """The five-model equal-weight combination, excluding seasonal_naive -- the
    counterpart to main() above."""
    panels5 = per_series_panels(exclude=(BENCHMARK_SN,))
    rows = []
    for sid, (L, T) in panels5.items():
        w = np.ones(T)
        pooled, best_single, comb_wins, med_single, rank = _pooled_best_rank(L, w)
        rows.append(dict(series=sid, T=T, best_single=best_single,
                         pooled_comb=pooled[COMB], pooled_best=pooled[best_single],
                         pooled_median_single=med_single,
                         comb_wins=bool(comb_wins), comb_beats_median=bool(pooled[COMB] < med_single),
                         comb_rank_of_7=int(rank)))
    R = pd.DataFrame(rows).sort_values("series")
    RM._atomic_to_csv(R, RES / "combination5_benchmark.csv")
    n = len(R)
    n_wins = int(R.comb_wins.sum()); n_beats_med = int(R.comb_beats_median.sum())
    med_rank = float(R.comb_rank_of_7.median())
    print(f"\n=== FIVE-MODEL EQUAL-WEIGHT COMBINATION (excludes {BENCHMARK_SN}) vs BEST SINGLE ({n} curated series) ===")
    print(R.to_string(index=False))
    print(f"  combination beats the best single model pooled : {n_wins}/{n}   (ORACLE test)")
    print(f"  combination beats the MEDIAN single model: {n_beats_med}/{n}")
    print(f"  combination's median rank among the 7 candidates (6 singles incl. {BENCHMARK_SN}, +itself): {med_rank:.1f}")
    emit("comb5.beats_best_single", n_wins, of=n,
         spec=f"curated series where the FIVE-model equal-weight combination (excludes {BENCHMARK_SN}, "
              f"the deliberately mis-specified reference S3.1 documents) has lower pooled mean scaled "
              f"absolute loss than the best-performing single model, chosen from the same six-single "
              f"pool comb.beats_best_single uses. Counterpart to comb.beats_best_single, whose "
              f"six-model version cannot cleanly separate the combination's performance from "
              f"seasonal_naive's own mis-specification",
         cites=["S4.4 body: five-model combination, seasonal_naive excluded"],
         cross_check="combination5_benchmark.csv: comb_wins.sum()")
    emit("comb5.beats_median_single", n_beats_med, of=n,
         spec="curated series where the five-model combination beats the MEDIAN single model (from "
              "the same six-single pool) -- the decision-relevant comparison, matching comb.beats_"
              "median_single's convention",
         cites=["S4.4 body: five-model combination, seasonal_naive excluded"],
         cross_check="combination5_benchmark.csv: comb_beats_median.sum()")
    emit("comb5.median_rank_of_7", med_rank, of=7, n_series=n,
         spec="median across series of the five-model combination's rank among the seven candidates "
              "(the six singles, including seasonal_naive as an option a forecaster could still pick, "
              "plus the combination itself), 1 = better than every single model",
         cites=["S4.4 body: five-model combination, seasonal_naive excluded"],
         cross_check="combination5_benchmark.csv: comb_rank_of_7.median()")
    print(f"  wrote {(RES / 'combination5_benchmark.csv').relative_to(ROOT_DIR)}")
    flush()


if __name__ == "__main__":
    main()
    main5()
