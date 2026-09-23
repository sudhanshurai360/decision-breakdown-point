"""
flagship_fragility.py — give the FLAGSHIP (UI weekly initial claims, 1988-2025) its own k*/MCS.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "code" / "pipeline"))
import fragility as F
import mcs_test as MCS
import run_multiseries as RM
import null_budget as NB
from canonical_emit import emit, flush
RES = ROOT / "results"


def main():
    sb = pd.read_csv(RES / "extension_scoreboard_1988-2025.csv", index_col=0)
    if sb.isna().any().any():
        raise RuntimeError(
            f"extension_scoreboard_1988-2025.csv has NaN(s) in column(s) "
            f"{sb.columns[sb.isna().any()].tolist()} -- F.fragility() needs a complete panel; drop or "
            f"impute before calling it, following the conventions already used upstream: "
            f"score_multiseries.py drops rows with `.dropna(subset=['scale_abs', 'scale_sq'])`, and "
            f"run_multiseries.py keeps only complete years via `_complete_years()` (an np.isfinite "
            f"mask requiring every model to have a finite loss in that year)")
    models = list(sb.columns); years = list(sb.index)
    L = {m: sb[m].to_numpy(float) for m in models}
    w = np.ones(len(years))
    r = F.fragility(L, w, labels=years, n_boot=5000, seed=0)
    _stab = NB.stable_stability(L, w, n_boot=5000)
    _fragile_avg = bool((_stab["value"] < F.FRAGILE_STABILITY_THRESH)
                         or (r["reversal"] and r["k_over_T"] < F.FRAGILE_KSTAR_RATIO_THRESH))
    if r["n_binding_tied"] != 1:
        raise RuntimeError(
            f"n_binding_tied={r['n_binding_tied']} != 1: flagship.concentration is now one of several "
            f"defensible values (range [{r['concentration_lo']:.3f}, {r['concentration_hi']:.3f}]) -- "
            f"surface concentration_lo/concentration_hi explicitly (see social_choice.py's conc_lo/"
            f"conc_hi columns) before emitting a single point estimate for flagship.concentration")

    # MCS on the year x model loss matrix (absolute per-year MASE)
    Lmat = sb.to_numpy(float)
    surv10, p10 = MCS.mcs(Lmat, alpha=0.10, seed=0)
    surv25, p25 = MCS.mcs(Lmat, alpha=0.25, seed=0)
    mcs10 = [models[i] for i in surv10]; mcs25 = [models[i] for i in surv25]

    pooled_rank = sb.mean(0).sort_values()
    print("=== FLAGSHIP (UI initial claims, 1988-2025) — its own k*/MCS ===")
    print(f"  models: {models}")
    print(f"  pooled winner (1988-2025): {r['pooled_winner']}   per-period winner: {r['per_period_winner']}")
    print(f"  k* = {r['k_star']}  (k*/T = {r['k_over_T']:.3f})  -> the pooled winner flips if the {r['k_star']} year(s) "
          f"{r['responsible']} are deleted")
    print(f"  binding runner-up: {r['runner_up']}   winner_stability (20-seed avg) = {_stab['value']:.3f} "
          f"(seed=0 alone: {r['winner_stability']:.3f})   fragile={_fragile_avg}")
    print(f"  MODEL CONFIDENCE SET  alpha=0.10: {mcs10}  (|MCS|={len(mcs10)}, stop p={p10:.3f})")
    print(f"  MODEL CONFIDENCE SET  alpha=0.25: {mcs25}  (|MCS|={len(mcs25)}, stop p={p25:.3f})")
    print("  pooled MASE ranking: " + ", ".join(f"{m} {v:.3f}" for m, v in pooled_rank.items()))

    # window-inversion exhibit (short COVID-heavy vs long vs ex-COVID)
    print("\n=== window inversion (contrast_exhibit.csv) — the flagship's headline fragility ===")
    ce = pd.read_csv(RES / "contrast_exhibit.csv", index_col=0)
    for col in ce.columns:
        rk = ce[col].rank()
        print(f"  {col:26s}: winner={ce[col].idxmin():14s} | ar1 rank #{rk.get('ar1', float('nan')):g} ({ce[col].get('ar1', float('nan')):.3f})")

    out = pd.DataFrame([{
        "window": "1988-2025", "pooled_winner": r["pooled_winner"], "per_period_winner": r["per_period_winner"],
        "k_star": r["k_star"], "k_over_T": round(r["k_over_T"], 3), "responsible": ";".join(map(str, r["responsible"])),
        "winner_stability": round(_stab["value"], 3), "fragile": _fragile_avg,
        "winner_stability_seed0": round(r["winner_stability"], 3),
        "winner_stability_sd": round(_stab["sd"], 4),
        "concentration": round(r["concentration"], 3),
        "reversal": r["reversal"],
        "condorcet_winner": r["condorcet_winner"], "condorcet_status": r["condorcet_status"],
        "reversal_condorcet": r["reversal_condorcet"],
        "reversal_ambiguous": r["reversal_ambiguous"],
        "MCS_10": "|".join(mcs10), "MCS_10_size": len(mcs10), "MCS_25": "|".join(mcs25), "MCS_25_size": len(mcs25),
    }])
    _csv_out = RES / "flagship_fragility.csv"
    RM._atomic_to_csv(out, _csv_out)
    print(f"\nwrote {_csv_out}")

    # Every emit() from flagship.pooled_winner through flagship.mcs_25 is a direct readout of a
    # value main() already computed above; no new computation added. Most of those values were also
    # printed to stdout, but concentration, reversal, reversal_ambiguous and the three condorcet_*
    # fields reach the reader only through flagship_fragility.csv, written just above. The last four do
    # compute: flagship.ar1_rank_long_window reads ar1's position out of the already-computed
    # pooled_rank, and the three flagship.stability_null_* emits are fed by a single fresh
    # NB.stable_benchmark(_K, _T) Monte Carlo (called just below) that is neither computed nor
    # printed above.
    emit("flagship.pooled_winner", str(r["pooled_winner"]),
         spec="the pooled (score-rule) winner over the flagship's full 1988-2025 window",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: pooled_winner")
    emit("flagship.k_star", int(r["k_star"]), of=int(len(years)),
         rival=str(r["runner_up"]), n_binding_tied=int(r["n_binding_tied"]),
         spec="decision-level breakdown point -- fewest years whose deletion flips the flagship's "
              "pooled winner against its BINDING rival (the opponent that overtakes the champion "
              "fastest, named here as `rival`; see flagship.t_binding, which reports the realised "
              "statistic for the same pair). k_star is a property of a specific champion-rival pair, "
              "not of the champion alone; "
              "n_binding_tied flags cells where more than one rival ties for binding",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: k_star")
    emit("flagship.winner_stability", round(float(_stab["value"]), 3),
         sd=round(_stab["sd"], 4), n_seeds=_stab["n_seeds"], seed0=round(float(r["winner_stability"]), 3),
         spec="block-bootstrap winner-stability for the flagship's pooled winner, averaged over "
              "20 independent bootstrap seeds via NB.stable_stability (seed0= carries the single "
              "seed=0 draw for reference)",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: winner_stability")
    emit("flagship.concentration", round(float(r["concentration"]), 3),
         n_binding_tied=int(r["n_binding_tied"]),
         spec="share of the flagship's k*-margin against its binding rival contributed by the single "
              "most influential year -- the same statistic Table 1 reports per curated series (S4)",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: concentration")
    emit("flagship.responsible_years", ";".join(map(str, r["responsible"])),
         of=int(len(years)),
         spec="the year(s) whose deletion flips the flagship's pooled winner",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: responsible")
    emit("flagship.fragile", bool(_fragile_avg),
         spec=f"whether the flagship's pooled winner meets the paper's fixed fragile rule "
              f"(winner_stability<{F.FRAGILE_STABILITY_THRESH:.2f}, or reversal with "
              f"k*/T<{F.FRAGILE_KSTAR_RATIO_THRESH:.2f}) -- the same rule Table 1 "
              f"applies per curated series. Computed from the 20-seed-averaged winner_stability "
              f"(reversal/k_over_T are exact and seed-independent, so only the "
              f"stability<{F.FRAGILE_STABILITY_THRESH:.2f} half of this rule depends on the bootstrap)",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: fragile")
    emit("flagship.reversal", bool(r["reversal"]),
         spec="whether the flagship's pooled (score-rule) winner over 1988-2025 differs from its "
              "per-period (plurality) winner -- the same statistic Table 1 reports per curated "
              "series as its `Reversal` column",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: reversal")
    emit("flagship.reversal_ambiguous", bool(r["reversal_ambiguous"]),
         spec="whether flagship.reversal's verdict rests on an arbitrary per-period plurality "
              "tie-break (the per-period winner is one of >=2 models tied for that period) rather "
              "than a clean, unambiguous per-period winner -- the flagship-exhibit analogue of "
              "election.reversals_lo/election.reversals_hi's tie-break-robustness disclosure for "
              "the curated panel",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: reversal_ambiguous")
    emit("flagship.per_period_winner", str(r["per_period_winner"]),
         spec="the per-period (plurality) winner over the flagship's full 1988-2025 window -- "
              "compare against flagship.pooled_winner; flagship.reversal reports whether they differ",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: per_period_winner")
    emit("flagship.condorcet_winner", r["condorcet_winner"],
         spec="the CONDORCET winner over the flagship's full 1988-2025 window (the model that "
              "beats every other model in pairwise per-period majority) -- null if no Condorcet "
              "winner exists (a majority cycle or pairwise tie; see flagship.condorcet_status). "
              "Completes the same {score, plurality, Condorcet} triple as flagship.pooled_winner/"
              "flagship.per_period_winner",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: condorcet_winner")
    emit("flagship.condorcet_status", str(r["condorcet_status"]),
         spec="classification of the flagship's pairwise-majority tournament: 'winner' (a Condorcet "
              "winner exists), 'cycle' (a genuine intransitive majority cycle, no Condorcet winner), "
              "or 'tie' (no Condorcet winner and no cycle -- driven by pairwise ties)",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: condorcet_status")
    emit("flagship.reversal_condorcet", bool(r["reversal_condorcet"]),
         spec="whether the flagship's pooled (score-rule) winner over 1988-2025 differs from its "
              "Condorcet winner (undefined/False when no Condorcet winner exists -- see "
              "flagship.condorcet_status) -- the same statistic election.rev_condorcet reports "
              "per curated series",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: reversal_condorcet")
    emit("flagship.mcs_10", "|".join(mcs10), of=int(len(models)),
         spec="Hansen (2011) Model Confidence Set at alpha=0.10 on the flagship's 7-model year x "
              "model MASE matrix",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: MCS_10")
    emit("flagship.mcs_25", "|".join(mcs25), of=int(len(models)),
         spec="Hansen (2011) Model Confidence Set at alpha=0.25 on the flagship's 7-model year x "
              "model MASE matrix",
         cites=["S3.6 body: flagship exhibit"],
         cross_check="flagship_fragility.csv: MCS_25")
    emit("flagship.ar1_rank_long_window", int(pooled_rank.index.get_loc("ar1") + 1),
         of=int(len(pooled_rank)),
         spec="AR1's rank by pooled MASE over the FULL 1988-2025 window, all models",
         cites=["S3.6 body: window-vs-basis attribution -- long-window leg"],
         cross_check="extension_scoreboard_1988-2025.csv: rank of ar1 in full-sample mean")
    _K, _T = len(models), len(years)
    _nbd = NB.stable_benchmark(_K, _T)
    emit("flagship.stability_null_median", round(float(_nbd["median"]), 3), n_models=_K, n_periods=_T,
         n_eff=_nbd["n_eff"],
         spec=f"median winner_stability under EXCHANGEABLE models (K={_K}, T={_T}, {_nbd['n_eff']} "
              f"panels across {_nbd['n_seeds']} independent seeds): what the statistic returns when "
              f"the winner is arbitrary by construction",
         cites=["S3.6 body: flagship exhibit"], cross_check=None)
    emit("flagship.stability_null_p05", round(float(_nbd["p05"]), 3), n_models=_K, n_periods=_T,
         band=[round(_nbd["p05_lo"], 3), round(_nbd["p05_hi"], 3)], sd=round(_nbd["p05_sd"], 4),
         n_eff=_nbd["n_eff"],
         spec=f"5th percentile of winner_stability under exchangeable models (K={_K}, T={_T}) at the "
              f"flagship's own dimensions; a value below this is unusually fragile RELATIVE TO CHANCE",
         cites=["S3.6 body: flagship exhibit"], cross_check=None)
    emit("flagship.stability_null_p95", round(float(_nbd["p95"]), 3), n_models=_K, n_periods=_T,
         band=[round(_nbd["p95_lo"], 3), round(_nbd["p95_hi"], 3)], sd=round(_nbd["p95_sd"], 4),
         n_eff=_nbd["n_eff"],
         spec=f"95th percentile of winner_stability under exchangeable models (K={_K}, T={_T}) at "
              f"the flagship's own dimensions -- the upper edge of the same band as "
              f"flagship.stability_null_p05, reported so both tails of the null are on record. The "
              f"sd here is the p95 edge's OWN seed-to-seed spread, not to be conflated with "
              f"flagship.stability_null_p05's sd",
         cites=["S3.6 body: flagship exhibit"], cross_check=None)
    flush()


if __name__ == "__main__":
    main()
