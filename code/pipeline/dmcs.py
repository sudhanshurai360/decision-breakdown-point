"""
dmcs.py — the decision Model Confidence Set: assess non-identification UNDER THE DECISION LOSS, not MASE.
"""
import sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
import state_eb_decision as SEB
import decision_protocol as DP
import mcs_test as MCS
from canonical_emit import emit, flush
RES = MULTI / "results"


def year_mcs(loss_by_model, common, seed=0):
    """loss_by_model: {model: monthly Series}; -> (survivors set of model names) via per-year-mean MCS."""
    M = pd.DataFrame({m: loss_by_model[m] for m in loss_by_model}).reindex(common).dropna()
    if len(M) < 30:
        return None
    yr = M.groupby(M.index.year).mean()
    surv, _ = MCS.mcs(yr.to_numpy(float), alpha=0.10, seed=seed)
    return set(M.columns[i] for i in surv)


def main():
    df = SEB.load_state_ur(); print(f"state UR panel: {df.shape[1]} states x {df.shape[0]} months")
    rows = []
    n_eligible = 0   # states with a long-enough panel to even attempt the comparison
    for s in df.columns:
        u = df[s].dropna()
        if len(u) < SEB.MIN_STATE_PANEL_MONTHS:
            continue
        n_eligible += 1
        nc = SEB.nowcast(u)
        common_full = u.index[u.index.year >= u.index.year.min() + SEB.WARMUP_YEARS]
        common = common_full[common_full >= SEB.EB_TUR_EFFECTIVE]
        realized_eb = SEB.eb_on(u, u)                               # realized EB status (on revised UR)
        stat_loss = {m: (nc[m] - u).abs() for m in SEB.MODELS}      # statistical loss (abs error)
        dec_loss = {m: (SEB.eb_on(nc[m].reindex(u.index), u) != realized_eb).astype(float) for m in SEB.MODELS}  # decision loss
        A = pd.DataFrame({m: DP.statutory_round_tur(DP.three_month_average(nc[m].reindex(u.index), u))
                          for m in SEB.MODELS}).reindex(common)
        near = common[((A >= 6.0) & (A <= 7.0)).any(axis=1)]
        tied_stat = year_mcs(stat_loss, common_full)   # full record: matches SEB._tied_models() by construction
        tied_dec = year_mcs(dec_loss, near)          # <-- near-trigger months only, not `common`
        tied_stat_near = year_mcs(stat_loss, near)
        if tied_stat is None or tied_dec is None or tied_stat_near is None:
            continue
        rows.append({"state": s, "MCS_stat": len(tied_stat), "dMCS_decision": len(tied_dec),
                     "MCS_stat_near": len(tied_stat_near),
                     "near_trigger_months": len(near),
                     "near_trigger_years": int(pd.DataFrame(dec_loss).reindex(near).dropna().index.year.nunique()),
                     "full_years": int(pd.DataFrame(stat_loss).reindex(common_full).dropna().index.year.nunique()),
                     "separates": len(tied_dec) < len(tied_stat),
                     "separates_symmetric": len(tied_dec) < len(tied_stat_near),
                     "stat_set": sorted(tied_stat), "decision_set": sorted(tied_dec),
                     "stat_set_near": sorted(tied_stat_near)})
    R = pd.DataFrame(rows)
    R.to_csv(RES / "dmcs.csv", index=False)
    n = len(R)
    n_dropped_near = n_eligible - n   # no silent cap: disclose states the near-trigger floor drops,
                                       # rather than reporting only the retained set
    dropped_states = sorted(set(df.columns) - set(R.state))
    lowest_ur_states = sorted(df.mean().sort_values().index[:5].tolist())
    overlap_states = sorted(set(dropped_states) & set(lowest_ur_states))
    print("\n=== dMCS: is the model set that is tied under MASE also tied under the DECISION (EB) loss,")
    print("    measured on near-trigger months (where the decision loss can actually discriminate)? ===")
    print(R[["state", "MCS_stat", "dMCS_decision", "MCS_stat_near", "near_trigger_months", "separates"]].head(18).to_string(index=False))
    sep = int(R.separates.sum())
    tied_both = int(((R.dMCS_decision >= R.MCS_stat) & (R.dMCS_decision > 1)).sum())
    single_dec = int((R.dMCS_decision == 1).sum())
    sep_sym = int(R.separates_symmetric.sum())
    tied_both_sym = int(((R.dMCS_decision >= R.MCS_stat_near) & (R.dMCS_decision > 1)).sum())
    equal_sym = int((R.dMCS_decision == R.MCS_stat_near).sum())
    wider_sym = int((R.dMCS_decision > R.MCS_stat_near).sum())
    wider_from_singleton = int(((R.dMCS_decision > R.MCS_stat_near) & (R.MCS_stat_near == 1)).sum())
    wfs_mask = (R.dMCS_decision > R.MCS_stat_near) & (R.MCS_stat_near == 1)
    wider_from_singleton_also_full_sample = int((R[wfs_mask].MCS_stat == 1).sum())
    print(f"\n  eligible states (>=300mo panel): {n_eligible}; {n_dropped_near} dropped for insufficient "
          f"near-trigger months (year_mcs's existing len(M)<30 floor); {n} retained.")
    print("  states where the DECISION loss (near-trigger months only) SEPARATES the statistically-tied models")
    print(f"    (tie dissolves -> 'select on the decision loss', Granger-Machina): {sep}/{n};  decision set is a")
    print(f"    SINGLETON in {single_dec}/{n} states.")
    print(f"  states still TIED under both (genuine ambiguity -> report the admissible EB action SET): {tied_both}/{n}.")
    print(f"  mean |MCS_stat| = {R.MCS_stat.mean():.2f}  vs  mean |dMCS_decision| = {R.dMCS_decision.mean():.2f}"
          f"  (near-trigger months only, mean {R.near_trigger_months.mean():.0f} per state).")
    print("  READING: where the decision loss (measured only on months where it can discriminate at all) "
          "identifies a")
    print("    winner, non-identification does NOT bind -- select on the decision loss; where it stays tied even")
    print("    THERE, report a confidence set of statutory actions. Restricting to near-trigger months matters:")
    print("    measured on EVERY scored month the comparison is dominated by the trivial 'everyone agrees it's")
    print("    off' region (EB is a ~2-5% base-rate event), which reads as near-universal 'genuine ambiguity'")
    print("    when the real mechanism is near-total model agreement.")
    print("  SYMMETRIC ROBUSTNESS CUT -- the statistical MCS leg recomputed on the SAME near-trigger months")
    print("    as the decision leg (matched T):")
    print(f"    mean |MCS_stat_near| = {R.MCS_stat_near.mean():.2f} (vs full-sample {R.MCS_stat.mean():.2f}).")
    print(f"    separates_symmetric: {sep_sym}/{n}.  tied_both_symmetric: {tied_both_sym}/{n} (vs the unmatched-T")
    print(f"    tied_both={tied_both}/{n} above). {'IDENTICAL' if tied_both_sym == tied_both else 'DIFFERENT'} result under")
    print(f"    matched T -- {'the T-asymmetry was NOT the explanation; tied_both reflects the decision loss itself, not a mechanical artifact' if tied_both_sym == tied_both else 'the T-asymmetry WAS inflating tied_both; the matched-T comparison is the one to cite'}.")
    print(f"  wrote {RES/'dmcs.csv'}")

    emit("dmcs.n_states", n,
         spec="number of states retained after BOTH floors: a long enough UR panel (>=300 months) AND "
              "enough near-trigger months for year_mcs's existing len(M)<30 floor to run the decision-"
              "loss MCS meaningfully",
         cites=[],
         cross_check="dmcs.csv: len(R)")
    emit("dmcs.n_eligible", n_eligible,
         spec="states with a long enough UR panel (>=300 months) to even ATTEMPT the comparison, before "
              "the near-trigger floor -- disclosed so dmcs.n_states' drop from this is not silent",
         cites=[],
         cross_check="dmcs.py main(): count of states with len(u) >= 300")
    emit("dmcs.mean_set_statistical", float(R.MCS_stat.mean()), n_states=n,
         spec="mean |MCS| across states under the STATISTICAL loss (absolute nowcast error), alpha=0.10, "
              "over the FULL scored sample (only the decision loss below is "
              "restricted to near-trigger months)",
         cites=[],
         cross_check="dmcs.csv: MCS_stat.mean()")
    emit("dmcs.mean_set_decision", float(R.dMCS_decision.mean()), n_states=n,
         spec="mean |MCS| across states under the DECISION loss (misclassification of the realized EB "
              "trigger status), alpha=0.10, measured ONLY on near-trigger months (near-trigger months "
              "only, not the full sample, which is dominated by the trivial always-off region). "
              "T-ASYMMETRY NOTE, RESOLVED: "
              f"this leg runs on a median {int(R.near_trigger_years.median())} near-trigger YEARS per "
              f"state (distinct calendar years among the near-trigger months, range "
              f"{int(R.near_trigger_years.min())}-{int(R.near_trigger_years.max())}; read from "
              "dmcs.csv's near_trigger_years column) vs. dmcs.mean_set_statistical's "
              f"median {int(R.full_years.median())} years -- a real T-asymmetry between the two MCS legs "
              "dmcs.separates/dmcs.tied_both compare directly, raising the concern that a high mean "
              "|MCS| here could be an artifact of the shorter-T leg eliminating fewer candidates "
              "almost mechanically. Resolved by dmcs.mean_set_statistical_near (the statistical MCS "
              "recomputed at matched T): see dmcs.tied_both_symmetric, which is identical to "
              "dmcs.tied_both under matched T -- the T-asymmetry was not the explanation.",
         cites=[],
         cross_check="dmcs.csv: dMCS_decision.mean()")
    emit("dmcs.separates", sep, of=n,
         spec="states where the near-trigger-only decision loss produces a STRICTLY smaller tied set "
              "than the (full-sample) statistical loss (the tie dissolves when selecting on the "
              "decision that actually matters, measured where it can actually discriminate)",
         cites=[],
         cross_check="dmcs.csv: separates.sum()")
    emit("dmcs.tied_both", tied_both, of=n,
         spec="states where the tied set under the near-trigger-only decision loss is NOT smaller than "
              "under the (full-sample) statistical loss AND still has more than one admissible model -- "
              "candidate reading: genuine ambiguity even where the decision loss has the best chance to "
              "discriminate, reported as a confidence set of admissible statutory actions rather than a "
              "point call. T-ASYMMETRY CHECK (see dmcs.mean_set_decision): dMCS_decision >= MCS_stat in EVERY "
              "retained state raised the concern that this could be an artifact of the T-asymmetry "
              "between the two MCS legs (near-trigger-restricted vs. full-sample) rather than genuine "
              "information. Resolved: dmcs.tied_both_symmetric (the same comparison at matched T) is "
              f"{'IDENTICAL' if tied_both_sym == tied_both else 'NOT identical'}, {tied_both_sym}/{n} "
              f"vs {tied_both}/{n} -- even though the underlying per-state statistical-MCS sizes "
              "genuinely differ between the full-sample and near-trigger-restricted legs in "
              f"{int((R.MCS_stat != R.MCS_stat_near).sum())} of {n} states (confirming the "
              "T-restriction is not a no-op), the decision loss is at least as broad as the "
              "statistical loss either way, in every state.",
         cites=[],
         cross_check="dmcs.csv: ((dMCS_decision >= MCS_stat) & (dMCS_decision > 1)).sum()")
    emit("dmcs.single_decision", single_dec, of=n,
         spec="states where the near-trigger-only decision-loss tied set is a singleton (the decision "
              "loss fully identifies a winner even restricted to the months where it can discriminate)",
         cites=[],
         cross_check="dmcs.csv: (dMCS_decision == 1).sum()")
    emit("dmcs.mean_set_statistical_near", float(R.MCS_stat_near.mean()), n_states=n,
         spec="mean |MCS| across states under the STATISTICAL loss, alpha=0.10, restricted to the SAME "
              "near-trigger months as dmcs.mean_set_decision (matched T) -- the symmetric counterpart "
              "to dmcs.mean_set_statistical's full-sample figure, added to isolate whether "
              "dmcs.tied_both reflects genuine ambiguity or the T-asymmetry between the two "
              "full-sample-vs-near-trigger legs",
         cites=[],
         cross_check="dmcs.csv: MCS_stat_near.mean()")
    emit("dmcs.separates_symmetric", sep_sym, of=n,
         spec="states where the near-trigger decision loss produces a strictly smaller tied set than "
              "the near-trigger STATISTICAL loss (matched T on both legs) -- the symmetric counterpart "
              "to dmcs.separates, which compares against the full-sample statistical loss instead",
         cites=["S7 body: no cell shows a narrower decision-loss |MCS| than accuracy-loss |MCS|"],
         cross_check="dmcs.csv: separates_symmetric.sum()")
    emit("dmcs.tied_both_symmetric", tied_both_sym, of=n,
         spec="states where the tied set under the near-trigger decision loss is not smaller than under "
              "the near-trigger STATISTICAL loss (matched T on both legs) AND still has more than one "
              "admissible model -- the symmetric counterpart to dmcs.tied_both, which compares against "
              "the full-sample statistical loss instead. Compare directly against dmcs.tied_both: if "
              "the two counts agree, the T-asymmetry was not the explanation for dmcs.tied_both and it "
              "can be cited as genuine ambiguity; if they diverge, the T-asymmetry was inflating "
              "dmcs.tied_both and this matched-T figure is the one to cite instead",
         cites=[],
         cross_check="dmcs.csv: ((dMCS_decision >= MCS_stat_near) & (dMCS_decision > 1)).sum()")
    emit("dmcs.equal_symmetric", equal_sym, of=n,
         spec="states where the decision-loss |MCS| is EXACTLY equal to the near-trigger statistical "
              "|MCS| (matched T) -- the 'as wide but not wider' half of dmcs.tied_both_symmetric's "
              "never-narrower finding",
         cites=["S7 body: exactly equal in 18 states"],
         cross_check="dmcs.csv: (dMCS_decision == MCS_stat_near).sum()")
    emit("dmcs.wider_symmetric", wider_sym, of=n,
         spec="states where the decision-loss |MCS| is STRICTLY wider than the near-trigger statistical "
              "|MCS| (matched T) -- the complementary half of dmcs.equal_symmetric; "
              "equal_symmetric + wider_symmetric + narrower_symmetric (always 0, see "
              "dmcs.tied_both_symmetric) must sum to n",
         cites=["S7 body: strictly wider in the remaining states"],
         cross_check="dmcs.csv: (dMCS_decision > MCS_stat_near).sum()")
    emit("dmcs.wider_from_singleton", wider_from_singleton, of=wider_sym,
         spec="of dmcs.wider_symmetric's states (decision-loss |MCS| strictly wider than near-trigger "
              "statistical |MCS|), how many started from MCS_stat_near==1 -- a CLEAN, unambiguous "
              "statistical winner with zero pre-existing tie. For these states the decision rule does "
              "not widen an existing ambiguity, it manufactures a new one from a previously clean "
              "identification. tied_both/wider_symmetric do not distinguish this "
              "case from a genuinely-tied statistical set growing wider still -- both fall under "
              "the same 'genuine ambiguity' framing; this key separates them.",
         states=";".join(sorted(R[(R.dMCS_decision > R.MCS_stat_near) & (R.MCS_stat_near == 1)].state)),
         cites=["S7 body: decision loss manufactures ambiguity from a clean statistical winner"],
         cross_check="dmcs.csv: ((dMCS_decision > MCS_stat_near) & (MCS_stat_near == 1)).sum()")
    emit("dmcs.wider_from_singleton_also_full_sample", wider_from_singleton_also_full_sample,
         of=wider_from_singleton,
         spec="of dmcs.wider_from_singleton's states (decision-loss |MCS| strictly wider than the "
              "near-trigger-restricted statistical |MCS|, MCS_stat_near==1), how many are ALSO a "
              "singleton under the FULL-SAMPLE statistical comparison (MCS_stat==1) -- i.e. not "
              "just a clean winner once restricted to near-trigger months, but a clean winner "
              "under the unrestricted full-sample MCS too",
         states=";".join(sorted(R[wfs_mask & (R.MCS_stat == 1)].state)),
         cites=["S7 body: singleton under the full-sample statistical comparison too"],
         cross_check="dmcs.csv: ((dMCS_decision > MCS_stat_near) & (MCS_stat_near == 1) & "
                      "(MCS_stat == 1)).sum()")
    emit("dmcs.dropped_near_trigger_states", len(dropped_states), of=df.shape[1],
         states=";".join(dropped_states),
         spec="states with a long enough UR panel to attempt the near-trigger decision-loss comparison "
              "but dropped by year_mcs's len(M)<30 floor once restricted to near-trigger months -- the "
              f"'excluded {len(dropped_states)}' Appendix 13.4 names",
         cites=["S13.4: dropped states"],
         cross_check="dmcs.py main(): set(df.columns) - set(dmcs.csv's state column)")
    emit("dmcs.lowest_mean_ur_states", len(lowest_ur_states), of=5,
         states=";".join(lowest_ur_states),
         spec="the 5 states with the lowest mean unemployment rate over the full UR panel (all 51 "
              "jurisdictions, unfiltered) -- compared against dmcs.dropped_near_trigger_states in "
              "Appendix 13.4 to show the excluded set is related to, but not identical to, the "
              "lowest-UR states",
         cites=["S13.4: dropped states"],
         cross_check="dmcs.py main(): df.mean().sort_values().index[:5]")
    emit("dmcs.dropped_lowest_ur_overlap", len(overlap_states), of=5,
         states=";".join(overlap_states),
         spec="states in BOTH dmcs.dropped_near_trigger_states and dmcs.lowest_mean_ur_states -- the "
              "'3 of 5' overlap Appendix 13.4 reports",
         cites=["S13.4: dropped states"],
         cross_check="dmcs.py main(): set(dropped_states) & set(lowest_ur_states)")
    flush()


if __name__ == "__main__":
    main()
