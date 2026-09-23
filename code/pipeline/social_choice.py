"""
social_choice.py — the selection-as-election reframe: k* is a margin of victory, and the crowned model is usually a
MINORITY winner carried by a pivotal bloc.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(HERE))                 # ar1_null() below imports its sibling election_null_ar1.py
sys.path.insert(0, str(INSTR))
import fragility as F
import null_budget as NB
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
PRED = MULTI / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
RES = MULTI / "results"
METRICS = {"mase": ("sae", lambda s: s.mean()), "rmsse": ("msse", lambda s: np.sqrt(s.mean()))}

N_PERM = 2000
PERM_SEED = 20260902


def _cell_stats(L, w):
    """The five Section 5 statistics for one cell, using fragility()'s own primitives and
    definitions."""
    a = F.pooled_winner(L, w)
    pp = F.per_period_winner(L)
    k, b, _removed = F.decision_breakdown(L, w, a)
    la = np.asarray(L[a], float)
    lb = np.asarray(L[b], float) if b is not None else la
    c = w * (lb - la)
    M = float(c.sum())
    conc = np.nan if F._margin_below_floor(la, lb, w, M) else float(np.max(c) / M)
    cstat = F.condorcet_status(L)
    return (a != pp,
            bool(np.isfinite(conc) and conc > 1),
            k == 1,
            cstat != "winner",
            cstat == "cycle")


def ar1_null(cells, ar_hat, n_perm=N_PERM, seed=PERM_SEED):
    """Same interface and return shape as permutation_null(), but the null distribution comes from
    fresh synthetic AR(1)-exchangeable panels at each cell's own (K, T), not a permutation of the
    real data."""
    from election_null_ar1 import panel_ar1
    obs = np.sum([_cell_stats(L, w) for L, w in cells], axis=0)
    dims = [(len(L), len(w)) for L, w in cells]                   # each cell's own (K, T)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, 5), dtype=int)
    for i in range(n_perm):
        tot = np.zeros(5, dtype=int)
        for (K, T), (L, w) in zip(dims, cells):
            synth = panel_ar1(rng, K, T, ar_hat)
            tot += np.asarray(_cell_stats(synth, w), dtype=int)
        null[i] = tot
    return obs.astype(int), null


def permutation_null(cells, n_perm=N_PERM, seed=PERM_SEED):
    """`cells` is a list of (L, w) panels -- one per (series, metric) cell, exactly the panels main()
    scored."""
    obs = np.sum([_cell_stats(L, w) for L, w in cells], axis=0)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, 5), dtype=int)
    for i in range(n_perm):
        tot = np.zeros(5, dtype=int)
        for L, w in cells:
            models = list(L)
            A = np.column_stack([L[m] for m in models])          # T x K
            for t in range(A.shape[0]):
                rng.shuffle(A[t])                                 # permute labels WITHIN the period
            tot += np.asarray(_cell_stats({m: A[:, j] for j, m in enumerate(models)}, w), dtype=int)
        null[i] = tot
    return obs.astype(int), null


def main():
    d = pd.read_csv(PRED)
    rows = []
    perm_cells = []          # (L, w) per cell, reused by permutation_null() below
    for sid, g in d.groupby("series"):
        for metric, (col, agg) in METRICS.items():
            per = g.groupby(["model", "year"])[col].apply(agg).reset_index()
            piv = per.pivot(index="year", columns="model", values=col).dropna()
            years = list(piv.index)
            L = {m: piv[m].to_numpy(float) for m in piv.columns}
            w = np.ones(len(years))
            perm_cells.append((L, w))
            r = F.fragility(L, w, labels=years, n_boot=1500, seed=0)
            _stab = NB.stable_stability(L, w, n_boot=1500)
            resp = r["responsible"]
            top_year = resp[0] if resp else None                 # the single most pivotal period (k*=1 => that year)
            top_years_by_opp = [rl[0] for rl in r["responsible_by_opponent"].values() if rl]
            pivot_2020_certain = bool(top_years_by_opp) and all(y == 2020 for y in top_years_by_opp)
            pivot_2020_possible = any(y == 2020 for y in top_years_by_opp)
            rows.append({
                "series": sid, "metric": metric,
                "score": r["pooled_winner"], "plurality": r["per_period_winner"], "condorcet": r["condorcet_winner"],
                # agree3 requires a Condorcet WINNER to exist AND equal score/plurality. A cycle or a tie (no
                # winner) can never count as "agreement" (a no-winner cell has condorcet=None).
                "agree3": (r["condorcet_status"] == "winner") and len({r["pooled_winner"], r["per_period_winner"], r["condorcet_winner"]}) == 1,
                "reversal": r["reversal"], "rev_condorcet": r["reversal_condorcet"],
                "condorcet_status": r["condorcet_status"],   # 'winner' | 'cycle' | 'tie'
                "condorcet_cycle": r["condorcet_cycle"],     # TRUE intransitive cycle only
                "k_star": r["k_star"], "concentration": round(r["concentration"], 2),
                "winner_stability": _stab["value"],
                "winner_stability_seed0": r["winner_stability"],
                "winner_stability_sd": round(_stab["sd"], 4),
                "n_binding_tied": r["n_binding_tied"],
                "conc_lo": round(r["concentration_lo"], 2) if pd.notna(r["concentration_lo"]) else np.nan,
                "conc_hi": round(r["concentration_hi"], 2) if pd.notna(r["concentration_hi"]) else np.nan,
                "pivotal_year": top_year, "pivot_2020": (top_year == 2020),
                "reversal_ambiguous": r["reversal_ambiguous"],
                "pivot_2020_certain": pivot_2020_certain, "pivot_2020_possible": pivot_2020_possible,
            })
    R = pd.DataFrame(rows)
    RM._atomic_to_csv(R, RES / "social_choice.csv")
    print("=== SELECTION-AS-ELECTION: do the three canonical voting rules agree on the 'best' model? ===")
    show = R[["series", "metric", "score", "plurality", "condorcet", "reversal", "condorcet_status", "k_star", "concentration", "pivotal_year"]]
    print(show.to_string(index=False))
    n = len(R)
    n_winner = int((R.condorcet_status == "winner").sum())
    n_cycle = int((R.condorcet_status == "cycle").sum())     # TRUE intransitive cycles only
    n_tie = int((R.condorcet_status == "tie").sum())         # no winner, no cycle -> pairwise ties
    print(f"\n  score-vs-plurality REVERSAL: {int(R.reversal.sum())}/{n} cells "
          f"({100*R.reversal.mean():.0f}%)")
    print(f"  Condorcet: winner EXISTS in {n_winner}/{n} cells; genuine intransitive CYCLE in {n_cycle}/{n}; "
          f"pairwise-TIE (no winner, no cycle) in {n_tie}/{n}.")
    print(f"  Of the {n_winner} cells where a Condorcet winner exists, score-vs-Condorcet reversal in "
          f"{int(R.rev_condorcet.sum())}/{n_winner} ({100*R.rev_condorcet.sum()/max(n_winner,1):.0f}%)")
    print(f"  all-three-agree (Condorcet winner exists AND all three coincide): {int(R.agree3.sum())}/{n} cells")
    print(f"  MINORITY winner on a pivotal bloc (concentration > 1, i.e. one period outweighs the net margin): "
          f"{int((R.concentration > 1).sum())}/{n} cells (max {R.concentration.max():.1f}x)")
    piv = R[R.pivotal_year.notna()]
    print(f"  the single most-pivotal period is 2020 in {int(piv.pivot_2020.sum())}/{len(piv)} cells "
          f"-> UNIFIES 'shape-not-magnitude' with selection (the crown rides the 2020 cliff).")
    rob = R.groupby("series").reversal.any()
    print(f"  robust series (no reversal on any metric): {list(rob[~rob].index)}  (expect PI = the one identified MCS case).")
    print("  FRAME: rule-sensitivity + margin of victory (computational social choice). NOT an Arrow impossibility.")

    # Both blocks below reuse this file's own already-computed R.
    _pi = R[R.series == "PI"]
    emit("election.pi_concentration", float(_pi.concentration.min()),
         spec="Personal Income's concentration -- the identified control, lowest in the panel",
         cites=["S4 body: Table 1 PI Concentration column"],
         cross_check="social_choice.csv: concentration.min() where series=='PI'")
    from scipy.stats import spearmanr as _sp
    _mt = pd.read_csv(RES / "mcs_test.csv")
    for _metric, _col in (("mase", "MCS_abs_10"), ("rmsse", "MCS_rmsse_10")):
        _j = R[R.metric == _metric].merge(_mt[["series", _col]], on="series").dropna(
            subset=[_col, "winner_stability"])
        _r, _pv = _sp(_j[_col], _j["winner_stability"])
        emit(f"election.spearman_mcs_stability_{_metric}", float(_r), n_series=len(_j),
             spec=f"Spearman(|MCS|, winner_stability) across curated series on the {_metric} metric, "
                  f"paired with the MATCHING loss family ({_col}). §2.4 directs the reader to "
                  f"Section 5.4 for both instruments (winner_stability read against the "
                  f"exchangeable null's 5th percentile, and the Model Confidence Set's tied set); "
                  f"it does not itself promise a rank correlation between them -- this key is "
                  f"Section 5.4's own reporting of how the two instruments relate",
             cites=["S5.4 body: the two instruments"],
             cross_check=f"social_choice.csv + mcs_test.csv: spearman(winner_stability, {_col}) on "
                         f"metric=='{_metric}'")
        emit(f"election.spearman_mcs_stability_{_metric}_p", float(_pv),
             spec=f"two-sided p-value for election.spearman_mcs_stability_{_metric} at n={len(_j)}",
             cites=["S5.4 body: the two instruments"],
             cross_check=f"social_choice.csv + mcs_test.csv: spearmanr p on metric=='{_metric}'")

    _rho = R[["k_star", "winner_stability"]].dropna().corr(method="spearman").iloc[0, 1]
    emit("election.spearman_kstar_stability", float(_rho),
         spec="Spearman rank correlation between k* (descriptive) and winner_stability (inferential), "
              "across the curated panel's cells",
         cites=["S5.4 body: rank correlation between k* and winner_stability"],
         cross_check="social_choice.csv: corr(k_star, winner_stability, method='spearman')")

    _pobs, _pnull = permutation_null(perm_cells)
    _pnames = ["reversal", "conc_gt1", "kstar1", "no_condorcet", "cycle"]
    _pspec = {
        "reversal": "score winner != plurality winner",
        "conc_gt1": "the single most pivotal period contributes more than the champion's whole net margin",
        "kstar1": "a single period's deletion flips the pooled winner",
        "no_condorcet": "no Condorcet winner exists (cycle or tie)",
        "cycle": "a TRUE intransitive majority cycle",
    }
    for _j, _nm in enumerate(_pnames):
        _col = _pnull[:, _j]
        _p_ge = float((_col >= _pobs[_j]).mean())
        _p_le = float((_col <= _pobs[_j]).mean())
        emit(f"election.permnull_{_nm}_mean", round(float(_col.mean()), 2), of=int(n),
             n_perm=int(N_PERM), observed=int(_pobs[_j]),
             p05=int(round(np.percentile(_col, 5))), p95=int(round(np.percentile(_col, 95))),
             spec=f"MEAN count of cells ({_pspec[_nm]}) under the within-period model-label "
                  f"permutation null, {N_PERM} draws, seed {PERM_SEED}. The null permutes model "
                  f"labels within each period independently, preserving each period's own multiset of losses "
                  f"(so its difficulty and dispersion, including the 2020 cliff at full size) and every "
                  f"marginal loss value -- destroying the association between model identity and "
                  f"performance, and with it each named model's serial persistence across periods and "
                  f"the correlation between named models' loss series (the real panel's median "
                  f"cross-model level correlation is recorded as modelset.loss_corr_median; the "
                  f"permuted counterpart is not measured by this producer). Compare against the "
                  f"observed count in "
                  f"the `observed` field: this is the arbitrary-winner benchmark Section 5's own "
                  f"language implicitly claims to beat.",
             cites=["S5.0 body: permutation-null calibration"],
             cross_check=f"social_choice.permutation_null(): null[:, {_j}].mean()")
        emit(f"election.permnull_{_nm}_p_ge", round(_p_ge, 3), n_perm=int(N_PERM),
             observed=int(_pobs[_j]),
             spec=f"P(null count >= observed) for {_pspec[_nm]}. A LARGE value means the observed "
                  f"count is at or BELOW what an arbitrary winner produces -- i.e. the raw count is "
                  f"not evidence of the aggregation fragility Section 5 describes.",
             cites=["S5.0 body: permutation-null calibration"],
             cross_check=f"social_choice.permutation_null(): (null[:, {_j}] >= obs).mean()")
        # reversal's and cycle's p_le are the two quoted in prose, and they are quoted in the
        # subsections that discuss them (5.1 and 5.2), not in 5.0's table.
        _le_cite = {"reversal": "S5.1 body: reversal below the null",
                    "cycle": "S5.2 body: transitivity above chance"}.get(
                        _nm, "S5.0 body: permutation-null calibration")
        emit(f"election.permnull_{_nm}_p_le", round(_p_le, 3), n_perm=int(N_PERM),
             observed=int(_pobs[_j]),
             spec=f"P(null count <= observed) for {_pspec[_nm]}. A SMALL value means the observed "
                  f"count is significantly BELOW the null -- a finding in the OPPOSITE direction "
                  f"from the one the raw count suggests.",
             cites=[_le_cite],
             cross_check=f"social_choice.permutation_null(): (null[:, {_j}] <= obs).mean()")

    from election_null_ar1 import estimate_ar1_coefficients
    _ar_hat = float(np.median(estimate_ar1_coefficients(d)["ar1_estimate"].to_numpy(float)))
    _sobs, _snull = ar1_null(perm_cells, _ar_hat)
    print(f"\n  AR(1)-calibrated exchangeable null (ar_hat={_ar_hat:.4f}): std per statistic = "
          f"{_snull.std(0).round(2).tolist()} (a real, non-degenerate null, unlike the whole-series "
          f"permutation this replaces)")
    emit("election.ar1null_coefficient", round(_ar_hat, 4),
         spec="pooled median lag-1 AR coefficient used to calibrate this null -- IDENTICAL "
              "computation and value to election.ar1_coefficient_empirical (election_null_ar1.py); "
              "re-derived here rather than imported as a value so this file's own emit is "
              "self-contained, but the two must and do agree",
         cites=["S5.0 body: AR(1)-calibrated exchangeable null"],
         cross_check=None)
    for _j, _nm in enumerate(_pnames):
        _col = _snull[:, _j]
        _p_ge = float((_col >= _sobs[_j]).mean())
        _p_le = float((_col <= _sobs[_j]).mean())
        emit(f"election.ar1null_{_nm}_mean", round(float(_col.mean()), 2), of=int(n),
             n_perm=int(N_PERM), observed=int(_sobs[_j]),
             p05=int(round(np.percentile(_col, 5))), p95=int(round(np.percentile(_col, 95))),
             sd=round(float(_col.std()), 3),
             spec=f"MEAN count of cells ({_pspec[_nm]}) under the AR(1)-calibrated exchangeable null "
                  f"-- fresh synthetic K-model panels at each cell's own (K,T), genuinely exchangeable "
                  f"(no model has a real population edge) but carrying the same serial dependence "
                  f"real models do (AR(1) at the empirically-estimated pooled coefficient), not a "
                  f"permutation of the fixed real data. Reported alongside, not instead of, the "
                  f"within-period figure.",
             cites=["S5.0 body: AR(1)-calibrated exchangeable null"],
             cross_check=f"social_choice.ar1_null(): null[:, {_j}].mean()")
        emit(f"election.ar1null_{_nm}_p_ge", round(_p_ge, 3), n_perm=int(N_PERM),
             observed=int(_sobs[_j]),
             spec=f"P(AR(1)-null count >= observed) for {_pspec[_nm]}, the serial-dependence-"
                  f"calibrated counterpart to election.permnull_{_nm}_p_ge",
             cites=["S5.0 body: AR(1)-calibrated exchangeable null"],
             cross_check=f"social_choice.ar1_null(): (null[:, {_j}] >= obs).mean()")
        _sle_cite = {"reversal": "S5.1 body: reversal below the null, AR(1)-calibrated check",
                    "cycle": "S5.2 body: transitivity above chance, AR(1)-calibrated check"}.get(
                        _nm, "S5.0 body: AR(1)-calibrated exchangeable null")
        emit(f"election.ar1null_{_nm}_p_le", round(_p_le, 3), n_perm=int(N_PERM),
             observed=int(_sobs[_j]),
             spec=f"P(AR(1)-null count <= observed) for {_pspec[_nm]}, the serial-dependence-"
                  f"calibrated counterpart to election.permnull_{_nm}_p_le",
             cites=[_sle_cite],
             cross_check=f"social_choice.ar1_null(): (null[:, {_j}] <= obs).mean()")

    emit("election.reversals", int(R.reversal.sum()), of=int(n),
         spec="cells where the SCORE (pooled-mean) winner differs from the PLURALITY (per-period) winner; "
              "as reported, resolves any plurality tie by sorted model name (see reversals_lo/hi for the "
              "range under any tie-break)",
         cites=["S5.1 body: score-vs-plurality reversal"],
         cross_check="social_choice.csv: reversal.sum()")
    emit("election.reversals_lo", int((R.reversal & ~R.reversal_ambiguous).sum()), of=int(n),
         spec="cells where the score-vs-plurality reversal holds under EVERY possible plurality "
              "tie-break -- a lower bound on election.reversals",
         cites=["S5.1 body: score-vs-plurality reversal"],
         cross_check="social_choice.csv: (reversal & ~reversal_ambiguous).sum()")
    emit("election.reversals_hi", int((R.reversal & ~R.reversal_ambiguous).sum() + R.reversal_ambiguous.sum()), of=int(n),
         spec="cells where the score-vs-plurality reversal holds under AT LEAST ONE possible plurality "
              "tie-break -- an upper bound on election.reversals",
         cites=["S5.1 body: score-vs-plurality reversal"],
         cross_check="social_choice.csv: (reversal & ~reversal_ambiguous).sum() + reversal_ambiguous.sum()")
    emit("election.reversals_pct", round(100 * int(R.reversal.sum()) / int(n)),
         spec="election.reversals as a share of the panel, rounded to the nearest percent",
         cites=["S5.1 body: score-vs-plurality reversal"],
         cross_check="round(100 * election.reversals / n)")
    _piv = R[R.pivotal_year.notna()]
    emit("election.pivot_2020", int(_piv.pivot_2020.sum()), of=int(len(_piv)),
         spec="cells (of those with a defined pivotal period) where the single most-pivotal period is "
              "2020, as reported under the deterministic (sorted-opponent-name) tie-break -- see "
              "pivot_2020_lo/hi for the range under any tie-break among opponents tied at k*",
         cites=["S6 body: 2020 is the most-pivotal period in 5-7 of 24 cells"],
         cross_check="social_choice.csv: pivotal_year.notna() -> pivot_2020.sum()")
    emit("election.pivot_2020_lo", int(_piv.pivot_2020_certain.sum()), of=int(len(_piv)),
         spec="cells where the most-pivotal period is 2020 under EVERY opponent tied at k* -- a lower "
              "bound on election.pivot_2020",
         cites=["S6 body: 2020 is the most-pivotal period in 5-7 of 24 cells"],
         cross_check="social_choice.csv: pivotal_year.notna() -> pivot_2020_certain.sum()")
    emit("election.pivot_2020_hi", int(_piv.pivot_2020_possible.sum()), of=int(len(_piv)),
         spec="cells where the most-pivotal period is 2020 under AT LEAST ONE opponent tied at k* -- "
              "an upper bound on election.pivot_2020",
         cites=["S6 body: 2020 is the most-pivotal period in 5-7 of 24 cells"],
         cross_check="social_choice.csv: pivotal_year.notna() -> pivot_2020_possible.sum()")
    emit("election.condorcet_winner_cells", n_winner, of=int(n),
         spec="cells where a Condorcet (pairwise-majority) winner exists",
         cites=["S5.2 body: Condorcet winner existence"],
         cross_check="social_choice.csv: (condorcet_status == 'winner').sum()")
    emit("election.rev_condorcet", int(R.rev_condorcet.sum()), of=n_winner,
         spec="of the cells where a Condorcet winner exists, how many have the score-voting winner "
              "disagree with the Condorcet winner",
         cites=["S5.2 body: score-vs-Condorcet reversal"],
         cross_check="social_choice.csv: (condorcet_status == 'winner') -> rev_condorcet.sum()")
    emit("election.agree3", int(R.agree3.sum()), of=int(n),
         spec="cells where a Condorcet winner exists AND the score, plurality, and Condorcet "
              "winners all coincide -- full agreement among the three canonical voting rules",
         cites=[],
         cross_check="social_choice.csv: agree3.sum()")
    emit("election.condorcet_tie_cells", n_tie, of=int(n),
         spec="cells with no Condorcet winner and no cycle -- a pairwise tie",
         cites=["S5.2 body: pairwise-tie cells"],
         cross_check="social_choice.csv: (condorcet_status == 'tie').sum()")
    emit("election.condorcet_cycle_cells", n_cycle, of=int(n),
         spec="cells with a genuine intransitive Condorcet cycle",
         cites=["S5.2 body: intransitive-cycle cells"],
         cross_check="social_choice.csv: (condorcet_status == 'cycle').sum()")
    emit("election.concentration_gt1", int((R.concentration > 1).sum()), of=int(n),
         spec="cells where the single most-pivotal period contributes MORE than the champion's entire "
              "net margin over its binding opponent -- deleting it reverses the verdict",
         cites=["S5.3 body: concentration > 1 cells"],
         cross_check="social_choice.csv: (concentration > 1).sum()")
    emit("election.concentration_max", float(R.concentration.max()),
         spec="largest single-period concentration value in the panel",
         cites=["S5.3 body: maximum concentration"],
         cross_check="social_choice.csv: concentration.max()")
    emit("election.concentration_gt1_pct", round(100 * int((R.concentration > 1).sum()) / int(n)),
         spec="election.concentration_gt1 as a share of the panel, rounded to the nearest percent",
         cites=["S5.3 body: concentration > 1 cells"],
         cross_check="round(100 * election.concentration_gt1 / n)")
    emit("election.stability_median", float(R.winner_stability.median()),
         sd_median=round(float(R.winner_stability_sd.median()), 4),
         sd_max=round(float(R.winner_stability_sd.max()), 4),
         spec="median winner_stability across the curated panel's cells (fragility()'s own "
              "per-cell statistic, seed-averaged via null_budget.stable_stability)",
         cites=["S5.4 body: observed median winner_stability"],
         cross_check="social_choice.csv: winner_stability.median()")

    st = R.winner_stability
    _K = int(d.model.nunique())
    _T = int(d.groupby(["series", "year"]).ngroups / d.series.nunique())
    _nbd = NB.stable_benchmark(_K, _T)
    _nb_med, _nb_lo = _nbd["median"], _nbd["p05"]
    emit("election.stability_null_median", round(float(_nb_med), 3), n_models=_K, n_periods=_T,
         n_eff=_nbd["n_eff"],
         spec=f"median winner_stability under EXCHANGEABLE models (K={_K}, T={_T}, {_nbd['n_eff']} "
              f"panels across {_nbd['n_seeds']} independent seeds): what the statistic returns when "
              f"the winner is arbitrary by construction. The benchmark the observed median must be "
              f"read against",
         cites=["S5.4 body: the calibrated companion"], cross_check=None)
    emit("election.stability_null_p05", round(float(_nb_lo), 3), n_models=_K, n_periods=_T,
         band=[round(_nbd["p05_lo"], 3), round(_nbd["p05_hi"], 3)], sd=round(_nbd["p05_sd"], 4),
         n_eff=_nbd["n_eff"],
         spec=f"5th percentile of winner_stability under exchangeable models (K={_K}, T={_T}); a cell "
              f"below this is unusually fragile RELATIVE TO CHANCE, which is the claim the word "
              f"'calibrated' makes",
         cites=["S5.4 body: the calibrated companion"], cross_check=None)
    emit("election.stability_min", float(st.min()),
         spec="lowest observed winner_stability in the curated panel's cells",
         cites=["S5.4 body: the lowest cell"],
         cross_check="social_choice.csv: winner_stability.min()")
    _below = int((st < _nb_lo).sum())
    _below_band = sorted({int((st < _nbd["p05_lo"]).sum()), int((st < _nbd["p05_hi"]).sum())})
    emit("election.stability_below_null_p05", _below, of=int(len(st)),
         band_count=_below_band, threshold=round(float(_nb_lo), 3),
         expected=round(0.05 * len(st), 1),
         spec="cells whose winner_stability falls below the exchangeable null's OWN fifth percentile "
              "-- the only fragility claim this panel supports, since a fixed cut like 0.60 sits above "
              "the arbitrary-winner benchmark. Read against `expected`: at a fifth percentile, about "
              "one such cell in twenty is expected by chance, so this count is a tail description and "
              "not evidence. `band_count` is the count at each end of the threshold's own seed band",
         cites=["S5.4 body: cells below the null's fifth percentile"],
         cross_check="social_choice.csv: (winner_stability < election.stability_null_p05).sum()")
    emit("election.stability_below_060", int((st < 0.60).sum()), of=int(len(st)),
         spec=f"cells with winner_stability below 0.60. NOT EVIDENCE OF FRAGILITY ON ITS OWN, and "
              f"must not be reported as such: the exchangeable null's own median sits well BELOW "
              f"0.60 (it is {round(float(_nb_med), 3)} -- see election.stability_null_median), so a "
              f"0.60 cut sits ABOVE the arbitrary-winner benchmark and would flag most of a "
              f"pure-null panel. Only cells below the null's 5th percentile "
              f"(election.stability_null_p05) are unusually fragile relative to chance; that count "
              f"is election.stability_below_null_p05, reported with its own expected value",
         cites=[],  # NOT REPORTED in prose by decision -- see election.stability_below_null_p05
         cross_check="social_choice.csv: (winner_stability < 0.60).sum()")
    _tied = R[R.n_binding_tied > 1]
    emit("election.conc_tied_cells", int(len(_tied)), of=int(n),
         spec="cells where more than one opponent ties at the minimum k*, so `concentration` is one "
              "of several defensible values rather than a point",
         cites=["S5.3 body: why a single concentration figure is not a bound"],
         cross_check="social_choice.csv: (n_binding_tied > 1).sum()")
    emit("election.conc_reported_not_max", int(((_tied.concentration - _tied.conc_hi).abs() > 1e-9).sum()),
         of=int(len(_tied)),
         spec="of the tied cells, those whose REPORTED concentration is not the largest in its range "
              "-- i.e. where quoting the single figure UNDERSTATES how concentrated the margin can be",
         cites=["S5.3 body: why a single concentration figure is not a bound"],
         cross_check="social_choice.csv: among n_binding_tied>1, concentration != conc_hi")
    _argmax = R.loc[R.concentration.idxmax()]
    emit("election.concentration_max_cell_lo", float(_argmax.conc_lo),
         spec="the same cell as election.concentration_max (largest single-period concentration in the "
              "panel), but against its most DISTANT tied opponent rather than the alphabetically-first "
              "one -- the low end of the range election.concentration_max is drawn from",
         cites=["S5.3 body: why a single concentration figure is not a bound"],
         cross_check="social_choice.csv: conc_lo where concentration is max")

    wrows = []
    for sid, g in d.groupby("series"):
        n_models_per_year = g.groupby("year")["model"].nunique()
        if n_models_per_year.nunique() > 1:
            raise RuntimeError(
                f"{sid}: model count varies by year ({dict(n_models_per_year)}) -- nmonths_year is no "
                f"longer a constant rescaling of the true month count, and w_mo below would silently "
                f"weight years by (models x months) instead of the intended months alone")
        nmonths_year = g.groupby("year").size()
        for metric, (col, agg) in METRICS.items():
            per = g.groupby(["model", "year"])[col].apply(agg).reset_index()
            piv = per.pivot(index="year", columns="model", values=col).dropna()
            years = list(piv.index)
            L = {m: piv[m].to_numpy(float) for m in piv.columns}
            w_eq = np.ones(len(years))
            w_mo = np.array([float(nmonths_year.get(y, 12)) for y in years])
            ra = F.fragility(L, w_eq, labels=years, n_boot=1500, seed=0)
            rb = F.fragility(L, w_mo, labels=years, n_boot=1500, seed=0)
            wrows.append({
                "series": sid, "metric": metric,
                "winner_equal_year": ra["pooled_winner"], "winner_month_wtd": rb["pooled_winner"],
                "k_equal_year": ra["k_star"], "k_month_wtd": rb["k_star"],
                "conc_equal_year": round(ra["concentration"], 2),
                "conc_month_wtd": round(rb["concentration"], 2),
            })
    W = pd.DataFrame(wrows)
    RM._atomic_to_csv(W, RES / "social_choice_weights.csv")
    n_w = len(W)
    _tcu = W[(W.series == "TCU") & (W.metric == "mase")].iloc[0]
    emit("election.weight_winner_changes", int((W.winner_equal_year != W.winner_month_wtd).sum()), of=int(n_w),
         spec="cells where the pooled (score) winner differs between equal-year weighting and "
              "month-count weighting of the evaluation periods",
         cites=["S5 body: period-weighting robustness"],
         cross_check="social_choice_weights.csv: (winner_equal_year != winner_month_wtd).sum()")
    emit("election.weight_kstar_changes", int((W.k_equal_year != W.k_month_wtd).sum()), of=int(n_w),
         spec="cells where k* differs between equal-year weighting and month-count weighting",
         cites=["S5 body: period-weighting robustness"],
         cross_check="social_choice_weights.csv: (k_equal_year != k_month_wtd).sum()")
    emit("election.weight_conc_gt1_equal", int((W.conc_equal_year > 1).sum()), of=int(n_w),
         spec="cells with concentration > 1 under equal-year weighting -- matches election.concentration_gt1, "
              "since equal-year weighting is what the rest of this file already uses",
         cites=["S5 body: period-weighting robustness"],
         cross_check="social_choice_weights.csv: (conc_equal_year > 1).sum()")
    emit("election.weight_conc_gt1_monthwtd", int((W.conc_month_wtd > 1).sum()), of=int(n_w),
         spec="cells with concentration > 1 under month-count weighting",
         cites=["S5 body: period-weighting robustness"],
         cross_check="social_choice_weights.csv: (conc_month_wtd > 1).sum()")
    emit("election.weight_tcu_conc_equal", float(_tcu.conc_equal_year),
         spec="TCU/mase's own concentration under equal-year weighting, used as the worked example of "
              "weight-sensitivity -- NOT the panel's largest concentration cell (that is "
              "election.concentration_max)",
         cites=["S5 body: period-weighting robustness"],
         cross_check="social_choice_weights.csv: conc_equal_year where series=='TCU' and metric=='mase'")
    emit("election.weight_tcu_conc_monthwtd", float(_tcu.conc_month_wtd),
         spec="the same TCU/mase cell's concentration under month-count weighting",
         cites=["S5 body: period-weighting robustness"],
         cross_check="social_choice_weights.csv: conc_month_wtd where series=='TCU' and metric=='mase'")
    flush()


if __name__ == "__main__":
    main()
