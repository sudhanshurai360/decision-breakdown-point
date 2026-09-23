"""
ml_witness.py — a third arena: does the k*/non-identification/crowding signature appear on an ML leaderboard?
"""
import sys, json, time, urllib.request, urllib.error, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(INSTR))
import fragility as F
import null_budget as NB
from canonical_emit import emit, flush
import arena_crowding as AC
CACHE = MULTI / "data" / "open_llm.parquet"
TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
API = "https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents&config=default&split=train"


def fetch(nmax=20000):
    if CACHE.exists(): return pd.read_parquet(CACHE)
    rows = []
    total = None
    completed = False
    for off in range(0, nmax, 100):
        req = urllib.request.Request(f"{API}&offset={off}&length=100", headers={"User-Agent": "Mozilla/5.0"})
        d = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=40) as resp:
                    d = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    wait = 5 * (attempt + 1)
                    print(f"  429 at offset {off}, attempt {attempt+1}/6 -- waiting {wait}s"); time.sleep(wait)
                    continue
                print(f"  stop at offset {off}: {e}"); break
            except Exception as e:
                if attempt < 5:
                    wait = 5 * (attempt + 1)
                    print(f"  transient error at offset {off}, attempt {attempt+1}/6: {e} -- waiting {wait}s")
                    time.sleep(wait)
                    continue
                print(f"  stop at offset {off} after 6 attempts: {e}"); break
        if d is None: break
        if total is None:
            total = d.get("num_rows_total")
            print(f"  API reports num_rows_total={total}")
        r = d.get("rows", [])
        if not r:
            completed = True
            break
        for x in r:
            row = x["row"]
            rows.append({"model": row.get("fullname") or row.get("Model"),
                         **{t: row.get(t) for t in TASKS}})
        if len(r) < 100:
            completed = True
            break
        if total is not None and off + 100 >= total:
            completed = True
            break
    else:
        print(f"  [warn] fetch loop hit its nmax={nmax} backstop without exhausting the API "
              f"(num_rows_total={total}) -- the candidate pool may again be silently truncated.")
    if not completed:
        raise RuntimeError(
            f"fetch() aborted before exhausting the leaderboard (collected {len(rows)} rows, "
            f"API reports num_rows_total={total}) -- refusing to write a silently-truncated "
            f"{CACHE.name}. Re-run once the underlying network issue is resolved; do not delete "
            f"the (nonexistent) cache to work around this."
        )
    df = pd.DataFrame(rows)
    for t in TASKS: df[t] = pd.to_numeric(df[t], errors="coerce")
    df = df.dropna(subset=TASKS).drop_duplicates("model").reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(CACHE); print(f"  fetched {len(df)} complete models")
    return df


def crowd(mat, rng):
    """For each pool size k in [2,4,8,16,32,64] (stopping once k exceeds len(mat)), draw 500 random
    k-model sub-panels and check how often the mean-loss winner (mw) differs from the rank-
    aggregation/Borda-style winner (lowest mean RANK across tasks, not lowest mean raw loss) --
    the flip rate at that pool size."""
    return AC.crowd(mat, rng, ks=[2, 4, 8, 16, 32, 64])


def main():
    df = fetch(); print(f"Open LLM Leaderboard: {len(df)} models x {len(TASKS)} tasks")
    S = df[TASKS].to_numpy(float)                       # scores (higher=better)
    L = 100.0 - S                                       # loss (lower=better)
    mean = L.mean(1); order = np.argsort(mean)          # leaderboard order (by mean loss)
    w = np.ones(len(TASKS))

    # (1) k* of the #1: fewest tasks to flip the top model
    Ld = {df.model.iloc[i]: L[i] for i in order[:30]}   # top-30 as the candidate pool
    # stability_block=1: the 6 tasks are an exchangeable battery with no adjacency
    r = F.fragility(Ld, w, n_boot=2000, seed=0, stability_block=1)
    _stab = NB.stable_stability(Ld, w, n_boot=2000, block=1)
    ws_stable, ws_sd, ws_lo, ws_hi, ws_nseeds = (_stab["value"], _stab["sd"], _stab["lo"], _stab["hi"],
                                                  _stab["n_seeds"])
    print("\n=== (1) k* OF THE LEADERBOARD #1 (over 6 tasks) ===")
    print(f"  #1 by mean: {df.model.iloc[order[0]]}")
    print(f"  k* = {r['k_star']} of 6 tasks flip the top model (binding rival: {r['runner_up']}); "
          f"winner_stability={ws_stable:.3f} over {ws_nseeds} seeds (seed=0 alone: {r['winner_stability']:.2f})")
    print(f"  -> the crown rests on {r['k_star']}/6 task(s); MCS-style top set is {'a near-tie' if ws_stable<0.9 else 'identified'}.")
    emit("llm.kstar", int(r["k_star"]), of=6, rival=str(r["runner_up"]),
         spec="fewest of the 6 Open LLM Leaderboard tasks whose deletion flips the top (mean-score) "
              "model against its BINDING rival (the opponent that overtakes the top model fastest, "
              "named here as `rival`)",
         cites=["N2 body: the finding"],
         cross_check=None)
    emit("llm.top_model", str(df.model.iloc[order[0]]),
         spec="the #1 Open LLM Leaderboard model by mean loss over the 6 tasks, among the top-30 "
              "candidate pool k*/winner_stability are computed on -- the model llm.kstar's margin "
              "and llm.kstar's `rival` qualifier are both measured against",
         cites=[], cross_check=None)
    emit("llm.concentration", round(float(r["concentration"]), 3),
         n_binding_tied=int(r["n_binding_tied"]),
         spec="share of the top model's k*-margin against its binding rival (llm.kstar's `rival`) "
              "contributed by the single most influential task, over the top-30 candidate pool -- "
              "the same statistic Table 1 reports per curated series (S4) and "
              "flagship.concentration reports for the legacy-neural exhibit",
         cites=[], cross_check=None)
    emit("llm.reversal", bool(r["reversal"]),
         spec="whether the top-30 pool's pooled (mean-loss) winner differs from its per-task "
              "(plurality) winner -- the same statistic Table 1 reports per curated series as its "
              "`Reversal` column and flagship.reversal reports for the legacy-neural exhibit",
         cites=[], cross_check=None)
    emit("llm.fragile", bool(r["fragile"]),
         spec=f"whether the Open LLM Leaderboard's top model meets the paper's fixed fragile rule "
              f"(winner_stability<{F.FRAGILE_STABILITY_THRESH:.2f}, or reversal with "
              f"k*/T<{F.FRAGILE_KSTAR_RATIO_THRESH:.2f}) -- the same rule Table 1 applies "
              f"per curated series and flagship.fragile reports for the legacy-neural exhibit",
         cites=[], cross_check=None)
    emit("llm.condorcet_winner", r["condorcet_winner"],
         spec="the Condorcet winner among the top-30 candidate pool (the model beating every other "
              "model on a majority of the 6 tasks), if one exists -- None if the pairwise-comparison "
              "graph has a majority cycle or a tie (see llm.condorcet_status)",
         cites=[], cross_check=None)
    emit("llm.condorcet_status", str(r["condorcet_status"]),
         spec="whether a Condorcet winner exists among the top-30 pool, and if not, why (cycle or "
              "tie) -- the same {score, plurality, Condorcet} triple that "
              "election.condorcet_winner_cells / _cycle_cells / _tie_cells count across the "
              "curated-panel cells, and that flagship.condorcet_status reports for the "
              "legacy-neural exhibit",
         cites=[], cross_check=None)
    emit("llm.reversal_condorcet", bool(r["reversal_condorcet"]),
         spec="whether the top-30 pool's pooled (mean-loss) winner differs from its Condorcet "
              "winner (when one exists) -- the Condorcet-comparison analogue of llm.reversal's "
              "plurality comparison",
         cites=[], cross_check=None)
    emit("llm.concentration_lo", round(float(r["concentration_lo"]), 3),
         n_binding_tied=int(r["n_binding_tied"]),
         spec="lower edge of llm.concentration's ambiguity range across all binding-tied rivals "
              "(n_binding_tied>1 here, unlike flagship_fragility.py's degenerate n_binding_tied==1 "
              "case) -- the range a reader should read llm.concentration against, not a single "
              "point estimate against only one arbitrarily-chosen rival",
         cites=[], cross_check=None)
    emit("llm.concentration_hi", round(float(r["concentration_hi"]), 3),
         n_binding_tied=int(r["n_binding_tied"]),
         spec="upper edge of llm.concentration's ambiguity range across all binding-tied rivals -- "
              "see llm.concentration_lo",
         cites=[], cross_check=None)
    _K, _T = len(Ld), len(TASKS)
    _nbd = NB.stable_benchmark(_K, _T, block=1)
    emit("llm.winner_stability", round(float(ws_stable), 3),
         band=[round(ws_lo, 3), round(ws_hi, 3)], sd=round(ws_sd, 4), n_seeds=ws_nseeds,
         spec="iid-bootstrap (stability_block=1 -- the 6 tasks are an exchangeable battery with no "
              "adjacency, see this call site's own comment) winner_stability of the Open LLM "
              "Leaderboard's top model over its 6 tasks, computed on the top-30 candidate pool "
              f"(K={_K}), AVERAGED OVER {ws_nseeds} independent bootstrap seeds via NB.stable_stability "
              f"(seed=0 alone gives {r['winner_stability']:.2f}, within this value's own "
              f"seed-to-seed sd={ws_sd:.4f}). "
              "Read against llm.winner_stability_null_median/p05/p95 -- its own calibrated "
              "exchangeable null at this exact (K, T), not the K=6/T=32 null quoted elsewhere in the "
              "project (a different panel's dimensions) and not a bare fragile/robust threshold",
         cites=["N4 body: calibrating winner_stability"],
         cross_check=None)
    emit("llm.winner_stability_null_median", round(float(_nbd["median"]), 3), n_models=_K, n_periods=_T,
         n_eff=_nbd["n_eff"],
         spec=f"median winner_stability under EXCHANGEABLE models (K={_K}, T={_T}, {_nbd['n_eff']} "
              f"panels across {_nbd['n_seeds']} independent seeds): what the statistic returns when "
              f"the winner is arbitrary by construction, at the Open LLM Leaderboard's own dimensions",
         cites=["N4 body: calibrating winner_stability"], cross_check=None)
    emit("llm.winner_stability_null_p05", round(float(_nbd["p05"]), 3), n_models=_K, n_periods=_T,
         band=[round(_nbd["p05_lo"], 3), round(_nbd["p05_hi"], 3)], sd=round(_nbd["p05_sd"], 4),
         n_eff=_nbd["n_eff"],
         spec=f"5th percentile of winner_stability under exchangeable models (K={_K}, T={_T}) at the "
              f"leaderboard's own dimensions; a value below this is unusually fragile RELATIVE TO CHANCE",
         cites=["N4 body: calibrating winner_stability"], cross_check=None)
    emit("llm.winner_stability_null_p95", round(float(_nbd["p95"]), 3), n_models=_K, n_periods=_T,
         band=[round(_nbd["p95_lo"], 3), round(_nbd["p95_hi"], 3)], sd=round(_nbd["p95_sd"], 4),
         n_eff=_nbd["n_eff"],
         spec=f"95th percentile of winner_stability under exchangeable models (K={_K}, T={_T}) at the "
              f"leaderboard's own dimensions -- the upper edge of the same band as "
              f"llm.winner_stability_null_p05, reported so both tails of the null are on record. The "
              f"sd here is the p95 edge's OWN seed-to-seed spread, not to be "
              f"conflated with llm.winner_stability_null_p05's sd -- the companion note's observed-vs-p95 "
              f"comparison needs this one specifically",
         cites=["N4 body: calibrating winner_stability"], cross_check=None)

    # (2) crowding vs an INDEPENDENCE NULL -- the RISE is mechanical (nulls rise too); what matters is real-vs-null.
    rng = np.random.default_rng(0)
    Lnull = np.column_stack([rng.permutation(L[:, j]) for j in range(L.shape[1])])  # shuffle each task -> kill cross-model structure
    real = crowd(L, rng)
    null, null_sd, null_first = AC.average_null(L, rng, [2, 4, 8, 16, 32, 64], Lnull)
    print("\n=== (2) CROWDING vs INDEPENDENCE NULL (is the leaderboard MORE or LESS identified than chance?) ===")
    print(f"  {'#models':>8s} | REAL flip | col-shuffle NULL | ")
    for k in real:
        print(f"  {k:>8d} |   {real[k]:.1%}   |     {null[k]:.1%}      | {'REAL < null -> comparatively IDENTIFIED' if real[k] < null[k] else 'REAL >= null -> fragile'}")
        emit(f"llm.crowd_real_k{k}", real[k],
             spec=f"share of {k}-model random draws from the real Open LLM Leaderboard where the mean-loss "
                  f"winner differs from the rank-aggregation (Borda-style) winner",
             cites=["N2 body: the finding"],
             cross_check=None)
        emit(f"llm.crowd_null_k{k}", null[k],
             null_sd=round(float(null_sd[k]), 4), n_perm=AC.N_NULL_PERM, first_permutation=float(null_first[k]),
             spec=f"same statistic as llm.crowd_real_k{k} but on a column-shuffled (independence-null) version "
                  f"of the score matrix -- isolates the mechanical crowding effect from real structure. "
                  f"MEAN over {AC.N_NULL_PERM} column permutations (companions null_sd, "
                  f"first_permutation = the single seed-0 draw)",
             cites=["N2 body: the finding"],
             cross_check=None)
    tcorr = np.corrcoef(L.T)[np.triu_indices(len(TASKS), 1)].mean()
    print(f"\n  mean cross-task correlation, FULL {len(df)}-model board: {tcorr:.2f}  (positive => good models win everywhere => mean & Borda agree => identified)")
    emit("llm.mean_cross_task_corr", float(tcorr),
         n_models=int(len(df)),
         spec=f"mean pairwise correlation across the 6 Open LLM Leaderboard tasks' loss columns, computed "
              f"on the FULL {len(df)}-model board (not the top-30 pool) -- positive means good models tend "
              "to win on every task (mean and Borda aggregation agree). This is the FULL-BOARD figure; "
              "llm.mean_cross_task_corr_top30 is the value that belongs to any claim about the top-30 pool",
         cites=["N2 body: the crowding finding"],
         cross_check=None)
    _top30_mat = L[order[:30]]
    _top30_corr_pairs = np.corrcoef(_top30_mat.T)[np.triu_indices(len(TASKS), 1)]
    tcorr_top30 = float(_top30_corr_pairs.mean())
    print(f"  mean cross-task correlation, TOP-30 pool only: {tcorr_top30:.4f}  (pairwise range {_top30_corr_pairs.min():.2f} to {_top30_corr_pairs.max():.2f})")
    emit("llm.mean_cross_task_corr_top30", round(tcorr_top30, 4), n_models=30,
         band=[round(float(_top30_corr_pairs.min()), 2), round(float(_top30_corr_pairs.max()), 2)],
         spec="mean pairwise correlation across the 6 Open LLM Leaderboard tasks' loss columns, computed "
              "on the ACTUAL top-30 candidate pool (K=30, the same Ld k*/winner_stability are computed on) "
              "-- the value the companion note's top-30 claim cites, distinct from "
              "llm.mean_cross_task_corr's full-board figure",
         cites=["N2 body: the finding"],
         cross_check=None)
    rng = np.random.default_rng(20260909)
    n_models_total = L.shape[0]
    if n_models_total < 30:
        raise ValueError(
            f"ml_witness range-restriction check needs >=30 complete-task models to draw a same-size "
            f"K=30 random pool for comparison against the top-30 candidate pool; only "
            f"{n_models_total} available. The top-30 k*/winner_stability exhibit above does not apply "
            f"at this scale either.")
    random30_corrs = []
    for _ in range(2000):
        idx = rng.choice(n_models_total, size=30, replace=False)
        pairs = np.corrcoef(L[idx].T)[np.triu_indices(len(TASKS), 1)]
        random30_corrs.append(float(pairs.mean()))
    random30_corrs = np.asarray(random30_corrs)
    print(f"\n  RANGE-RESTRICTION CHECK: random K=30 draws (not selected by mean score) give mean "
          f"cross-task correlation {random30_corrs.mean():.3f} (5-95%: [{np.percentile(random30_corrs,5):.3f}, "
          f"{np.percentile(random30_corrs,95):.3f}]) vs the real top-30's {tcorr_top30:.4f} and the "
          f"board-wide {tcorr:.3f}.")
    emit("llm.random30_cross_task_corr_mean", round(float(random30_corrs.mean()), 3), n_draws=2000,
         band=[round(float(np.percentile(random30_corrs, 5)), 3), round(float(np.percentile(random30_corrs, 95)), 3)],
         spec="mean pairwise cross-task correlation on RANDOM (not top-ranked) K=30 draws from the full "
              "leaderboard, 2000 draws -- the range-restriction null the top-30 pool's own correlation "
              "(llm.mean_cross_task_corr_top30) must be read against before treating it as an "
              "independent finding rather than a consequence of selecting on mean score",
         cites=["N2 body: the range-restriction check"],
         cross_check=None)
    flush()
    print("  DISCRIMINANT VALIDITY (not 'generality'): the crowding RISE is MECHANICAL (the null rises too), so it is NOT")
    print("  evidence by itself. The REAL leaderboard flips LESS than the independence null -> it is COMPARATIVELY IDENTIFIED,")
    print("  unlike the shock-prone macro panels. So the instrument DISTINGUISHES fragile (macro) from robust (this")
    print("  leaderboard) arenas. (Appendix witness only; Chatbot-Arena / Bradley-Terry deferred to arXiv 2508.11847.)")


if __name__ == "__main__":
    main()
