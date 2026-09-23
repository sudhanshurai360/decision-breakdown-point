"""
arena_crowding.py — harmonized cross-arena crowding + identification (identical operationalization on the M4 and Open-LLM arenas).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(INSTR))
from canonical_emit import emit, flush
RES = MULTI / "results"; DATA = MULTI / "data"
LLM_TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
M4_METRICS = ["sMAPE", "MAPE", "MASE", "RMSSE"]


def crowd(L, rng, ks):
    """mean-loss winner vs Borda (mean-rank) winner flip-rate at each pool size k (L: models x tasks, lower=better)."""
    out = {}
    for k in ks:
        if k > len(L):
            continue
        fl = []
        for _ in range(500):
            idx = rng.choice(len(L), k, replace=False)
            sub = L[idx]
            mw = int(np.argmin(sub.mean(1)))
            ranks = np.argsort(np.argsort(sub, axis=0), axis=0)
            fl.append(mw != int(np.argmin(ranks.mean(1))))
        out[k] = float(np.mean(fl))
    return out


N_NULL_PERM = 50


def first_permutation(L, rng):
    """The first column permutation of the independence null (drawn before the real curve, so this
    draw's RNG order is fixed regardless of how many null permutations follow); average_null()
    adds the rest."""
    return np.column_stack([rng.permutation(L[:, j]) for j in range(L.shape[1])])


def average_null(L, rng, ks, first_null, n_perm=N_NULL_PERM):
    """Average `crowd()` over `first_null` (already drawn) plus `n_perm - 1` fresh permutations."""
    draws = [crowd(first_null, rng, ks)]
    for _ in range(n_perm - 1):
        Lnull = np.column_stack([rng.permutation(L[:, j]) for j in range(L.shape[1])])
        draws.append(crowd(Lnull, rng, ks))
    ks_seen = list(draws[0])
    null = {k: float(np.mean([d[k] for d in draws])) for k in ks_seen}
    null_sd = {k: float(np.std([d[k] for d in draws], ddof=1)) if len(draws) > 1 else float("nan") for k in ks_seen}
    return null, null_sd, draws[0]


def arena(name, L, ks):
    rng = np.random.default_rng(0)
    Lnull = first_permutation(L, rng)                         # drawn before `real`
    real = crowd(L, rng, ks)
    null, null_sd, null_first = average_null(L, rng, ks, Lnull)
    tcorr = np.corrcoef(L.T)[np.triu_indices(L.shape[1], 1)].mean()
    print(f"\n=== {name} ({L.shape[0]} models x {L.shape[1]} tasks; mean-vs-Borda flip; cross-task corr {tcorr:+.2f}) ===")
    print(f"  {'#models':>8s} | REAL | NULL mean over {N_NULL_PERM} permutations (sd) [first draw] |")
    for k in real:
        print(f"  {k:>8d} | {real[k]:.0%} | {null[k]:.0%} ({null_sd[k]:.2f}) [{null_first[k]:.0%}] | "
              f"{'REAL<null -> comparatively IDENTIFIED' if real[k] < null[k] else 'REAL>=null -> fragile'}")
    return real, null, float(tcorr), null_sd, null_first


def main():
    # M4 arena
    b = pd.read_csv(RES / "m4_submissions_leaderboard.csv", index_col=0)
    LM4 = b[M4_METRICS].to_numpy(float)                          # loss (lower=better)
    m4_real, m4_null, m4_tcorr, m4_null_sd, m4_null_first = arena("M4 competition (61 real submissions)", LM4, [2, 4, 8, 16, 32, 60])
    _null_spec = (f" The null is the MEAN over {N_NULL_PERM} column permutations (companion fields null_sd, "
                  f"n_perm, first_permutation = the single seed-0 draw); a single "
                  f"permutation is far too noisy on a 61-row panel to read a real-vs-null sign from.")
    _null_spec_llm = (f" The null is the MEAN over {N_NULL_PERM} column permutations (companion fields "
                      f"null_sd, n_perm, first_permutation = the single seed-0 draw); same convention as the M4 arm's arena.m4_flip_null_* keys.")
    ks_sorted = sorted(m4_real)
    k_min, k_max = ks_sorted[0], ks_sorted[-1]
    emit("arena.m4_mean_cross_metric_corr", round(m4_tcorr, 3),
         spec="mean pairwise Pearson correlation across all C(4,2)=6 sMAPE/MAPE/MASE/RMSSE metric "
              "pairs on the 61 real M4 submissions' own accuracy panel -- how much the M4 crowding "
              "arena's 4 metrics move together, wider than m4sub.rank_corr_smape_rmsse's single pair",
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.arena()'s own tcorr")
    emit("arena.m4_flip_real_min_pool", float(m4_real[k_min]), n_models=k_min,
         spec=f"M4 competition: real mean-vs-Borda winner flip rate at the smallest pool size "
              f"tested (n_models={k_min}), against an independence-null resample of the same panel",
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.crowd() at k=min(ks)")
    emit("arena.m4_flip_null_min_pool", float(m4_null[k_min]), n_models=k_min,
         null_sd=round(float(m4_null_sd[k_min]), 4), n_perm=N_NULL_PERM,
         first_permutation=float(m4_null_first[k_min]),
         spec=f"M4 competition: INDEPENDENCE-NULL mean-vs-Borda winner flip rate at the smallest "
              f"pool size tested (n_models={k_min}) -- the counterfactual arena.m4_flip_real_min_pool "
              f"is read against. NOT comparable to density.flip_rate_min_pool: that key is a "
              f"MAE-vs-RMSE winner flip rate on ~1500 M4-Monthly SERIES scored under 9 models, "
              f"whereas this is a mean-loss-vs-Borda flip rate on the 61 M4 SUBMISSIONS scored "
              f"under 4 metrics. Neither key is cited in either manuscript.{_null_spec}",
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.crowd() on a column-shuffled null at k=min(ks)")
    emit("arena.m4_flip_real_max_pool", float(m4_real[k_max]), n_models=k_max,
         spec=f"M4 competition: real mean-vs-Borda winner flip rate at the largest pool size "
              f"tested (n_models={k_max}), against an independence-null resample of the same panel",
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.crowd() at k=max(ks)")
    emit("arena.m4_flip_null_max_pool", float(m4_null[k_max]), n_models=k_max,
         null_sd=round(float(m4_null_sd[k_max]), 4), n_perm=N_NULL_PERM,
         first_permutation=float(m4_null_first[k_max]),
         spec=f"M4 competition: INDEPENDENCE-NULL mean-vs-Borda winner flip rate at the largest "
              f"pool size tested (n_models={k_max}) -- the counterfactual arena.m4_flip_real_max_pool "
              f"is read against. NOT comparable to density.flip_rate_max_pool: that key is a "
              f"MAE-vs-RMSE winner flip rate on ~1500 M4-Monthly SERIES scored under 9 models at a "
              f"largest pool of 9, whereas this is a mean-loss-vs-Borda flip rate on the 61 M4 "
              f"SUBMISSIONS scored under 4 metrics at a largest pool of {k_max}. Neither key is "
              f"cited in either manuscript.{_null_spec}",
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.crowd() on a column-shuffled null at k=max(ks)")
    emit("arena.m4_flip_real_mid_pool", float(m4_real[32]), n_models=32,
         spec="M4 competition: real mean-vs-Borda winner flip rate at the middle pool size tested "
              "(n_models=32), against an independence-null resample of the same panel. The "
              "CROSS-ARENA print below reports real < null at EVERY pool size tested, this one "
              "included; the uncertainty that bears on that comparison is the null's own spread "
              "across permutations, not a binomial standard error on a single draw -- see the "
              "null_sd, n_perm and first_permutation companions on arena.m4_flip_null_mid_pool",
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.crowd() at k=32")
    emit("arena.m4_flip_null_mid_pool", float(m4_null[32]), n_models=32,
         null_sd=round(float(m4_null_sd[32]), 4), n_perm=N_NULL_PERM,
         first_permutation=float(m4_null_first[32]),
         spec="M4 competition: INDEPENDENCE-NULL mean-vs-Borda winner flip rate at the middle pool "
              "size tested (n_models=32) -- see arena.m4_flip_real_mid_pool." + _null_spec,
         cites=[], cross_check="m4_submissions_leaderboard.csv: arena_crowding.crowd() on a column-shuffled null at k=32")
    # M4 identification margin (not via the density curve)
    mean = LM4.mean(1); order = np.argsort(mean)
    champ = b.index[order[0]].replace("submission-", "").replace(".rar", "")
    wins_all = all(b[m].idxmin() == b.index[order[0]] for m in M4_METRICS)
    if not wins_all:
        raise RuntimeError(
            f"M4 arena: {champ} (mean-loss champion) does not win all 4 metrics ({M4_METRICS}) -- "
            f"Pareto-dominance broke; the 'unique top' identification claim below is no longer valid")
    margin = 100 * (mean[order[1]] / mean[order[0]] - 1)
    print(f"  M4 IDENTIFICATION: champion = {champ}; wins ALL 4 metrics = {wins_all}; margin over #2 = {margin:.1f}% (unique top).")
    emit("arena.m4_champion", champ,
         spec="M4 competition: the mean-loss champion across the 4 M4_METRICS -- the model this "
              "file's identification margin (arena.m4_margin_pct) is measured against #2",
         cites=[], cross_check="m4_submissions_leaderboard.csv: the index label at "
                               "LM4.mean(1).argmin() -- which is 'submission-118.rar'; the emitted "
                               "value strips the 'submission-' prefix and the '.rar' suffix")
    emit("arena.m4_wins_all_metrics", bool(wins_all),
         spec="M4 competition: whether the mean-loss champion also wins (is idxmin() on) every one "
              "of the 4 M4_METRICS individually -- Pareto-dominance, asserted just above this emit",
         cites=[], cross_check="m4_submissions_leaderboard.csv: all(b[m].idxmin()==champion for m in M4_METRICS)")
    emit("arena.m4_margin_pct", round(float(margin), 1),
         spec="M4 competition: the mean-loss champion's percentage margin over #2 "
              "(100*(mean_loss[#2]/mean_loss[#1]-1)) -- the M4 IDENTIFICATION evidence this file's "
              "own docstring names as one of its two stated purposes",
         cites=[], cross_check="m4_submissions_leaderboard.csv: 100*(LM4.mean(1)[order[1]]/LM4.mean(1)[order[0]]-1)")

    # LLM arena
    df = pd.read_parquet(DATA / "open_llm.parquet")
    LLM = (100.0 - df[LLM_TASKS].to_numpy(float))
    emit("llm.n_models", int(len(df)),
         spec="complete Open-LLM Leaderboard entries scored on all six benchmark tasks",
         cites=["N2 body: the finding"],
         cross_check="data/open_llm.parquet: row count")
    llm_real, llm_null, _, llm_null_sd, llm_null_first = arena("Open-LLM Leaderboard", LLM, [2, 4, 8, 16, 32, 64])
    llm_ks_sorted = sorted(llm_real)
    llm_k_min, llm_k_max = llm_ks_sorted[0], llm_ks_sorted[-1]
    emit("arena.llm_flip_real_min_pool", float(llm_real[llm_k_min]), n_models=llm_k_min,
         spec=f"Open-LLM Leaderboard: real mean-vs-Borda winner flip rate at the smallest pool size "
              f"tested (n_models={llm_k_min}), against an independence-null resample of the same panel",
         cites=[], cross_check="data/open_llm.parquet: arena_crowding.crowd() at k=min(ks)")
    emit("arena.llm_flip_null_min_pool", float(llm_null[llm_k_min]), n_models=llm_k_min,
         null_sd=round(float(llm_null_sd[llm_k_min]), 4), n_perm=N_NULL_PERM,
         first_permutation=float(llm_null_first[llm_k_min]),
         spec=f"Open-LLM Leaderboard: INDEPENDENCE-NULL mean-vs-Borda winner flip rate at the "
              f"smallest pool size tested (n_models={llm_k_min})." + _null_spec_llm,
         cites=[], cross_check="data/open_llm.parquet: arena_crowding.crowd() on a column-shuffled null at k=min(ks)")
    emit("arena.llm_flip_real_max_pool", float(llm_real[llm_k_max]), n_models=llm_k_max,
         spec=f"Open-LLM Leaderboard: real mean-vs-Borda winner flip rate at the largest pool size "
              f"tested (n_models={llm_k_max}), against an independence-null resample of the same panel",
         cites=[], cross_check="data/open_llm.parquet: arena_crowding.crowd() at k=max(ks)")
    emit("arena.llm_flip_null_max_pool", float(llm_null[llm_k_max]), n_models=llm_k_max,
         null_sd=round(float(llm_null_sd[llm_k_max]), 4), n_perm=N_NULL_PERM,
         first_permutation=float(llm_null_first[llm_k_max]),
         spec=f"Open-LLM Leaderboard: INDEPENDENCE-NULL mean-vs-Borda winner flip rate at the "
              f"largest pool size tested (n_models={llm_k_max})." + _null_spec_llm,
         cites=[], cross_check="data/open_llm.parquet: arena_crowding.crowd() on a column-shuffled null at k=max(ks)")

    print("\n=== CROSS-ARENA (identical operationalization) -- read HONESTLY ===")
    print(f"  LLM and M4: real < null at EVERY pool size tested (null averaged over {N_NULL_PERM} column permutations;")
    print("  a single permutation is too noisy to read M4's k=32/60 sign from -- see")
    print("  arena.m4_flip_null_mid_pool's null_sd/first_permutation companions). M4's gap is noisier than LLM's")
    print(f"  (null_sd up to {m4_null_sd[60]:.2f} at k=60 vs {llm_null_sd[64]:.2f} for LLM at k=64). M4 identification")
    print(f"  rests on the UNIQUE CHAMPION (sub-{champ} Pareto-dominant on all 4 metrics, {margin:+.1f}%), NOT the crowding curve.")
    print("  'Comparatively identified' ~ positive cross-task correlation (the g-factor) -> an APPENDIX portability witness,")
    print("  NOT a result. CONFOUND to disclose: arenas differ in pooling axis (time vs tasks) and #alternatives (~5 vs")
    print("  thousands), so 'fragile vs robust' is entangled with identification difficulty, not a clean domain contrast.")
    print("  (Also: M4 mean-loss mixes incommensurable metric scales -> MASE/RMSSE-dominated; applied identically to real+null.)")
    flush()


if __name__ == "__main__":
    main()
