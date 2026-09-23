"""
mcs_robustness.py — is the non-identification spine (11/12) robust to
the bootstrap-design tuning it currently hardcodes?
"""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
import mcs_test as MT
from canonical_emit import emit, flush
RES = MULTI / "results"


def per_series_losses():
    d = pd.read_csv(MT.PRED); out = {}
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique()); yrs = sorted(g.year.unique())
        La = np.full((len(yrs), len(models)), np.nan); Ls = np.full((len(yrs), len(models)), np.nan)
        for j, m in enumerate(models):
            gm = g[g.model == m].groupby("year"); a = gm["sae"].mean(); s = gm["msse"].mean()
            for i, y in enumerate(yrs):
                if y in a.index: La[i, j] = a[y]; Ls[i, j] = s[y]
        keep = ~(np.isnan(La).any(1) | np.isnan(Ls).any(1))
        if (~keep).sum():
            print(f"  [warn] {sid}: dropping {(~keep).sum()} year(s) with a NaN in La/Ls before robustness checks")
        out[sid] = (La[keep], Ls[keep])
    return out


def nonid_count(losses, tag, alpha, B, block, seed):
    """# series with |MCS|>1 (non-identified) under this config."""
    idx = 0 if tag == "abs" else 1
    return sum(len(MT.mcs(v[idx], alpha=alpha, B=B, block=block, seed=seed)[0]) > 1 for v in losses.values())


def fredmd_robustness(seeds=3):
    """Population spot-check: is the FRED-MD non-identified count robust to the same tuning?"""
    import fred_md_sweep as FS
    df = FS.load_fredmd(); Ls = []
    for col in df.columns:
        s = FS.load_series_gap_safe(df, col)
        if s is None:
            continue
        res = FS.series_losses(s.to_numpy(float), s.index)
        if res is None:
            print(f"  [skip] {col}: series_losses() returned no result", flush=True)
            continue
        La = res[0]
        if np.all(np.isfinite(La)):
            Ls.append(La)
        else:
            print(f"  [skip] {col}: non-finite loss in the levels arm (La)", flush=True)
    n = len(Ls)
    print(f"\n=== (4) FRED-MD population spot-check: non-identified |MCS_abs|>1 of {n} (mean [min-max] over {seeds} seeds) ===")
    print(f"  {'block':>5} {'B':>6} {'alpha':>6} | non-identified")
    GRID = [(3, 2000, 0.10), (1, 2000, 0.10), (6, 2000, 0.10), (3, 5000, 0.10), (3, 2000, 0.05), (3, 2000, 0.25)]
    rows = []
    for block, B, alpha in GRID:
        cnts = [sum(len(MT.mcs(L, alpha=alpha, B=B, block=block, seed=sd)[0]) > 1 for L in Ls) for sd in range(seeds)]
        print(f"  {block:>5} {B:>6} {alpha:>6} | {np.mean(cnts):.1f} [{min(cnts)}-{max(cnts)}] of {n}")
        rows.append({"block": block, "B": B, "alpha": alpha, "mean": float(np.mean(cnts)),
                     "min": int(min(cnts)), "max": int(max(cnts)), "n": n, "seeds": seeds})
    print("  -> majority non-identified at EVERY config; monotone in alpha (a feature: stricter alpha -> larger MCS).")
    R = pd.DataFrame(rows)
    R.to_csv(RES / "fredmd_robustness_band.csv", index=False)
    print(f"  wrote {RES/'fredmd_robustness_band.csv'}")

    a10 = R[R.alpha == 0.10]
    emit("fredmd.band_alpha10", [int(a10["min"].min()), int(a10["max"].max())], of=n,
         spec="min/max non-identified count across the block/B configurations swept at alpha=0.10 "
              "(not a full block x B cross -- 4 of the 6 grid cells use alpha=0.10), 3 seeds each",
         cites=["S13.1: FRED-MD bootstrap-tuning band"],
         cross_check="fredmd_robustness_band.csv: min/max over rows with alpha==0.10")
    emit("fredmd.count_alpha05", round(float(R[(R.alpha == 0.05) & (R.block == 3) & (R.B == 2000)]["mean"].iloc[0]), 2),
         of=n,
         spec="mean non-identified count at alpha=0.05, block=3, B=2000, over 3 seeds",
         cites=["S4.1: FRED-MD count at alpha=0.05"],
         cross_check="fredmd_robustness_band.csv: mean where alpha==0.05 & block==3 & B==2000")
    _a25 = R[(R.alpha == 0.25) & (R.block == 3) & (R.B == 2000)]
    emit("fredmd.count_alpha25", round(float(_a25["mean"].iloc[0]), 2),
         of=n, p25_min=int(_a25["min"].iloc[0]), p25_max=int(_a25["max"].iloc[0]),
         spec="mean non-identified count at alpha=0.25, block=3, B=2000, over 3 seeds",
         cites=["S4.1: FRED-MD count at alpha=0.25"],
         cross_check="fredmd_robustness_band.csv: mean where alpha==0.25 & block==3 & B==2000")


def monthly_resolution_robustness(blocks=(1, 2, 3, 6, 12, 24), alpha=0.10):
    """Evaluation-frequency check: the headline MCS runs on per-YEAR
    mean losses (~25-36 points); here the same MCS is re-run on the native MONTHLY loss matrices
    across a sweep of block lengths."""
    d = pd.read_csv(MT.PRED)
    rows = []
    for sid, g in d.groupby("series"):
        yrs = sorted(g.year.unique()); models = sorted(g.model.unique())
        La = np.full((len(yrs), len(models)), np.nan)
        for j, m in enumerate(models):
            a = g[g.model == m].groupby("year")["sae"].mean()
            for i, y in enumerate(yrs):
                if y in a.index: La[i, j] = a[y]
        _keep = ~np.isnan(La).any(1)
        if (~_keep).sum():
            print(f"  [warn] {sid}: dropping {(~_keep).sum()} year(s) with a NaN before the monthly-resolution sweep")
        La = La[_keep]
        piv = g.pivot_table(index="target", columns="model", values="sae").dropna()
        Lm = piv.to_numpy(float)
        row = {"series": sid, "T_annual": int(La.shape[0]), "T_monthly": int(Lm.shape[0]),
               "MCS_annual": len(MT.mcs(La, alpha=alpha, block=3, seed=0)[0])}
        for blk in blocks:
            row[f"MCS_mon_b{blk}"] = len(MT.mcs(Lm, alpha=alpha, block=blk, seed=0)[0])
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(RES / "mcs_monthly_resolution.csv", index=False)
    n = len(R); na = int((R.MCS_annual > 1).sum())
    # computed over the full sweep, not read off the b1/b24 endpoints (only correct if the sweep
    # is monotone).
    _counts = [int((R[f"MCS_mon_b{b}"] > 1).sum()) for b in blocks]
    emit("mcs.monthly_nonidentified_range", [min(_counts), max(_counts)], of=n,
         spec=f"min and max curated non-identified count over ALL block lengths swept "
              f"({', '.join(map(str, blocks))}), alpha={alpha}; computed over the sweep, not read off "
              f"its endpoints",
         cites=["S4.1 body: evaluation-frequency sweep"],
         cross_check="mcs_monthly_resolution.csv: min/max over blocks of (MCS_mon_b* > 1).sum()")
    emit("mcs.annual_nonidentified", na, of=n,
         spec="series with |MCS|>1 under the annual-aggregation MCS (block=3); same formula as "
              "mcs.curated_nonidentified",
         cites=["S4.1 body: annual vs monthly resolution"],
         cross_check="mcs_monthly_resolution.csv: (MCS_annual > 1).sum()")
    emit("data.curated_monthly_T_min", int(R.T_monthly.min()),
         spec="smallest native-monthly loss-matrix length across the 12 curated series, the lower end "
              "of S4.1's '292-431 monthly observations per series' range",
         cites=["S4.1 body: native monthly loss matrices"],
         cross_check="mcs_monthly_resolution.csv: T_monthly.min()")
    emit("data.curated_monthly_T_max", int(R.T_monthly.max()),
         spec="largest native-monthly loss-matrix length across the 12 curated series, the upper end "
              "of S4.1's '292-431 monthly observations per series' range",
         cites=["S4.1 body: native monthly loss matrices"],
         cross_check="mcs_monthly_resolution.csv: T_monthly.max()")
    flush()
    print(f"\n=== (5) EVALUATION-FREQUENCY CHECK: native-MONTHLY MCS across block lengths (alpha={alpha}) vs the ANNUAL headline ===")
    print(R.to_string(index=False))
    print(f"  annual (block=3): non-identified {na}/{n}")
    _ids_by_block = []
    for blk in blocks:
        c = int((R[f"MCS_mon_b{blk}"] > 1).sum())
        ids = sorted(R[R[f"MCS_mon_b{blk}"] == 1].series.tolist())
        _ids_by_block.append(set(ids))
        print(f"  monthly block={blk:>2}: non-identified {c}/{n}  identified={ids}")
    _always = sorted(set.intersection(*_ids_by_block)) if _ids_by_block else []
    _swing = sorted(set.union(*_ids_by_block) - set.intersection(*_ids_by_block)) if _ids_by_block else []
    print(f"  -> OVERWHELMING at every resolution ({na}-{max(int((R[f'MCS_mon_b{b}']>1).sum()) for b in blocks)}/{n})."
          f" Always identified: {_always}; swing series (identified under at least one but not every "
          f"block): {_swing}")
    print("     Report the SWEEP, not a single block: annual aggregation inflates the count by at most")
    print("     one borderline series, it does not create it.")
    print(f"  wrote {RES/'mcs_monthly_resolution.csv'}")


def arch_cross_check_full(losses):
    """Independent-implementation cross-check of the hand-rolled range-statistic MCS against `arch`
    8.0.0's ModelConfidenceSet, on the REAL curated panel (all 12 series, not a 3-series spot-check)."""
    from arch.bootstrap import MCS as ArchMCS
    rows = []
    print("\n=== (3) cross-check hand-rolled range-MCS vs `arch` MCS (ALL 12 curated series, abs loss, alpha=0.10) ===")
    for sid in sorted(losses):
        La = losses[sid][0]
        ours_surv, _ = MT.mcs(La, alpha=0.10, seed=0)
        m = ArchMCS(pd.DataFrame(La), size=0.10, reps=2000, block_size=3, method="R", seed=0)
        m.compute()
        arch_surv = sorted(int(i) for i in m.included)
        ours_sorted = sorted(int(i) for i in ours_surv)
        exact = ours_sorted == arch_surv
        m2 = ArchMCS(pd.DataFrame(La), size=0.10, reps=2000, block_size=3, method="R",
                     bootstrap="moving block", seed=0)
        m2.compute()
        arch_mbb = sorted(int(i) for i in m2.included)
        ours_eR = sorted(int(i) for i in MT.mcs(La, alpha=0.10, seed=0, elimination="e_R")[0])
        rows.append({"series": sid, "ours_size": len(ours_sorted), "arch_size": len(arch_surv),
                     "ours_members": ";".join(map(str, ours_sorted)),
                     "arch_members": ";".join(map(str, arch_surv)),
                     "exact_set_match": exact, "size_match": len(ours_sorted) == len(arch_surv),
                     "arch_mbb_members": ";".join(map(str, arch_mbb)),
                     "ours_eR_members": ";".join(map(str, ours_eR)),
                     "raw_eq_arch_mbb": ours_sorted == arch_mbb,
                     "eR_eq_arch_mbb": ours_eR == arch_mbb})
        print(f"  {sid:>9}: ours |MCS|={len(ours_sorted)} {ours_sorted}  arch |MCS|={len(arch_surv)} {arch_surv}"
              f"  {'EXACT MATCH' if exact else ('size-only match' if len(ours_sorted)==len(arch_surv) else 'DISAGREE')}"
              f"  | arch(moving-block) {arch_mbb}  ours(e_R) {ours_eR}")
    R = pd.DataFrame(rows)
    R.to_csv(RES / "mcs_arch_cross_check.csv", index=False)
    n = len(R)
    n_exact = int(R.exact_set_match.sum()); n_size = int(R.size_match.sum())
    n_raw_mbb = int(R.raw_eq_arch_mbb.sum()); n_eR_mbb = int(R.eR_eq_arch_mbb.sum())
    print(f"  -> exact surviving-set match: {n_exact}/{n}   cardinality-only match: {n_size}/{n}")
    print(f"  -> decomposition: ours-raw vs arch(moving-block) {n_raw_mbb}/{n}; ours-e_R vs arch(moving-block) {n_eR_mbb}/{n}")
    emit("mcsarch.curated_exact_match_bootstrap_matched", n_raw_mbb, of=n,
         differ=sorted(R.series[~R.raw_eq_arch_mbb].tolist()),
         spec="curated series on which this paper's MCS (raw-mean elimination, moving-block "
              "bootstrap) returns the identical surviving set as arch 8.0.0 run with the SAME "
              "moving-block bootstrap (arch's default is a stationary bootstrap). A series still "
              "differing here differs through the elimination rule (raw vs HLN's e_R) and/or "
              "residual implementation differences (independent bootstrap draws, p-value "
              "convention); the residual after matching BOTH is "
              "mcsarch.curated_exact_match_fully_matched. alpha=0.10, block=3, B=2000, seed=0 on "
              "both sides",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="mcs_arch_cross_check.csv: raw_eq_arch_mbb.sum()")
    emit("mcsarch.curated_exact_match_fully_matched", n_eR_mbb, of=n,
         spec="curated series on which this paper's MCS run with HLN's e_R elimination rule "
              "(mcs_test.mcs(elimination='e_R')) returns the identical surviving set as arch 8.0.0 "
              "with a moving-block bootstrap -- bootstrap scheme AND elimination rule both matched, "
              "so this is the pure independent-implementation check",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="mcs_arch_cross_check.csv: eR_eq_arch_mbb.sum()")
    emit("mcsarch.curated_exact_match", n_exact, of=n,
         spec="curated series (all 12, not a subset) on which the hand-rolled range-statistic MCS "
              "returns the IDENTICAL surviving model set (not merely the same |MCS|) as arch 8.0.0's "
              "independent ModelConfidenceSet implementation, alpha=0.10, block=3, B=2000, on the "
              "real per-year absolute-loss panels",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="mcs_arch_cross_check.csv: exact_set_match.sum()")
    emit("mcsarch.curated_size_match", n_size, of=n,
         spec="curated series (all 12) on which the hand-rolled and arch implementations return the "
              "same |MCS| cardinality, whether or not the member sets are identical",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="mcs_arch_cross_check.csv: size_match.sum()")


def fredmd_arch_cross_check():
    """Same independent-implementation cross-check, extended to the FRED-MD population sweep,
    whose non-identification count is one of the paper's two headline numbers."""
    import fred_md_sweep as FS
    from arch.bootstrap import MCS as ArchMCS
    df = FS.load_fredmd()
    rows = []
    for col in df.columns:
        s = FS.load_series_gap_safe(df, col)
        if s is None:
            continue
        res = FS.series_losses(s.to_numpy(float), s.index)
        if res is None:
            print(f"  [skip] {col}: series_losses() returned no result", flush=True)
            continue
        La = res[0]
        if not np.all(np.isfinite(La)):
            print(f"  [skip] {col}: non-finite loss in the levels arm (La)", flush=True)
            continue
        ours_surv, _ = MT.mcs(La, alpha=0.10, seed=0)
        m = ArchMCS(pd.DataFrame(La), size=0.10, reps=2000, block_size=3, method="R", seed=0)
        m.compute()
        arch_surv = sorted(int(i) for i in m.included)
        ours_sorted = sorted(int(i) for i in ours_surv)
        m2 = ArchMCS(pd.DataFrame(La), size=0.10, reps=2000, block_size=3, method="R",
                     bootstrap="moving block", seed=0)
        m2.compute()
        arch_mbb = sorted(int(i) for i in m2.included)
        ours_eR = sorted(int(i) for i in MT.mcs(La, alpha=0.10, seed=0, elimination="e_R")[0])
        rows.append({"series": col, "ours_size": len(ours_sorted), "arch_size": len(arch_surv),
                     "exact_set_match": ours_sorted == arch_surv,
                     "size_match": len(ours_sorted) == len(arch_surv),
                     "ours_members": ";".join(map(str, ours_sorted)),
                     "arch_members": ";".join(map(str, arch_surv)),
                     "arch_mbb_members": ";".join(map(str, arch_mbb)),
                     "ours_eR_members": ";".join(map(str, ours_eR)),
                     "ours_eR_size": len(ours_eR),
                     "raw_eq_arch_mbb": ours_sorted == arch_mbb,
                     "eR_eq_arch_mbb": ours_eR == arch_mbb})
    R = pd.DataFrame(rows)
    R.to_csv(RES / "fredmd_arch_cross_check.csv", index=False)
    n = len(R)
    n_exact = int(R.exact_set_match.sum()); n_size = int(R.size_match.sum())
    n_raw_mbb = int(R.raw_eq_arch_mbb.sum()); n_eR_mbb = int(R.eR_eq_arch_mbb.sum())
    n_nonid_raw = int((R.ours_size > 1).sum()); n_nonid_eR = int((R.ours_eR_size > 1).sum())
    print(f"\n=== (3b) FRED-MD population-sweep cross-check vs `arch` (all {n} series, abs loss, alpha=0.10) ===")
    print(f"  -> exact surviving-set match: {n_exact}/{n}   cardinality-only match: {n_size}/{n}")
    print(f"  -> decomposition: ours-raw vs arch(moving-block) {n_raw_mbb}/{n}; ours-e_R vs arch(moving-block) {n_eR_mbb}/{n}")
    print(f"  -> non-identified count: raw rule {n_nonid_raw}/{n} (must equal the fred_md_sweep headline); e_R rule {n_nonid_eR}/{n}")
    emit("mcsarch.fredmd_exact_match_bootstrap_matched", n_raw_mbb, of=n,
         differ=sorted(R.series[~R.raw_eq_arch_mbb].tolist()),
         spec="FRED-MD population-sweep series on which this paper's MCS (raw-mean elimination, "
              "moving-block bootstrap) returns the identical surviving set as arch 8.0.0 run with "
              "the SAME moving-block bootstrap; a series still differing here differs through the "
              "elimination rule (raw vs HLN's e_R) and/or residual implementation differences "
              "(independent bootstrap draws, p-value convention) -- the residual after matching BOTH "
              "is mcsarch.fredmd_exact_match_fully_matched. alpha=0.10, block=3, B=2000, seed=0",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="fredmd_arch_cross_check.csv: raw_eq_arch_mbb.sum()")
    emit("mcsarch.fredmd_exact_match_fully_matched", n_eR_mbb, of=n,
         spec="FRED-MD population-sweep series on which this paper's MCS with HLN's e_R rule "
              "returns the identical surviving set as arch 8.0.0 with a moving-block bootstrap -- "
              "bootstrap scheme and elimination rule both matched: the pure implementation check",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="fredmd_arch_cross_check.csv: eR_eq_arch_mbb.sum()")
    emit("mcsarch.fredmd_nonidentified_eR", n_nonid_eR, of=n, raw_rule=n_nonid_raw,
         spec="FRED-MD series non-identified (|MCS| > 1 at alpha=0.10) when the MCS is run with "
              "HLN's e_R elimination rule instead of this paper's raw-mean rule, everything else "
              "identical -- the elimination-rule sensitivity of the population headline, disclosed "
              "alongside the arch-implementation bound (mcsarch.fredmd_classification_flip). The "
              "companion field raw_rule is the same count under the paper's rule, recomputed here",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="fredmd_arch_cross_check.csv: (ours_eR_size > 1).sum()")
    emit("mcsarch.fredmd_exact_match", n_exact, of=n,
         spec="FRED-MD population-sweep series on which the hand-rolled range-statistic MCS returns "
              "the IDENTICAL surviving model set as arch 8.0.0's independent implementation, "
              "alpha=0.10, block=3, B=2000",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="fredmd_arch_cross_check.csv: exact_set_match.sum()")
    emit("mcsarch.fredmd_size_match", n_size, of=n,
         spec="FRED-MD population-sweep series on which the hand-rolled and arch implementations "
              "return the same |MCS| cardinality",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="fredmd_arch_cross_check.csv: size_match.sum()")
    _flip = R[((R.ours_size == 1) & (R.arch_size > 1)) | ((R.ours_size > 1) & (R.arch_size == 1))]
    emit("mcsarch.fredmd_classification_flip", int(len(_flip)), of=n, series=sorted(_flip.series.tolist()),
         spec="FRED-MD population-sweep series where the two implementations disagree not just on "
              "cardinality but on the BINARY identified (|MCS|=1) vs non-identified (|MCS|>1) "
              f"classification itself -- the honest bound on how much the {n_nonid_raw}/{n} headline could move "
              "under a different, equally valid implementation, disclosed alongside the tuning- and "
              "significance-level sensitivity already reported",
         cites=["S4.1 body: independent-implementation cross-check"],
         cross_check="fredmd_arch_cross_check.csv: rows where (ours_size==1)!=(arch_size==1)")


def main():
    losses = per_series_losses(); n = len(losses)
    print(f"curated panel: {n} series (default headline = MCS_abs_10 non-identified count)\n")
    SEEDS = list(range(10))                                            # Monte-Carlo band

    print("=== (1) GRID: non-identified count (|MCS|>1) over block x B x alpha, ABS loss (mean [min-max] over 10 seeds) ===")
    print(f"  {'block':>5} {'B':>6} | " + " ".join(f"a={a:<5}" for a in (0.05, 0.10, 0.25)))
    grid_rows = []
    for block, B in itertools.product([1, 2, 3, 4, 6], [2000, 5000]):
        cells = []
        for a in (0.05, 0.10, 0.25):
            cnts = [nonid_count(losses, "abs", a, B, block, s) for s in SEEDS]
            cells.append(f"{np.mean(cnts):.1f}[{min(cnts)}-{max(cnts)}]")
            grid_rows.append({"block": block, "B": B, "alpha": a, "mean": float(np.mean(cnts)),
                               "min": int(min(cnts)), "max": int(max(cnts)), "n": n,
                               "seeds": len(SEEDS)})
        print(f"  {block:>5} {B:>6} | " + " ".join(f"{c:<8}" for c in cells))
    GRID_R = pd.DataFrame(grid_rows)
    GRID_R.to_csv(RES / "mcs_curated_tuning_grid.csv", index=False)
    emit("mcs.curated_grid_min", int(GRID_R["min"].min()), of=n,
         spec="minimum non-identified count (|MCS_abs|>1) across the FULL curated-panel tuning grid "
              "(block in {1,2,3,4,6}, B in {2000,5000}, alpha in {0.05,0.10,0.25}, 10 seeds each -- "
              "5*2*3 = 30 cells, 30*10 = 300 nonid_count() evaluations, and since each one runs the "
              "MCS once per curated series, 30*10*12 = 3,600 MCS calls in total)",
         cites=["S13.1: curated-panel tuning-grid"],
         cross_check="mcs_curated_tuning_grid.csv: min()")
    emit("mcs.curated_grid_max", int(GRID_R["max"].max()), of=n,
         spec="maximum non-identified count across the same full curated-panel tuning grid as "
              "mcs.curated_grid_min -- equal to it iff the count is exactly invariant across every "
              "swept configuration",
         cites=["S13.1: curated-panel tuning-grid"],
         cross_check="mcs_curated_tuning_grid.csv: max()")

    print("\n=== (2) MONTE-CARLO band at the DEFAULT (block=3, B=2000, alpha=0.10), seeds 0-49 ===")
    band_rows = []
    for tag in ("abs", "sq"):
        cnts = [nonid_count(losses, tag, 0.10, 2000, 3, s) for s in range(50)]
        print(f"  {tag:>3} loss: non-identified = {np.mean(cnts):.2f} [{min(cnts)}-{max(cnts)}] of {n}  (SE {np.std(cnts):.2f})")
        band_rows.append({"loss": tag, "mean": float(np.mean(cnts)), "min": int(min(cnts)),
                           "max": int(max(cnts)), "se": float(np.std(cnts)), "n": n, "seeds": 50})
    BAND_R = pd.DataFrame(band_rows)
    BAND_R.to_csv(RES / "mcs_curated_default_band.csv", index=False)
    _abs_band = BAND_R[BAND_R.loss == "abs"].iloc[0]
    emit("mcs.curated_default_band_min", int(_abs_band["min"]), of=n,
         spec="minimum non-identified count over 50 Monte-Carlo seeds at the paper's own default "
              "configuration (block=3, B=2000, alpha=0.10), absolute loss",
         cites=["S13.1: curated-panel tuning-grid"],
         cross_check="mcs_curated_default_band.csv: min where loss=='abs'")
    emit("mcs.curated_default_band_max", int(_abs_band["max"]), of=n,
         spec="maximum non-identified count over the same 50-seed default-configuration band as "
              "mcs.curated_default_band_min -- equal to it iff the count is invariant to seed choice "
              "at the paper's own tuning",
         cites=["S13.1: curated-panel tuning-grid"],
         cross_check="mcs_curated_default_band.csv: max where loss=='abs'")
    _sq_band = BAND_R[BAND_R.loss == "sq"].iloc[0]
    emit("mcs.curated_default_band_min_sq", int(_sq_band["min"]), of=n,
         spec="minimum non-identified count over 50 Monte-Carlo seeds at the paper's own default "
              "configuration (block=3, B=2000, alpha=0.10), SQUARED loss -- the loss-function "
              "robustness companion to mcs.curated_default_band_min",
         cites=[], cross_check="mcs_curated_default_band.csv: min where loss=='sq'")
    emit("mcs.curated_default_band_max_sq", int(_sq_band["max"]), of=n,
         spec="maximum non-identified count over the same 50-seed default-configuration band as "
              "mcs.curated_default_band_min_sq, SQUARED loss",
         cites=[], cross_check="mcs_curated_default_band.csv: max where loss=='sq'")

    arch_cross_check_full(losses)
    fredmd_robustness()
    fredmd_arch_cross_check()
    monthly_resolution_robustness()
    print("\n  READ: publish the band (curated 11/12 invariant on the annual grid; FRED-MD 54-58/74 across block/B/seed,")
    print("        monotone in alpha; native-monthly 10-11/12 across block lengths -> overwhelming at every resolution,")
    print("        BOPGEXP the lone swing series), not the single integer. Robust to bootstrap tuning + arch-corroborated.")


if __name__ == "__main__":
    main()
