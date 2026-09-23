"""mcs_circular_check.py — does switching the MCS bootstrap from moving-block to circular close the
gap against `arch`, and does it move the headline non-identification counts?
"""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "instrument"))
import mcs_test as MT
import mcs_robustness as MR
import mcs_size as MS
from canonical_emit import emit, flush

RES = ROOT / "multiseries" / "results"
K_PAPER, T_PAPER, BLOCK_PAPER = MS.K_PAPER, MS.T_PAPER, MS.BLOCK_PAPER
N_SIZE = MS.N_SIZE
SEED = MS.SEED
null_panel = MS.null_panel


def size_rate(circular, n_rep=N_SIZE, seed=SEED):
    g = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_rep):
        surv, _p = MT.mcs(null_panel(g, K_PAPER, T_PAPER), alpha=0.10, B=800, block=BLOCK_PAPER,
                           seed=0, circular=circular)
        hits += int(len(surv) < K_PAPER)
    rate = hits / n_rep
    return rate, float(np.sqrt(rate * (1 - rate) / n_rep))


def three_series_cross_check(losses):
    """CARDINALITY *and* exact-set match, both implementations, on a 3-series subset."""
    from arch.bootstrap import MCS as ArchMCS
    rows = []
    for sid in list(losses)[:3]:
        La = losses[sid][0]
        mb_set = set(MT.mcs(La, alpha=0.10, seed=0, circular=False)[0])
        circ_set = set(MT.mcs(La, alpha=0.10, seed=0, circular=True)[0])
        m = ArchMCS(pd.DataFrame(La), size=0.10, reps=2000, block_size=3, method="R", seed=0)
        m.compute()
        arch_set = set(int(i) for i in m.included)
        ours_mb, ours_circ, n_arch = len(mb_set), len(circ_set), len(arch_set)
        rows.append({"series": sid, "ours_moving_block": ours_mb, "ours_circular": ours_circ,
                     "arch": n_arch, "exact_match_mb": mb_set == arch_set,
                     "exact_match_circ": circ_set == arch_set})
        print(f"  {sid:>9}: moving-block |MCS|={ours_mb}  circular |MCS|={ours_circ}  arch included={n_arch}"
              f"  exact_match(mb/circ)=({mb_set == arch_set}/{circ_set == arch_set})")
    return pd.DataFrame(rows)


def curated_band(losses, circular, seeds=range(10)):
    n = len(losses)
    cnts = [sum(len(MT.mcs(v[0], alpha=0.10, B=2000, block=BLOCK_PAPER, seed=s, circular=circular)[0]) > 1
                for v in losses.values()) for s in seeds]
    return np.mean(cnts), min(cnts), max(cnts), n


def main():
    print("=== (1) SIZE: empirical rejection rate on exactly-null panels, moving-block vs circular vs arch ===")
    rate_mb, se_mb = size_rate(circular=False)
    rate_circ, se_circ = size_rate(circular=True)
    from arch.bootstrap import MCS as ArchMCS
    g = np.random.default_rng(SEED)
    hits_arch = 0
    hits_arch_mbb = hits_arch_cbb = hits_ours_mb_here = hits_ours_circ_here = 0
    for _ in range(N_SIZE):
        L = null_panel(g, K_PAPER, T_PAPER)
        a = ArchMCS(pd.DataFrame(L), size=0.10, reps=800, block_size=BLOCK_PAPER, method="R", seed=0)
        a.compute()
        hits_arch += int(len(a.included) < K_PAPER)
        for kind, bucket in (("moving block", "mbb"), ("circular", "cbb")):
            a2 = ArchMCS(pd.DataFrame(L), size=0.10, reps=800, block_size=BLOCK_PAPER, method="R",
                         bootstrap=kind, seed=0)
            a2.compute()
            if bucket == "mbb":
                hits_arch_mbb += int(len(a2.included) < K_PAPER)
            else:
                hits_arch_cbb += int(len(a2.included) < K_PAPER)
        hits_ours_mb_here += int(len(MT.mcs(L, alpha=0.10, B=800, block=BLOCK_PAPER, seed=0, circular=False)[0]) < K_PAPER)
        hits_ours_circ_here += int(len(MT.mcs(L, alpha=0.10, B=800, block=BLOCK_PAPER, seed=0, circular=True)[0]) < K_PAPER)
    rate_arch = hits_arch / N_SIZE
    rate_arch_mbb = hits_arch_mbb / N_SIZE
    rate_arch_cbb = hits_arch_cbb / N_SIZE
    if hits_ours_mb_here / N_SIZE != rate_mb or hits_ours_circ_here / N_SIZE != rate_circ:
        raise RuntimeError("mcs_circular_check: the matched-scheme loop did not reproduce size_rate()'s "
                           "panels -- the two generators have diverged; the like-for-like gaps below would be "
                           "computed on different draws")
    gap_matched_mb = abs(rate_mb - rate_arch_mbb)
    gap_matched_circ = abs(rate_circ - rate_arch_cbb)
    print(f"  moving-block: {rate_mb:.4f} +/- {se_mb:.4f}")
    print(f"  circular    : {rate_circ:.4f} +/- {se_circ:.4f}")
    print(f"  arch        : {rate_arch:.4f}  (stationary default)   arch moving-block: {rate_arch_mbb:.4f}   arch circular: {rate_arch_cbb:.4f}")
    print(f"  moving-block gap to arch: {abs(rate_mb - rate_arch):.4f}   circular gap to arch: {abs(rate_circ - rate_arch):.4f}")
    print(f"  LIKE-FOR-LIKE (same scheme, same panels): moving-block gap {gap_matched_mb:.4f}   circular gap {gap_matched_circ:.4f}")

    _mcssize_frag = json.loads((RES / "canonical" / "mcs_size.json").read_text())
    rate_arch_canonical = float(_mcssize_frag["mcssize.size_arch"]["value"])
    gap_mb_canonical = abs(rate_arch_canonical - rate_mb)
    gap_circ_canonical = abs(rate_arch_canonical - rate_circ)
    gap_reduction = (gap_mb_canonical - gap_circ_canonical) / gap_mb_canonical
    print(f"  vs the published arch figure ({rate_arch_canonical:.3f}): moving-block gap {gap_mb_canonical:.4f}, "
          f"circular gap {gap_circ_canonical:.4f} ({gap_reduction:.1%} reduction)")

    print("\n=== (2) 3-series cross-check vs arch, moving-block vs circular ===")
    losses = MR.per_series_losses()
    cross = three_series_cross_check(losses)
    cross.to_csv(RES / "mcs_circular_cross_check.csv", index=False)
    n_match_mb = int((cross.ours_moving_block == cross.arch).sum())
    n_match_circ = int((cross.ours_circular == cross.arch).sum())
    n_exact_mb = int(cross.exact_match_mb.sum())
    n_exact_circ = int(cross.exact_match_circ.sum())

    print("\n=== (3) curated panel (12 series): non-identified count, moving-block vs circular, 10-seed band ===")
    mean_mb, lo_mb, hi_mb, n = curated_band(losses, circular=False)
    mean_circ, lo_circ, hi_circ, _ = curated_band(losses, circular=True)
    print(f"  moving-block: {mean_mb:.1f} [{lo_mb}-{hi_mb}] of {n}")
    print(f"  circular    : {mean_circ:.1f} [{lo_circ}-{hi_circ}] of {n}")

    pd.DataFrame([{
        "rate_moving_block": rate_mb, "rate_circular": rate_circ, "rate_arch": rate_arch,
        "gap_moving_block": abs(rate_mb - rate_arch), "gap_circular": abs(rate_circ - rate_arch),
        "rate_arch_canonical": rate_arch_canonical, "gap_moving_block_canonical": gap_mb_canonical,
        "gap_circular_canonical": gap_circ_canonical, "gap_reduction": gap_reduction,
        "rate_arch_moving_block": rate_arch_mbb, "rate_arch_circular": rate_arch_cbb,
        "gap_matched_moving_block": gap_matched_mb, "gap_matched_circular": gap_matched_circ,
        "three_series_match_moving_block": n_match_mb, "three_series_match_circular": n_match_circ,
        "three_series_exact_match_moving_block": n_exact_mb, "three_series_exact_match_circular": n_exact_circ,
        "curated_mean_moving_block": mean_mb, "curated_lo_moving_block": lo_mb, "curated_hi_moving_block": hi_mb,
        "curated_mean_circular": mean_circ, "curated_lo_circular": lo_circ, "curated_hi_circular": hi_circ,
        "curated_n": n,
    }]).to_csv(RES / "mcs_circular_check.csv", index=False)

    emit("mcssize.size_circular", round(float(rate_circ), 4), n_rep=N_SIZE, se=round(float(se_circ), 4),
         spec=f"empirical rejection rate on exactly-null panels at K={K_PAPER}, T={T_PAPER}, "
              f"block={BLOCK_PAPER}, alpha=0.10, with circular=True instead of the default "
              f"moving-block scheme -- the same measurement mcssize.size_at_nominal_10 makes, with "
              f"only the resampling scheme changed",
         cites=["S4.2 body: circular-bootstrap size check"],
         cross_check="mcs_circular_check.csv: rate_circular")
    emit("mcssize.arch_gap_moving_block", round(gap_mb_canonical, 4), rate_arch=rate_arch_canonical,
         spec="gap between the published arch figure (mcssize.size_arch) and our moving-block size "
              "(mcssize.size_at_nominal_10) -- the discrepancy S4.2's prose names before testing "
              "whether the bootstrap scheme explains it",
         cites=["S4.2 body: circular-bootstrap size check"],
         cross_check="mcs_circular_check.csv: gap_moving_block_canonical")
    emit("mcssize.arch_gap_circular", round(gap_circ_canonical, 4), rate_arch=rate_arch_canonical,
         reduction_pct=round(gap_reduction * 100, 1),
         spec="the same gap to the published arch figure, with circular resampling instead of "
              "moving-block -- narrower but not closed. NOTE: both this gap and "
              "mcssize.arch_gap_moving_block compare our scheme against arch's STATIONARY-bootstrap "
              "default, and against an arch rate measured on a different panel draw; the "
              "like-for-like gaps are mcssize.arch_gap_matched_*",
         cites=["S4.2 body: circular-bootstrap size check"],
         cross_check="mcs_circular_check.csv: gap_circular_canonical")
    emit("mcssize.arch_gap_matched_moving_block", round(float(gap_matched_mb), 4),
         rate_ours=round(float(rate_mb), 4), rate_arch=round(float(rate_arch_mbb), 4), n_panels=N_SIZE,
         spec="|our moving-block size - arch's size| when arch is ALSO run with a moving-block "
              f"bootstrap (bootstrap='moving block'), on the identical {N_SIZE} null panels, "
              f"K={K_PAPER}, T={T_PAPER}, block={BLOCK_PAPER}, B=800, alpha=0.10: the like-for-like "
              "implementation gap with the resampling scheme held fixed",
         cites=["S4.2 body: circular-bootstrap size check"],
         cross_check="mcs_circular_check.csv: gap_matched_moving_block")
    emit("mcssize.arch_gap_matched_circular", round(float(gap_matched_circ), 4),
         rate_ours=round(float(rate_circ), 4), rate_arch=round(float(rate_arch_cbb), 4), n_panels=N_SIZE,
         spec="|our circular size - arch's size| when arch is ALSO run with a circular block "
              "bootstrap (bootstrap='circular'), on the identical null panels -- the like-for-like "
              "gap for the circular scheme",
         cites=["S4.2 body: circular-bootstrap size check"],
         cross_check="mcs_circular_check.csv: gap_matched_circular")
    emit("mcs.circular_three_series_match", n_match_circ, of=3,
         spec="cardinality-only match on a 3-series subset (NOT the same check as S4.1's own "
              "arch cross-check, which runs on all 12 curated series -- see mcsarch.curated_exact_match "
              "and mcs_robustness.py's arch_cross_check_full()): how many of 3 series match arch's "
              f"|MCS| cardinality exactly under circular resampling (moving-block matches "
              f"{n_match_mb} of 3). See mcs.circular_three_series_exact_match for the stronger, "
              "surviving-set-identity version of this same comparison",
         cites=["S13.1: circular-bootstrap cross-check"],
         cross_check="mcs_circular_cross_check.csv: (ours_circular == arch).sum()")
    emit("mcs.circular_three_series_exact_match", n_exact_circ, of=3,
         moving_block=n_exact_mb,
         spec="of the same 3-series subset, how many now have an IDENTICAL surviving model set "
              "(not just equal cardinality) to arch's own, when OUR bootstrap uses a fixed-length "
              f"circular resampling scheme instead of moving-block (moving-block: {n_exact_mb} of 3) "
              "-- the exact-set-identity check mcs_robustness.py::arch_cross_check_full() already "
              "applies to the full 12-series panel, extended here to this file's own "
              "block-structure comparison",
         cites=[],
         cross_check="mcs_circular_cross_check.csv: exact_match_circ.sum()")
    emit("mcs.curated_nonidentified_circular", int(round(mean_circ)), of=n,
         band=[int(lo_circ), int(hi_circ)],
         spec=f"mean non-identified count on the curated panel at the default configuration "
              f"(block={BLOCK_PAPER}, B=2000, alpha=0.10) over a 10-seed band (this file's own "
              f"seeds=range(10) default -- NOT the same band as S4.1's 50-seed Monte-Carlo band), "
              f"with circular=True instead of moving-block resampling "
              f"(moving-block mean over the identical seeds: {mean_mb:.1f})",
         cites=["S4.1 body: circular-bootstrap robustness check"],
         cross_check="mcs_circular_check.csv: curated_mean_circular")
    flush()

    print("\n=== READ ===")
    print(f"  arch gap: moving-block {abs(rate_mb - rate_arch):.4f} -> circular {abs(rate_circ - rate_arch):.4f}")
    print(f"  3-series cardinality match: moving-block {n_match_mb}/3 -> circular {n_match_circ}/3")
    print(f"  3-series exact-set match:   moving-block {n_exact_mb}/3 -> circular {n_exact_circ}/3")
    print(f"  curated count: moving-block {mean_mb:.1f} -> circular {mean_circ:.1f} (both of {n})")


if __name__ == "__main__":
    main()
