"""
mcs_power.py — how much power does the MCS actually have here?
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "instrument"))
import mcs_test as MCS
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
PRED = HERE.parent.parent / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
B, ALPHA = 800, 0.10
DELTAS = [0.0, 0.02, 0.05, 0.10, 0.30, 0.50]
CONTROL_EDGE, CONTROL_N, CONTROL_MIN = 0.30, 2000, 0.90


def load_panels():
    d = pd.read_csv(PRED)
    mats, models = [], None
    for sid, g in d.groupby("series"):
        ms = sorted(g.model.unique()); ys = sorted(g.year.unique())
        M = np.array([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in ms] for y in ys])
        if np.all(np.isfinite(M)) and M.shape[0] >= 20:
            mats.append(M); models = ms
    if len(mats) != 12:
        raise AssertionError(
            f"load_panels(): expected 12 curated series with fully rectangular, fully-finite loss "
            f"matrices, got {len(mats)}. A missing (model, year) cell silently dropped a whole series -- "
            f"decide explicitly (per-year filtering, matching mcs_test.py's `keep` mask, vs. accepting "
            f"fewer trials) rather than letting mcspower.n_trials quietly shrink."
        )
    return mats, models


_CONTROL_SEEDS = (11, 10_000_030, 20_000_049, 30_000_068, 40_000_087)


def _stable_negative_control(elimination):
    rates = []
    for seed in _CONTROL_SEEDS:
        g = np.random.default_rng(seed)
        det = []
        for r in range(CONTROL_N):
            M = g.normal(1.0, 0.30, (30, 6))
            s, _ = MCS.mcs(M, alpha=ALPHA, B=B, seed=0, elimination=elimination)
            det.append(len(s) == 1)                  # ANY singleton is a false discovery under the null
        rates.append(float(np.mean(det)))
    rates = np.asarray(rates, float)
    return {"mean": float(rates.mean()), "sd": float(rates.std(ddof=1)),
            "min": float(rates.min()), "max": float(rates.max()), "n_seeds": len(_CONTROL_SEEDS)}


def negative_control():
    """A genuine NULL: i.i.d. Gaussian loss panels (K=6, T=30) in which no model is better than any
    other, so ANY singleton MCS is a false discovery. Neither arm of sweep() can supply this figure,
    because both run on the real curated panels, which have a real best model even at a zero injected
    edge. Studentized elimination; averaged over _CONTROL_SEEDS."""
    return _stable_negative_control("studentized")


def negative_control_raw():
    """SAME construction as negative_control() above, but under the RAW elimination rule -- the one
    the paper's own headline non-identification count actually uses (mcs_test.py's default, "raw
    mean" removal), not the studentized rule this file otherwise reports throughout."""
    return _stable_negative_control("raw")


def positive_control():
    """Panels with a known answer. Gates every other number in this file."""
    g = np.random.default_rng(7)
    det = []
    for r in range(CONTROL_N):
        M = g.normal(1.0, 0.30, (30, 6))
        f = r % 6
        L = M.copy(); L[:, f] = L[:, f] - CONTROL_EDGE * float(np.mean(M))
        s, _ = MCS.mcs(L, alpha=ALPHA, B=B, seed=0, elimination="studentized")
        det.append(len(s) == 1 and s[0] == f)
    return float(np.mean(det))


def sweep(mats, delta, recenter):
    """Detection share at one edge."""
    det, sizes, det_cond = [], [], []
    for M in mats:
        base = M - M.mean(0) + M.mean() if recenter else M
        scale = float(np.mean(base))
        for f in range(base.shape[1]):
            L = base.copy()
            if recenter:
                L[:, f] = L[:, f] * (1 - delta)
            else:
                L[:, f] = L[:, f] - delta * scale
            s, _ = MCS.mcs(L, alpha=ALPHA, B=B, seed=0, elimination="studentized")
            hit = len(s) == 1 and s[0] == f
            det.append(hit)
            sizes.append(len(s))
            if int(np.argmin(L.mean(0))) == f:          # the edge actually made f the pooled best
                det_cond.append(hit)
    cond = float(np.mean(det_cond)) if det_cond else float("nan")
    return float(np.mean(det)), float(np.median(sizes)), len(det), cond, len(det_cond)


def main():
    mats, _ = load_panels()
    n_trials = int(sum(M.shape[1] for M in mats))
    print(f"=== MCS POWER (real per-year dispersion from {len(mats)} panels, {n_trials} focal-model trials/level) ===")

    ctrl = positive_control()
    print(f"\n  POSITIVE CONTROL  iid K=6 T=30, {CONTROL_EDGE:.0%} additive edge -> detected {ctrl:.3f} "
          f"of {CONTROL_N} (floor {CONTROL_MIN:.2f})")
    if ctrl < CONTROL_MIN:
        raise SystemExit(f"ABORT: positive control {ctrl:.3f} < {CONTROL_MIN}. The harness cannot "
                         f"detect an edge it was handed, so every power number below is meaningless. "
                         f"Fix the harness before reading anything else.")
    print("  -> PASS\n")
    emit("mcspower.positive_control", round(ctrl, 3), n=CONTROL_N, edge=CONTROL_EDGE,
         spec=f"share of {CONTROL_N} i.i.d. K=6 T=30 panels with a {CONTROL_EDGE:.0%} additive edge on "
              f"one model where the studentized MCS singles that model out. A VALIDITY GATE, not a "
              f"result: the stage raises if this falls below {CONTROL_MIN}, because a harness that "
              f"cannot find an edge it planted says nothing about one it cannot find in the data",
         cites=["S4.2 body: positive control"], cross_check=None)

    emit("mcspower.n_trials", n_trials, n_panels=len(mats),
         spec="focal-model trials per injected-edge level: one per model per panel, swept "
              "deterministically rather than sampled. Every mcspower.* share is out of this many",
         cites=["S4.2 body: power of the studentized rule"], cross_check=None)

    print(f"  {'edge':>6s} | {'CORRECTED (uncentered, additive)':>34s} | {'superseded (recentred, multiplicative)':>38s}")
    rows = []
    for delta in DELTAS:
        pw, med, _, pw_cond, n_qual = sweep(mats, delta, recenter=False)
        pw_old, med_old, _, _, _ = sweep(mats, delta, recenter=True)
        pct = int(round(delta * 100))
        rows.append({"edge_pct": pct, "power_corrected": pw, "median_set_corrected": med,
                     "power_recentred": pw_old, "median_set_recentred": med_old,
                     "power_conditional": pw_cond, "n_qualifying": n_qual})
        # NOT "specificity": the corrected arm is uncentered and has a real best model, so this row
        # is correct identification, not false discovery. Specificity is the negative control.
        tag = "   <- no edge planted; see the negative control for false discovery" if delta == 0 else ""
        print(f"  {delta:>5.0%} |   {pw:.3f}  (median |MCS| {med:.1f})           |   "
              f"{pw_old:.3f}  (median |MCS| {med_old:.1f}){tag}")
        emit(f"mcspower.studentized_power_{pct}pct", pw,
             spec=f"share of {n_trials} focal-model trials at a {pct}% ADDITIVE true edge on the "
                  f"UNCENTERED panel where the studentized-rule MCS singles out the focal model alone. "
                  f"The edge is {pct}% of the panel's grand mean loss over all six candidates "
                  f"(np.mean(base)), a larger unit than the best model's own mean loss used in S3.1. "
                  f"This is the design the paper reports; see mcspower.recentred_power_{pct}pct for "
                  f"the superseded re-centred figure (a multiplicative edge on an equal-mean "
                  f"panel structurally protected one high-variance model from elimination)",
             cites=["S4.2 body: power of the studentized rule"],
             cross_check=f"mcs_power.csv: power_corrected where edge_pct=={pct}")
        emit(f"mcspower.conditional_power_{pct}pct", round(pw_cond, 4), n_qualifying=n_qual,
             n_trials=n_trials, n_hits=int(round(pw_cond * n_qual)),
             spec=f"detection share at a {pct}% additive edge RESTRICTED to the {n_qual} of {n_trials} "
                  f"trials where the planted edge actually made the focal model the pooled best. The "
                  f"unconditional figure mixes two failures -- the test could not resolve the edge, and "
                  f"the edge was too small to put the focal model in front at all -- and only the first "
                  f"is a statement about resolution. The gap between the two is the arithmetic ceiling "
                  f"S4.2 previously asserted without measuring",
             cites=["S4.2 body: conditional power"],
             cross_check=f"mcs_power.csv: power_conditional where edge_pct=={pct}")
        emit(f"mcspower.recentred_power_{pct}pct", pw_old,
             spec="SUPERSEDED, never to be quoted as a power estimate: the same trial under the "
                  "re-centred equal-mean construction with a multiplicative edge, which made one "
                  "high-variance model structurally uneliminable. Emitted so the artifact is a "
                  "published number rather than an assertion about an earlier draft",
             cites=(["S4.2 body: the superseded power design"] if pct == 10 else []),
             cross_check=f"mcs_power.csv: power_recentred where edge_pct=={pct}")
    RM._atomic_to_csv(pd.DataFrame(rows), HERE.parent.parent / "multiseries" / "results" / "mcs_power.csv")

    spec0 = [r for r in rows if r["edge_pct"] == 0][0]["power_corrected"]
    emit("mcspower.zero_edge_detection", spec0,
         spec="share of trials at a ZERO injected edge where the corrected sweep singles out the focal "
              "model. NOT a false-discovery rate: that arm runs on the UNCENTERED panel, which has a "
              "genuine best model, so these are trials where the focal model really is best -- the "
              "curated panel's own point-identification showing through. Emitted to be explicit about "
              "what the zero row of mcs_power.csv is",
         cites=["S4.2 body: the zero-edge row"],
         cross_check="mcs_power.csv: power_corrected where edge_pct==0")
    neg = negative_control()
    print(f"\n  NEGATIVE CONTROL  iid K=6 T=30, no edge -> false discovery {neg['mean']:.4f} of {CONTROL_N} "
          f"(range {neg['min']:.4f}-{neg['max']:.4f} across {neg['n_seeds']} seeds, sd={neg['sd']:.4f})")
    # cross_check=None -- deterministic synthetic Monte Carlo, fixed seeds, per-trial detections
    # never persisted to a CSV, so there is nothing on disk to cross-check against.
    emit("mcspower.false_discovery_null", round(neg["mean"], 4), n=CONTROL_N,
         sd=round(neg["sd"], 4), aux_max=round(neg["max"], 4),
         spec=f"false-discovery rate on a genuine null: {CONTROL_N} i.i.d. panels at K=6, T=30 with NO "
              f"planted edge, counting trials where the studentized MCS collapses to any single model, "
              f"AVERAGED over {neg['n_seeds']} independent seeds (reported as a mean with its own "
              f"seed-to-seed sd, since a single-seed estimate at this budget could "
              f"plausibly differ by roughly this range). This is the specificity figure the paper "
              f"reports, and it is measured on synthetic i.i.d. panels because neither arm of the sweep "
              f"can supply one: both run on the real curated panels, which have a real best model even "
              f"at a zero injected edge",
         cites=["S4.2 body: specificity"], cross_check=None)
    neg_raw = negative_control_raw()
    print(f"  NEGATIVE CONTROL (RAW elimination, the paper's headline rule)  iid K=6 T=30, no edge -> "
          f"false discovery {neg_raw['mean']:.4f} of {CONTROL_N} (range {neg_raw['min']:.4f}-"
          f"{neg_raw['max']:.4f} across {neg_raw['n_seeds']} seeds, sd={neg_raw['sd']:.4f})")
    emit("mcspower.false_discovery_null_raw", round(neg_raw["mean"], 4), n=CONTROL_N,
         sd=round(neg_raw["sd"], 4), aux_max=round(neg_raw["max"], 4),
         spec=f"SAME false-discovery-rate check as mcspower.false_discovery_null, but under the RAW "
              f"elimination rule -- the one the paper's own headline non-identification count actually "
              f"uses -- rather than the studentized rule reported elsewhere in this section. Same "
              f"{neg_raw['n_seeds']}-seed averaging as mcspower.false_discovery_null.",
         cites=["S4.2 body: specificity under the headline rule"], cross_check=None)

    print("\n  HONEST read: the CORRECTED design (uncentered, additive edge on the real panel) is what the paper")
    print("  reports. Specificity comes from a genuine i.i.d. null (the negative control above), not from the")
    print("  zero-edge row of the sweep itself, which has a real best model and is not a null. So:")
    print("   (a) the discriminant validity rests on the DIRECT evidence -- PI, the M4 champion, and 18/74 FRED-MD series")
    print("       ARE identified at the SAME sample sizes -- plus this near-perfect specificity; NOT on a clean power claim.")
    print("   (b) '11/12 non-identified' must be SCOPED honestly: classical models, short annual shock-prone panels")
    print("       (part genuine near-tie, part shock-limited power). The conditional power figures separate how much")
    print("       of the small-edge shortfall is arithmetic (the edge never made the focal model best at all) from how")
    print("       much is a real resolution limit.")
    flush()


if __name__ == "__main__":
    main()
