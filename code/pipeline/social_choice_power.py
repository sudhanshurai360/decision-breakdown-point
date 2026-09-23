"""social_choice_power.py — does the Section 5 permutation null have power to detect a real edge, or
is "at or below chance" partly a low-power negative rather than a demonstrated absence of fragility?
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "instrument"))
import social_choice as SC
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper

RES = ROOT / "multiseries" / "results"
STAT_NAMES = ["reversal", "concentration_gt1", "kstar_is_1", "no_condorcet", "cycle"]
DELTAS = [0.005, 0.01, 0.02, 0.05]


def _tag(delta):
    """Canonical-key-safe suffix for a delta, e.g. 0.005 -> "0p5"."""
    pct = delta * 100
    s = f"{pct:g}"
    return s.replace(".", "p")
N_PERM = SC.N_PERM  # 2000, matching the real headline permutation_null call's own precision


def base_cells():
    d = pd.read_csv(SC.PRED)
    cells = []
    for sid, g in d.groupby("series"):
        for metric, (col, agg) in SC.METRICS.items():
            per = g.groupby(["model", "year"])[col].apply(agg).reset_index()
            piv = per.pivot(index="year", columns="model", values=col).dropna()
            L = {m: piv[m].to_numpy(float) for m in piv.columns}
            w = np.ones(len(piv))
            cells.append((L, w))
    return cells


def with_edge(cells, model, delta):
    """Subtract delta*scale from `model`'s series in every cell -- the same uncentered, additive
    convention mcs_power.py's corrected design uses, applied panel-wide rather than to one panel."""
    out = []
    for L, w in cells:
        if model not in L:
            out.append((L, w)); continue
        models = list(L)
        scale = float(np.mean(np.column_stack([L[m] for m in models])))
        L2 = dict(L)
        L2[model] = L[model] - delta * scale
        out.append((L2, w))
    return out


def main():
    cells = base_cells()
    models = sorted(set().union(*[set(L) for L, w in cells]))
    missing = [(m, sid_metric) for m in models for sid_metric, (L, w) in enumerate(cells) if m not in L]
    if missing:
        raise RuntimeError(
            f"social_choice_power.py: {len(missing)} (model, cell) pair(s) missing a model this file's "
            f"panel-wide edge-injection design assumes present everywhere -- with_edge() would silently "
            f"no-op on these rather than plant a genuine panel-wide edge. First few: {missing[:5]}"
        )
    print(f"=== SECTION 5 PERMUTATION-NULL POWER ({len(cells)} cells, {len(models)} models, "
          f"n_perm={N_PERM}) ===")
    rows = []
    obs0, null0 = SC.permutation_null(cells, n_perm=N_PERM)
    p05_0 = np.percentile(null0, 5, axis=0)
    hit0 = obs0 < p05_0
    rows.append({"delta": 0.0, "model": "__baseline__",
                 **{f"obs_{n}": int(o) for n, o in zip(STAT_NAMES, obs0)},
                 **{f"p05_{n}": float(p) for n, p in zip(STAT_NAMES, p05_0)},
                 **{f"hit_{n}": bool(h) for n, h in zip(STAT_NAMES, hit0)}})
    print("  delta=0 (unperturbed baseline): " + "  ".join(
        f"{n}: obs {int(o)} vs p05 {p:.0f} -> {'ALREADY IN REJECTION REGION' if h else 'outside'}"
        for n, o, p, h in zip(STAT_NAMES, obs0, p05_0, hit0)))
    for delta in DELTAS:
        detections = np.zeros(5, dtype=int)
        for m in models:
            perturbed = with_edge(cells, m, delta)
            obs, null = SC.permutation_null(perturbed, n_perm=N_PERM)
            p05 = np.percentile(null, 5, axis=0)
            hit = obs < p05
            detections += hit.astype(int)
            rows.append({"delta": delta, "model": m, **{f"obs_{n}": int(o) for n, o in zip(STAT_NAMES, obs)},
                         **{f"p05_{n}": float(p) for n, p in zip(STAT_NAMES, p05)},
                         **{f"hit_{n}": bool(h) for n, h in zip(STAT_NAMES, hit)}})
        rate = detections / len(models)
        print(f"  delta={delta:.2%}: " + "  ".join(f"{n}={r:.2f}" for n, r in zip(STAT_NAMES, rate)))
    R = pd.DataFrame(rows)
    RM._atomic_to_csv(R, RES / "social_choice_power.csv")

    for stat, o, p, h in zip(STAT_NAMES, obs0, p05_0, hit0):
        emit(f"scpower.{stat}_baseline_in_rejection_region", int(h), obs=int(o), p05=float(p),
             spec=f"whether the UNPERTURBED (delta=0) curated panel's panel-wide '{stat}' count "
                  f"already sits below the permutation null's own fifth percentile (obs={int(o)} vs "
                  f"p05={p:.0f}). 1 means the detection rule used by the scpower.{stat}_* power "
                  f"figures fires with NO edge planted, so those figures measure P(hit | no edge) = 1 "
                  f"rather than power and must not be read as a power result for this statistic; 0 "
                  f"means the statistic starts outside the rejection region and its hits reflect "
                  f"genuine movement",
             cites=["S5.0 body: permutation-null power check"],
             cross_check=f"social_choice_power.csv: hit_{stat} where model=='__baseline__'")
    for delta in DELTAS:
        sub = R[(R.delta == delta) & (R.model != "__baseline__")]
        for stat in STAT_NAMES:
            hits = int(sub[f"hit_{stat}"].sum())
            emit(f"scpower.{stat}_{_tag(delta)}", hits, of=len(models),
                 spec=f"of the {len(models)} curated models (one per trial), how many whose "
                      f"panel-wide '{stat}' count, after a {delta:.1%} additive edge planted "
                      f"panel-wide for that model, falls below the permutation null's own fifth "
                      f"percentile -- the direction a real edge should push it, mirroring "
                      f"mcs_power.py's edge-injection design applied to the Section 5 statistics",
                 cites=["S5.0 body: permutation-null power check"],
                 cross_check=f"social_choice_power.csv: hit_{stat} where delta=={delta}")
    flush()
    print(f"\n  wrote {RES/'social_choice_power.csv'}")


if __name__ == "__main__":
    main()
