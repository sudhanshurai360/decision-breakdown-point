"""
mcs_test.py — per-series Model Confidence Set (Hansen, Lunde, Nason 2011) + the PREDICTIVE-LAW test.
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
import fragility as F
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
CIRCULAR_BOOTSTRAP = False   # MCS resamples moving-block (HLN); winner_stability in fragility.py resamples circularly
_SE_REL_FLOOR = 1e-12        # a bootstrap se below 1e-12 x the loss scale is float noise, not a real spread
MIN_T_FOR_ZERO_SE = 8        # below 8 periods a zero-se pair is more likely a degenerate resample than a certain gap
PRED = MULTI / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
OUT = MULTI / "results" / "mcs_test.csv"


def _block_idx(T, block, rng):
    out = []
    while len(out) < T:
        s = int(rng.integers(0, max(1, T - block + 1)))
        out.extend(range(s, min(s + block, T)))
    return np.asarray(out[:T])


def _circular_block_idx(T, block, rng):
    """Circular block bootstrap (Politis & Romano 1992): blocks wrap past the end instead of being
    drawn only from starts in [0, T-block], so first/last observations are resampled at the same
    rate as an interior one."""
    if block <= 1:
        return rng.integers(0, T, size=T)
    starts = rng.integers(0, T, size=int(np.ceil(T / block)))
    return np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]


ELIMINATION = "raw"   # the default elimination rule, below and in every spec= string that reports it


def mcs(L, alpha=0.10, B=2000, block=3, seed=0, elimination=ELIMINATION, circular=CIRCULAR_BOOTSTRAP):
    """L: T x K per-period loss matrix."""
    if elimination not in ("raw", "studentized", "e_R"):
        raise ValueError(f"mcs(): unknown elimination rule {elimination!r} -- use 'raw', 'studentized' or 'e_R'")
    rng = np.random.default_rng(seed)
    T, K = L.shape
    if T == 0:
        raise ValueError("mcs(): T=0 -- the loss matrix has zero evaluation periods, nothing to test")
    if T <= block:
        block_eff = max(1, T // 2)
        print(f"  [mcs() WARNING] T={T} <= block={block}: bootstrap would be a no-op (zero resampling variability). "
              f"Shrinking block to {block_eff} for this call so the test retains real power.")
        block = block_eff
    idx_fn = _circular_block_idx if circular else _block_idx
    boot = [idx_fn(T, block, rng) for _ in range(B)]
    surv = list(range(K))
    p_stop = 1.0
    while len(surv) > 1:
        sub = L[:, surv]; k = len(surv)
        dbar = sub.mean(0)
        bmeans = np.array([sub[idx].mean(0) for idx in boot])          # B x k
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        TRb = np.zeros(B); TR = 0.0
        e_R_drop = None   # the losing (higher-loss) model of the pair attaining T_R -- see elimination="e_R"
        for (a, b) in pairs:
            diff = bmeans[:, a] - bmeans[:, b]
            obs_diff = dbar[a] - dbar[b]
            se = float(np.sqrt(np.mean((diff - obs_diff) ** 2)))
            scale = max(abs(dbar[a]), abs(dbar[b]), abs(obs_diff))
            if se <= _SE_REL_FLOOR * scale:
                if abs(obs_diff) > 0 and T >= MIN_T_FOR_ZERO_SE:
                    if not np.isinf(TR):          # first certain rejection fixes the e_R candidate
                        e_R_drop = a if obs_diff > 0 else b
                    TR = np.inf
                continue   # obs_diff==0 too: a genuine exact tie, correctly uninformative, skip
            t_obs = abs(obs_diff) / se
            if t_obs > TR:                        # strict: a later equal t never displaces an earlier one
                TR = t_obs
                e_R_drop = a if obs_diff > 0 else b
            TRb = np.maximum(TRb, np.abs(diff - obs_diff) / se)
        p = float((np.sum(TRb >= TR) + 1) / (B + 1)) if np.isfinite(TR) else 0.0
        if p >= alpha:
            p_stop = p; break
        if elimination == "studentized":
            d_dot = dbar - dbar.mean()
            b_dot = bmeans - bmeans.mean(axis=1, keepdims=True)
            se_dot = np.sqrt(np.mean((b_dot - d_dot) ** 2, axis=0))
            scale = np.maximum(np.maximum(np.abs(dbar), np.abs(d_dot)), 1e-300)
            safe = se_dot > _SE_REL_FLOOR * scale
            t_dot = np.where(safe, d_dot / np.where(safe, se_dot, 1.0), np.where(d_dot > 0, np.inf, -np.inf))
            drop = int(np.argmax(t_dot))
        elif elimination == "e_R":
            if e_R_drop is None:
                raise RuntimeError("mcs(elimination='e_R'): rejected with no pair attaining T_R -- impossible state")
            drop = int(e_R_drop)
        else:                                                           # "raw", validated on entry
            drop = int(np.argmax(dbar))                                 # default: raw-mean rule
        surv.pop(drop)
        p_stop = p
    return surv, p_stop


def main():
    d = pd.read_csv(PRED)
    rows = []
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        yrs = sorted(g.year.unique())
        # per-year mean loss matrices (years x models)
        La = np.full((len(yrs), len(models)), np.nan)   # absolute (MASE-style)
        Ls = np.full((len(yrs), len(models)), np.nan)   # squared (MSSE -- NOT rmsse, no sqrt)
        Lr = np.full((len(yrs), len(models)), np.nan)   # RMSSE proper: sqrt(mean(msse)) PER YEAR
        for j, m in enumerate(models):
            gm = g[g.model == m].groupby("year")
            a = gm["sae"].mean(); s = gm["msse"].mean(); r = np.sqrt(s)
            for i, y in enumerate(yrs):
                if y in a.index: La[i, j] = a[y]; Ls[i, j] = s[y]; Lr[i, j] = r[y]
        keep = ~(np.isnan(La).any(1) | np.isnan(Ls).any(1) | np.isnan(Lr).any(1))
        if (~keep).sum():
            print(f"  [warn] {sid}: dropping {(~keep).sum()} year(s) with a NaN in La/Ls/Lr before MCS")
        La, Ls, Lr = La[keep], Ls[keep], Lr[keep]
        # shock-proneness (surprise concentration on per-year loss)
        Ld = {m: La[:, j] for j, m in enumerate(models)}
        sc = F.surprise_concentration(Ld)
        out = {"series": sid, "sci": round(sc["sci"], 3), "eff_hard_yrs": round(sc["eff_surprising_periods"], 2)}
        for tag, L in [("abs", La), ("sq", Ls), ("rmsse", Lr)]:
            for al in (0.10, 0.25):
                surv, _ = mcs(L, alpha=al, seed=0)
                out[f"MCS_{tag}_{int(al*100)}"] = len(surv)
        out["best_abs"] = models[int(np.argmin(La.mean(0)))]
        rows.append(out)
    r = pd.DataFrame(rows).sort_values("MCS_abs_10")
    RM._atomic_to_csv(r, OUT)
    n = len(r)
    emit("mcs.curated_nonidentified", int((r.MCS_abs_10 > 1).sum()), of=int(n),
         spec=f"series with |MCS|>1 under absolute loss at alpha=0.10; circular={CIRCULAR_BOOTSTRAP}, "
              f"elimination={ELIMINATION}",
         cites=["Abstract: headline curated non-identification count",
                "Introduction: headline curated non-identification count",
                "S4.1 body: headline curated count"],
         cross_check="mcs_test.csv: (MCS_abs_10 > 1).sum()")
    emit("mcs.curated_total", int(n),
         spec="series in the curated panel",
         cites=["S4.1 body: curated panel size"],
         cross_check="mcs_test.csv: len(r)")
    emit("mcs.curated_nonidentified_rmsse", int((r.MCS_rmsse_10 > 1).sum()), of=int(n),
         spec=f"series with |MCS|>1 under RMSSE loss (sqrt(mean(msse)) per year) at alpha=0.10; "
              f"circular={CIRCULAR_BOOTSTRAP}, elimination={ELIMINATION} -- the same sensitivity "
              f"check as mcs.curated_nonidentified but under the paper's OTHER stated loss "
              f"function (Section 5 scores both MASE and RMSSE; Table 1 reports MASE/absolute-loss "
              f"only)",
         cites=["S4.1 body: RMSSE sensitivity check"],
         cross_check="mcs_test.csv: (MCS_rmsse_10 > 1).sum()")
    flush()
    pd.set_option("display.width", 200)
    print("=== per-series MODEL CONFIDENCE SET (Hansen 2011; |MCS|=1 => point-identified) ===")
    print(r.to_string(index=False))
    for c in ["MCS_abs_10", "MCS_abs_25", "MCS_sq_10"]:
        print(f"  {c}: identified(|MCS|=1) {int((r[c]==1).sum())}/{n} ; near-tie(|MCS|>1) {int((r[c]>1).sum())}/{n} ; mean |MCS| {r[c].mean():.2f}")
    print("\n=== PREDICTIVE LAW: does shock-proneness predict non-identification? ===")
    for c in ["MCS_abs_10", "MCS_abs_25", "MCS_sq_10"]:
        rho = r[["sci", c]].corr(method="spearman").iloc[0, 1]
        print(f"  Spearman(sci, {c}) = {rho:+.3f}")
    print("  (M4 external anchor: champion metric-invariant => |MCS| small at the identified corner.)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
