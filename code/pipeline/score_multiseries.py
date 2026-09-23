"""
score_multiseries.py — RE-SCORE the frozen, committed per-series predictions under a chosen
MASE/RMSSE scale, WITHOUT re-running any model.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

RESULTS = MULTI / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INSTR))
import walkforward as WF          # noqa: E402
import fragility as F             # noqa: E402
import run_multiseries as RM      # noqa: E402  -- for its shared _atomic_to_csv() helper

TRAIL = int(os.environ.get("MS_TRAIL", "120")) or None   # months; 0 -> None (full expanding)
SHOCK_YEARS = [2008, 2009, 2020, 2021]
N_BOOT = int(os.environ.get("MS_NBOOT", "2000"))
M = 12


def rescore_series(sid: str, pred: pd.DataFrame) -> list:
    """Recompute per-year scales (trailing) from the vintages, re-derive sae/msse on the FROZEN y_true/y_pred,
    then run the fragility diagnostic per metric."""
    years = sorted(pred["year"].unique())
    truth = WF.realized(WF.load_vintages(sid), sid, guard_before=f"{int(max(years))}-01-01")
    sc_abs = WF.year_scales(truth, years, m=M, squared=False, trail=TRAIL)
    sc_sq = WF.year_scales(truth, years, m=M, squared=True, trail=TRAIL)
    g = pred.copy()
    g["scale_abs"] = g["year"].map(sc_abs)
    g["scale_sq"] = g["year"].map(sc_sq)
    g = g.dropna(subset=["scale_abs", "scale_sq"])
    err = g["y_true"] - g["y_pred"]
    g["sae"] = err.abs() / g["scale_abs"]
    g["msse"] = (err ** 2) / g["scale_sq"]

    rows = []
    for metric in ("mase", "rmsse"):
        agg = WF.agg_metric(g, metric)
        L, w, yrs = WF.to_fragility_inputs(agg)
        keep = [i for i, _ in enumerate(yrs) if all(np.isfinite(L[m][i]) for m in L)]
        L2 = {m: L[m][keep] for m in L}; w2 = w[keep]; ys = [yrs[i] for i in keep]
        r = F.fragility(L2, w2, shock_periods=SHOCK_YEARS, labels=ys, n_boot=N_BOOT, seed=0)
        pooled = float(np.average(L2[r["pooled_winner"]], weights=w2))
        rows.append({
            "series": sid, "metric": metric, "scale": f"trail{TRAIL}" if TRAIL else "expanding",
            "n_years": len(ys), "pooled_winner": r["pooled_winner"], "per_period_winner": r["per_period_winner"],
            "runner_up": r["runner_up"], "winner_pooled_loss": round(pooled, 3),
            "k_star": r["k_star"], "k_over_T": round(r["k_over_T"], 3), "reversal": r["reversal"],
            "responsible": ";".join(map(str, r["responsible"])), "shock_coincidence": r["shock_coincidence"],
            "winner_stability": round(r["winner_stability"], 3), "fragile": r["fragile"],
        })
    return rows


def main():
    pred = pd.read_csv(RESULTS / "multiseries_predictions.csv")
    rows = []
    for sid in sorted(pred["series"].unique()):
        rows.extend(rescore_series(sid, pred[pred["series"] == sid]))
    out = pd.DataFrame(rows)
    tag = f"trail{TRAIL}" if TRAIL else "expanding"
    _csv_out = RESULTS / f"multiseries_fragility_{tag}.csv"
    RM._atomic_to_csv(out, _csv_out)
    print(f"=== MULTISERIES FRAGILITY — scale = {tag} (re-scored from frozen predictions) ===")
    print(out.to_string(index=False))
    print(f"\nwrote {RESULTS/('multiseries_fragility_'+tag+'.csv')}")


if __name__ == "__main__":
    main()
