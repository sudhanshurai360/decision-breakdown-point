"""m4_margins.py — the champion-vs-runner-up margin on each of M4's 4 scoring metrics.
"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "instrument"))
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
RES = ROOT / "multiseries" / "results"
METRICS = ["sMAPE", "MAPE", "MASE", "RMSSE"]


def main():
    B = pd.read_csv(RES / "m4_submissions_leaderboard.csv", index_col=0)
    B.index = B.index.str.replace(".rar", "", regex=False)
    champion = B["MASE"].idxmin()   # same champion on every metric (Pareto-dominant, m4sub.pareto_dominant)
    if not all(B[mt].idxmin() == champion for mt in METRICS):
        raise AssertionError(
            f"Pareto-dominance broke: {champion} (MASE champion) is not the argmin on every metric -- "
            f"the per-metric champion_value/champion labeling below is no longer valid")
    rows = []
    for mt in METRICS:
        s = B[mt].sort_values()
        champ_val = float(s.iloc[0])
        runner_up = s.index[1]
        runner_val = float(s.iloc[1])
        gap_pct = 100 * (runner_val - champ_val) / champ_val
        rows.append({"metric": mt, "champion": champion, "champion_value": champ_val,
                     "runner_up": runner_up, "runner_up_value": runner_val, "gap_pct": gap_pct})
        print(f"  {mt:6s}: champion {champ_val:.5f}  runner-up {runner_up} {runner_val:.5f}  gap {gap_pct:.2f}%")
    R = pd.DataFrame(rows)
    RM._atomic_to_csv(R, RES / "m4_margins.csv")
    for r in rows:
        key = f"m4sub.gap_{r['metric'].lower()}_pct"
        emit(key, round(r["gap_pct"], 2), champion=r["champion"], runner_up=r["runner_up"],
             spec=f"champion-vs-runner-up percentage gap on {r['metric']}, "
                  f"({r['runner_up_value']:.5f} - {r['champion_value']:.5f}) / {r['champion_value']:.5f} "
                  f"* 100 -- read against S3.1's 0.93% median best-vs-runner-up gap on the home panel",
             cites=["S9 body: M4 margin table"],
             cross_check=f"m4_margins.csv: gap_pct where metric=='{r['metric']}'")
    flush()


if __name__ == "__main__":
    main()
