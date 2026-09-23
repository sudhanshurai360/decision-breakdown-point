"""
shape_not_magnitude.py — the SHAPE-NOT-MAGNITUDE main exhibit.
"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'instrument'))
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
ROOT_DIR = HERE.parent.parent   # repo root -- used to print RELATIVE output paths in logs
PRED = HERE.parent.parent / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
OUT = HERE.parent.parent / "multiseries" / "results" / "shape_not_magnitude.csv"


def main():
    d = pd.read_csv(PRED); d["target"] = pd.to_datetime(d["target"])
    diff = d.groupby(["series", "target"])["sae"].min().reset_index()          # difficulty = best any model achieved
    diff["ym"] = diff.target.dt.strftime("%Y-%m"); diff["yr"] = diff.target.dt.year
    hardest = diff.loc[diff.groupby("series")["sae"].idxmax()][["series", "ym", "sae"]].sort_values("sae", ascending=False)
    n = hardest.series.nunique()
    med = diff.groupby("series")["sae"].median()
    apr20 = diff[diff.ym == "2020-04"].set_index("series")["sae"]
    ratio = (apr20 / med).dropna()
    ann = diff.groupby(["series", "yr"])["sae"].mean().reset_index()
    ann_ratio = (ann[ann.yr == 2020].set_index("series")["sae"] / ann.groupby("series")["sae"].median()).dropna()
    ann_ratio_matched = ann_ratio.reindex(ratio.index).dropna()

    print("=== SHAPE-NOT-MAGNITUDE (12 distinct macro series) ===")
    print(hardest.to_string(index=False))
    n_apr20 = int((hardest.ym == "2020-04").sum())
    n_2020 = int(hardest.ym.str.startswith("2020").sum())
    n_gfc = int(hardest.ym.str.startswith(("2008", "2009")).sum())
    med_month, med_annual = float(ratio.median()), float(ann_ratio.median())
    med_annual_matched = float(ann_ratio_matched.median())
    max_month, max_series = float(ratio.max()), str(ratio.idxmax())

    emit("shape.hardest_april2020", n_apr20, of=int(n),
         spec="series whose single hardest month is April 2020; difficulty = min scaled abs error over models",
         cites=["S6 body: hardest-month count"],
         cross_check="shape_not_magnitude.csv: (ym == '2020-04').sum()")
    emit("shape.hardest_2020", n_2020, of=int(n),
         spec="series whose hardest month falls anywhere in calendar 2020 (not just April) -- "
              "currently identical to shape.hardest_april2020 by coincidence, not by construction",
         cites=[], cross_check="shape_not_magnitude.csv: ym.str.startswith('2020').sum()")
    emit("shape.hardest_gfc", n_gfc, of=int(n),
         spec="series whose hardest month falls in calendar 2008-09",
         cites=["S6 body: hardest-month count in 2008-09"],
         cross_check="shape_not_magnitude.csv: ym.str.startswith(('2008','2009')).sum()")
    emit("shape.monthly_median_x", med_month, n_series=int(len(ratio)),
         spec="median April-2020 difficulty divided by that series' median month; n=10 (BOPGEXP/"
              "BOPGIMP have no 2020-04 evaluation month, so this is a 10-of-12-series sample -- see "
              "shape.annual_median_x_matched for the like-for-like annual comparison on this same "
              "10-series sample)",
         cites=["S6 body: monthly median"],
         cross_check="derived from breadth_distinct_predictions.csv per shape_not_magnitude.main()")
    emit("shape.annual_median_x", med_annual, n_series=int(len(ann_ratio)),
         spec="same errors aggregated to annual frequency, all 12 series -- NOT the same sample as "
              "shape.monthly_median_x's n=10 (BOPGEXP/BOPGIMP excluded there); see "
              "shape.annual_median_x_matched for the matched-sample comparison",
         cites=["S6 body: annual median"],
         cross_check="derived from breadth_distinct_predictions.csv per shape_not_magnitude.main()")
    emit("shape.annual_median_x_matched", med_annual_matched, n_series=int(len(ann_ratio_matched)),
         spec="same errors aggregated to annual frequency, restricted to the same 10 series "
              "shape.monthly_median_x uses (excludes BOPGEXP/BOPGIMP, which have no 2020-04 "
              "evaluation month) -- the like-for-like counterpart to the MONTHLY->ANNUAL dilution "
              "claim; shape.annual_median_x itself is the unmatched, all-12-series figure",
         cites=[],
         cross_check="derived from breadth_distinct_predictions.csv per shape_not_magnitude.main()")
    emit("shape.monthly_max_x", max_month, series=max_series,
         spec="largest April-2020 multiple across series; full precision, round once at render",
         cites=["S6 body: maximum multiple"],
         cross_check="derived from breadth_distinct_predictions.csv per shape_not_magnitude.main()")

    print(f"\n  hardest month in 2020:   {(hardest.ym.str.startswith('2020')).sum()}/{n}")
    print(f"  hardest month in 2008-09: {n_gfc}/{n}  <- the GFC NEVER tops the list")
    print(f"  2020-04 vs normal month: median {med_month:.1f}x  (max {max_month:.0f}x = {max_series})")
    print(f"  ANNUAL aggregation hides the cliff: MONTHLY {med_month:.1f}x  ->  ANNUAL "
          f"{med_annual_matched:.1f}x (matched 10-series sample; all-12-series ANNUAL is "
          f"{med_annual:.1f}x)")

    out = hardest.copy()
    out["x_normal_2020_04"] = out.series.map((apr20 / med).round(1))
    RM._atomic_to_csv(out, OUT)
    # persist the aggregate headline numbers as a citable artifact, not just to stdout
    _summary = pd.DataFrame([{
        "n_series": int(n),
        "hardest_in_2020": int(hardest.ym.str.startswith("2020").sum()),
        "hardest_in_2008_09": int(hardest.ym.str.startswith(("2008", "2009")).sum()),
        "monthly_2020_04_median_x": round(float(ratio.median()), 1),
        "monthly_2020_04_max_x": round(float(ratio.max()), 1),
        "monthly_2020_04_max_series": ratio.idxmax(),
        "annual_2020_median_x": round(float(ann_ratio.median()), 1),
    }])
    RM._atomic_to_csv(_summary, HERE.parent.parent / "multiseries" / "results" / "shape_not_magnitude_summary.csv")
    print(f"\nwrote {OUT.relative_to(ROOT_DIR)}")
    flush()


if __name__ == "__main__":
    main()
