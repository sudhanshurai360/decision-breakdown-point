"""data_config.py — RECORD THE PANEL DESCRIPTIONS §3 MUST STATE, SO THEY CANNOT BE TYPED BY HAND.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT_DIR / "code" / "instrument"))
from canonical_emit import emit, flush

RES = HERE.parent.parent / "multiseries" / "results"
ROOT_RES = ROOT_DIR / "results"

# --- frozen input atom (SHA-pinned; reproduce.py --list) --------------------------------------
PRED = RES / "breadth_distinct" / "breadth_distinct_predictions.csv"   # curated panel loss matrix
SCOREBOARD = ROOT_RES / "extension_scoreboard_1988-2025.csv"           # flagship, neural tier
CONTRAST = ROOT_RES / "contrast_exhibit.csv"                           # flagship, window-inversion
# --- regenerated pipeline (each named in a canonical cross_check) ------------------------------
MCSRES = RES / "mcs_monthly_resolution.csv"                            # per-series annual T
SOCIAL = RES / "social_choice.csv"                                     # per-cell metric axis
MANIFEST = ROOT_DIR / "multiseries" / "data" / "raw" / "breadth_manifest_distinct.json"

SEP = ";"


def _joined(values) -> str:
    """Sorted, delimiter-joined membership."""
    return SEP.join(sorted(map(str, set(values))))


def contrast_rt_start(contrast_columns) -> int:
    """First year of the window-inversion exhibit's SHORT (real-time) pool, PARSED from the
    exhibit's own `pooled_RT_<start>-<end>` column name -- not retyped -- so a regenerated exhibit
    on a different window cannot silently disagree with this file."""
    for c in contrast_columns:
        m = re.fullmatch(r"pooled_RT_(\d{4})-\d{2}", str(c))
        if m:
            return int(m.group(1))
    raise ValueError(f"no column matching 'pooled_RT_<YYYY>-<YY>' found in {list(contrast_columns)!r} "
                      f"-- contrast_exhibit.csv's real-time window column is missing or renamed")


def main():
    pred = pd.read_csv(PRED)
    mcsres = pd.read_csv(MCSRES)
    social = pd.read_csv(SOCIAL)
    score = pd.read_csv(SCOREBOARD, index_col=0)
    contrast = pd.read_csv(CONTRAST, index_col=0)
    CONTRAST_RT_START = contrast_rt_start(contrast.columns)

    # ---- B: the curated macro panel ----------------------------------------------------------
    emit("data.curated_series", _joined(pred["series"]), of=int(pred["series"].nunique()),
         spec=f"identifiers of the curated panel, sorted and '{SEP}'-joined; the distinct `series` "
              f"of {PRED.name}, the frozen loss matrix every curated result is scored from",
         cites=["S3.1 body: panel composition"],
         cross_check=f"{PRED.name}: sorted set of series")

    emit("data.curated_models", _joined(pred["model"]), of=int(pred["model"].nunique()),
         spec=f"the forecasting models entered on the curated panel, sorted and '{SEP}'-joined; the "
              f"distinct `model` of {PRED.name}. Recorded from the artifact rather than from "
              f"run_multiseries.MODELS, which resolves from the environment and describes what would "
              f"run now rather than what ran",
         cites=["S3.1 body: model pool"],
         cross_check=f"{PRED.name}: sorted set of model")

    emit("data.curated_n_models", int(pred["model"].nunique()),
         spec=f"size of the curated panel's model pool; distinct `model` of {PRED.name}",
         cites=["S3.1 body: model pool"],
         cross_check=f"{PRED.name}: model.nunique()")

    emit("data.curated_year_max", int(pred["year"].max()),
         spec=f"last evaluation year on the curated panel; max `year` of {PRED.name}. Emitted alone, "
              f"WITHOUT a matching first year: the panel's series begin in several different years, so "
              f"a single minimum would state an envelope as though it were uniform. The per-series "
              f"evaluation LENGTH carries that information honestly and is emitted instead",
         cites=["S3.1 body: evaluation window"],
         cross_check=f"{PRED.name}: year.max()")

    emit("data.curated_T_min", int(mcsres["T_annual"].min()),
         spec=f"shortest per-series annual evaluation length on the curated panel; min `T_annual` of "
              f"{MCSRES.name}. This is the quantity the propositions in the k* section are stated at, "
              f"and it is what every pooled result conditions on",
         cites=["S3.1 body: evaluation length"],
         cross_check=f"{MCSRES.name}: T_annual.min()")

    emit("data.curated_T_max", int(mcsres["T_annual"].max()),
         spec=f"longest per-series annual evaluation length on the curated panel; max `T_annual` of "
              f"{MCSRES.name}",
         cites=["S3.1 body: evaluation length"],
         cross_check=f"{MCSRES.name}: T_annual.max()")

    emit("data.curated_metrics", _joined(social["metric"]), of=int(social["metric"].nunique()),
         spec=f"the accuracy metrics each curated series is scored on, sorted and '{SEP}'-joined; the "
              f"distinct `metric` of {SOCIAL.name}. The panel's cell count is the product of this and "
              f"the series count, so the two must be stated from the same source",
         cites=["S3.1 body: metric axis"],
         cross_check=f"{SOCIAL.name}: sorted set of metric")

    # ---- A: the flagship, which shares only part of B's pool and adds neural models -----------
    emit("data.flagship_scoreboard_models", _joined(score.columns),
         of=int(len(score.columns)),
         spec=f"models entered on the flagship claims series, sorted and '{SEP}'-joined; the columns "
              f"of {SCOREBOARD.name}. Named for its TIER: the flagship is scored on two frozen "
              f"artifacts with different pools, and an unqualified key would not say which",
         cites=["S3.6 body: flagship exhibit"],
         cross_check=f"{SCOREBOARD.name}: sorted set of columns")

    emit("data.flagship_contrast_models", _joined(contrast.index),
         of=int(len(contrast.index)),
         spec=f"models carried on the flagship window-inversion exhibit, sorted and '{SEP}'-joined; "
              f"the index of {CONTRAST.name}. A strict subset of the scoreboard tier, which is why the "
              f"rank denominator quoted from this exhibit differs from the scoreboard's pool size",
         cites=["S3.6 body: flagship exhibit"],
         cross_check=f"{CONTRAST.name}: sorted set of index")

    emit("data.flagship_year_min", int(min(score.index)),
         spec=f"first evaluation year on the flagship series; min index of {SCOREBOARD.name}. Unlike "
              f"the curated panel this is a single series, so a first year is uniform by construction "
              f"and states no envelope",
         cites=["S3.6 body: flagship exhibit"],
         cross_check=f"{SCOREBOARD.name}: min of index")

    emit("data.flagship_year_max", int(max(score.index)),
         spec=f"last evaluation year on the flagship series; max index of {SCOREBOARD.name}",
         cites=["S3.6 body: flagship exhibit"],
         cross_check=f"{SCOREBOARD.name}: max of index")

    man = json.loads(MANIFEST.read_text())
    _passed = sorted(man["passed"])
    _dropped = sorted(s for s, r in man["series"].items() if r["status"] != "PASS")
    if len(_passed) + len(_dropped) != len(man["series"]):
        raise RuntimeError(
            f"breadth manifest's passed ({len(_passed)}) and dropped ({len(_dropped)}) candidates do not "
            f"partition its full series set ({len(man['series'])}) -- some candidate's status is neither "
            f"'PASS' nor counted as dropped, which should be impossible by construction")
    if set(_passed) != set(pred["series"].unique()):
        raise RuntimeError(
            "breadth manifest's passed set does not match the panel actually scored -- the selection log "
            "and the loss matrix have diverged, and one of them is describing a run that is not the run "
            "the paper reports")

    emit("data.curated_candidates", int(len(man["series"])),
         spec=f"size of the PRE-COMMITTED candidate universe screened for the curated panel; number of "
              f"entries in {MANIFEST.name}, the decision log breadth_fetch.py writes at fetch time. The "
              f"denominator that makes the panel size meaningful: {len(_passed)} of this many, not "
              f"{len(_passed)} chosen freely",
         cites=["S3.1 body: panel selection rule"],
         cross_check=f"{MANIFEST.name}: len(series)")

    emit("data.curated_dropped", _joined(_dropped), of=int(len(_dropped)),
         spec=f"candidates screened OUT of the curated panel, sorted and '{SEP}'-joined; entries in "
              f"{MANIFEST.name} with status != PASS (DROP, or a fetch failure). Emitted so the paper "
              f"reports its exclusions by name rather than only its inclusions -- a selection rule "
              f"stated without its casualties is not checkable",
         cites=["S3.1 body: panel selection rule"],
         cross_check=f"{MANIFEST.name}: sorted series with status != PASS")

    emit("data.curated_vintage_cutoff", str(man["criteria"]["vintages_cover"]),
         spec=f"the ALFRED vintage-depth criterion a candidate had to meet to enter the curated panel; "
              f"`criteria.vintages_cover` of {MANIFEST.name}. Pre-committed before results were seen, "
              f"and chosen so the 2008 recession is reconstructable in real time",
         cites=["S3.1 body: panel selection rule"],
         cross_check=f"{MANIFEST.name}: criteria.vintages_cover")

    emit("data.curated_rebase_tol", float(man["criteria"]["rebase_tol"]),
         spec=f"the rebasing tolerance a candidate had to meet to enter the curated panel: the "
              f"real-time/realized ratio at the {man['criteria']['probe_month']} probe had to sit "
              f"within this fraction of 1, screening out rebased indices in favour of natural-unit "
              f"levels; `criteria.rebase_tol` of {MANIFEST.name}",
         cites=["S3.1 body: panel selection rule"],
         cross_check=f"{MANIFEST.name}: criteria.rebase_tol")

    _short = [y for y in score.index if y >= CONTRAST_RT_START]
    _rank_short_revised = (score.loc[_short, list(contrast.index)].mean()
                           .sort_values().index.get_loc("ar1") + 1)
    emit("flagship.ar1_rank_short_window_revised", int(_rank_short_revised),
         of=int(len(contrast.index)), n_periods=int(len(_short)),
         spec=f"AR1's rank by pooled MASE on the REVISED-basis record restricted to the exhibit's own "
              f"short window ({CONTRAST_RT_START}-{max(score.index)}), over the exhibit's model set. "
              f"Holds the forecast basis fixed while the window varies, which the frozen exhibit does "
              f"not: read against flagship.ar1_rank_long_window (same basis, long window) to attribute "
              f"the inversion to the WINDOW rather than to the real-time-vs-revised convention",
         cites=["S3.6 body: window-vs-basis attribution -- short-window leg"],
         cross_check=f"{SCOREBOARD.name}: rank of ar1 in mean over years >= {CONTRAST_RT_START}")

    print("=== DATA CONFIG ===")
    print(f"  curated panel : {pred['series'].nunique()} series x {social['metric'].nunique()} metric(s), "
          f"{pred['model'].nunique()} models, T in [{mcsres['T_annual'].min()}, {mcsres['T_annual'].max()}]")
    print(f"  selection     : {len(_passed)} of {len(man['series'])} candidates passed; dropped {_dropped or '[]'} "
          f"(vintages must reach {man['criteria']['vintages_cover']}, rebase tol {man['criteria']['rebase_tol']})")
    print(f"  flagship      : {len(score.columns)} models (scoreboard), {len(contrast.index)} (contrast), "
          f"{min(score.index)}-{max(score.index)}")
    print(f"  window/basis  : ar1 rank {_rank_short_revised} of {len(contrast.index)} on REVISED basis over "
          f"{CONTRAST_RT_START}-{max(score.index)} (vs rank on the long window -- isolates window from basis)")
    flush()


if __name__ == "__main__":
    main()
