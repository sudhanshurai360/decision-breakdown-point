"""canonical.py — merge every producer's emitted values, and re-derive them independently.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                     # <repo>/code/gates/ -> <repo>
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
from data_config import _joined  # noqa: E402 -- needed after the sys.path insert directly above
MULTI = ROOT / "multiseries"
RES = MULTI / "results"                       # the multi-series macro panel
FLAG_RES = ROOT / "results"                   # flagship artifacts live at the REPO ROOT, not
                                              # under multiseries/ -- keep the two apart.
ROOT_RES = ROOT / "results"                   # the flagship single-series panel
FRAGDIR = RES / "canonical"
OUT = ROOT / "paper" / "canonical.json"

SENTINEL = 987654                             # selftest_canonical.py's planted wrong value

N_VALUES = 447
N_CHECKED = 230


def _keyset_hash(keys) -> str:
    return hashlib.sha256(",".join(sorted(keys)).encode()).hexdigest()


VALUES_KEYS_HASH = "7c8c22345b8a720896b4161f438cab3453bcbb8c1c5ef0d7947caa94a7a52986"
CHECKED_KEYS_HASH = "5469215ebb76846b9bf36fcc8d3465945520e3eaad5de048f0d74f661bf64c5d"

KNOWN_OUT_DEPENDENT = frozenset({"vintage.window_start", "election.stability_below_null_p05"})
OUT_DEPENDENT_KEYS_HASH = "07a169127f28cf968c08ca31565978f36a53fb7ce1ad6e5f76e2268d804bd011"


def _perturb_value(v):
    """A deterministic, type-preserving change to a canonical `value` (or any nested element)
    that no faithful re-derivation from a producer CSV could reproduce."""
    if isinstance(v, bool):
        return not v
    if v is None:
        return "PERTURBED_NONE"      # None is otherwise a fixed point: any perturbation that maps
                                      # None -> None would silently pass an independence probe that
                                      # only checks equality, hiding a real dependency
    if isinstance(v, float) and np.isnan(v):
        return 12345.678             # NaN*1.5+1 is NaN: also a fixed point
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v * 1.5 + 1 if isinstance(v, float) else v + 1
    if isinstance(v, str):
        return v + "_PERTURBED"
    if isinstance(v, list):
        return [_perturb_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _perturb_value(x) for k, x in v.items()}
    return v


def _run_all_checks(payload):
    """Every registered CHECKS[] result (or the exception's type name), keyed by canonical key."""
    out = {}
    for key, fn in CHECKS.items():
        if key not in payload:
            continue
        try:
            out[key] = fn()
        except Exception as e:   # noqa: BLE001 -- the probe compares outcomes, including failures
            out[key] = f"<{type(e).__name__}>"
    return out


def _independence_probe(doc, real_results):
    """Re-run every registered check against a perturbed temp copy of the merged file and return
    the keys whose result changed (or newly failed) and are NOT in KNOWN_OUT_DEPENDENT."""
    import tempfile
    global OUT, FRAGDIR
    def _perturb_doc(d):
        return {k: (v if k.startswith("_") else
                    {f: (x if f in _PROSE_ONLY_FIELDS else _perturb_value(x)) for f, x in v.items()})
                for k, v in d.items()}
    perturbed = _perturb_doc(doc)
    tmp = Path(tempfile.mkstemp(suffix=".canonical_probe.json")[1])
    tmpdir = Path(tempfile.mkdtemp(suffix=".canonical_probe_frags"))
    real_out, real_fragdir = OUT, FRAGDIR
    try:
        tmp.write_text(json.dumps(perturbed, indent=2, sort_keys=True, default=str))
        if real_fragdir.exists():
            for f in real_fragdir.glob("*.json"):
                (tmpdir / f.name).write_text(json.dumps(_perturb_doc(json.loads(f.read_text())),
                                                        indent=2, sort_keys=True, default=str))
        OUT, FRAGDIR = tmp, tmpdir
        probe_results = _run_all_checks(_payload(perturbed))
    finally:
        OUT, FRAGDIR = real_out, real_fragdir
        try:
            tmp.unlink()
        except OSError:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)
    dependent = []
    for key, real in real_results.items():
        got = probe_results.get(key, "<missing>")
        if not _close(got, real) and key not in KNOWN_OUT_DEPENDENT:
            dependent.append((key, real, got))
    return dependent


def _source_ratchet():
    """Static guards on THIS file's source that the runtime probe cannot express."""
    import ast, inspect
    src = inspect.getsource(sys.modules[__name__])
    tree = ast.parse(src)
    problems = []
    own = {id(x) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "_source_ratchet"
           for x in ast.walk(fn)}   # this function's own literals do not count
    n_lit = sum(1 for n in ast.walk(tree) if id(n) not in own
                if isinstance(n, ast.Constant) and n.value in ("canonical.json", "paper/canonical.json"))
    if n_lit != 1:
        problems.append(f"{n_lit} string constants name the merged file (expected exactly 1: the OUT definition)")
    allowed = {"merge", "_independence_probe"}
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda))]:
        name = getattr(fn, "name", "<lambda>")
        if name in allowed:
            continue
        if any(isinstance(x, ast.Name) and x.id == "FRAGDIR" for x in ast.walk(fn)):
            problems.append(f"FRAGDIR referenced outside merge()/_independence_probe(): in {name}()")
    for n in ast.walk(tree):
        ident = n.id if isinstance(n, ast.Name) else (n.attr if isinstance(n, ast.Attribute) else None)
        if ident in ("lru_cache", "cache") and isinstance(n, (ast.Name, ast.Attribute)):
            src_line = src.splitlines()[n.lineno - 1].strip()
            if "functools" in src_line or ident == "lru_cache":
                problems.append(f"memoisation at line {n.lineno}: {src_line[:80]}")
    return problems


# ============================ re-derivation helpers ==================================


def _shape_ratio(which):
    """Recompute the April-2020 difficulty multiples from the frozen predictions, independently of
    shape_not_magnitude.py's own path."""
    d = pd.read_csv(RES / "breadth_distinct" / "breadth_distinct_predictions.csv")
    _expect_numeric(d["sae"], "breadth_distinct_predictions.csv")
    d["target"] = pd.to_datetime(d["target"])
    diff = d.groupby(["series", "target"])["sae"].min().reset_index()
    diff["ym"] = diff.target.dt.strftime("%Y-%m")
    diff["yr"] = diff.target.dt.year
    med = diff.groupby("series")["sae"].median()
    monthly = (diff[diff.ym == "2020-04"].set_index("series")["sae"] / med).dropna()
    if which == "monthly_max":
        return float(monthly.max())
    if which != "monthly_median":
        raise ValueError(f"_shape_ratio: unregistered 'which' {which!r} -- only 'monthly_max' and "
                          f"'monthly_median' are ever invoked from CHECKS")
    ann = diff.groupby(["series", "yr"])["sae"].mean().reset_index()
    annual = (ann[ann.yr == 2020].set_index("series")["sae"] / ann.groupby("series")["sae"].median()).dropna()
    shared = monthly.index.intersection(annual.index)
    return float(monthly.loc[shared].median())


def _maudit(freq, which):
    """Re-derive the M4 winner-disagreement rates for one frequency from m_audit_<Freq>.csv:
    "flip_mase_rmsse" is the share of series whose MASE and RMSSE winners differ; anything else is
    the share whose five metric winners (MASE/RMSSE/sMAPE/RMSE/MAE) are not unanimous."""
    d = pd.read_csv(RES / f"m_audit_{freq.capitalize()}.csv")
    if which == "flip_mase_rmsse":
        return float((d.win_MASE != d.win_RMSSE).mean())
    win_cols = [f"win_{m}" for m in ("MASE", "RMSSE", "sMAPE", "RMSE", "MAE")]
    return float((d[win_cols].nunique(axis=1) > 1).mean())


def _fredmd_law(which):
    """Re-derive the FRED-MD "predictive law" statistics from fred_md_sweep.csv. "max_abs_rho" is
    the largest absolute Spearman correlation between any shape predictor (abrupt/kurt/vol) and any
    outcome (gap_t/MCS_abs/MCS_sq); any other `which` names an outcome column and returns the R^2
    of an OLS regression of it on all three standardised predictors."""
    r = pd.read_csv(RES / "fred_md_sweep.csv")
    preds, dvs = ["abrupt", "kurt", "vol"], ["gap_t", "MCS_abs", "MCS_sq"]
    if which == "max_abs_rho":
        return float(max(abs(r[[p, dv]].corr(method="spearman").iloc[0, 1]) for p in preds for dv in dvs))
    X = r[preds].fillna(r[preds].median()); X = (X - X.mean()) / X.std().replace(0, 1)
    X = np.column_stack([np.ones(len(X)), X.to_numpy()])
    y = r[which].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _expect_numeric(series: "pd.Series", source: str) -> "pd.Series":
    """Fail loudly unless `series` is numeric. `idxmax()`/`idxmin()`/`.rank()`/`sort_values()` do
    NOT raise on an object/string-dtype column -- they silently operate lexicographically instead
    (e.g. sorting the string "10" before "9"), so a mis-typed column yields a wrong-but-plausible
    answer rather than an error."""
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"{source}: expected column {series.name!r} to be numeric, got dtype {series.dtype} "
            f"-- idxmax/idxmin/rank/sort_values would silently operate lexicographically instead "
            f"of numerically on this dtype, producing a wrong-but-plausible answer"
        )
    return series


def _sc_argmax(what):
    s = pd.read_csv(RES / "social_choice.csv")
    r = s.loc[_expect_numeric(s.concentration, "social_choice.csv").idxmax()]
    return f"{r.series}/{r.metric}" if what == "cell" else str(r[what])


def _density_sorted_by_n_models():
    """m4_density_curve.csv sorted by n_models -- shared by 3 CHECKS entries below (min/max pool
    flip_rate, monotonicity)."""
    d = pd.read_csv(RES / "m4_density_curve.csv")
    _expect_numeric(d.n_models, "m4_density_curve.csv")
    return d.sort_values("n_models")


def _vint():
    return pd.read_csv(RES / "state_eb_vintage.csv")


def _vint_window_start():
    """Independent re-derivation of vintage.window_start: the first of the month containing the
    panel start. The reported arms begin at panel_start itself, with no look-back offset."""
    doc = json.loads(OUT.read_text())
    panel_start = pd.Timestamp(doc["vintage.window_start"]["panel_start"])
    win_start = panel_start.normalize().replace(day=1)
    return str(win_start.date())


def _mcspower_n_trials():
    """Independent re-derivation of mcspower.n_trials: the total number of (series, model) MCS
    trials across the curated panel, recounted from the raw prediction file."""
    d = pd.read_csv(RES / "breadth_distinct" / "breadth_distinct_predictions.csv")
    n = 0; kept = 0
    for sid, g in d.groupby("series"):
        ms = sorted(g.model.unique()); ys = sorted(g.year.unique())
        M = np.array([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in ms] for y in ys])
        if np.all(np.isfinite(M)) and M.shape[0] >= 20:
            n += M.shape[1]; kept += 1
    if kept != 12:
        raise RuntimeError(
            f"_mcspower_n_trials(): expected 12 curated series with fully rectangular, fully-finite "
            f"loss matrices, got {kept} -- see load_panels()'s matching guard in mcs_power.py."
        )
    return n


def _mcspower_n_panels():
    """Companion-field re-derivation of mcspower.n_trials's own `n_panels` field."""
    d = pd.read_csv(RES / "breadth_distinct" / "breadth_distinct_predictions.csv")
    kept = 0
    for sid, g in d.groupby("series"):
        ms = sorted(g.model.unique()); ys = sorted(g.year.unique())
        M = np.array([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in ms] for y in ys])
        if np.all(np.isfinite(M)) and M.shape[0] >= 20:
            kept += 1
    return kept


def _rt():
    """The whole realised_t.csv."""
    return pd.read_csv(RES / "realised_t.csv")


def _rt_flag(col):
    """One column of the FLAGSHIP row of realised_t.csv, selected BY NAME, not by position."""
    d = pd.read_csv(RES / "realised_t.csv")
    return d.loc[d.panel == "FLAGSHIP", col].iloc[0]


def _proto(col):
    """The superseded-protocol comparison row (§7 disclosure). Never cite as current."""
    return pd.read_csv(RES / "state_eb_alt_protocol.csv")[col].iloc[0]


def _ebwin(window, col):
    """One row of state_eb_decision.py's window-contrast CSV, selected by window name."""
    d = pd.read_csv(RES / "state_eb_window_comparison.csv")
    return d.loc[d.window == window, col].iloc[0]


def _realtime(key, col):
    """One row of sahm_evans_summary.csv, selected by rule (SAHM/EVANS) and protocol."""
    d = pd.read_csv(RES / "sahm_evans_summary.csv")
    rule, protocol = key
    return d.loc[(d.rule == rule) & (d.protocol == protocol), col].iloc[0]


def _ns(config, field):
    """One field of one regime row of null_sensitivity.csv, selected BY CONFIG NAME."""
    d = pd.read_csv(RES / "null_sensitivity.csv")
    m = d[d.config == config]
    if len(m) != 1:
        raise LookupError(f"null_sensitivity.csv: expected exactly 1 row for config={config!r}, got {len(m)}")
    return m.iloc[0][field]


def _flag(col):
    """One column of the single-row flagship window from results/flagship_fragility.csv."""
    d = pd.read_csv(FLAG_RES / "flagship_fragility.csv")
    m = d[d.window == "1988-2025"]
    if len(m) != 1:
        raise LookupError(f"flagship_fragility.csv: expected 1 row for window 1988-2025, got {len(m)}")
    return m.iloc[0][col]


def _mcssize_row(K, T, block):
    """The size-grid row for one (K, T, block) cell, selected by key rather than by position."""
    d = pd.read_csv(RES / "mcs_size.csv")
    m = d[(d.K == K) & (d["T"] == T) & (d.block == block)]
    if len(m) != 1:
        raise LookupError(f"mcs_size.csv: expected exactly 1 row for K={K},T={T},block={block}, got {len(m)}")
    return m.iloc[0]


def _mcssize_sweep():
    d = pd.read_csv(RES / "mcs_size_alpha_sweep.csv")
    _expect_numeric(d["alpha"], "mcs_size_alpha_sweep.csv")
    return d


def _comb_win():
    """The single combination_benchmark.csv row where the combination beats the best single."""
    d = pd.read_csv(RES / "combination_benchmark.csv")
    m = d[d.comb_wins]
    if len(m) != 1:
        raise LookupError(f"combination_benchmark.csv: expected exactly 1 comb_wins row, got {len(m)}")
    return m.iloc[0]


def _manifest():
    return json.loads((MULTI / "data" / "raw" / "breadth_manifest_distinct.json").read_text())


def _elec_mcs(metric, col):
    """social_choice.csv joined to mcs_test.csv on series, filtered to one metric -- the exact join
    election.spearman_mcs_stability_* re-derives, kept independent by reading both CSVs fresh here."""
    R = pd.read_csv(RES / "social_choice.csv")
    mt = pd.read_csv(RES / "mcs_test.csv")
    j = R[R.metric == metric].merge(mt[["series", col]], on="series").dropna(subset=[col, "winner_stability"])
    return spearmanr(j[col], j["winner_stability"])


def _mcspow(col, pct):
    d = pd.read_csv(RES / "mcs_power.csv")
    m = d[d.edge_pct == pct]
    if len(m) != 1:
        raise LookupError(f"mcs_power.csv: expected exactly 1 row for edge_pct={pct}, got {len(m)}")
    return m.iloc[0][col]


def _flag_score():
    return pd.read_csv(ROOT_RES / "extension_scoreboard_1988-2025.csv", index_col=0)


def _contrast_rt_start(columns):
    """First year of the window-inversion exhibit's SHORT (real-time) pool, PARSED from the
    exhibit's own `pooled_RT_<start>-<end>` column name -- not retyped -- so a regenerated exhibit
    on a different window cannot silently disagree with this gate."""
    for c in columns:
        m = re.fullmatch(r"pooled_RT_(\d{4})-\d{2}", str(c))
        if m:
            return int(m.group(1))
    raise ValueError(f"no column matching 'pooled_RT_<YYYY>-<YY>' found in {list(columns)!r} -- "
                      f"contrast_exhibit.csv's real-time window column is missing or renamed")


_LLM_TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]


def _llm_loss():
    """Independently re-derived from the cached Open LLM Leaderboard parquet, mirroring
    ml_witness.py's own S/L/order construction exactly (score->loss flip, mean-loss ranking)."""
    df = pd.read_parquet(MULTI / "data" / "open_llm.parquet")
    S = df[_LLM_TASKS].to_numpy(float)
    L = 100.0 - S
    order = np.argsort(L.mean(1))
    return L, order


def _shape_annual_median_x():
    """Matches shape_not_magnitude.py's formula for shape.annual_median_x exactly -- NOT
    `_shape_ratio("monthly_median")`, which restricts to the index also present in the MONTHLY
    ratio. The producer does not apply that restriction to this key: its own ann_ratio is computed
    over every series with a 2020 annual value, unrestricted."""
    d = pd.read_csv(RES / "breadth_distinct" / "breadth_distinct_predictions.csv")
    _expect_numeric(d["sae"], "breadth_distinct_predictions.csv")
    d["target"] = pd.to_datetime(d["target"])
    diff = d.groupby(["series", "target"])["sae"].min().reset_index()
    diff["yr"] = diff.target.dt.year
    ann = diff.groupby(["series", "yr"])["sae"].mean().reset_index()
    ann_ratio = (ann[ann.yr == 2020].set_index("series")["sae"]
                 / ann.groupby("series")["sae"].median()).dropna()
    return float(ann_ratio.median())


CHECKS = {
    "comb.beats_best_single": lambda: int(pd.read_csv(RES/"combination_benchmark.csv").comb_wins.sum()),
    "comb.beats_median_single": lambda: int(pd.read_csv(RES/"combination_benchmark.csv").comb_beats_median.sum()),
    "comb.in_mcs": lambda: int(pd.read_csv(RES/"combination_benchmark.csv").comb_in_mcs.sum()),
    "comb.median_kstar_vs_best": lambda: float(pd.read_csv(RES/"combination_benchmark.csv").k_star.median()),
    "comb.median_rank_of_7": lambda: float(pd.read_csv(RES/"combination_benchmark.csv").comb_rank_of_7.median()),
    "comb.median_winner_stability": lambda: float(pd.read_csv(RES/"combination_benchmark.csv").winner_stability.median()),
    # The headline size row is selected BY ITS (K, T, block) KEY, never by row position -- the
    # grid's order is a property of the producer's loop, not of the value.
    "mcssize.size_at_nominal_10": lambda: round(float(_mcssize_row(6, 30, 3)["size"]), 3),
    "mcssize.alpha_sweep_nonidentified": lambda: int(_mcssize_sweep().sort_values("alpha").iloc[0].nonidentified),
    "mcssize.alpha_sweep_invariant": lambda: bool(_mcssize_sweep().nonidentified.nunique() == 1),
    "prop.spearman_kstar_t": lambda: round(float(spearmanr(_rt().k_star, _rt().t_binding).statistic), 3),
    "comb.rank6_series": lambda: int((pd.read_csv(RES/"combination_benchmark.csv").comb_rank_of_7 == 6).sum()),
    "comb.sn_worst_ratio_min": lambda: float(_expect_numeric(pd.read_csv(RES/"combination_benchmark.csv")
        .query("worst_single == 'seasonal_naive'").worst_best_ratio, "combination_benchmark.csv").min()),
    "comb.sn_worst_series": lambda: int((pd.read_csv(RES/"combination_benchmark.csv").worst_single == "seasonal_naive").sum()),
    "dmcs.mean_set_decision": lambda: float(pd.read_csv(RES/"dmcs.csv").dMCS_decision.mean()),
    "dmcs.mean_set_statistical": lambda: float(pd.read_csv(RES/"dmcs.csv").MCS_stat.mean()),
    "dmcs.separates": lambda: int(pd.read_csv(RES/"dmcs.csv").separates.sum()),
    "dmcs.n_states": lambda: len(pd.read_csv(RES/"dmcs.csv")),
    "dmcs.n_eligible": lambda: int((pd.read_parquet(MULTI/"data"/"state_ur.parquet")
                                     .count() >= 300).sum()),
    "dmcs.tied_both": lambda: int(((pd.read_csv(RES/"dmcs.csv").dMCS_decision
                                     >= pd.read_csv(RES/"dmcs.csv").MCS_stat)
                                    & (pd.read_csv(RES/"dmcs.csv").dMCS_decision > 1)).sum()),
    "dmcs.single_decision": lambda: int((pd.read_csv(RES/"dmcs.csv").dMCS_decision == 1).sum()),
    "dmcs.wider_from_singleton": lambda: int((
        (pd.read_csv(RES/"dmcs.csv").dMCS_decision > pd.read_csv(RES/"dmcs.csv").MCS_stat_near)
        & (pd.read_csv(RES/"dmcs.csv").MCS_stat_near == 1)).sum()),
    "bestarm.spearman_gap_mcs": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").gap2_t,
                                                         pd.read_csv(RES/"best_arm.csv").MCS,
                                                         nan_policy="omit").statistic),
    "bestarm.spearman_gap_mcs_p": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").gap2_t,
                                                           pd.read_csv(RES/"best_arm.csv").MCS,
                                                           nan_policy="omit").pvalue),
    "bestarm.spearman_H_mcs": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").H,
                                                       pd.read_csv(RES/"best_arm.csv").MCS,
                                                       nan_policy="omit").statistic),
    "bestarm.spearman_H_mcs_p": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").H,
                                                         pd.read_csv(RES/"best_arm.csv").MCS,
                                                         nan_policy="omit").pvalue),
    "bestarm.spearman_gap_H": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").gap2_t,
                                                       pd.read_csv(RES/"best_arm.csv").H,
                                                       nan_policy="omit").statistic),
    "bestarm.spearman_sci_mcs": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").sci,
                                                         pd.read_csv(RES/"best_arm.csv").MCS,
                                                         nan_policy="omit").statistic),
    "bestarm.spearman_sci_mcs_p": lambda: float(spearmanr(pd.read_csv(RES/"best_arm.csv").sci,
                                                           pd.read_csv(RES/"best_arm.csv").MCS,
                                                           nan_policy="omit").pvalue),
    # --- m4sub.* / density.*: S9's discriminant-validity controls ------------------------
    "m4sub.n_submissions": lambda: len(pd.read_csv(RES/"m4_submissions_leaderboard.csv", index_col=0)),
    "m4sub.distinct_winners": lambda: len({_expect_numeric(
        pd.read_csv(RES/"m4_submissions_leaderboard.csv", index_col=0)[mt], "m4_submissions_leaderboard.csv").idxmin()
                                            for mt in ["sMAPE", "MAPE", "MASE", "RMSSE"]}),
    "m4sub.pareto_dominant": lambda: len({_expect_numeric(
        pd.read_csv(RES/"m4_submissions_leaderboard.csv", index_col=0)[mt], "m4_submissions_leaderboard.csv").idxmin()
                                           for mt in ["sMAPE", "MAPE", "MASE", "RMSSE"]}) == 1,
    "m4sub.rank_corr_smape_rmsse": lambda: float(
        _expect_numeric(pd.read_csv(RES/"m4_submissions_leaderboard.csv", index_col=0)["sMAPE"],
                         "m4_submissions_leaderboard.csv").rank().corr(
            _expect_numeric(pd.read_csv(RES/"m4_submissions_leaderboard.csv", index_col=0)["RMSSE"],
                             "m4_submissions_leaderboard.csv").rank())),
    "density.n_models_max": lambda: int(_expect_numeric(
        pd.read_csv(RES/"m4_density_curve.csv").n_models, "m4_density_curve.csv").max()),
    "density.flip_rate_min_pool": lambda: float(_density_sorted_by_n_models().flip_rate.iloc[0]),
    "density.flip_rate_max_pool": lambda: float(_density_sorted_by_n_models().flip_rate.iloc[-1]),
    "density.flip_rate_monotonic_increasing": lambda: bool(np.all(np.diff(
        _density_sorted_by_n_models().flip_rate.to_numpy()) > 0)),
    "eb.alt_protocol_episodes": lambda: int(_proto("episodes")),
    "eb.alt_protocol_near_trigger_pct": lambda: float(_proto("near_trigger_pct")),
    "eb.episodes": lambda: int(pd.read_csv(RES/"state_eb_decision.csv").flip_episodes.sum()),
    "eb.fullwindow_episodes": lambda: int(_ebwin("unrestricted_1991_01", "episodes")),
    "eb.fullwindow_states_divergent": lambda: int(_ebwin("unrestricted_1991_01", "states_divergent")),
    "eb.fullwindow_flip_months": lambda: int(_ebwin("unrestricted_1991_01", "flip_months")),
    "eb.prestatute_flip_months": lambda: int(_ebwin("unrestricted_1991_01", "flip_months")
                                              - _ebwin("statutory_1993_03", "flip_months")),
    "realtime.sahm_divergent": lambda: int(_realtime(("SAHM", "published"), "n_divergent")),
    "realtime.evans_divergent": lambda: int(_realtime(("EVANS", "published"), "n_divergent")),
    "realtime.sahm_divergent_all_forecast": lambda: int(_realtime(("SAHM", "all_forecast"), "n_divergent")),
    "realtime.evans_divergent_all_forecast": lambda: int(_realtime(("EVANS", "all_forecast"), "n_divergent")),
    "eb.knife_edge_pct": lambda: float(pd.read_csv(RES/"state_eb_summary.csv").knife_edge_pct.iloc[0]),
    "eb.near_trigger_pct": lambda: float(pd.read_csv(RES/"state_eb_summary.csv").near_trigger_pct.iloc[0]),
    "eb.states_divergent": lambda: int((pd.read_csv(RES/"state_eb_decision.csv").flip_episodes > 0).sum()),
    "eb.states_total": lambda: len(pd.read_csv(RES/"state_eb_decision.csv")),
    "eb.unconditional_pct": lambda: float(pd.read_csv(RES/"state_eb_summary.csv").unconditional_pct.iloc[0]),
    "election.concentration_max": lambda: float(_expect_numeric(
        pd.read_csv(RES/"social_choice.csv").concentration, "social_choice.csv").max()),
    "election.concentration_max_cell_lo": lambda: float(_sc_argmax("conc_lo")),
    "election.reversals": lambda: int(pd.read_csv(RES/"social_choice.csv").reversal.sum()),
    "election.reversals_lo": lambda: int((
        lambda t: (t.reversal & ~t.reversal_ambiguous).sum())(pd.read_csv(RES/"social_choice.csv"))),
    "election.reversals_hi": lambda: int((
        lambda t: (t.reversal & ~t.reversal_ambiguous).sum() + t.reversal_ambiguous.sum())(
        pd.read_csv(RES/"social_choice.csv"))),
    "election.pivot_2020": lambda: int((
        lambda t: t[t.pivotal_year.notna()].pivot_2020.sum())(pd.read_csv(RES/"social_choice.csv"))),
    "election.pivot_2020_lo": lambda: int((
        lambda t: t[t.pivotal_year.notna()].pivot_2020_certain.sum())(pd.read_csv(RES/"social_choice.csv"))),
    "election.pivot_2020_hi": lambda: int((
        lambda t: t[t.pivotal_year.notna()].pivot_2020_possible.sum())(pd.read_csv(RES/"social_choice.csv"))),
    "election.stability_median": lambda: float(pd.read_csv(RES/"social_choice.csv").winner_stability.dropna().median()),
    # --- flagship.* -----------------------------------------------------------------------
    "flagship.k_star": lambda: int(_flag("k_star")),
    "flagship.pooled_winner": lambda: str(_flag("pooled_winner")),
    "flagship.winner_stability": lambda: round(float(_flag("winner_stability")), 3),
    "flagship.responsible_years": lambda: str(_flag("responsible")),
    "flagship.t_binding": lambda: round(float(_rt_flag("t_binding")), 3),
    "flagship.t_max": lambda: round(float(_rt_flag("t_max")), 3),
    "flagship.mcs_10": lambda: str(_flag("MCS_10")),
    "flagship.mcs_25": lambda: str(_flag("MCS_25")),
    "fredmd.band_alpha10": lambda: [
        int(_expect_numeric(pd.read_csv(RES/"fredmd_robustness_band.csv")
            .pipe(lambda d: d[d.alpha == 0.10])["min"], "fredmd_robustness_band.csv").min()),
        int(_expect_numeric(pd.read_csv(RES/"fredmd_robustness_band.csv")
            .pipe(lambda d: d[d.alpha == 0.10])["max"], "fredmd_robustness_band.csv").max())],
    "fredmd.count_alpha05": lambda: round(float(pd.read_csv(RES/"fredmd_robustness_band.csv")
        .pipe(lambda d: d[(d.alpha == 0.05) & (d.block == 3) & (d.B == 2000)])["mean"].iloc[0]), 2),
    "fredmd.count_alpha25": lambda: round(float(pd.read_csv(RES/"fredmd_robustness_band.csv")
        .pipe(lambda d: d[(d.alpha == 0.25) & (d.block == 3) & (d.B == 2000)])["mean"].iloc[0]), 2),
    "mcspower.n_trials": _mcspower_n_trials,
    "nullsens.rho_delta_p05": lambda: round(float(abs(_ns("rho08", "p05") - _ns("rho05", "p05"))), 4),
    "nullsens.ar_delta_p05": lambda: round(float(_ns("ar06", "p05") - _ns("iid", "p05")), 4),
    "nullsens.p05_ar06": lambda: round(float(_ns("ar06", "p05")), 4),
    "nullsens.p05_iid": lambda: round(float(_ns("iid", "p05")), 4),
    "nullsens.p05_rho05": lambda: round(float(_ns("rho05", "p05")), 4),
    "nullsens.p05_rho08": lambda: round(float(_ns("rho08", "p05")), 4),
    "prop.hac_over_ceiling": lambda: int(_rt().hac_over_ceiling.sum()),
    "prop.median_slack": lambda: round(float(_rt().slack.median()), 3),
    "prop.panels_sig_small_kstar": lambda: int(((_rt().n_sig_5pct > 0) & (_rt().k_star <= 3)).sum()),
    "prop.realised_violations": lambda: int(_rt().violates.sum()),
    "shape.hardest_april2020": lambda: int((pd.read_csv(RES/"shape_not_magnitude.csv").ym == "2020-04").sum()),
    "shape.hardest_gfc": lambda: int(pd.read_csv(RES/"shape_not_magnitude.csv").ym.str.startswith(("2008","2009")).sum()),
    "shape.monthly_max_x": lambda: _shape_ratio("monthly_max"),
    "vintage.episodes_initial": lambda: int(_vint().episodes_initial.sum()),
    "vintage.episodes_revised": lambda: int(_vint().episodes_revised.sum()),
    "vintage.flip_months_initial": lambda: int(_vint().flip_months_initial.sum()),
    "vintage.flip_months_revised": lambda: int(_vint().flip_months_revised.sum()),
    "vintage.states_divergent_initial": lambda: int((_vint().flip_months_initial > 0).sum()),
    "vintage.states_divergent_revised": lambda: int((_vint().flip_months_revised > 0).sum()),
    "vintage.window_start": _vint_window_start,

    # --- comb.* (combination_benchmark.csv) ------------------------------------------------
    "comb.stability_null_median": lambda: float(pd.read_csv(RES/"combination_benchmark.csv").null_median.median()),
    "comb.stability_null_p05": lambda: float(pd.read_csv(RES/"combination_benchmark.csv").null_p05.median()),
    "comb.winning_series": lambda: str(_comb_win().series),
    "comb.winning_series_null_p05": lambda: round(float(_comb_win().null_p05), 3),
    "comb.winning_series_stability": lambda: round(float(_comb_win().winner_stability), 3),
    "comb.winning_series_verdict": lambda: str(_comb_win().fragility_verdict),

    # --- data.* (data_config.py's own frozen/regenerated artifacts) ------------------------
    "data.curated_T_max": lambda: int(_expect_numeric(
        pd.read_csv(RES/"mcs_monthly_resolution.csv").T_annual, "mcs_monthly_resolution.csv").max()),
    "data.curated_T_min": lambda: int(_expect_numeric(
        pd.read_csv(RES/"mcs_monthly_resolution.csv").T_annual, "mcs_monthly_resolution.csv").min()),
    "data.curated_candidates": lambda: int(len(_manifest()["series"])),
    "data.curated_dropped": lambda: _joined(s for s, r in _manifest()["series"].items() if r["status"] != "PASS"),
    "data.curated_metrics": lambda: _joined(pd.read_csv(RES/"social_choice.csv")["metric"]),
    "data.curated_models": lambda: _joined(pd.read_csv(RES/"breadth_distinct"/"breadth_distinct_predictions.csv")["model"]),
    "data.curated_n_models": lambda: int(pd.read_csv(RES/"breadth_distinct"/"breadth_distinct_predictions.csv")["model"].nunique()),
    "data.curated_rebase_tol": lambda: float(_manifest()["criteria"]["rebase_tol"]),
    "data.curated_series": lambda: _joined(pd.read_csv(RES/"breadth_distinct"/"breadth_distinct_predictions.csv")["series"]),
    "data.curated_vintage_cutoff": lambda: str(_manifest()["criteria"]["vintages_cover"]),
    "data.curated_year_max": lambda: int(_expect_numeric(
        pd.read_csv(RES/"breadth_distinct"/"breadth_distinct_predictions.csv")["year"],
        "breadth_distinct_predictions.csv").max()),
    "data.flagship_contrast_models": lambda: _joined(pd.read_csv(ROOT_RES/"contrast_exhibit.csv", index_col=0).index),
    "data.flagship_scoreboard_models": lambda: _joined(_flag_score().columns),
    "data.flagship_year_max": lambda: int(max(_expect_numeric(
        _flag_score().index, "extension_scoreboard_1988-2025.csv"))),
    "data.flagship_year_min": lambda: int(min(_expect_numeric(
        _flag_score().index, "extension_scoreboard_1988-2025.csv"))),
    "flagship.ar1_rank_short_window_revised": lambda: int(
        (lambda _ce: _flag_score().loc[
            [y for y in _flag_score().index if y >= _contrast_rt_start(_ce.columns)],
            list(_ce.index)]
        .mean().sort_values().index.get_loc("ar1") + 1)(
            pd.read_csv(ROOT_RES/"contrast_exhibit.csv", index_col=0))),

    # --- election.* (social_choice.csv / social_choice_weights.csv / mcs_test.csv) ---------
    "election.conc_reported_not_max": lambda: int((
        lambda t: ((t.concentration - t.conc_hi).abs() > 1e-9).sum())(
        pd.read_csv(RES/"social_choice.csv").pipe(lambda d: d[d.n_binding_tied > 1]))),
    "election.conc_tied_cells": lambda: int((pd.read_csv(RES/"social_choice.csv").n_binding_tied > 1).sum()),
    "election.concentration_gt1": lambda: int((pd.read_csv(RES/"social_choice.csv").concentration > 1).sum()),
    "election.condorcet_cycle_cells": lambda: int((pd.read_csv(RES/"social_choice.csv").condorcet_status == "cycle").sum()),
    "election.condorcet_tie_cells": lambda: int((pd.read_csv(RES/"social_choice.csv").condorcet_status == "tie").sum()),
    "election.condorcet_winner_cells": lambda: int((pd.read_csv(RES/"social_choice.csv").condorcet_status == "winner").sum()),
    "election.pi_concentration": lambda: float(_expect_numeric(pd.read_csv(RES/"social_choice.csv")
        .pipe(lambda d: d[d.series == "PI"]).concentration, "social_choice.csv").min()),
    "election.spearman_kstar_stability": lambda: float(
        pd.read_csv(RES/"social_choice.csv")[["k_star", "winner_stability"]].dropna()
        .corr(method="spearman").iloc[0, 1]),
    "election.spearman_mcs_stability_mase": lambda: float(_elec_mcs("mase", "MCS_abs_10").statistic),
    "election.spearman_mcs_stability_mase_p": lambda: float(_elec_mcs("mase", "MCS_abs_10").pvalue),
    "election.spearman_mcs_stability_rmsse": lambda: float(_elec_mcs("rmsse", "MCS_rmsse_10").statistic),
    "election.spearman_mcs_stability_rmsse_p": lambda: float(_elec_mcs("rmsse", "MCS_rmsse_10").pvalue),
    "election.stability_below_060": lambda: int((pd.read_csv(RES/"social_choice.csv").winner_stability < 0.60).sum()),
    # threshold is itself a canonical value (a Monte Carlo estimate with no CSV of its own) -- reads
    # the merged output rather than re-deriving it, since there is no second path to the threshold.
    "election.stability_below_null_p05": lambda: int((
        pd.read_csv(RES/"social_choice.csv").winner_stability
        < json.loads(OUT.read_text())["election.stability_null_p05"]["value"]).sum()),
    "election.stability_min": lambda: float(_expect_numeric(
        pd.read_csv(RES/"social_choice.csv").winner_stability, "social_choice.csv").min()),
    "election.weight_conc_gt1_equal": lambda: int((pd.read_csv(RES/"social_choice_weights.csv").conc_equal_year > 1).sum()),
    "election.weight_conc_gt1_monthwtd": lambda: int((pd.read_csv(RES/"social_choice_weights.csv").conc_month_wtd > 1).sum()),
    "election.weight_kstar_changes": lambda: int((
        lambda w: (w.k_equal_year != w.k_month_wtd).sum())(pd.read_csv(RES/"social_choice_weights.csv"))),
    "election.weight_tcu_conc_equal": lambda: float(pd.read_csv(RES/"social_choice_weights.csv")
        .pipe(lambda w: w[(w.series == "TCU") & (w.metric == "mase")]).conc_equal_year.iloc[0]),
    "election.weight_tcu_conc_monthwtd": lambda: float(pd.read_csv(RES/"social_choice_weights.csv")
        .pipe(lambda w: w[(w.series == "TCU") & (w.metric == "mase")]).conc_month_wtd.iloc[0]),
    "election.weight_winner_changes": lambda: int((
        lambda w: (w.winner_equal_year != w.winner_month_wtd).sum())(pd.read_csv(RES/"social_choice_weights.csv"))),

    # --- fredmd.* (fred_md_sweep.csv) -------------------------------------------------------
    "fredmd.identified_complement": lambda: int((pd.read_csv(RES/"fred_md_sweep.csv").MCS_abs == 1).sum()),
    "fredmd.nonidentified": lambda: int((pd.read_csv(RES/"fred_md_sweep.csv").MCS_abs > 1).sum()),
    "fredmd.spearman_gap_mcs": lambda: float(spearmanr(
        pd.read_csv(RES/"fred_md_sweep.csv")["gap_t"], pd.read_csv(RES/"fred_md_sweep.csv")["MCS_abs"]).statistic),
    "fredmd.squared_nonidentified": lambda: int((pd.read_csv(RES/"fred_md_sweep.csv").MCS_sq > 1).sum()),
    "fredmd.total": lambda: len(pd.read_csv(RES/"fred_md_sweep.csv")),
    "fredmd.law_max_abs_rho": lambda: _fredmd_law("max_abs_rho"),
    "fredmd.law_r2_gap_t": lambda: _fredmd_law("gap_t"),
    "fredmd.law_r2_mcs_abs": lambda: _fredmd_law("MCS_abs"),
    "fredmd.law_r2_mcs_sq": lambda: _fredmd_law("MCS_sq"),

    # --- llm.* / flagship.* / mcs.* misc ----------------------------------------------------
    "llm.n_models": lambda: int(len(pd.read_parquet(MULTI/"data"/"open_llm.parquet"))),
    "llm.mean_cross_task_corr": lambda: float(
        np.corrcoef(_llm_loss()[0].T)[np.triu_indices(len(_LLM_TASKS), 1)].mean()),
    "llm.mean_cross_task_corr_top30": lambda: round(float(
        np.corrcoef(_llm_loss()[0][_llm_loss()[1][:30]].T)[np.triu_indices(len(_LLM_TASKS), 1)].mean()), 4),
    "flagship.ar1_rank_long_window": lambda: int(_flag_score().mean(0).sort_values().index.get_loc("ar1") + 1),
    "mcs.annual_nonidentified": lambda: int((pd.read_csv(RES/"mcs_monthly_resolution.csv").MCS_annual > 1).sum()),
    "mcs.curated_nonidentified": lambda: int((pd.read_csv(RES/"mcs_test.csv").MCS_abs_10 > 1).sum()),
    "mcs.curated_total": lambda: len(pd.read_csv(RES/"mcs_test.csv")),
    "mcs.monthly_nonidentified_range": lambda: [
        int(min(int((pd.read_csv(RES/"mcs_monthly_resolution.csv")[f"MCS_mon_b{b}"] > 1).sum())
                for b in (1, 2, 3, 6, 12, 24))),
        int(max(int((pd.read_csv(RES/"mcs_monthly_resolution.csv")[f"MCS_mon_b{b}"] > 1).sum())
                for b in (1, 2, 3, 6, 12, 24))),
    ],

    # --- mcspower.* (mcs_power.csv) ---------------------------------------------------------
    **{f"mcspower.studentized_power_{p}pct": (lambda p=p: float(_mcspow("power_corrected", p)))
       for p in (0, 2, 5, 10, 30, 50)},
    **{f"mcspower.conditional_power_{p}pct": (lambda p=p: round(float(_mcspow("power_conditional", p)), 4))
       for p in (0, 2, 5, 10, 30, 50)},
    **{f"mcspower.recentred_power_{p}pct": (lambda p=p: float(_mcspow("power_recentred", p)))
       for p in (0, 2, 5, 10, 30, 50)},
    "mcspower.zero_edge_detection": lambda: float(_mcspow("power_corrected", 0)),

    # --- shape.* (reuses the _shape_ratio helper) ------------------------------------------
    "shape.annual_median_x": lambda: _shape_annual_median_x(),
    "shape.monthly_median_x": lambda: _shape_ratio("monthly_median"),

    # --- regime.* / fredmd.tcode_* -- re-derivation drivers not present in the original producer files
    "regime.nonidentified_full": lambda: int((pd.read_csv(RES/"regime_robustness.csv").MCS_full > 1).sum()),
    "regime.nonidentified_excovid": lambda: int((pd.read_csv(RES/"regime_robustness.csv").MCS_ex_covid > 1).sum()),
    "regime.placebo_nonidentified_pct": lambda: float(
        (pd.read_csv(RES/"regime_robustness.csv").MCS_placebo_mean > 1).mean() * 100),
    "regime.placebo_below_excovid_median": lambda: float(
        pd.read_csv(RES/"regime_robustness.csv").placebo_pctile_of_excovid.median()),
    "fredmd.tcode_subset": lambda: int(pd.read_csv(RES/"fred_md_sweep.csv").MCS_abs_tcode.notna().sum()),
    "fredmd.nonidentified_levels_subset": lambda: int(
        pd.read_csv(RES/"fred_md_sweep.csv").pipe(lambda d: (d.MCS_abs > 1)[d.MCS_abs_tcode.notna()]).sum()),
    "fredmd.nonidentified_tcode": lambda: int((pd.read_csv(RES/"fred_md_sweep.csv").MCS_abs_tcode > 1).sum()),

    "election.ar1_coefficient_empirical": lambda: round(float(
        pd.read_csv(RES/"election_null_ar1_coefficients.csv").query("basis=='differential'").ar1_estimate.median()), 4),
    "election.stability_null_p05_ar1": lambda: round(float(pd.read_csv(RES/"election_null_ar1.csv").null_p05.iloc[0]), 3),
    "election.stability_null_median_ar1": lambda: round(float(pd.read_csv(RES/"election_null_ar1.csv").null_median.iloc[0]), 3),
    "election.stability_below_null_p05_ar1": lambda: int(pd.read_csv(RES/"election_null_ar1.csv").below_p05.iloc[0]),
    "mcssize.size_circular": lambda: round(float(pd.read_csv(RES/"mcs_circular_check.csv").rate_circular.iloc[0]), 4),
    "mcs.circular_three_series_match": lambda: int(
        (pd.read_csv(RES/"mcs_circular_cross_check.csv").ours_circular
         == pd.read_csv(RES/"mcs_circular_cross_check.csv").arch).sum()),
    "mcs.curated_nonidentified_circular": lambda: int(
        round(float(pd.read_csv(RES/"mcs_circular_check.csv").curated_mean_circular.iloc[0]))),
    "m4sub.k_star": lambda: int(pd.read_csv(RES/"m4_perseries_kstar.csv").k_star.iloc[0]),
    "m4sub.winner_stability": lambda: round(float(pd.read_csv(RES/"m4_perseries_kstar.csv").winner_stability.iloc[0]), 4),
    "election.ar1_delta_p05": lambda: round(float(pd.read_csv(RES/"election_null_ar1.csv").delta_p05.iloc[0]), 4),
    "m4sub.gap_smape_pct": lambda: round(float(pd.read_csv(RES/"m4_margins.csv").query("metric=='sMAPE'").gap_pct.iloc[0]), 2),
    "m4sub.gap_mape_pct": lambda: round(float(pd.read_csv(RES/"m4_margins.csv").query("metric=='MAPE'").gap_pct.iloc[0]), 2),
    "m4sub.gap_mase_pct": lambda: round(float(pd.read_csv(RES/"m4_margins.csv").query("metric=='MASE'").gap_pct.iloc[0]), 2),
    "m4sub.gap_rmsse_pct": lambda: round(float(pd.read_csv(RES/"m4_margins.csv").query("metric=='RMSSE'").gap_pct.iloc[0]), 2),
    "mcssize.arch_gap_moving_block": lambda: round(float(pd.read_csv(RES/"mcs_circular_check.csv").gap_moving_block_canonical.iloc[0]), 4),
    "mcssize.arch_gap_circular": lambda: round(float(pd.read_csv(RES/"mcs_circular_check.csv").gap_circular_canonical.iloc[0]), 4),
    **{f"maudit.flip_mase_rmsse_{fx}": (lambda fx=fx: _maudit(fx, "flip_mase_rmsse"))
       for fx in ("weekly", "monthly")},
    **{f"maudit.any_flip_{fx}": (lambda fx=fx: _maudit(fx, "any_flip"))
       for fx in ("weekly", "monthly")},

    "election.ar1_flagged_cells_at_or_below_pooled": lambda: bool(
        pd.read_csv(RES/"election_null_ar1_flagged_cells.csv").own_ar1_below_pooled.all()),
    "election.tcu_own_ar1_median": lambda: round(float(
        pd.read_csv(RES/"election_null_ar1_flagged_cells.csv").query("series=='TCU'").own_ar1_median.iloc[0]), 3),
    "election.houst_own_ar1_median": lambda: round(float(
        pd.read_csv(RES/"election_null_ar1_flagged_cells.csv").query("series=='HOUST'").own_ar1_median.iloc[0]), 3),
    "election.houst_ar1_excess_over_pooled": lambda: round(
        float(pd.read_csv(RES/"election_null_ar1_flagged_cells.csv").query("series=='HOUST'").own_ar1_median.iloc[0])
        - float(pd.read_csv(RES/"election_null_ar1.csv").ar_hat.iloc[0]), 3),
    "election.houst_mase_winner_stability": lambda: round(float(
        pd.read_csv(RES/"election_null_ar1_flagged_cells.csv").query("series=='HOUST'").winner_stability.iloc[0]), 3),
    "flagship.ar1_coefficient_empirical": lambda: round(float(
        pd.read_csv(FLAG_RES/"flagship_null_ar1_coefficients.csv").query("basis=='differential'").ar1_estimate.median()), 4),
    "flagship.stability_null_p05_ar1": lambda: round(float(pd.read_csv(FLAG_RES/"flagship_null_ar1.csv").null_p05.iloc[0]), 3),
    "flagship.ar1_delta_p05": lambda: round(float(pd.read_csv(FLAG_RES/"flagship_null_ar1.csv").delta_p05.iloc[0]), 4),
    "flagship.above_ar1_adjusted_p05": lambda: bool(pd.read_csv(FLAG_RES/"flagship_null_ar1.csv").above_ar1_p05.iloc[0]),
    "scpower.reversal_0p5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.005").hit_reversal.sum()),
    "scpower.reversal_1": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.01").hit_reversal.sum()),
    "scpower.concentration_gt1_0p5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.005").hit_concentration_gt1.sum()),
    "scpower.concentration_gt1_1": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.01").hit_concentration_gt1.sum()),
    "scpower.kstar_is_1_0p5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.005").hit_kstar_is_1.sum()),
    "scpower.kstar_is_1_1": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.01").hit_kstar_is_1.sum()),
    "scpower.no_condorcet_0p5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.005").hit_no_condorcet.sum()),
    "scpower.no_condorcet_1": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.01").hit_no_condorcet.sum()),
    "scpower.cycle_0p5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.005").hit_cycle.sum()),
    "scpower.cycle_1": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.01").hit_cycle.sum()),

    # --- Section 5's permutation-null power check, continued at the larger deltas ----------
    "scpower.reversal_2": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.02").hit_reversal.sum()),
    "scpower.reversal_5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.05").hit_reversal.sum()),
    "scpower.concentration_gt1_2": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.02").hit_concentration_gt1.sum()),
    "scpower.concentration_gt1_5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.05").hit_concentration_gt1.sum()),
    "scpower.kstar_is_1_2": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.02").hit_kstar_is_1.sum()),
    "scpower.kstar_is_1_5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.05").hit_kstar_is_1.sum()),
    "scpower.no_condorcet_2": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.02").hit_no_condorcet.sum()),
    "scpower.no_condorcet_5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.05").hit_no_condorcet.sum()),
    "scpower.cycle_2": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.02").hit_cycle.sum()),
    "scpower.cycle_5": lambda: int(pd.read_csv(RES/"social_choice_power.csv").query("delta==0.05").hit_cycle.sum()),
}


# ================================== machinery ==================================================

_PROSE_ONLY_FIELDS = {"producer", "spec", "cites", "cross_check", "provenance"}


def _strip_prose(fields: dict) -> dict:
    """One key's fields with the prose-only ones (producer/spec/cites/cross_check/provenance)
    removed --
    factored out of _payload() so merge()'s clash-detection can compare the same "does this key
    carry the same real data" notion _payload()/_digest() use, one key at a time, before the full
    doc is assembled."""
    return {f: fv for f, fv in fields.items() if f not in _PROSE_ONLY_FIELDS}


def _payload(doc):
    """Value plus every companion data field (n, numerator, of, band, sd, ...), sorted -- the
    prose-only fields (producer/spec/cites/cross_check/provenance) are deliberately excluded so
    editing a spec's wording or fixing a cite pointer does not invalidate a published number."""
    return {
        k: _strip_prose(v)
        for k, v in sorted(doc.items()) if not k.startswith("_")
    }


def _digest(doc):
    return hashlib.sha256(
        json.dumps(_payload(doc), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_write_json(path: Path, doc: dict) -> None:
    """Write `doc` to `path` atomically: serialise to a sibling .tmp file, copy the destination's
    mode if it already exists, then replace() it into place, so a reader never observes a
    half-written canonical.json and a crash mid-write leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    if path.exists():
        shutil.copymode(path, tmp)
    tmp.replace(path)


def merge():
    if not FRAGDIR.exists():
        print(f"  no fragment directory at {FRAGDIR.relative_to(ROOT)} — run the producers first")
        return 2
    frags = sorted(FRAGDIR.glob("*.json"))
    if not frags:
        print(f"  no fragments in {FRAGDIR.relative_to(ROOT)} — run the producers first")
        return 2
    doc, stages, clashes = {}, [], []
    for f in frags:
        d = json.loads(f.read_text())
        stage = d.get("_stage", f.stem)
        stages.append(stage)
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if k in doc and _strip_prose(doc[k]) != _strip_prose(v):
                clashes.append(k)
            doc[k] = v
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    version = int(prev.get("_meta", {}).get("version", 0)) + 1
    doc["_meta"] = {"version": version, "n_values": len(_payload(doc)),
                    "n_stages": len(set(stages)), "hash": _digest(doc)}
    _atomic_write_json(OUT, doc)
    for k in clashes:
        print(f"  CLASH  {k} — two producers emitted different values; last write wins")
    print(f"  v{version}: {doc['_meta']['n_values']} values from {doc['_meta']['n_stages']} stages, "
          f"each traceable to a producing script and (where applicable) a cross_check")
    print(f"  -> {OUT.relative_to(ROOT)}")
    return 1 if clashes else 0


def _close(a, b):
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if np.isnan(fa) and np.isnan(fb):
        return True
    return abs(fa - fb) <= max(1e-9, abs(fb) * 1e-6)


def _contains_sentinel(v) -> bool:
    """True if `v` itself is the SENTINEL, or a list/dict containing it at any depth."""
    if isinstance(v, dict):
        return any(_contains_sentinel(x) for x in v.values())
    if isinstance(v, list):
        return any(_contains_sentinel(x) for x in v)
    return v == SENTINEL or str(v) == str(SENTINEL)


def check():
    if not OUT.exists():
        print(f"  FAIL  {OUT.relative_to(ROOT)} does not exist — run the merge first")
        return 2
    doc = json.loads(OUT.read_text())
    meta = doc.get("_meta", {})

    stated, actual = meta.get("hash"), _digest(doc)
    hash_ok = stated == actual
    if not hash_ok:
        print(f"  _meta INCONSISTENT — hash says '{stated}', the payload gives '{actual}'")

    payload = _payload(doc)
    sentinel_found = False
    for k, fields in payload.items():
        for f, v in fields.items():
            if _contains_sentinel(v):
                print(f"  *** SENTINEL LEFT LIVE: {k}{'' if f == 'value' else '.' + f} = {SENTINEL} ***")
                print("      selftest_canonical.py was interrupted between mutating and restoring.")
                print("      Restore from git before trusting anything downstream.")
                sentinel_found = True

    ok = bad = unchk = 0
    failures = []
    real_results = {}   # every registered check's real-OUT outcome, for the independence probe below
    for key, fn in sorted(CHECKS.items()):
        if key not in payload:
            continue
        try:
            got = fn()
            real_results[key] = got
        except FileNotFoundError as e:
            unchk += 1
            failures.append(f"      {key}: re-derivation UNCHECKABLE — input file missing: {e}")
            continue
        except Exception as e:
            unchk += 1
            failures.append(f"      {key}: re-derivation ERRORED — {type(e).__name__}: {str(e)[:70]}")
            continue
        canon_value = payload[key]["value"]
        if _close(got, canon_value):
            ok += 1
        else:
            bad += 1
            failures.append(f"      {key}: canonical={canon_value!r}  re-derived={got!r}")

    registered = sum(1 for k in CHECKS if k in payload)
    print(f"  values {len(payload)} | machine-checked {registered} "
          f"(ok {ok}, MISMATCH {bad}, uncheckable {unchk}) | unregistered {len(payload)-registered}")
    for f in failures:
        print(f)

    for name, want, got in (("N_VALUES", N_VALUES, len(payload)),
                            ("N_CHECKED", N_CHECKED, registered)):
        if want is not None and want != got:
            print(f"  _meta INCONSISTENT — {name} says {want}, the payload gives {got}")
            hash_ok = False

    for name, want, got_keys in (
        ("VALUES_KEYS_HASH", VALUES_KEYS_HASH, payload.keys()),
        ("CHECKED_KEYS_HASH", CHECKED_KEYS_HASH, (k for k in CHECKS if k in payload)),
    ):
        got = _keyset_hash(got_keys)
        if want != got:
            print(f"  _meta INCONSISTENT — {name} says '{want}', the actual key set hashes to '{got}'")
            hash_ok = False

    if "mcspower.n_trials" in payload:
        try:
            want_n_panels = payload["mcspower.n_trials"].get("n_panels")
            got_n_panels = _mcspower_n_panels()
            if want_n_panels is not None and want_n_panels != got_n_panels:
                print(f"  mcspower.n_trials.n_panels INCONSISTENT — canonical={want_n_panels!r} "
                      f"re-derived={got_n_panels!r}")
                hash_ok = False
        except Exception as e:
            print(f"  mcspower.n_trials.n_panels companion check ERRORED — "
                  f"{type(e).__name__}: {str(e)[:70]}")
            hash_ok = False

    dep_hash = _keyset_hash(KNOWN_OUT_DEPENDENT)
    if dep_hash != OUT_DEPENDENT_KEYS_HASH:
        print(f"  _meta INCONSISTENT — OUT_DEPENDENT_KEYS_HASH says '{OUT_DEPENDENT_KEYS_HASH}', "
              f"KNOWN_OUT_DEPENDENT hashes to '{dep_hash}'")
        hash_ok = False
    dependent = _independence_probe(doc, real_results)
    for key, real, got in dependent:
        print(f"  NOT INDEPENDENT — {key}: re-derivation changed when canonical.json was perturbed "
              f"(real-OUT={real!r}, perturbed-OUT={got!r}) -- a readback or hidden dependence on the "
              f"merged file, not an independent re-derivation from the producer's CSV")
    print(f"  independence probe: {len(real_results)} registered checks re-run against a perturbed "
          f"canonical.json, {len(dependent)} undisclosed dependence(s) "
          f"({len(KNOWN_OUT_DEPENDENT)} disclosed in KNOWN_OUT_DEPENDENT)")
    if dependent:
        hash_ok = False
    for p in _source_ratchet():
        print(f"  SOURCE RATCHET — {p}")
        hash_ok = False

    good = hash_ok and bad == 0 and unchk == 0 and not sentinel_found
    print(f"  {'PASS' if good else 'FAIL'}  canonical matches the fragments")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="re-derive and compare; never writes")
    ap.add_argument("--rehash", action="store_true", help="deliberately re-stamp a stale digest")
    ap.add_argument("--rehash-keysets", action="store_true",
                     help="print the current VALUES_KEYS_HASH/CHECKED_KEYS_HASH literals to paste "
                          "into this file after a deliberate key-set change; never writes")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.rehash_keysets:
        doc = json.loads(OUT.read_text())
        payload = _payload(doc)
        print(f'VALUES_KEYS_HASH = "{_keyset_hash(payload.keys())}"')
        print(f'CHECKED_KEYS_HASH = "{_keyset_hash(k for k in CHECKS if k in payload)}"')
        print(f'OUT_DEPENDENT_KEYS_HASH = "{_keyset_hash(KNOWN_OUT_DEPENDENT)}"')
        return 0
    if a.rehash:
        doc = json.loads(OUT.read_text())
        doc["_meta"]["hash"] = _digest(doc)
        doc["_meta"]["n_values"] = len(_payload(doc))
        _atomic_write_json(OUT, doc)
        print(f"  digest re-stamped: {doc['_meta']['hash'][:16]}...")
        return 0
    return merge()


if __name__ == "__main__":
    sys.exit(main())
