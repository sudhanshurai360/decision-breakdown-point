"""make_figures.py — generates the paper's exhibits from committed result CSVs.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MULTI = ROOT / "multiseries"
RES = MULTI / "results"
FLAG_RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CANON = ROOT / "paper" / "canonical.json"

sys.path.insert(0, str(ROOT / "code" / "instrument"))
import fragility as F_frag

# --- shared print-friendly style ---------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
BLACK = "#000000"
GREY = "#888888"
LGREY = "#cfcfcf"
VERMILLION = "#D55E00"   # risk/fragile accent
ORANGE = "#E69F00"       # Okabe-Ito "orange" -- used for intermediate/knife-edge cases
BLUE = "#0072B2"
SKY_BLUE = "#56B4E9"
BLUISH_GREEN = "#009E73"
REDDISH_PURPLE = "#CC79A7"
PALE_RED = "#F4CFC4"     # null-band tint (pale wash of VERMILLION)


_CANON_CACHE = None


def _canon():
    global _CANON_CACHE
    if _CANON_CACHE is None:
        _CANON_CACHE = json.loads(CANON.read_text())
    return _CANON_CACHE


def cval(key):
    return _canon()[key]["value"]


def savefig(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def table_accuracy():
    """Table R -- the raw per-model pooled accuracy table."""
    R = pd.read_csv(RES / "model_set_audit.csv").sort_values("series")
    models = ["ar1", "ets", "theta", "sarima", "dhr", "seasonal_naive"]
    lines = ["| Series | AR1 | ETS | Theta | SARIMA | DHR | Seasonal-naive |",
             "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        cells = " | ".join(f"{r[f'loss_{m}']:.3f}" for m in models)
        lines.append(f"| {r.series} | {cells} |")
    caption = ("**Table R.** Pooled mean scaled absolute error (equal-year weighted, the same "
               "quantity Table 1's |MCS| and Section 3.1's gap/correlation figures are computed "
               "from), all six candidate models, all 12 curated series. The scale each series' own "
               "column is normalized on differs from series to series (MASE is unitless within a series, "
               "not comparable across series), so compare within a row, not down a column.")
    path = OUT / "tableR_accuracy.md"
    path.write_text("\n".join(lines) + "\n\n" + caption + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def table1():
    mcs = pd.read_csv(RES / "mcs_test.csv")
    sc = pd.read_csv(RES / "social_choice.csv")
    sc = sc[sc.metric == "mase"].set_index("series")
    frag = pd.read_csv(RES / "multiseries_fragility.csv")
    frag = frag[frag.metric == "mase"].set_index("series")
    rows = []
    needs_reversal_only = []
    for _, r in mcs.sort_values("MCS_abs_10").iterrows():
        e = sc.loc[r.series]
        T = int(frag.loc[r.series, "n_years"])
        k = int(e.k_star)
        ws = float(e.winner_stability)
        reversal = bool(e.reversal)
        stable_alone = ws < F_frag.FRAGILE_STABILITY_THRESH
        fragile = bool(stable_alone or (reversal and (k / T) < F_frag.FRAGILE_KSTAR_RATIO_THRESH))
        if fragile and not stable_alone:
            needs_reversal_only.append(r.series)
        rows.append({
            "Series": r.series,
            "|MCS|": int(r.MCS_abs_10),
            "Winner": r.best_abs,
            "T": T,
            "k*": k,
            "Concentration": f"{float(e.concentration):.2f}",
            "winner_stability": f"{ws:.2f}",
            "Reversal": "yes" if reversal else "no",
            "Fragile": "yes" if fragile else "no",
        })
    t = pd.DataFrame(rows)
    n_nonident = int((mcs.MCS_abs_10 > 1).sum())
    lines = ["| Series | \\|MCS\\| | Winner | T | k* | Concentration | winner_stability | Reversal | Fragile |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['Series']} | {r['|MCS|']} | {r['Winner']} | {r['T']} | {r['k*']} | "
                      f"{r['Concentration']} | {r['winner_stability']} | {r['Reversal']} | {r['Fragile']} |")
    if needs_reversal_only:
        reversal_examples = (", ".join(needs_reversal_only[:-1]) + " and " + needs_reversal_only[-1]
                              if len(needs_reversal_only) > 1 else needs_reversal_only[0])
        reversal_clause = (f", including {reversal_examples}, whose winner_stability column alone "
                            f"would not otherwise explain a \"yes\"")
    else:
        reversal_clause = ""
    caption = (f"**Table 1.** The identification panel: all {len(t)} curated series, sorted by "
               f"|MCS| (Model Confidence Set size, α=0.10, absolute loss). {n_nonident} of "
               f"{len(t)} series have |MCS| > 1 (non-identified). k* is the decision breakdown "
               f"point for the pooled winner against its most dangerous rival, over T evaluation "
               f"years; read it against T, not as a bare count. Reversal is Section 5's "
               f"score-vs-plurality disagreement; Fragile = winner_stability < "
               f"{F_frag.FRAGILE_STABILITY_THRESH:.2f} OR (Reversal AND k*/T < "
               f"{F_frag.FRAGILE_KSTAR_RATIO_THRESH:.2f}) (Section 2) -- both columns are shown so "
               f"every Fragile verdict is verifiable directly from this table{reversal_clause}. "
               f"All columns use equal-year weighting (w_t = 1), the convention used "
               f"throughout the paper; Section 5 reports the observation-count-weighted robustness "
               f"arm.")
    path = OUT / "table1_identification.md"
    path.write_text("\n".join(lines) + "\n\n" + caption + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")
    return t


def figure_a():
    sc = pd.read_csv(RES / "social_choice.csv")
    nyr = pd.read_csv(RES / "multiseries_fragility.csv")
    nyr = nyr[nyr.metric == "mase"].set_index("series")["n_years"]
    d = sc[sc.metric == "mase"].copy()
    d["n_years"] = d.series.map(nyr)
    FRAGILE_STAB_THRESH = F_frag.FRAGILE_STABILITY_THRESH
    d["fragile"] = ((d.winner_stability < FRAGILE_STAB_THRESH)
                    | (d.reversal.astype(bool) & ((d.k_star / d.n_years) < F_frag.FRAGILE_KSTAR_RATIO_THRESH)))
    d = d.sort_values("winner_stability").reset_index(drop=True)
    null_med = cval("election.stability_null_median")
    null_p05 = cval("election.stability_null_p05")
    if null_p05 > FRAGILE_STAB_THRESH:
        raise RuntimeError(
            f"figure_a's color logic assumes the calibrated band (null_p05={null_p05}) sits inside "
            f"the fragile threshold ({FRAGILE_STAB_THRESH}) -- a band member above "
            f"{FRAGILE_STAB_THRESH} would render uncolored")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    y = np.arange(len(d))

    ax.axvspan(0, null_p05, color=PALE_RED, alpha=0.7, zorder=0)
    ax.axvline(null_med, color=GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(null_med + 0.015, len(d) + 0.6, "null\nmedian", fontsize=6.5, ha="left", color=GREY)
    for yi, ws, fragile in zip(y, d.winner_stability, d.fragile):
        in_band = ws < null_p05
        marker = "o" if not fragile else "s"
        color = "white" if not fragile else (VERMILLION if in_band else ORANGE)
        edge = BLACK if not fragile else (VERMILLION if in_band else ORANGE)
        ax.plot(ws, yi, marker=marker, markerfacecolor=color, markeredgecolor=edge,
                markersize=6, linewidth=0)
    ax.axvline(0, color=BLACK, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d.series, fontsize=7.5)
    ax.set_xlabel("winner_stability")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.8, len(d) + 1.4)
    ax.text(null_p05 / 2, len(d) + 0.6, "null\n5th pct.", fontsize=6.5, ha="center", color=VERMILLION)

    handles = [plt.Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=BLACK,
                          linewidth=0, label="not fragile"),
               plt.Line2D([0], [0], marker="s", markerfacecolor=ORANGE, markeredgecolor=ORANGE,
                          linewidth=0, label="fragile (Table 1 rule), above the calibrated band"),
               plt.Line2D([0], [0], marker="s", markerfacecolor=VERMILLION, markeredgecolor=VERMILLION,
                          linewidth=0, label="fragile AND below the null's 5th pct.")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.3, frameon=False)
    _n_frag = int(d.fragile.sum())
    _n_out = int((d.fragile & (d.winner_stability >= null_p05)).sum())
    ax.set_title(f"Figure 3. Calibrated stability vs. the arbitrary-winner null\n"
                 f"(k* per series is in Table 1; 'fragile' is the Table 1 "
                 f"stability<{FRAGILE_STAB_THRESH:.2f}, or reversal with k*/T<{F_frag.FRAGILE_KSTAR_RATIO_THRESH:.2f}, rule,\n"
                 f"not band membership -- {_n_out} of {_n_frag} fragile series sit outside the "
                 f"calibrated tail)",
                 fontsize=8.3)
    fig.tight_layout()
    savefig(fig, "figA_forest_kstar_stability")


def figure_b():
    d = pd.read_csv(RES / "social_choice.csv")
    d = d.assign(label=d.series + " (" + d.metric + ")").sort_values("concentration").reset_index(drop=True)
    n_rev = int(d.reversal.sum())

    # vermillion for reversal points only makes the n_rev/len(d) headline visually immediate;
    # shape (square/circle) is the greyscale-safe backup encoding.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y = np.arange(len(d))
    for yi, row in zip(y, d.itertuples()):
        marker = "s" if row.reversal else "o"
        face = VERMILLION if row.reversal else "white"
        edge = VERMILLION if row.reversal else BLACK
        ax.plot(row.concentration, yi, marker=marker, markerfacecolor=face,
                markeredgecolor=edge, markersize=6, linewidth=0)
    ax.axvline(1.0, color=GREY, linewidth=0.8, linestyle="--")
    ax.text(1.03, len(d) - 0.4, "concentration = 1\n(pivotal period's margin\n= champion's total margin)",
            fontsize=6.3, color=GREY, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(d.label, fontsize=7)
    ax.set_xlabel("concentration (pivotal-period margin share of champion's total margin)", fontsize=8)
    ax.set_xscale("log")
    ax.set_title(f"Figure 4. The election reframe: {n_rev} of {len(d)} cells reverse (score ≠\n"
                 f"plurality winner, marker shape); concentration is the same fact in a more vivid\n"
                 f"unit (§5.3), not an independent measurement", fontsize=8.7)
    handles = [plt.Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=BLACK,
                          linewidth=0, label="score = plurality winner"),
               plt.Line2D([0], [0], marker="s", markerfacecolor=VERMILLION, markeredgecolor=VERMILLION,
                          linewidth=0, label="reversal (score ≠ plurality)")]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    savefig(fig, "figB_election_reversal")


def figure_c():
    d = pd.read_csv(RES / "state_eb_decision.csv")
    s = pd.read_csv(RES / "state_eb_summary.csv").iloc[0]
    d["genuine"] = d.flip_episodes - d.knife_edge

    genuine = sorted(d[d.genuine > 0].state)
    knife_only = sorted(d[(d.genuine == 0) & (d.flip_episodes > 0)].state)
    no_flip = sorted(d[d.flip_episodes == 0].state)
    n_partitioned = len(genuine) + len(knife_only) + len(no_flip)
    if n_partitioned != len(d):
        raise RuntimeError(
            f"figure_c's genuine/knife_only/no_flip partition covers {n_partitioned} states, "
            f"not all {len(d)} -- a state was double-counted or dropped; Table C would render "
            f"an incomplete or double-counted state list with no error otherwise.")

    lines = ["| Category | Count | States |",
             "|---|---|---|",
             f"| ≥1 genuine flip | {len(genuine)} | {', '.join(genuine)} |",
             f"| Knife-edge flip only | {len(knife_only)} | {', '.join(knife_only)} |",
             f"| No flip episode | {len(no_flip)} | {', '.join(no_flip)} |"]
    caption = (f"**Table C.** The decision cost by state: {int(s.states_divergent)} of "
               f"{int(s.states_total)} states have ≥1 flip episode ({int(s.distinct_episodes)} "
               f"episodes total, {s.knife_edge_pct:.0f}% knife-edge -- the tied models' rounded three-month "
               f"trigger values lie within 0.3 percentage points at the flip month) -- only {' and '.join(genuine)} are "
               f"genuine, non-knife-edge flips.")
    path = OUT / "tableC_decision_cost_by_state.md"
    path.write_text("\n".join(lines) + "\n\n" + caption + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def table_permnull():
    rows = [("score-vs-plurality reversal", "reversal"),
            ("concentration > 1",           "conc_gt1"),
            ("k* = 1",                      "kstar1"),
            ("no Condorcet winner",         "no_condorcet"),
            ("intransitive cycle",          "cycle")]
    lines = ["| statistic | observed | null mean | null 5-95% | P(null >= obs) |",
             "|---|---|---|---|---|"]
    n_perm = None
    for label, key in rows:
        m = _canon()[f"election.permnull_{key}_mean"]
        p_ge = cval(f"election.permnull_{key}_p_ge")
        n_perm = int(m["n_perm"])
        lines.append(f"| {label} | {int(m['observed'])} of {int(m['of'])} | {float(m['value']):.2f} | "
                     f"[{int(m['p05'])}, {int(m['p95'])}] | {float(p_ge):.2f} |")
    caption = (f"**Table P.** Section 5's election statistics against the within-period model-label "
               f"permutation null ({n_perm} draws). The null permutes model labels within each period "
               f"independently, preserving every period's own multiset of losses (its difficulty and "
               f"dispersion, including the April-2020 cliff at full size) and every marginal loss value -- "
               f"destroying the association between a model's identity and its performance, and with it "
               f"each named model's serial persistence across periods and the correlation between named "
               f"models' loss series (Table Q recalibrates against a null that keeps the serial dependence). A "
               f"large P(null >= obs) means the observed count is at or below what an "
               f"arbitrarily-crowned winner produces.")
    path = OUT / "tableP_permutation_null.md"
    path.write_text("\n".join(lines) + "\n\n" + caption + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def table_ar1null():
    """Table Q -- the AR(1)-calibrated exchangeable null companion to Table P above."""
    rows = [("score-vs-plurality reversal", "reversal"),
            ("concentration > 1",           "conc_gt1"),
            ("k* = 1",                      "kstar1"),
            ("no Condorcet winner",         "no_condorcet"),
            ("intransitive cycle",          "cycle")]
    lines = ["| statistic | observed | AR(1) null mean | null 5-95% | P(null >= obs) | P(null <= obs) |",
             "|---|---|---|---|---|---|"]
    n_perm = None
    for label, key in rows:
        m = _canon()[f"election.ar1null_{key}_mean"]
        p_ge = cval(f"election.ar1null_{key}_p_ge")
        p_le = cval(f"election.ar1null_{key}_p_le")
        n_perm = int(m["n_perm"])
        lines.append(f"| {label} | {int(m['observed'])} of {int(m['of'])} | {float(m['value']):.2f} | "
                     f"[{int(m['p05'])}, {int(m['p95'])}] | {float(p_ge):.3f} | {float(p_le):.3f} |")
    ar_hat = float(_canon()["election.ar1null_coefficient"]["value"])
    caption = (f"**Table Q.** Section 5's election statistics against an AR(1)-calibrated "
               f"exchangeable null ({n_perm} draws, AR(1)={ar_hat:.3f} -- the same empirically-"
               f"estimated pooled coefficient election_null_ar1.py uses): fresh synthetic K-model "
               f"panels at each cell's own (K,T), genuinely exchangeable (no model has a real "
               f"population edge) but carrying the same serial dependence real models do, not a "
               f"permutation of the fixed real data.")
    path = OUT / "tableQ_ar1null.md"
    path.write_text("\n".join(lines) + "\n\n" + caption + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def figure_d():
    d = pd.read_csv(RES / "realised_t.csv")
    lo = min(d.ceiling_binding.min(), d.t_binding[d.t_binding > 0].min()) * 0.7
    hi = max(d.ceiling_binding.max(), d.t_binding.max()) * 1.4

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    is_flag = d.panel == "FLAGSHIP"
    ax.plot(d.ceiling_binding[~is_flag], d.t_binding[~is_flag], marker="o", markerfacecolor="white",
            markeredgecolor=BLACK, linewidth=0, markersize=6.5, label="curated series")
    ax.plot(d.ceiling_binding[is_flag], d.t_binding[is_flag], marker="D", markerfacecolor=BLACK,
            markeredgecolor=BLACK, linewidth=0, markersize=7.5, label="legacy neural exhibit")
    rows = sorted(d.itertuples(), key=lambda r: r.t_binding)
    log_t = [np.log(r.t_binding) for r in rows]
    clusters, current = [], [0]
    for i in range(1, len(rows)):
        if log_t[i] - log_t[i - 1] < 0.12:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)
    nudge = {}
    for cluster in clusters:
        n = len(cluster)
        if n == 1:
            continue
        for rank, idx in enumerate(cluster):
            nudge[rows[idx].panel] = rank - (n - 1) / 2
    for row in rows:
        label = "legacy neural exhibit" if row.panel == "FLAGSHIP" else row.panel
        dx = 1.06 if row.t_binding > 0.15 else 1.1
        dy = 1 + 0.09 * nudge.get(row.panel, 0)
        ax.annotate(label, (row.ceiling_binding * dx, row.t_binding * dy), fontsize=5.8, color=GREY,
                    va="center")
    ax.plot([lo, hi], [lo, hi], color=BLACK, linewidth=0.8, linestyle="--")
    ax.text(hi * 0.5, hi * 0.56, "sharpness line: |t| = ceiling", fontsize=6.5, color=GREY,
            rotation=45, ha="center")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Proposition 2.2 ceiling  √(Tk*/(T−k*))  [log scale]")
    ax.set_ylabel("realised |t| against the binding rival  [log scale]")
    n = len(d)
    n_viol = int((d.t_binding > d.ceiling_binding + 1e-9).sum())
    ax.set_title(f"Figure 1. Proposition 2.2's bound in practice\n"
                 f"({n} panels, {n_viol} violate the ceiling; median |t|/ceiling "
                 f"{cval('prop.median_slack'):.3f})", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    savefig(fig, "figD_prop22_ceiling_scatter")


def figure_e():
    """Legacy neural-model trajectory exhibit -- supplementary, not part of the curated-panel
    main text."""
    d = pd.read_csv(FLAG_RES / "extension_scoreboard_1988-2025.csv", index_col=0)
    responsible = [int(y) for y in str(cval("flagship.responsible_years")).split(";")]
    winner = cval("flagship.pooled_winner")
    binding_rival = _canon()["flagship.k_star"]["rival"]

    styles = {
        "seasonal_naive": dict(color=BLUE, linestyle=":", marker=None, lw=1.1),
        "ar1": dict(color=SKY_BLUE, linestyle="--", marker=None, lw=1.1),
        "theta": dict(color=BLUISH_GREEN, linestyle="-", marker=None, lw=1.1),
        "nbeats": dict(color=BLACK, linestyle="-", marker="o", lw=1.8, ms=3.5),
        "nhits": dict(color=ORANGE, linestyle="-", marker="^", lw=1.2, ms=3),
        "tft": dict(color=VERMILLION, linestyle="--", marker="s", lw=1.2, ms=3),
        "itransformer": dict(color=REDDISH_PURPLE, linestyle=":", marker="x", lw=1.2, ms=3),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in d.columns:
        st = dict(styles.get(m, dict(color=GREY, linestyle="-", lw=0.8)))
        marker = st.pop("marker", None)
        ax.plot(d.index, d[m], marker=marker, markersize=st.pop("ms", 0) if marker else 0,
                label=m, zorder=(5 if m == "nbeats" else 2), **st)
    for yr in responsible:
        ax.axvspan(yr - 0.5, yr + 0.5, color=LGREY, alpha=0.5, zorder=0)
    ax.set_ylabel("MASE (pooled, lower = better)")
    ax.set_xlabel("year")
    ax.set_title(f"Figure S1 (supplementary). The legacy neural exhibit: classical (seasonal_naive/ar1/\n"
                 f"theta, cool colors) vs. neural (nhits/tft/itransformer, warm; nbeats, pooled winner,\n"
                 f"black) forecasters, 1988–2025 — pooled winner: {winner}  ·  "
                 f"k*={int(cval('flagship.k_star'))}, binding rival: {binding_rival}  ·  shaded = the "
                 f"{len(responsible)} years whose deletion flips it", fontsize=8.2)
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    savefig(fig, "figS1_flagship_trajectory_supplementary")


def figure_f():
    sys.path.insert(0, str(HERE))
    import mcs_test as MT
    d = pd.read_csv(MT.PRED)
    rows = {}
    all_models = set()
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        yrs = sorted(g.year.unique())
        La = np.full((len(yrs), len(models)), np.nan)
        for j, m in enumerate(models):
            gm = g[g.model == m].groupby("year")["sae"].mean()
            for i, y in enumerate(yrs):
                if y in gm.index:
                    La[i, j] = gm[y]
        keep = ~np.isnan(La).any(1)
        if (~keep).sum():
            print(f"  [warn] {sid}: dropping {(~keep).sum()} year(s) with a NaN before MCS (figure F)")
        La = La[keep]
        surv, _p = MT.mcs(La, alpha=0.10, seed=0)
        survivors = {models[i] for i in surv}
        rows[sid] = {m: (m in survivors) for m in models}
        all_models |= set(models)

    all_models = sorted(all_models)
    series = sorted(rows, key=lambda s: -sum(rows[s].values()))
    mat = np.array([[1 if rows[s].get(m, False) else (np.nan if m not in rows[s] else 0)
                      for m in all_models] for s in series], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = matplotlib.colors.ListedColormap(["white", BLACK])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series, fontsize=7)
    for i in range(len(series)):
        for j in range(len(all_models)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "×", ha="center", va="center", fontsize=6, color=GREY)
    ax.set_xticks(np.arange(-0.5, len(all_models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(series), 1), minor=True)
    ax.grid(which="minor", color=LGREY, linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Figure 2. Which models survive the Model Confidence Set?\n"
                 "(α=0.10, absolute loss; black = in the MCS, white = eliminated)", fontsize=8.7)
    fig.tight_layout()
    savefig(fig, "figF_mcs_membership_heatmap")


def figure_note_llm_crowding():
    ks = [2, 4, 8, 16, 32, 64]
    real = [cval(f"llm.crowd_real_k{k}") for k in ks]
    null = [cval(f"llm.crowd_null_k{k}") for k in ks]

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(ks, real, marker="o", markerfacecolor=BLACK, markeredgecolor=BLACK, color=BLACK,
            linewidth=1.2, label="real leaderboard")
    ax.plot(ks, null, marker="s", markerfacecolor="white", markeredgecolor=BLACK, color=GREY,
            linewidth=1.0, linestyle="--", label="independence null")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("pool size (# models)")
    ax.set_ylabel("share of k-model draws where mean-score ≠\nrank-agg. winner", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_title("Figure 1 (companion note). On the Open LLM Leaderboard, mean and rank\n"
                 "aggregation agree more often than under task independence, at every pool size", fontsize=8.7)
    fig.tight_layout()
    savefig(fig, "figNote1_llm_crowding_vs_null")


def figure_h():
    d = pd.read_csv(RES / "breadth_distinct" / "breadth_distinct_predictions.csv")
    d["target"] = pd.to_datetime(d["target"])
    diff = d.groupby(["series", "target"])["sae"].min().reset_index()  # difficulty = best any model achieved
    diff["ym"] = diff.target.dt.to_period("M")

    win20 = pd.period_range("2019-06", "2021-06", freq="M")
    win08 = pd.period_range("2007-06", "2009-06", freq="M")
    series = sorted(diff.series.unique())

    _month_counts = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, win, title, shock in [(ax1, win20, "(a) 2020: an abrupt shock", "2020-04"),
                                   (ax2, win08, "(b) 2008–09: a gradual shock", None)]:
        sub = diff[diff.ym.isin(win)]
        med = sub.groupby("ym")["sae"].median()
        lo = sub.groupby("ym")["sae"].quantile(0.25)
        hi = sub.groupby("ym")["sae"].quantile(0.75)
        _month_counts.append(sub.groupby("ym")["series"].nunique())
        x = [p.to_timestamp() for p in med.index]
        ax.fill_between(x, lo.values, hi.values, color=LGREY, alpha=0.6, linewidth=0)
        ax.plot(x, med.values, color=BLACK, linewidth=1.3)
        if shock:
            # vermillion accent on the pivotal month only
            shock_x = pd.Period(shock, freq="M").to_timestamp()
            ax.axvline(shock_x, color=VERMILLION, linestyle="--", linewidth=0.9)
            ax.text(shock_x, 0.97, "COVID onset", transform=ax.get_xaxis_transform(),
                    fontsize=6, color=VERMILLION, rotation=90, ha="right", va="top")
        ax.set_title(title, fontsize=8.7)
        ax.set_xlabel("month")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45); lbl.set_ha("right")
    _n_lo = int(min(c.min() for c in _month_counts))
    _n_hi = int(max(c.max() for c in _month_counts))
    _n_desc = f"{_n_lo}" if _n_lo == _n_hi else f"{_n_lo}-{_n_hi} of {len(series)}"
    ax1.set_ylabel(f"difficulty: min scaled error across models\n"
                    f"(median across {_n_desc} series/month, IQR shaded)")
    fig.suptitle("Figure 5. An abrupt shock (2020) and a gradual one (2008-09)\nat monthly "
                 "resolution", fontsize=9, y=1.04)
    fig.tight_layout()
    savefig(fig, "figH_shape_not_magnitude")


def _figure_i_equal_cells(xs, ys, counts):
    """The value-equal, populated (MCS_stat_near, dMCS_decision) grid cells figure_i() outlines,
    sorted most-populated first (the callout points at cells[0])."""
    equal_cells = [(i, j) for i, x in enumerate(xs) for j, y in enumerate(ys)
                   if x == y and int(counts.get((x, y), 0)) > 0]
    equal_cells.sort(key=lambda ij: -int(counts.get((xs[ij[0]], ys[ij[1]]), 0)))
    return equal_cells


def figure_i():
    d = pd.read_csv(RES / "dmcs.csv")
    tied_both = ((d.dMCS_decision >= d.MCS_stat_near) & (d.dMCS_decision > 1))
    n = len(d)

    xs = sorted(d.MCS_stat_near.unique())
    ys = sorted(d.dMCS_decision.unique())
    counts = d.groupby(["MCS_stat_near", "dMCS_decision"]).size()

    # 5.9 wide, not the usual 5.0: the value-equal-cell callout label added below needs the extra
    # room or it crowds the title.
    fig, ax = plt.subplots(figsize=(5.9, 4.6))
    cmax = counts.max()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            c = int(counts.get((x, y), 0))
            if c == 0:
                continue
            face = matplotlib.colormaps["viridis"](0.15 + 0.85 * (c / cmax))
            ax.add_patch(plt.Rectangle((i, j), 0.94, 0.94, facecolor=face,
                                        edgecolor=BLACK, linewidth=0.7))
            luminance = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            tcol = "black" if luminance > 0.55 else "white"
            ax.text(i + 0.47, j + 0.47, str(c), ha="center", va="center", fontsize=9, color=tcol)
    equal_cells = _figure_i_equal_cells(xs, ys, counts)
    for i, j in equal_cells:
        ax.add_patch(plt.Rectangle((i, j), 0.94, 0.94, facecolor="none",
                                    edgecolor=GREY, linewidth=2.2, linestyle="--", zorder=2))
    if equal_cells:
        i0, j0 = equal_cells[0]
        ax.annotate("decision |MCS| =\naccuracy |MCS|\n(outlined cell)",
                     xy=(i0 + 0.94, j0 + 0.47), xytext=(len(xs) + 0.12, j0 + 0.47),
                     fontsize=6.3, color=GREY, ha="left", va="center", clip_on=False,
                     arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6))
    ax.set_xticks([i + 0.47 for i in range(len(xs))])
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_yticks([j + 0.47 for j in range(len(ys))])
    ax.set_yticklabels([str(int(y)) for y in ys])
    ax.set_xlim(0, len(xs))
    ax.set_ylim(0, len(ys))
    ax.set_aspect("equal")
    ax.set_xlabel("|MCS| under statistical (accuracy) loss, near-trigger months")
    ax.set_ylabel("|MCS| under decision (statutory-trigger) loss")
    ax.set_title(f"Figure 6. Decision-loss ambiguity is at least as wide as accuracy-loss ambiguity,\n"
                 f"in every state ({int(tied_both.sum())} of {n}) -- ties are never narrower under the "
                 f"statutory decision rule",
                 fontsize=8.3)
    fig.tight_layout()
    savefig(fig, "figI_accuracy_vs_decision")


def main():
    print("Building Table 1, Tables C/P/Q/R and Figures A, B, D, F, H, I (read directly from "
          "committed CSVs and canonical.json -- nothing hand-typed)...")
    table1()
    table_permnull()
    table_ar1null()
    table_accuracy()
    for fn in (figure_a, figure_b, figure_c, figure_d, figure_f, figure_h, figure_i):
        fn()
    print("\nSupplementary (not main-text):")
    figure_e()
    print("\nFor the companion note (paper/note_ssrn.md), not this paper:")
    figure_note_llm_crowding()
    print(f"\nAll artifacts written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
