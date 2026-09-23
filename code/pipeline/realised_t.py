"""realised_t.py — the t-statistic the paper BOUNDS, finally computed on the paper's own data.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "instrument"))
from canonical_emit import emit, flush
from scipy.stats import spearmanr
import fragility as F
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
ROOT = HERE.parent.parent
RES = ROOT / "multiseries" / "results"
PRED = RES / "breadth_distinct" / "breadth_distinct_predictions.csv"
FLAG = ROOT / "results" / "extension_scoreboard_1988-2025.csv"
CRIT_5PCT = 1.96


def hac_t(d: np.ndarray) -> float:
    """Newey-West studentized mean, Bartlett kernel, standard 4(T/100)^(2/9) lag rule."""
    T = len(d); m = float(d.mean()); e = d - m
    lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    s = float(e @ e) / T
    for L in range(1, lag + 1):
        s += 2.0 * (1 - L / (lag + 1)) * float(e[L:] @ e[:-L]) / T
    return float("nan") if s <= 0 else float(abs(np.sqrt(T) * m / np.sqrt(s)))


def analyse(name: str, M: pd.DataFrame):
    T = len(M)
    win = M.mean().idxmin()
    rows = []
    for riv in M.columns:
        if riv == win:
            continue
        d = (M[riv] - M[win]).to_numpy(float)   # positive where the champion is better
        if d.sum() <= 0:                        # the champion must win the pair pooled
            continue
        sd = float(d.std(ddof=0))               # POPULATION sd (ddof=0): the studentization
                                                # Proposition 2.2 is stated for
        if sd <= 0:
            continue
        # the shipped breakdown-number definition, so k* here is the same object the paper bounds
        k = F.breakdown_number(M[win].to_numpy(float), M[riv].to_numpy(float), np.ones(T))[0]
        rows.append(dict(rival=riv, k=k,
                         t=float(abs(np.sqrt(T) * d.mean() / sd)),
                         ceiling=float(np.sqrt(T * k / (T - k))) if k < T else np.inf,
                         t_hac=hac_t(d)))
    if not rows:
        return None
    binding = min(rows, key=lambda r: (r["k"], r["rival"]))
       # k* is the MIN over rivals -- this is the pair 2.2 binds
    strongest = max(rows, key=lambda r: r["t"])
    if not np.isfinite(binding["t_hac"]):
        print(f"  [warn] {name}: HAC statistic undefined (non-positive HAC variance) for binding "
              f"rival {binding['rival']} -- hac_over_ceiling recorded as False, not verified")
    return dict(panel=name, T=T, winner=win, k_star=binding["k"], binding_rival=binding["rival"],
                t_binding=binding["t"], ceiling_binding=binding["ceiling"],
                slack=binding["t"] / binding["ceiling"] if np.isfinite(binding["ceiling"]) else np.nan,
                t_hac_binding=binding["t_hac"], t_max=strongest["t"], t_max_rival=strongest["rival"],
                n_sig_5pct=int(sum(1 for r in rows if r["t"] > CRIT_5PCT)), n_rivals=len(rows),
                violates=bool(np.isfinite(binding["ceiling"]) and binding["t"] > binding["ceiling"] + 1e-9),
                hac_over_ceiling=bool(np.isfinite(binding["ceiling"]) and np.isfinite(binding["t_hac"])
                                      and binding["t_hac"] > binding["ceiling"] + 1e-9))


def main():
    out = []
    n_excluded_short, n_excluded_novalid = 0, 0
    d = pd.read_csv(PRED)
    for sid, g in d.groupby("series"):
        ms = sorted(g.model.unique()); ys = sorted(g.year.unique())
        M = pd.DataFrame([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in ms] for y in ys],
                         columns=ms, index=ys)
        if M.notna().all().all() and len(M) >= 20:
            r = analyse(sid, M)
            if r:
                out.append(r)
            else:
                n_excluded_novalid += 1
                print(f"  [drop] {sid}: no valid binding rival (every candidate tied or degenerate)")
        else:
            n_excluded_short += 1
            print(f"  [drop] {sid}: incomplete or <20-year panel ({len(M)} years, "
                  f"{'complete' if M.notna().all().all() else 'has gaps'})")
    if n_excluded_short or n_excluded_novalid:
        print(f"  {n_excluded_short + n_excluded_novalid} series excluded before scoring "
              f"({n_excluded_short} incomplete/short, {n_excluded_novalid} no valid rival)")
    _flag_M = pd.read_csv(FLAG, index_col=0)
    if _flag_M.notna().all().all() and len(_flag_M) >= 20:
        r = analyse("FLAGSHIP", _flag_M)
        if r:
            out.append(r)
        else:
            print("  [drop] FLAGSHIP: no valid binding rival (every candidate tied or degenerate)")
    else:
        print(f"  [drop] FLAGSHIP: incomplete or <20-year panel ({len(_flag_M)} years, "
              f"{'complete' if _flag_M.notna().all().all() else 'has gaps'})")
    R = pd.DataFrame(out)
    RM._atomic_to_csv(R, RES / "realised_t.csv")
    n = len(R)
    rho, pval = spearmanr(R.k_star, R.t_binding)
    n_sig_small = int(((R.n_sig_5pct > 0) & (R.k_star <= 3)).sum())
    print(f"=== REALISED |t| BESIDE k* ({n} panels) ===")
    print(R[["panel", "T", "winner", "k_star", "binding_rival", "t_binding", "ceiling_binding",
             "slack", "t_max", "n_sig_5pct", "t_hac_binding"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    emit("prop.realised_violations", int(R.violates.sum()), of=n,
         spec="panels where the realised i.i.d.-studentized |t| against the BINDING rival exceeds "
              "Proposition 2.2's ceiling for that pair. The proposition is stated per pair, so the "
              "binding rival -- the one that sets k* -- is the pair it constrains",
         cites=["S2.2 body: realised statistics"],
         cross_check="realised_t.csv: violates.sum()")
    emit("prop.median_slack", round(float(R.slack.median()), 3),
         spec="median of |t| / ceiling on the binding pair -- the fraction of the bound the realised "
              "statistic CONSUMES, so a HIGHER value means LESS room, not more. Named `median_slack` "
              "for continuity with the acceptance ledger; the manuscript reports it as a "
              "ratio and does not call it slack, since that word inverts the reading. "
              "The bound is sharp in the sense that an "
              "extremal configuration attains it; this says how far the actual panels sit from that "
              "configuration, which is a different and more useful fact for a reader",
         cites=["S2.2 body: realised statistics"],
         cross_check="realised_t.csv: median of slack")
    emit("prop.hac_over_ceiling", int(R.hac_over_ceiling.sum()), of=n,
         spec="panels where the HAC-corrected |t| exceeds the i.i.d. ceiling. Proposition 2.2 is NOT "
              "stated for the HAC statistic and this does not extend it -- zero here is empirical "
              "reassurance on the panels at hand, not a proof, and S2.2 keeps the hedge",
         cites=["S2.2 body: the HAC hedge"],
         cross_check="realised_t.csv: hac_over_ceiling.sum()")
    emit("prop.spearman_kstar_t", round(float(rho), 3), p_value=float(pval), n_panels=n,
         spec="Spearman correlation between k* and the realised |t| on the binding pair. Reported "
              "because it is a COST to this paper's framing, not a confirmation: a high value means "
              "k* is substantially redundant with the studentized margin a reader could compute "
              "anyway, and the case for reporting it rests on interpretability, not independence",
         cites=["S2.2 body: what k* adds over a t-statistic"],
         cross_check=None)
    emit("prop.panels_sig_small_kstar", n_sig_small, of=n,
         spec="panels with k* <= 3 AND a champion significant at the two-sided five-percent level "
              "against at least one rival -- a configuration §2.2 rules out only if the "
              "minimum-over-rivals bound is misread as binding every pair",
         cites=["S2.2 body: what the ceiling constrains"],
         cross_check="realised_t.csv: rows with k_star<=3 and n_sig_5pct>0")
    fl = R[R.panel == "FLAGSHIP"]
    if len(fl) == 1:
        emit("flagship.t_binding", round(float(fl.t_binding.iloc[0]), 3),
             rival=str(fl.binding_rival.iloc[0]),
             spec="the flagship champion's realised i.i.d.-studentized |t| against the rival that "
                  "sets its k*, i.e. the one pair Proposition 2.2 constrains for it",
             cites=["S2.3 body: the flagship's realised statistic"],
             cross_check="realised_t.csv: t_binding where panel=='FLAGSHIP'")
        emit("flagship.t_max", round(float(fl.t_max.iloc[0]), 3),
             rival=str(fl.t_max_rival.iloc[0]),
             spec="the flagship champion's LARGEST realised |t| over all rivals. Reported beside "
                  "t_binding because the gap between them is the whole point of the §2.2 correction",
             cites=["S2.3 body: the flagship's realised statistic"],
             cross_check="realised_t.csv: t_max where panel=='FLAGSHIP'")
        emit("flagship.ceiling_binding", round(float(fl.ceiling_binding.iloc[0]), 3),
             spec="Proposition 2.2's ceiling sqrt(T*k/(T-k)) for the flagship's binding pair (its own "
                  "T and k*=flagship.k_star) -- what flagship.t_binding is compared against",
             cites=["S2.3 body: the flagship's realised statistic"],
             cross_check="realised_t.csv: ceiling_binding where panel=='FLAGSHIP'")
    print(f"\n  Prop 2.2 violated on the binding pair: {int(R.violates.sum())} of {n}")
    print(f"  median |t|/ceiling (fraction of the bound consumed): {R.slack.median():.3f}")
    print(f"  HAC |t| over the i.i.d. ceiling: {int(R.hac_over_ceiling.sum())} of {n}")
    print(f"  Spearman(k*, |t| binding) = {rho:+.3f}  p = {pval:.4g}")
    print(f"  panels with k*<=3 AND a significant champion: {n_sig_small} of {n}"
          f"   <- the configuration S2.2 warns against ruling out (a per-pair bound is not a"
          f" per-champion bound)")
    flush()


if __name__ == "__main__":
    main()
