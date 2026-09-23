"""state_eb_vintage.py — does using REVISED state unemployment rates, where the statute specifies the INITIAL RELEASE, change Section 7's result?
"""
import sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "instrument"))
import state_eb_decision as SEB
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
RES = ROOT_DIR / "multiseries" / "results"
VINT = ROOT_DIR / "multiseries" / "data" / "raw" / "state_ur_alfred_vintages.parquet"
LOOKBACK_MONTHS = 26
      # a 3-month average shifted 24 => the trigger reads 26 months of history. Only the superseded
      # each-leg arm needs that history from the initial-release series; the as-of arm reads its
      # look-back legs from the vintage itself and the revised/one-leg arms from revised data, so the
      # reported window starts at panel_start and the each-leg arm is disclosed on its own
      # determined months.
WAVE = (pd.Timestamp("2007-06-01"), pd.Timestamp("2009-07-01"))
      # the 2008-09 trigger-on wave: the model and revision channels are also counted inside it so
      # the multipliers can be read by sub-period rather than as a full-record average only


def initial_release_panel():
    """Each reference month's FIRST published value, per state."""
    true_rt, archive_start = _true_first_releases()
    ir = true_rt.pivot(index="date", columns="state", values="value").sort_index()
    return ir, archive_start


def _true_first_releases():
    """(true_rt, archive_start): one row per (state, date) whose earliest realtime_start is a
    genuine release (strictly after that state's archive start), with its first-published value."""
    v = pd.read_parquet(VINT).dropna(subset=["value"]).sort_values("realtime_start")
    archive_start = v.groupby("state").realtime_start.min()
    first = v.groupby(["state", "date"], as_index=False).first()
    first["archive_start"] = first.state.map(archive_start)
    true_rt = first[first.realtime_start > first.archive_start]
    return true_rt, archive_start


def first_release_dates():
    """dates x states DataFrame of each reference month's first TRUE release date (realtime_start),
    NaT where the month has no recoverable release (archive-start snapshot months)."""
    true_rt, _ = _true_first_releases()
    return true_rt.pivot(index="date", columns="state", values="realtime_start").sort_index()
_REMOVED_SENTINEL = -999.0   # a vintage row whose value is NaN = "removed from this vintage on"


def asof_vintage_pivot(v_state):
    """dates x vintage-dates matrix of the value CURRENT at each vintage, for one state."""
    x = v_state[["date", "realtime_start", "value"]].copy()
    x["value"] = x["value"].fillna(_REMOVED_SENTINEL)
    P = x.pivot(index="date", columns="realtime_start", values="value").sort_index().sort_index(axis=1)
    P = P.ffill(axis=1)
    return P.replace(_REMOVED_SENTINEL, float("nan"))


def eb_on_asof(fc, P, first_release, full_index, months):
    """The trigger for each determination month t, computed from the vintage CURRENT AT t's OWN
    FIRST RELEASE -- the statutory quantity (20 CFR 615.12(e)(1)(i)/615.11(d); DOL UIPL 07-94,
    recorded verbatim as external_facts uipl_07_94: "these revised historical data will be
    incorporated in the computation of TUR percentages of prior years as soon as they are
    published. However, no State's EB status will change retroactively based on revised data.")."""
    out = pd.Series(pd.NA, index=full_index, dtype="boolean")
    for t in months:
        r = first_release.get(t)
        if r is None or r not in P.columns:
            continue
        u_t = P[r].reindex(full_index)
        f = u_t if fc is None else fc
        e = SEB.eb_on(f.reindex(full_index), u_t)
        if t in e.index:
            out.loc[t] = e.loc[t]
    return out


def _scored_window(ir, lookback_months=0):
    """(panel_start, win_start): the earliest month at which EVERY state in `ir` has a recoverable
    initial release (`panel_start`), plus `lookback_months` of history (`win_start`; 0 for the
    reported arms, which are determinable from panel_start -- see LOOKBACK_MONTHS)."""
    balanced = ir.index[ir.notna().all(axis=1)]
    panel_start = balanced.min()
    win_start = (panel_start + pd.DateOffset(months=lookback_months)).normalize().replace(day=1)
    return panel_start, win_start


def _window_restrict(x, common, win_start):
    """Restrict a per-state Series or DataFrame to the SAME window episodes_initial/
    episodes_revised score on: reindex to `common` (that state's own `year >= year.min()+15`
    warm-up cutoff), keep only dates >= `win_start`, drop any remaining NaN row/entry."""
    x = x.reindex(common)
    x = x[x.index >= win_start]
    return x.dropna()


def main():
    rev = SEB.load_state_ur()
    ir, _ = initial_release_panel()
    ir = ir.reindex(columns=[c for c in rev.columns if c in ir.columns])
    print(f"revised panel: {rev.shape[1]} states x {rev.shape[0]} months")
    print(f"initial-release panel: {ir.shape[1]} states x {ir.shape[0]} months "
          f"({ir.index.min().date()}..{ir.index.max().date()})")
    # The earliest month at which EVERY state has an initial release, plus the trigger's own lookback.
    panel_start, win_start = _scored_window(ir)
    print(f"  all-states initial-release coverage from {panel_start.date()} -> scored window starts "
          f"{win_start.date()} (the each-leg arm alone needs +{LOOKBACK_MONTHS}mo of initial-release history)")
    diff_parts = []
    rows = []
    tot = {"revised": dict(flip=0, ep=0, div=0, months=0),
           "initial": dict(flip=0, ep=0, div=0, months=0),
           "first_release_each_leg": dict(flip=0, ep=0, div=0, months=0)}
    realized_changed = realized_total = 0
    realized_changed_each_leg = realized_total_each_leg = 0
    realized_changed_one_leg = 0
    wave = {"ep_revised": 0, "ep_initial": 0, "flip_revised": 0, "flip_initial": 0,
            "changed": 0, "changed_one_leg": 0, "months": 0}
    v_all = pd.read_parquet(VINT)
    fr_dates = first_release_dates()
    for s in ir.columns:
        u = rev[s].dropna()
        if len(u) < SEB.MIN_STATE_PANEL_MONTHS:
            continue
        nc = SEB.nowcast(u)
        tied, common = SEB._tied_models(u, nc)
        if len(tied) < 2:
            continue
        # The initial-release realized series, extended with nothing: months outside the archive stay
        # NaN so eb_on's own comparisons drop them rather than silently falling back to revised.
        u_ir = ir[s].reindex(u.index)
        diff_parts.append(_window_restrict(u_ir - u, common, win_start))
        # the as-of machinery for this state: its vintage matrix and each month's first release
        P = asof_vintage_pivot(v_all[v_all.state == s])
        frs = fr_dates[s].dropna() if s in fr_dates.columns else pd.Series(dtype="datetime64[ns]")
        asof_months = [t for t in frs.index if t >= win_start and t in common]
        per_arm = {}
        for arm, realized in (("revised", u), ("initial", None), ("first_release_each_leg", u_ir)):
            if arm == "initial":
                E = pd.DataFrame({m: eb_on_asof(nc[m].reindex(u.index), P, frs, u.index, asof_months)
                                  for m in tied})
            else:
                E = pd.DataFrame({m: SEB.eb_on(nc[m].reindex(u.index), realized) for m in tied})
            E = _window_restrict(E, common, win_start)
            flip = E.any(axis=1) & ~E.all(axis=1)
            ep = len(SEB._episode_lengths(flip))
            if arm in ("revised", "initial"):
                _fw = flip[(flip.index >= WAVE[0]) & (flip.index <= WAVE[1])]
                wave[f"flip_{arm}"] += int(_fw.sum()); wave[f"ep_{arm}"] += len(SEB._episode_lengths(_fw))
            per_arm[arm] = dict(flip=int(flip.sum()), ep=ep, months=len(E),
                                div=int(int(flip.sum()) > 0))
            t = tot[arm]
            t["flip"] += int(flip.sum()); t["ep"] += ep; t["months"] += len(E)
            t["div"] += int(int(flip.sum()) > 0)
        eb_rev = SEB.eb_on(u, u)
        eb_ir = eb_on_asof(None, P, frs, u.index, asof_months)   # statutory as-of status
        eb_ir_each_leg = SEB.eb_on(u_ir, u_ir)                    # the old per-leg construct, disclosed
        eb_one_leg = SEB.eb_on(u_ir, u)
        both_idx = (eb_rev.dropna().index.intersection(eb_ir.dropna().index)
                    .intersection(eb_one_leg.dropna().index))
        both_idx = both_idx[both_idx >= win_start]
        realized_changed += int((eb_rev.reindex(both_idx) != eb_ir.reindex(both_idx)).sum())
        realized_changed_one_leg += int((eb_rev.reindex(both_idx) != eb_one_leg.reindex(both_idx)).sum())
        realized_total += len(both_idx)
        _w = both_idx[(both_idx >= WAVE[0]) & (both_idx <= WAVE[1])]
        wave["changed"] += int((eb_rev.reindex(_w) != eb_ir.reindex(_w)).sum())
        wave["changed_one_leg"] += int((eb_rev.reindex(_w) != eb_one_leg.reindex(_w)).sum())
        wave["months"] += len(_w)
        # the superseded each-leg construct needs LOOKBACK_MONTHS of initial-release history, so it is
        # determined on a shorter window; disclosed on its own months with its own denominator
        el_idx = both_idx.intersection(eb_ir_each_leg.dropna().index)
        realized_changed_each_leg += int((eb_rev.reindex(el_idx) != eb_ir_each_leg.reindex(el_idx)).sum())
        realized_total_each_leg += len(el_idx)
        rows.append({"state": s, "tied": len(tied),
                     "flip_months_revised": per_arm["revised"]["flip"],
                     "flip_months_initial": per_arm["initial"]["flip"],
                     "flip_months_first_release_each_leg": per_arm["first_release_each_leg"]["flip"],
                     "episodes_revised": per_arm["revised"]["ep"],
                     "episodes_initial": per_arm["initial"]["ep"],
                     "episodes_first_release_each_leg": per_arm["first_release_each_leg"]["ep"],
                     "scored_months": per_arm["revised"]["months"],
                     "scored_months_initial": per_arm["initial"]["months"]})
    diff = pd.concat(diff_parts) if diff_parts else pd.Series(dtype=float)
    R = pd.DataFrame(rows)
    RM._atomic_to_csv(R, RES / "state_eb_vintage.csv")
    _summary = pd.DataFrame([
        {"arm": "revised", "states_divergent": tot["revised"]["div"], "states_total": len(R),
         "episodes": tot["revised"]["ep"], "flip_months": tot["revised"]["flip"],
         "scored_months": tot["revised"]["months"]},
        {"arm": "initial_release", "states_divergent": tot["initial"]["div"], "states_total": len(R),
         "episodes": tot["initial"]["ep"], "flip_months": tot["initial"]["flip"],
         "scored_months": tot["initial"]["months"]},
        {"arm": "first_release_each_leg", "states_divergent": tot["first_release_each_leg"]["div"],
         "states_total": len(R), "episodes": tot["first_release_each_leg"]["ep"],
         "flip_months": tot["first_release_each_leg"]["flip"],
         "scored_months": tot["first_release_each_leg"]["months"]},
    ])
    RM._atomic_to_csv(_summary, RES / "state_eb_vintage_summary.csv")
    _scope = (f"sub-window {win_start.date()}..{ir.index.max().date()}, the months for which every "
              f"state has a recoverable INITIAL RELEASE (the as-of arm reads its look-back legs from the "
              f"vintage current at each month's first release, so it is determinable from the first "
              f"all-state release; an earlier construction began {LOOKBACK_MONTHS} months later, at "
              f"2009-08-01, a look-back only the superseded each-leg construct needs -- that construct is "
              f"disclosed on its own determined months). Forecasts and tie sets are held fixed at the "
              f"revised-data versions, so only the realized series feeding the statutory legs differs "
              f"between arms. NOT the paper's headline window and not a replacement for it")
    _asof = ("INITIAL-RELEASE here means the AS-OF vintage current at each month's own first "
             "release (20 CFR 615.12(e)(1)(i), 615.11(d); DOL UIPL 07-94, quoted verbatim as "
             "external_facts uipl_07_94): month t at its first "
             "release, t-1/t-2 at their then-current revisions, the prior-year comparison legs "
             "benchmark-revised as of that date. The companion field each_leg_first_release is the "
             "same figure under an alternative construct that used every leg-month's OWN first "
             "release (a mixed-vintage series no determination ever saw), kept for disclosure")
    emit("vintage.window_start", str(win_start.date()),
         panel_start=str(panel_start.date()),
         spec=f"first scored month of the vintage robustness check: the earliest month with all-state "
              f"initial-release coverage (the each-leg construct additionally needs a "
              f"{LOOKBACK_MONTHS}-month look-back, i.e. 2009-08-01)",
         cites=["S7 body: vintage robustness window"],
         cross_check="state_eb_vintage_summary.csv: implied by scored_months")
    emit("vintage.flip_months_revised", int(tot["revised"]["flip"]),
         spec=f"flip months on REVISED data over the {_scope}",
         cites=["S7 body: vintage robustness"],
         cross_check="state_eb_vintage.csv: flip_months_revised.sum()")
    emit("vintage.flip_months_initial", int(tot["initial"]["flip"]),
         each_leg_first_release=int(tot["first_release_each_leg"]["flip"]),
         spec=f"flip months on INITIAL-RELEASE data (what the statute specifies) over the {_scope}. "
              f"{_asof}",
         cites=["S7 body: vintage robustness"],
         cross_check="state_eb_vintage.csv: flip_months_initial.sum()")
    emit("vintage.episodes_revised", int(tot["revised"]["ep"]),
         spec=f"distinct flip-episodes on REVISED data over the {_scope}",
         cites=["S7 body: vintage robustness"],
         cross_check="state_eb_vintage.csv: episodes_revised.sum()")
    emit("vintage.episodes_initial", int(tot["initial"]["ep"]),
         each_leg_first_release=int(tot["first_release_each_leg"]["ep"]),
         spec=f"distinct flip-episodes on INITIAL-RELEASE data over the {_scope}. {_asof}",
         cites=["S7 body: vintage robustness"],
         cross_check="state_eb_vintage.csv: episodes_initial.sum()")
    emit("vintage.states_divergent_initial", int(tot["initial"]["div"]), of=int(len(R)),
         each_leg_first_release=int(tot["first_release_each_leg"]["div"]),
         spec=f"states with at least one flip month on INITIAL-RELEASE data over the {_scope}. {_asof}",
         cites=["S7 body: vintage robustness"],
         cross_check="state_eb_vintage_summary.csv: states_divergent where arm=='initial_release'")
    emit("vintage.states_divergent_revised", int(tot["revised"]["div"]), of=int(len(R)),
         spec=f"states with at least one flip month on REVISED data over the {_scope}",
         cites=["S7 body: vintage robustness"],
         cross_check="state_eb_vintage_summary.csv: states_divergent where arm=='revised'")
    emit("vintage.revision_sd", round(float(diff.std()), 3),
         n_state_months=int(len(diff)),
         mean=round(float(diff.mean()), 3),
         spec=f"standard deviation of (initial release - revised) in the state unemployment rate, "
              f"measured on this paper's own panel over every state-month where both exist, "
              f"restricted to the {_scope}. Reported alongside the 0.34pp of Chodorow-Reich & "
              f"Karabarbounis (2016), but NOT as a like-for-like check: this figure is a monthly "
              f"initial-release-minus-revised SD, while the quote recorded for theirs in "
              f"external_facts.json "
              f"(chodorow_reich_karabarbounis_2016.tur_realtime_error_sd_0_34pp) defines it only as "
              f"the difference between the real-time and the revised unemployment rate and states "
              f"no averaging or smoothing basis. The two are therefore not established to be on the "
              f"same basis; the comparison is indicative of magnitude only",
         cites=["S7 body: vintage robustness"],
         cross_check=None)
    _ep_lo, _ep_hi = int(tot["revised"]["ep"]), int(tot["initial"]["ep"])   # model-driven episodes on the revised and as-of arms
    emit("vintage.realized_trigger_changed", int(realized_changed), of=int(realized_total),
         ratio_to_model_episodes_lo=int(round(realized_changed / max(_ep_hi, _ep_lo))),
         ratio_to_model_episodes_hi=int(round(realized_changed / min(_ep_hi, _ep_lo))),
         each_leg_first_release=int(realized_changed_each_leg),
         each_leg_first_release_of=int(realized_total_each_leg),
         spec="state-months where the REALIZED statutory trigger status itself differs between the "
              "initial release and the revised series -- no model choice involved. The nearest "
              "published counterpart is Chodorow-Reich & Karabarbounis's (2016) 618 CASES in which "
              "a state 'would have had a different duration of extensions using the revised data', "
              "but the units are not directly comparable: theirs counts episodes, which "
              "external_facts.json's important_context for that key records as averaging about four "
              f"months each, while this key counts state-MONTHS. {_asof}. "
              "This is the TOTAL revision channel (all nine legs of the determination as of month t's "
              "first release); vintage.realized_trigger_changed_one_leg is the like-for-like one-leg "
              "figure. Both REPORTED arms share the `of` denominator: the state-months on which the "
              "revised and as-of arms' status is both determined. The superseded each-leg companion "
              "does NOT: like every other vintage.* construct of its kind it is disclosed on its own "
              "determined months, carried here as each_leg_first_release_of, which is smaller than "
              "`of` because that mixed-vintage series needs a longer look-back.",
         cites=["S7 body: vintage robustness"],
         cross_check=None)
    emit("vintage.realized_trigger_changed_one_leg", int(realized_changed_one_leg), of=int(realized_total),
         ratio_to_model_episodes_lo=int(round(realized_changed_one_leg / max(_ep_hi, _ep_lo))),
         ratio_to_model_episodes_hi=int(round(realized_changed_one_leg / min(_ep_hi, _ep_lo))),
         spec="state-months where the REALIZED statutory trigger status differs between (a) the revised "
              "series and (b) the revised series with ONLY the current month t replaced by its first "
              "release (t-1, t-2 and the six prior-year comparison months revised) -- the single-leg "
              "revision channel, i.e. exactly the leg the model channel perturbs when a nowcast stands in "
              "for month t. The like-for-like numerator for the 'revision vs model choice' comparison; "
              "vintage.realized_trigger_changed is the total (nine-leg) revision channel. Same "
              "denominator as that key.",
         cites=["S7 body: vintage robustness"],
         cross_check=None)
    _wep_lo, _wep_hi = min(wave["ep_revised"], wave["ep_initial"]), max(wave["ep_revised"], wave["ep_initial"])
    emit("vintage.wave_realized_trigger_changed_one_leg", int(wave["changed_one_leg"]), of=int(wave["months"]),
         window_start=str(WAVE[0].date()), window_end=str(WAVE[1].date()),
         model_episodes_revised=int(wave["ep_revised"]), model_episodes_initial=int(wave["ep_initial"]),
         nine_leg=int(wave["changed"]),
         ratio_to_model_episodes_lo=round(wave["changed_one_leg"] / max(_wep_hi, 1), 1),
         ratio_to_model_episodes_hi=round(wave["changed_one_leg"] / max(_wep_lo, 1), 1),
         spec="the one-leg revision channel (vintage.realized_trigger_changed_one_leg) restricted to the "
              "2008-09 trigger-on wave, 2007-06-01..2009-07-01 -- the 26 months an earlier version of this "
              "check excluded. Companions: the nine-leg revision count on the same months, the tied-model "
              "episodes on revised and as-of data inside the wave, and the like-for-like multiple's range "
              "(one-leg changes over the larger and the smaller model-episode count). Read with the "
              "full-record figures: the multiple is a full-record average, and inside the wave the two "
              "channels are closer in size",
         cites=["S7 body: vintage robustness"],
         cross_check=None)
    print(f"\n=== VINTAGE ROBUSTNESS (initial release vs revised, {win_start.date()} onward) ===")
    print(f"  WAVE {WAVE[0].date()}..{WAVE[1].date()}: model episodes revised {wave['ep_revised']} / as-of {wave['ep_initial']}; "
          f"revision one-leg {wave['changed_one_leg']} / nine-leg {wave['changed']} of {wave['months']} state-months")
    print(f"  scored state-months per arm : {tot['revised']['months']}")
    el = tot["first_release_each_leg"]
    print(f"  flip months   revised {tot['revised']['flip']:>4}   initial-release (as-of) {tot['initial']['flip']:>4}"
          f"   [superseded per-leg construct {el['flip']:>4}]")
    print(f"  episodes      revised {tot['revised']['ep']:>4}   initial-release (as-of) {tot['initial']['ep']:>4}"
          f"   [superseded per-leg construct {el['ep']:>4}]")
    print(f"  states diverg revised {tot['revised']['div']:>4}   initial-release (as-of) {tot['initial']['div']:>4}"
          f"  of {len(R)}   [superseded per-leg construct {el['div']:>4}]")
    print(f"  (initial - revised) UR: mean {diff.mean():+.3f}pp  sd {diff.std():.3f}pp  "
          f"over {len(diff)} state-months")
    print(f"  REALIZED trigger status differs in {realized_changed}/{realized_total} state-months "
          f"({100*realized_changed/max(realized_total,1):.1f}%) on the statutory as-of basis "
          f"[superseded per-leg construct: {realized_changed_each_leg}/{realized_total_each_leg}; one-leg "
          f"(month t only at first release): {realized_changed_one_leg}/{realized_total}] "
          f"-- no model choice involved")
    print(f"  wrote {(RES/'state_eb_vintage.csv').relative_to(ROOT_DIR)}")
    flush()


if __name__ == "__main__":
    main()
