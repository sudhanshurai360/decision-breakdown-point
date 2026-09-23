"""
state_eb_decision.py — the powered decision-projection: k*_decision on the state Extended-Benefits trigger.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
import m_audit as MA
import mcs_test as MCS
import fetch_alfred as FA
import decision_protocol as DP
from canonical_emit import emit, flush

RES = MULTI / "results"; CACHE = MULTI / "data" / "state_ur.parquet"
STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
          "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
          "TX","UT","VT","VA","WA","WV","WI","WY","DC"]
MODELS = MA.EB_MODELS

MIN_TRAIN = 60          # months of history before a month is nowcast: 5 years, enough for a seasonal fit
WARMUP_YEARS = 15       # panel starts 1976; scoring from 1991 gives every state a full cycle of history and matches
                        # the unrestricted-window contrast the paper reports (1991-01)
MIN_STATE_PANEL_MONTHS = 300   # 25 years of state UR: below this the post-warm-up scored window is too short for an MCS

EB_TUR_EFFECTIVE = pd.Timestamp("1993-03-01")


def load_state_ur():
    if CACHE.exists(): return pd.read_parquet(CACHE)
    from fredapi import Fred
    fred = Fred(api_key=FA._fred_key()); cols = {}
    for s in STATES:
        try:
            x = fred.get_series(f"{s}UR")
            if x is not None and len(x.dropna()) > 240: cols[s] = x
        except Exception as e:
            print(f"  [skip] {s}UR: {e}", flush=True)
    df = pd.DataFrame(cols); df.index = pd.to_datetime(df.index); df = df.sort_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(CACHE); print(f"  fetched {df.shape[1]} states")
    return df


def eb_on(fc, realized, protocol="published"):
    """EB TUR trigger -> nullable ("boolean" dtype) Series.

    Follows 20 CFR 615.12(e): 6.5 floor on the (e)(3)-rounded 3-month average, 110% look-back
    against either of the two prior years under the (e)(1)(ii) rounding cascade. Two approximations:
    the 3-month value averages three published SA rates (the regulation averages levels), and the
    temporary 2011-13 third-year look-back is not modelled. Verified against DOL trigger notices
    2009-2013: ON/OFF agrees in 99.7% of state-weeks, the 3-month value within 0.1pp in 98%
    (3,314 state-weeks from 67 DOL trigger notices; see external_facts.json,
    dol_trigger_notice_comparison_2026_09_20)."""
    a = DP.statutory_round_tur(DP.three_month_average(fc, realized, protocol))   # current leg: 2 published + 1 forecast, (e)(3)-rounded
    ar = DP.statutory_round_tur(DP.reference_average(realized))                  # reference legs: entirely historical, (e)(3)-rounded
    ref1, ref2 = DP.date_shift(ar, 12), DP.date_shift(ar, 24)   # calendar-aware: the state UR panel has a genuine Oct-2025 gap
    flat_ok = (a >= 6.5).astype("boolean").mask(a.isna())
    lookback_ok = DP.statutory_lookback_ok(a, ref1) | DP.statutory_lookback_ok(a, ref2)   # (e)(1)(ii)-rounded ratio, either prior year
    return flat_ok & lookback_ok


def _calendar_filled(u: pd.Series) -> pd.Series:
    """`u` reindexed onto its full monthly calendar grid, with any single-month gap bridged by
    forward-fill (ffill(limit=1)) -- built for `nowcast()`'s training array only."""
    grid = DP._monthly_grid(u)
    filled = u.reindex(grid).ffill(limit=1)
    if filled.isna().any():
        bad = filled[filled.isna()].index[0]
        raise ValueError(
            f"nowcast(): un-bridgeable gap at {bad.date()} (more than 1 consecutive missing month) "
            f"in the state UR panel -- ffill(limit=1) only bridges a single-month gap; a wider "
            f"gap needs an explicit, audited decision, not a silent fill."
        )
    return filled


def nowcast(u):
    """One-step-ahead forecast for every month, refit on all data strictly before it."""
    u_filled = _calendar_filled(u)
    out = {m: pd.Series(index=u.index, dtype=float) for m in MODELS}
    first_year = u.index.year.min() + WARMUP_YEARS
    for t in u.index:
        if t.year < first_year:
            continue
        tr = u_filled[u_filled.index < t].to_numpy(float)
        if len(tr) < MIN_TRAIN:
            continue
        for m, fn in MODELS.items():
            fc = float(MA.safe_forecast(fn, m, tr, 12, 1, context=f"state={u.name}, t={t}")[0])
            out[m].loc[t] = fc
    return out


def _episode_lengths(flip: pd.Series) -> pd.Series:
    """Length in months of each contiguous True-run in a boolean Series, indexed by episode order
    (1, 2, 3, ...)."""
    grid = pd.date_range(flip.index.min(), flip.index.max(), freq="MS")
    flip = flip.reindex(grid, fill_value=False)
    starts = flip & ~flip.shift(1, fill_value=False)
    if not starts.any():
        return pd.Series(dtype=int)
    return flip[flip].groupby(starts.cumsum()[flip]).size()


def _new_acc():
    return dict(tot_flip=0, tot_eval=0, tot_near=0, tot_near_flip=0, tot_knife=0, tot_ep=0,
                tot_isolated=0,
                states_div=0, per=[], trigger_counts={}, trigger_counts_clean={}, trigger_tied_states=[],
                flip_spreads=[], flip_dates=[])


def _isolated_flips(flip: pd.Series, all_on: pd.Series) -> pd.Series:
    """Flip months with NO adjacent month (t-1 or t+1, calendar-aware) on which every tied model
    agrees the trigger is ON."""
    grid = pd.date_range(flip.index.min() - pd.DateOffset(months=1),
                         flip.index.max() + pd.DateOffset(months=1), freq="MS")
    f = flip.reindex(grid, fill_value=False).astype(bool)
    on = all_on.reindex(grid, fill_value=False).astype(bool)
    adjacent_on = on.shift(1, fill_value=False) | on.shift(-1, fill_value=False)
    return (f & ~adjacent_on).reindex(flip.index)


def _tied_models(u, nc, models=None):
    """Tied-model-set determination on the full error record: reindex to the post-warm-up common
    index, floor at 30 observed months, run MCS.mcs(alpha=0.10, seed=0) on the annual-mean error
    matrix, and map surviving indices back to model names via `models`'s own declaration order."""
    if models is None:
        models = MODELS
    common = u.index[u.index.year >= u.index.year.min() + WARMUP_YEARS]
    err = pd.DataFrame({m: (nc[m] - u).abs() for m in models}).reindex(common).dropna()
    if len(err) < 30:
        return [], common
    surv, _ = MCS.mcs(err.groupby(err.index.year).mean().to_numpy(float), alpha=0.10, seed=0)
    tied = [list(models)[i] for i in surv]
    return tied, common


def scan(df, protocol="published", windows=None):
    """Per-state EB divergence scan under one protocol, across one or more scored windows."""
    if windows is None:
        windows = {"statutory": EB_TUR_EFFECTIVE, "full": None}
    acc = {w: _new_acc() for w in windows}
    bias_sum = {}; bias_n = {}     # robustness: is one biased model driving the flips? window-invariant, accumulated once.
    for s in df.columns:
        u = df[s].dropna()
        if len(u) < MIN_STATE_PANEL_MONTHS: continue
        nc = nowcast(u)
        tied, common = _tied_models(u, nc)
        if len(tied) < 2: continue
        E_all = pd.DataFrame({m: eb_on(nc[m].reindex(u.index), u, protocol)
                              for m in tied}).reindex(common).dropna()
        for m in tied:
            b = float((nc[m].reindex(common) - u.reindex(common)).mean())
            bias_sum[m] = bias_sum.get(m, 0.0) + b; bias_n[m] = bias_n.get(m, 0) + 1
        for wname, start in windows.items():
            E = E_all if start is None else E_all[E_all.index >= start]
            if E.empty:
                continue
            A = pd.DataFrame({m: DP.statutory_round_tur(DP.three_month_average(nc[m].reindex(u.index), u, protocol))
                              for m in tied}).reindex(E.index)
            flip = E.any(axis=1) & ~E.all(axis=1)                         # months tied models disagree on EB status
            isolated = _isolated_flips(flip, E_all.all(axis=1))          # flips with no commonly-on neighbour (full record)
            near = ((A >= 6.0) & (A <= 7.0)).any(axis=1)                  # any tied model's TRIGGER VALUE within 0.5pp of 6.5
            spread = np.round(A.max(axis=1) - A.min(axis=1), 1)
            knife = flip & (spread < 0.3)                                 # knife-edge: tied models' 3-month TRIGGER VALUES within
                                                                          # 0.3pp (= forecasts within ~0.9pp, one leg of three differs)
            ep_lengths = _episode_lengths(flip)
            ep = len(ep_lengths)                                          # distinct contiguous flip-EPISODES
            max_ep_len = int(ep_lengths.max()) if ep else 0
            a = acc[wname]
            a["tot_flip"] += int(flip.sum()); a["tot_eval"] += len(E); a["tot_near"] += int(near.sum())
            a["tot_near_flip"] += int((flip & near).sum()); a["tot_knife"] += int(knife.sum())
            a["tot_ep"] += ep
            a["tot_isolated"] += int(isolated.sum())
            a["states_div"] += int(flip.sum() > 0)
            if flip.any():
                a["flip_spreads"].append((s, [float(x) for x in spread[flip].tolist()]))
                a["flip_dates"].append(
                    (s, [(str(d.date()), float(x)) for d, x in zip(spread[flip].index, spread[flip].tolist())]))
            # robustness: the "trigger" model = net most-often ON while others OFF at flip months
            if int(flip.sum()) > 0:
                net_on = {m: int((E[m] & flip).sum()) - int((~E[m] & flip).sum()) for m in tied}
                _max_net = max(net_on.values())
                winners = [m for m in net_on if net_on[m] == _max_net]
                trg = winners[0]
                a["trigger_counts"][trg] = a["trigger_counts"].get(trg, 0) + 1
                if len(winners) == 1:
                    a["trigger_counts_clean"][trg] = a["trigger_counts_clean"].get(trg, 0) + 1
                else:
                    a["trigger_tied_states"].append({"state": s, "tied_models": winners, "net_on": net_on})
            a["per"].append({"state": s, "tied": len(tied), "flip_months": int(flip.sum()),
                             "flip_episodes": ep, "knife_edge": int(knife.sum()),
                             "max_episode_months": max_ep_len,
                             "isolated_flip_months": int(isolated.sum())})
    for w in acc:
        acc[w]["bias_sum"] = bias_sum; acc[w]["bias_n"] = bias_n
    return acc


def main():
    df = load_state_ur(); print(f"state UR panel: {df.shape[1]} states x {df.shape[0]} months")
    acc = scan(df, protocol="published")
    S, S_FULL = acc["statutory"], acc["full"]

    acc_alt = scan(df, protocol="all_forecast")
    S_alt = acc_alt["statutory"]
    pd.DataFrame([{
        "episodes": S_alt["tot_ep"], "states_divergent": S_alt["states_div"], "states_total": len(S_alt["per"]),
        "near_trigger_pct": round(100 * S_alt["tot_near_flip"] / max(S_alt["tot_near"], 1), 1),
    }]).to_csv(RES / "state_eb_alt_protocol.csv", index=False)
    tot_ep, states_div = S["tot_ep"], S["states_div"]
    tot_flip, tot_eval, tot_near, tot_near_flip, tot_knife = (
        S["tot_flip"], S["tot_eval"], S["tot_near"], S["tot_near_flip"], S["tot_knife"])
    per = S["per"]

    r = pd.DataFrame(per).sort_values("flip_episodes", ascending=False)
    r.to_csv(RES / "state_eb_decision.csv", index=False)

    pd.DataFrame([{
        "states_divergent": states_div, "states_total": len(per),
        "distinct_episodes": tot_ep,
        "near_trigger_flip": tot_near_flip, "near_trigger_total": tot_near,
        "near_trigger_pct": round(100 * tot_near_flip / max(tot_near, 1), 1),
        "unconditional_flip": tot_flip, "unconditional_total": tot_eval,
        "unconditional_pct": round(100 * tot_flip / max(tot_eval, 1), 1),
        "knife_edge": tot_knife, "knife_edge_pct": round(100 * tot_knife / max(tot_flip, 1), 0),
    }]).to_csv(RES / "state_eb_summary.csv", index=False)

    pd.DataFrame([
        {"window": "statutory_1993_03", "states_divergent": states_div, "states_total": len(per),
         "episodes": tot_ep, "flip_months": tot_flip, "evaluated_months": tot_eval},
        {"window": "unrestricted_1991_01", "states_divergent": S_FULL["states_div"],
         "states_total": len(S_FULL["per"]), "episodes": S_FULL["tot_ep"],
         "flip_months": S_FULL["tot_flip"], "evaluated_months": S_FULL["tot_eval"]},
    ]).to_csv(RES / "state_eb_window_comparison.csv", index=False)

    trig = dict(sorted(S["trigger_counts"].items(), key=lambda x: -x[1]))
    bias = {m: round(S["bias_sum"][m] / S["bias_n"][m], 3) for m in S["bias_sum"]}
    top_share = (max(S["trigger_counts"].values()) / sum(S["trigger_counts"].values())) if S["trigger_counts"] else 0
    pd.DataFrame([{"model": m, "trigger_states": trig.get(m, 0),
                   "share_of_divergent": round(trig.get(m, 0) / states_div, 4) if states_div else 0.0}
                 for m in MODELS]).to_csv(RES / "state_eb_trigger_model.csv", index=False)

    print("\n=== STATE EB DECISION-PROJECTION (statute-faithful protocol: monthly refit, statutory-window "
          "scoring, (e)(3)/(e)(1)(ii) rounding) ===")
    print(r.head(15).to_string(index=False))
    print(f"\n  states where accuracy-tied models FLIP the EB decision: {states_div}/{len(per)}")
    print(f"  DISTINCT EB flip MONTHS switched on/off by the arbitrary tied-model choice: {tot_ep}")
    print(f"  flip MONTHS conditional on NEAR-TRIGGER (any tied model 3mo-avg in [6,7]): {tot_near_flip}/{tot_near} "
          f"({100*tot_near_flip/max(tot_near,1):.1f}%)   [unconditional {tot_flip}/{tot_eval} = {100*tot_flip/max(tot_eval,1):.1f}%]")
    print(f"  knife-edge share (flip months with tied-model spread < 0.3pp): {tot_knife}/{tot_flip} "
          f"({100*tot_knife/max(tot_flip,1):.0f}%)")
    print(f"  ROBUSTNESS (not one biased model): 'trigger' model across flip-states = {trig} (top share {top_share:.0%});")
    print(f"    mean state-UR forecast bias by model (pp) = {bias}")
    print(f"  WINDOW CONTRAST: unrestricted (from 1991-01) gives {S_FULL['tot_ep']} episodes / "
          f"{S_FULL['states_div']} of {len(S_FULL['per'])} states -- {S_FULL['tot_ep'] - tot_ep} of those episodes "
          f"predate the optional TUR indicator's {EB_TUR_EFFECTIVE.date()} effective date and are dropped from "
          f"the statutory-window headline above.")
    print("  CAVEAT (data limitation, disclose honestly): REVISED state UR (real-time state UR is not available "
          "back to 1991); see sahm_evans_realtime.py for the real-time-honest national complement.")
    print(f"  wrote {RES/'state_eb_decision.csv'}")

    _spec = (f"EB TUR trigger: 3mo-avg state UR (statutory rounding, 20 CFR 615.12(e)(3)) >= 6.5% AND "
             f">= 110% of the same period in either prior year (615.12(e)(1)(ii) rounding); tied models "
             f"from per-year MCS at alpha=0.10; models={list(MODELS)}. SCORED FROM {EB_TUR_EFFECTIVE.date()}, "
             f"the effective date of the optional TUR indicator (EUCA 70 s203(f), P.L. 102-318 s201; "
             f"external_facts uipl_45_92) -- lookback legs still read earlier data, as the statute "
             f"contemplates, but no month is SCORED before the provision existed. The unrestricted "
             f"reading is published alongside as eb.fullwindow_*")
    emit("eb.episodes", int(tot_ep),
         spec=f"distinct contiguous flip-episodes summed across states. {_spec}",
         cites=["S7 body: episode count"],
         cross_check="state_eb_decision.csv: flip_episodes.sum()")
    emit("eb.flip_months", int(tot_flip),
         spec=f"flip MONTHS summed across states -- equal to eb.episodes because every flip episode "
              f"in this exercise is one month long. That is episode LENGTH and says nothing about "
              f"eb.isolated_flip_months, which asks a different question (does a flip month have an "
              f"adjacent commonly-ON month?) and answers it for only a small minority of these "
              f"months. {_spec}",
         cites=["S7 body: episode count"],
         cross_check="state_eb_decision.csv: flip_months.sum()")
    emit("eb.isolated_flip_months", int(S["tot_isolated"]), of=int(tot_flip),
         adjacent=int(tot_flip) - int(S["tot_isolated"]),
         spec=f"of the flip months, how many have NO adjacent month (t-1 or t+1, on the full "
              f"record) on which every tied model agrees the trigger is ON. Such a flip is not a "
              f"timing shift at the edge of an agreed-on spell: under the tied model that says ON it "
              f"starts an EB period the other tied models never start, and 20 CFR 615.11(c) makes "
              f"that period at least 13 weeks long -- i.e. an entire benefit spell that turns on "
              f"which tied model is used. The complement (flip months adjacent to a commonly-on "
              f"month) are the timing shifts S7's prose describes. Computed PER MONTH: the neighbour "
              f"test is 'commonly ON', so the leading month of a multi-month early start (flip, flip, "
              f"commonly-on) counts as isolated -- a conservative (over-)count; an undetermined "
              f"neighbour (NA) also reads as not commonly-on. Exact today because every flip episode "
              f"is one month long (see eb.max_episode_months). {_spec}",
         cites=["S7 body: episode count"],
         cross_check="state_eb_decision.csv: isolated_flip_months.sum()")
    _ep = r[r.flip_episodes > 0]
    emit("eb.max_episode_months",
         int(_ep.max_episode_months.max()) if len(_ep) else 0,
         spec="longest single flip episode, in months, across all states -- the true per-episode max, "
              "not an average. A value of 1 means no episode spans more than a single month, so none "
              "can correspond to a statutory EB period (minimum 13 consecutive weeks, 20 CFR 615.11).",
         cites=["S7 body: episode count"],
         cross_check="state_eb_decision.csv: max_episode_months.max()")
    emit("eb.states_divergent", int(states_div), of=int(len(per)),
         spec=f"states with at least one flip month. {_spec}",
         cites=["S7 body: state count"],
         cross_check="state_eb_decision.csv: (flip_episodes > 0).sum()")
    emit("eb.states_total", int(len(per)),
         spec="MCS-eligible states (>=2 tied models and enough history) with any scored data under the "
              "statutory window. Window-invariant: the tie set is determined on the full error record, "
              "so restricting the SCORED window changes which states diverge and can drop a state "
              "entirely if its post-effective-date history is empty, never which models a state's "
              "record can separate",
         cites=["S7 body: state count"],
         cross_check="state_eb_decision.csv: row count")

    _wspec = (f"SAME computation as the headline key of this name, but with NO statutory-date "
              f"restriction on the scored window (from 1991-01, the panel's own start). Reported only "
              f"as the robustness contrast for the re-scoping; the optional TUR indicator did not "
              f"exist before {EB_TUR_EFFECTIVE.date()}, so this reading counts months in which no state "
              f"could have triggered on it. Never to be quoted as a result")
    emit("eb.fullwindow_episodes", int(S_FULL["tot_ep"]),
         spec=f"distinct contiguous flip-episodes, unrestricted window. {_wspec}",
         cites=["S7 body: unrestricted-window contrast"],
         cross_check="state_eb_window_comparison.csv: episodes where window=='unrestricted_1991_01'")
    emit("eb.fullwindow_states_divergent", int(S_FULL["states_div"]), of=int(len(S_FULL["per"])),
         spec=f"states with at least one flip month, unrestricted window. {_wspec}",
         cites=["S7 body: unrestricted-window contrast"],
         cross_check="state_eb_window_comparison.csv: states_divergent where window=='unrestricted_1991_01'")
    emit("eb.fullwindow_flip_months", int(S_FULL["tot_flip"]),
         spec=f"flip months, unrestricted window. {_wspec}",
         cites=["S7 body: unrestricted-window contrast"],
         cross_check="state_eb_window_comparison.csv: flip_months where window=='unrestricted_1991_01'")
    emit("eb.prestatute_flip_months", int(S_FULL["tot_flip"] - tot_flip),
         of=int(S_FULL["tot_flip"]),
         episodes=int(S_FULL["tot_ep"] - tot_ep),
         spec=f"flip months falling BEFORE the optional TUR indicator's effective date, i.e. those "
              f"dropped by scoring from {EB_TUR_EFFECTIVE.date()} -- months in which no state could "
              f"have triggered on the modelled indicator at all",
         cites=["S7 body: unrestricted-window contrast"],
         cross_check="state_eb_window_comparison.csv: flip_months(unrestricted) - flip_months(statutory)")
    emit("eb.near_trigger_pct", round(100 * tot_near_flip / max(tot_near, 1), 1),
         numerator=int(tot_near_flip), denominator=int(tot_near),
         spec="flip months as a share of near-trigger months (any tied model 3mo-avg in [6,7])",
         cites=["S7 body: near-trigger share"],
         cross_check="state_eb_summary.csv: near_trigger_pct")
    emit("eb.unconditional_pct", round(100 * tot_flip / max(tot_eval, 1), 1),
         numerator=int(tot_flip), denominator=int(tot_eval),
         spec="flip months as a share of all evaluated months (unconditional)",
         cites=["S7 body: unconditional flip share"],
         cross_check="state_eb_summary.csv: unconditional_pct")
    emit("eb.knife_edge_pct", round(100 * tot_knife / max(tot_flip, 1), 0),
         numerator=int(tot_knife), denominator=int(tot_flip),
         spec="flip months where the tied models' (e)(3)-rounded 3-month trigger values are within "
              "0.3pp of each other (only the forecast month differs, so the forecasts themselves are "
              "within roughly 0.9pp) and the trigger conditions still resolve differently, as a share of "
              "flip months",
         cites=["S7 body: knife-edge share"],
         cross_check="state_eb_summary.csv: knife_edge_pct")
    emit("eb.alt_protocol_episodes", int(S_alt["tot_ep"]),
         spec="SAME as eb.episodes but under the 'all_forecast' protocol (all three months of the "
              "CURRENT 3-month average model-produced, vs. 'published''s one month; the two "
              "prior-year comparator averages are never model-produced in either protocol). A "
              "leg-exposure sensitivity arm, not a competing estimate of the flip count -- "
              "'all_forecast' is further from what the statute actually does, not closer",
         cites=["S7 body: leg-exposure sensitivity"],
         cross_check="state_eb_alt_protocol.csv: episodes")
    emit("eb.alt_protocol_states_divergent", int(S_alt["states_div"]), of=int(len(S_alt["per"])),
         spec="SAME as eb.states_divergent but under the 'all_forecast' protocol",
         cites=["S7 body: leg-exposure sensitivity"],
         cross_check="state_eb_alt_protocol.csv: states_divergent")
    emit("eb.alt_protocol_near_trigger_pct",
         round(100 * S_alt["tot_near_flip"] / max(S_alt["tot_near"], 1), 1),
         spec="SAME as eb.near_trigger_pct but under the 'all_forecast' protocol",
         cites=["S7 body: leg-exposure sensitivity"],
         cross_check="state_eb_alt_protocol.csv: near_trigger_pct")

    _genuine = r[(r.flip_episodes > 0) & (r.flip_months > r.knife_edge)]
    emit("eb.genuine_divergent_states", int(len(_genuine)), of=int(states_div),
         states=sorted(_genuine.state.tolist()),
         spec="divergent states with at least one flip MONTH that is not knife-edge (tied-model spread "
              ">= 0.3pp) -- the complement of a state being knife-edge in its entirety",
         cites=["S7 body: genuine (non-knife-edge) divergence"],
         cross_check="state_eb_decision.csv: states where flip_episodes>0 and flip_months>knife_edge")
    emit("eb.knife_edge_only_states", int(states_div) - int(len(_genuine)), of=int(states_div),
         spec="divergent states whose entire flip-month record is knife-edge; "
              "states_divergent - eb.genuine_divergent_states",
         cites=["S7 body: knife-edge share"],
         cross_check="state_eb_decision.csv: states_divergent - genuine_divergent_states")

    _flip_dates_by_state = {}
    for _st, _pairs in S["flip_dates"]:
        _flip_dates_by_state.setdefault(_st, []).extend(_pairs)
    _genuine_detail = {}
    for _st in sorted(_genuine.state.tolist()):
        _genuine_pairs = [(d, sp) for d, sp in _flip_dates_by_state.get(_st, []) if sp >= 0.3]
        if _genuine_pairs:
            _d, _sp = _genuine_pairs[0]
            _genuine_detail[_st] = {"date": _d, "spread_pp": round(_sp, 1)}
    emit("eb.genuine_divergent_states_detail", len(_genuine_detail), of=int(len(_genuine)),
         dates={k: v["date"] for k, v in _genuine_detail.items()},
         spreads_pp={k: v["spread_pp"] for k, v in _genuine_detail.items()},
         spec="for each state in eb.genuine_divergent_states, the exact calendar month and tied-model "
              "spread (pp) of that state's one non-knife-edge flip month/episode",
         cites=["S7 body: genuine (non-knife-edge) divergence"],
         cross_check="state_eb_decision.csv + per-flip-month spread: the one flip month per genuine "
                     "state with spread>=0.3pp")

    _sweep_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    _spread_by_state = {}
    for _st, _spreads in S["flip_spreads"]:
        _spread_by_state.setdefault(_st, []).extend(_spreads)
    _sweep_genuine_states = {}
    for _thr in _sweep_thresholds:
        _n_genuine = sum(1 for _st, _sps in _spread_by_state.items() if any(_sp >= _thr for _sp in _sps))
        _sweep_genuine_states[_thr] = _n_genuine
    print(f"  KNIFE-EDGE THRESHOLD SWEEP: genuine-divergence state count at "
          f"{{{', '.join(f'{t}pp: {n}' for t, n in _sweep_genuine_states.items())}}}")
    pd.DataFrame([{"threshold_pp": t, "genuine_states": n} for t, n in _sweep_genuine_states.items()]
                 ).to_csv(RES / "state_eb_knife_edge_sweep.csv", index=False)
    emit("eb.knife_edge_sweep_genuine_states", _sweep_genuine_states[0.3], of=int(states_div),
         thresholds_pp=_sweep_thresholds,
         counts=[_sweep_genuine_states[t] for t in _sweep_thresholds],
         spec="genuine-divergence state count (>=1 flip month with tied-model spread at or above the "
              "threshold) swept across knife-edge thresholds 0.1/0.2/0.3/0.4/0.5 percentage points; "
              "the 0.3pp entry matches eb.genuine_divergent_states by construction (same definition, "
              "computed from the same underlying per-flip-month spreads) -- reported to show how much "
              "the headline 2-state figure moves under a different, equally defensible cutoff, not to "
              "propose a different headline",
         cites=["S7 body: knife-edge threshold sweep"],
         cross_check="state_eb_knife_edge_sweep.csv: genuine_states where threshold_pp==0.3")
    for _m in MODELS:
        emit(f"eb.trigger_{_m}", int(trig.get(_m, 0)), of=int(states_div),
             spec=f"divergent states where {_m} is the 'net-on' tied model (the model whose forecast "
                  f"alone would trigger the decision at that state's flip months)",
             cites=["S7 body: robustness across tied models"],
             cross_check="state_eb_trigger_model.csv: trigger_states where model=='%s'" % _m)
    emit("eb.trigger_ar1_pct", round(100 * trig.get("ar1", 0) / max(states_div, 1), 0),
         spec="eb.trigger_ar1 as a share of states_divergent, rounded to the nearest percent",
         cites=["S7 body: robustness across tied models"],
         cross_check="round(100 * eb.trigger_ar1 / eb.states_divergent)")
    emit("eb.trigger_ar1_ets_pct",
         round(100 * (trig.get("ar1", 0) + trig.get("ets", 0)) / max(states_div, 1), 0),
         spec="eb.trigger_ar1 + eb.trigger_ets as a share of states_divergent, rounded to the nearest "
              "percent",
         cites=["S7 body: robustness across tied models"],
         cross_check="round(100 * (eb.trigger_ar1 + eb.trigger_ets) / eb.states_divergent)")
    _tied_states = sorted(t["state"] for t in S["trigger_tied_states"])
    trig_clean = dict(sorted(S["trigger_counts_clean"].items(), key=lambda x: -x[1]))
    pd.DataFrame(S["trigger_tied_states"]).to_csv(RES / "state_eb_trigger_ties.csv", index=False)
    emit("eb.trigger_tied_states", len(_tied_states), of=int(states_div),
         states=_tied_states,
         spec="divergent states where the 'net-on' trigger-model attribution above is an EXACT tie "
              "among 2+ models, broken only by MODELS' declaration order (ar1 before ets/theta) with "
              "no other basis for the attribution -- the complement of a state's trigger-model credit "
              "being a genuine, unique net-dominance win",
         cites=["S7 body: robustness across tied models"],
         cross_check="state_eb_trigger_ties.csv: row count")
    for _m in MODELS:
        emit(f"eb.trigger_{_m}_clean", int(trig_clean.get(_m, 0)),
             of=int(states_div) - len(_tied_states),
             spec=f"SAME as eb.trigger_{_m} but excluding the {len(_tied_states)} states in "
                  f"eb.trigger_tied_states -- a model only counts here if it is the UNIQUE net-on "
                  f"maximizer at that state, not merely the insertion-order tie-break winner",
             cites=["S7 body: robustness across tied models"],
             cross_check="state_eb_trigger_ties.csv / state_eb_trigger_model.csv: recomputed excluding exact-tie states")
    emit("eb.trigger_ar1_pct_clean",
         round(100 * trig_clean.get("ar1", 0) / max(states_div - len(_tied_states), 1), 0),
         spec="eb.trigger_ar1_clean as a share of (states_divergent - eb.trigger_tied_states), "
              "rounded to the nearest percent -- the tie-excluded analogue of eb.trigger_ar1_pct",
         cites=["S7 body: robustness across tied models"],
         cross_check="round(100 * eb.trigger_ar1_clean / (eb.states_divergent - eb.trigger_tied_states))")
    emit("eb.trigger_ar1_ets_pct_clean",
         round(100 * (trig_clean.get("ar1", 0) + trig_clean.get("ets", 0))
               / max(states_div - len(_tied_states), 1), 0),
         spec="eb.trigger_ar1_clean + eb.trigger_ets_clean as a share of (states_divergent - "
              "eb.trigger_tied_states), rounded to the nearest percent",
         cites=["S7 body: robustness across tied models"],
         cross_check="round(100 * (eb.trigger_ar1_clean + eb.trigger_ets_clean) / "
                      "(eb.states_divergent - eb.trigger_tied_states))")
    for _m, _b in bias.items():
        emit(f"eb.trigger_model_bias_{_m}_pp", _b,
             spec=f"mean state-UR forecast bias of {_m}, in percentage points -- the average, "
                  f"across every state where {_m} was among the accuracy-tied models, of that "
                  f"state's own mean forecast bias over its full post-warmup evaluation window -- "
                  f"backs S7's claim that the trigger ordering 'is not unrelated to' the models' "
                  f"own small forecast biases",
             cites=["S7 body: robustness across tied models"],
             cross_check=f"state_eb_decision.py main(): bias['{_m}']")
    flush()


if __name__ == "__main__":
    main()
