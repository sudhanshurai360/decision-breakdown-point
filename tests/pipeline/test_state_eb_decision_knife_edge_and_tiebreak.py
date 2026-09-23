"""Two properties of `state_eb_decision.py::scan()` that a silent defect would otherwise move.

Property 1: `spread = A.max(axis=1) - A.min(axis=1)` must be rounded to 1 decimal before being
compared against the literal `0.3`. Compared raw it misclassifies ~67% of true-0.3pp gaps as
knife-edge, purely from float-subtraction noise on `DP.statutory_round_tur()`'s 1-decimal output
(most 1-decimal percentages are not exactly representable in binary). Rounding first is exact by
construction, since every entry of `A` is itself exact to 1 decimal.

Property 2: `trg = max(net_on, key=net_on.get)` silently broke an EXACT net_on tie by `MODELS`'
declaration order, with no disclosure. scan() therefore tracks `winners` (all models achieving the
max) and records ties in `trigger_tied_states`/`trigger_counts_clean`, while `trg = winners[0]`
reproduces `max()`'s selection bit-for-bit (verified below against scan()'s OWN recorded net_on),
so `state_eb_trigger_model.csv`'s headline counts are unaffected.

TESTING APPROACH -- rewritten. The previous version of this file claimed a controlled scenario
could not be driven through scan() ("a ~30-60-minute real-data integration") and so tested both
properties by REIMPLEMENTING scan()'s expressions inline, backed by one `inspect.getsource()`
string match. That was not falsifiable: deleting the `np.round(..., 1)` from scan() left both
behavioural-sounding tests green, and only the string match failed. Worse,
test_winners_first_reproduces_old_max_selection_bit_for_bit never referenced `SED` at all -- it
asserted a property of CPython's `max()`.

scan()'s cost is entirely in `nowcast()` (a per-month refit of four models across 51 states).
Everything this file cares about -- `_tied_models`, `eb_on`, the A matrix, `flip`, `spread`,
`knife`, `net_on`/`winners`/`trigger_tied_states` -- is downstream of that and cheap. So the tests
below monkeypatch `SED.nowcast` ONLY, hand scan() a single synthetic state, and assert on scan()'s
real returned accumulator. Real production run for reference: 6 tied states (GA, NM, NV, RI, VA,
VT -- NM a literal 3-way exact-zero tie), all correctly excluded from `eb.trigger_*_clean`, with
`eb.trigger_ar1`/`eb.trigger_ets`/`eb.trigger_theta` at 20/2/2 -- see
multiseries/results/state_eb_trigger_ties.csv.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "code" / "pipeline"
sys.path.insert(0, str(CODE))

import decision_protocol as DP  # noqa: E402
import state_eb_decision as SED  # noqa: E402

# --------------------------------------------------------------------------------------------
# A synthetic single-state panel that scan() will actually accept.
#   * 480 months (>= MIN_STATE_PANEL_MONTHS = 300), starting 1980 so the post-WARMUP_YEARS common
#     index is 300 months (>= the 30 _tied_models() needs).
#   * a flat 4.0 realized rate, so every reference leg is 4.0 and the (e)(1)(ii) look-back ratio is
#     comfortably above 110 for any trigger value at or above 6.5 -- i.e. the 6.5 FLOOR alone
#     decides ON/OFF, which is what lets a chosen A matrix drive flip/knife directly.
#   * per-model Gaussian forecast noise of equal scale, so MCS keeps all four models tied.
# --------------------------------------------------------------------------------------------
PANEL_INDEX = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
BASE_RATE = 4.0
TARGET = pd.Timestamp("2005-06-01")


def _panel() -> pd.DataFrame:
    return pd.DataFrame({"ZZ": pd.Series(BASE_RATE, index=PANEL_INDEX, name="ZZ")})


def _nowcast_hitting(trigger_values):
    """A drop-in replacement for SED.nowcast() whose forecasts land each model's statutory trigger
    value (the (e)(3)-rounded 3-month average, i.e. scan()'s `A` matrix) exactly on
    `trigger_values[i]` at TARGET, and leave it at the flat baseline everywhere else.

    Inverts DP.three_month_average()'s 'published' protocol: A = round((fc + u[t-1] + u[t-2]) / 3),
    and u is flat at BASE_RATE, so fc = 3*A - 2*BASE_RATE."""
    assert len(trigger_values) == len(SED.MODELS)

    def _fake(u):
        rng = np.random.default_rng(7)
        out = {m: u + pd.Series(rng.normal(0, 0.05, len(u)), index=u.index) for m in SED.MODELS}
        for m, a in zip(list(SED.MODELS), trigger_values):
            out[m].loc[TARGET] = 3.0 * a - 2.0 * BASE_RATE
        return out
    return _fake


def _scan_full(monkeypatch, trigger_values):
    """Run the REAL SED.scan() over the synthetic panel and return its 'full'-window accumulator,
    after asserting the fixture actually set up the intended A matrix and tied-model set."""
    fake = _nowcast_hitting(trigger_values)
    monkeypatch.setattr(SED, "nowcast", fake)
    u = _panel()["ZZ"]
    nc = fake(u)
    tied, _common = SED._tied_models(u, nc)
    assert len(tied) == len(SED.MODELS), (
        f"fixture precondition: all {len(SED.MODELS)} models must survive MCS so the A matrix "
        f"below is the one scan() actually spreads over; got {tied}"
    )
    A = pd.DataFrame({m: DP.statutory_round_tur(DP.three_month_average(nc[m], u, "published"))
                      for m in SED.MODELS})
    assert list(A.loc[TARGET]) == list(trigger_values), (
        f"fixture precondition: A at {TARGET.date()} is {A.loc[TARGET].to_dict()}, "
        f"not the requested {trigger_values}"
    )
    return SED.scan(_panel(), windows={"full": None})["full"]


# --------------------------------- Property 1: the rounded spread ---------------------------------

def test_float_subtraction_artifact_is_real_in_this_environment():
    """NOT a scan() test: this only establishes that the underlying float artifact still exists on
    this numpy/pandas build, so the scan()-level tests below are exercising a live hazard rather
    than a historical one. A true 0.3pp gap between two statutory-rounded rates raw-subtracts to
    slightly LESS than 0.3 for many pairs in the realistic UR range."""
    a_vals = [round(x, 1) for x in np.arange(4.0, 10.0, 0.1)]
    m1 = DP.statutory_round_tur(pd.Series(a_vals)).to_numpy(float)
    m2 = DP.statutory_round_tur(pd.Series([round(a - 0.3, 1) for a in a_vals])).to_numpy(float)
    A = pd.DataFrame({"m1": m1, "m2": m2})
    raw_spread = A.max(axis=1) - A.min(axis=1)
    assert (raw_spread < 0.3).any(), (
        "the float-subtraction artifact no longer reproduces on this numpy/pandas version -- the "
        "scan()-level knife-edge tests below would then be testing nothing; update the swept range"
    )
    # and the specific pair the scan()-level fixture uses must be one of the affected ones
    assert (6.6 - 6.3) < 0.3, "the 6.6/6.3 pair used by the scan() fixtures is no longer affected"


def test_scan_does_not_classify_a_true_0_3pp_gap_as_knife_edge(monkeypatch):
    """THE behavioural test for Property 1, through the real SED.scan(). Tied models split 6.6/6.6
    vs 6.3/6.3 at one month: the 6.5 floor puts two models ON and two OFF, so that month IS a flip,
    and the tied models' trigger values are a TRUE 0.3pp apart -- by the rule's own definition not
    knife-edge. Raw (unrounded) subtraction gives 0.29999999999999982 < 0.3 and would miscount it.

    Deleting the `np.round(..., 1)` from state_eb_decision.py's `spread = ...` line makes this test
    fail (tot_knife 0 -> 1); the reimplementation it replaced stayed green under that mutation."""
    a = _scan_full(monkeypatch, [6.6, 6.6, 6.3, 6.3])
    assert a["tot_eval"] == 300, "the whole post-warm-up window must be evaluated"
    assert a["tot_flip"] == 1, "exactly the one constructed month must register as a flip"
    assert a["flip_spreads"] == [("ZZ", [0.3])], (
        f"scan() must record the spread as an exact 0.3, not a float-noise 0.2999...; "
        f"got {a['flip_spreads']}"
    )
    assert a["tot_knife"] == 0, (
        "a TRUE 0.3pp gap was counted as knife-edge -- scan()'s spread is being compared against "
        "0.3 without the 1-decimal rounding, reopening the float-subtraction misclassification"
    )


def test_scan_still_counts_a_genuine_sub_0_3pp_gap_as_knife_edge(monkeypatch):
    """The other half: the rounding fix must not turn the knife-edge count into a constant zero.
    Tied models split 6.5/6.5 vs 6.4/6.4 -- still a flip (6.5 clears the floor, 6.4 does not), and
    a genuine 0.1pp spread, which MUST be counted as knife-edge."""
    a = _scan_full(monkeypatch, [6.5, 6.5, 6.4, 6.4])
    assert a["tot_flip"] == 1
    assert a["flip_spreads"] == [("ZZ", [0.1])]
    assert a["tot_knife"] == 1, (
        "a genuine 0.1pp gap at a flip month is no longer counted as knife-edge -- the knife-edge "
        "rule has become vacuously false rather than correctly rounded"
    )


def test_scan_source_contains_the_rounded_spread_fix():
    """Belt-and-braces on top of the two behavioural tests above: pin the exact expression, so a
    rewrite that happens to preserve behaviour on this fixture but loses the rounding is still
    visible in review."""
    src = inspect.getsource(SED.scan)
    assert "np.round(A.max(axis=1) - A.min(axis=1), 1)" in src, (
        "scan()'s spread computation no longer rounds to 1 decimal before the knife-edge "
        "comparison -- this reopens the float-subtraction misclassification bug"
    )


# --------------------------------- Property 2: the disclosed tie-break ---------------------------------

def test_scan_records_an_exact_net_on_tie_instead_of_silently_breaking_it(monkeypatch):
    """THE behavioural test for Property 2, through the real SED.scan(). The 6.6/6.6/6.3/6.3 split
    puts the first two models ON and the last two OFF at the single flip month, so net_on is an
    EXACT two-way tie at +1. scan() must disclose it: the state goes into `trigger_tied_states`
    with both winners and the full net_on map, and must NOT appear in `trigger_counts_clean`."""
    a = _scan_full(monkeypatch, [6.6, 6.6, 6.3, 6.3])
    assert len(a["trigger_tied_states"]) == 1, (
        f"the constructed exact tie was not disclosed: {a['trigger_tied_states']}"
    )
    rec = a["trigger_tied_states"][0]
    assert rec["state"] == "ZZ"
    assert rec["tied_models"] == ["snaive", "ar1"], f"unexpected winner set {rec['tied_models']}"
    assert rec["net_on"] == {"snaive": 1, "ar1": 1, "ets": -1, "theta": -1}
    assert a["trigger_counts_clean"] == {}, (
        "a tied state must not contribute to trigger_counts_clean -- that is the entire point of "
        "the clean/raw split"
    )


def test_scan_winners_first_reproduces_old_max_selection_bit_for_bit(monkeypatch):
    """The compatibility claim, now checked against scan()'s OWN output rather than against a
    hand-written dict (the previous version of this test never referenced SED at all -- it asserted
    a property of CPython's `max()`). `trigger_counts` (the raw headline counts that feed
    state_eb_trigger_model.csv) must still credit exactly the model the old
    `max(net_on, key=net_on.get)` would have picked from the very net_on map scan() recorded."""
    a = _scan_full(monkeypatch, [6.6, 6.6, 6.3, 6.3])
    net_on = a["trigger_tied_states"][0]["net_on"]
    old_choice = max(net_on, key=net_on.get)          # the pre-fix selection, on scan()'s own data
    assert a["trigger_counts"] == {old_choice: 1}, (
        f"trigger_counts {a['trigger_counts']} no longer matches the old max()-based selection "
        f"{old_choice!r} on scan()'s own recorded net_on {net_on!r} -- the disclosure fix was "
        f"supposed to change reporting, not the headline eb.trigger_* counts"
    )
    assert a["trigger_tied_states"][0]["tied_models"][0] == old_choice, (
        "winners[0] must be the model max() would have returned (first maximal element in "
        "MODELS declaration order)"
    )


def test_scan_source_tie_break_is_disclosed_not_silent():
    """Structural companion to the two behavioural tie tests above: `winners` must be computed from
    ALL models achieving the max net_on, `trg` must still be `winners[0]`, and the clean-count
    increment must be gated on a UNIQUE winner."""
    src = inspect.getsource(SED.scan)
    assert "winners = [m for m in net_on if net_on[m] == _max_net]" in src
    assert "trg = winners[0]" in src
    assert "trigger_tied_states" in src
    assert "trigger_counts_clean" in src
    clean_branch = src[src.index("if len(winners) == 1:"):]
    assert "trigger_counts_clean" in clean_branch[:200]


def test_new_acc_initializes_tie_tracking_fields():
    acc = SED._new_acc()
    assert acc["trigger_counts_clean"] == {}
    assert acc["trigger_tied_states"] == []
