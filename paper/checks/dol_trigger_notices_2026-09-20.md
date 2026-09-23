# EB trigger code vs. DOL's published trigger notices (2026-09-20)

Scope: `code/pipeline/decision_protocol.py`, `eb_on()` in `code/pipeline/state_eb_decision.py`; read-only. External source: 67 weekly ETA Trigger Notices spanning Jan-2009 to Dec-2013, sampled at roughly 4-week intervals (about 13 per year; a strict 4-week cadence across those five years would give about 65, so the sample carries a few extra notices — the exact notice dates are in the parsed rows, which are not committed with this report), from `oui.doleta.gov/unemploy/trigger/<yyyy>/trig_<mmddyy>.html`; 67 × 51 = 3,417 state-week rows, 51 jurisdictions. Regulation: 20 CFR 615.12 via the eCFR API.

## 1. Rule the code implements

Current leg at month t: `(fc[t] + UR[t-1] + UR[t-2]) / 3` ("published": two published months plus one model month; "all_forecast": rolling mean of fc), rounded half-away-from-zero to tenths ((e)(3)). Reference legs: rolling 3-month mean of realized UR, rounded to tenths, shifted 12 and 24 calendar months. Look-back: `round(round(a/ref, 2)*100, 0) >= 110` for either year ((e)(1)(ii)/(e)(4)). Floor `a >= 6.5`. ON = floor AND (y1 OR y2). Applied to all 51 jurisdictions, 2-year look-back only; IUR trigger ignored.

## 2. Design

Each notice states its TUR-end month (notice effective 2010-04-11 uses the average ending Feb-2010) and reports per state the 3-month SA TUR, "% of prior year", "% of second year" (third-year column from 2011), flags (`@` ON by TUR, `&` no TUR option in law) and a B/E status date. Matched on (state, tur_end) with fc = realized (tests the rule, not a forecast). Two arms: (a) `multiseries/data/state_ur.parquet` (2026 vintage); (b) as-of reconstruction from `multiseries/data/raw/state_ur_alfred_vintages.parquet` (latest vintage with realtime_start <= notice date; monthly vintage coverage confirmed).

## 3. Results

| Arm | TUR exact | within 0.1pp | %-y1 exact | ON/OFF agree (2-yr arithmetic) |
|---|---|---|---|---|
| (a) revised | 10% | 30% | 9% | 3,203/3,366 (95.2%) |
| (b) as-of | 86% | 98% | 62% (71% where TUR agrees) | 3,305/3,314 (99.7%) |

All 163 arm-(a) mismatches trace to an input difference (87 floor, 72 look-back, 4 both); where all three intermediates coincide (17 rows) ON/OFF agrees 100%. DOL's `@` flags confirm inclusive thresholds and the either-year OR.

Selected rows — DOL | code as-of | code revised:

| State | TUR-end (notice) | DOL TUR/y1/y2/flag | as-of TUR/y1/y2/ON | revised TUR/y1/y2/ON | Verdict |
|---|---|---|---|---|---|
| KS | 2010-02 (04-11-10) | 6.5/116/166 `@` ON | 6.5/116/167 ON | 7.3/124/174 ON | MATCH (exact floor) |
| MN | 2010-02 | 7.3/101/155 `@` ON | 7.3/101/155 ON | 7.7/110/164 ON | MATCH (2nd-yr leg) |
| MI | 2010-02 | 14.3/126/201 ON | 14.3/127/204 ON | 13.9/126/193 ON | MATCH |
| ND | 2010-02 | 4.2/105/144 OFF | 4.2/105/145 OFF | 3.9/105/134 OFF | MATCH |
| CO | 2010-02 | 7.4/110/172 (`&`) | 7.5/112/174 ON | 8.8/133/215 ON | TUR off 0.1: 7.3/7.4/7.7 -> 7.467; DOL 7.4 (levels) |
| AZ | 2008-12 (02-01-09) | 6.5/158/166 ON-arith | 6.4/156/164 OFF | 7.3/183/178 ON | MISMATCH: 6.1/6.3/6.9 -> 6.4; DOL 6.5 (levels) |
| AR | 2009-03 (04-26-09) | 6.5/135/127 ON-arith | 6.4/133/125 OFF | 7.3/146/138 ON | MISMATCH: 6.4/6.4/6.5 -> 6.4; DOL 6.5 (levels) |
| KS | 2011-03 (04-24-11) | 6.8/94/109 (3rd yr 170) | 6.8/94/110 ON | 6.6/92/105 OFF | MISMATCH: 6.8/6.2 -> 110 vs DOL 109 (ref precision) |
| NY | 2012-07 (09-09-12) | 8.9/110/103 `@` ON | 8.9/109/103 OFF | 8.8/106/101 OFF | MISMATCH: 8.9/8.2 -> 109 vs DOL 110 (ref precision) |
| AK | 2008-12 (03-01-09) | 7.3/115/115 `@` ON | 6.8/108/108 OFF | 6.8/108/106 OFF | MISMATCH: (d)(2) freeze, DOL kept pre-benchmark 7.3 |
| UT | 2009-06 (07-19-09) | 5.4/163/207 `&` OFF | 5.4/164/208 OFF | 7.5/250/300 ON | as-of MATCH; revised 2.1pp off (vintage) |
| CA | 2011-12 (01-29-12) | 11.4/91/93/131 `@` ON | 11.4/91/93 OFF | 11.3/92/93 OFF | 3-yr look-back (P.L. 111-312) not modelled |

## 4. Diagnosis

1. **(e)(3) computes the rate from averaged levels; the code averages three tenths-rounded rates.** Signature: when the rate-mean ends in .x33, DOL is 0.1 higher in 20% of rows; .x67, DOL 0.1 lower in 17%; .x00, they differ in 3%. Net 0.1pp disagreement in ~13% of state-months, flipping exact-floor cases (AZ, AR). The look-back shows the same precision effect (+1 in 23% of rows with identical TUR; KS, NY). Rounded/rounded fits DOL best of four numerator/denominator variants: the cascade is right, input granularity is the deviation.
2. **Vintage.** (d)(2): determinations use the initial BLS release, never revised. Against the 2026 vintage, |gap| >= 0.3pp in 40-58% of 2008-13 state-months (median 0.2-0.3pp) — the size of the paper's 0.3pp knife-edge threshold. The existing `state_eb_vintage.py` arm (7 vs 9 vs 14 flip months) is therefore not optional. (Those three counts are from the then-current 2009-08-01 window start; superseded — on the shipped 2007-06-01 window the arm reports 24 revised / 26 as-of / 23 first-release-each-leg, as quoted in Section 7.)
3. **Temporary 3-year look-back (Dec-2010 to Dec-2013):** 463 of 1,801 3-yr-ON state-rows in 2011-13 were ON only via the third year. Not modelled; disclose as scoping.
4. **Law adoption:** 24-38% of state-rows (78% in 2008) carry `&`; the code triggers every state. Counterfactual framing, should be stated.

## 5. Verdict

Trigger *logic* — leg composition, calendar-correct 12/24-month references, either-year OR, inclusive 6.5/110, hundredths-then-integer look-back rounding — is VERIFIED against DOL's determinations: 99.7% ON/OFF agreement on as-of data, every disagreement attributable to inputs, none to a comparison or reference-period bug. Two input-level deviations are NOT verified and matter for a knife-edge argument: the levels-vs-rates proxy (0.1pp in ~13% of months) and revised vs initial-release data (0.2-0.3pp median). Neither is fixable from `state_ur.parquet` (needs SA unemployment and labor-force levels). For §7's 24/50 states, 29 flip months, 93% knife-edge: these count forecast-side coin-flips at a 0.1pp granularity the statutory computation itself does not reproduce; the prose should say the trigger is matched to +/-0.1pp and promote the vintage arm to headline sensitivity.


## Addendum — definition of the 3,314-row arm-(b) subset

Re-derived from the underlying comparison's merged output (not committed as a standalone script). Of the 3,417 parsed rows (67 notices × 51
jurisdictions), 103 have `asof_ok = False` and are excluded from arm (b): 102 rows are the two notices
effective 2012-11-04 and 2012-12-30, whose headers print "3-month period ending October, 2012" and
"December, 2012" respectively — months BLS had not released on those dates (the 2012-12-02 notice
correctly prints October), so no as-of vintage can reproduce them; 1 row is DC on the 2012-10-07 notice,
whose ALFRED vintage as of that date ended at 2012-07 (August not yet present). These four figures
(3,417 / 103 / 102 / 3,305) are recorded in `paper/external_facts.json` under
`dol_trigger_notice_comparison_2026_09_20` and quoted in Section 7.

Arm (a) (revised data) compares **3,366** rows, and that denominator is **not** reconciled here.
3,417 − 3,366 = 51 rows — exactly one notice's worth, at 51 jurisdictions per notice. The
as-of exclusions above do not explain it: the revised series covers the two misprinted months and
the DC row alike, so the reason stated for arm (b) would put arm (a) at 3,417 (at most one row, not
51, could plausibly fall out of it). The comparison's merged output is not committed, and neither
is the notice-parsing script, so the specific notice dropped from arm (a) and the rule that dropped
it cannot be recovered from the material in this repository. Stated plainly rather than
rationalised: **51 arm-(a) rows are unaccounted for.**

What this does and does not affect. The headline of this report is the arm-(b) figure
(3,305/3,314 = 99.7%), whose denominator is fully reconciled above and which is what Section 7 and
`external_facts.json` quote; it is untouched. The affected number is arm (a)'s 3,203/3,366 = 95.2%
in Section 3, and the 163 = 87 + 72 + 4 mismatch decomposition in Section 3, which are internally
consistent with 3,366 but would read 3,203/3,417 = 93.7% on the full parse. Arm (a) is the
deliberately mis-specified comparison (revised data against real-time determinations) whose only
role in the argument is to be much worse than arm (b); a 1.5pp shift in how much worse does not
bear on any claim made from it.
