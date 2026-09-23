# FREEZE — code state behind this deposit

## What this deposit is

The code, tests and frozen inputs behind the working paper posted to SSRN (Abstract ID 7503218,
2026-09-22; companion note 7503319). This file records the state that deposit was cut from; the
release it describes is version 1.0.0.

## Determinism, measured

Two independent full `reproduce.py` runs on the same code (6.7 h and 5.5 h) produced byte-identical
values and companion fields on every shared key.

## Gate results at freeze

`canonical.py --check`, `selftest_canonical.py`, `acceptance.py` (relocated ledger,
`paper/ledger_recorded.json`), `claims.py` (repointed at `paper/draft_ssrn.md`/`note_ssrn.md`),
`differential_test.py` (against `pip install selection-fragility==1.0.3`) and
`metric_consistency.py` all PASS.

## Tests

`pytest tests/`: 1,144 passed, 10 skipped, 2 xfailed (in this public deposit, which drops the
gates/tests tied to non-shipping material — `ssrn_consistency.py`, `mechanism_workbench.py`, the
two incomplete-stub producers and their tests — relative to the private development tree).

Those counts are for an environment with **both** `pip install -r requirements-lock.txt` **and**
`pip install selection-fragility==1.0.3`. The second package is deliberately outside the lock file
(it is the released instrument the differential gate checks the frozen `code/instrument/` copy
against), and the README quickstart does not install it. Without it, the 16 differential tests in
`tests/gates/test_differential_test_field_level.py` skip instead of running: that many fewer passes
and that many more skips, with the xfail count unchanged. A run that reports 16 extra skips is
therefore a missing optional package, not a regression.

## Canonical object

`paper/canonical.json` v203, 447 values, `_meta.hash`
`854e90f850e6b23d051f7282eb15a37a7f4cfc9f6adc62a668e00a7b22cf5f73`.

The digest excludes the prose-only fields (`producer`, `spec`, `cites`, `cross_check`,
`provenance`), so text edits to those do not move it. It was re-stamped once, after
`provenance` was added to that exclusion set: `mcs_size.py` passes nine `provenance=` strings
through `emit()`, and until then they were being hashed as if they were data and perturbed by
the independence probe. Adding them to the exclusion set changed the digest from
`8a64b080f063a595a97b119a5d384072675fbb8c6fe72125d0d1e2fb20d88682` without changing any value:
comparing every key's non-prose fields before and after returns an empty diff, and the count
stays at 447. `VALUES_KEYS_HASH` and `CHECKED_KEYS_HASH` are unchanged, since the key set is
identical.

## Known limitations of the gates

- `claims.py` is a context-free number matcher: it has near-0% single-digit detection power and no
  right-number-wrong-sentence check.
- 15 `MISPLACED` entries are accepted deliberately: booleans the prose states in words, values
  quoted at a different rounding, and superseded arms (e.g. `mcssize.arch_gap_moving_block`) no
  longer quoted directly.
- Two tests are strict `xfail`s recording that `claims.tokenise()` does not treat a U+2212 typographic
  minus sign as a sign; they flip to XPASS if that is fixed.

## Known limitations of the frozen manuscripts

The manuscript files in `paper/` carry the text posted to SSRN on 2026-09-22. `note_ssrn.md` is
byte-identical to it. `draft_ssrn.md` differs on exactly five lines, all typographic and none
touching a number or a claim: a doubled comma (`2%,,` to `2%,`), two HTML entities left unescaped
in the markdown (`&lt;` to `<`), and two curly apostrophes made straight. A `git diff` against this
repository's first commit shows those five lines and nothing else.

One statement in the manuscripts is narrower than what this repository can support:

- **Section 7, the Chodorow-Reich & Karabarbounis (2016) 0.34pp comparison.** `draft_ssrn.md`
  describes the 0.34pp as being "for the smoothed 3-month-average basis", against which this
  paper's own 0.45pp monthly figure is then the expected larger number. The machine record
  disagrees with that reconciliation: `paper/canonical.json`'s `vintage.revision_sd` spec, and
  `paper/external_facts.json`'s
  `chodorow_reich_karabarbounis_2016.tur_realtime_error_sd_0_34pp`, record that the source defines
  its figure only as the difference between the real-time and the revised unemployment rate and
  states no averaging or smoothing basis — so the two are **not established to be on the same
  basis**, and the comparison is indicative of magnitude only. Nothing else in Section 7 depends on
  the reconciliation: the 0.45pp figure, its denominator and the revision-vs-model-choice
  comparison are all computed on this paper's own panel. The manuscript sentence will be corrected
  in the revised version posted to SSRN.

## The one post-freeze change that touched shipped numbers

`state_eb_vintage.py`: the reported arms (revised, as-of, one-leg) start at `panel_start`
(2007-06-01, the first month with an initial release for every state) rather than `panel_start` +
26 months (2009-08-01). The 26-month look-back is needed only by a disclosed, superseded each-leg
construct, which is now reported on its own determined months. The excluded 26 months are the
2008-09 trigger-on wave. Effect: model-choice episodes 7/9 → 24/26; revision changes 61/203 of
9,950 → 84/271 of 11,250; the Section 7 multipliers 7-9× → 3-4× like-for-like and 23-29× → 10-11×
every-leg; revision SD 0.449 → 0.451 over 11,350; states divergent 6/9 → 22/22 of 50. Prose requoted
at 21 sites in both documents; the old window and its multipliers are disclosed in Section 7 as an
earlier version of the check. A companion field also counts the model and revision channels inside
the 2007-06..2009-07 wave itself: `vintage.wave_realized_trigger_changed_one_leg` = 23 of 1,300
(nine-leg 68; model episodes 17 on both arms; like-for-like multiple 1.4×).

`mcs_size.py` separately gained two companion emits that did not move any pre-existing value: the
first-step size measured on exactly-null panels carrying the panel's own pooled lag-1 autocorrelation
(`mcssize.size_ar1_at_nominal_10` = 0.367, i.i.d. companion 0.265) and at Section 7's state
dimensions (`mcssize.size_state_dims` = 0.179, AR(1) companion 0.307).

## Text-only post-freeze edits

Docstrings, spec strings, figure titles and captions were edited after the freeze with every
emitted value re-verified byte-identical. A second pass, after an independent review of this
deposit, removed internal code-review process labels (dated review-round and reviewer identifiers)
from comments and docstrings across the codebase and test suite; no code logic, test assertion, or
emitted value was touched by that pass — confirmed by a full gate-stack and `pytest tests/` re-run
(1,144 passed, 10 skipped, 2 xfailed) afterward.
