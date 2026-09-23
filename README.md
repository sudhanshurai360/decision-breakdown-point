# decision-breakdown-point

Code, tests and frozen inputs behind two working papers:

> Rai, S. (2026). *When the Best Forecasting Model Is a Tie: Pricing Model Choice Against Data
> Revision in the Extended Benefits Trigger.* SSRN Working Paper 7503218.
> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7503218

> Rai, S. (2026). *One Task Flips the Top of a 4,497-Model Leaderboard: A Companion Note on the
> Decision Breakdown Point.* SSRN Working Paper 7503319.
> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7503319

Both are also in this repository as posted, at `paper/draft_ssrn.md` (paper) and
`paper/note_ssrn.md` (note); the PDFs actually uploaded to SSRN are linked above.

**Note on data availability (2026-09-23).** Section 12 of the paper, and Section 6 of the note,
state that the pipeline, the frozen inputs of Section 3.6, the per-year loss arrays behind Table 1
and `paper/external_facts.json` were not yet publicly deposited and were available from the author
on request. That was accurate when the manuscripts were posted to SSRN on 2026-09-22; it is
superseded by this repository, which contains all four of those artifacts. A revised version citing
this repository will be posted to SSRN. The manuscript files here carry the posted text (see `FREEZE.md` for the five
typographic lines that differ), so that statement still stands in them.

**Note on one Section 7 comparison (2026-09-23).** The paper's Section 7 describes Chodorow-Reich
& Karabarbounis's (2016) 0.34pp real-time state-UR error SD as being "for the smoothed
3-month-average basis", which makes this paper's own 0.45pp monthly figure the expected larger
number. This repository's machine record does not support that reconciliation: the source defines
its figure only as the difference between the real-time and the revised unemployment rate and
states no averaging or smoothing basis, so the two bases are **not established to be the same**
(see `vintage.revision_sd` in `paper/canonical.json` and
`chodorow_reich_karabarbounis_2016.tur_realtime_error_sd_0_34pp` in `paper/external_facts.json`).
The comparison is indicative of magnitude only; nothing computed in Section 7 depends on it. The
manuscript files here carry the posted text; the sentence will be corrected in the revised
version. See `FREEZE.md`, "Known limitations of the frozen manuscripts".

## What this is

Two forecasting models can score identically on a held-out evaluation and still be treated, in
practice, as if one clearly won. The paper asks how often that actually happens, on real
macroeconomic series, and what it costs when the winner is used for something with consequences:
projecting a state's real-world eligibility trigger for federal Extended Unemployment Benefits.

The core idea is the **decision breakdown point**, *k\**: the fewest observations (years, in most
of the paper) whose removal would flip which model wins. A large *k\** means the ranking is robust;
a small one means it is fragile, sensitive to which few observations happened to be in the sample.
The paper computes *k\** — and its calibrated companion, `winner_stability` — across a curated panel
of macro series, a population of 74 FRED-MD series, the 100,000-series M4 competition, an
election-theoretic reframing of "winner," and a real statutory rule (the state Extended Benefits
trigger). The companion note applies the identical instrument to the Hugging Face Open LLM
Leaderboard: a single evaluation task swap flips the reported #1 model.

The statistical instrument itself (*k\**, `winner_stability`, the Model Confidence Set glue code) is
maintained as its own versioned package, **[`selection-fragility`](https://github.com/sudhanshurai360/selection-fragility)**
(PyPI: `pip install selection-fragility`). **This repository is everything around it**: the ~40
analysis scripts that pull real data, run the instrument on it, and produce every table, figure and
number the two papers cite — plus a verification layer that checks the manuscripts' prose against
that machine-computed record, so a stale or hand-typed number cannot silently drift from the code
that was supposed to produce it.

## Repository layout

```
code/
  pipeline/      40 files: 30 producer scripts (one per analysis, each registering results via
                 canonical_emit) plus shared data/model/figure utilities (data fetchers,
                 walk-forward scoring, make_figures.py)
  instrument/    a frozen copy of the k*/MCS machinery, contemporaneous with the papers (adds a
                 31st producer, flagship_fragility.py)
  gates/         the verification layer (see "How the numbers are checked" below)
tests/
  pipeline/      tests for code/pipeline/ (112 files), incl. mutation-verified regression tests
                 (test_r33_*.py; each docstring names the specific number-changing code mutation
                 it was verified to catch)
  gates/         tests for code/gates/ (43 files)
  instrument/    tests for code/instrument/ (4 files)
multiseries/
  data/          frozen input data: ALFRED vintages, FRED-MD, the Open LLM Leaderboard snapshot
  results/       every producer's output CSV, plus results/canonical/ (paper/canonical.json's
                 per-producer JSON fragments) and results/parts/, results/breadth_distinct/
                 (the curated-panel predictions, per series and combined)
results/         the legacy single-series ("flagship") neural-forecast scoreboard and its outputs
paper/
  draft_ssrn.md, note_ssrn.md       the two manuscripts, as posted to SSRN (PDFs are on SSRN; see
                                     the links above)
  canonical.json                    every numbered result in either manuscript, one entry per number
  external_facts.json               facts drawn from outside sources (statutes, other papers),
                                     each with its exact quote and location
  ledger_recorded.json              a tolerance ledger for acceptance.py (see below)
  checks/                           the read-only verification report behind Section 7's DOL
                                     cross-check, cited via external_facts.json
  figures/                          every rendered figure and markdown table
reproduce.py                        one command to regenerate + reverify the whole pipeline
REPRODUCE.md, FREEZE.md             how to reproduce it, and the exact state this was cut from
```

## How the numbers are checked

This is the part of the repository that is not just "the code behind a paper" — it is a small
verification system, because the project's working rule was that no number reaches the manuscript
without a machine-checkable trail back to the code that computed it.

1. Every producer script in `code/pipeline/` calls `emit()` (from `code/instrument/canonical_emit.py`)
   for each number it computes, together with a plain-English `spec` describing what the number
   means, which sentence of the manuscript it backs (`cites`), and — where practical — a
   `cross_check` expression that re-derives the number a different way from the same output CSV.
2. `code/gates/canonical.py` merges every producer's fragment into `paper/canonical.json` (447
   values as of this deposit), and **independently re-derives about half of them** straight from the
   committed result CSVs, rather than trusting the producer's own in-memory value.
3. `code/gates/claims.py` reads `paper/draft_ssrn.md` and `paper/note_ssrn.md` and cross-checks
   every number that appears in the prose against `canonical.json`: a number with no canonical
   source is an **ORPHAN**; a canonical value that doesn't appear where its own `cites` field says it
   should is **MISPLACED**.
4. `code/gates/differential_test.py` runs the frozen instrument copy in `code/instrument/` against
   the released `selection-fragility` PyPI package on thousands of synthetic panels, to confirm they
   still agree (with any deliberate divergence disclosed and named).
5. `code/gates/acceptance.py`, `code/gates/metric_consistency.py` and
   `code/gates/selftest_canonical.py` run further consistency checks — see their own docstrings.

`FREEZE.md` records the exact gate and test results this deposit was cut from.

## Paper section ↔ code map

Built directly from `paper/canonical.json`'s own `cites`/`producer` fields — this is not a
hand-maintained summary, it is what the code itself declares. To verify any specific number in the
paper: find it in `paper/draft_ssrn.md`, search `paper/canonical.json` for a `cites` entry naming
that section, read the `producer` field, then open that script in `code/pipeline/`.

| Section | Claim | Code | Output |
|---|---|---|---|
| §2 (the instrument) | *k\**, `winner_stability`, the Model Confidence Set | `code/instrument/fragility.py` (frozen copy of `selection-fragility`) | — |
| §2.2–2.3 (Proposition 2.2, worked example) | the realised-\|t\| ceiling *k\** implies | `code/pipeline/realised_t.py` | `multiseries/results/realised_t.csv` |
| §3.1 (curated panel) | which 12 series, how selected | `code/pipeline/data_config.py`, `code/pipeline/model_set_audit.py` | `multiseries/results/model_set_audit.csv` |
| §3.2 (FRED-MD population) | 74-series sweep construction | `code/pipeline/fred_md_sweep.py` | `multiseries/results/fred_md_sweep.csv` |
| §3.4 (national UNRATE, ALFRED vintages) | real-time Sahm-rule and FOMC-threshold onsets | `code/pipeline/sahm_evans_realtime.py` | `sahm_evans_summary.csv`, `sahm_evans_onsets.csv` |
| §3.6 (frozen inputs / flagship exhibit) | the single-series neural-forecast exhibit | `code/pipeline/data_config.py`, `code/instrument/flagship_fragility.py`, `code/pipeline/flagship_null_ar1.py` | `results/flagship_fragility.csv`, `results/flagship_null_ar1.csv` |
| §4 (Table 1, the headline) | per-series identification/MCS/fragility | `code/pipeline/mcs_test.py`, `code/pipeline/social_choice.py`, `code/pipeline/fred_md_sweep.py` | `multiseries/results/mcs_test.csv`, `social_choice.csv` |
| §4.1 (robustness of the spine) | COVID placebo, bootstrap tuning, circular-block, monthly resolution | `code/pipeline/regime_robustness.py`, `mcs_robustness.py`, `mcs_circular_check.py`, `model_set_audit.py`, `mcs_test.py`, `fred_md_sweep.py` | `regime_robustness.csv`, `mcs_monthly_resolution.csv`, `mcs_circular_check.csv` |
| §4.2 (is this just low power?) | discriminant-power simulation, alpha sweep | `code/pipeline/mcs_power.py`, `mcs_size.py`, `mcs_circular_check.py`, `fred_md_sweep.py` | `mcs_power.csv`, `mcs_size.csv` |
| §4.3 (conceptual explanation) | best-arm reframing, FRED-MD "law" regression | `code/pipeline/best_arm.py`, `fred_md_sweep.py` | `best_arm.csv`, `fred_md_sweep.csv` |
| §4.4 (combination benchmark) | equal-weight combination vs best single model | `code/pipeline/combination_benchmark.py` | `combination_benchmark.csv` |
| §5 (election reframe) | score-vs-plurality reversal, Condorcet, concentration | `code/pipeline/social_choice.py` | `social_choice.csv` |
| §5.0 (calibration) | permutation null, AR(1)-calibrated null | `code/pipeline/social_choice.py`, `social_choice_power.py` | `social_choice_power.csv` |
| §5.4 (AR(1) companion) | serial-dependence-adjusted null | `code/pipeline/election_null_ar1.py`, `null_sensitivity.py`, `social_choice.py` | `election_null_ar1.csv`, `null_sensitivity.csv` |
| §6 (abrupt vs. gradual shocks) | shape, not magnitude, drives dilution | `code/pipeline/shape_not_magnitude.py`, `social_choice.py` | `shape_not_magnitude.csv` |
| §7 (decision cost: the EB trigger) | the decision-loss MCS, the real statutory projection | `code/pipeline/dmcs.py`, `mcs_size.py`, `sahm_evans_realtime.py`, `state_eb_decision.py`, `state_eb_vintage.py` | `dmcs.csv`, `state_eb_decision.csv`, `state_eb_vintage.csv` |
| §9 (control: the M4 competition) | where selection *is* possible | `code/pipeline/m4_margins.py`, `m4_perseries_kstar.py`, `m4_submissions_audit.py`, `m_audit.py` | `m4_margins.csv`, `m4_perseries_kstar.csv`, `m4_submissions_leaderboard.csv`, `m_audit_Monthly.csv` |
| §13.1 (bootstrap-tuning detail) | Appendix B robustness detail | `mcs_circular_check.py`, `mcs_robustness.py` | `mcs_circular_check.csv` |
| §13.2 (model-set similarity) | Appendix B detail | `model_set_audit.py` | `model_set_audit.csv` |
| §13.3 (COVID-regime placebo detail) | Appendix B detail | `regime_robustness.py` | `regime_robustness.csv` |
| §13.4 (near-trigger screen detail) | Appendix B detail | `dmcs.py` | `dmcs.csv` |
| §13.5 (split-half persistence) | Appendix B detail | `kstar_persistence.py` | `kstar_persistence.csv` |
| §13.6 (raw per-model accuracy) | Appendix B detail | `code/pipeline/make_figures.py::table_accuracy()` | `paper/figures/tableR_accuracy.md` |

**Companion note.** Note §2 ("the finding") and note §4 (calibrating `winner_stability`) are backed
by `code/pipeline/ml_witness.py` and `code/pipeline/arena_crowding.py`, which pull and score the
Open LLM Leaderboard snapshot in `multiseries/data/open_llm.parquet`.

## Figures and tables

Every rendered artifact in `paper/figures/` is produced by one function in
`code/pipeline/make_figures.py`, reading only committed result CSVs and `canonical.json` — nothing
in a figure or table is hand-typed.

| File | Section | Built by |
|---|---|---|
| `table1_identification.md` (Table 1) | §4 | `table1()` from `mcs_test.csv` + `social_choice.csv` |
| `tableC_decision_cost_by_state.md` (Table 2; the generated file's own caption says "Table C") | §7 | `figure_c()` from `state_eb_decision.csv` |
| `tableP_permutation_null.md` (Table P) | §5.0 | `table_permnull()` |
| `tableQ_ar1null.md` (Table Q) | §5.4 | `table_ar1null()` |
| `tableR_accuracy.md` (Table R) | §3.1 / §13.6 | `table_accuracy()` |
| `figA_forest_kstar_stability.png` | §4 | `figure_a()` from `social_choice.csv` + `multiseries_fragility.csv` |
| `figB_election_reversal.png` | §5.1 | `figure_b()` from `social_choice.csv` |
| `figD_prop22_ceiling_scatter.png` | §2.3 | `figure_d()` from `realised_t.csv` |
| `figF_mcs_membership_heatmap.png` | §4.1 | `figure_f()` from the frozen predictions + `mcs_test` |
| `figH_shape_not_magnitude.png` | §6 | `figure_h()` from `breadth_distinct_predictions.csv` |
| `figI_accuracy_vs_decision.png` | §7 | `figure_i()` from `dmcs.csv` |
| `figS1_flagship_trajectory_supplementary.png` | §3.6 | `figure_e()` from `extension_scoreboard_1988-2025.csv` |
| `figNote1_llm_crowding_vs_null.png` | note §2 | `figure_note_llm_crowding()` from `canonical.json`'s `llm.*` keys |

## Reproducing the results

```bash
pip install -r requirements-lock.txt          # needs Python 3.11+ (see REPRODUCE.md)
pip install selection-fragility==1.0.3        # optional: only the differential gate/tests need it
python reproduce.py --list     # manifest only, runs nothing
python reproduce.py --fast     # fast + medium stages (skips the slow 51-state / population sweeps)
python reproduce.py            # everything; several hours
python -m pytest tests/
```

Without the second install, the 16 differential tests in
`tests/gates/test_differential_test_field_level.py` skip rather than run, and
`code/gates/differential_test.py` cannot be run at all; everything else is unaffected. See
`FREEZE.md` for the test counts that assume it.

`reproduce.py --list` prints the full pipeline (31 canonical-emitting stages plus a unit-test stage
and 4 gate stages) with its plain-English description, output
file and speed tier — that list, not this README, is the authoritative one. See `REPRODUCE.md` for
what regenerates vs. what is frozen and why, environment/determinism notes, and two platform
caveats; see `FREEZE.md` for the exact gate/test results and canonical-object hash this deposit was
cut from.

## Data sources and licensing

Macro series (`multiseries/data/raw/*_alfred_vintages.parquet`, `state_ur.parquet`,
`unrate_vintages.parquet`) are point-in-time vintages pulled from the Federal Reserve's ALFRED
archive of BLS, Census, BEA and Federal Reserve Board data. Originating-agency data are U.S.
government works (public domain); FRED/ALFRED's terms permit redistribution of the data with
attribution to FRED and the originating agency. `multiseries/data/fredmd/` is the FRED-MD series
list (McCracken, M. W., & Ng, S. (2016), "FRED-MD: A Monthly Database for Macroeconomic Research",
*Journal of Business & Economic Statistics*), with values pulled live from the FRED API, not the
McCracken-Ng CSV directly. `multiseries/data/open_llm.parquet` is a snapshot (2026-09-15) of the
Hugging Face `open-llm-leaderboard/contents` dataset; that dataset declares no license on the Hub,
so this snapshot is included with attribution rather than under a stated redistribution license —
treat it as a factual record of a retired public leaderboard's own published scores, not as
independently licensed data.

Four files under `multiseries/data/` are **not** third-party data and **are** MIT-licensed with the
rest of the repository: `multiseries/data/raw/breadth_manifest.json`,
`multiseries/data/raw/breadth_manifest_distinct.json`, `multiseries/data/raw/lineage.json` and
`multiseries/data/fredmd/fredmd_api_codes.txt`. These are this repository's own selection and
provenance logs — which series were chosen, in what order, and where each was pulled from (the last
is a list of 76 FRED series identifiers, not series values). Everything else under
`multiseries/data/` is a `*.parquet` cache of the third-party sources above. `LICENSE` states the
same split.

## License

Code, tests and this repository's own documentation are released under the MIT License (see
`LICENSE`). The data files above carry the licensing terms described in "Data sources and
licensing," not MIT.

## Citing this work

See `CITATION.cff`, or cite the SSRN papers directly (links above).
