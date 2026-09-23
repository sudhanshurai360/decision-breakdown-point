"""regime_robustness.py integration tests, run against the REAL curated data (not synthetic
fixtures) -- this is the one piece of this file that genuinely needs real data: its own internal
cross-check (main()'s `assert (_cmp == _headline).all()`) exists specifically to catch a regression
in the reconstructed mcs_size()/per_series_panels() plumbing by comparing against mcs_test.csv's
independently-computed MCS_abs_10 column. A synthetic panel can't exercise that comparison.

The three data-dependent tests require multiseries/results/breadth_distinct/
breadth_distinct_predictions.csv and multiseries/results/mcs_test.csv (both real, committed pipeline
outputs) to be present -- each is individually skipped if either is missing (e.g. a partial
checkout) rather than failing. The skip is applied per-test, NOT module-wide, so the two
placebo-disclosure tests at the bottom -- which read regime_robustness.py's own source and need no
data files at all -- keep running on a clone without multiseries/results/.
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mcs_test as MT
import regime_robustness as RR

needs_real_data = pytest.mark.skipif(
    not (Path(MT.PRED).exists() and Path(MT.OUT).exists()),
    reason="real curated-panel data files not present in this checkout",
)


@needs_real_data
def test_per_series_panels_covers_the_curated_twelve():
    # The curated panel is 12 series: the screened universe has 13 candidates and TTLCONS is the one
    # that failed the pre-committed validity screen (S3.1; canonical data.curated_candidates=13,
    # data.curated_dropped="TTLCONS"). UMCSENT is NOT one of the 13 -- it was excluded on licensing
    # before the screen was applied and never entered the manifest.
    panels = RR.per_series_panels()
    assert len(panels) == 12


@needs_real_data
def test_mcs_size_full_sample_matches_mcs_test_csv_headline():
    # this is exactly main()'s own internal assertion, run here as a standing regression test rather
    # than only as a side effect of running the whole script -- same full curated panel, same
    # estimator (mcs_test.mcs at alpha=0.10, seed=0), so MCS_full must equal MCS_abs_10 for every
    # series or the reconstructed per_series_panels()/mcs_size() plumbing has diverged from the
    # paper's headline estimator.
    panels = RR.per_series_panels()
    headline = pd.read_csv(MT.OUT).set_index("series")["MCS_abs_10"]
    for sid, (years, L) in panels.items():
        T = len(years)
        full_mask = np.ones(T, bool)
        mcs_full = RR.mcs_size(L, full_mask)
        assert mcs_full == headline[sid], (
            f"{sid}: regime_robustness MCS_full={mcs_full} != mcs_test.csv MCS_abs_10={headline[sid]}")


@needs_real_data
def test_covid_mask_selects_exactly_2020_and_2021():
    panels = RR.per_series_panels()
    for sid, (years, _L) in panels.items():
        years = np.asarray(years)
        covid_mask = np.isin(years, RR.COVID)
        assert covid_mask.sum() == 2, f"{sid}: expected exactly 2 COVID years present, got {covid_mask.sum()}"
        assert set(years[covid_mask]) == {2020, 2021}


# --- the undersized-placebo-pool disclosure ------------------------------------------------------
# main() ends with a hard assertion cross-checking MCS_full against the REAL mcs_test.csv for the
# real 12-series curated panel, so a synthetic short-panel substitution cannot reach this branch
# through main() without also faking that whole comparison. Instead the branch's OWN source is
# lifted out of regime_robustness.py and executed against a synthetic year vector, so the text and
# the condition under test are the shipped ones rather than a copy that can drift.

def _disclosure_src() -> str:
    lines = Path(RR.__file__).read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines)
              if ln.strip() == "eligible = np.flatnonzero(~covid_mask)"]
    assert len(starts) == 1, (
        f"expected exactly one eligible-pool block in regime_robustness.py, found {len(starts)}"
    )
    end = next(i for i in range(starts[0], len(lines)) if lines[i].rstrip().endswith("flush=True)"))
    src = textwrap.dedent("\n".join(lines[starts[0]:end + 1]))
    assert "[warn]" in src, (
        "the lifted eligible-pool block does not contain the [warn] disclosure it is supposed to "
        f"carry -- the source extraction anchors have drifted from regime_robustness.py:\n{src}"
    )
    return src


def _placebo_disclosure(years, covid_years, capsys):
    years = np.asarray(years)
    covid_mask = np.isin(years, covid_years)
    ns = {"np": np, "covid_mask": covid_mask, "n_drop": int(covid_mask.sum()), "sid": "TESTSID"}
    exec(_disclosure_src(), ns)
    return capsys.readouterr().out


def test_undersized_eligible_pool_prints_a_warn_disclosure(capsys):
    """A direct repro of the undersized-pool case: T=3, 2 of 3 years are COVID, leaving only 1 eligible non-COVID year
    -- fewer than n_drop=2. Must print a [warn], not silently draw an undersized placebo."""
    out = _placebo_disclosure([2019, 2020, 2021], (2020, 2021), capsys)
    assert "[warn] TESTSID: only 1 non-COVID year(s) available, fewer than the 2 COVID year(s)" in out


def test_sufficient_eligible_pool_prints_nothing(capsys):
    """Negative control, matching real current data (every curated series has >=23 eligible years
    against n_drop=2): no warning when the pool is large enough."""
    years = list(range(1990, 2020)) + [2020, 2021]   # 30 non-COVID years, 2 COVID years
    assert _placebo_disclosure(years, (2020, 2021), capsys) == ""
