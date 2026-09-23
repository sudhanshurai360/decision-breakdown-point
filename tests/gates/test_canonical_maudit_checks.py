"""Regression test: canonical.py's `_maudit()` helper, a sibling of `_contrast_rt_start()`,
had ZERO cross-check test anywhere -- no test
imported canonical.py or called `C._maudit()`/`C.CHECKS["maudit.*"]`. Unlike its sibling
duplicated-not-imported helpers (`_contrast_rt_start()` has a full cross-check test comparing both
implementations directly; `_fredmd_law()` has a weaker but real regression test against the stored
canonical.json value), `_maudit()` was the one helper in this family with no guard at all, despite
hand-verification confirming its 4 CHECKS entries were numerically correct at the time they were
added.

This locks in two things: (1) each of the 4 maudit.* CHECKS entries matches the live canonical.json
value at full precision (mirroring test_fredmd_law_checks.py's pattern), and (2) `_maudit()`'s own
computation genuinely agrees with an INDEPENDENTLY re-derived statistic computed directly from the
per-series CSV using a different code path than `_maudit()` itself -- not just a second call to the
same formula, which would prove nothing about correctness.
"""
import json

import pandas as pd
import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")

MAUDIT_KEYS = [
    "maudit.flip_mase_rmsse_weekly",
    "maudit.flip_mase_rmsse_monthly",
    "maudit.any_flip_weekly",
    "maudit.any_flip_monthly",
]


@pytest.mark.parametrize("key", MAUDIT_KEYS)
def test_maudit_key_registered_and_matches(key):
    assert key in C.CHECKS, f"{key} is not registered in canonical.py's CHECKS"
    doc = json.loads(C.OUT.read_text())
    assert key in doc, f"{key} is not present in the live canonical.json"
    assert C.CHECKS[key]() == doc[key]["value"], (
        f"{key}: canonical.py's independent re-derivation no longer matches the emitted value at "
        f"full precision -- either m_audit.py's computation or canonical.py's own _maudit() "
        f"re-derivation has drifted."
    )


@pytest.mark.parametrize("freq", ["Weekly", "Monthly"])
def test_maudit_flip_mase_rmsse_matches_an_independently_computed_statistic(freq):
    """Real cross-check, not a second call to the same formula: recomputes the MASE-vs-RMSSE
    winner-flip rate directly from the per-series CSV using a hand-written comparison, not
    `_maudit()`'s own `(d.win_MASE != d.win_RMSSE).mean()` one-liner (even though the underlying
    arithmetic is necessarily similar for a rate-of-disagreement statistic, this at least proves
    _maudit() reads the right file/columns and the CHECKS wiring points at the right CSV)."""
    d = pd.read_csv(C.RES / f"m_audit_{freq}.csv")
    n_total = len(d)
    n_flip = sum(1 for _, row in d.iterrows() if row["win_MASE"] != row["win_RMSSE"])
    expected = n_flip / n_total

    got = C.CHECKS[f"maudit.flip_mase_rmsse_{freq.lower()}"]()
    assert got == pytest.approx(expected), (
        f"maudit.flip_mase_rmsse_{freq.lower()}: _maudit()'s vectorized computation ({got}) "
        f"disagrees with a row-by-row independent recount ({expected}) over {n_total} series"
    )


@pytest.mark.parametrize("freq", ["Weekly", "Monthly"])
def test_maudit_any_flip_matches_an_independently_computed_statistic(freq):
    d = pd.read_csv(C.RES / f"m_audit_{freq}.csv")
    win_cols = [f"win_{m}" for m in ("MASE", "RMSSE", "sMAPE", "RMSE", "MAE")]
    n_total = len(d)
    n_flip = sum(1 for _, row in d.iterrows() if len(set(row[c] for c in win_cols)) > 1)
    expected = n_flip / n_total

    got = C.CHECKS[f"maudit.any_flip_{freq.lower()}"]()
    assert got == pytest.approx(expected), (
        f"maudit.any_flip_{freq.lower()}: _maudit()'s vectorized computation ({got}) disagrees "
        f"with a row-by-row independent recount ({expected}) over {n_total} series"
    )


def test_maudit_reads_the_true_full_population_not_a_stale_sample():
    """Locks in the other half of an earlier fix: m_audit_Monthly.csv must be the true ~48,000-series
    M4 Monthly population, not the undocumented 3,000-series sample that was committed before that
    fix. A future accidental re-run with a --sample flag regressing this would silently corrupt
    maudit.*_monthly's denominator with no error."""
    d = pd.read_csv(C.RES / "m_audit_Monthly.csv")
    assert len(d) > 40000, (
        f"m_audit_Monthly.csv has only {len(d)} rows -- expected the true M4 Monthly population "
        f"(~48,000 series), not a small sample"
    )
