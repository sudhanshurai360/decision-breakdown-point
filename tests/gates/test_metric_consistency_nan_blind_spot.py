"""Regression test: metric_consistency.py's
check_real_panel() computed `diff = (merged[ref] - merged[other]).abs()` and flagged a mismatch
only when `diff > 1e-12` -- but NaN comparisons are always False in pandas/numpy, so if the two
producers' own aggregators ever disagreed on how to handle a missing/NaN cell in a (series, model,
year) group (one skipping NaN, one not), the resulting NaN-vs-real-number disagreement was
completely invisible to this gate. Currently latent against the real frozen loss matrix (verified:
zero NaNs in its sae/msse columns today), but exactly the blind spot the gate exists to close --
"the same metric NAME must mean the same COMPUTATION in every producer", per metric_consistency.py's
own module docstring.

Fixed by also flagging wherever exactly one side is NaN and the other is not:
`(diff > 1e-12) | (merged[ref].isna() != merged[other].isna())`.
"""
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

import sys
sys.path.insert(0, str(ROOT / "code" / "gates"))
import metric_consistency as MC  # noqa: E402


def _write_panel(path):
    # One (series, model, year) GROUP with two rows: [NaN, 5.0]. An aggregator that skips NaN
    # (pandas' own default .mean()) returns 5.0 from this group; an aggregator that does NOT skip
    # NaN (.mean(skipna=False)) returns NaN from the identical group -- a genuine disagreement, not
    # a construction artifact.
    rows = [
        {"series": "S", "model": "M", "year": 2020, "col": float("nan")},
        {"series": "S", "model": "M", "year": 2020, "col": 5.0},
        # a second, ordinary group with no NaN at all, both aggregators must still agree here --
        # confirms the fix doesn't over-flag ordinary data.
        {"series": "S", "model": "M", "year": 2021, "col": 3.0},
        {"series": "S", "model": "M", "year": 2021, "col": 3.0},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_nan_vs_real_number_disagreement_is_caught(monkeypatch, tmp_path):
    """THE CONFIRMED GAP CLOSED: before this fix, a NaN-skipping aggregator vs a NaN-propagating
    aggregator on the identical group silently passed (diff was NaN, NaN > 1e-12 is False)."""
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    monkeypatch.setattr(MC, "PRED", panel)

    metrics = {
        "skips_nan": {"m": ("col", lambda s: s.mean())},
        "keeps_nan": {"m": ("col", lambda s: s.mean(skipna=False))},
    }
    bad = MC.check_real_panel(metrics)
    assert len(bad) == 1, f"expected exactly one flagged metric mismatch, got: {bad}"
    assert "differ on 1 of 2" in bad[0], (
        f"expected exactly the one NaN-mismatched group (of 2 total groups) to be flagged, "
        f"the ordinary 2021 group must NOT be flagged: {bad[0]}"
    )


def test_agreeing_aggregators_including_on_nan_are_not_flagged(monkeypatch, tmp_path):
    """Sanity: two aggregators that agree on EVERY group, including producing the identical NaN on
    the NaN group, must not be flagged -- the fix must not over-trigger on legitimate agreement."""
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    monkeypatch.setattr(MC, "PRED", panel)

    metrics = {
        "a": {"m": ("col", lambda s: s.mean(skipna=False))},
        "b": {"m": ("col", lambda s: s.mean(skipna=False))},
    }
    bad = MC.check_real_panel(metrics)
    assert bad == []


def test_no_nans_in_the_real_frozen_panel_today():
    """Confirms the fix is genuinely latent against the real data, mirroring this project's own
    established 'latent, not active' verification style for this class of finding."""
    if not MC.PRED.exists():
        pytest.skip("frozen loss matrix not available")
    d = pd.read_csv(MC.PRED)
    for col in ("sae", "msse"):
        if col in d.columns:
            assert d[col].isna().sum() == 0, (
                f"{col} now has NaN values in the real frozen panel -- the NaN-mismatch fix is no "
                f"longer purely latent, verify metric_consistency.py's output before trusting it"
            )
