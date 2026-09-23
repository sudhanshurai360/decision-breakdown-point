"""breadth_fetch.py's honest validity filter has two ways to silently pass a series it should
check, both guarded here.

(1) The "already present" cache-reuse path must still run validity(). Marking a reused series PASS
unconditionally bypasses the filter for exactly the series most likely to be stale -- and that is
most of them: 12 of 13 CANDIDATES_DISTINCT series (PAYEMS, UNEMPLOY, HOUST, RSAFS, DGORDER, TCU,
PCE, PI, BOPGEXP, BOPGIMP, M2SL, BUSINV) are reused in the real committed manifest.

(2) validity()'s rebasing check computes `ratio = early/realized`, and `abs(nan - 1) > REBASE_TOL`
is ALWAYS False for a NaN ratio (NaN comparisons never trigger a Python `>` test) -- so a NaN probe
value would fall through to "ok" with no actual rebasing check performed, instead of being caught
by the DROP branch immediately below it.

Both are latent against the real data today (all 12 reused series independently re-check PASS; 0
real vintage files have a NaN probe value), but a future re-run with new criteria, a corrupted
cache, or a series whose probe genuinely goes missing would otherwise silently miss a real drop.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import breadth_fetch as BF  # noqa: E402


def _panel(probe_value, realized_probe_value, vintage_min="2000-01-01"):
    """A minimal 2-row-class panel: one 'early' vintage of the probe month (what a real-time user
    saw back then) and one 'later' vintage of the same month (the eventually-realized value) --
    enough for validity()'s ratio computation, regardless of what other dates exist."""
    rows = [
        {"realtime_start": vintage_min, "date": str(BF.PROBE.date()), "value": probe_value},
        {"realtime_start": "2020-01-01", "date": str(BF.PROBE.date()), "value": realized_probe_value},
    ]
    df = pd.DataFrame(rows)
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_nan_probe_ratio_is_dropped_not_passed():
    """THE CONFIRMED REGRESSION: a NaN probe value must be treated as a failed rebasing check
    (fail-safe DROP), not silently fall through to 'ok' via a comparison that's always False for NaN."""
    df = _panel(probe_value=np.nan, realized_probe_value=100.0)
    ok, reason, ratio, vmin = BF.validity(df)
    assert ok is False, f"expected DROP on a NaN probe ratio, got PASS (reason={reason!r})"
    assert np.isnan(ratio)


def test_empty_fetch_result_is_dropped_not_passed():
    """The same comparison-always-False-for-a-missing-value trap covered above for a NaN ratio also
    applies to the vintage-depth check above it in validity(): a wholly-empty fetch result (0 rows)
    makes `rs.min()` return `NaT`, and `NaT > VINT_COVER` is ALWAYS False, so a broken/empty fetch
    could silently fall through to 'ok (no probe)' instead of being rejected outright."""
    df = pd.DataFrame({"realtime_start": pd.to_datetime([]), "date": pd.to_datetime([]), "value": []})
    ok, reason, ratio, vmin = BF.validity(df)
    assert ok is False, f"expected DROP on an empty fetch result, got PASS (reason={reason!r})"
    assert pd.isna(vmin)


def test_genuine_rebase_still_dropped():
    """Positive control: an ordinary, non-NaN rebasing violation (ratio far from 1) must still be
    dropped exactly as before -- the NaN guard must not weaken the existing check."""
    df = _panel(probe_value=50.0, realized_probe_value=100.0)   # ratio = 0.5, way outside REBASE_TOL
    ok, reason, ratio, vmin = BF.validity(df)
    assert ok is False
    assert ratio == pytest.approx(0.5)


def test_missing_probe_date_passes_with_rebasing_unchecked():
    """A candidate whose date
    history doesn't reach back to PROBE (2005-01) at all -- a STRUCTURAL fact about the series, not
    a data error -- must PASS with the rebasing check explicitly skipped, not be silently rejected.
    This is deliberately asymmetric with test_nan_probe_ratio_is_dropped_not_passed above: a NaN
    ratio means the probe date exists but its value is corrupted for an otherwise-checkable series
    (fail closed); a missing probe date means the check cannot be performed at all for a structural
    reason (fail open, disclosed via the "ok (no probe)" reason string)."""
    rows = [
        {"realtime_start": "2007-01-01", "date": "2008-01-01", "value": 50.0},
        {"realtime_start": "2020-01-01", "date": "2008-01-01", "value": 51.0},
    ]
    df = pd.DataFrame(rows)
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["date"] = pd.to_datetime(df["date"])
    ok, reason, ratio, vmin = BF.validity(df)
    assert ok is True
    assert reason == "ok (no probe)"
    assert ratio is None


def test_genuine_pass_still_passes():
    """Negative control: a ratio well within tolerance must still pass."""
    df = _panel(probe_value=99.0, realized_probe_value=100.0)   # ratio = 0.99, within REBASE_TOL
    ok, reason, ratio, vmin = BF.validity(df)
    assert ok is True
    assert ratio == pytest.approx(0.99)


def test_load_or_fetch_reused_file_is_still_returned_for_validation(tmp_path):
    """THE CONFIRMED REGRESSION (structural half): load_or_fetch() must return the CACHED data (and
    reused=True) without calling the fetch function -- and, critically, the caller (main()) must
    still run validity() on it. This test locks the loading contract; the full pipeline behavior
    (validity() actually being called after) is exercised directly in main()'s source, not re-derived
    here, to avoid a second parallel implementation of the loop."""
    out = tmp_path / "FAKE_alfred_vintages.parquet"
    src = _panel(probe_value=99.0, realized_probe_value=100.0)
    src.to_parquet(out, index=False)

    def _fetch_should_not_be_called():
        raise AssertionError("fetch_fn must not be called when the cached file already exists")

    df, reused = BF.load_or_fetch("FAKE", out, _fetch_should_not_be_called)
    assert reused is True
    assert len(df) == len(src)
    # the loaded frame must be independently validity()-checkable, not just returned opaquely
    ok, reason, ratio, vmin = BF.validity(df)
    assert ok is True


def test_load_or_fetch_missing_file_calls_fetch_fn(tmp_path):
    """Negative control: when the file does NOT exist, load_or_fetch() must call fetch_fn() and
    report reused=False."""
    out = tmp_path / "MISSING_alfred_vintages.parquet"
    sentinel = _panel(probe_value=99.0, realized_probe_value=100.0)
    called = []

    def _fetch():
        called.append(True)
        return sentinel

    df, reused = BF.load_or_fetch("MISSING", out, _fetch)
    assert reused is False
    assert called == [True]
    assert df is sentinel


def test_no_real_cached_series_currently_fails_validity():
    """Confirms the guard is genuinely latent on real data: every currently-cached ('already
    present' in the real committed manifest) vintage parquet independently re-checks PASS under
    validity().

    The manifest match is a PREFIX match, deliberately. A REUSED series' reason is
    "already present, {validity_reason}" (e.g. "already present, ok"), so an exact match against
    the bare string "already present" would find NO series in a regenerated manifest and silently
    skip the check this test exists to run."""
    manifest = ROOT / "multiseries" / "data" / "raw" / "breadth_manifest_distinct.json"
    if not manifest.exists():
        pytest.skip("breadth_manifest_distinct.json not available")
    import json
    m = json.loads(manifest.read_text())
    reused_sids = [sid for sid, rec in m["series"].items()
                   if rec.get("reason", "").startswith("already present")]
    assert reused_sids, "expected at least one 'already present' series in the real manifest"
    for sid in reused_sids:
        f = ROOT / "multiseries" / "data" / "raw" / f"{sid}_alfred_vintages.parquet"
        if not f.exists():
            pytest.skip(f"{sid}'s vintage parquet not available")
        df = pd.read_parquet(f)
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["date"] = pd.to_datetime(df["date"])
        ok, reason, ratio, vmin = BF.validity(df)
        assert ok, f"{sid} was previously bypassed as 'already present' but FAILS validity(): {reason}"
