"""Raw ALFRED pulls are IMMUTABLE once written. An unconditional
`df.to_parquet(out, index=False)` in fetch_alfred.py's main(), with no `out.exists()` check, would
mean that simply re-running the script (an argv typo re-triggering the default full series list, or
just running it again) silently refetches and overwrites every already-frozen parquet with whatever
ALFRED returns today -- which need not match the original pull if FRED has since revised its own
vintage history. That failure mode hit fetch_state_ur_vintages.py's own final output for real: a
re-fetch there moved every vintage value away from the recorded ones, 0/6 matching (see
test_fetch_state_ur_vintages_immutability.py).

The guard is that sibling file's `exists() and not force` / `--force` convention, in the shared
load_cached_or_fetch() helper.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import fetch_alfred as FA  # noqa: E402


def _sentinel_df(tag):
    return pd.DataFrame({
        "series_id": ["X"], "date": pd.to_datetime(["2020-01-01"]),
        "realtime_start": pd.to_datetime(["2020-01-02"]), "value": [float(tag)],
    })


def test_existing_file_is_not_refetched(tmp_path):
    """THE CONFIRMED REGRESSION: a cached file must be reused, not overwritten by a fresh fetch,
    when --force is not passed."""
    out = tmp_path / "X_alfred_vintages.parquet"
    cached = _sentinel_df(1)
    cached.to_parquet(out, index=False)
    mtime_before = out.stat().st_mtime_ns

    def _fetch_should_not_be_called():
        raise AssertionError("fetch_fn must not be called when the cached file exists and force=False")

    df, fetched = FA.load_cached_or_fetch("X", out, _fetch_should_not_be_called, force=False)
    assert fetched is False
    assert out.stat().st_mtime_ns == mtime_before, "the cached file must not be rewritten"
    assert df["value"].iloc[0] == 1.0


def test_force_flag_overwrites_deliberately(tmp_path):
    """Positive control: --force must still allow a deliberate re-fetch, i.e. fetch_fn() IS called
    and the caller is told to (re)write the file."""
    out = tmp_path / "X_alfred_vintages.parquet"
    _sentinel_df(1).to_parquet(out, index=False)
    fresh = _sentinel_df(2)

    df, fetched = FA.load_cached_or_fetch("X", out, lambda: fresh, force=True)
    assert fetched is True
    assert df is fresh


def test_missing_file_always_fetches(tmp_path):
    """Negative control: when no cached file exists, fetch_fn() must be called regardless of force."""
    out = tmp_path / "MISSING_alfred_vintages.parquet"
    fresh = _sentinel_df(3)
    called = []

    def _fetch():
        called.append(True)
        return fresh

    df, fetched = FA.load_cached_or_fetch("MISSING", out, _fetch, force=False)
    assert fetched is True
    assert called == [True]
    assert df is fresh
