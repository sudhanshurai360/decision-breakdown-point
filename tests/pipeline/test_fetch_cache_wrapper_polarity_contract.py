"""fetch_alfred.py::load_cached_or_fetch() and breadth_fetch.py::load_or_fetch() are structurally
near-identical cache-or-fetch wrappers (same parquet-read + datetime-coercion body), but their
shared boolean return value means the OPPOSITE thing in each file:
  - fetch_alfred.load_cached_or_fetch()  -> (df, fetched):  True  on a FRESH fetch, False on cache hit
  - breadth_fetch.load_or_fetch()        -> (df, reused):   True  on a CACHE HIT,   False on fresh fetch
Each file is internally self-consistent today and no live bug exists, but a future "obvious" merge
of these two functions, done without noticing the polarity flip, would silently invert one
caller's fresh-fetch/cache-hit logic. This test pins each function's documented contract so that
change would fail loudly here first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import pandas as pd
import fetch_alfred as FA
import breadth_fetch as BF


def _fake_df():
    return pd.DataFrame({
        "series_id": ["X"], "date": pd.to_datetime(["2020-01-01"]),
        "realtime_start": pd.to_datetime(["2020-01-02"]), "value": [1.0],
    })


def _write_cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _fake_df().to_parquet(path)


def test_fetch_alfred_reports_fresh_fetch_as_true(tmp_path):
    out = tmp_path / "not_cached.parquet"
    calls = []

    def fetch_fn():
        calls.append(1)
        return _fake_df()

    df, fetched = FA.load_cached_or_fetch("X", out, fetch_fn, force=False)
    assert fetched is True, "fetch_alfred.load_cached_or_fetch must report True on a FRESH fetch"
    assert calls == [1]


def test_fetch_alfred_reports_cache_hit_as_false(tmp_path):
    out = tmp_path / "cached.parquet"
    _write_cache(out)
    calls = []

    def fetch_fn():
        calls.append(1)
        return _fake_df()

    df, fetched = FA.load_cached_or_fetch("X", out, fetch_fn, force=False)
    assert fetched is False, "fetch_alfred.load_cached_or_fetch must report False on a CACHE HIT"
    assert calls == [], "fetch_fn must not be called on a cache hit"


def test_breadth_fetch_reports_fresh_fetch_as_false(tmp_path):
    out = tmp_path / "not_cached.parquet"
    calls = []

    def fetch_fn():
        calls.append(1)
        return _fake_df()

    df, reused = BF.load_or_fetch("X", out, fetch_fn)
    assert reused is False, "breadth_fetch.load_or_fetch must report False on a FRESH fetch"
    assert calls == [1]


def test_breadth_fetch_reports_cache_hit_as_true(tmp_path):
    out = tmp_path / "cached.parquet"
    _write_cache(out)
    calls = []

    def fetch_fn():
        calls.append(1)
        return _fake_df()

    df, reused = BF.load_or_fetch("X", out, fetch_fn)
    assert reused is True, "breadth_fetch.load_or_fetch must report True on a CACHE HIT"
    assert calls == [], "fetch_fn must not be called on a cache hit"


def test_the_two_functions_disagree_on_polarity_for_the_same_scenario(tmp_path):
    """The load-bearing check: given the IDENTICAL scenario (a fresh fetch, no cache present),
    the two functions' boolean flags must currently be OPPOSITE of each other. If this ever starts
    passing with `==` instead of `!=`, someone already "fixed" the polarity mismatch without
    updating both call sites, which is exactly the silent-inversion risk this cluster exists to
    catch -- that would need an explicit, deliberate change to this test, not a silent pass."""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    _, fetched = FA.load_cached_or_fetch("X", out_a, _fake_df, force=False)
    _, reused = BF.load_or_fetch("X", out_b, _fake_df)
    assert fetched != reused, (
        "fetch_alfred's `fetched` and breadth_fetch's `reused` are documented to have opposite "
        "polarity for the same fresh-fetch scenario -- if they now agree, the two functions' "
        "contracts have converged and every caller of either needs re-auditing before any merge."
    )
