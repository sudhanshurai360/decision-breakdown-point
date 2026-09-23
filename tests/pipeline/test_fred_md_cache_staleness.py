"""load_fredmd()'s cache must be invalidated against the current FREDMD_CODES list. When the cache
was returned unconditionally whenever multiseries/data/fredmd/fredmd_api.parquet existed, editing
FREDMD_CODES (e.g. removing a licensed series) had NO effect on the actual sweep population until someone
thought to manually delete the cache -- which is exactly what happened when UMCSENT was removed:
the first rerun kept scoring UMCSENT's data from a stale, pre-removal cache despite the source-list
edit having already landed.

Exercises the real load_fredmd() against a monkeypatched DATA dir and FREDMD_CODES, with fetching
faked out (no network) -- confirms a cache written for one code list is invalidated the moment
FREDMD_CODES changes, and is still reused when it hasn't.
"""
import numpy as np
import pandas as pd

import fred_md_sweep as FS


def _fake_fetch(monkeypatch, codes, calls):
    """Replaces the FRED API call path with an in-memory fetch that records how many times it ran
    and returns one clean column per code -- exercises load_fredmd()'s own logic around it (mkdir,
    parquet round-trip, fingerprint write) without hitting the network."""
    class _FakeSeries:
        def get_series(self, c):
            calls.append(c)
            idx = pd.date_range("2000-01-01", periods=250, freq="MS")
            return pd.Series(np.arange(250, dtype=float), index=idx)

    def _fake_fred_ctor(api_key):
        return _FakeSeries()

    monkeypatch.setattr(FS, "FREDMD_CODES", codes)
    monkeypatch.setitem(
        __import__("sys").modules, "fredapi",
        type("m", (), {"Fred": staticmethod(_fake_fred_ctor)})(),
    )
    fa = type("m", (), {"_fred_key": staticmethod(lambda: "unused")})()
    monkeypatch.setitem(__import__("sys").modules, "fetch_alfred", fa)


def test_cache_reused_when_codes_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "DATA", tmp_path)
    calls = []
    _fake_fetch(monkeypatch, ["AAA", "BBB"], calls)

    df1 = FS.load_fredmd()
    assert set(df1.columns) == {"AAA", "BBB"}
    assert calls == ["AAA", "BBB"], "first call must fetch (no cache yet)"

    calls.clear()
    df2 = FS.load_fredmd()
    assert calls == [], "second call with the SAME FREDMD_CODES must reuse the cache, not refetch"
    assert set(df2.columns) == {"AAA", "BBB"}


def test_cache_invalidated_when_a_code_is_removed(tmp_path, monkeypatch):
    """THE CONFIRMED REGRESSION: removing a code from FREDMD_CODES (e.g. UMCSENT) must force a
    refetch, not silently keep serving the old, now-stale column set."""
    monkeypatch.setattr(FS, "DATA", tmp_path)
    calls = []
    _fake_fetch(monkeypatch, ["AAA", "BBB", "UMCSENT"], calls)
    df1 = FS.load_fredmd()
    assert "UMCSENT" in df1.columns

    calls.clear()
    _fake_fetch(monkeypatch, ["AAA", "BBB"], calls)  # UMCSENT removed from the source list
    df2 = FS.load_fredmd()
    assert calls == ["AAA", "BBB"], "removing a code must trigger a fresh fetch, not reuse the stale cache"
    assert "UMCSENT" not in df2.columns, (
        "THE BUG: a stale cache would silently keep UMCSENT in the population after it was removed "
        "from FREDMD_CODES")


def test_cache_invalidated_when_a_code_is_added(tmp_path, monkeypatch):
    """Symmetric case: adding a code must also invalidate (not just removal)."""
    monkeypatch.setattr(FS, "DATA", tmp_path)
    calls = []
    _fake_fetch(monkeypatch, ["AAA"], calls)
    FS.load_fredmd()

    calls.clear()
    _fake_fetch(monkeypatch, ["AAA", "CCC"], calls)
    df2 = FS.load_fredmd()
    assert calls == ["AAA", "CCC"]
    assert "CCC" in df2.columns


def test_fingerprint_file_written_alongside_cache(tmp_path, monkeypatch):
    """Negative-control-adjacent structural check: confirms the fingerprint mechanism this fix
    relies on is actually present on disk (not just working by coincidence of mtime or similar)."""
    monkeypatch.setattr(FS, "DATA", tmp_path)
    calls = []
    _fake_fetch(monkeypatch, ["AAA", "BBB"], calls)
    FS.load_fredmd()
    fp_file = tmp_path / "fredmd_api_codes.txt"
    assert fp_file.exists()
    assert fp_file.read_text() == "AAA\nBBB"


def test_committed_fingerprint_file_matches_the_module_and_so_reuses_the_cache():
    """The shipped cache must actually be reusable on a clean clone.

    load_fredmd() reuses multiseries/data/fredmd/fredmd_api.parquet only when the committed
    fingerprint file equals "\\n".join(FREDMD_CODES) exactly. The comparison is byte-exact, so a
    single stray trailing newline silently disables the pin: the stage then hard-fails without a
    FRED_API_KEY, or refetches live FRED data with one -- which is precisely what pinning the cache
    exists to prevent. That happened: a cosmetic "add a trailing newline" edit broke it, and it
    stayed broken because every other test in this file monkeypatches FS.DATA to a tmp_path and so
    never reads the committed file.
    """
    committed = (FS.DATA / "fredmd_api_codes.txt").read_text()
    expected = "\n".join(FS.FREDMD_CODES)
    assert committed == expected, (
        "the committed FRED-MD fingerprint no longer matches FREDMD_CODES, so load_fredmd() will "
        "ignore the committed parquet and refetch from the FRED API.\n"
        f"  committed ends: {committed[-40:]!r}\n"
        f"  expected  ends: {expected[-40:]!r}")
