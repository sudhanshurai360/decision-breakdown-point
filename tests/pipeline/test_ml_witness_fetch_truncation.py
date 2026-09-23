"""Regression test: ml_witness.py::fetch()'s paginated HF datasets-server loop used to treat ANY non-429 exception
(timeout, ConnectionError, a 5xx HTTPError, malformed JSON) identically to a genuine "no more rows"
completion -- both hit `break` and the collected rows were written to CACHE (open_llm.parquet) as if
the fetch were complete, with no comparison against the API's own `num_rows_total` and no flag
distinguishing a clean stop from an aborted one. Because `CACHE.exists()` short-circuits the whole
function with no `--force`/refresh mechanism, a single mid-run transient failure would permanently
freeze a truncated leaderboard on disk, undetectable later (every downstream `llm.*` canonical value
is documented `cross_check=None`, i.e. never independently re-derived).

Fixed: a `completed` flag now tracks whether the loop actually reached a genuine exhaustion
condition; any other exception is retried with the same bounded backoff as a 429 instead of being
treated as a clean stop; and if the loop still isn't `completed` when it exits, fetch() raises
instead of silently writing CACHE.
"""
import json
import time

import pytest

import ml_witness as MLW


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_persistent_transient_error_refuses_to_write_a_truncated_cache(monkeypatch, tmp_path):
    """THE CORE REGRESSION: a non-429 exception that never recovers (6 attempts, all fail) must
    raise, not silently cache whatever partial rows were collected before it."""
    cache = tmp_path / "open_llm.parquet"
    monkeypatch.setattr(MLW, "CACHE", cache)
    monkeypatch.setattr(time, "sleep", lambda *_: None)  # skip real backoff delays in the test

    def _always_timeout(*a, **kw):
        raise TimeoutError("simulated network timeout")

    monkeypatch.setattr(MLW.urllib.request, "urlopen", _always_timeout)

    with pytest.raises(RuntimeError, match="aborted before exhausting"):
        MLW.fetch(nmax=200)
    assert not cache.exists(), "a truncated fetch must never be written to CACHE"


def test_a_transient_error_that_recovers_still_completes_normally(monkeypatch, tmp_path):
    """Companion sanity check: a transient error on the FIRST attempt that succeeds on retry must
    not block a normal, complete fetch -- the fix must not become over-eager."""
    cache = tmp_path / "open_llm.parquet"
    monkeypatch.setattr(MLW, "CACHE", cache)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    calls = {"n": 0}

    def _flaky_then_ok(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated transient failure")
        # second call: return a short, complete page (fewer than 100 rows -> natural completion)
        payload = json.dumps({
            "num_rows_total": 3,
            "rows": [{"row": {"fullname": f"model_{i}", **{t: 0.5 for t in MLW.TASKS}}} for i in range(3)],
        }).encode()
        return _FakeResponse(payload)

    monkeypatch.setattr(MLW.urllib.request, "urlopen", _flaky_then_ok)

    df = MLW.fetch(nmax=200)
    assert len(df) == 3
    assert cache.exists(), "a fetch that eventually completes must still write CACHE"


def test_a_genuine_clean_completion_still_writes_cache_as_before(monkeypatch, tmp_path):
    """Sanity check on the `completed` flag itself: a normal single-page fetch (no errors at all)
    must behave exactly as before this fix."""
    cache = tmp_path / "open_llm.parquet"
    monkeypatch.setattr(MLW, "CACHE", cache)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def _clean_page(req, timeout=None):
        payload = json.dumps({
            "num_rows_total": 2,
            "rows": [{"row": {"fullname": f"model_{i}", **{t: 0.5 for t in MLW.TASKS}}} for i in range(2)],
        }).encode()
        return _FakeResponse(payload)

    monkeypatch.setattr(MLW.urllib.request, "urlopen", _clean_page)

    df = MLW.fetch(nmax=200)
    assert len(df) == 2
    assert cache.exists()
