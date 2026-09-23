"""fetch_alfred.py::main()'s per-series loop must isolate a per-series fetch failure. Without a
try/except, a transient failure fetching ONE series (timeout, ConnectionError, malformed response)
propagates as a raw unhandled traceback, aborting the run for EVERY series in `wanted`, not just
the failing one. breadth_fetch.py's analogous loop wraps the identical fetch call in try/except and
continues; this loop matches it.

Fixed by wrapping the fetch call in try/except, printing a clear message, and continuing to the
next series. This test verifies isolation directly: one series' fetch raises, the other succeeds,
and main() must complete (not crash) with lineage.json reflecting only the successful series.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import fetch_alfred as FA  # noqa: E402


def _df(tag):
    return pd.DataFrame({
        "series_id": ["X"], "date": pd.to_datetime(["2020-01-01"]),
        "realtime_start": pd.to_datetime(["2020-01-02"]), "value": [float(tag)],
    })


def test_one_series_failure_does_not_abort_the_whole_run(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(FA, "RAW", raw)
    monkeypatch.setattr(FA, "SERIES", {"BAD": ("bad series", "test"), "GOOD": ("good series", "test")})
    monkeypatch.setattr(FA, "_fred_key", lambda: "fake-key-for-test")
    monkeypatch.setattr("fredapi.Fred", lambda api_key: object())
    monkeypatch.setattr(sys, "argv", ["fetch_alfred.py"])

    def _fake_load_cached_or_fetch(sid, out, fetch_fn, force):
        if sid == "BAD":
            raise TimeoutError("simulated network timeout")
        df = _df(1)
        return df, True

    monkeypatch.setattr(FA, "load_cached_or_fetch", _fake_load_cached_or_fetch)

    FA.main()  # must not raise

    printed = capsys.readouterr().out
    assert "FETCH FAILED for BAD" in printed
    assert "GOOD" in printed

    lineage = (raw / "lineage.json")
    assert lineage.exists()
    import json
    doc = json.loads(lineage.read_text())
    assert "GOOD" in doc["per_series"]
    assert "BAD" not in doc["per_series"], "a failed series must not appear in lineage.json's per_series"
