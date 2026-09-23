"""fetch_state_ur_vintages.py's --force flag must protect the FINAL merged output (OUT -- the
frozen artifact state_eb_vintage.py consumes, backing S7's vintage.* canonical keys), not only the
intermediate per-state PARTS cache. Writing OUT unconditionally is the failure mode that already
happened once for real here: a re-fetch moved every vintage value away from the recorded ones,
"0 of 6 matching". The guard is the same exists()-and-not-force gate on the final
`out.to_parquet(OUT, ...)` write that fetch_alfred.py applies to its own parquets
(test_fetch_alfred_immutability.py).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import fetch_alfred as FA  # noqa: E402
import fetch_state_ur_vintages as FSV  # noqa: E402


def _sentinel_part(st, tag):
    return pd.DataFrame({
        "state": [st], "date": pd.to_datetime(["2020-01-01"]),
        "realtime_start": pd.to_datetime(["2020-01-02"]), "value": [float(tag)],
    })


def _wire_tmp_layout(monkeypatch, tmp_path, states):
    """Redirect every module-level path constant + states() to an isolated tmp layout, and
    pre-populate the per-state PARTS cache so main() never needs a real Fred/ALFRED call."""
    raw = tmp_path / "raw"
    parts = raw / "_state_ur_parts"
    parts.mkdir(parents=True)
    out = raw / "state_ur_alfred_vintages.parquet"
    monkeypatch.setattr(FSV, "ROOT", tmp_path)
    monkeypatch.setattr(FSV, "RAW", raw)
    monkeypatch.setattr(FSV, "PARTS", parts)
    monkeypatch.setattr(FSV, "OUT", out)
    monkeypatch.setattr(FSV, "states", lambda: states)
    for i, st in enumerate(states):
        _sentinel_part(st, i + 1).to_parquet(parts / f"{st}.parquet", index=False)
    return out


def _stub_fred(monkeypatch, fetch_one=None):
    """main() constructs a real `Fred(api_key=FA._fred_key())` unconditionally near its top, BEFORE the
    per-state loop even checks whether any state actually needs fetching -- so any test that
    reaches that line requires a real FRED_API_KEY in the environment unless this is stubbed,
    which would fail on a clean clone or in CI. Only tests that reach past the early OUT.exists()
    guard need this."""
    monkeypatch.setattr(FA, "_fred_key", lambda: "fake-key-for-test")

    class _FakeFred:
        def __init__(self, api_key):
            pass

    monkeypatch.setattr("fredapi.Fred", _FakeFred)
    if fetch_one is not None:
        monkeypatch.setattr(FA, "fetch_one", fetch_one)


def test_existing_merged_output_is_not_overwritten_without_force(monkeypatch, tmp_path):
    """THE CONFIRMED REGRESSION: a pre-existing OUT must be left untouched -- and main() must
    return 0, not crash or silently proceed -- when --force is not passed, even though every
    per-state PARTS file is already cached (so the old code's ONLY gate was satisfied while the
    real one, on OUT itself, did not exist at all)."""
    out = _wire_tmp_layout(monkeypatch, tmp_path, ["AA", "BB"])
    sentinel = pd.DataFrame({"state": ["ZZ"], "date": pd.to_datetime(["1999-01-01"]),
                              "realtime_start": pd.to_datetime(["1999-01-02"]), "value": [-999.0]})
    sentinel.to_parquet(out, index=False)
    mtime_before = out.stat().st_mtime_ns

    monkeypatch.setattr(sys, "argv", ["fetch_state_ur_vintages.py"])

    rc = FSV.main()
    assert rc == 0
    assert out.stat().st_mtime_ns == mtime_before, "OUT must not be rewritten without --force"
    assert pd.read_parquet(out)["state"].iloc[0] == "ZZ", "OUT's content must be the untouched sentinel"


def test_force_flag_overwrites_the_merged_output_deliberately(monkeypatch, tmp_path):
    """Positive control: --force must still allow a deliberate rebuild of OUT. --force also forces
    the per-state PARTS refetch (the original, already-correct gate), so fetch_one is monkeypatched
    to synthetic data rather than hitting the real ALFRED API."""
    out = _wire_tmp_layout(monkeypatch, tmp_path, ["AA", "BB"])
    sentinel = pd.DataFrame({"state": ["ZZ"], "date": pd.to_datetime(["1999-01-01"]),
                              "realtime_start": pd.to_datetime(["1999-01-02"]), "value": [-999.0]})
    sentinel.to_parquet(out, index=False)

    def _fake_fetch_one(sid, fred):
        return pd.DataFrame({"series_id": [sid], "date": pd.to_datetime(["2020-01-01"]),
                              "realtime_start": pd.to_datetime(["2020-01-02"]), "value": [1.0]})

    _stub_fred(monkeypatch, fetch_one=_fake_fetch_one)
    monkeypatch.setattr(sys, "argv", ["fetch_state_ur_vintages.py", "--force", "--as-of=none"])

    rc = FSV.main()
    assert rc == 0
    fresh = pd.read_parquet(out)
    assert set(fresh["state"]) == {"AA", "BB"}, "OUT must be rebuilt from freshly-fetched parts when --force is passed"


def test_as_of_silently_collapsing_state_coverage_refuses_to_write(monkeypatch, tmp_path, capsys):
    """THE CORE HAZARD: `--as-of=` filters the merged panel by realtime_start, and without a guard
    a state's rows can be filtered out ENTIRELY (as_of earlier than that state's own ALFRED archive
    start). Writing the merged file unconditionally would then make it the new "immutable" OUT,
    missing that state -- permanently poisoned until an explicit --force. Two states with genuinely
    different archive-start dates, an as_of chosen
    to sit between them: main() must refuse to write (return 1) and name the dropped state,
    rather than writing a 1-state panel as if it were complete."""
    parts = tmp_path / "raw" / "_state_ur_parts"
    parts.mkdir(parents=True)
    out = tmp_path / "raw" / "state_ur_alfred_vintages.parquet"
    monkeypatch.setattr(FSV, "ROOT", tmp_path)
    monkeypatch.setattr(FSV, "RAW", tmp_path / "raw")
    monkeypatch.setattr(FSV, "PARTS", parts)
    monkeypatch.setattr(FSV, "OUT", out)
    monkeypatch.setattr(FSV, "states", lambda: ["EARLY", "LATE"])
    pd.DataFrame({"state": ["EARLY"], "date": pd.to_datetime(["2005-06-01"]),
                  "realtime_start": pd.to_datetime(["2005-06-17"]), "value": [1.0]}
                 ).to_parquet(parts / "EARLY.parquet", index=False)
    pd.DataFrame({"state": ["LATE"], "date": pd.to_datetime(["2007-06-01"]),
                  "realtime_start": pd.to_datetime(["2007-06-19"]), "value": [2.0]}
                 ).to_parquet(parts / "LATE.parquet", index=False)
    _stub_fred(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["fetch_state_ur_vintages.py", "--as-of=2006-06-01"])
    rc = FSV.main()

    assert rc == 1, "must refuse (nonzero exit), not silently write a collapsed panel"
    assert not out.exists(), "OUT must not be written when --as-of collapses a state's coverage entirely"
    printed = capsys.readouterr().out
    assert "LATE" in printed  # the dropped state must be named


def test_as_of_that_preserves_all_states_writes_normally(monkeypatch, tmp_path):
    """Companion sanity check: an as_of that keeps every state's rows must write normally, so the
    new guard is not over-eager."""
    parts = tmp_path / "raw" / "_state_ur_parts"
    parts.mkdir(parents=True)
    out = tmp_path / "raw" / "state_ur_alfred_vintages.parquet"
    monkeypatch.setattr(FSV, "ROOT", tmp_path)
    monkeypatch.setattr(FSV, "RAW", tmp_path / "raw")
    monkeypatch.setattr(FSV, "PARTS", parts)
    monkeypatch.setattr(FSV, "OUT", out)
    monkeypatch.setattr(FSV, "states", lambda: ["EARLY", "LATE"])
    pd.DataFrame({"state": ["EARLY"], "date": pd.to_datetime(["2005-06-01"]),
                  "realtime_start": pd.to_datetime(["2005-06-17"]), "value": [1.0]}
                 ).to_parquet(parts / "EARLY.parquet", index=False)
    pd.DataFrame({"state": ["LATE"], "date": pd.to_datetime(["2007-06-01"]),
                  "realtime_start": pd.to_datetime(["2007-06-19"]), "value": [2.0]}
                 ).to_parquet(parts / "LATE.parquet", index=False)

    _stub_fred(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["fetch_state_ur_vintages.py", "--as-of=2026-01-01"])
    rc = FSV.main()

    assert rc == 0
    assert out.exists()
    assert set(pd.read_parquet(out)["state"]) == {"EARLY", "LATE"}


def test_missing_merged_output_is_always_written(monkeypatch, tmp_path):
    """Negative control: when OUT does not exist yet, main() must write it regardless of --force.
    PARTS is pre-cached for both states, so no per-state fetch actually happens -- but main()
    still constructs a real Fred client before checking that, so this is stubbed too."""
    out = _wire_tmp_layout(monkeypatch, tmp_path, ["AA", "BB"])
    assert not out.exists()
    _stub_fred(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["fetch_state_ur_vintages.py", "--as-of=none"])

    rc = FSV.main()
    assert rc == 0
    assert out.exists()
    assert set(pd.read_parquet(out)["state"]) == {"AA", "BB"}
