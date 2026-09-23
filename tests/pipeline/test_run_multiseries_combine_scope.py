"""run_multiseries.py's combine() globs results/parts/*_fragility.csv and takes no argument
recording WHICH series the caller intended, so on its own it would merge whatever part files happen
to be sitting in the shared parts/ directory. A prior, unrelated invocation (a small ad-hoc test
run, or a stale part left over from an earlier full run) can leave part files there that a LATER
--combine call has no way to distinguish from the current run's own fresh output, silently mixing
stale and fresh series (e.g. a leftover 4-series test run's parts blended with a fresh 12-series
run's parts) into one combined table.

The guard is combine_scope_warnings(), which compares the series being combined against the
curated panel's own pre-committed selection log (breadth_manifest_distinct.json's `passed` list)
and returns named warnings on any mismatch -- a warning, not a hard failure, since this pipeline
does not currently feed any canonical/gated value and a deliberate partial-scope test combine is a
legitimate use case this must not block.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import run_multiseries as RM  # noqa: E402


def _manifest(tmp_path, passed):
    p = tmp_path / "breadth_manifest_distinct.json"
    p.write_text(json.dumps({"passed": passed}))
    return p


def test_exact_match_produces_no_warnings(tmp_path):
    """The confirmed-safe case: parts found exactly match the manifest -- no warning."""
    m = _manifest(tmp_path, ["PAYEMS", "UNEMPLOY", "HOUST"])
    warnings = RM.combine_scope_warnings(["PAYEMS", "UNEMPLOY", "HOUST"], manifest_path=m)
    assert warnings == []


def test_extra_stale_series_is_flagged():
    """THE CONFIRMED REGRESSION shape: a stale/leftover series not in the curated manifest must be
    named explicitly, not silently merged in."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = _manifest(Path(d), ["PAYEMS", "UNEMPLOY"])
        warnings = RM.combine_scope_warnings(["PAYEMS", "UNEMPLOY", "STALE_TEST_SERIES"], manifest_path=m)
    assert any("STALE_TEST_SERIES" in w and "NOT in the curated panel" in w for w in warnings), warnings


def test_missing_series_is_flagged():
    """A deliberate partial combine (fewer series than the full curated panel) must be visibly
    disclosed as a partial scope, not silently presented as if it were the full panel."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = _manifest(Path(d), ["PAYEMS", "UNEMPLOY", "HOUST"])
        warnings = RM.combine_scope_warnings(["PAYEMS"], manifest_path=m)
    assert any("MISSING" in w and "UNEMPLOY" in w and "HOUST" in w for w in warnings), warnings


def test_missing_manifest_produces_no_warnings(tmp_path):
    """If no manifest is available to check against, this must not fail or fabricate a warning --
    it silently no-ops (the pre-existing behavior, unweakened by this fix)."""
    nonexistent = tmp_path / "does_not_exist.json"
    warnings = RM.combine_scope_warnings(["PAYEMS", "ANYTHING"], manifest_path=nonexistent)
    assert warnings == []


def _write_parts(parts_dir, frag_series, pred_series):
    parts_dir.mkdir(parents=True, exist_ok=True)
    for s in frag_series:
        (parts_dir / f"{s}_fragility.csv").write_text("series,metric,value\n" f"{s},k_star,1\n")
    for s in pred_series:
        (parts_dir / f"{s}_predictions.csv").write_text("series,year,pred\n" f"{s},2020,0.5\n")


def test_combine_desync_between_fragility_and_predictions_is_flagged(tmp_path, monkeypatch, capsys):
    """THE CORE DEFECT-CATCHING PATH for combine()'s own desync guard -- the
    fragility-parts/predictions-parts set-equality check at run_multiseries.py:118-123, which lives
    in combine(), not main(). Its sibling combine_scope_warnings() has its own tests; this covers
    the guard itself. An interrupted per-series write (fragility part
    written, predictions part not yet written before a crash) must surface a visible SCOPE WARNING
    naming the exact series affected, not silently combine a desynced table."""
    parts_dir = tmp_path / "parts"
    _write_parts(parts_dir, frag_series=["PAYEMS", "UNEMPLOY"], pred_series=["PAYEMS"])
    monkeypatch.setattr(RM, "PARTS", parts_dir)
    monkeypatch.setattr(RM, "RESULTS", tmp_path)
    monkeypatch.setattr(RM, "MANIFEST", tmp_path / "does_not_exist.json")
    RM.combine()
    out = capsys.readouterr().out
    assert "DIFFERENT series sets" in out
    assert "UNEMPLOY" in out  # the fragility-only series must be named


def test_combine_with_zero_prediction_parts_refuses_cleanly_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """THE CORE REGRESSION: at least one
    *_fragility.csv part exists but zero *_predictions.csv parts do (e.g. an interrupted run,
    between a series' two separate atomic writes, on the very first/only series present). Before
    the fix, combine() printed the SCOPE WARNING, wrote a stale multiseries_fragility.csv, THEN
    crashed with a bare `ValueError: No objects to concatenate` on the empty pparts concat. Must
    now refuse cleanly, before either file is written."""
    parts_dir = tmp_path / "parts"
    _write_parts(parts_dir, frag_series=["PAYEMS"], pred_series=[])
    monkeypatch.setattr(RM, "PARTS", parts_dir)
    monkeypatch.setattr(RM, "RESULTS", tmp_path)
    monkeypatch.setattr(RM, "MANIFEST", tmp_path / "does_not_exist.json")

    RM.combine()  # must not raise

    out = capsys.readouterr().out
    assert "no prediction parts" in out
    assert not (tmp_path / "multiseries_fragility.csv").exists(), (
        "combine() must not write a stale, unpaired fragility.csv when there are zero prediction parts"
    )
    assert not (tmp_path / "multiseries_predictions.csv").exists()


def test_combine_matched_parts_produces_no_desync_warning(tmp_path, monkeypatch, capsys):
    """The confirmed-safe case: fragility and predictions parts cover the identical series set --
    no desync warning, and the combined files are written."""
    parts_dir = tmp_path / "parts"
    _write_parts(parts_dir, frag_series=["PAYEMS", "UNEMPLOY"], pred_series=["PAYEMS", "UNEMPLOY"])
    monkeypatch.setattr(RM, "PARTS", parts_dir)
    monkeypatch.setattr(RM, "RESULTS", tmp_path)
    monkeypatch.setattr(RM, "MANIFEST", tmp_path / "does_not_exist.json")
    RM.combine()
    out = capsys.readouterr().out
    assert "DIFFERENT series sets" not in out
    assert (tmp_path / "multiseries_fragility.csv").exists()
    assert (tmp_path / "multiseries_predictions.csv").exists()


def test_real_manifest_matches_real_parts_directory():
    """Direct check against the real, currently-committed state: the series actually present in
    results/parts/ today must match the real breadth_manifest_distinct.json exactly (0 warnings) --
    confirming the fix is genuinely latent against today's real data."""
    import pytest
    parts = ROOT / "multiseries" / "results" / "parts"
    if not parts.exists() or not RM.MANIFEST.exists():
        pytest.skip("parts/ or the real manifest not available")
    found = sorted({p.name.replace("_fragility.csv", "") for p in parts.glob("*_fragility.csv")})
    if not found:
        pytest.skip("no part files present")
    warnings = RM.combine_scope_warnings(found)
    assert warnings == [], f"real parts/ directory currently disagrees with the real manifest: {warnings}"
