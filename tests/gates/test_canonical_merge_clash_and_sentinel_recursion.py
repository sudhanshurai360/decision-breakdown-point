"""Regression tests for four canonical.py infrastructure changes (each
documented inline in canonical.py at the site): merge() clash detection on companion fields,
sentinel-scan recursion into lists and dicts, and two stale re-derivation formulas.
"""
import json

import pytest

import canonical as C


# --- merge()'s clash detection used to be value-only -------------------------------------------

def _write_frag(dirpath, stage, key, value, extra_field=None, extra_value=None):
    fields = {"value": value, "producer": stage, "spec": "x", "cites": [], "cross_check": None}
    if extra_field is not None:
        fields[extra_field] = extra_value
    doc = {"_stage": stage, "_generated": "2026-01-01T00:00:00+00:00", "_n": 1, key: fields}
    (dirpath / f"{stage}.json").write_text(json.dumps(doc, indent=2))


def test_merge_flags_same_value_different_companion_field_as_a_clash(tmp_path, monkeypatch, capsys):
    """THE CORE REGRESSION: two fragments emitting the SAME key with an IDENTICAL 'value' but
    DIFFERENT companion data (e.g. 'n') used to merge silently, last-write-wins, no CLASH printed
    -- even though companion fields are hash-covered and trusted by claims.py's orphan check as
    legitimizing evidence. Must now be flagged."""
    frag_dir = tmp_path / "frags"
    frag_dir.mkdir()
    _write_frag(frag_dir, "stageA", "shared.key", 5.0, extra_field="n", extra_value=12)
    _write_frag(frag_dir, "stageB", "shared.key", 5.0, extra_field="n", extra_value=99)
    out = tmp_path / "canonical.json"
    monkeypatch.setattr(C, "FRAGDIR", frag_dir)
    monkeypatch.setattr(C, "OUT", out)
    monkeypatch.setattr(C, "ROOT", tmp_path)

    C.merge()
    printed = capsys.readouterr().out
    assert "CLASH" in printed and "shared.key" in printed, (
        "same-value/different-companion-field collision was not flagged as a CLASH"
    )


def test_merge_does_not_flag_a_clash_when_fragments_genuinely_agree(tmp_path, monkeypatch, capsys):
    """Companion sanity check: two fragments emitting the identical key with identical value AND
    identical companion fields must NOT be flagged -- the fix must not become over-eager."""
    frag_dir = tmp_path / "frags"
    frag_dir.mkdir()
    _write_frag(frag_dir, "stageA", "shared.key", 5.0, extra_field="n", extra_value=12)
    _write_frag(frag_dir, "stageB", "shared.key", 5.0, extra_field="n", extra_value=12)
    out = tmp_path / "canonical.json"
    monkeypatch.setattr(C, "FRAGDIR", frag_dir)
    monkeypatch.setattr(C, "OUT", out)
    monkeypatch.setattr(C, "ROOT", tmp_path)

    C.merge()
    printed = capsys.readouterr().out
    assert "CLASH" not in printed


def test_merge_still_flags_a_genuine_different_value_clash(tmp_path, monkeypatch, capsys):
    """The pre-existing, already-working case must still work after the fix."""
    frag_dir = tmp_path / "frags"
    frag_dir.mkdir()
    _write_frag(frag_dir, "stageA", "shared.key", 5.0)
    _write_frag(frag_dir, "stageB", "shared.key", 7.0)
    out = tmp_path / "canonical.json"
    monkeypatch.setattr(C, "FRAGDIR", frag_dir)
    monkeypatch.setattr(C, "OUT", out)
    monkeypatch.setattr(C, "ROOT", tmp_path)

    C.merge()
    printed = capsys.readouterr().out
    assert "CLASH" in printed and "shared.key" in printed


# --- sentinel scan didn't recurse into list-valued fields ---------------------------------------

def test_contains_sentinel_detects_a_bare_sentinel_value():
    assert C._contains_sentinel(C.SENTINEL) is True
    assert C._contains_sentinel(42) is False


def test_contains_sentinel_recurses_into_a_list():
    """THE CORE REGRESSION: a sentinel embedded inside a list-valued companion field (e.g. a
    band/counts/states/alphas field -- 32 such fields exist in the live canonical.json, counting
    every list-valued entry field other than `cites` and `value`) used to
    escape detection entirely, since `[0.1, SENTINEL] == SENTINEL` is False and
    `str([0.1, SENTINEL])` never equals `str(SENTINEL)`."""
    assert C._contains_sentinel([0.1, C.SENTINEL]) is True
    assert C._contains_sentinel([0.1, 0.2, 0.3]) is False


def test_check_reports_sentinel_left_live_when_hidden_inside_a_list(monkeypatch, tmp_path, capsys):
    """End-to-end: check() must print the SENTINEL warning for a sentinel hidden inside a list
    field, not just a bare value field (a synthetic doc with a sentinel inside a 'band' field,
    hash correctly stamped, previously produced NO warning)."""
    doc = {"fake.key_with_list_field": {"value": 1.0, "band": [0.1, C.SENTINEL]}}
    doc["_meta"] = {"version": 1, "n_values": 1, "n_stages": 1, "hash": C._digest(doc)}
    out = tmp_path / "canonical.json"
    out.write_text(json.dumps(doc, indent=2))
    monkeypatch.setattr(C, "OUT", out)

    C.check()
    printed = capsys.readouterr().out
    assert "SENTINEL LEFT LIVE" in printed, (
        "a sentinel hidden inside a list-valued field produced no warning"
    )


# --- blind re-derivation check: sentinel scan also didn't recurse into dicts -------------------

def test_contains_sentinel_recurses_into_a_dict():
    """Companion coverage to the list case above: a sentinel hidden inside a dict-valued companion
    field must also be caught. Not a hypothetical shape -- eb.genuine_divergent_states_detail's
    `dates`/`spreads_pp` fields ARE dict-valued ({"MA": ..., "NV": ...} per-state maps) in the real,
    live canonical.json, so this recursion path is exercised by production data, not just by a
    synthetic edge case."""
    assert C._contains_sentinel({"MA": 0.5, "NV": C.SENTINEL}) is True
    assert C._contains_sentinel({"MA": 0.5, "NV": 0.8}) is False


def test_contains_sentinel_recurses_into_a_dict_nested_inside_a_list_and_vice_versa():
    """Mixed nesting must also be caught -- list-of-dicts and dict-of-lists, not just one level."""
    assert C._contains_sentinel([{"MA": C.SENTINEL}]) is True
    assert C._contains_sentinel({"MA": [0.1, C.SENTINEL]}) is True
    assert C._contains_sentinel([{"MA": 0.5}, {"NV": 0.8}]) is False


def test_check_reports_sentinel_left_live_when_hidden_inside_a_dict(monkeypatch, tmp_path, capsys):
    """End-to-end: check() must print the SENTINEL warning for a sentinel hidden inside a
    dict-valued companion field -- reproduces a real planted-sentinel scenario (planting SENTINEL
    into the real eb.genuine_divergent_states_detail.spreads_pp dict, hash correctly re-stamped,
    previously produced a clean PASS with no warning)."""
    doc = {"fake.key_with_dict_field": {"value": 1.0, "spreads_pp": {"MA": 0.5, "NV": C.SENTINEL}}}
    doc["_meta"] = {"version": 1, "n_values": 1, "n_stages": 1, "hash": C._digest(doc)}
    out = tmp_path / "canonical.json"
    out.write_text(json.dumps(doc, indent=2))
    monkeypatch.setattr(C, "OUT", out)

    C.check()
    printed = capsys.readouterr().out
    assert "SENTINEL LEFT LIVE" in printed, (
        "a sentinel hidden inside a dict-valued field produced no warning"
    )


def test_real_eb_genuine_divergent_states_detail_field_is_dict_valued():
    """Sanity check that the scenario this fix protects against is not hypothetical: confirms the
    real, committed canonical.json still has a dict-valued companion field today, so this
    regression coverage stays load-bearing rather than testing a scenario that no longer exists."""
    doc = json.loads(C.OUT.read_text())
    detail = doc.get("eb.genuine_divergent_states_detail")
    if detail is None:
        pytest.skip("eb.genuine_divergent_states_detail not present in the real canonical.json")
    assert isinstance(detail.get("dates"), dict)
    assert isinstance(detail.get("spreads_pp"), dict)


# --- two of the "stale re-derivation formula" pattern instances --------------------------------

def test_rho_delta_p05_check_uses_abs(monkeypatch):
    """Regression for the missing abs(): nullsens.rho_delta_p05's re-derivation must match the
    real producer's `d_rho = abs(cell("rho08","p05") - cell("rho05","p05"))` exactly, including
    the sign-insensitivity -- swapping which of rho08/rho05 is larger must not change the result."""
    calls = {"rho08": 0.10, "rho05": 0.30}  # rho05 > rho08, so a signed diff would be negative
    monkeypatch.setattr(C, "_ns", lambda arm, stat: calls[arm])
    result = C.CHECKS["nullsens.rho_delta_p05"]()
    assert result == pytest.approx(0.20), f"expected abs(0.10-0.30)=0.20, got {result}"
    assert result >= 0, "nullsens.rho_delta_p05 must never be negative (it's an absolute change)"


def test_curated_dropped_check_includes_fetch_err_status(monkeypatch):
    """Regression: data.curated_dropped's re-derivation must match data_config.py's own current
    `!= "PASS"` filter, not the narrower `== "DROP"` this key had regressed to -- a FETCH_ERR-status
    candidate must be included in the disclosed drop-list, not silently omitted."""
    fake_manifest = {"series": {
        "A": {"status": "PASS"},
        "B": {"status": "DROP"},
        "C": {"status": "FETCH_ERR some transient network error"},
    }}
    monkeypatch.setattr(C, "_manifest", lambda: fake_manifest)
    result = C.CHECKS["data.curated_dropped"]()
    assert "B" in result and "C" in result, (
        f"expected both DROP and FETCH_ERR candidates in the disclosed drop-list, got {result!r}"
    )
    assert "A" not in result


def test_the_real_committed_canonical_still_passes_check():
    """Sanity check on the fixture itself: all 4 fixes above must not turn today's real, correct
    canonical.json into a false FAIL."""
    if not C.OUT.exists():
        pytest.skip("paper/canonical.json not available")
    rc = C.check()
    assert rc == 0
