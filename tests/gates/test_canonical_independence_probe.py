"""Regression tests: CHECKED_KEYS_HASH
pins WHICH keys are registered, not what their bodies do. Swapping a CHECKS[] body for a readback
of canonical.json's own stored value left the key-set hash unchanged and made canonical.py --check
(and acceptance.py) agree with themselves -- live-demonstrated on eb.episodes 29 -> 39, both gates
PASS. 135 of 230 registered keys were exposed. check() now re-runs every registered check against
a PERTURBED temp copy of canonical.json and requires each result to be identical to the real-OUT
run; a readback cannot help but change.
"""
import json

import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")


def _readback(key):
    return lambda: json.loads(C.OUT.read_text())[key]["value"]


def test_probe_flags_a_readback_swap_that_the_keyset_hash_cannot_see(monkeypatch):
    """THE CORE REGRESSION: replace one genuine re-derivation with a readback. The registered key
    SET is unchanged (so CHECKED_KEYS_HASH still matches), but the probe must report the key."""
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    victim = next(k for k in sorted(C.CHECKS) if k in payload and k not in C.KNOWN_OUT_DEPENDENT
                  and isinstance(payload[k]["value"], (int, float)))
    swapped = dict(C.CHECKS)
    swapped[victim] = _readback(victim)
    monkeypatch.setattr(C, "CHECKS", swapped)
    assert C._keyset_hash(k for k in C.CHECKS if k in payload) == C.CHECKED_KEYS_HASH, (
        "fixture assumption: the swap must leave the key set -- hence the old hash ratchet -- untouched"
    )
    real = C._run_all_checks(payload)
    dependent = C._independence_probe(doc, real)
    assert [k for k, _, _ in dependent] == [victim]


def test_probe_is_silent_on_the_real_registry_except_the_two_disclosed_entries(monkeypatch):
    doc = json.loads(C.OUT.read_text())
    real = C._run_all_checks(C._payload(doc))
    assert C._independence_probe(doc, real) == []
    # and the disclosed two really ARE OUT-dependent -- with the allowlist emptied, the probe must
    # report exactly them (otherwise the allowlist is stale and would hide a future readback there)
    monkeypatch.setattr(C, "KNOWN_OUT_DEPENDENT", frozenset())
    assert {k for k, _, _ in C._independence_probe(doc, real)} == {
        "vintage.window_start", "election.stability_below_null_p05"}


def test_disclosed_dependence_allowlist_is_hashed_and_matches():
    assert C._keyset_hash(C.KNOWN_OUT_DEPENDENT) == C.OUT_DEPENDENT_KEYS_HASH


def test_check_fails_when_a_readback_is_present(monkeypatch, capsys):
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    victim = next(k for k in sorted(C.CHECKS) if k in payload and k not in C.KNOWN_OUT_DEPENDENT
                  and isinstance(payload[k]["value"], (int, float)))
    swapped = dict(C.CHECKS)
    swapped[victim] = _readback(victim)
    monkeypatch.setattr(C, "CHECKS", swapped)
    rc = C.check()
    out = capsys.readouterr().out
    assert rc != 0
    assert f"NOT INDEPENDENT — {victim}" in out


def test_perturb_value_changes_every_supported_type():
    assert C._perturb_value(3) == 4
    assert C._perturb_value(2.0) == 4.0
    assert C._perturb_value(True) is False
    assert C._perturb_value("x") == "x_PERTURBED"
    assert C._perturb_value([1, 2.0]) == [2, 4.0]
    assert C._perturb_value({"a": 1}) == {"a": 2}
    assert C._perturb_value(None) == "PERTURBED_NONE"       # None must not be a fixed point
    assert C._perturb_value(float("nan")) == 12345.678       # nor NaN


# The probe used to repoint only OUT, so a body
# reading the producer FRAGMENT (the same emitted value merge() copies) passed unflagged. FRAGDIR is
# now mirrored (perturbed) too, and the remaining bypasses are closed statically.
def _fragment_readback(key):
    def body():
        for f in C.FRAGDIR.glob("*.json"):
            d = json.loads(f.read_text())
            if key in d:
                return d[key]["value"]
        raise KeyError(key)
    return body


def test_probe_flags_a_fragment_readback(monkeypatch):
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    victim = next(k for k in sorted(C.CHECKS) if k in payload and k not in C.KNOWN_OUT_DEPENDENT
                  and isinstance(payload[k]["value"], (int, float))
                  and any(k in json.loads(f.read_text()) for f in C.FRAGDIR.glob("*.json")))
    swapped = dict(C.CHECKS)
    swapped[victim] = _fragment_readback(victim)
    monkeypatch.setattr(C, "CHECKS", swapped)
    real = C._run_all_checks(payload)
    assert [k for k, _, _ in C._independence_probe(doc, real)] == [victim]
    assert C.FRAGDIR.exists() and any(C.FRAGDIR.glob("*.json"))   # restored, not left pointing at the temp mirror


def test_source_ratchet_is_clean_on_the_shipped_file():
    assert C._source_ratchet() == []


def test_source_ratchet_catches_a_literal_path_and_memoisation(tmp_path, monkeypatch):
    """Feed _source_ratchet a doctored copy of canonical.py's source through a stand-in module."""
    import importlib.util, inspect, sys as _sys
    src = inspect.getsource(C)
    doctored = src.replace(
        "def _shape_ratio(which):",
        'def _shape_ratio(which):\n    _ = (ROOT / "paper" / "canonical.json").read_text()  # literal-path readback\n', 1)
    doctored = doctored.replace("import shutil\n", "import shutil\nimport functools\n", 1)
    doctored = doctored.replace("def _payload(", "@functools.lru_cache(maxsize=None)\ndef _payload(", 1)
    p = tmp_path / "canonical_doctored.py"
    p.write_text(doctored)
    spec = importlib.util.spec_from_file_location("canonical_doctored", p)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["canonical_doctored"] = mod
    try:
        spec.loader.exec_module(mod)
        problems = mod._source_ratchet()
    finally:
        _sys.modules.pop("canonical_doctored", None)
    joined = "\n".join(problems)
    assert "2 string constants name the merged file" in joined
    assert "memoisation" in joined
