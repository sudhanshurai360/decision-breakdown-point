"""Regression test: `_payload()`/
`_digest()` used to include ONLY each key's `"value"` field -- every companion field a stage passes
through `emit()`'s `**extra` (n, numerator, denominator, of, p_value, aux_max, n_qualifying, band,
sd, n_eff, ...) sat entirely outside the hash and every CHECKS re-derivation, even on keys counted
"machine-checked". Demonstrated live before the fix: setting `bestarm.spearman_H_mcs`'s own `n`
field from 12 to 1011 required no hash re-stamp at all (the digest was bit-identical before and
after that mutation) and `canonical.py --check` still reported 172/172 ok, PASS. This matters
specifically because `claims.py`'s own `orphans()` trusts these same companion fields as
legitimizing evidence for prose numbers in the manuscript -- the one consumer that relies on them
treated them as verified while nothing upstream ever verified them.

Fixed by making `_payload()` include every field except the genuinely prose-only ones
(producer/spec/cites/cross_check -- human-readable documentation, deliberately still excluded so
editing a spec's wording or fixing a cite pointer does not invalidate a published number), via an
explicit denylist rather than an allowlist so a new companion-data kwarg some future emit() call
introduces is covered by default.
"""
import json

import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")


def test_payload_includes_companion_fields_not_just_value():
    """A key with a real companion field (n=) must expose it in _payload(), not just 'value'."""
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    assert "bestarm.spearman_H_mcs" in payload
    fields = payload["bestarm.spearman_H_mcs"]
    assert "value" in fields
    assert "n" in fields, "companion field 'n' is missing from _payload()'s output"


def test_payload_excludes_prose_only_fields():
    """producer/spec/cites/cross_check are deliberately NOT hashed -- editing a spec's wording or
    fixing a cite pointer must not invalidate a published number's hash."""
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    for key, fields in payload.items():
        for prose_field in ("producer", "spec", "cites", "cross_check"):
            assert prose_field not in fields, (
                f"{key}: prose-only field {prose_field!r} leaked into the hashed payload"
            )


def test_corrupting_a_companion_field_changes_the_digest():
    """THE CONFIRMED REGRESSION, locked in: mutating ONLY a companion field (never touching
    'value' itself) must change _digest()'s output, so canonical.py --check would report
    _meta INCONSISTENT rather than silently passing."""
    doc = json.loads(C.OUT.read_text())
    before = C._digest(doc)
    assert "bestarm.spearman_H_mcs" in doc and "n" in doc["bestarm.spearman_H_mcs"]
    doc["bestarm.spearman_H_mcs"]["n"] = 1011  # a plausible-looking, fully wrong corruption
    after = C._digest(doc)
    assert before != after, (
        "corrupting a companion field (n) alone did not change the digest -- companion fields "
        "are not actually covered by the hash"
    )


def test_corrupting_a_companion_field_fails_check(monkeypatch, capsys, tmp_path):
    """End-to-end: a companion-field-only corruption, with the hash left STALE (as a real
    corruption would be -- an attacker or a bug wouldn't also re-stamp the hash to match), must
    make `canonical.py --check` FAIL, not PASS."""
    doc = json.loads(C.OUT.read_text())
    doc["bestarm.spearman_H_mcs"]["n"] = 1011
    # hash intentionally NOT re-stamped -- this is what a real silent corruption looks like
    tmp_out = tmp_path / "canonical.json"
    tmp_out.write_text(json.dumps(doc, indent=2))
    monkeypatch.setattr(C, "OUT", tmp_out)

    rc = C.check()
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "INCONSISTENT" in out


def test_clean_check_still_passes_after_the_fix():
    """The richer payload must not turn today's real, correct canonical.json into a false FAIL."""
    rc = C.check()
    assert rc == 0
