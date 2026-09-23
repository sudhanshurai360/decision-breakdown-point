"""Regression test: claims.py's
`orphans()` companion-field allowlist (the `extra` tuple it consults per canonical value) was
missing every numeric array-shaped companion field currently live in canonical.json ("counts",
"thresholds_pp", "distinct_counts", "distinct_pi", "alphas", "band_count") -- reopening, in the
checker's own general mechanism, the exact defect a previous fix found and patched with a bespoke,
single-key pytest test for `eb.knife_edge_sweep_genuine_states.counts` alone, never touching this
allowlist itself.

Verified live that the S7 prose tokens the "counts" array should cover ("24", "5") were
covered ONLY by incidental collision with dozens of unrelated canonical values' own renderings --
never by the companion array itself. That means a future edit that changes the threshold set or
introduces a sweep-aggregation bug, without updating the prose sentence to match, would have left
orphans() reporting UNRESOLVED: 0 regardless -- "false comfort via incidental collision".

This test proves the fix does real detection work, not just "doesn't crash": it builds a minimal
canon with a single key carrying one of the six newly-allowlisted array fields and an
otherwise-unique token (not producible by any other rendering, so no incidental collision is
possible), confirms the token is covered while the array holds its correct value, then corrupts
the array on a copy of canon and confirms the SAME token is now reported as an ORPHAN -- i.e. the
allowlist entry is actually load-bearing for detecting a stale/corrupted companion array, exactly
the near-miss this generalizes the fix for.
"""
import json

import pytest

import claims as C

NEW_ALLOWLIST_FIELDS = [
    "counts", "thresholds_pp", "distinct_counts", "distinct_pi", "alphas", "band_count",
]


def _patch_body(monkeypatch, tmp_path, body: str):
    """Point claims.py's manuscript-loading machinery at a synthetic draft.md containing exactly
    `body`, with no note.md, so orphans()'s internal load_body() call sees a fully controlled prose
    surface instead of the real paper/draft_ssrn.md -- required because orphans() reads the manuscript
    via module-level DRAFT/NOTE paths, not as a function argument."""
    draft = tmp_path / "draft.md"
    draft.write_text(f"## 1. Section\n\n{body}\n")
    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "NOTE", tmp_path / "does_not_exist_note.md")
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)


@pytest.mark.parametrize("field", NEW_ALLOWLIST_FIELDS)
def test_companion_array_field_covers_its_own_token(field, tmp_path, monkeypatch):
    """The direct claim: a value carrying one of the six newly-allowlisted array fields must have
    that array's own renderings covered by orphans(), the same class of coverage aux_max/
    n_qualifying/sd/band/etc. already get."""
    token = "847213"  # large, arbitrary 6-digit number -- won't collide with any BENIGN rule or
    # any other canonical rendering in this minimal, single-key canon.
    canon = {
        "test.companion_key": {
            "value": 1, "producer": "test", "cites": [], "cross_check": None,
            field: [int(token), 55555],
        }
    }
    _patch_body(monkeypatch, tmp_path, f"The sweep identifies {token} qualifying cases in this run.")
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert token not in orph, (
        f"{field!r} must cover its own array rendering, the same way the other companion-data "
        f"fields already allowlisted in orphans() do"
    )


@pytest.mark.parametrize("field", NEW_ALLOWLIST_FIELDS)
def test_corrupting_the_companion_array_surfaces_the_stale_token_as_orphan(field, tmp_path, monkeypatch):
    """The mechanism-level proof this fix actually changes detection behavior, not just "doesn't
    crash": with the array at its correct value the token is covered (not an orphan); on a
    corrupted copy of canon (same key, same field name, DIFFERENT array contents), the identical,
    otherwise-unique token must now be reported as an ORPHAN. This is the exact near-miss
    scenario generalized: a companion array going stale relative to the prose must be catchable by
    the checker's own general mechanism, not require a bespoke one-off test."""
    token = "847213"
    canon = {
        "test.companion_key": {
            "value": 1, "producer": "test", "cites": [], "cross_check": None,
            field: [int(token), 55555],
        }
    }
    _patch_body(monkeypatch, tmp_path, f"The sweep identifies {token} qualifying cases in this run.")

    orph_correct, _, _ = C.orphans(canon)
    assert token not in orph_correct, "sanity check: must be covered before corruption"

    corrupted = json.loads(json.dumps(canon))  # deep copy, so the two runs don't share state
    corrupted["test.companion_key"][field] = [999999, 111111]  # stale/corrupted companion array
    orph_corrupted, _, _ = C.orphans(corrupted)
    assert token in orph_corrupted, (
        f"REGRESSION: corrupting the {field!r} companion array must surface the now-stale prose "
        f"token {token!r} as an ORPHAN -- if this fails, the allowlist entry for {field!r} is not "
        f"actually load-bearing for detecting a stale/corrupted companion array, which is exactly "
        f"the finding this test guards against"
    )


def test_field_not_in_allowlist_is_never_covered_by_its_own_array(tmp_path, monkeypatch):
    """Control: a companion field NOT on the allowlist (a made-up name) must never cover its own
    array's renderings -- confirms the six fields above are covered BECAUSE they are on the
    allowlist, not because of some other, broader coverage mechanism that would make this whole
    fix moot."""
    token = "358214"
    canon = {
        "test.companion_key": {
            "value": 1, "producer": "test", "cites": [], "cross_check": None,
            "not_an_allowlisted_field": [int(token), 55555],
        }
    }
    _patch_body(monkeypatch, tmp_path, f"The sweep identifies {token} qualifying cases in this run.")
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert token in orph, (
        "a companion field outside the allowlist must not silently cover its own array's "
        "renderings -- if it does, the allowlist mechanism itself is not doing anything"
    )


def test_real_canonical_json_currently_uses_all_six_fields():
    """Sanity check tying this test to the real corpus: confirms the six fields this fix
    allowlists are genuinely live in paper/canonical.json today (not a hypothetical):
    eb.knife_edge_sweep_genuine_states.{counts,thresholds_pp},
    mcssize.alpha_sweep_nonidentified.{alphas,counts},
    mcssize.alpha_sweep_invariant.{distinct_counts,distinct_pi},
    election.stability_below_null_p05{,_ar1}.band_count."""
    canon = C._canon()
    found = {f: [k for k, v in canon.items() if isinstance(v, dict) and v.get(f) is not None]
             for f in NEW_ALLOWLIST_FIELDS}
    for field, keys in found.items():
        assert keys, f"expected {field!r} to be live on at least one canonical.json key today"
