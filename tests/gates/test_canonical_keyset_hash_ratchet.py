"""Regression for the gate-ratchet swap-exploit: N_VALUES/N_CHECKED are
exact-equality on COUNT alone, which cannot see
a same-count substitution of WHICH keys are covered -- removing one real CHECKS[] entry and adding
an unrelated passthrough registration for a different key leaves the count unchanged. The new
VALUES_KEYS_HASH/CHECKED_KEYS_HASH ratchets hash the actual sorted key set instead, and catch
exactly this swap.
"""
import json

import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")


def test_keyset_hashes_match_the_real_committed_canonical_json():
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    assert C._keyset_hash(payload.keys()) == C.VALUES_KEYS_HASH
    assert C._keyset_hash(k for k in C.CHECKS if k in payload) == C.CHECKED_KEYS_HASH


def test_a_same_count_check_swap_is_caught_by_the_hash_even_though_the_count_ratchet_would_miss_it(
    monkeypatch,
):
    """THE CORE REGRESSION (a live-demonstrated exploit): remove one real, registered
    check and add an unrelated passthrough for a different, previously-unregistered key. The count
    of registered checks is unchanged (N_CHECKED's own ratchet is satisfied), but the SET of covered
    keys is different -- the hash ratchet must catch this even when the count ratchet cannot."""
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    checked = sorted(k for k in C.CHECKS if k in payload)
    unregistered = sorted(set(payload.keys()) - set(C.CHECKS.keys()))
    assert checked and unregistered, "fixture assumption: both a checked and an unregistered key exist"

    victim, replacement = checked[0], unregistered[0]
    new_checks = dict(C.CHECKS)
    del new_checks[victim]
    new_checks[replacement] = lambda: payload[replacement]["value"]  # trivial always-agreeing passthrough
    monkeypatch.setattr(C, "CHECKS", new_checks)

    registered_after = sum(1 for k in C.CHECKS if k in payload)
    assert registered_after == C.N_CHECKED, (
        "fixture assumption: the swap must be same-count, or this isn't exercising the exploit"
    )

    got_hash = C._keyset_hash(k for k in C.CHECKS if k in payload)
    assert got_hash != C.CHECKED_KEYS_HASH, (
        "the swap changed which keys are covered but the hash didn't change -- ratchet is not closing "
        "the exploit"
    )


def test_check_reports_inconsistent_when_the_checked_keyset_hash_is_wrong(monkeypatch, capsys):
    monkeypatch.setattr(C, "CHECKED_KEYS_HASH", "0" * 64)
    rc = C.check()
    printed = capsys.readouterr().out
    assert "CHECKED_KEYS_HASH" in printed and "INCONSISTENT" in printed
    assert rc != 0


def test_check_reports_inconsistent_when_the_values_keyset_hash_is_wrong(monkeypatch, capsys):
    monkeypatch.setattr(C, "VALUES_KEYS_HASH", "0" * 64)
    rc = C.check()
    printed = capsys.readouterr().out
    assert "VALUES_KEYS_HASH" in printed and "INCONSISTENT" in printed
    assert rc != 0


def test_rehash_keysets_prints_the_currently_correct_literals(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["canonical.py", "--rehash-keysets"])
    C.main()
    printed = capsys.readouterr().out
    assert C.VALUES_KEYS_HASH in printed
    assert C.CHECKED_KEYS_HASH in printed
