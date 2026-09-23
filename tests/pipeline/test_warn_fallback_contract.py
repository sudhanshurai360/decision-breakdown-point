"""m_audit.py::_warn_model_fallback(), models_freq.py::_warn_fallback(), and
density_audit.py::_warn_fallback() are a 3-way fan-out of the identical "print a fallback warning
the FIRST time a (model, exception-type) pair is seen this run, then silently tally further
occurrences" contract, each against its own module-level seen-set. m_audit.py's version is a
strict superset (adds a `context=` kwarg); the other two are structurally identical modulo the
fallback-description wording ("last-value" vs "naive persistence" -- accurate per-file, not a
bug). Diagnostics-only (stderr messages, not computed/emitted values), so lower correctness risk
than the other duplication clusters, but still worth pinning: this test asserts all 3 honor the
"print once, tally silently thereafter" contract, so a future edit to one that breaks the pattern
(e.g. reintroducing a print-every-time regression) fails here instead of just flooding stderr on a
real run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import m_audit as MA
import models_freq as MF
import density_audit as DA


class _FakeError(Exception):
    pass


class _OtherError(Exception):
    pass


def _fresh_key(prefix):
    """A model-name string unique enough that no other test/run has seen this (name, exc-type)
    pair before, so each test is independent of run order without needing to reach into and
    clear each module's private seen-set."""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_m_audit_warn_model_fallback_prints_once_then_tallies_silently(capsys):
    name = _fresh_key("m_audit")
    MA._warn_model_fallback(name, _FakeError("boom"))
    first = capsys.readouterr().err
    assert name in first and "FakeError" in first

    MA._warn_model_fallback(name, _FakeError("boom again"))
    second = capsys.readouterr().err
    assert second == "", "second occurrence of the SAME (name, exc-type) pair must not re-print"

    MA._warn_model_fallback(name, _OtherError("different"))
    third = capsys.readouterr().err
    assert name in third and "OtherError" in third, (
        "a DIFFERENT exception type for the same model must print again (key is (name, exc-type))"
    )


def test_models_freq_warn_fallback_prints_once_then_tallies_silently(capsys):
    name = _fresh_key("models_freq")
    MF._warn_fallback(name, _FakeError("boom"))
    first = capsys.readouterr().err
    assert name in first and "FakeError" in first

    MF._warn_fallback(name, _FakeError("boom again"))
    second = capsys.readouterr().err
    assert second == "", "second occurrence of the SAME (name, exc-type) pair must not re-print"


def test_density_audit_warn_fallback_prints_once_then_tallies_silently(capsys):
    name = _fresh_key("density_audit")
    DA._warn_fallback(name, _FakeError("boom"))
    first = capsys.readouterr().err
    assert name in first and "FakeError" in first

    DA._warn_fallback(name, _FakeError("boom again"))
    second = capsys.readouterr().err
    assert second == "", "second occurrence of the SAME (name, exc-type) pair must not re-print"


def test_all_three_seen_sets_are_keyed_the_same_way():
    """All three implementations key their seen-set on (name, type(exc).__name__), not on the
    exception instance or message -- confirmed by checking a SECOND distinct exception instance
    of the SAME type is still treated as "already seen"."""
    import io
    import contextlib

    for mod, fn in ((MA, MA._warn_model_fallback), (MF, MF._warn_fallback), (DA, DA._warn_fallback)):
        name = _fresh_key(f"keying_{mod.__name__}")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            fn(name, _FakeError("first instance"))
        assert buf.getvalue() != ""

        buf2 = io.StringIO()
        with contextlib.redirect_stderr(buf2):
            fn(name, _FakeError("a completely different message, same type"))
        assert buf2.getvalue() == "", (
            f"{mod.__name__}'s warn-fallback must key on (name, exc TYPE), not the exception "
            f"instance or message -- a second _FakeError with a different message should be "
            f"silently tallied, not re-printed"
        )


def test_models_freq_and_density_audit_fallback_tally_actually_counts(capsys):
    """models_freq.py and density_audit.py track fallbacks in `_FALLBACK_TALLY` (a dict counter),
    not a bare set: a set can only answer "has this happened", which would make the printed claim
    "further occurrences... will be silently tallied" untrue and unqueryable. The tests above check
    only the print-once side effect, so this one asserts the counter's actual VALUE: 3 occurrences
    of the same (name, exc-type) pair must leave `_FALLBACK_TALLY[key] == 3`, not merely
    "greater than 1"."""
    for mod, fn in ((MF, MF._warn_fallback), (DA, DA._warn_fallback)):
        name = _fresh_key(f"tally_{mod.__name__}")
        key = (name, "_FakeError")
        assert key not in mod._FALLBACK_TALLY

        fn(name, _FakeError("one"))
        assert mod._FALLBACK_TALLY[key] == 1
        fn(name, _FakeError("two"))
        assert mod._FALLBACK_TALLY[key] == 2
        fn(name, _FakeError("three"))
        assert mod._FALLBACK_TALLY[key] == 3, (
            f"{mod.__name__}._FALLBACK_TALLY must genuinely count occurrences, not just track "
            f"membership -- got {mod._FALLBACK_TALLY[key]} after 3 calls"
        )
