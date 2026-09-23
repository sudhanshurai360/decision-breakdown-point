"""Regression test: acceptance.py's `classify()` and `blocked_reason()` used `abs(g - w) <= tol`-style
comparisons everywhere, none of which special-case NaN==NaN -- any NaN comparison in Python is
always False, so a key whose `got` AND `want` are both legitimately NaN (a true match) would be
reported DIFFER, even though NaN is treated as a legitimate value elsewhere in this system (e.g.
canonical.py::_close() already special-cases NaN==NaN as a match, for the same reason -- that
special-casing previously existed only there, never mirrored here).

Fixed by mirroring canonical.py::_close()'s `np.isnan(fa) and np.isnan(fb)` convention in both
functions (acceptance.py's own copies read `math.isnan(g) and math.isnan(w)`).
"""
import acceptance as A


def test_classify_treats_nan_equal_to_nan_as_a_match():
    verdict, note = A.classify("some.key", float("nan"), float("nan"))
    assert verdict == "MATCH"
    assert "nan" in note.lower()


def test_classify_still_flags_nan_vs_a_real_number_as_a_genuine_difference():
    """Sanity check the fix isn't over-eager: NaN vs a real, finite number must still DIFFER."""
    verdict, note = A.classify("some.key", float("nan"), 1.0)
    assert verdict == "DIFFER"
    verdict, note = A.classify("some.key", 1.0, float("nan"))
    assert verdict == "DIFFER"


def test_classify_ordinary_numeric_comparison_is_unaffected():
    """The fix must not change behavior for the ordinary, non-NaN case."""
    assert A.classify("some.key", 1.0, 1.0) == ("MATCH", "")
    assert A.classify("some.key", 1.0, 2.0)[0] == "DIFFER"


def test_blocked_reason_treats_nan_equal_to_nan_as_matching_canon():
    """blocked_reason()'s value-aware fallback must also treat NaN==NaN as a match against
    canon, consistent with classify() above -- otherwise a legitimately-NaN BLOCKED key would fall
    through to a spurious DIFFER the moment it's added to the ledger's comparable set."""
    why = A.blocked_reason("eb.fake_nan_key", got=float("nan"),
                            canon={"eb.fake_nan_key": {"value": float("nan")}})
    # blocked_reason() only returns non-None if the key matches a real BLOCKED prefix; "eb." is
    # one of the documented exempt prefixes, so this exercises the real code path.
    assert why is not None, "expected 'eb.' to match a real BLOCKED prefix for this test to be meaningful"
