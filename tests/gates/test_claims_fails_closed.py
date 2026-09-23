"""Regression test: claims.py's main() used
to unconditionally `return 0` on every path -- ORPHAN/DRIFT/MISPLACED were all computed and printed,
but none was wired into the exit code, so `sys.exit(main())` always exited 0 regardless of how many
real problems were found. Also, unlike every sibling gate in code/gates/ (acceptance.py's
N_COMPARABLE/N_BLOCKED, differential_test.py's 3 exact-equality ratchets, metric_consistency.py's
N_COVERED_PRODUCERS/N_COMPARED_NAMES, canonical.py's N_VALUES/N_CHECKED), MISPLACED's count had no
ratchet at all -- nothing distinguished a newly-introduced, unreviewed MISPLACED entry from the
already-reviewed-and-accepted ones (that accepted set is now pinned by `claims.py`'s `N_MISPLACED`,
whose size drifts as the draft changes, so it is not restated here).

Fixed: ORPHAN and TABLE-1-DRIFT now gate the exit code directly (target is exact 0); MISPLACED gets
an exact-equality ratchet (N_MISPLACED) matching every sibling gate's discipline.
"""
import io
from contextlib import redirect_stdout

import pytest

import claims as CL

pytestmark = pytest.mark.skipif(not CL._draft_exists(), reason="paper/draft_ssrn.md not available")


@pytest.fixture(autouse=True)
def _clean_argv(monkeypatch):
    """claims.py's main() now validates sys.argv against an allowlist (closing
    the CLI-argv-silent-ignore gap) -- every test in this file calls CL.main() directly in-process,
    so without this it would see pytest's OWN argv (e.g. '-q', the test file path) and reject it as
    unrecognized, unrelated to what each test actually means to exercise."""
    monkeypatch.setattr("sys.argv", ["claims.py"])


def test_real_run_passes_and_exits_0():
    """The confirmed-safe case: today's real draft_ssrn.md/canonical.json state must still PASS and
    exit 0 -- this fix must not become over-eager and fail a genuinely clean run."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CL.main()
    assert rc == 0
    assert "PASS" in buf.getvalue()


def test_misplaced_ratchet_fails_closed_on_a_regression(monkeypatch):
    """THE CORE REGRESSION: if N_MISPLACED no longer matches the live MISPLACED+unknown count (a
    genuine new entry, reviewed or not), main() must return a nonzero exit code, not silently
    return 0 as it did before this fix."""
    monkeypatch.setattr(CL, "N_MISPLACED", CL.N_MISPLACED - 1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CL.main()
    assert rc == 1
    assert "FAIL" in buf.getvalue()
    assert "RATCHET" in buf.getvalue()


def test_orphan_regression_fails_closed(monkeypatch):
    """An ORPHAN regression (a prose number that no longer resolves to any canonical value or
    benign rule) must fail the exit code, not just print a nonzero UNRESOLVED count that nothing
    acts on."""
    monkeypatch.setattr(CL, "orphans", lambda canon, sections: ({"999999": "…fake…"}, 1, 0))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CL.main()
    assert rc == 1
    assert "FAIL" in buf.getvalue()


def test_table_drift_regression_fails_closed(monkeypatch):
    """A TABLE 1 drift regression (the draft's copy no longer matches the generated one) must fail
    the exit code."""
    monkeypatch.setattr(CL, "table_drift", lambda: ["fake drift line"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CL.main()
    assert rc == 1
    assert "FAIL" in buf.getvalue()


def test_misplaced_set_hash_matches_the_real_misplaced_entries():
    import hashlib

    canon = CL._canon()
    sections = CL.index_draft()
    bad, unknown, _, _ = CL.misplaced(canon, sections)
    got = hashlib.sha256(",".join(sorted(bad + unknown)).encode()).hexdigest()
    assert got == CL.MISPLACED_SET_HASH


def test_a_same_count_misplaced_swap_is_caught_by_the_hash_even_though_the_count_ratchet_would_miss_it(
    monkeypatch,
):
    """Regression test: closes the same swap-exploit shape already found and fixed in
    canonical.py's own ratchets. N_MISPLACED is exact-equality on COUNT alone, so
    replacing the entire MISPLACED+unknown set with a different, same-size set of entries -- one
    real drift fixed, a different unreviewed one appearing in the same edit -- would satisfy
    N_MISPLACED while smuggling an unreviewed regression past this gate. MISPLACED_SET_HASH must
    catch it."""
    fake_bad = ["totally.different.key = 1 is not in S1 — declared by 'fake' (producer: fake)"]
    monkeypatch.setattr(CL, "misplaced", lambda canon, sections: (fake_bad, [], 0, 0))
    # match N_MISPLACED to this fake entry's own count, so the count ratchet alone is satisfied --
    # isolating the hash ratchet as the only thing that can still catch the substitution.
    monkeypatch.setattr(CL, "N_MISPLACED", len(fake_bad))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CL.main()
    assert rc == 1
    assert "FAIL" in buf.getvalue()
    assert "MISPLACED_SET_HASH" in buf.getvalue()
