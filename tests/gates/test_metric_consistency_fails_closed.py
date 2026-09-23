"""Regression test: the same
fail-open shape acceptance.py was already fixed for -- a shrinking comparable-set has
no floor, so the check silently returns "everything agrees" instead of failing loud.

metric_consistency.py's `check_synthetic()`/`check_real_panel()` both only compare a metric name
when >=2 producers expose it. If COVERED shrinks to 1 (or 0) entries -- a COVERED-list edit, or a
producer's METRICS mapping losing entries -- every metric name drops below that floor, `bad` stays
`[]`, and main() prints "every shared metric name computes identically in every covered producer"
and returns 0 (PASS), even though zero real comparisons happened. Confirmed live before fixing (not
just reasoned about): monkeypatching COVERED to a single entry reproduced this exactly.

Fixed by adding an exact-equality ratchet (N_COVERED_PRODUCERS, N_COMPARED_NAMES) mirroring
acceptance.py's own N_COMPARABLE/N_BLOCKED discipline -- N_COMPARED_NAMES specifically counts metric
names with >=2 producers, the real "did a comparison happen" signal, not just "how many names exist
in total" (which a single-producer COVERED would still report as nonzero)."""
import io
from contextlib import redirect_stdout

import metric_consistency as MCON


def _run_main_with(monkeypatch, covered):
    monkeypatch.setattr(MCON, "COVERED", covered)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = MCON.main()
    return rc, buf.getvalue()


def test_clean_run_still_passes():
    """The real, unmodified repo state (run as a subprocess, exercising the actual script on disk)
    must still pass -- confirms the fix didn't break the legitimate case."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(MCON.__file__)], capture_output=True, text=True)
    assert r.returncode == 0
    assert "PASS" in r.stdout
    assert "ratchet ok" in r.stdout


def test_shrunk_to_one_producer_fails_closed(monkeypatch):
    """The exact scenario found: COVERED shrunk to a single entry. Before the fix this
    silently printed 'every shared metric name computes identically' and returned 0. After the fix
    it must FAIL, and specifically via the ratchet (not by coincidentally finding a fake
    inconsistency), isolating the ratchet as the thing that caught it."""
    rc, out = _run_main_with(monkeypatch, [MCON.COVERED[0]])
    assert rc == 1
    assert "FAIL" in out
    assert "RATCHET INCONSISTENT" in out
    assert "0 inconsistency(ies)" in out  # not caught by coincidence -- bad is genuinely empty


def test_shrunk_to_zero_producers_fails_closed(monkeypatch):
    """The even more degenerate case: COVERED emptied out entirely. Must also fail, not raise an
    unrelated exception and not silently pass."""
    rc, out = _run_main_with(monkeypatch, [])
    assert rc == 1
    assert "FAIL" in out
    assert "RATCHET INCONSISTENT" in out


def test_a_metric_name_losing_coverage_fails_closed(monkeypatch):
    """A subtler shrink than losing a whole producer: 'rmsse' is dropped from every producer but
    one (COVERED itself is unchanged, and 'mase' still has full coverage), so 'rmsse' now has fewer
    than 2 producers exposing it -- below check_real_panel()'s own `len(producers) < 2: continue`
    floor -- even though N_COVERED_PRODUCERS alone would not catch this (len(metrics) is unchanged).
    N_COMPARED_NAMES must catch it."""
    real_metrics = MCON._load_metrics()

    def fake_load_metrics():
        out = {}
        for i, (mod, M) in enumerate(real_metrics.items()):
            out[mod] = dict(M) if i == 0 else {k: v for k, v in M.items() if k != "rmsse"}
        return out

    monkeypatch.setattr(MCON, "_load_metrics", fake_load_metrics)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = MCON.main()
    assert rc == 1
    assert "RATCHET INCONSISTENT — N_COMPARED_NAMES" in buf.getvalue()


def test_ratchet_matches_real_repo_state():
    """Confirms the current real repo state satisfies both ratchets, so the constants don't
    silently drift out of sync with a legitimate change to COVERED or a producer's METRICS mapping
    without this test flagging it."""
    metrics = MCON._load_metrics()
    all_names = sorted({n for M in metrics.values() for n in M})
    n_compared = sum(1 for n in all_names if sum(1 for M in metrics.values() if n in M) >= 2)
    assert len(metrics) == MCON.N_COVERED_PRODUCERS, (
        f"len(COVERED) drifted to {len(metrics)} -- if legitimate, bump N_COVERED_PRODUCERS"
    )
    assert n_compared == MCON.N_COMPARED_NAMES, (
        f"names actually compared drifted to {n_compared} -- if legitimate, bump N_COMPARED_NAMES"
    )


def test_keyset_hashes_match_the_real_committed_state():
    import hashlib
    metrics = MCON._load_metrics()
    all_names = sorted({n for M in metrics.values() for n in M})
    compared_names = sorted(n for n in all_names if sum(1 for M in metrics.values() if n in M) >= 2)
    assert hashlib.sha256(",".join(sorted(MCON.COVERED)).encode()).hexdigest() == MCON.COVERED_PRODUCERS_HASH
    assert hashlib.sha256(",".join(compared_names).encode()).hexdigest() == MCON.COMPARED_NAMES_HASH


def test_a_same_count_covered_producers_swap_is_caught_by_the_hash_even_though_the_count_ratchet_would_miss_it(
    monkeypatch,
):
    """Regression (focused fix, closing the same swap-exploit shape found in canonical.py's own
    ratchets): N_COVERED_PRODUCERS is exact-equality on COUNT alone,
    so replacing COVERED with an entirely different, same-size list of producer names would satisfy
    N_COVERED_PRODUCERS while genuinely changing what's verified. COVERED_PRODUCERS_HASH must catch
    it even when the count ratchet cannot."""
    import hashlib

    fake_mean = lambda s: s.mean()  # noqa: E731 -- trivial test aggregator, matches real METRICS shape
    fake_metrics = {name: {"mase": ("sae", fake_mean)} for name in ("fake_a", "fake_b")}
    monkeypatch.setattr(MCON, "_load_metrics", lambda: fake_metrics)
    monkeypatch.setattr(MCON, "N_COMPARED_NAMES", 1)
    monkeypatch.setattr(MCON, "COMPARED_NAMES_HASH", hashlib.sha256(b"mase").hexdigest())
    # same COUNT as N_COVERED_PRODUCERS (2), but COVERED itself names entirely different producers
    monkeypatch.setattr(MCON, "COVERED", ["totally", "different"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = MCON.main()
    out = buf.getvalue()
    assert rc == 1
    assert "RATCHET INCONSISTENT — COVERED_PRODUCERS_HASH" in out
    assert "RATCHET INCONSISTENT — N_COVERED_PRODUCERS" not in out, (
        f"the count ratchet must stay silent ({MCON.N_COVERED_PRODUCERS} == "
        f"{MCON.N_COVERED_PRODUCERS}) -- only the hash may catch this swap"
    )
