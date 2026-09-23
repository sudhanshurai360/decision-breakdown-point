"""Regression test: differential_test.py's
dict-shaped comparisons (`fragility()`, `decision_breakdown()` when dict-shaped, etc.) used
`_same(ra, rb)`, which short-circuits at `a.keys() == b.keys()` -- so any schema drift between the
FROZEN pipeline copy and the redesigned package copy (extra/renamed fields from the package's
v1.0 redesign, enumerated in differential_test.py's own `KNOWN_SCHEMA_DRIFT` registry) made the
WHOLE dict report as a mismatch, with no indication of which shared field, if any, actually
disagreed in VALUE. Investigated directly: field-by-field over 100 panels, every field `fragility()` shares
between the two copies agrees except `fragile`, which disagrees 100/100 -- not a regression, but
the package's v1.0 redesign deliberately retiring that field (returns None without an explicit
`benchmark=`) after an independent review found it measured INVERTED under the exchangeable null
(see the installed package's fragility.py, its `fragile =`
assignment and its inline comment). The gate previously failed on this every single run
(permanently red) and pointed at a non-existent `MERGED_DECISIONS.md`.

Fixed with a `KNOWN_DIVERGENT` registry (mirroring acceptance.py's `BLOCKED`) and a field-level
diff (`_dict_field_diff`) that separates: schema drift (extra/missing keys, informational only),
known-divergent shared fields (disclosed, does not fail the gate, but is always printed -- not
hidden behind --verbose), and genuinely live mismatches (fails the gate, same as before).
"""
import hashlib
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))
import differential_test as DT  # noqa: E402

def _package_importable():
    try:
        import selection_fragility._internals  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not DT.PIPELINE_COPY.exists() or not _package_importable(),
    reason="fragility.py copy or the installed selection-fragility package not available",
)


def test_clean_run_passes_despite_known_divergent_field():
    """The exact scenario this fix addresses: fragility's `fragile` field disagrees on every
    panel (a real, disclosed divergence), but the gate must still PASS overall since every OTHER
    shared field agrees."""
    r = subprocess.run([sys.executable, str(DT.HERE / "differential_test.py"), "--panels", "20"],
                        cwd=DT.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"expected PASS:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout
    assert "known-divergent  fragility.fragile" in r.stdout
    assert "LIVE MISMATCH" not in r.stdout


def test_fragility_field_diff_isolates_known_divergent_field():
    """Directly exercises _dict_field_diff on real fragility() output from both copies -- confirms
    `fragile` is classified as known-divergent (not live) and every other shared field agrees."""
    pipe = DT._load(DT.PIPELINE_COPY, "_diff_pipeline_fragility_test")
    pkg = DT._load_package_copy()
    import numpy as np
    rng = np.random.default_rng(20260824)
    L, w = DT._panels(1, rng)[0]
    kw = DT.SEEDED["fragility"]
    ra, rb = pipe.fragility(L, w, **kw), pkg.fragility(L, w, **kw)

    live, div, extra_a, extra_b = DT._dict_field_diff("fragility", ra, rb)

    assert live == [], f"unexpected live mismatch(es): {live}"
    assert "fragile" in div


def test_unexpected_field_mismatch_still_fails_closed(monkeypatch):
    """The other half of the fix: a field NOT in KNOWN_DIVERGENT that starts disagreeing must
    still fail the gate -- KNOWN_DIVERGENT is a narrow, named exemption, not a way to silence
    fragility() comparisons generally."""
    pipe = DT._load(DT.PIPELINE_COPY, "_diff_pipeline_fragility_test2")
    pkg = DT._load_package_copy()

    orig_fragility = pkg.fragility

    def poisoned(*args, **kwargs):
        r = dict(orig_fragility(*args, **kwargs))
        r["k_star"] = -999  # not in KNOWN_DIVERGENT["fragility"]
        return r

    monkeypatch.setattr(pkg, "fragility", poisoned)
    monkeypatch.setattr(DT, "_load", lambda path, name: pipe)
    monkeypatch.setattr(DT, "_load_package_copy", lambda: pkg)
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])

    rc = DT.main()
    assert rc == 1


def test_known_divergent_entries_have_a_real_shared_field():
    """Guards KNOWN_DIVERGENT itself against drift: every key it names for `fragility` must
    actually be a field BOTH copies emit today (not a stale entry naming a field one side already
    renamed away, which would silently stop exempting anything and just look like it still does)."""
    pipe = DT._load(DT.PIPELINE_COPY, "_diff_pipeline_fragility_test3")
    pkg = DT._load_package_copy()
    import numpy as np
    rng = np.random.default_rng(1)
    L, w = DT._panels(1, rng)[0]
    kw = DT.SEEDED["fragility"]
    ra, rb = pipe.fragility(L, w, **kw), pkg.fragility(L, w, **kw)

    for fn, fields in DT.KNOWN_DIVERGENT.items():
        assert fn == "fragility", f"test only covers the fragility entry, update it for {fn}"
        for field in fields:
            assert field in ra and field in rb, (
                f"KNOWN_DIVERGENT['fragility']['{field}'] names a field not emitted by both "
                f"copies anymore -- update or remove this entry"
            )


def test_known_divergent_ratchet_matches_and_catches_an_unjustified_addition(monkeypatch):
    """The KNOWN_DIVERGENT fix was found to lack acceptance.py's own
    anti-abuse safeguard: an exact-equality ratchet, so a future edit could quietly add a second,
    unjustified exemption (hiding a real regression as 'known-divergent') with nothing to catch
    it. N_KNOWN_DIVERGENT_FIELDS must match the real count today, AND adding an entry without
    bumping it must fail the gate."""
    n_known = sum(len(fields) for fields in DT.KNOWN_DIVERGENT.values())
    assert n_known == DT.N_KNOWN_DIVERGENT_FIELDS

    monkeypatch.setitem(DT.KNOWN_DIVERGENT["fragility"], "k_star", "unjustified test addition")
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])
    rc = DT.main()
    assert rc == 1


# ---- K=1 input shape -------------------------------------

def test_k1_comparison_is_exercised_and_known_divergent_not_silently_uncovered():
    """CONFIRMED GAP CLOSED: _panels() never drew K=1 (rng.integers(2, 8) floors at 2), so this
    input shape was never exercised across any of the ~3000 comparisons despite the two copies
    genuinely diverging there (the package's v1.0 K<2 guard vs. the frozen pipeline copy's lack of
    one). The dedicated K=1 block must actually run, correctly classify the divergence as
    known-divergent (not error, not silently uncovered), and the gate must still PASS."""
    r = subprocess.run([sys.executable, str(DT.HERE / "differential_test.py"), "--panels", "10"],
                        cwd=DT.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"expected PASS:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout
    for fn in ("pooled_winner", "winner_stability", "decision_breakdown", "surprise_concentration"):
        assert f"known-divergent  K=1.{fn}: 20 of 20" in r.stdout, (
            f"K=1 comparison for {fn} was not exercised or not classified as known-divergent"
        )
    assert "ERROR     K=1 comparison" not in r.stdout


def test_k1_comparison_would_fail_closed_on_the_opposite_asymmetry(monkeypatch):
    """The other half of the fix: if the FROZEN pipeline copy started raising on K=1 while the
    package didn't (the opposite of the documented, expected asymmetry), that must surface as a
    real error, not be silently absorbed as if it were the known, disclosed direction."""

    class _FlippedPipe:
        """Wraps the real pipeline copy but makes every K=1-relevant function raise -- simulating
        the pipeline copy unexpectedly GAINING a K<2 guard, the reverse of today's known gap."""
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            if name in ("pooled_winner", "winner_stability", "decision_breakdown",
                        "surprise_concentration"):
                def _raises(*a, **kw):
                    raise ValueError("simulated: pipeline copy now also rejects K<2")
                return _raises
            return getattr(self._real, name)

    real_pipe = DT._load(DT.PIPELINE_COPY, "_diff_pipeline_fragility_k1flip_test")
    monkeypatch.setattr(DT, "_load", lambda *a, **kw: _FlippedPipe(real_pipe))
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])
    rc = DT.main()
    assert rc == 1, "both sides raising on K=1 must not be silently treated as the documented divergence"


# ---- coverage ratchets -------------------------------------------

def test_shared_functions_ratchet_matches_today():
    """N_SHARED_FUNCTIONS must match what this gate actually compares today -- the same discipline
    N_KNOWN_DIVERGENT_FIELDS already gets above, applied to coverage itself rather than to
    documented divergence."""
    r = subprocess.run([sys.executable, str(DT.HERE / "differential_test.py"), "--panels", "5"],
                        cwd=DT.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"expected PASS:\n{r.stdout}\n{r.stderr}"
    assert "RATCHET INCONSISTENT" not in r.stdout
    assert f"functions compared : {DT.N_SHARED_FUNCTIONS}" in r.stdout


def test_shared_functions_ratchet_catches_a_silently_dropped_function(monkeypatch):
    """THE CONFIRMED GAP CLOSED: before this fix, a function silently disappearing from either
    fragility.py copy (rename, typo, dropped export) was only reported as 'NOT in both copies' --
    printed, never failing the gate. Simulates that exact scenario by dropping one CASES entry
    (as if `pooled_winner` had been renamed on one side) and confirms the gate now fails closed
    instead of passing with quietly reduced coverage."""
    monkeypatch.setattr(DT, "CASES", [c for c in DT.CASES if c[0] != "pooled_winner"])
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])
    rc = DT.main()
    assert rc == 1, "a silently-dropped shared function must fail the gate, not just print 'NOT in both copies'"


def test_every_public_function_in_fragility_py_is_named_somewhere_in_differential_test():
    """N_SHARED_FUNCTIONS is a ratchet
    on the COUNT of comparisons that ran, but it never checked that count against fragility.py's
    OWN actual public-function list -- a brand-new public function added identically to both the
    pipeline and package copies, but never added to CASES or special-cased in main() (the way
    breakdown_number/exchangeable_benchmark are), would be completely invisible: N_SHARED_FUNCTIONS
    stays put, the gate keeps passing. The exact blind-spot shape metric_consistency.py's
    COVERED list had before its own fix (tests/gates/test_metric_consistency_uncovered_lines.py::
    test_every_module_with_a_metrics_dict_is_either_covered_or_disclosed_uncovered).
    This is the introspection-based completeness check that closes it: every top-level, non-
    underscore-prefixed function actually defined in code/instrument/fragility.py must be named
    literally somewhere in differential_test.py's own source (as a CASES entry, or via a
    hasattr(pipe, "name")/pipe.name-style special-cased comparison like breakdown_number's)."""
    import ast

    fragility_src = (DT.ROOT / "code" / "instrument" / "fragility.py").read_text()
    tree = ast.parse(fragility_src)
    public_funcs = sorted(
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    )
    assert public_funcs, "sanity check: fragility.py should define at least one public function"

    dt_src = (DT.ROOT / "code" / "gates" / "differential_test.py").read_text()
    missing = [name for name in public_funcs if name not in dt_src]
    assert not missing, (
        f"{missing} are public functions in fragility.py but are never referenced anywhere in "
        f"differential_test.py -- they are invisible to this gate's comparison entirely. Add them "
        f"to CASES, or special-case them in main() the way breakdown_number/exchangeable_benchmark "
        f"are, and bump N_SHARED_FUNCTIONS."
    )


def test_schema_drift_was_silently_informational_now_fails_closed_on_a_field_rename(monkeypatch):
    """THE CONFIRMED GAP CLOSED: schema_drift (extra/missing dict keys between the two fragility.py
    copies) was collected every run but only ever PRINTED under --verbose, and never affected the
    exit code -- a gate that collects a discrepancy and then exits 0 on it decays silently, which
    is the same shape as the whole-dict short-circuit this file's module docstring describes, just
    at the field level within a dict-shaped output. The reproduction: the package copy's
    fragility() renames `k_star` -> `k_star_v2`. That is NOT in KNOWN_SCHEMA_DRIFT, so it must now
    fail the gate (previously it was silently absorbed into schema_drift and the gate still
    PASSed)."""
    pipe = DT._load(DT.PIPELINE_COPY, "_diff_pipeline_fragility_schemadrift_test")
    pkg = DT._load_package_copy()

    orig_fragility = pkg.fragility

    def renamed(*args, **kwargs):
        r = dict(orig_fragility(*args, **kwargs))
        r["k_star_v2"] = r.pop("k_star")
        return r

    monkeypatch.setattr(pkg, "fragility", renamed)
    monkeypatch.setattr(DT, "_load", lambda path, name: pipe)
    monkeypatch.setattr(DT, "_load_package_copy", lambda: pkg)
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])

    rc = DT.main()
    assert rc == 1, (
        "a field rename producing undocumented schema drift (extra/missing keys not in "
        "KNOWN_SCHEMA_DRIFT) must fail the gate, not be silently absorbed as informational-only "
        "schema_drift"
    )


def test_schema_drift_printed_unconditionally_not_gated_behind_verbose():
    """The other half of the fix: schema drift must be visible in a default (non---verbose) run,
    not just under --verbose -- a reader who never passes --verbose must still be able to see it."""
    r = subprocess.run([sys.executable, str(DT.HERE / "differential_test.py"), "--panels", "5"],
                        cwd=DT.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"expected PASS:\n{r.stdout}\n{r.stderr}"
    assert "schema drift  fragility:" in r.stdout, (
        "schema drift must print in a default run, not only under --verbose"
    )


def test_known_schema_drift_ratchet_matches_and_catches_an_unjustified_addition(monkeypatch):
    """Mirrors test_known_divergent_ratchet_matches_and_catches_an_unjustified_addition above, for
    the new KNOWN_SCHEMA_DRIFT allowlist: N_KNOWN_SCHEMA_DRIFT_KEYS must match the real count
    today, and silently widening the allowlist without bumping it must fail the gate."""
    n_known = sum(
        len(v.get("pipeline_only", ())) + len(v.get("package_only", ()))
        for v in DT.KNOWN_SCHEMA_DRIFT.values()
    )
    assert n_known == DT.N_KNOWN_SCHEMA_DRIFT_KEYS

    # frozenset has no in-place add; swap in a widened copy instead.
    monkeypatch.setitem(
        DT.KNOWN_SCHEMA_DRIFT["fragility"],
        "pipeline_only",
        DT.KNOWN_SCHEMA_DRIFT["fragility"]["pipeline_only"] | {"unjustified_test_field"},
    )
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])
    rc = DT.main()
    assert rc == 1


# ---- same-count swap exploits ----------------------------------------------
# Closes the same swap-exploit shape live-demonstrated in canonical.py's own ratchets: all
# 3 ratchets above are exact-equality on COUNT alone, so a same-count substitution of WHICH
# (function, field)/function/key is covered would satisfy the scalar ratchet while genuinely
# changing what's exempted or compared. The 3 tests below confirm the new *_HASH constants hash the
# real committed state correctly, then demonstrate the swap is caught by the hash even in a
# scenario engineered so the pre-existing count ratchet alone would stay satisfied.

def test_all_three_keyset_hashes_match_the_real_committed_state():
    got_kd = hashlib.sha256(",".join(sorted(
        f"{fn}.{field}" for fn, fields in DT.KNOWN_DIVERGENT.items() for field in fields
    )).encode()).hexdigest()
    assert got_kd == DT.KNOWN_DIVERGENT_HASH

    got_sd = hashlib.sha256(",".join(sorted(
        f"{name}.{side}:{key}"
        for name, v in DT.KNOWN_SCHEMA_DRIFT.items()
        for side in ("pipeline_only", "package_only")
        for key in v.get(side, ())
    )).encode()).hexdigest()
    assert got_sd == DT.SCHEMA_DRIFT_KEYS_HASH


def test_a_same_count_known_divergent_swap_is_caught_by_the_hash_even_though_the_count_ratchet_would_miss_it(
    monkeypatch,
):
    """Removes one real exemption (fragility.pooled_runner_up) and adds an unrelated one
    (fragility.k_star) in its place -- same total count (N_KNOWN_DIVERGENT_FIELDS unaffected), but
    a different exempted field. KNOWN_DIVERGENT_HASH must catch it."""
    new_fragility = dict(DT.KNOWN_DIVERGENT["fragility"])
    del new_fragility["pooled_runner_up"]
    new_fragility["k_star"] = "unrelated test substitution, not a real exemption"
    monkeypatch.setitem(DT.KNOWN_DIVERGENT, "fragility", new_fragility)

    n_known = sum(len(fields) for fields in DT.KNOWN_DIVERGENT.values())
    assert n_known == DT.N_KNOWN_DIVERGENT_FIELDS, "fixture assumption: swap must be same-count"

    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "5"])
    rc = DT.main()
    assert rc == 1


def test_a_same_count_shared_functions_swap_is_caught_by_the_hash_even_though_the_count_ratchet_would_miss_it(
    monkeypatch,
):
    """Drops `pooled_winner` from CASES (11 of 12 functions compared) and shows that state passes
    once N_SHARED_FUNCTIONS/SHARED_FUNCTIONS_HASH are aligned to it, then drops a DIFFERENT
    function (`condorcet_winner`) instead -- same count (11), different membership. The count
    ratchet alone cannot distinguish these two scenarios; the hash must."""
    monkeypatch.setattr(sys, "argv", ["differential_test.py", "--panels", "3"])
    monkeypatch.setattr(DT, "N_SHARED_FUNCTIONS", DT.N_SHARED_FUNCTIONS - 1)

    original_cases = list(DT.CASES)  # captured before any patching below -- see the condorcet_winner
    # variant further down, which must derive from this same original list, not an already-patched one
    cases_drop_pooled_winner = [c for c in original_cases if c[0] != "pooled_winner"]
    monkeypatch.setattr(DT, "CASES", cases_drop_pooled_winner)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc0 = DT.main()
    assert rc0 == 1, "fixture assumption: must fail on the (not-yet-aligned) hash alone"
    assert "RATCHET INCONSISTENT — SHARED_FUNCTIONS_HASH" in buf.getvalue()
    assert "RATCHET INCONSISTENT — N_SHARED_FUNCTIONS" not in buf.getvalue(), (
        "the count ratchet must already be satisfied at this point -- only the hash should differ"
    )
    line = next(l for l in buf.getvalue().splitlines() if l.startswith("  functions compared"))
    shared_v1 = line.split("(", 1)[1].rstrip(")").split(", ")
    hash_v1 = hashlib.sha256(",".join(sorted(shared_v1)).encode()).hexdigest()
    monkeypatch.setattr(DT, "SHARED_FUNCTIONS_HASH", hash_v1)

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc1 = DT.main()
    assert rc1 == 0, f"fixture assumption: aligned hash must now cleanly pass:\n{buf2.getvalue()}"

    cases_drop_condorcet_winner = [c for c in original_cases if c[0] != "condorcet_winner"]
    monkeypatch.setattr(DT, "CASES", cases_drop_condorcet_winner)
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        rc2 = DT.main()
    out3 = buf3.getvalue()
    assert rc2 == 1
    assert "RATCHET INCONSISTENT — SHARED_FUNCTIONS_HASH" in out3
    assert "RATCHET INCONSISTENT — N_SHARED_FUNCTIONS" not in out3, (
        "both scenarios compare exactly 11 functions -- the count ratchet must stay silent; only "
        "the hash may catch this swap"
    )
