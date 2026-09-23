"""Regression test: metric_consistency.py's
disclosed UNCOVERED blind-spot registry -- files it names as doing inline sae/msse aggregation
without a named METRICS mapping -- had all 5 of its recorded line-number references rotted (every
one pointed at unrelated code). The registry's own comment said "verify by reading the line before
trusting the note"; nobody had. Corrected once, then rotted twice more from unrelated edits to
mcs_test.py (the RMSSE fix, then the degenerate-SE fix) -- three rots from three genuinely
unrelated changes made clear the exact-line design itself was too fragile for a file under active
development, not just unlucky timing.

UNCOVERED therefore now matches by CONTENT anywhere in the named file (`"file.py": note`,
no `:LINE` suffix), not a pinned line number -- immune to line-number drift from unrelated edits
elsewhere in the same file, while still failing loudly if the described code is ever actually
removed.

A bare `"sae" in content or "msse"
in content` check would be vacuous for any file whose DOCSTRING happens to mention "sae"/"msse" in
prose (several of these files do, describing what the columns mean) -- confirmed by direct
experiment: deleting mcs_test.py's real `gm["sae"].mean()`/`gm["msse"].mean()` aggregation code, or
literally FLIPPING combination_benchmark.py's real aggregation from sae to msse (making its "no
msse aggregation" note false), both left the bare-substring check passing anyway, because the
docstrings independently contain the words. So this checks the bracket-quoted, code-only form
(`["sae"]`/`["msse"]`, which appears only in real aggregation code and never in any docstring --
once per file, except mcs_robustness.py, which has two genuine `["sae"]` aggregation sites)
instead of the bare word, plus an explicit `["msse"]`-ABSENT check for the two files whose note
specifically claims "sae only, no msse".
"""
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PIPELINE = ROOT / "code" / "pipeline"

import sys
sys.path.insert(0, str(ROOT / "code" / "gates"))
import metric_consistency as MC  # noqa: E402

pytestmark = pytest.mark.skipif(not PIPELINE.exists(), reason="code/pipeline not available")

# Files whose UNCOVERED note specifically claims "sae only, no msse aggregation" -- these get an
# explicit absence check for `["msse"]`, not just a presence check for `["sae"]`.
_SAE_ONLY = {"combination_benchmark.py", "mcs_power.py"}


def test_uncovered_registry_is_not_empty():
    """Non-vacuity guard for the parametrize below: an empty MC.UNCOVERED would generate zero test
    cases, and a file with zero cases reports green -- indistinguishable from "every entry passes"."""
    assert MC.UNCOVERED, (
        "metric_consistency.UNCOVERED is empty -- the parametrized check below would silently "
        "generate no cases at all"
    )
    missing = _SAE_ONLY - set(MC.UNCOVERED)
    assert missing == set(), (
        f"_SAE_ONLY names {sorted(missing)}, which UNCOVERED no longer registers -- the stricter "
        f"no-msse assertion for those files would never run"
    )


@pytest.mark.parametrize("key,note", list(MC.UNCOVERED.items()))
def test_uncovered_entry_content_matches_real_file(key, note):
    assert ":" not in key, f"{key}: UNCOVERED keys should be bare filenames now, no :LINE suffix"
    f = PIPELINE / key
    assert f.exists(), f"{key} names a file that doesn't exist: {f}"
    content = f.read_text()

    assert '["sae"]' in content or '["msse"]' in content, (
        f"{key} has no bracket-quoted [\"sae\"]/[\"msse\"] reference anywhere in the file -- "
        f"UNCOVERED has rotted, the described inline aggregation may have been removed or renamed "
        f"(a bare substring check would be vacuous here -- {key}'s own docstring may still "
        f"mention the words even if the code changed, see this test's own module docstring)"
    )
    if key in _SAE_ONLY:
        assert '["msse"]' not in content, (
            f"{key}'s note specifically claims 'sae only, no msse aggregation', but the code now "
            f"contains a bracket-quoted [\"msse\"] reference -- the note is stale, this producer "
            f"may need to move to metric_consistency.py's COVERED list instead"
        )


def test_metric_consistency_gate_still_passes():
    rc = MC.main()
    assert rc == 0


def test_every_module_with_a_metrics_dict_is_either_covered_or_disclosed_uncovered():
    """election_null_ar1.py had defined a
    {name: (column, aggregator)} METRICS dict since it was written, and is a live producer, but was
    in neither COVERED nor UNCOVERED -- completely invisible to this gate. Fixed by adding it to
    COVERED. This test locks in the general property so a future module with the same shape can't
    silently repeat the gap: every MODULE-LEVEL `METRICS = {...}` dict whose values are 2-tuples
    (the {name: (column, aggregator)} shape this gate actually compares) must appear in COVERED.

    NOTE: `METRICS` is also used, unrelatedly, in m4_margins.py and m4_submissions_audit.py as a
    plain list of metric-name strings (`["sMAPE", "MAPE", "MASE", "RMSSE"]`) -- a genuinely
    different concept (M4 leaderboard column labels, not a column/aggregator mapping) that happens
    to share the variable name. The dict-of-2-tuples shape check below excludes those on purpose;
    it is not testing "any module-level name METRICS", only this gate's actual comparable shape."""
    import ast

    ALLOWED_SILENT = set()  # no known exceptions today
    pipeline_dir = ROOT / "code" / "pipeline"
    offenders = []
    for f in sorted(pipeline_dir.glob("*.py")):
        if f.stem in MC.COVERED or f.stem in ALLOWED_SILENT:
            continue
        tree = ast.parse(f.read_text())
        for node in tree.body:  # module level only, not inside functions
            if not (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "METRICS" for t in node.targets)):
                continue
            if not isinstance(node.value, ast.Dict):
                continue  # the unrelated plain-list METRICS convention -- not this gate's shape
            if all(isinstance(v, ast.Tuple) and len(v.elts) == 2 for v in node.value.values):
                offenders.append(f.stem)
    assert not offenders, (
        f"{offenders} define a module-level METRICS = {{name: (column, aggregator)}} dict but are "
        f"not in metric_consistency.py's COVERED list -- add them or, if deliberately excluded, "
        f"update this test's ALLOWED_SILENT set with a one-line reason."
    )
