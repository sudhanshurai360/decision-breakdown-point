"""The prose an emit() call passes must equal the prose shipped in canonical.json.

canonical.py's digest deliberately excludes the prose-only fields (producer, spec, cites,
cross_check, provenance) so that rewording a spec does not invalidate a frozen hash. The cost is a
blind spot: edit a spec= string in a producer without re-running it, and the merged object keeps
the old text forever. Every gate still passes, because no gate compares the two. That is not
hypothetical -- it is how canonical.json came to ship "2020 is the most-pivotal period in 7-9 of
24 cells" when the code, the values (lo=5, hi=7) and the manuscript all said 5-7.

This test closes most of that hole from the source side: it reads emit() calls statically and
asserts the shipped copies agree, in paper/canonical.json and in the producer's own fragment.

A plain literal is compared exactly. An f-string cannot be evaluated statically -- its
interpolations are runtime values -- so it is compared as a pattern: the literal segments must
appear, in order, with anything in the interpolation slots. That is weaker than an exact match,
but it catches the defect class, because drift is in the prose a human edited, not in the number
the producer computed. Checking only literals would leave a real hole: three f-string specs had
drifted, one attributing an argument to a section of the paper that does not make it.

What this gate does NOT see, stated plainly so a pass is not read for more than it carries:

- An emit() whose KEY is itself an f-string. Those keys cannot be resolved statically, so their
  prose is never compared. test_this_gate_still_reaches_most_of_the_shipped_prose pins how much
  is reachable, so the gap cannot widen unnoticed.
- Text appended to the very END of an f-string spec that ends in an interpolation (23 of 96 do).
  A trailing wildcard cannot tell appended prose from a longer interpolated value, and no length
  bound helps -- interpolated content here runs to ~1100 characters. Drift anywhere else in those
  strings, and anywhere at all in the other 73, is caught.
- Prose assembled some other way (a join over a computed list) is skipped, not guessed at.
"""
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CANON = ROOT / "paper" / "canonical.json"
FRAGMENTS = ROOT / "multiseries" / "results" / "canonical"

PROSE_FIELDS = ("spec", "cites", "cross_check", "provenance")


def _literal(node):
    """The node's value if it is a literal, else None (a runtime-built value we cannot check)."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _fstring_pattern(node) -> str | None:
    """A regex for an f-string: literal segments escaped and kept in order, each interpolation
    standing in as a lazy wildcard. None if the node is not an f-string."""
    if not isinstance(node, ast.JoinedStr):
        return None
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(re.escape(value.value))
        else:
            parts.append(".*?")
    return "".join(parts)


def _emitted_prose() -> tuple[dict[str, dict], dict[str, dict]]:
    """Across every emit() under code/, split by how the prose can be checked:
    ({key: {field: exact literal}}, {key: {field: f-string regex}})."""
    exact: dict[str, dict] = {}
    patterns: dict[str, dict] = {}
    for py in sorted((ROOT / "code").rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "emit"):
                continue
            if not node.args:
                continue
            key = _literal(node.args[0])
            if not isinstance(key, str):
                continue
            for kw in node.keywords:
                if kw.arg not in PROSE_FIELDS:
                    continue
                value = _literal(kw.value)
                if value is not None:
                    exact.setdefault(key, {})[kw.arg] = value
                    continue
                pattern = _fstring_pattern(kw.value)
                if pattern is not None:
                    patterns.setdefault(key, {})[kw.arg] = pattern


    return exact, patterns


def _stale(doc: dict, where: str) -> list[str]:
    """Every prose field in `doc` that disagrees with the emit() call that produced it."""
    exact, patterns = _emitted_prose()
    out = []
    for key, fields in exact.items():
        entry = doc.get(key)
        if not isinstance(entry, dict):
            continue
        for field, source_value in fields.items():
            if field in entry and entry[field] != source_value:
                out.append(f"{where}{key}.{field}\n"
                           f"      source : {source_value!r}\n"
                           f"      shipped: {entry[field]!r}")
    for key, fields in patterns.items():
        entry = doc.get(key)
        if not isinstance(entry, dict):
            continue
        for field, pattern in fields.items():
            shipped = entry.get(field)
            if isinstance(shipped, str) and not re.fullmatch(pattern, shipped, re.S):
                out.append(f"{where}{key}.{field}  (f-string)\n"
                           f"      source pattern: {pattern[:200]!r}\n"
                           f"      shipped       : {shipped[:200]!r}")
    return out


def _coverage() -> tuple[int, int]:
    """(source-comparable prose fields, prose fields shipped in canonical.json) -- so the ratchet
    below fails if a refactor quietly moves prose out of this test's reach."""
    exact, patterns = _emitted_prose()
    comparable = sum(len(v) for v in exact.values()) + sum(len(v) for v in patterns.values())
    shipped = json.loads(CANON.read_text(encoding="utf-8"))
    total = sum(1 for k, e in shipped.items()
                if not k.startswith("_") and isinstance(e, dict)
                for f in PROSE_FIELDS if e.get(f) is not None)
    return comparable, total


def test_every_emit_prose_field_matches_paper_canonical_json():
    exact, patterns = _emitted_prose()
    assert exact and patterns, ("found no literal and/or no f-string emit() prose at all -- the AST "
                                "walk is broken, not the data")

    stale = _stale(json.loads(CANON.read_text(encoding="utf-8")), "")
    assert not stale, (
        f"{len(stale)} prose field(s) in paper/canonical.json disagree with the emit() call that "
        f"produced them. The producer was edited without being re-run, so the stale text is what "
        f"ships. Re-run the producer and re-merge, or patch the shipped copy to match source -- "
        f"the digest excludes prose, so neither moves the hash:\n    " + "\n    ".join(stale))


def test_every_emit_prose_field_matches_its_producer_fragment():
    exact, patterns = _emitted_prose()
    assert exact and patterns, "found no emit() prose at all -- the AST walk is broken"

    fragments = {p: json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(FRAGMENTS.glob("*.json"))}
    assert fragments, "no canonical fragments found"

    stale = []
    for path, doc in fragments.items():
        stale += _stale(doc, f"{path.name}::")

    assert not stale, (
        f"{len(stale)} prose field(s) in multiseries/results/canonical/ disagree with the emit() "
        f"call that produced them. A fragment left stale here will be merged back into "
        f"paper/canonical.json on the next run, silently reverting a source fix:\n    "
        + "\n    ".join(stale))


def test_this_gate_still_reaches_most_of_the_shipped_prose():
    """A ratchet on the gate itself.

    An emit() whose KEY is an f-string (`f"eb.trigger_{m}"`) cannot be resolved statically, so the
    prose on those keys is invisible here -- as is prose built by anything other than a literal or
    an f-string. That gap is acceptable but must not silently widen: if a refactor converts more
    literal keys to computed ones, coverage drops and this fails, rather than the gate quietly
    checking less and less while still reporting green.
    """
    comparable, total = _coverage()
    assert total > 0, "no prose fields found in canonical.json at all"
    assert comparable >= 945, (
        f"this gate can now reach only {comparable} source-side prose fields, down from 951 "
        f"({total} prose fields ship in canonical.json; the two counts are measured on different "
        f"sides, so this is a floor on reach, not a coverage fraction). Prose moved out of reach -- "
        f"most likely an emit() key became an f-string. Restore the literal key, or extend "
        f"_emitted_prose() to resolve the computed form, rather than lowering this floor.")
