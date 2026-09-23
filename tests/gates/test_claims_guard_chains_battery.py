"""claims.py::renderings() and percent_renderings() each carry an independent isinstance-guard chain
(None, bool, dict, list, str/int, ...) that must agree on WHICH types are safe to fall through to
the final numeric-format code, even though the two chains are two separately-maintained blocks of
code with no structural link between them. They have already drifted twice in practice: one pass
added a None guard to both in the same sitting (after a live crash), another pass added a dict
guard to both in the same sitting (found pre-emptively). A full merge is
not worth the risk (the two functions intentionally return different sets for the same numeric
input), so instead this is a PIN: a battery of edge-case values run through both functions, so the
NEXT type that needs a guard fails this test loudly instead of waiting for a live canonical.json
crash -- how both prior instances were actually found.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "gates"))
import claims as C

EDGE_CASES = [
    None,
    True,
    False,
    {},
    {"a": 1, "b": 2.5},
    {"nested": {"x": None}},
    [],
    [1, 2],
    [1, 2, 3],
    [None, True, {"a": 1}],
    "",
    "a string",
    0,
    -5,
    12345,
    0.0,
    -0.5,
    3.14159,
    1e-9,
    float("nan"),
]


def test_renderings_never_raises_on_any_edge_case():
    for v in EDGE_CASES:
        C.renderings(v)  # must not raise


def test_percent_renderings_never_raises_on_any_edge_case():
    for v in EDGE_CASES:
        C.percent_renderings(v)  # must not raise


def test_renderings_returns_a_set_of_strings_for_every_edge_case():
    for v in EDGE_CASES:
        out = C.renderings(v)
        assert isinstance(out, set)
        assert all(isinstance(x, str) for x in out)


def test_percent_renderings_returns_a_set_of_strings_for_every_edge_case():
    for v in EDGE_CASES:
        out = C.percent_renderings(v)
        assert isinstance(out, set)
        assert all(isinstance(x, str) for x in out)


def test_both_functions_agree_on_which_types_are_categorically_empty():
    """None, bool, and empty dict/list carry no defensible rendering in EITHER function -- if one
    function starts rendering something for one of these types and the other doesn't, that's
    exactly the kind of silent guard-chain drift this test exists to catch."""
    categorically_empty = [None, True, False, {}, []]
    for v in categorically_empty:
        assert C.renderings(v) == set(), f"renderings({v!r}) should be empty, got {C.renderings(v)!r}"
        assert C.percent_renderings(v) == set(), (
            f"percent_renderings({v!r}) should be empty, got {C.percent_renderings(v)!r}"
        )
