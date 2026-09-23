"""make_figures.py::_canon() and claims.py::_canon() are two independent memoized loaders of the
same paper/canonical.json, and they currently disagree on one thing: claims.py's version drops
every key starting with "_" (i.e. excludes "_meta"), make_figures.py's doesn't. This is inert
today because make_figures.py only ever accesses _canon() by specific named key (via cval()),
never iterates the full dict -- but it would silently bite the moment anyone adds a full-key
iteration/dump to make_figures.py. This test codifies today's one intentional difference instead
of relying on nobody ever adding that iteration.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "gates"))

import make_figures as MF
import claims as C


def test_both_canon_loaders_agree_on_every_key_except_underscore_prefixed():
    mf_keys = set(MF._canon().keys())
    claims_keys = set(C._canon().keys())

    underscore_keys = {k for k in mf_keys if k.startswith("_")}
    assert underscore_keys, (
        "expected at least one underscore-prefixed key (e.g. _meta) in the real canonical.json -- "
        "if this is empty, the one documented difference between the two loaders is currently "
        "untested against real data and this test should be revisited"
    )

    assert claims_keys == (mf_keys - underscore_keys), (
        "claims.py's _canon() is documented to drop exactly the underscore-prefixed keys that "
        "make_figures.py's _canon() keeps -- if this assertion fails, either a NEW kind of "
        "disagreement has appeared (a real bug) or claims.py's filter no longer matches "
        "make_figures.py's key set for a reason beyond the underscore convention (also worth "
        "investigating before assuming it's fine)"
    )


def test_both_canon_loaders_agree_on_the_value_for_every_shared_key():
    mf_canon = MF._canon()
    claims_canon = C._canon()
    for k in claims_canon:
        assert claims_canon[k] == mf_canon[k], f"canon disagreement on key {k!r}"
