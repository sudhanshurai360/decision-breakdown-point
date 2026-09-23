"""canonical_emit.py backs
every canonical value in this project (every single one goes through emit()/flush()) but had
ZERO direct unit tests of its own logic before this file -- the reason two latent bugs
(_plain()'s dict-recursion gap and 0-d-numpy-array crash, both fixed when this test file
was added) went unnoticed for a long time despite the exact same bug SHAPE already
being caught and fixed once in claims.py. Covers: the duplicate-key guard, the
protected-kwarg-collision guard, _plain()'s full type table (including the two new guards), and
flush()'s empty-pending guard and basic write shape.
"""
import json
import math
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "instrument"))
import canonical_emit as CE


def _key():
    """A fresh, collision-free emit() key per test -- _PENDING[stage] is a module-level dict keyed
    by the CALLING FILE's stem (this test file's own name, for every call made from here), so
    reusing a literal key string across test functions would trip the real "emitted twice" guard
    for reasons unrelated to what any individual test is checking."""
    return f"test.{uuid.uuid4().hex[:12]}"


# --- _plain() ----------------------------------------------------------------------------------

def test_plain_converts_numpy_scalar_types_to_native():
    """NOTE on `type(...) is float` rather than `isinstance(..., float)`: np.float64 IS a subclass
    of Python's float, so `isinstance(np.float64(1.5), float)` is True even with NO conversion at
    all -- the isinstance form this test used to carry could not fail. (np.int64/np.bool_ are NOT
    subclasses of int/bool, so those two lines were already meaningful; they use the exact-type
    form here for consistency.)"""
    assert type(CE._plain(np.int64(5))) is int and CE._plain(np.int64(5)) == 5
    assert type(CE._plain(np.bool_(True))) is bool and CE._plain(np.bool_(True)) is True
    assert type(CE._plain(np.float64(1.5))) is float and CE._plain(np.float64(1.5)) == 1.5


def test_plain_recurses_into_list_tuple_ndarray():
    assert CE._plain([np.int64(1), np.int64(2)]) == [1, 2]
    assert CE._plain((np.int64(1), np.int64(2))) == [1, 2]
    assert CE._plain(np.array([1, 2, 3])) == [1, 2, 3]


def test_plain_recurses_into_dict():
    """FIXED -- previously fell through unconverted to `return v`."""
    out = CE._plain({"a": np.int64(3), "b": np.float64(2.5)})
    assert out == {"a": 3, "b": 2.5}
    # exact type, not isinstance: np.float64 subclasses float, so isinstance cannot detect a
    # dict value that was never converted (see test_plain_converts_numpy_scalar_types_to_native)
    assert type(out["a"]) is int and type(out["b"]) is float


def test_plain_recurses_into_nested_dict_and_list():
    out = CE._plain({"a": [np.int64(1), {"b": np.float64(2.0)}]})
    assert out == {"a": [1, {"b": 2.0}]}


def test_plain_handles_0d_ndarray():
    """FIXED -- previously raised TypeError: iteration over a 0-d array."""
    assert CE._plain(np.array(5.0)) == 5.0
    assert type(CE._plain(np.array(5.0))) is float   # exact type: np.float64 subclasses float
    assert CE._plain(np.array(3)) == 3
    assert type(CE._plain(np.array(3))) is int


def test_plain_rejects_infinity():
    with pytest.raises(AssertionError):
        CE._plain(float("inf"))
    with pytest.raises(AssertionError):
        CE._plain(float("-inf"))


def test_plain_allows_nan_as_a_first_class_value():
    """NaN is deliberately NOT rejected -- fragility.py's `concentration` field is genuinely
    NaN-valued on a documented degenerate-margin path."""
    assert math.isnan(CE._plain(float("nan")))


def test_plain_passes_through_plain_python_scalars_unchanged():
    assert CE._plain(5) == 5 and isinstance(CE._plain(5), int)
    assert CE._plain("x") == "x"
    assert CE._plain(None) is None


# --- emit() --------------------------------------------------------------------------------

def test_emit_raises_on_duplicate_key_within_the_same_stage():
    k = _key()
    CE.emit(k, 1, spec="first")
    with pytest.raises(ValueError, match="emitted twice"):
        CE.emit(k, 2, spec="second")


def test_emit_raises_on_extra_kwarg_colliding_with_a_protected_field():
    # `value`/`spec`/`cites`/`cross_check` are already named parameters of emit() itself, so
    # passing them a second time as a kwarg is caught by Python's own argument binding before
    # reaching **extra. `producer` is the one protected-field name that is NOT a named parameter
    # (it only appears in the frag dict this function builds), so it's what actually exercises
    # this function's own internal collision guard.
    k = _key()
    with pytest.raises(AssertionError, match="collide"):
        CE.emit(k, 1, spec="x", producer="not allowed")


def test_emit_records_cites_cross_check_and_extra_kwargs():
    k = _key()
    CE.emit(k, 7, spec="x", cites=["S1 body"], cross_check="foo.csv: bar", n_models=3)
    stage = CE._caller_stage()
    frag = CE._PENDING[stage][k]
    assert frag["value"] == 7
    assert frag["cites"] == ["S1 body"]
    assert frag["cross_check"] == "foo.csv: bar"
    assert frag["n_models"] == 3
    assert frag["producer"] == stage


def test_emit_defaults_cites_to_empty_list_and_cross_check_to_none():
    k = _key()
    CE.emit(k, 1, spec="x")
    stage = CE._caller_stage()
    frag = CE._PENDING[stage][k]
    assert frag["cites"] == []
    assert frag["cross_check"] is None


def test_emit_copies_cites_not_aliases_the_callers_list():
    """Regression: `cites` used to be
    stored by reference (`cites or []`), unlike every other field, which goes through _plain()'s
    defensive copy. A caller passing the SAME list object to multiple emit() calls -- confirmed
    live-exercised by kstar_persistence.py's module-level `CITE` constant across 16 emit() calls --
    would have every one of those fragment entries silently change if that shared list were later
    mutated. Two emit() calls sharing one list object, then mutating it after both calls, must NOT
    retroactively change either already-recorded "cites" field."""
    shared = ["S1 body"]
    k1, k2 = _key(), _key()
    CE.emit(k1, 1, spec="x", cites=shared)
    CE.emit(k2, 2, spec="y", cites=shared)
    shared.append("S2 body (a later, unrelated citation)")
    stage = CE._caller_stage()
    assert CE._PENDING[stage][k1]["cites"] == ["S1 body"]
    assert CE._PENDING[stage][k2]["cites"] == ["S1 body"]


# --- flush() -------------------------------------------------------------------------------

def test_flush_raises_on_nothing_pending(monkeypatch):
    """A stage reaching flush() with an EMPTY (not missing) pending dict must fail loudly rather
    than skip silently, for the reason flush()'s own RuntimeError message states: "The previous
    fragment is still on disk and would be merged as if current." """
    stage = CE._caller_stage()
    monkeypatch.setitem(CE._PENDING, stage, {})
    with pytest.raises(RuntimeError, match="nothing to write"):
        CE.flush()


def test_flush_writes_the_expected_json_shape(tmp_path, monkeypatch):
    stage = CE._caller_stage()
    k = _key()
    monkeypatch.setitem(CE._PENDING, stage, {
        k: {"value": 42, "producer": stage, "spec": "x", "cites": [], "cross_check": None},
    })
    monkeypatch.setattr(CE, "_FRAGDIR", tmp_path)
    monkeypatch.setattr(CE, "_ROOT", tmp_path)  # flush()'s print uses out.relative_to(_ROOT)
    out = CE.flush()
    assert out == tmp_path / f"{stage}.json"
    doc = json.loads(out.read_text())
    assert doc["_stage"] == stage
    assert doc["_n"] == 1
    assert doc[k]["value"] == 42
    assert "_generated" in doc


def test_flush_preserves_existing_file_permission_mode(tmp_path, monkeypatch):
    stage = CE._caller_stage()
    k = _key()
    out_path = tmp_path / f"{stage}.json"
    out_path.write_text("{}")
    out_path.chmod(0o640)
    monkeypatch.setitem(CE._PENDING, stage, {
        k: {"value": 1, "producer": stage, "spec": "x", "cites": [], "cross_check": None},
    })
    monkeypatch.setattr(CE, "_FRAGDIR", tmp_path)
    monkeypatch.setattr(CE, "_ROOT", tmp_path)
    CE.flush()
    assert (out_path.stat().st_mode & 0o777) == 0o640
