"""Regression test:
canonical.py's own writes of paper/canonical.json (merge() and --rehash) used a bare
`path.write_text()` with no atomicity at all, despite this being the single most-read,
most-contended file in the project (a confirmed real corruption incident from
concurrent gate access to this exact file). Fixed with `_atomic_write_json()`, which follows
canonical_emit.py::flush()'s existing tmp-file + shutil.copymode + Path.replace pattern.

`_atomic_write_json()` then had ZERO pytest coverage -- not even a happy-path test of the write
itself, let alone the permission-mode preservation its docstring promises ("copy the destination's
mode if it already exists"). This is a real gap given canonical_emit.py::
flush()'s analogous helper already has a dedicated permission-mode test
(test_flush_preserves_existing_file_permission_mode in tests/instrument/test_canonical_emit.py).
These tests close that specific gap for canonical.py's own copy of the pattern, directly -- no
call to merge()/check() or the real paper/canonical.json involved.
"""
import json

import canonical as C


def test_atomic_write_json_writes_the_expected_content(tmp_path):
    p = tmp_path / "out.json"
    C._atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_atomic_write_json_creates_missing_parent_directories(tmp_path):
    p = tmp_path / "nested" / "dir" / "out.json"
    C._atomic_write_json(p, {"a": 1})
    assert p.exists()
    assert json.loads(p.read_text()) == {"a": 1}


def test_atomic_write_json_preserves_existing_file_permission_mode(tmp_path):
    """Mirrors canonical_emit.py::flush()'s own permission-mode test. _atomic_write_json()'s
    docstring promises to "copy the destination's mode if it already exists", but (unlike flush())
    it had no test confirming the shutil.copymode() call actually preserves an existing file's mode
    rather than silently defaulting to umask-determined permissions on the new file."""
    p = tmp_path / "out.json"
    p.write_text("{}")
    p.chmod(0o640)
    C._atomic_write_json(p, {"a": 1})
    assert (p.stat().st_mode & 0o777) == 0o640


def test_atomic_write_json_leaves_no_tmp_file_behind(tmp_path):
    """The tmp-file + Path.replace() pattern's whole point is that the file at `path` is either
    the old complete content or the new complete content, never a half-written intermediate -- and
    the .tmp sibling must not linger after a successful write."""
    p = tmp_path / "out.json"
    C._atomic_write_json(p, {"a": 1})
    assert not (tmp_path / "out.json.tmp").exists()
    assert json.loads(p.read_text()) == {"a": 1}


def test_atomic_write_json_overwrites_existing_content_completely(tmp_path):
    p = tmp_path / "out.json"
    C._atomic_write_json(p, {"a": 1, "b": 2})
    C._atomic_write_json(p, {"c": 3})
    assert json.loads(p.read_text()) == {"c": 3}
