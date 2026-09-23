"""density_audit.py's raw-M4-data access.

THE HAZARD: reading `MA.DATA / "Monthly-train.csv"` / `"Monthly-test.csv"` directly assumes the raw
M4-Monthly CSVs are already on disk. `multiseries/data/m_audit/` is NOT committed to this repo (it
is a static, identically re-fetchable public archive), so on a genuinely clean clone that raises
`FileNotFoundError` -- a failure only a full `reproduce.py --fast` against a fresh `git clone`
surfaces, not any unit test or `canonical.py --check`. `m_audit.py`'s own `main()` calls
`m_audit.fetch()`; `density_audit.py` must do the same.

This test guards that structurally (no network access, no dependency on the real ~99MB cache being
present): it confirms `main()`'s source calls `MA.fetch(` rather than constructing the raw data
path directly, so an edit that reverts to a hardcoded path is caught immediately rather than only
on the next real clean-clone run.
"""
import inspect

import pytest

import density_audit as DA
import m_audit as MA


def test_main_calls_m_audit_fetch_not_a_hardcoded_path():
    src = inspect.getsource(DA.main)
    assert "MA.fetch(" in src, (
        "density_audit.main() must obtain the raw M4-Monthly data via m_audit.fetch() (which "
        "downloads it if missing) -- reading MA.DATA / '<name>.csv' directly assumes the data is "
        "already on disk, which is false on a genuinely clean clone (multiseries/data/m_audit/ is "
        "not committed to this repo -- it is a re-fetchable static archive)."
    )


def test_m_audit_fetch_returns_paths_that_exist_after_calling():
    """fetch() must return usable Path objects pointing at files that actually exist once fetch()
    has run. Skips (rather than hitting the network) if the raw M4-Monthly cache isn't already
    present -- this is a same-machine sanity check on fetch()'s return value, not a network test."""
    if not (MA.DATA / "Monthly-train.csv").exists():
        pytest.skip("raw M4-Monthly cache not present locally; not forcing a network fetch in tests")
    tr_f, te_f = MA.fetch("Monthly")
    assert tr_f.exists(), f"fetch() returned a train path that does not exist: {tr_f}"
    assert te_f.exists(), f"fetch() returned a test path that does not exist: {te_f}"
