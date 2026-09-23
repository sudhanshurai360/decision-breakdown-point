"""The two canonical.json values that CAN be independently re-derived from a producer CSV. The
other unregistered values are deterministic Monte Carlo simulations (fixed seeds) or in-memory-only
computations with no per-draw detail persisted to disk, so they cannot be re-aggregated from a
producer CSV the way `canonical.py`'s other checks are. These two can:

- `vintage.window_start`: the producer's own `panel_start` extra field plus a fixed 26-month
  lookback constant fully determines it via simple date arithmetic.
- `mcspower.n_trials`: a deterministic count over a real, persisted input CSV
  (breadth_distinct_predictions.csv), independently re-derivable without touching the original
  `load_panels()` code.
"""
import json

import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")


def test_vintage_window_start_registered_and_matches():
    assert "vintage.window_start" in C.CHECKS
    doc = json.loads(C.OUT.read_text())
    assert C.CHECKS["vintage.window_start"]() == doc["vintage.window_start"]["value"]


def test_vintage_window_start_formula_matches_producer_constant():
    """Locks in the LOOKBACK_MONTHS=26 constant this check reproduces independently rather than
    imports -- if state_eb_vintage.py's own constant ever changes, this test (not just the live
    canonical.json comparison) should be the thing that notices."""
    import state_eb_vintage as SEV
    assert SEV.LOOKBACK_MONTHS == 26


def test_mcspower_n_trials_registered_and_matches():
    assert "mcspower.n_trials" in C.CHECKS
    doc = json.loads(C.OUT.read_text())
    assert C.CHECKS["mcspower.n_trials"]() == doc["mcspower.n_trials"]["value"]


def test_mcspower_n_trials_is_a_positive_int_derived_from_real_data():
    n = C.CHECKS["mcspower.n_trials"]()
    assert isinstance(n, int)
    assert n > 0
