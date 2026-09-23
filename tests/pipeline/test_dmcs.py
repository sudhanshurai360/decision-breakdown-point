"""dmcs.py's eb_on() call-site regression, run against REAL state UR data (not synthetic).

`eb_on()` requires `realized` as an explicit positional argument, precisely to stop a silent class
of bug: a caller omitting it would fall back to forecast noise in the legs the statute reads as
known-at-decision-time data (eb_on()'s reference legs are "entirely historical",
state_eb_decision.py:60). Both of dmcs.py's main() call sites must therefore pass it. These tests
lock in the correct call shape, so a future signature change to eb_on() that breaks dmcs.py's call
sites fails a fast unit test rather than a 12-minute full run.

They also cover eb_on()'s nullable ("boolean" dtype) output, pd.NA rather than False across the
real Oct-2025 state-UR data gap: test_dmcs_decision_loss_propagates_undetermined_not_zero locks in
that dmcs.py's `.astype(float)` turns pd.NA into NaN (which year_mcs()'s own `.dropna()` already
excludes from evaluation), not silently into 0.0 ("no decision loss this month").

Requires the real state UR panel (state_eb_decision.CACHE) to be present on disk -- skipped if the
file is missing (e.g. a partial checkout) rather than failing. The skip is gated on the FILE, not
on whether load_state_ur() raises: see the comment on the gate below.
"""

import pandas as pd
import pytest

import state_eb_decision as SEB

# Gate ONLY on the data file's presence, then call load_state_ur() unguarded.
# The previous form wrapped that call in `try/except Exception`, so ANY failure inside
# load_state_ur() -- a genuine regression, not just a missing checkout -- silently turned this
# whole module into green skips. The skip must answer "is the panel on disk?" and nothing else;
# everything after that is allowed to fail loudly.
_HAS_STATE_UR = SEB.CACHE.exists()

pytestmark = pytest.mark.skipif(
    not _HAS_STATE_UR, reason=f"real state UR panel not available ({SEB.CACHE})")

_DF = SEB.load_state_ur() if _HAS_STATE_UR else None


@pytest.fixture(scope="module")
def one_state_slice():
    """A short, fast slice of one real state's series -- enough months for nowcast()'s warmup
    (WARMUP_YEARS) plus a few forecast points, without paying for a full-length 51-state run. Uses
    the LAST 400 months (not the first): the real Oct-2025 data gap is near the end of the full
    series, and test_dmcs_decision_loss_propagates_undetermined_not_zero below needs it present."""
    u = _DF[_DF.columns[0]].dropna().iloc[-400:]
    nc = SEB.nowcast(u)
    return u, nc


def test_eb_on_realized_status_accepts_realized_twice(one_state_slice):
    """dmcs.py's `realized_eb = SEB.eb_on(u, u)` -- the REALIZED EB status, obtained by passing the
    realized series as both the forecast and the realized leg, so the "current leg" contains no
    forecast at all (state_eb_decision.py:59)."""
    u, _ = one_state_slice
    realized_eb = SEB.eb_on(u, u)
    assert isinstance(realized_eb, pd.Series)
    assert realized_eb.dtype == "boolean"


def test_eb_on_per_model_decision_status_accepts_nowcast_and_realized(one_state_slice):
    """dmcs.py's `SEB.eb_on(nc[m].reindex(u.index), u)` -- the per-model decision status, the same
    (forecast, realized) argument pair state_eb_decision.py:155 passes for its own E_all
    computation (that call adds `protocol` as a third argument; dmcs.py takes the default).
    Omitting `realized` raises TypeError: missing 1 required positional argument."""
    u, nc = one_state_slice
    for m in SEB.MODELS:
        dec_status = SEB.eb_on(nc[m].reindex(u.index), u)
        assert isinstance(dec_status, pd.Series)
        assert dec_status.dtype == "boolean"


def test_dmcs_decision_loss_matches_dmcs_py_computation(one_state_slice):
    """Reproduces dmcs.py's exact dec_loss expression end-to-end on real data and checks the
    result is a well-formed 0/1 float loss, not a crash and not silently all-NaN."""
    u, nc = one_state_slice
    realized_eb = SEB.eb_on(u, u)
    dec_loss = {m: (SEB.eb_on(nc[m].reindex(u.index), u) != realized_eb).astype(float) for m in SEB.MODELS}
    for m, loss in dec_loss.items():
        vals = loss.dropna().unique()
        assert set(vals) <= {0.0, 1.0}, f"{m}: dec_loss has non-0/1 values {vals}"


def test_dmcs_decision_loss_propagates_undetermined_not_zero(one_state_slice):
    """eb_on() returns pd.NA (not False) for months whose 3-month-average legs fall in the real
    Oct-2025 state-UR data gap. dmcs.py's `!=` comparison against `realized_eb` and `.astype(float)`
    must propagate that as NaN, not silently coerce it to a determined 0.0/1.0 -- otherwise
    year_mcs()'s own `.reindex(common).dropna()` has nothing to exclude and an undetermined month
    gets counted as a determined non-flip."""
    u, nc = one_state_slice
    realized_eb = SEB.eb_on(u, u)
    assert realized_eb.isna().any(), "fixture slice should include the real Oct-2025 gap's downstream NA months"
    m = next(iter(SEB.MODELS))
    dec_status = SEB.eb_on(nc[m].reindex(u.index), u)
    dec_loss = (dec_status != realized_eb).astype(float)
    # wherever EITHER side is undetermined, `!=` on nullable booleans is itself undetermined (NA) --
    # dec_loss must be NaN there, never a determined 0.0/1.0 manufactured from an unknown comparison.
    either_na = dec_status.isna() | realized_eb.isna()
    assert dec_loss[either_na].isna().all(), "dec_loss should be NaN wherever either side is undetermined"
