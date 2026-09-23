"""Two explicit `raise` branches (converted from bare `assert` so they survive `python -O`) that
no test had exercised: canonical.py's own gate re-derivation and claims.py's own location-index
build.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import canonical as C  # noqa: E402
import claims as CL  # noqa: E402


def test_mcspower_n_trials_raises_when_fewer_than_12_series_qualify(monkeypatch, tmp_path):
    """canonical.py::_mcspower_n_trials() must raise RuntimeError, not silently re-derive a wrong
    n_trials, if a fresh read of breadth_distinct_predictions.csv finds fewer than 12 series with a
    fully rectangular, fully-finite >=20-year loss matrix."""
    bd = tmp_path / "breadth_distinct"
    bd.mkdir()
    # One series, 20 years x 2 models, fully finite -- qualifies (kept=1, far short of the required 12).
    rows = [{"series": "ONLY", "model": m, "year": 2000 + y, "sae": 1.0}
            for y in range(20) for m in ("ar1", "ets")]
    pd.DataFrame(rows).to_csv(bd / "breadth_distinct_predictions.csv", index=False)
    monkeypatch.setattr(C, "RES", tmp_path)

    with pytest.raises(RuntimeError, match="expected 12 curated series"):
        C._mcspower_n_trials()


def test_index_draft_raises_on_a_genuine_s_n_location_collision(monkeypatch):
    """claims.py::index_draft()'s S/N collision guard is structurally near-unreachable in practice
    (draft_ssrn.md's headings are always 'S'-prefixed, note_ssrn.md's always 'N'-prefixed, so real keys never
    collide) -- but the raise itself must actually fire if that invariant is ever violated, rather
    than silently letting one manuscript's location entry overwrite the other's. Forces a collision
    directly by monkeypatching _index_headings() to (incorrectly) use the same prefix for both."""
    monkeypatch.setattr(CL, "_draft_text", lambda: "fake draft text")
    monkeypatch.setattr(CL, "_note_text", lambda: "fake note text")
    monkeypatch.setattr(CL, "_index_headings", lambda text, prefix: {"S4": text})

    with pytest.raises(RuntimeError, match="location collision"):
        CL.index_draft()
