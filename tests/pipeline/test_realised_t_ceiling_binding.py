"""Regression test for `flagship.ceiling_binding`, the canonical key behind the Proposition 2.2
write-up (§2.2/§2.3 of paper/draft_ssrn.md).

realised_t.py emits flagship.t_binding and flagship.t_max; without a citable key for the
Proposition 2.2 ceiling itself (sqrt(T*k/(T-k)) at the flagship's own T and k*), §2.3's worked
example ("`t_binding` = **0.490**, far inside the ceiling") would have a number for one side of the
comparison and not the other. This test locks the emitted value to a direct
re-derivation from the committed realised_t.csv, and checks the invariant the whole write-up rests
on: the realised statistic must not exceed the ceiling it is being bounded by (Proposition 2.2 is a
proved bound, so a violation here would mean either a coding error or a genuinely sharp panel worth
its own footnote -- not silently plausible either way).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CSV = ROOT / "multiseries" / "results" / "realised_t.csv"
CANON = ROOT / "multiseries" / "results" / "canonical" / "realised_t.json"


def _flagship_row():
    d = pd.read_csv(CSV)
    fl = d[d.panel == "FLAGSHIP"]
    assert len(fl) == 1
    return fl.iloc[0]


def test_ceiling_binding_matches_direct_formula_on_committed_csv():
    """sqrt(T*k/(T-k)) recomputed from the flagship row's own T and k_star must match the emitted
    value -- not some other panel's row, not a stale cached number."""
    r = _flagship_row()
    T, k = float(r["T"]), float(r.k_star)
    expected = float(np.sqrt(T * k / (T - k)))
    frag = json.loads(CANON.read_text())
    assert frag["flagship.ceiling_binding"]["value"] == round(expected, 3)


def test_ceiling_binding_registered_and_cited_in_canonical():
    frag = json.loads(CANON.read_text())
    assert "flagship.ceiling_binding" in frag
    assert frag["flagship.ceiling_binding"]["cites"] == ["S2.3 body: the flagship's realised statistic"]


def test_t_binding_does_not_exceed_its_own_ceiling():
    """The whole point of §2.3's worked example: Proposition 2.2 is a proved bound, so the realised
    i.i.d.-studentized |t| against the binding rival must sit at or below ceiling_binding. A failure
    here means either a bug in realised_t.py or ceiling_binding/t_binding disagreeing about which
    rival is binding -- not a fact to report as-is."""
    frag = json.loads(CANON.read_text())
    t_binding = frag["flagship.t_binding"]["value"]
    ceiling = frag["flagship.ceiling_binding"]["value"]
    assert t_binding <= ceiling + 1e-9


def test_ceiling_binding_equals_committed_csv_column_for_flagship():
    r = _flagship_row()
    frag = json.loads(CANON.read_text())
    assert frag["flagship.ceiling_binding"]["value"] == round(float(r.ceiling_binding), 3)
