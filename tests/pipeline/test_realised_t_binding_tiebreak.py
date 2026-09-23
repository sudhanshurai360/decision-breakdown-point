"""realised_t.py::analyse()'s binding-rival tie-break must match its sibling computation.

fragility.py's decision_breakdown() (the analogous binding-opponent computation, same k* concept,
used by flagship_fragility.py for flagship.k_star's `rival` field) resolves a tie in k by SORTED
rival name, explicitly and by design. realised_t.py's `analyse()` -- which computes
flagship.t_binding, the "realised statistic for the same pair" per flagship.k_star's own canonical
spec string -- must do the same. Breaking ties by whichever rival appears FIRST in the input
DataFrame's column order happens to agree for the curated per-series panels (whose caller builds
`ms = sorted(...)`), but follows the RAW CSV COLUMN ORDER for FLAGSHIP
(extension_scoreboard_1988-2025.csv): a caller-order-dependent convention, silently inconsistent
with the sibling computation.

The distinction is latent on the current data: of the panels with a genuine tie in k (4 of 12
curated -- DGORDER, HOUST, TCU, UNEMPLOY -- plus FLAGSHIP, which has no tie at its own minimum k),
none pick a different binding_rival under the two conventions, since the curated panels already
iterate sorted. No currently-cited canonical value depends on it.
"""
import numpy as np
import pandas as pd

import fragility as F
import realised_t as RT


def test_binding_rival_ties_broken_by_sorted_name_not_column_order():
    """Construct a synthetic panel where two rivals genuinely tie at the minimum k, in a
    DataFrame whose column order deliberately puts the alphabetically-LATER rival first --
    the exact scenario a column-order-dependent tie-break gets wrong."""
    rng = np.random.default_rng(0)
    T = 30
    win = rng.uniform(0.5, 1.0, T)
    # two rivals engineered to have the SAME breakdown number against `win`
    riv_z = win + 0.5   # constant positive margin -> same k for both rivals below
    riv_a = win + 0.5
    M = pd.DataFrame({"zzz_rival": riv_z, "aaa_rival": riv_a, "win": win})[["zzz_rival", "win", "aaa_rival"]]
    r = RT.analyse("synthetic", M)
    assert r is not None
    assert r["winner"] == "win"
    # both rivals genuinely tie in k (by construction) -- verify the fixture is real, not vacuous
    ks = {}
    for riv in ("zzz_rival", "aaa_rival"):
        ks[riv] = F.breakdown_number(M["win"].to_numpy(float), M[riv].to_numpy(float), np.ones(T))[0]
    assert ks["zzz_rival"] == ks["aaa_rival"], "fixture must produce a genuine tie to test the tie-break"
    # sorted-name convention -> "aaa_rival" wins the tie, regardless of it appearing SECOND in M's columns
    assert r["binding_rival"] == "aaa_rival"


def test_binding_rival_matches_decision_breakdowns_convention_on_real_flagship_data():
    """Direct cross-check against fragility.py's decision_breakdown() -- the sibling computation
    flagship_fragility.py uses for flagship.k_star's own binding-rival field -- on the real
    FLAGSHIP panel. Both must agree, since they are documented as describing the same pair."""
    df = pd.read_csv(RT.FLAG, index_col=0)   # absolute path from the module, not a cwd-relative read
    r = RT.analyse("FLAGSHIP", df)
    L = {m: df[m].to_numpy(float) for m in df.columns}
    w = np.ones(len(df))
    _k, binding_opp, _removed = F.decision_breakdown(L, w, a=r["winner"])
    assert r["binding_rival"] == binding_opp
