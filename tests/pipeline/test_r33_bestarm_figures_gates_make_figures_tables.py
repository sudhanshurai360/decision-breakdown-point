"""Mutation-verified tests, including pinning R3 (named in this file only):
make_figures.py's Table 1 and Table R renderers otherwise have only count-level coverage, so
column-permutation mutations survive:
  A  `models = ["ar1", "ets", ...]` with ar1/ets swapped  -> every Table R cell under the wrong header
  B  the Table 1 row f-string with T and k* swapped        -> every Table 1 row wrong
  C  Fragile rule `k/T < 0.25` -> `<=`                     -> latent (no series at exactly 0.25 today)
  D  Fragile rule `ws < 0.60` -> `<=`                      -> latent (no series at exactly 0.60 today)
  R3 the rule is applied to the UNROUNDED winner_stability float while the column renders at 2 dp,
     so 0.595 <= ws < 0.60 prints "0.60 | no | yes" -- pinned here as the current, documented
     behaviour (a mutant that rounds before applying the rule is killed).

Every test here redirects RES / OUT / ROOT to tmp_path: NOTHING is written under paper/figures/
(unlike tests/pipeline/test_make_figures.py's first test). Fixtures are tiny hand-built CSVs with
the exact columns table1()/table_accuracy() read; the expected rows are written out by hand.
Relies on tests/conftest.py for sys.path. Every test here was verified to FAIL on its named
mutant (A, B, C, D, R3) and PASS on the real module.
"""
import pandas as pd
import pytest

import make_figures as MF  # noqa: E402  (path via tests/conftest.py)

MODELS = ["ar1", "ets", "theta", "sarima", "dhr", "seasonal_naive"]


@pytest.fixture
def redirect(monkeypatch, tmp_path):
    res = tmp_path / "results"
    out = tmp_path / "paper" / "figures"
    res.mkdir(); out.mkdir(parents=True)
    monkeypatch.setattr(MF, "RES", res)
    monkeypatch.setattr(MF, "OUT", out)
    monkeypatch.setattr(MF, "ROOT", tmp_path)   # savefig/print use path.relative_to(ROOT)
    return res, out


def _body_rows(md_path):
    lines = [l for l in md_path.read_text().splitlines() if l.startswith("| ") and not l.startswith("|---")]
    return lines[0], lines[1:]


# --- Table R -------------------------------------------------------------------------------------

def test_table_r_cells_sit_under_their_own_model_header(redirect, capsys):
    """kills: A (swap "ar1"/"ets" in table_accuracy()'s models list).
    One series with loss_<m> = 0.1, 0.2, ..., 0.6 in HEADER order AR1, ETS, Theta, SARIMA, DHR,
    Seasonal-naive -- every cell is distinct so any permutation of the models list changes the
    rendered row. The full row is asserted, not 'the AR1 value appears somewhere'."""
    res, out = redirect
    row = {"series": "S1"}
    for i, m in enumerate(MODELS):
        row[f"loss_{m}"] = 0.1 * (i + 1)
    row2 = {"series": "S2"}
    for i, m in enumerate(MODELS):
        row2[f"loss_{m}"] = 1.0 + 0.111 * (i + 1)     # 1.111, 1.222, ... (3-dp distinct)
    pd.DataFrame([row2, row]).to_csv(res / "model_set_audit.csv", index=False)   # unsorted on purpose

    MF.table_accuracy()
    capsys.readouterr()
    header, body = _body_rows(out / "tableR_accuracy.md")
    assert header == "| Series | AR1 | ETS | Theta | SARIMA | DHR | Seasonal-naive |"
    assert body == [
        "| S1 | 0.100 | 0.200 | 0.300 | 0.400 | 0.500 | 0.600 |",
        "| S2 | 1.111 | 1.222 | 1.333 | 1.444 | 1.555 | 1.666 |",
    ], "rows must be sorted by series and every cell must sit under its own model's header"


# --- Table 1 -------------------------------------------------------------------------------------

def _write_table1_inputs(res, rows):
    """rows: list of dict(series, mcs, winner, T, k, conc, ws, reversal)."""
    pd.DataFrame([{"series": r["series"], "MCS_abs_10": r["mcs"], "best_abs": r["winner"], "sci": 0.2}
                  for r in rows]).to_csv(res / "mcs_test.csv", index=False)
    sc = []
    for r in rows:
        for metric in ("mase", "rmsse"):
            # the rmsse rows carry garbage so a missing `metric == "mase"` filter would show
            sc.append({"series": r["series"], "metric": metric,
                       "k_star": r["k"] if metric == "mase" else 99,
                       "winner_stability": r["ws"] if metric == "mase" else 0.01,
                       "reversal": r["reversal"] if metric == "mase" else True,
                       "concentration": r["conc"] if metric == "mase" else 0.99})
    pd.DataFrame(sc).to_csv(res / "social_choice.csv", index=False)
    frag = []
    for r in rows:
        for metric in ("mase", "rmsse"):
            frag.append({"series": r["series"], "metric": metric,
                         "n_years": r["T"] if metric == "mase" else 7})
    pd.DataFrame(frag).to_csv(res / "multiseries_fragility.csv", index=False)


ROWS = [
    # PI-like: identified, T=36, k*=20, stable -> not fragile.  T != k* so a column swap shows.
    dict(series="P1", mcs=1, winner="ar1", T=36, k=20, conc=0.66, ws=0.95, reversal=False),
    # k*/T = 5/20 = 0.25 EXACTLY with a reversal: strict `<` -> NOT fragile
    dict(series="Q2", mcs=3, winner="ets", T=20, k=5, conc=0.40, ws=0.80, reversal=True),
    # k*/T = 4/20 = 0.20 with a reversal: fragile via the reversal clause only
    dict(series="Q3", mcs=2, winner="dhr", T=20, k=4, conc=0.41, ws=0.80, reversal=True),
    # ws = 0.60 EXACTLY, no reversal: strict `<` -> NOT fragile
    dict(series="Q4", mcs=4, winner="theta", T=30, k=9, conc=0.50, ws=0.60, reversal=False),
    # ws = 0.5999 -> renders "0.60" but the rule sees the float -> fragile (R3)
    dict(series="Q5", mcs=5, winner="sarima", T=30, k=9, conc=0.51, ws=0.5999, reversal=False),
    # plainly fragile on stability
    dict(series="Q6", mcs=6, winner="ar1", T=27, k=3, conc=0.30, ws=0.55, reversal=False),
]

EXPECTED_ROWS = [
    "| P1 | 1 | ar1 | 36 | 20 | 0.66 | 0.95 | no | no |",
    "| Q3 | 2 | dhr | 20 | 4 | 0.41 | 0.80 | yes | yes |",
    "| Q2 | 3 | ets | 20 | 5 | 0.40 | 0.80 | yes | no |",
    "| Q4 | 4 | theta | 30 | 9 | 0.50 | 0.60 | no | no |",
    "| Q5 | 5 | sarima | 30 | 9 | 0.51 | 0.60 | no | yes |",
    "| Q6 | 6 | ar1 | 27 | 3 | 0.30 | 0.55 | no | yes |",
]


def test_table1_rows_render_every_column_in_its_own_place(redirect, capsys):
    """kills: B (swap T and k* in the row f-string); also any Winner/Concentration/winner_stability
    column permutation, the missing metric=='mase' filter, and the |MCS| sort. Rows are asserted
    in full against hand-written expectations (sorted by |MCS| ascending, as table1() promises)."""
    res, out = redirect
    _write_table1_inputs(res, ROWS)
    t = MF.table1()
    capsys.readouterr()
    header, body = _body_rows(out / "table1_identification.md")
    assert header == "| Series | \\|MCS\\| | Winner | T | k* | Concentration | winner_stability | Reversal | Fragile |"
    assert body == EXPECTED_ROWS
    assert list(t["Series"]) == ["P1", "Q3", "Q2", "Q4", "Q5", "Q6"]
    assert list(t["T"]) == [36, 20, 20, 30, 30, 27] and list(t["k*"]) == [20, 4, 5, 9, 9, 3]


def test_table1_fragile_rule_is_strict_at_both_thresholds(redirect, capsys):
    """kills: C (`k/T < 0.25` -> `<=`) via Q2 (k*/T = 0.25 exactly, reversal, must be 'no');
    kills: D (`ws < 0.60` -> `<=`) via Q4 (ws = 0.60 exactly, must be 'no').
    Q3 (0.20 with reversal) and Q6 (0.55) are the positive controls on each clause."""
    res, out = redirect
    _write_table1_inputs(res, ROWS)
    t = MF.table1().set_index("Series")
    capsys.readouterr()
    assert t.loc["Q2", "Fragile"] == "no", "k*/T == 0.25 exactly must NOT be fragile (strict <)"
    assert t.loc["Q4", "Fragile"] == "no", "winner_stability == 0.60 exactly must NOT be fragile (strict <)"
    assert t.loc["Q3", "Fragile"] == "yes"
    assert t.loc["Q6", "Fragile"] == "yes"
    assert t.loc["P1", "Fragile"] == "no"


def test_table1_fragile_rule_reads_the_unrounded_float_not_the_rendered_2dp(redirect, capsys):
    """kills: a mutant that rounds winner_stability to the displayed 2 dp BEFORE applying the rule
    (`stable_alone = round(ws, 2) < 0.60`). Pins R3 as the CURRENT behaviour: Q5
    (ws = 0.5999) renders '0.60' yet is Fragile 'yes', while Q4 (ws = 0.60) renders identically
    and is 'no'. The caption's 'every Fragile verdict is verifiable directly from this table' promise is
    therefore not met at 2 dp for such a row; the caption's reversal clause must NOT name Q5
    (its verdict comes from the stability clause, not the reversal clause)."""
    res, out = redirect
    _write_table1_inputs(res, ROWS)
    t = MF.table1().set_index("Series")
    capsys.readouterr()
    assert t.loc["Q5", "winner_stability"] == "0.60" and t.loc["Q4", "winner_stability"] == "0.60"
    assert t.loc["Q5", "Fragile"] == "yes" and t.loc["Q4", "Fragile"] == "no"
    text = (out / "table1_identification.md").read_text()
    assert "including Q3, whose winner_stability column alone" in text
    assert "Q5, whose" not in text and "Q5 and" not in text and "and Q5," not in text


def test_table1_caption_counts_nonidentified_series_from_mcs_size(redirect, capsys):
    """The caption's 'N of M series have |MCS| > 1' is derived, not typed: 5 of 6 here."""
    res, out = redirect
    _write_table1_inputs(res, ROWS)
    MF.table1()
    capsys.readouterr()
    text = (out / "table1_identification.md").read_text()
    assert "all 6 curated series" in text
    assert "5 of 6 series have |MCS| > 1" in text
