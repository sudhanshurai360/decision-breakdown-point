"""best_arm.py must store full-precision `gap2_t`/`H` in `rows`/best_arm.csv, rounding only for the
printed display table (via a `.round()`'d copy, never into storage). These are the exact columns
`spearmanr()` (in this file) and `canonical.py`'s own CHECKS re-derivation (reading best_arm.csv
fresh) both correlate on, so rounding at storage time corrupts the correlation itself, not just its
display.

The effect is real, not hypothetical: PAYEMS (H=2.905150) and UNEMPLOY (H=2.902528) round to an
artificial tie at 2.9 despite having different |MCS| (5 vs 4), which moves the H-vs-|MCS| Spearman
correlation and, more consequentially, the "excl-PI leverage" sensitivity check the manuscript
quotes directly (S4.3). The gap-based numbers are essentially unaffected (gap2_t at 2 decimal
places never produces a PAYEMS/UNEMPLOY-style tie), which is why the exposure is on the H side."""
from pathlib import Path

import pytest

# No production module is imported here -- every test reads committed artefacts off disk -- so no
# sys.path setup is needed; ROOT exists only to locate those files.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def test_committed_best_arm_csv_stores_full_precision_not_rounded():
    """Locked against the real committed data: best_arm.csv's H column must retain enough precision
    to distinguish PAYEMS (2.905150) from UNEMPLOY (2.902528) -- if a change introduces rounding at
    storage time, these two collapse to an identical 2.9 and silently corrupt the H-vs-|MCS|
    correlation."""
    import pandas as pd

    csv = ROOT / "multiseries" / "results" / "best_arm.csv"
    if not csv.exists():
        pytest.skip("best_arm.csv not available")
    d = pd.read_csv(csv).set_index("series")
    payems_h = d.loc["PAYEMS", "H"]
    unemploy_h = d.loc["UNEMPLOY", "H"]
    assert payems_h != unemploy_h, (
        "PAYEMS and UNEMPLOY's H values are stored identically -- if this is because they were "
        "rounded before storage, the H-vs-|MCS| correlation is silently corrupted (see this "
        "file's module docstring)"
    )
    # both must actually be close to the known real values, not merely "different from each other"
    assert payems_h == pytest.approx(2.905150, abs=1e-4)
    assert unemploy_h == pytest.approx(2.902528, abs=1e-4)


def test_rounding_before_correlation_materially_changes_the_excl_pi_h_drop():
    """Reproduces the defect mechanism end-to-end on the real committed loss data: recomputes the
    H-vs-|MCS| excl-PI relative drop BOTH the rounded-before-correlation way and the full-precision
    way, and confirms they genuinely differ -- this is not a hypothetical edge case, it measurably
    moves a number the manuscript quotes (S4.3's "relative drop" figure)."""
    import pandas as pd
    from scipy import stats

    csv = ROOT / "multiseries" / "results" / "best_arm.csv"
    if not csv.exists():
        pytest.skip("best_arm.csv not available")
    r = pd.read_csv(csv)

    def excl_pi_drop_from(df):
        h_rho, _ = stats.spearmanr(df["H"], df["MCS"], nan_policy="omit")
        excl = df[df.series != "PI"]
        h_rho_excl, _ = stats.spearmanr(excl["H"], excl["MCS"], nan_policy="omit")
        return round(100 * (abs(h_rho) - abs(h_rho_excl)) / abs(h_rho))

    # full precision (current, correct)
    drop_full = excl_pi_drop_from(r)

    # reproduce the OLD bug deliberately: round before correlating, exactly as the pre-fix code did
    drop_old_buggy = excl_pi_drop_from(r.assign(H=r.H.round(1), gap2_t=r.gap2_t.round(2)))

    # On the committed curated panel: 65% at full precision vs 69% under the rounding bug. The
    # mechanism -- rounding before correlating moves the quoted figure -- is what is being pinned.
    assert drop_full == 65, f"expected the fixed, full-precision drop to be 65%, got {drop_full}%"
    assert drop_old_buggy == 69, (
        f"expected reproducing the old rounded-before-correlation bug to give 69%, got {drop_old_buggy}%"
    )
    assert drop_full != drop_old_buggy, (
        "rounding before correlation no longer changes the reported drop -- either the underlying "
        "data changed, or this test's reproduction of the old bug is no longer valid; re-verify "
        "before assuming the fix is unnecessary"
    )


def test_canonical_bestarm_h_mcs_excl_pi_drop_pct_matches_full_precision():
    """End-to-end: canonical.json's own bestarm.spearman_H_mcs_excl_pi_drop_pct (what draft_ssrn.md S4.3
    actually cites) must equal the full-precision-computed 65 (regenerated panel), not the
    rounding-corrupted 69."""
    import json

    canon_path = ROOT / "paper" / "canonical.json"
    if not canon_path.exists():
        pytest.skip("canonical.json not available")
    canon = json.loads(canon_path.read_text())
    assert canon["bestarm.spearman_H_mcs_excl_pi_drop_pct"]["value"] == 65
