"""apply_tcode() golden-value tests -- locks the McCracken-Ng (2016) transformation-code math in
fred_md_sweep.py against hand-computed values, independent of numpy's own diff/log implementation
being "obviously right". Also locks the FREDMD_TCODE table's most error-prone entry (tcode=7, the
one series -- NONBORRES -- that uses it) so a future edit can't silently swap it back to tcode=5.
"""
import numpy as np
import pytest

import fred_md_sweep as FS


def test_tcode_1_is_level():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(FS.apply_tcode(y, 1), y)


def test_tcode_2_is_first_difference():
    y = np.array([1.0, 3.0, 6.0, 10.0])
    assert np.allclose(FS.apply_tcode(y, 2), [2.0, 3.0, 4.0])


def test_tcode_3_is_second_difference():
    y = np.array([1.0, 3.0, 6.0, 10.0])
    assert np.allclose(FS.apply_tcode(y, 3), [1.0, 1.0])


def test_tcode_4_is_log():
    y = np.array([1.0, np.e, np.e**2])
    assert np.allclose(FS.apply_tcode(y, 4), [0.0, 1.0, 2.0])


def test_tcode_5_is_diff_log():
    y = np.array([1.0, np.e, np.e**3])
    assert np.allclose(FS.apply_tcode(y, 5), [1.0, 2.0])


def test_tcode_6_is_second_diff_log():
    y = np.array([1.0, np.e, np.e**3, np.e**6])
    assert np.allclose(FS.apply_tcode(y, 6), [1.0, 1.0])


def test_tcode_7_is_diff_of_pct_change():
    # y = 100, 110, 121, 133.1 -> pct change is a constant 10% each step -> diff of pct change is 0
    y = np.array([100.0, 110.0, 121.0, 133.1])
    assert np.allclose(FS.apply_tcode(y, 7), [0.0, 0.0], atol=1e-10)


def test_tcode_7_detects_changing_growth_rate():
    # growth accelerates 10% -> 20% -> diff of pct change should be +0.10
    y = np.array([100.0, 110.0, 132.0])
    assert np.allclose(FS.apply_tcode(y, 7), [0.10], atol=1e-10)


def test_tcode_unknown_raises():
    with pytest.raises(ValueError):
        FS.apply_tcode(np.array([1.0, 2.0]), 8)


# ---------------------------------------------------------------------------
# FREDMD_TCODE table spot-checks against the published McCracken-Ng (2016) appendix, independently
# verified against a pdftotext extraction of the appendix PDF.
# ---------------------------------------------------------------------------
# UMCSENT is deliberately absent from the spot-check list below: it is dropped from FRED-MD
# entirely on U-Michigan licensing grounds. It is neither a deliberately-excluded close cousin
# (test_fredmd_tcode_table_only_covers_confirmed_codes below) nor a confirmed code -- it is absent
# from both FREDMD_CODES and FREDMD_TCODE, checked directly by
# test_licensed_series_removed_from_fredmd below.
@pytest.mark.parametrize("mnemonic,expected", [
    ("RPI", 5), ("INDPRO", 5), ("CUMFNS", 2), ("UNRATE", 2), ("PAYEMS", 5),
    ("AWHMAN", 1), ("AWOTMAN", 2), ("HOUST", 4),
    ("DPCERA3M086SBEA", 5), ("RSAFS", 5), ("BUSINV", 5),
    ("M1SL", 6), ("M2REAL", 5), ("NONBORRES", 7), ("FEDFUNDS", 2), ("GS10", 2),
    ("WPSID61", 6), ("CPIAUCSL", 6),
])
def test_fredmd_tcode_table_matches_published_appendix(mnemonic, expected):
    assert FS.FREDMD_TCODE[mnemonic] == expected


def test_fredmd_tcode_table_only_covers_confirmed_codes():
    # deliberately-excluded close cousins (TCU, UNEMPLOY, PCE, PI, DGORDER, AMTMNO, BOGMBASE,
    # TOTALSL, EXPGS, IMPGS, BOPGEXP, BOPGIMP) must stay absent from FREDMD_TCODE -- their presence
    # in FREDMD_CODES does not mean a transform code was ever confirmed for them.
    excluded = {"TCU", "UNEMPLOY", "PCE", "PI", "DGORDER", "AMTMNO", "BOGMBASE",
                "TOTALSL", "EXPGS", "IMPGS", "BOPGEXP", "BOPGIMP"}
    assert excluded.isdisjoint(FS.FREDMD_TCODE)


# ---------------------------------------------------------------------------
# tcode_outlier_ratio() -- regression test:
# NONBORRES (tcode=7) crosses near zero during the 2008 financial crisis, producing a FINITE but
# 863.9x relative-magnitude outlier in its transformed series -- exactly what the function's own
# docstring calls "the failure mode np.isfinite() cannot see" (fred_md_sweep.py:77-78).
# ---------------------------------------------------------------------------
def test_tcode_outlier_ratio_flat_series_is_one():
    y_t = np.array([1.0, 1.0, 1.0, 1.0])
    assert FS.tcode_outlier_ratio(y_t) == pytest.approx(1.0, rel=1e-6)


def test_tcode_outlier_ratio_below_threshold_for_ordinary_variation():
    # a legitimate, moderately volatile transformed series (no sign-crossing artifact)
    rng = np.random.default_rng(0)
    y_t = rng.normal(0, 1, 500)
    y_t[250] = 50.0  # one real 50x-ish shock, e.g. a COVID month -- still far under the threshold
    assert FS.tcode_outlier_ratio(y_t) < FS.TCODE_OUTLIER_RATIO_THRESHOLD


def test_tcode_outlier_ratio_catches_the_real_nonborres_shape():
    """THE CORE DEFECT-CATCHING PATH: reproduces the real NONBORRES near-zero-crossing shape
    (a level series with a sign flip during 2008) and confirms tcode=7's output crosses the
    threshold, exactly as it does on the real fetched series (verified 863.9x during a read-only
    sensitivity check)."""
    y = np.concatenate([
        np.linspace(50.0, 5.0, 20),          # decaying toward zero
        np.array([-0.8, -16.3, -333.5, -20.0, 10.0]),   # the actual 2008 sign-crossing shape
        np.linspace(15.0, 60.0, 20),
    ])
    y_t = FS.apply_tcode(y, 7)
    assert FS.tcode_outlier_ratio(y_t) > FS.TCODE_OUTLIER_RATIO_THRESHOLD


def test_tcode_outlier_ratio_on_real_nonborres_series_exceeds_threshold():
    """Direct real-data confirmation, skipped if the cached FRED-MD panel is unavailable."""
    try:
        df = FS.load_fredmd()
    except Exception:
        pytest.skip("FRED-MD cache not available")
    s = FS.load_series_gap_safe(df, "NONBORRES")
    if s is None:
        pytest.skip("NONBORRES series not available")
    y_t = FS.apply_tcode(s.to_numpy(float), FS.FREDMD_TCODE["NONBORRES"])
    ratio = FS.tcode_outlier_ratio(y_t)
    assert ratio > FS.TCODE_OUTLIER_RATIO_THRESHOLD, (
        f"expected the real NONBORRES 2008 sign-crossing artifact to exceed the threshold, got {ratio:.1f}")


def test_licensed_series_removed_from_fredmd():
    """UMCSENT, AAA, and BAA must be absent from BOTH FREDMD_CODES and FREDMD_TCODE -- not merely
    excluded from TCODE like the "close cousins" above, which are a different category (present in
    CODES, deliberately never given a transform code). All three carry an explicit
    licensing/redistribution restriction (UMCSENT: U-Michigan citation requirement; AAA/BAA: a
    Moody's copyright notice prohibiting redistribution without prior written consent, verified
    against FRED series metadata), so they are removed from the FRED-MD universe entirely rather
    than merely disclosed."""
    for sid in ("UMCSENT", "AAA", "BAA"):
        assert sid not in FS.FREDMD_CODES, f"{sid} should be excluded from FREDMD_CODES (licensed)"
        assert sid not in FS.FREDMD_TCODE, f"{sid} should be excluded from FREDMD_TCODE (licensed)"
