"""claims.py cannot see a silent change in
~36% of cited numeric values (132 of 362 today, per claims.py's own N_UNDETECTABLE_CITED ratchet --
a wrong number hides behind an unrelated value's rendering). The
structural fix is a matcher redesign, deliberately not attempted; instead `claims.py --coverage`
measures the undetectable set and ratchets it (exact count + key-set hash) so it cannot silently
grow. The measurement costs ~5 minutes (one orphans()+misplaced() pass per cited value), so this
test runs only under SLOW_GATES=1 -- set BY HAND at the end of a prose pass (nothing in reproduce.py sets it), not by a routine
`pytest -q`.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCRIPT = ROOT / "code" / "gates" / "claims.py"

# NOTE: the skipif belongs on the slow subprocess test ONLY. Applied module-wide (as a
# `pytestmark`) it also silences test_perturbation_helper_matches_the_measured_convention, an
# in-memory microsecond unit test with no reason to be gated -- and since nothing in the repo
# ever sets SLOW_GATES, that test would never run at all.
@pytest.mark.skipif(os.environ.get("SLOW_GATES") != "1",
                    reason="~5-minute measurement; set SLOW_GATES=1 to run")
def test_detection_coverage_ratchet_holds():
    r = subprocess.run([sys.executable, str(SCRIPT), "--coverage"], capture_output=True, text=True,
                       cwd=str(ROOT), timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS  detection-coverage ratchet" in r.stdout


def test_perturbation_helper_matches_the_measured_convention():
    sys.path.insert(0, str(ROOT / "code" / "gates"))
    import claims as CL
    assert CL._perturb_cited(11) == 12
    assert CL._perturb_cited(2.0) == 2.1
    assert CL._perturb_cited([1, 2.0]) == [2, 2.1]
    assert CL._perturb_cited(True) is None
    assert CL._perturb_cited("x") is None
    assert CL._perturb_cited([1, "a"]) is None
