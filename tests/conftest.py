"""conftest.py -- path setup for the paper-pipeline test suite.

code/{instrument,pipeline,gates} import each other via ad-hoc sys.path.insert() (the pipeline's own
convention, e.g. regime_robustness.py's `sys.path.insert(0, str(HERE.parent / "instrument"))`) rather
than being an installed package. Mirror that here so `import fragility as F` etc. work the same way
inside tests as they do inside the real pipeline scripts.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("instrument", "pipeline", "gates"):
    p = str(ROOT / "code" / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
