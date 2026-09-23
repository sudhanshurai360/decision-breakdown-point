"""canonical_emit.py — the ONE way a stage records a number the paper cites.
"""
from __future__ import annotations
import inspect
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent   # <repo root>/code/instrument/ -> <repo root>
_FRAGDIR = _ROOT / "multiseries" / "results" / "canonical"
# Accumulated per calling module, flushed on write. Deliberately NOT cleared after flush(): a
# stage that calls flush() twice must write the same complete fragment both times, and a stage
# owns its fragment file completely (flush() overwrites it).
_PENDING: dict[str, dict] = {}


def _plain(v):
    """numpy -> json-native, preserving precision. No rounding here, ever."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return _plain(v.item())
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating, float)):
        v = float(v)
        if math.isinf(v):
            raise AssertionError(
                f"Infinity cannot be recorded in canonical.json: {v!r}. (NaN IS allowed here, as a "
                f"first-class 'not computable on this sample' value; an infinity is always a bug.)")
        return v
    return v


def _caller_stage() -> str:
    """The stage file that called emit()."""
    for fr in inspect.stack()[1:]:
        f = Path(fr.filename)
        if f.name != Path(__file__).name:
            return f.stem
    return "unknown"


def emit(key: str, value, *, spec: str, cites: list[str] | None = None,
         cross_check: str | None = None, **extra) -> None:
    """Record one value the manuscript cites."""
    if not set(extra).isdisjoint({"value", "producer", "spec", "cites", "cross_check"}):
        raise AssertionError(
            f"emit({key!r}): extra kwarg(s) {set(extra) & {'value', 'producer', 'spec', 'cites', 'cross_check'}} "
            f"collide with a protected field name and would silently clobber it -- rename the kwarg(s)")
    stage = _caller_stage()
    frag = _PENDING.setdefault(stage, {})
    if key in frag:
        raise ValueError(f"canonical key '{key}' emitted twice by {stage} — keys are unique")
    frag[key] = {
        "value": _plain(value),
        "producer": stage,
        "spec": spec,
        "cites": list(cites) if cites else [],
        "cross_check": cross_check,
        **{k: _plain(v) for k, v in extra.items()},
    }


def flush() -> Path | None:
    """Write this stage's fragment, overwriting any previous one: a stage owns its keys
    completely, so every emit() a stage makes must happen in the same process before flush()."""
    stage = _caller_stage()
    frag = _PENDING.get(stage)
    if not frag:
        raise RuntimeError(
            f"{stage} called flush() with nothing to write. Either it emitted no values (in which case "
            f"it should not call flush) or its emits did not run. The previous fragment is still on "
            f"disk and would be merged as if current.")
    _FRAGDIR.mkdir(parents=True, exist_ok=True)
    out = _FRAGDIR / f"{stage}.json"
    doc = {"_stage": stage,
           "_generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "_n": len(frag), **frag}
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    if out.exists():
        shutil.copymode(out, tmp)
    tmp.replace(out)   # atomic: a crash mid-write leaves the previous fragment intact, never a truncated one
    print(f"  canonical: {len(frag)} value(s) -> {out.relative_to(_ROOT)}")
    return out
