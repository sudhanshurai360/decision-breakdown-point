"""
breadth_fetch.py — fetch a PRE-COMMITTED universe of monthly macro series for the breadth appendix, apply the
VALIDITY FILTER (vintage-depth and rebasing checks; see `validity()`), and keep only passers.
"""
import sys, json, hashlib, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/

RAW = MULTI / "data" / "raw"
sys.path.insert(0, str(HERE))
import fetch_alfred as FA

CANDIDATES_DISTINCT = {
    "PAYEMS":   "Labor: nonfarm payrolls",              "UNEMPLOY": "Labor: unemployment level (household)",
    "HOUST":    "Housing: starts",                      "RSAFS":    "Consumption: retail & food sales",
    "DGORDER":  "Manufacturing: durable-goods orders",  "TCU":      "Industrial: capacity utilization (%)",
    "PCE":      "Consumption: personal consumption ($)",
    "PI":       "Income: personal income ($)",          "BOPGEXP":  "Trade: goods exports ($)",
    "BOPGIMP":  "Trade: goods imports ($)",             "M2SL":     "Money: M2 ($)",
    "BUSINV":   "Inventories: total business inv ($)",  "TTLCONS":  "Construction: total spending ($)",
}
# Heavily collinear — 'robust-to-disaggregation' exhibit ONLY. They do NOT all sum to PAYEMS: only
# the pair {USGOOD, SRVPRD} partitions it. Everything else is nested inside one of those two (MANEMP
# /DMANEMP/NDMANEMP/USCONS/USMINE within USGOOD; USTRADE/USWTRADE/USTPU/USFIRE/USINFO/USPBS/USEHS/
# USLAH/USGOVT within SRVPRD, and USTRADE/USWTRADE nest again inside USTPU), and CE16OV is not a CES
# payroll series at all — it is CPS household employment, a different survey with a different universe.
CANDIDATES_EMPLOYMENT = {
    "MANEMP": "Mfg emp", "DMANEMP": "Durable-mfg emp", "NDMANEMP": "Nondurable-mfg emp", "USCONS": "Construction emp",
    "USGOOD": "Goods-producing emp", "SRVPRD": "Service-providing emp", "USTRADE": "Retail-trade emp",
    "USWTRADE": "Wholesale-trade emp", "USTPU": "Trade/transport/util emp", "USFIRE": "Financial emp",
    "USINFO": "Information emp", "USPBS": "Prof/business emp", "USEHS": "Education/health emp",
    "USLAH": "Leisure/hospitality emp", "USGOVT": "Government emp", "USMINE": "Mining/logging emp", "CE16OV": "Civilian emp",
}

VINT_COVER = pd.Timestamp("2007-06-01")   # vintages must reach here
PROBE = pd.Timestamp("2005-01-01")        # rebasing probe month
REBASE_TOL = 0.15                          # |real-time/realized - 1| must be < this


def validity(df):
    rs = df["realtime_start"]; vmin = rs.min()
    if pd.isna(vmin):
        return False, "empty fetch result (0 rows, no realtime_start)", None, vmin
    if vmin > VINT_COVER:
        return False, f"vintages_from={vmin.date()} (> {VINT_COVER:%Y-%m})", None, vmin
    realized = df.sort_values("realtime_start").groupby("date").last()["value"]
    early = df[df["date"] == PROBE].sort_values("realtime_start")
    if not len(early) or PROBE not in realized.index:
        return True, "ok (no probe)", None, vmin
    ratio = float(early["value"].iloc[0] / realized.loc[PROBE])
    if np.isnan(ratio):
        return False, "rebased? ratio=nan (undefined probe/realized value)", ratio, vmin
    if abs(ratio - 1) > REBASE_TOL:
        return False, f"rebased? ratio={ratio:.2f}", ratio, vmin
    return True, "ok", ratio, vmin


def load_or_fetch(sid, out, fetch_fn):
    """Return (df, reused): the cached parquet at `out` if it exists, else the result of `fetch_fn()`."""
    if out.exists():
        df = pd.read_parquet(out)
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["date"] = pd.to_datetime(df["date"])
        return df, True
    return fetch_fn(), False


def main():
    from fredapi import Fred
    unknown = [a for a in sys.argv[1:] if a != "--employment"]
    if unknown:
        print(f"  UNKNOWN ARGUMENT(S): {unknown} -- refusing to silently ignore a possible typo. "
              f"Recognized: --employment.", file=sys.stderr)
        sys.exit(2)
    which = "employment" if "--employment" in sys.argv else "distinct"
    SET = CANDIDATES_EMPLOYMENT if which == "employment" else CANDIDATES_DISTINCT
    fred = Fred(api_key=FA._fred_key())
    RAW.mkdir(parents=True, exist_ok=True)
    man = {"pulled_at": dt.datetime.now().isoformat(timespec="seconds"), "universe": which,
           "criteria": {"vintages_cover": str(VINT_COVER.date()), "rebase_tol": REBASE_TOL,
                        "probe_month": str(PROBE.date())}, "series": {}}
    passed = []
    for sid, label in SET.items():
        out = RAW / f"{sid}_alfred_vintages.parquet"
        try:
            df, reused = load_or_fetch(sid, out, lambda: FA.fetch_one(sid, fred))
        except Exception as e:
            man["series"][sid] = {"label": label, "status": f"FETCH_ERR {e}"}; print(f"{sid:9s} FETCH_ERR {e}"); continue
        ok, reason, ratio, vmin = validity(df)
        if reused and ok:
            reason = f"already present, {reason}"
        rec = {"label": label, "status": "PASS" if ok else "DROP", "reason": reason,
               "n_rows": int(len(df)), "vintage_min": None if pd.isna(vmin) else str(vmin.date()),
               "ratio_2005": round(ratio, 3) if ratio is not None else None}
        if ok:
            if not reused:
                df.to_parquet(out, index=False)
            rec["sha16"] = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
            passed.append(sid)
        man["series"][sid] = rec
        print(f"{sid:9s} {'PASS' if ok else 'DROP':4s} {reason:26s} vintages_from={rec['vintage_min']}")
    man["passed"] = passed; man["n_passed"] = len(passed)
    (RAW / f"breadth_manifest_{which}.json").write_text(json.dumps(man, indent=2))
    print(f"\n{len(passed)}/{len(SET)} passed [{which}] -> {passed}")


if __name__ == "__main__":
    main()
