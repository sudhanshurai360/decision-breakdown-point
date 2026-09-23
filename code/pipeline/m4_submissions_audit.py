"""
m4_submissions_audit.py — re-score every frozen M4 competition submission from the public archive and
check whether the leaderboard's top-1 flips across sMAPE, MAPE, MASE and RMSSE.

Downloads the submissions and the M4 train/test sets over the network, so it needs an internet
connection and room on disk for the archives. Extraction shells out to the `unar` binary (The
Unarchiver) -- not RARLab's `unrar`, which is a different tool and will not satisfy this dependency.
"""
import sys, json, subprocess, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

DATA = MULTI / "data" / "m4_sub"
RESULTS = MULTI / "results"
sys.path.insert(0, str(INSTR))
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
REPO = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master"
API = "https://api.github.com/repos/Mcompetitions/M4-methods/contents/Point%20Forecasts"
FREQ = {"Y": ("Yearly", 1, 6), "Q": ("Quarterly", 4, 8), "M": ("Monthly", 12, 18),
        "W": ("Weekly", 1, 13), "D": ("Daily", 1, 14), "H": ("Hourly", 24, 48)}   # prefix -> (name, m, h)


def _get(url, dest):
    if not Path(dest).exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def list_submissions():
    r = json.load(urllib.request.urlopen(urllib.request.Request(API, headers={"User-Agent": "x"}), timeout=30))
    return [e["name"] for e in r if e["name"].endswith(".rar")]


def _leading_run(row):
    """The row's leading run of non-NaN values, as a float array. M4's CSVs are ragged -- short
    series are right-padded with NaN -- so the prefix before the first NaN is the real series;
    dropping every NaN instead would silently close an interior gap and shift later values forward."""
    arr = row.to_numpy(float)
    nan_idx = np.flatnonzero(np.isnan(arr))
    return arr[:nan_idx[0]] if len(nan_idx) else arr


def load_data():
    """id -> (actuals, m, scale_abs, scale_sq)."""
    DATA.mkdir(parents=True, exist_ok=True)
    truth = {}
    for pfx, (name, m, h) in FREQ.items():
        te = _get(f"{REPO}/Dataset/Test/{name}-test.csv", DATA / f"{name}-test.csv")
        tr = _get(f"{REPO}/Dataset/Train/{name}-train.csv", DATA / f"{name}-train.csv")
        dte = pd.read_csv(te); dte = dte.set_index(dte.columns[0])
        dtr = pd.read_csv(tr); dtr = dtr.set_index(dtr.columns[0])
        for sid, row in dte.iterrows():
            yt = _leading_run(row)
            ytr = _leading_run(dtr.loc[sid])
            dd = np.abs(ytr[m:] - ytr[:-m]) if (m > 1 and len(ytr) > m) else np.abs(np.diff(ytr))
            sa = dd.mean() if len(dd) and dd.mean() > 0 else (np.abs(ytr).mean() + 1e-9)
            dsq = ((ytr[m:] - ytr[:-m]) ** 2) if (m > 1 and len(ytr) > m) else (np.diff(ytr) ** 2)
            ssq = dsq.mean() if len(dsq) and dsq.mean() > 0 else sa ** 2
            truth[str(sid)] = (yt, m, float(sa), float(ssq))
    return truth


def unrar(rar, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unar", "-q", "-f", "-o", str(outdir), str(rar)], check=True)   # unar (macOS); handles .rar
    return list(outdir.rglob("*.csv")) + list(outdir.rglob("*.txt"))


def parse_submission(csvs):
    """Return id -> forecast array. Standard M4 layout: first col = id (Y1/Q1/...), rest = numeric forecasts."""
    for f in sorted(csvs, key=lambda p: -p.stat().st_size):      # biggest file = the forecasts
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  [skip] {f.name}: failed to parse as CSV: {e}")
            continue
        idcol = d.columns[0]
        ids = d[idcol].astype(str)
        if not ids.str.match(r"^[YQMWDH]\d+$").mean() > 0.5:     # first col must look like M4 ids
            print(f"  [skip] {f.name}: first column '{idcol}' doesn't look like M4 ids, trying next candidate")
            continue
        num = d.drop(columns=[idcol]).apply(pd.to_numeric, errors="coerce")
        return {str(i): _leading_run(row) for i, (_, row) in zip(ids, num.iterrows())}
    return {}


def score(y_pred, y_true, scale_abs, scale_sq):
    """SCALE-FREE per-series metrics (so the cross-series mean is meaningful): sMAPE, MAPE, MASE, RMSSE."""
    n = min(len(y_pred), len(y_true))
    if n == 0:
        return None
    yp, yt = y_pred[:n], y_true[:n]; e = yt - yp; ae = np.abs(e)
    dm = np.abs(yt) > 0; ds = (np.abs(yt) + np.abs(yp)) > 0
    return {"sMAPE": np.mean(2 * ae[ds] / (np.abs(yt) + np.abs(yp))[ds]) if ds.any() else np.nan,
            "MAPE": np.mean(ae[dm] / np.abs(yt)[dm]) if dm.any() else np.nan,
            "MASE": ae.mean() / scale_abs,
            "RMSSE": np.sqrt((e ** 2).mean() / scale_sq)}


def main():
    known = {"--inspect", "--limit"}
    unknown = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--limit":
            if i + 1 >= len(sys.argv) or not sys.argv[i + 1].isdigit():
                unknown.append(a if i + 1 >= len(sys.argv) else f"{a} {sys.argv[i + 1]}")
                i += 2  # consume the (invalid) value token too, so it isn't double-reported
                continue
            i += 2  # consume the flag and its numeric value together, not as 2 separate tokens
            continue
        if a not in known:
            unknown.append(a)
        i += 1
    if unknown:
        print(f"  UNKNOWN ARGUMENT(S): {unknown} -- refusing to silently fall back to a default "
              f"(a typo here would otherwise silently audit ALL submissions instead of the intended "
              f"subset). Recognized: --inspect, --limit N.", file=sys.stderr)
        sys.exit(2)
    inspect = "--inspect" in sys.argv
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    DATA.mkdir(parents=True, exist_ok=True)
    subs = list_submissions()
    print(f"{len(subs)} submissions on the repo")
    if inspect:
        s = subs[0]; rar = _get(f"{REPO}/Point%20Forecasts/{s}", DATA / s)
        files = unrar(rar, DATA / "inspect")
        print(f"--- {s} contains: ---")
        for f in files[:20]:
            print("  ", f.name, f.stat().st_size // 1024, "KB")
            try:
                print("     head:", pd.read_csv(f, nrows=2).iloc[:, :4].to_dict("records"))
            except Exception as e:
                print("     (parse:", e, ")")
        return
    truth = load_data(); print(f"loaded {len(truth)} test series")
    if limit:
        subs = subs[:limit]
    METRICS = ["sMAPE", "MAPE", "MASE", "RMSSE"]
    board = {}                          # submission -> {metric: mean score across its series}
    for s in subs:
        try:
            rar = _get(f"{REPO}/Point%20Forecasts/{s}", DATA / s)
            fc = parse_submission(unrar(rar, DATA / s.replace(".rar", "")))
        except Exception as e:
            print(f"  [skip] {s}: {e}"); continue
        rows = []
        for sid, yp in fc.items():
            if sid in truth:
                yt, _, sa, ssq = truth[sid]; sc = score(yp, yt, sa, ssq)
                if sc: rows.append(sc)
        if len(rows) > 100:
            df = pd.DataFrame(rows); board[s] = {mt: df[mt].mean() for mt in METRICS}
            print(f"  {s}: scored {len(rows)} series")
        else:
            print(f"  [drop] {s}: only {len(rows)} series scored (<=100), excluded from leaderboard")
    B = pd.DataFrame(board).T
    RM._atomic_to_csv(B, RESULTS / "m4_submissions_leaderboard.csv", index=True)
    print("\n=== LEADERBOARD TOP-1 PER METRIC (does the winner FLIP?) ===")
    for mt in METRICS:
        w = B[mt].idxmin(); print(f"  {mt:6s} winner: {w}  ({B[mt].min():.4f})")
    winners = {B[mt].idxmin() for mt in METRICS}
    print(f"\n  >>> distinct top-1 across metrics: {len(winners)}  ({'FLIPS' if len(winners) > 1 else 'STABLE'})")
    rank_corr = float(B["sMAPE"].rank().corr(B["RMSSE"].rank()))
    print(f"  full-leaderboard rank corr sMAPE vs RMSSE: {rank_corr:.3f}")
    print(f"  top-5 by sMAPE:  {list(B['sMAPE'].nsmallest(5).index)}")
    print(f"  top-5 by RMSSE:  {list(B['RMSSE'].nsmallest(5).index)}")
    print(f"  wrote {RESULTS/'m4_submissions_leaderboard.csv'}")

    if limit:
        print("  (--limit run: skipping canonical emit -- this is a partial-leaderboard proof run, not "
              "the full audit the manuscript cites)")
    else:
        emit("m4sub.n_submissions", len(B), spec="number of M4 competition submissions successfully re-scored",
             cites=["S9: Re-scoring all 61 real M4 submissions"],
             cross_check="m4_submissions_leaderboard.csv: len(B)")
        emit("m4sub.distinct_winners", len(winners), of=len(METRICS),
             spec="distinct top-1 submissions across the 4 re-scoring metrics (sMAPE, MAPE, MASE, RMSSE)",
             cites=["S9: it wins on all four standard accuracy metrics simultaneously"],
             cross_check="m4_submissions_leaderboard.csv: {B[mt].idxmin() for mt in METRICS}")
        emit("m4sub.pareto_dominant", len(winners) == 1, n_metrics=len(METRICS),
             spec="the single submission that is the argmin on every one of the 4 metrics simultaneously "
                  "cannot be dominated by any other submission on all metrics at once",
             cites=["S9: is dominated by none"],
             cross_check="m4_submissions_leaderboard.csv: len({B[mt].idxmin() for mt in METRICS}) == 1")
        emit("m4sub.rank_corr_smape_rmsse", rank_corr, spec="Spearman rank correlation of the full leaderboard under sMAPE vs RMSSE",
             cites=["S9 body: board-wide rank correlation complement to Pareto dominance"],
             cross_check="m4_submissions_leaderboard.csv: B['sMAPE'].rank().corr(B['RMSSE'].rank())")
        flush()


if __name__ == "__main__":
    main()
