#!/usr/bin/env python3
"""T04 deviation 12: QC that admits re-predicted / second-pass prediction files (fixed before any re-prediction existed).

Deviation: research/dg_mechanisms/plans/2026-11-14-T04-repredict-deviation.md. The registered QC (|Dice - stored| <= 1e-4 per
row, t04_analyze.py) stays the default for first-pass files; this script is the opt-in QC for the deviation-12 settings
(Glioma t1c / t2f, Pelvis ct / mri re-predicted on Vulcan; Glioma t1n / t2w whose files are a second pass).

Per setting x arm, over all (case, label, fold) rows of t04_signed's QC parquet with a stored value (t04_analyze.prep applied,
i.e. out-of-domain rows of the registered pool only):
  1. median |Dice - stored| <= 1e-3, 99th percentile <= 0.02, max <= 0.05;
  2. |mean signed diff| <= 1e-3 (no bias);
  3. per item and fold, mean Dice within 0.1 of the stored mean (same rows);
  4. every roster case predicted (prov: no missing case, arm directory found) and every prediction file on the GT grid
     (size, spacing, origin, direction read from the headers; spacing/origin tolerance 1e-3 mm, direction 1e-4).
A stored value that is finite while the new Dice is NaN (or the reverse) counts as |diff| = 1 (conservative; declared here).
A setting is admitted only if every scored arm (t04_arms.SCORED) is present and passes.

  t04_qc_dev12.py --dir DIR --settings Glioma_t1c,Glioma_t2f,Pelvis_ct,Pelvis_mri,Glioma_t1n,Glioma_t2w [--workers 8]
Reads DIR/qc/<s>.parquet, DIR/prov/<s>.tsv; writes DIR/qc_dev12.tsv (setting x arm) and DIR/qc_dev12_items.tsv (item x fold).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t04_arms as TA  # noqa: E402
from t04_analyze import prep  # noqa: E402

TOL = dict(median=1e-3, p99=0.02, max=0.05, bias=1e-3, item_mean=0.1)
GRID_TOL_MM, GRID_TOL_DIR = 1e-3, 1e-4


def _hdr(p: Path):
    r = sitk.ImageFileReader()
    r.SetFileName(str(p))
    r.ReadImageInformation()
    return r.GetSize(), np.array(r.GetSpacing()), np.array(r.GetOrigin()), np.array(r.GetDirection())


def grid_check(task):
    """(n_checked, n_bad, first bad message) for one prediction directory vs the GT directory."""
    pdir, gdir, cases = task
    n, bad, msg = 0, 0, ""
    for c in cases:
        pf, gf = Path(pdir) / f"{c}.nii.gz", Path(gdir) / f"{c}.nii.gz"
        if not pf.exists():
            continue  # completeness is counted from prov
        n += 1
        a, b = _hdr(pf), _hdr(gf)
        ok = (a[0] == b[0] and np.allclose(a[1], b[1], atol=GRID_TOL_MM) and np.allclose(a[2], b[2], atol=GRID_TOL_MM)
              and np.allclose(a[3], b[3], atol=GRID_TOL_DIR))
        if not ok:
            bad += 1
            msg = msg or f"{Path(pdir).name}: size {a[0]} vs {b[0]}"
    return n, bad, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    D = Path(a.dir)
    S = TA.settings()
    rows, item_rows, grid_tasks, grid_keys = [], [], [], []
    for s in a.settings.split(","):
        st = S[s]
        Q = prep(pd.read_parquet(D / "qc" / f"{s}.parquet"), s)
        P = pd.read_csv(D / "prov" / f"{s}.tsv", sep="\t")
        P = P[(P.fold >= 0) & P.item_key.isin(Q.item_key.unique())]
        items = {it["key"]: it for it in st["items"]}
        for arm in sorted(Q.arm.unique()):
            g = Q[Q.arm == arm]
            w = g[g.stored_dice.notna()]
            both = w[w.dice.notna()]
            d = (both.dice - both.stored_dice).to_numpy()
            n_nan = int(w.dice.isna().sum())
            ad = np.r_[np.abs(d), np.ones(n_nan)]
            per = (both.groupby(["item_key", "fold"])[["dice", "stored_dice"]].mean()
                   .assign(diff=lambda x: x.dice - x.stored_dice).reset_index())
            for r in per.itertuples(index=False):
                item_rows.append(dict(setting=s, arm=arm, item_key=r.item_key, fold=int(r.fold), mean_dice=r.dice,
                                      mean_stored=r.stored_dice, diff=r.diff))
            pa = P[P.arm == arm]
            n_missing = int(pa.missing_cases.sum()) + int((pa.dir == "ABSENT").sum())
            hows = ",".join(sorted(pa.how.astype(str).str.split(" ").str[0].unique()))
            row = dict(setting=s, arm=arm, role="scored" if arm in TA.SCORED else "reported", pred_source=hows,
                       n_rows=len(g), n_with_stored=len(w), n_new_nan_stored_finite=n_nan,
                       n_stored_nan_new_finite=int((g.stored_dice.isna() & g.dice.notna()).sum()),
                       median_abs=float(np.median(ad)) if ad.size else np.nan,
                       p99_abs=float(np.percentile(ad, 99)) if ad.size else np.nan,
                       max_abs=float(ad.max()) if ad.size else np.nan,
                       mean_signed=float(d.mean()) if d.size else np.nan,
                       max_item_fold_mean_diff=float(per["diff"].abs().max()) if len(per) else np.nan,
                       n_missing_cases=n_missing)
            rows.append(row)
            for r in pa.itertuples(index=False):
                if r.dir == "ABSENT":
                    continue
                cases = sorted(g[(g.item_key == r.item_key) & (g.fold == r.fold)].case.unique())
                it = items[r.item_key]
                gdir = Path(it["gt_dir"]) if it["gt_dir"].startswith("/") else TA.R / it["gt_dir"]
                grid_tasks.append((r.dir, str(gdir), cases))
                grid_keys.append((s, arm))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(grid_check, grid_tasks, chunksize=4))
    G = {}
    for k, (n, bad, msg) in zip(grid_keys, res):
        x = G.setdefault(k, [0, 0, ""])
        x[0] += n; x[1] += bad; x[2] = x[2] or msg
    for r in rows:
        n, bad, msg = G.get((r["setting"], r["arm"]), [0, 0, ""])
        r.update(n_grid_checked=n, n_grid_bad=bad, grid_note=msg)
        r["r1_abs"] = bool(r["median_abs"] <= TOL["median"] and r["p99_abs"] <= TOL["p99"] and r["max_abs"] <= TOL["max"])
        r["r2_bias"] = bool(abs(r["mean_signed"]) <= TOL["bias"])
        r["r3_item_mean"] = bool(r["max_item_fold_mean_diff"] <= TOL["item_mean"])
        r["r4_complete_grid"] = bool(r["n_missing_cases"] == 0 and r["n_grid_bad"] == 0 and n > 0 and r["n_with_stored"] > 0)
        r["passed"] = bool(r["r1_abs"] and r["r2_bias"] and r["r3_item_mean"] and r["r4_complete_grid"])
    T = pd.DataFrame(rows)
    adm = {}
    for s, g in T.groupby("setting"):
        sc = g[g.role == "scored"]
        adm[s] = bool(set(TA.SCORED) <= set(sc.arm) and sc.passed.all())
    T["setting_admitted"] = T.setting.map(adm)
    T.to_csv(D / "qc_dev12.tsv", sep="\t", index=False)
    pd.DataFrame(item_rows).to_csv(D / "qc_dev12_items.tsv", sep="\t", index=False)
    cols = ["setting", "arm", "role", "pred_source", "n_with_stored", "median_abs", "p99_abs", "max_abs", "mean_signed",
            "max_item_fold_mean_diff", "n_missing_cases", "n_grid_bad", "passed", "setting_admitted"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(T[cols].to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print("[qc_dev12] admitted:", {s: v for s, v in adm.items()})


if __name__ == "__main__":
    main()
