#!/usr/bin/env python3
"""Acceptance QC of in-domain test predictions (research/dg_mechanisms/plans/2026-11-16-repredict-indomain.md, sha256
d5692f43..., tag reg-T10-T28-20261116), fixed before any prediction existed. Same rule as T04 deviation 12 + clarification.

Per (setting, arm), over all (case, label, fold) rows of the setting's own in-domain item with a stored value in the stored
eval_all.csv of the same run (t04_signed.stored_dice, the T04 lookup):
  1. median |Dice - stored| <= 1e-3, 99th percentile <= 0.02, max <= 0.05;
  2. |mean signed diff| <= 1e-3;
  3. per fold, mean Dice within 0.1 of the stored mean (same rows);
  4. every stored case predicted in every fold, every prediction file on the GT grid (t04_qc_dev12.grid_check: size equal,
     spacing / origin within 1e-3 mm, direction within 1e-4).
A row stored finite / new NaN counts |diff| = 1 (clarification 1); stored NaN / new finite is counted and reported, not scored.
Tolerances and the grid check are imported from t04_qc_dev12 (TOL, grid_check); recomputed label Dice comes from
t11_cases.process (the T11 / E1 loaders: scoring.load_gt, scoring.read_stored_pred, gd2_lib.dice_from_counts on the item's
label triples). Prediction lookup: headline arms t04_arms.pred_dir with t04_arms.STAGE pointed at --stage (read only when the
repo tree has no hit); fill arms (texture = Palette) decompose_v2.pred_dir.

  qc_indomain.py --stage DIR --out DIR --pairs Glioma_t1c:synthseg_noEM,... [--workers 8]
Writes OUT/qc_indomain.tsv (setting x arm), OUT/qc_indomain_folds.tsv (fold means), OUT/qc_label_rows.parquet (untracked).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DGM = HERE.parent
sys.path.insert(0, str(DGM / "t04_arm_boundary"))
sys.path.insert(0, str(DGM / "t11_noem_errors"))
import t04_arms as TA  # noqa: E402
import t04_signed as TS  # noqa: E402
from t04_qc_dev12 import TOL, grid_check  # noqa: E402
import t11_cases as T11C  # noqa: E402

dv = TA.dv


def _init(stage: str):
    TA.STAGE = Path(stage)


def locate(st, it, arm, fold):
    """(run id, prediction dir or None, how)."""
    if arm in TA.FILL:
        rid = dv.run_id(st, arm)
        p, how = dv.pred_dir(st, it, arm, fold)
    else:
        rid = TA.run_ids(st)[arm]
        p, how = TA.pred_dir(st, it, arm, rid, fold)
    return rid, p, how


def run_case(task):
    out = T11C.process(task)
    s, arm = task["setting"].split(":", 1)
    rows = [dict(r, setting=s, arm=arm) for r in out["label_rows"]]
    return dict(label_rows=rows, errors=out["errors"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    _init(a.stage)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = TA.settings()
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",") if x]
    meta, tasks, grid_tasks, stored_rows = {}, [], [], []
    for s, arm in pairs:
        st = S[s]
        it = next(x for x in st["items"] if x["group"] == "in-domain")
        gdir = Path(it["gt_dir"]) if it["gt_dir"].startswith("/") else TA.R / it["gt_dir"]
        m = dict(setting=s, arm=arm, item_key=it["key"], n_stored_cases=0, n_missing_case_folds=0, hows=set(), csvs=set(),
                 dirs={})
        by_case = {}
        for f in dv.FOLDS:
            rid, p, how = locate(st, it, arm, f)
            m["run_id"] = rid
            d, src = TS.stored_dice(st, it, rid, f)
            m["csvs"].add(str(src))
            stored_rows += [dict(setting=s, arm=arm, case=c, fold=f, label=l, stored_dice=v) for (c, l), v in d.items()]
            cs = sorted({c for c, _ in d})
            m["n_stored_cases"] += len(cs)
            m["hows"].add(how.split(" ")[0])
            have = {x.name[:-7] for x in p.glob("*.nii.gz")} if p else set()
            m["n_missing_case_folds"] += len(set(cs) - have)
            m["dirs"][f] = str(p) if p else "ABSENT"
            if p is not None:
                grid_tasks.append(((s, arm), (str(p), str(gdir), cs)))
            for c in cs:
                if c in have:
                    by_case.setdefault(c, {})[f] = str(p)
        for c, folds in sorted(by_case.items()):
            tasks.append(dict(setting=f"{s}:{arm}", item=it, case=c, folds=folds))
        meta[(s, arm)] = m
        print(f"[qc_indomain] {s} {arm}: run {m['run_id']}, {m['n_stored_cases']} stored case-folds,"
              f" {m['n_missing_case_folds']} missing, how {sorted(m['hows'])}", flush=True)
    label_rows, errs = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(run_case, tasks, chunksize=2):
            label_rows += r["label_rows"]
            errs += r["errors"]
        grid = list(pool.map(grid_check, [g for _, g in grid_tasks], chunksize=1))
    L = pd.DataFrame(label_rows, columns=["setting", "arm", "case", "fold", "label", "dice"])
    ST = pd.DataFrame(stored_rows)
    Q = ST.merge(L, on=["setting", "arm", "case", "fold", "label"], how="left")
    Q.to_parquet(out / "qc_label_rows.parquet")
    G = {}
    for (k, _), (n, bad, msg) in zip(grid_tasks, grid):
        x = G.setdefault(k, [0, 0, ""])
        x[0] += n; x[1] += bad; x[2] = x[2] or msg
    rows, fold_rows = [], []
    for (s, arm), m in meta.items():
        g = Q[(Q.setting == s) & (Q.arm == arm)]
        w = g[g.stored_dice.notna()]
        both = w[w.dice.notna()]
        d = (both.dice - both.stored_dice).to_numpy()
        n_nan = int(w.dice.isna().sum())
        ad = np.r_[np.abs(d), np.ones(n_nan)]
        per = (both.groupby("fold")[["dice", "stored_dice"]].mean().assign(diff=lambda x: x.dice - x.stored_dice).reset_index())
        for r in per.itertuples(index=False):
            fold_rows.append(dict(setting=s, arm=arm, fold=int(r.fold), mean_dice=r.dice, mean_stored=r.stored_dice, diff=r.diff))
        n, bad, msg = G.get((s, arm), [0, 0, ""])
        row = dict(setting=s, arm=arm, run_id=m["run_id"], item_key=m["item_key"], pred_source=",".join(sorted(m["hows"])),
                   stored_csvs=";".join(sorted(m["csvs"])), n_rows=len(g), n_with_stored=len(w), n_new_nan_stored_finite=n_nan,
                   n_stored_nan_new_finite=int((g.stored_dice.isna() & g.dice.notna()).sum()),
                   median_abs=float(np.median(ad)) if ad.size else np.nan,
                   p99_abs=float(np.percentile(ad, 99)) if ad.size else np.nan,
                   max_abs=float(ad.max()) if ad.size else np.nan,
                   mean_signed=float(d.mean()) if d.size else np.nan,
                   max_fold_mean_diff=float(per["diff"].abs().max()) if len(per) else np.nan,
                   n_stored_case_folds=m["n_stored_cases"], n_missing_case_folds=m["n_missing_case_folds"],
                   n_grid_checked=n, n_grid_bad=bad, grid_note=msg)
        row["r1_abs"] = bool(row["median_abs"] <= TOL["median"] and row["p99_abs"] <= TOL["p99"] and row["max_abs"] <= TOL["max"])
        row["r2_bias"] = bool(abs(row["mean_signed"]) <= TOL["bias"])
        row["r3_fold_mean"] = bool(row["max_fold_mean_diff"] <= TOL["item_mean"])
        row["r4_complete_grid"] = bool(row["n_missing_case_folds"] == 0 and bad == 0 and n == row["n_stored_case_folds"]
                                       and row["n_with_stored"] > 0)
        row["passed"] = bool(row["r1_abs"] and row["r2_bias"] and row["r3_fold_mean"] and row["r4_complete_grid"])
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(out / "qc_indomain.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "qc_indomain_folds.tsv", sep="\t", index=False)
    cols = ["setting", "arm", "pred_source", "n_with_stored", "median_abs", "p99_abs", "max_abs", "mean_signed",
            "max_fold_mean_diff", "n_missing_case_folds", "n_grid_bad", "passed"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(T[cols].to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    for e in errs:
        print("[error]", e, flush=True)
    print(f"[qc_indomain] passed: {dict(zip(T.setting + ':' + T.arm, T.passed))}; errors {len(errs)}")
    sys.exit(3 if errs else 0)


if __name__ == "__main__":
    main()
