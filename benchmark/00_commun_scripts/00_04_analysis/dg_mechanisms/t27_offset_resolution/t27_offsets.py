#!/usr/bin/env python3
"""T27 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...):
does the shared inward offset track voxel size (M16) or is it a per-task constant (M17)?

Readout (from the T04 row parquets, read-only): set 'per_arm' (T04's primary matching), pool = t04_analyze.prep (E1's group
remap and pool restriction, in-domain dropped). Per item (item_key), arm and fold: mean over cases of the per-case value,
the per-case value being the mean over labels of the signed ASD (E1's agg, at item instead of group level). Per item and
arm: mean over folds 0-2. O(s, i) = median over the randomised arms (t04_arms.SCORED minus baseline). The plain model
(baseline) is kept apart for secondary (d).
Registered settings read from T04/rows/, deviation-12 settings from T04/dev12/rows/.

Predictors per item, fixed before any value was read:
  t(i), p(i): median over the item's test cases (the cases in the T04 rows) of the largest / smallest voxel spacing (mm) of
              the image header. Image = the native image where the item carries pred_transform.ref_dir (Brain), else the
              model input (input_dir/<case>_0000.nii.gz). The GT-grid spacing (scoring grid) is recorded alongside.
  training contrast: the same over the training images (imagesTr/*_0000.nii.gz next to the in-domain item's input_dir).
  target size: median over test cases of the GT volume (mL) of the union of the scored labels (the item's triples).
Gate: each arm's setting value recomputed from the parquet through the T04 path (prep, t04_analyze.fold_values, mean over
folds) equals per_setting.tsv (set per_arm) within 0.05 mm, and so does the median over the randomised arms.
Counts: cases per item in the parquet == n_cases of T04's prov table for that item.
Usage (Slurm): .venv/bin/python t27_offsets.py --out DIR [--workers 8]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t04_arm_boundary"))
import t04_analyze as T4A  # noqa: E402  (puts t04_arms, E1, v2 on sys.path)
import t04_arms as TA  # noqa: E402

dv = TA.dv
T04 = REPO / "benchmark/01_commun_results/dg_mechanisms/T04"
REGISTERED = ["MS_flair", "Brain_T1w", "Brain_T2w", "Brain_dwi_ap", "Mandible_cbct", "Pancreas_t1wce", "Pancreas_t2w",
              "Abdomen_t1in", "Abdomen_t2spir"]
DEV12 = ["Glioma_t1c", "Glioma_t1n", "Glioma_t2w", "Pelvis_mri"]
RAND = [a for a in TA.SCORED if a != "baseline"]
TOL = 0.05


def src_dir(s: str) -> Path:
    return T04 if s in REGISTERED else T04 / "dev12"


def item_values(sub: pd.DataFrame) -> pd.DataFrame:
    """rows (item_key, arm, fold) -> mean over cases of the per-case mean over labels of asd."""
    c = sub.groupby(["item_key", "arm", "fold", "case"])["asd"].mean().reset_index()
    n = c.groupby(["item_key", "arm", "fold"]).agg(v=("asd", "mean"), n_cases=("case", "nunique"),
                                                     n_cases_defined=("asd", lambda x: int(x.notna().sum())))
    return n.reset_index()


def spacing(p: Path):
    z = nib.load(str(p)).header.get_zooms()[:3]
    return float(max(z)), float(min(z))


def case_info(task):
    """(item_key, case, t_img, p_img, t_gt, p_gt, target_ml) for one test case."""
    it, case = task["item"], task["case"]
    tr = it.get("pred_transform") or {}
    img = (REPO / tr["ref_dir"] / f"{case}{tr['ref_suffix']}") if tr.get("ref_dir") else (REPO / it["input_dir"] / f"{case}_0000.nii.gz")
    out = dict(item_key=it["key"], case=case, img=str(img))
    try:
        out["t_img"], out["p_img"] = spacing(img)
        g_img, g = dv.scoring.load_gt(dv.R, it, case)
        sp = g_img.GetSpacing()
        out["t_gt"], out["p_gt"] = float(max(sp)), float(min(sp))
        ids = sorted({int(x) for t in it["triples"] for x in (t[2] if isinstance(t[2], list) else [t[2]])})
        out["target_ml"] = float(dv.binarize(g, ids).sum() * np.prod(sp) / 1000.0)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def train_info(path: str):
    try:
        t, p = spacing(Path(path))
        return dict(path=path, t=t, p=p)
    except Exception as e:  # noqa: BLE001
        return dict(path=path, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = TA.settings()
    gate, items, counts, tasks, train_tasks = [], [], [], [], {}
    for s in REGISTERED + DEV12:
        d = src_dir(s)
        st = S[s]
        R = T4A.prep(pd.read_parquet(d / "rows" / f"{s}.parquet"), s)
        sub = R[R.set == "per_arm"]
        P = pd.read_csv(d / "per_setting.tsv", sep="\t")
        P = P[(P.setting == s) & (P.set == "per_arm")].set_index("arm")
        # gate: T04 path, per arm
        mine = {}
        for arm in TA.SCORED:
            fv = T4A.fold_values(sub, arm)
            mine[arm] = float(np.mean(fv))
            pub = float(P.loc[arm, "mean_asd"]) if arm in P.index else np.nan
            gate.append(dict(setting=s, source=d.name if d != T04 else "T04", arm=arm, recomputed=mine[arm],
                             per_setting_tsv=pub, diff=mine[arm] - pub, ok=bool(abs(mine[arm] - pub) <= TOL)))
        med_m = float(np.median([mine[x] for x in RAND]))
        med_p = float(np.median([float(P.loc[x, "mean_asd"]) for x in RAND]))
        gate.append(dict(setting=s, source=d.name if d != T04 else "T04", arm="median_randomised", recomputed=med_m,
                         per_setting_tsv=med_p, diff=med_m - med_p, ok=bool(abs(med_m - med_p) <= TOL)))
        # items
        iv = item_values(sub)
        prov = pd.read_csv(d / "prov" / f"{s}.tsv", sep="\t")
        itmeta = {it["key"]: it for it in st["items"]}
        for ik, g in iv.groupby("item_key"):
            fm = g.groupby("arm")["v"].mean()                       # mean over folds per arm
            nf = g.groupby("arm")["fold"].nunique()
            ncase = int(g["n_cases"].max())
            pv = prov[(prov.item_key == ik)]
            n_prov = int(pv.n_cases.max()) if len(pv) else -1
            counts.append(dict(setting=s, item_key=ik, n_cases_rows=ncase, n_cases_prov=n_prov,
                               min_cases_any_arm_fold=int(g["n_cases"].min()), ok=bool(ncase == n_prov)))
            row = dict(setting=s, task=st["task"], group_type=st["group"], item_key=ik,
                       group=str(sub[sub.item_key == ik].group.iloc[0]), n_cases=ncase,
                       O=float(np.median([fm[x] for x in RAND if x in fm.index])), n_rand_arms=int(sum(x in fm.index for x in RAND)),
                       all_folds=bool((nf.reindex(RAND) == 3).all()),
                       O_plain=float(fm["baseline"]) if "baseline" in fm.index else np.nan)
            row.update({f"O_{x}": float(fm[x]) for x in TA.SCORED if x in fm.index})
            items.append(row)
            it = itmeta[ik]
            tasks += [dict(item=it, case=c) for c in sorted(sub[sub.item_key == ik].case.unique())]
        ind = [it for it in st["items"] if it["group"] == "in-domain"]
        if ind:
            trdir = (REPO / ind[0]["input_dir"]).parent / "imagesTr"
            train_tasks[s] = sorted(str(p) for p in trdir.glob("*_0000.nii.gz"))
    G = pd.DataFrame(gate)
    G.to_csv(out / "gate.tsv", sep="\t", index=False)
    C = pd.DataFrame(counts)
    C.to_csv(out / "counts.tsv", sep="\t", index=False)
    # headers + GT volumes (items shared across settings computed once per (item, case))
    uniq = {(t["item"]["key"], t["case"]): t for t in tasks}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        CI = pd.DataFrame(list(ex.map(case_info, uniq.values(), chunksize=4)))
        tr_flat = sorted({p for v in train_tasks.values() for p in v})
        TI = pd.DataFrame(list(ex.map(train_info, tr_flat, chunksize=16))) if tr_flat else pd.DataFrame()
    if "error" not in CI:
        CI["error"] = None
    CI.to_csv(out / "case_headers.tsv.gz", sep="\t", index=False)   # untracked output dir; contains case ids
    I = pd.DataFrame(items)
    agg = CI.groupby("item_key").agg(t=("t_img", "median"), p=("p_img", "median"), t_gt=("t_gt", "median"),
                                     p_gt=("p_gt", "median"), target_ml=("target_ml", "median"),
                                     n_header_cases=("case", "nunique"),
                                     n_errors=("error", lambda x: int(x.notna().sum())))
    I = I.merge(agg.reset_index(), on="item_key", how="left")
    tr = []
    for s, ps in train_tasks.items():
        T = TI[TI.path.isin(ps)] if len(TI) else TI
        tr.append(dict(setting=s, n_train=len(T), t_train=float(T.t.median()) if len(T) else np.nan,
                       p_train=float(T.p.median()) if len(T) else np.nan))
    I = I.merge(pd.DataFrame(tr), on="setting", how="left")
    I.to_csv(out / "items.tsv", sep="\t", index=False)
    print(G.to_string())
    print(C.to_string())
    print("gate:", "ALL OK" if bool(G.ok.all()) else "FAILURES", f"{int(G.ok.sum())}/{len(G)}; counts ok {int(C.ok.sum())}/{len(C)}")
    if CI.error.notna().any():
        print("header errors:", CI[CI.error.notna()].groupby("item_key").size().to_dict())


if __name__ == "__main__":
    main()
