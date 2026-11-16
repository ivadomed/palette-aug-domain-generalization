#!/usr/bin/env python3
"""T11 step 1 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...):
SynthSeg-noEM's in-domain errors, per case, and CF14.

Scope (fixed before any value was read): per setting, the setting's OWN in-domain test item (the in-domain item of the frozen
v2 settings snapshot that T04/E1 use), the SynthSeg-noEM run of the paper's headline config (t04_arms.run_ids), folds 0-2.
Case roster = the cases of that item in the stored eval_all.csv of the run (t04_signed.stored_dice, the T04 lookup), fold by
fold; prediction dirs = t04_arms.pred_dir; GT / prediction loading = the skip_perturbation scoring recipes used by E1 and
T04 (load_gt, read_stored_pred: on-harmony predictions on the native grid); label Dice = gd2_lib.dice_from_counts on the
item's label triples (T04's Dice QC). Multi-cohort in-domain rows of the paper (Breast T1-CE, HCC, Pancreas T2w) are reduced
to the own item (declared deviation). Spine is not in the snapshot and has no data on Vulcan: listed, not computed.

Per case and fold (foreground = any label > 0, prediction on the GT grid):
  dice_fg                whole-foreground Dice; empty prediction on non-empty GT -> 0; empty prediction on empty GT -> 1
  empty                  0 predicted foreground voxels
  FP components          26-connected predicted components with no voxel on GT foreground (gd2_lib._components); per
                         component: voxel count, min distance (mm) to the GT (Euclidean distance transform of the GT
                         background, voxel spacing; inf when the GT is empty); components < 10 voxels flagged
  label Dice             per scored label (the paper estimand, for the gate and the QC against eval_all.csv)
Per case (once): CF14 = GT foreground volume / volume of image voxels above the image's 5th intensity percentile (volumes,
so the ratio holds when image and GT grids differ; image = the model input input_dir/<case>_0000.nii.gz).
Usage (Slurm): .venv/bin/python t11_cases.py --out DIR [--workers 8] [--settings S1,S2]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t04_arm_boundary"))
import t04_arms as TA  # noqa: E402  (puts v2 + E1 on sys.path)
import t04_signed as TS  # noqa: E402
import gd2_lib as gl  # noqa: E402

dv = TA.dv
ARM = "synthseg_noEM"
SMALL = 10
SNAP = ["Glioma_t1n", "Glioma_t1c", "Glioma_t2w", "Glioma_t2f", "MS_flair", "MS_t1w", "Breast_t1wce", "Breast_t2w", "HCC_t1w",
        "Abdomen_t1in", "Abdomen_t2spir", "Brain_T1w", "Brain_T2w", "Brain_dwi_ap", "Mandible_cbct", "Pelvis_ct", "Pelvis_mri",
        "Pancreas_t1wce", "Pancreas_t2w"]


def label_ids(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def process(task):
    it, case, folds = task["item"], task["case"], task["folds"]
    out = dict(case_rows=[], label_rows=[], comp_rows=[], errors=[])
    try:
        g_img, g = dv.scoring.load_gt(dv.R, it, case)
        sp = np.array(g_img.GetSpacing()[::-1], float)
        vml = float(np.prod(sp)) / 1000.0
        gfg = g > 0
        n_g = int(gfg.sum())
        # CF14
        cf = dict(setting=task["setting"], case=case, gt_ml=n_g * vml)
        try:
            im = sitk.ReadImage(str(REPO / it["input_dir"] / f"{case}_0000.nii.gz"))
            a = sitk.GetArrayFromImage(im).astype(np.float32)
            p5 = float(np.percentile(a, 5))
            body_ml = float((a > p5).sum()) * float(np.prod(im.GetSpacing())) / 1000.0
            cf.update(body_ml=body_ml, cf14=(n_g * vml) / body_ml if body_ml > 0 else np.nan)
        except Exception as e:  # noqa: BLE001
            cf.update(body_ml=np.nan, cf14=np.nan, cf14_error=repr(e))
        out["cf14"] = cf
        dist = ndimage.distance_transform_edt(~gfg, sampling=sp) if n_g else None
        for f, d in folds.items():
            p = dv.scoring.read_stored_pred(Path(d) / f"{case}.nii.gz", it, dv.R, case)
            if p.shape != g.shape:
                raise RuntimeError(f"shape mismatch fold {f}: {p.shape} vs {g.shape}")
            for name, pid, gid in it["triples"]:
                pb, gb = dv.binarize(p, label_ids(pid)), dv.binarize(g, label_ids(gid))
                out["label_rows"].append(dict(setting=task["setting"], case=case, fold=int(f), label=name,
                                              dice=float(gl.dice_from_counts(int(pb.sum()), int((pb & gb).sum()), int(gb.sum())))))
            pfg = p > 0
            n_p, n_tp = int(pfg.sum()), int((pfg & gfg).sum())
            dfg = 1.0 if (n_p + n_g) == 0 else 2.0 * n_tp / (n_p + n_g)
            row = dict(setting=task["setting"], case=case, fold=int(f), n_gt=n_g, n_pred=n_p, n_tp=n_tp, dice_fg=dfg,
                       empty=bool(n_p == 0), gt_empty=bool(n_g == 0), pred_ml=n_p * vml, gt_ml=n_g * vml)
            nfp = nfp_small = 0
            fp_vox = fp_vox_far = fp_vox_small = 0
            if n_p:
                lab, n = gl._components(pfg)
                sizes = np.bincount(lab.ravel(), minlength=n + 1)
                hit = np.zeros(n + 1, bool)
                hit[np.unique(lab[gfg & pfg])] = True
                hit[0] = True
                fp_ids = np.nonzero(~hit)[0]
                if fp_ids.size:
                    if dist is None:
                        mind = {int(i): np.inf for i in fp_ids}
                    else:
                        mins = ndimage.minimum(dist, labels=lab, index=fp_ids)
                        mind = {int(i): float(m) for i, m in zip(fp_ids, np.atleast_1d(mins))}
                    for i in fp_ids:
                        sz = int(sizes[i])
                        small = sz < SMALL
                        out["comp_rows"].append(dict(setting=task["setting"], case=case, fold=int(f), n_vox=sz, ml=sz * vml,
                                                     min_dist_mm=mind[int(i)], small=small))
                        nfp += 1
                        nfp_small += int(small)
                        fp_vox += sz
                        if small:
                            fp_vox_small += sz
                        elif mind[int(i)] > 20.0:
                            fp_vox_far += sz
                row.update(n_comp=int(n), n_comp_tp=int(hit[1:].sum()))
            row.update(n_fp_comp=nfp, n_fp_comp_small=nfp_small, fp_vox=fp_vox, fp_vox_small=fp_vox_small, fp_vox_far_big=fp_vox_far)
            out["case_rows"].append(row)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{task['setting']}\t{case}\t{e!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--settings", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = TA.settings()
    want = [s for s in (a.settings.split(",") if a.settings else SNAP) if s]
    tasks, roster, stored_rows = [], [], []
    for s in want:
        st = S[s]
        it = next(x for x in st["items"] if x["group"] == "in-domain")
        rid = TA.run_ids(st).get(ARM)
        cases_by_fold, dirs = {}, {}
        for f in dv.FOLDS:
            d, src = TS.stored_dice(st, it, rid, f)
            stored_rows += [dict(setting=s, case=c, fold=f, label=l, stored_dice=v) for (c, l), v in d.items()]
            cs = sorted({c for c, _ in d})
            p, how = TA.pred_dir(st, it, ARM, rid, f)
            have = {x.name[:-7] for x in p.glob("*.nii.gz")} if p else set()
            roster.append(dict(setting=s, item_key=it["key"], run_id=rid, fold=f, stored_csv=src, n_cases_eval=len(cs),
                               pred_dir=str(p) if p else "ABSENT", how=how, n_pred_files=len(have),
                               n_eval_cases_with_pred=len(set(cs) & have), n_labels=len(it["triples"])))
            cases_by_fold[f] = cs
            if p is not None:
                dirs[f] = (str(p), have)
        allc = sorted(set().union(*cases_by_fold.values()))
        for c in allc:
            folds = {f: dirs[f][0] for f in dirs if c in dirs[f][1] and c in cases_by_fold[f]}
            tasks.append(dict(setting=s, item=it, case=c, folds=folds))
        print(f"[t11] {s}: {len(allc)} eval cases; prediction folds {sorted(dirs)}", flush=True)
    R = pd.DataFrame(roster)
    R.to_csv(out / "roster.tsv", sep="\t", index=False)
    pd.DataFrame(stored_rows).to_parquet(out / "stored_label_dice.parquet")
    case_rows, label_rows, comp_rows, cf_rows, errs = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(process, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            case_rows += r["case_rows"]; label_rows += r["label_rows"]; comp_rows += r["comp_rows"]; errs += r["errors"]
            if "cf14" in r:
                cf_rows.append(r["cf14"])
            if i % 100 == 0:
                print(f"[t11] {i}/{len(tasks)}", flush=True)
    pd.DataFrame(case_rows).to_parquet(out / "case_rows.parquet")
    pd.DataFrame(label_rows).to_parquet(out / "label_rows.parquet")
    pd.DataFrame(comp_rows, columns=["setting", "case", "fold", "n_vox", "ml", "min_dist_mm", "small"]).to_parquet(out / "fp_components.parquet")
    pd.DataFrame(cf_rows).to_parquet(out / "cf14_cases.parquet")
    print(f"[t11] case rows {len(case_rows)}, label rows {len(label_rows)}, FP components {len(comp_rows)}, cf14 cases {len(cf_rows)},"
          f" errors {len(errs)}", flush=True)
    for e in errs:
        print("[error]", e, flush=True)
    sys.exit(3 if errs else 0)


if __name__ == "__main__":
    main()
