#!/usr/bin/env python3
"""T10 step 1 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T10_T28.md, section T10, sha256 e761e43e...,
tag reg-T10-T28-20261116): voxel-level decomposition of the best randomised arm's error and of the specialist's error, per case,
label and fold, on the P15 cells; edge strength at the GT boundary (T28's operator) and per-object target-to-ring CNR (T02's CF03).

Scope (registration, read at run time from T02 cells.tsv): OOD cells of the P15 settings (P15_set True: Glioma T1w, T1-CE, T2w,
FLAIR, Pelvis CT) with spec_same_items True. Arm = the cell's best_randomised_arm; specialist = spec_setting / spec_arm on the
specialist setting's own in-domain item (same dataset item as the cell). Folds 0-2. Cases = those in the stored eval_all.csv of
every fold of both models AND with a prediction file in all six fold dirs ("test cases common to both").

Imports, not copies:
  predictions      t04_arms.pred_dir (headline arms; repo tree, then the T04 stage), decompose_v2.pred_dir (texture = Palette);
                   specialist baselines also from the in-domain stage of plans/2026-11-16-repredict-indomain.md
                   (/scratch/paulh/dg_mechanisms/indomain/staged_predictions) via t04_arms.pred_dir with STAGE pointed there.
  stored Dice      t04_signed.stored_dice (headline arms) / decompose_v2.stored_rows (texture).
  GT / prediction  skip_perturbation scoring.load_gt / read_stored_pred, decompose_v2.binarize, the item's label triples.
  components       gd2_lib._components (26-connectivity, cc3d), Dice gd2_lib.dice_from_counts.
  2 mm band        gd2_lib._edt_inside (spacing-aware EDT; inside: distance of a GT voxel to the nearest background voxel;
                   outside: _edt_inside(~GT), distance of a background voxel to the nearest GT voxel); within 2 mm = d <= 2 + gd2_lib.EPS.
  edge operator    t28_edge.profiles on the z-scored test image (t28_edge.zscore_body; image resampled onto the GT grid when the
                   grids differ, t28_edge.same_grid; image path t28_edge.image_path of the cell's own test item).
  CNR (CF03)       t02_features.case_features on a component-id label volume written to scratch (one id per GT component of one
                   label; ring = (0, 3 mm] around that component, MRI: nonzero voxels only, ct flag = t02_features.is_ct).

Per case, label, model and fold (foreground of the label; prediction on the GT grid). Choices the registration leaves open are [impl]:
  missed         FN voxels in GT components (26-conn) with no overlap with the prediction (= all voxels of such components).
  band_fn        FN voxels in detected GT components with d_in <= 2 mm.
  band_fp        FP voxels in predicted components that overlap GT, with d_out <= 2 mm.
  far_fp         FP voxels in predicted components with no GT overlap (whatever their distance: component status first).
  interior_fn / interior_fp   the rest. Exactness check: missed + band + far + interior == FN + FP (written per row).
  [impl] Edge points = T28's sampled GT surface points of this label's mask (up to 2,000, same seed); the coordinates are
         recomputed with t28_edge's own selection lines (surface = mask minus 6-erosion inside T28's crop, rng choice) and checked
         against profiles()' n_surface / n_sampled (profiles() does not return them). Valid = in-array profile, unit normal,
         SDM zero crossing (T28's conditions). Tertiles of the max |d intensity / ds| over the VALID points of the (case, label):
         weak = <= 1/3 quantile, strong = >= 2/3 quantile. Locatable (reported, never gating) = valid and >= the case's
         10th percentile (T28's drop rule).
  [impl] Each band error voxel is assigned to the nearest valid edge point (physical mm, KD-tree); per-point excess uses the
         count of weak / strong points of the (case, label), counted once (not per fold).
  [impl] Objects for ME04 / CNR = GT components of the label; a (component, fold) is missed when that fold's prediction does not
         overlap it.
Outputs (untracked, case ids): OUT/vox.parquet, OUT/points.parquet, OUT/comps.parquet, OUT/checks.parquet, OUT/roster.tsv.
Usage (Slurm only): .venv/bin/python t10_cases.py --out DIR --scratch DIR [--workers 8] [--max-cases N] [--items a,b]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
DGM = REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms"
for sub in ("t04_arm_boundary", "t28_edge_position", "t02_cell_features"):
    sys.path.insert(0, str(DGM / sub))
import t04_arms as TA  # noqa: E402  (puts v2 + E1 on sys.path)
import t04_signed as TS  # noqa: E402
import gd2_lib as gl  # noqa: E402
import t28_edge as T28  # noqa: E402
import t02_features as T02F  # noqa: E402

dv = TA.dv
CELLS_TSV = REPO / "benchmark/01_commun_results/dg_mechanisms/T02/cells.tsv"
STAGE_T04 = Path("/scratch/paulh/dg_mechanisms/T04/staged_predictions")
STAGE_IND = Path("/scratch/paulh/dg_mechanisms/indomain/staged_predictions")
BAND_MM = 2.0
# T02 display names -> settings-snapshot keys (checked against the snapshot's task / training contrast below)
SETTING_KEY = {"Glioma T1w": "Glioma_t1n", "Glioma T1-CE": "Glioma_t1c", "Glioma T2w": "Glioma_t2w", "Glioma FLAIR": "Glioma_t2f",
               "Pelvis CT": "Pelvis_ct", "Pelvis MRI": "Pelvis_mri"}
ARM_KEY = {"palette": "texture", "pa_auglab": "pa_auglab", "auglab": "auglab_default", "ssem": "synthseg_EM", "plain": "baseline"}


def label_ids(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def p15_cells() -> pd.DataFrame:
    C = pd.read_csv(CELLS_TSV, sep="\t", dtype=str)
    C = C[(C.P15_set == "True") & (C.spec_same_items == "True")].copy()
    C["s_key"] = C.setting.map(SETTING_KEY)
    C["spec_key"] = C.spec_setting.map(SETTING_KEY)
    C["arm"] = C.best_randomised_arm.map(ARM_KEY)
    C["spec_arm_key"] = C.spec_arm.map(ARM_KEY)
    if C[["s_key", "spec_key", "arm", "spec_arm_key"]].isna().any().any():
        raise RuntimeError("unmapped cell fields:\n" + C.to_string())
    C["dataset"] = C["items"].str.split(":").str[0]
    C["item"] = C["items"].str.split(":").str[1]
    return C


def locate(st, it, arm, fold):
    """(run id, dir or None, how): repo tree first, then the T04 stage, then the in-domain stage."""
    if arm in TA.FILL:
        p, how = dv.pred_dir(st, it, arm, fold)
        return dv.run_id(st, arm), p, how
    rid = TA.run_ids(st)[arm]
    for stage, tag in ((STAGE_T04, "staged_T04"), (STAGE_IND, "staged_indomain")):
        TA.STAGE = stage
        p, how = TA.pred_dir(st, it, arm, rid, fold)
        if p is not None:
            TA.STAGE = STAGE_T04
            return rid, p, (tag if how == "staged" else how)
    TA.STAGE = STAGE_T04
    return rid, None, "absent"


def stored(st, it, arm, rid, fold):
    if arm in TA.FILL:
        return dv.stored_rows(st, it, arm, fold)
    return TS.stored_dice(st, it, rid, fold)


def edge_points(m: np.ndarray, z: np.ndarray, sp: np.ndarray) -> dict:
    """T28's profiles() on mask m plus the point coordinates (full-array index), recomputed with T28's own selection lines."""
    P = T28.profiles(m, z, sp)
    if "error" in P:
        return P
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - T28.MARGIN, 0)
    hi = np.minimum(idx.max(0) + T28.MARGIN + 1, np.array(m.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    f = m[sl]
    surf = f & ~ndi.binary_erosion(f, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    pts = np.argwhere(surf)
    if len(pts) != P["n_surface"]:
        raise RuntimeError("edge point recomputation disagrees with t28_edge.profiles (n_surface)")
    if len(pts) > T28.NPTS:
        pts = pts[np.sort(np.random.default_rng(T28.SEED).choice(len(pts), T28.NPTS, replace=False))]
    if len(pts) != P["n_sampled"]:
        raise RuntimeError("edge point recomputation disagrees with t28_edge.profiles (n_sampled)")
    P["pts"] = pts + lo[None, :]
    return P


def process(task: dict) -> dict:
    out = dict(vox=[], points=[], comps=[], checks=[], errors=[])
    case, models = task["case"], task["models"]
    ref_it = task["ref_item"]
    try:
        g_img, g = dv.scoring.load_gt(dv.R, ref_it, case)
        sp = np.array(g_img.GetSpacing()[::-1], float)
        vml = float(np.prod(sp)) / 1000.0
        chk = dict(item=task["item_tag"], case=case)
        # GT of every model's own item must equal the reference GT (the stored Dice of each model used its own item's GT)
        chk["gt_equal_all"] = bool(all(np.array_equal(dv.scoring.load_gt(dv.R, m["item"], case)[1], g) for m in models))
        # test image: the cells' own test items (all must be the same image); first one is used
        imgs = sorted({str(T28.image_path(m["item"], case)) for m in models if m["role"] == "arm"})
        arrs = [sitk.GetArrayFromImage(sitk.ReadImage(p, sitk.sitkFloat32)) for p in imgs]
        chk["n_test_image_paths"] = len(imgs)
        chk["test_images_equal"] = bool(all(a.shape == arrs[0].shape and np.array_equal(a, arrs[0]) for a in arrs))
        im = sitk.ReadImage(imgs[0], sitk.sitkFloat32)
        chk["resampled"] = not T28.same_grid(im, g_img)
        if chk["resampled"]:
            im = sitk.Resample(im, g_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        z, _ = T28.zscore_body(sitk.GetArrayFromImage(im))
        # CNR: component-id volume in the GT file's own (nibabel) orientation
        gt_path = REPO / ref_it["gt_dir"] / f"{case}.nii.gz"
        lb = nib.load(str(gt_path))
        g_nib = np.rint(np.asanyarray(lb.dataobj)).astype(np.int32)
        chk["nib_transpose_ok"] = bool(g_nib.shape == g.shape[::-1] and np.array_equal(np.transpose(g_nib, (2, 1, 0)), g))
        comp_vol = np.zeros(g.shape, np.int32)
        info = {}
        nxt = 1
        for name, pid, gid in ref_it["triples"]:
            gb = dv.binarize(g, label_ids(gid))
            lab_g, n_g = gl._components(gb)
            size = np.bincount(lab_g.ravel(), minlength=n_g + 1)
            if n_g:
                comp_vol[gb] = lab_g[gb] + (nxt - 1)
            ids = list(range(nxt, nxt + n_g))
            nxt += n_g
            d_in = gl._edt_inside(gb, sp) if n_g else None
            d_out = gl._edt_inside(~gb, sp) if n_g else None
            band_in = (gb & (d_in <= BAND_MM + gl.EPS)) if n_g else np.zeros_like(gb)
            band_out = (~gb & (d_out <= BAND_MM + gl.EPS)) if n_g else np.zeros_like(gb)
            prow = dict(item=task["item_tag"], case=case, label=name, n_gt=int(gb.sum()), n_comp=n_g)
            tree, cls = None, None
            if n_g:
                P = edge_points(gb, z, sp)
                if "error" in P:
                    prow.update(edge_error=P["error"])
                else:
                    valid = P["inside"] & np.isfinite(P["s0"]) & np.isfinite(P["L"])
                    nv = int(valid.sum())
                    prow.update(n_surface=int(P["n_surface"]), n_sampled=int(P["n_sampled"]), n_valid=nv)
                    if nv >= 3:
                        gv = P["gmax"][valid]
                        q1, q2 = np.quantile(gv, [1 / 3, 2 / 3])
                        p10 = np.percentile(gv, T28.DROP_Q)
                        cls = np.where(gv <= q1, 0, np.where(gv >= q2, 2, 1))     # 0 weak, 1 middle, 2 strong
                        tree = cKDTree(P["pts"][valid] * sp[None, :])
                        prow.update(n_weak=int((cls == 0).sum()), n_strong=int((cls == 2).sum()), n_locatable=int((gv >= p10).sum()),
                                    gmax_q1=float(q1), gmax_q2=float(q2),
                                    n_edge_at_window_end=int((np.abs(P["s_edge"][valid]) >= 3.5).sum()))
            out["points"].append(prow)
            info[name] = dict(gb=gb, lab_g=lab_g, n_g=n_g, size=size, ids=ids, band_in=band_in, band_out=band_out, tree=tree, cls=cls)
            for k in range(1, n_g + 1):
                out["comps"].append(dict(item=task["item_tag"], case=case, label=name, comp=k, size_vox=int(size[k]),
                                         size_ml=float(size[k]) * vml, model="", fold=-1, missed=None))
        # CF03 per component (t02_features.case_features, unchanged) on a scratch label volume
        cnr = {}
        if nxt > 1 and chk["nib_transpose_ok"]:
            cpath = Path(task["scratch"]) / "comp_labels" / task["item_tag"].replace(":", "__") / f"{case}.nii.gz"
            cpath.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.transpose(comp_vol, (2, 1, 0)).astype(np.int32), lb.affine, lb.header), str(cpath))
            classes = [(str(i), i) for i in range(1, nxt)]
            try:
                rows = T02F.case_features(imgs[0], str(cpath), classes, T02F.is_ct(task["dataset"], task["item"]))
                cnr = {int(r["label"]): r["cnr"] for r in rows}
            except Exception as e:  # noqa: BLE001
                chk["cnr_error"] = repr(e)
        for r in out["comps"]:
            nm = r["label"]
            r["cnr"] = float(cnr.get(info[nm]["ids"][r["comp"] - 1], np.nan)) if cnr else np.nan
        # decomposition per model / fold / label
        for m in models:
            for f, d in m["dirs"].items():
                p = dv.scoring.read_stored_pred(Path(d) / f"{case}.nii.gz", m["item"], dv.R, case)
                if p.shape != g.shape:
                    raise RuntimeError(f"{m['model']} fold {f}: shape {p.shape} vs GT {g.shape}")
                for name, pid, gid in m["item"]["triples"]:
                    I = info[name]
                    gb = I["gb"]
                    pb = dv.binarize(p, label_ids(pid))
                    fn, fp = gb & ~pb, pb & ~gb
                    n_fn, n_fp = int(fn.sum()), int(fp.sum())
                    # GT components detected by this prediction
                    det = np.zeros(I["n_g"] + 1, bool)
                    if I["n_g"]:
                        det[np.unique(I["lab_g"][gb & pb])] = True
                    det[0] = False
                    in_det = det[I["lab_g"]]
                    missed = int((fn & ~in_det & gb).sum())
                    band_fn = int((fn & in_det & I["band_in"]).sum())
                    int_fn = n_fn - missed - band_fn
                    # predicted components overlapping GT
                    if pb.any():
                        lab_p, n_p = gl._components(pb)
                        hit = np.zeros(n_p + 1, bool)
                        hit[np.unique(lab_p[gb & pb])] = True
                        hit[0] = False
                        in_hit = hit[lab_p]
                    else:
                        in_hit = np.zeros_like(pb)
                    far_fp = int((fp & ~in_hit).sum())
                    band_fp_m = fp & in_hit & I["band_out"]
                    band_fp = int(band_fp_m.sum())
                    int_fp = n_fp - far_fp - band_fp
                    n_pv, n_tp, n_gv = int(pb.sum()), int((pb & gb).sum()), int(gb.sum())
                    row = dict(item=task["item_tag"], case=case, model=m["model"], fold=int(f), label=name, n_gt=n_gv, n_pred=n_pv,
                               n_tp=n_tp, fn=n_fn, fp=n_fp, missed=missed, band_fn=band_fn, band_fp=band_fp, far_fp=far_fp,
                               interior_fn=int_fn, interior_fp=int_fp,
                               sum_ok=bool(missed + band_fn + band_fp + far_fp + int_fn + int_fp == n_fn + n_fp and int_fn >= 0 and int_fp >= 0),
                               dice=float(gl.dice_from_counts(n_pv, n_tp, n_gv)),
                               stored_dice=float(m["stored"][int(f)].get((case, name), np.nan)),
                               stored_present=bool((case, name) in m["stored"][int(f)]))
                    # band errors -> nearest valid edge point
                    if I["tree"] is not None and (band_fn + band_fp):
                        bm = (fn & in_det & I["band_in"]) | band_fp_m
                        q = np.argwhere(bm) * sp[None, :]
                        _, j = I["tree"].query(q)
                        c = I["cls"][j]
                        row.update(band_weak=int((c == 0).sum()), band_mid=int((c == 1).sum()), band_strong=int((c == 2).sum()))
                    else:
                        row.update(band_weak=0, band_mid=0, band_strong=0, band_unassigned=int(band_fn + band_fp) if I["tree"] is None else 0)
                    out["vox"].append(row)
                    for k in range(1, I["n_g"] + 1):
                        out["comps"].append(dict(item=task["item_tag"], case=case, label=name, comp=k, size_vox=int(I["size"][k]),
                                                 size_ml=float(I["size"][k]) * vml, model=m["model"], fold=int(f), missed=bool(not det[k]),
                                                 cnr=np.nan))
        out["checks"].append(chk)
    except Exception as e:  # noqa: BLE001
        import traceback
        out["errors"].append(f"{task['item_tag']}\t{case}\t{e!r}\t{traceback.format_exc()[-600:]}")
    return out


def build(max_cases=0, items=None):
    S = TA.settings()
    C = p15_cells()
    roster, models_by_item = [], {}
    for r in C.itertuples():
        st, sst = S[r.s_key], S[r.spec_key]
        for k, s in ((r.setting, st), (r.spec_setting, sst)):
            want = {"Glioma": "brats2024-glioma", "Pelvis": "totalseg-pelvic"}[s["task"]]
            if s["dataset"] != want:
                raise RuntimeError(f"{k}: snapshot dataset {s['dataset']}")
        it_a = next(x for x in st["items"] if x["dataset"] == r.dataset and x["item"] == r.item and x["group"] != "in-domain")
        it_s = next(x for x in sst["items"] if x["group"] == "in-domain")
        if it_s["dataset"] != r.dataset or it_s["item"] != r.item or it_s["triples"] != it_a["triples"]:
            raise RuntimeError(f"{r.setting}->{r.item}: specialist item {it_s['dataset']}:{it_s['item']} differs from the cell item")
        tag = f"{r.dataset}:{r.item}"
        for role, s_key, s, it, arm in (("arm", r.s_key, st, it_a, r.arm), ("spec", r.spec_key, sst, it_s, r.spec_arm_key)):
            mid = f"{role}:{s_key}:{arm}"
            dirs, hows, stored_d, cases = {}, {}, {}, []
            for f in dv.FOLDS:
                rid, p, how = locate(s, it, arm, f)
                d, src = stored(s, it, arm, rid, f)
                stored_d[f] = d
                have = {x.name[:-7] for x in p.glob("*.nii.gz")} if p else set()
                cases.append({c for c, _ in d} & have)
                dirs[f], hows[f] = (str(p) if p else None), how
                roster.append(dict(cell=f"{r.setting}->{r.item}", role=role, model=mid, setting=s_key, arm=arm, run_id=rid, item=tag,
                                   fold=f, pred_dir=str(p) if p else "ABSENT", how=how, n_pred_files=len(have),
                                   stored_csv=src, n_stored_cases=len({c for c, _ in d})))
            M = models_by_item.setdefault(tag, {})
            if mid in M and M[mid]["dirs"] != dirs:
                raise RuntimeError(f"{mid}: two different prediction dirs")
            M[mid] = dict(model=mid, role=role, item=it, dirs=dirs, hows=hows, stored=stored_d, cases=set.intersection(*cases),
                          cell=f"{r.setting}->{r.item}")
    tasks = []
    for tag, M in models_by_item.items():
        if items and tag not in items:
            continue
        allc = sorted(set().union(*[m["cases"] for m in M.values()]))
        if max_cases:
            allc = allc[:max_cases]
        ref = next(m["item"] for m in M.values() if m["role"] == "spec")
        for c in allc:
            ms = [dict(model=m["model"], role=m["role"], item=m["item"], dirs={f: d for f, d in m["dirs"].items()},
                       stored=m["stored"]) for m in M.values() if c in m["cases"]]
            tasks.append(dict(item_tag=tag, dataset=tag.split(":")[0], item=tag.split(":")[1], case=c, ref_item=ref, models=ms))
    case_sets = {m["model"] + "|" + tag: sorted(m["cases"]) for tag, M in models_by_item.items() for m in M.values()}
    return C, pd.DataFrame(roster), tasks, case_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--items", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    C, R, tasks, case_sets = build(a.max_cases, [x for x in a.items.split(",") if x])
    C.to_csv(out / "cells_scope.tsv", sep="\t", index=False)
    R.to_csv(out / "roster.tsv", sep="\t", index=False)
    json.dump(case_sets, open(Path(a.scratch) / "case_sets.json", "w"))
    for t in tasks:
        t["scratch"] = a.scratch
    print(f"[t10] {len(C)} cells, {R.model.nunique()} models, {len(tasks)} (item, case) tasks; absent dirs: "
          f"{int((R.pred_dir == 'ABSENT').sum())}", flush=True)
    res = dict(vox=[], points=[], comps=[], checks=[], errors=[])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            for k in res:
                res[k] += r[k]
            if i % 25 == 0:
                print(f"[t10] {i}/{len(tasks)}", flush=True)
    for k in ("vox", "points", "comps", "checks"):
        pd.DataFrame(res[k]).to_parquet(out / f"{k}.parquet")
    print(f"[t10] vox rows {len(res['vox'])}, point rows {len(res['points'])}, comp rows {len(res['comps'])}, errors {len(res['errors'])}",
          flush=True)
    for e in res["errors"]:
        print("[error]", e, flush=True)
    sys.exit(3 if res["errors"] else 0)


if __name__ == "__main__":
    main()
