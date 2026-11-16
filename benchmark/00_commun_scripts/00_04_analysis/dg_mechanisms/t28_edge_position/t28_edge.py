#!/usr/bin/env python3
"""T28 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T10_T28.md, section T28, sha256 e761e43e...,
tag reg-T10-T28-20261116): edge position E (CF19) and ramp width W per test image, for exactly T27's items.

Scope: T27's (setting, item) rows (T27/items_scored.tsv). Cases per (setting, item) = the cases in that setting's T04 row
parquet (read exactly as t27_offsets does: src_dir + t04_analyze.prep, set per_arm). The item dict is the setting's own
(settings snapshot): one item key can point to different files in two settings (e.g. AMOS FOV crops per training contrast),
so E is computed per distinct (image file, GT file) and joined back per (setting, item).
Loaders as T27: image = native image where the item has pred_transform.ref_dir (Brain), else input_dir/<case>_0000.nii.gz;
GT = dv.scoring.load_gt; scored foreground = dv.binarize(gt, union of the item's scored label ids).

Operator (choices the registration leaves open are marked [impl]). Fixed before the gate; one change after a 1-case smoke run
(job 2385597): the first version sampled along the physical normal with a per-point voxel length 1/||n/spacing||; it was
moved to index space (the registration's units) before the gate was run (REPORT.md, DEVIATIONS). Nothing else changed.
  - Image z-scored with mean / sd inside its body mask (Otsu threshold, largest 26-connected component); applied to the
    whole image (affine: cannot move any edge position, drop or 10/90 point).
  - [impl] Image and GT read with SimpleITK; if size / spacing / origin / direction differ (tol 1e-3), the image is resampled
    linearly onto the GT grid (counted per case).
  - GT surface voxels = foreground voxels with a 6-neighbour in background; up to 2,000 per case drawn with a fixed seed
    (numpy default_rng(20261116), same seed for every case and variant).
  - Outward normal = normalised gradient of the GT signed distance map smoothed with a Gaussian of sigma 1 voxel.
    Index space throughout (the registration's units): SDM in voxel units (EDT without spacing; + outside), gradient in index
    units, unit normal d in index space.
  - Profile: image sampled trilinearly along d at s = -4..+4 voxels in 0.5 steps (17 samples). Points whose profile leaves
    the array are dropped (counted). mm per voxel step along d: L = sqrt(sum((d_i * spacing_i)^2)).
  - Edge = sample of maximal |d intensity / ds| (central differences, interior samples s = -3.5..+3.5).
  - [impl] Origin = the drawn boundary: the zero crossing of the (unsmoothed) SDM sampled along the same profile (the voxel
    face between the surface voxel and its outside neighbour), not the surface-voxel centre (which sits half a voxel inside
    the drawn boundary). Points without a zero crossing in the window are dropped (counted). The centre-origin variant
    (s0 = 0) is kept as a sensitivity column.
  - E (mm) = -(s_edge - s0) * L; + = the edge lies inside the drawn boundary. E_vox = -(s_edge - s0) (for the gate).
  - Drop: points whose max |gradient| is below the 10th percentile of the case's values (share reported).
  - W [impl]: from the edge sample, walk both ways while the profile keeps the sign of the step; the two ends are the 0% and
    100% levels; W = distance between the interpolated 10% and 90% crossings on that monotone run, times L, in mm.
  - Per case: medians over the kept points; per item: medians over cases.
Gate (registered): on 3 items per task (first 3 item keys in sorted order; MS has 2 and Pelvis 1 OOD item in scope), GT
dilated by one voxel (6-connected structuring element, 1 iteration): item E_vox must rise by 0.7-1.3; eroded: fall by 0.7-1.3.
Smoke: --mode smoke --limit N runs N files per gate item, original + dilated (code-path check only, no gate decision).
Usage (Slurm only): .venv/bin/python t28_edge.py --mode gate|full --out DIR [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t27_offset_resolution"))
import t27_offsets as T27  # noqa: E402  (src_dir, REGISTERED, DEV12; puts t04_analyze / t04_arms / dv on sys.path)

T4A, TA, dv = T27.T4A, T27.TA, T27.dv
T27_OUT = REPO / "benchmark/01_commun_results/dg_mechanisms/T27"
SEED = 20261116
NPTS = 2000
S_GRID = np.arange(-4.0, 4.0001, 0.5)          # 17 samples, in voxels along the normal
MARGIN = 8                                      # crop margin (voxels) around the GT bounding box
GATE_LO, GATE_HI = 0.7, 1.3
DROP_Q = 10


def image_path(it: dict, case: str) -> Path:
    """T27's header rule (t27_offsets.case_info): native image for Brain (pred_transform.ref_dir), else the model input."""
    tr = it.get("pred_transform") or {}
    if tr.get("ref_dir"):
        return REPO / tr["ref_dir"] / f"{case}{tr['ref_suffix']}"
    return REPO / it["input_dir"] / f"{case}_0000.nii.gz"


def label_ids(it: dict):
    return sorted({int(x) for t in it["triples"] for x in (t[2] if isinstance(t[2], list) else [t[2]])})


def same_grid(a: sitk.Image, b: sitk.Image, tol=1e-3) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def zscore_body(arr: np.ndarray) -> tuple[np.ndarray, float]:
    a = arr.astype(np.float32)
    thr = threshold_otsu(a[np.isfinite(a)])
    m = a > thr
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3), bool))
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(sizes.argmax())
    vals = a[m]
    sd = float(vals.std()) or 1.0
    return (a - float(vals.mean())) / sd, float(m.mean())


def profiles(fg: np.ndarray, img: np.ndarray, sp: np.ndarray, rng_seed=SEED) -> dict:
    """fg (bool, z,y,x) and img (float, same grid); sp = spacing per array axis (mm). Returns per-point arrays."""
    if not fg.any():
        return dict(error="empty foreground")
    idx = np.argwhere(fg)
    lo = np.maximum(idx.min(0) - MARGIN, 0)
    hi = np.minimum(idx.max(0) + MARGIN + 1, np.array(fg.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    f = fg[sl]
    im = img[sl]
    # signed distance (voxels): + outside (distance to nearest fg voxel), - inside (distance to nearest bg voxel)
    sdm = ndi.distance_transform_edt(~f) - ndi.distance_transform_edt(f)
    sm = ndi.gaussian_filter(sdm, sigma=1.0)
    grads = np.gradient(sm)                                           # index-space gradient
    surf = f & ~ndi.binary_erosion(f, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    pts = np.argwhere(surf)
    n_surface = len(pts)
    if n_surface > NPTS:
        pts = pts[np.sort(np.random.default_rng(rng_seed).choice(n_surface, NPTS, replace=False))]
    g = np.stack([gr[tuple(pts.T)] for gr in grads], 1)               # (N,3) index-space gradient
    gn = np.linalg.norm(g, axis=1)
    ok_norm = gn > 1e-8
    d_idx = np.zeros_like(g)
    d_idx[ok_norm] = g[ok_norm] / gn[ok_norm, None]                   # unit index-space normal
    L = np.where(ok_norm, np.linalg.norm(d_idx * sp, axis=1), np.nan)  # mm per index step along d
    coords = pts[:, None, :] + S_GRID[None, :, None] * d_idx[:, None, :]   # (N,17,3) index coords in the crop
    shp = np.array(f.shape)
    # out of the crop = out of the array only where the crop touches the array border (margin 8 > 4)
    abs_c = coords + lo[None, None, :]
    inside = np.all((abs_c >= 0) & (abs_c <= np.array(fg.shape) - 1), axis=(1, 2)) & ok_norm
    cc = np.clip(coords, 0, shp - 1).reshape(-1, 3).T
    prof = ndi.map_coordinates(im.astype(np.float64), cc, order=1, mode="nearest").reshape(len(pts), len(S_GRID))
    sprof = ndi.map_coordinates(sdm, cc, order=1, mode="nearest").reshape(len(pts), len(S_GRID))
    # edge: max |central difference| over interior samples
    der = (prof[:, 2:] - prof[:, :-2]) / (2 * 0.5)                   # per voxel, at samples 1..15
    k = np.abs(der).argmax(1) + 1
    gmax = np.abs(der).max(1)
    s_edge = S_GRID[k]
    # origin: zero crossing of the raw SDM closest to s = 0
    s0 = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        v = sprof[i]
        c = np.where((v[:-1] <= 0) & (v[1:] > 0))[0]
        if len(c):
            j = c[np.argmin(np.abs(S_GRID[c]))]
            s0[i] = S_GRID[j] + 0.5 * (-v[j]) / (v[j + 1] - v[j])
    # ramp width W (10-90 on the monotone run around the edge)
    W = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        p = prof[i]
        kk = k[i]
        sg = np.sign(p[kk + 1] - p[kk - 1])
        if sg == 0:
            continue
        a = kk
        while a > 0 and sg * (p[a] - p[a - 1]) > 0:
            a -= 1
        b = kk
        while b < len(p) - 1 and sg * (p[b + 1] - p[b]) > 0:
            b += 1
        seg = sg * p[a:b + 1]
        lo_v, hi_v = seg[0], seg[-1]
        if hi_v <= lo_v:
            continue
        s10 = np.interp(lo_v + 0.1 * (hi_v - lo_v), seg, S_GRID[a:b + 1])
        s90 = np.interp(lo_v + 0.9 * (hi_v - lo_v), seg, S_GRID[a:b + 1])
        W[i] = abs(s90 - s10)
    return dict(gmax=gmax, s_edge=s_edge, s0=s0, L=L, W_vox=W, inside=inside, n_surface=n_surface, n_sampled=len(pts))


def summarise(P: dict) -> dict:
    if "error" in P:
        return dict(error=P["error"])
    valid = P["inside"] & np.isfinite(P["s0"]) & np.isfinite(P["L"])
    n = len(valid)
    if valid.sum() < 10:
        return dict(error=f"only {int(valid.sum())} valid points")
    thr = np.percentile(P["gmax"][valid], DROP_Q)
    keep = valid & (P["gmax"] >= thr)
    e_vox = -(P["s_edge"] - P["s0"])
    e_mm = e_vox * P["L"]
    e_ctr_mm = -P["s_edge"] * P["L"]
    w_mm = P["W_vox"] * P["L"]
    return dict(n_surface=int(P["n_surface"]), n_sampled=n, share_out_of_array=float((~P["inside"]).mean()),
                share_no_zero_crossing=float((P["inside"] & ~np.isfinite(P["s0"])).mean()),
                share_dropped_weak=float((valid & ~keep).sum() / max(valid.sum(), 1)), gmax_p10=float(thr),
                n_kept=int(keep.sum()),
                E_mm=float(np.median(e_mm[keep])), E_vox=float(np.median(e_vox[keep])),
                E_center_mm=float(np.median(e_ctr_mm[keep])), s0_median=float(np.median(P["s0"][keep])),
                W_mm=float(np.nanmedian(w_mm[keep])) if np.isfinite(w_mm[keep]).any() else np.nan,
                W_vox=float(np.nanmedian(P["W_vox"][keep])) if np.isfinite(P["W_vox"][keep]).any() else np.nan,
                share_W_undefined=float((keep & ~np.isfinite(P["W_vox"])).sum() / max(keep.sum(), 1)),
                L_median=float(np.median(P["L"][keep])))


def run_case(task: dict) -> list[dict]:
    """task: img, gt_dir, case, ids, variants. One row per variant."""
    base = dict(img=task["img"], gt_dir=task["gt_dir"], case=task["case"])
    try:
        g_img, gt = dv.scoring.load_gt(dv.R, {"gt_dir": task["gt_dir"]}, task["case"])
        im = sitk.ReadImage(task["img"], sitk.sitkFloat32)
        resampled = False
        grid_note = ""
        if not same_grid(im, g_img):
            grid_note = (f"size {im.GetSize()}->{g_img.GetSize()} sp_maxdiff {np.abs(np.subtract(im.GetSpacing(), g_img.GetSpacing())).max():.3g} "
                         f"origin_maxdiff {np.abs(np.subtract(im.GetOrigin(), g_img.GetOrigin())).max():.3g} "
                         f"dir_maxdiff {np.abs(np.subtract(im.GetDirection(), g_img.GetDirection())).max():.3g}")
            im = sitk.Resample(im, g_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            resampled = True
        arr = sitk.GetArrayFromImage(im)
        z, body_frac = zscore_body(arr)
        sp = np.array(g_img.GetSpacing()[::-1], float)             # array axes (z, y, x)
        fg = dv.binarize(gt, task["ids"])
        out = []
        st6 = ndi.generate_binary_structure(3, 1)
        for var in task["variants"]:
            if var == "orig":
                m = fg
            elif var == "dilate":
                m = ndi.binary_dilation(fg, structure=st6, iterations=1)
            elif var == "erode":
                m = ndi.binary_erosion(fg, structure=st6, iterations=1, border_value=0)
            r = summarise(profiles(m, z, sp))
            out.append(dict(base, variant=var, resampled=resampled, grid_note=grid_note, body_frac=body_frac,
                            spacing=",".join(f"{x:.4g}" for x in sp), **r))
        return out
    except Exception as e:  # noqa: BLE001
        return [dict(base, variant=v, error=repr(e)) for v in task["variants"]]


def scope() -> pd.DataFrame:
    """(setting, task, item_key, case, img, gt_dir, ids) for T27's rows, cases from the T04 parquets as in t27_offsets."""
    I = pd.read_csv(T27_OUT / "items_scored.tsv", sep="\t")
    S = TA.settings()
    rows = []
    for s in T27.REGISTERED + T27.DEV12:
        d = T27.src_dir(s)
        R = T4A.prep(pd.read_parquet(d / "rows" / f"{s}.parquet"), s)
        sub = R[R.set == "per_arm"]
        itmeta = {it["key"]: it for it in S[s]["items"]}
        for ik in I[I.setting == s].item_key:
            it = itmeta[ik]
            for c in sorted(sub[sub.item_key == ik].case.unique()):
                rows.append(dict(setting=s, task=S[s]["task"], item_key=ik, case=c, img=str(image_path(it, c)),
                                 gt_dir=it["gt_dir"], ids=json.dumps(label_ids(it))))
    D = pd.DataFrame(rows)
    want = I.groupby(["setting", "item_key"]).n_cases.first()
    got = D.groupby(["setting", "item_key"]).case.nunique()
    bad = (want - got.reindex(want.index)).abs() > 0
    if bad.any():
        raise RuntimeError(f"case counts differ from T27 items_scored: {want[bad].to_dict()}")
    return D


def gate_items(D: pd.DataFrame) -> pd.DataFrame:
    """First 3 sorted item keys per task; each in its first setting (REGISTERED + DEV12 order)."""
    order = {s: i for i, s in enumerate(T27.REGISTERED + T27.DEV12)}
    D = D.assign(_o=D.setting.map(order))
    keep = []
    for t, g in D.groupby("task"):
        for ik in sorted(g.item_key.unique())[:3]:
            s = g[g.item_key == ik].sort_values("_o").setting.iloc[0]
            keep.append((s, ik))
    return D[[(s, i) in set(keep) for s, i in zip(D.setting, D.item_key)]].drop(columns="_o")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "full", "smoke"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = scope()
    if a.mode == "gate":
        D = gate_items(D)
        variants = ["orig", "dilate", "erode"]
    elif a.mode == "smoke":
        D = gate_items(D)
        D = D[D.case.isin(D.groupby("item_key").case.apply(lambda x: sorted(x)[: a.limit]).explode())]
        variants = ["orig", "dilate"]
    else:
        variants = ["orig"]
    uniq = D.drop_duplicates(["img", "gt_dir", "case"])
    tasks = [dict(img=r.img, gt_dir=r.gt_dir, case=r.case, ids=json.loads(r.ids), variants=variants) for r in uniq.itertuples()]
    print(f"[t28] mode {a.mode}: {len(D)} (setting,item,case) rows, {len(tasks)} distinct image/GT files", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_case, tasks, chunksize=1)):
            res += r
            if (i + 1) % 100 == 0:
                print(f"[t28] {i + 1}/{len(tasks)}", flush=True)
    C = pd.DataFrame(res)
    if "error" not in C:
        C["error"] = None
    M = D.merge(C, on=["img", "gt_dir", "case"], how="left")
    M.drop(columns=["ids"]).to_csv(out / f"cases_{a.mode}.tsv.gz", sep="\t", index=False)   # untracked; case ids
    n_err = int(M.error.notna().sum())
    print(f"[t28] case-variant rows {len(M)}, errors {n_err}", flush=True)
    if n_err:
        print(M[M.error.notna()].groupby(["item_key", "error"]).size().to_string())
    ok = M[M.error.isna()]
    agg = ok.groupby(["setting", "task", "item_key", "variant"]).agg(
        n_cases=("case", "nunique"), E_mm=("E_mm", "median"), E_vox=("E_vox", "median"), E_center_mm=("E_center_mm", "median"),
        W_mm=("W_mm", "median"), W_vox=("W_vox", "median"), L_median=("L_median", "median"),
        share_dropped_weak=("share_dropped_weak", "median"), share_out_of_array=("share_out_of_array", "mean"),
        share_no_zero_crossing=("share_no_zero_crossing", "mean"), share_W_undefined=("share_W_undefined", "mean"),
        n_resampled=("resampled", "sum"), n_points_kept_median=("n_kept", "median")).reset_index()
    nerr = M.groupby(["setting", "item_key", "variant"]).error.apply(lambda x: int(x.notna().sum())).rename("n_errors")
    agg = agg.merge(nerr.reset_index(), on=["setting", "item_key", "variant"], how="left")
    agg.to_csv(out / f"items_edge_{a.mode}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string())
    if a.mode == "gate":
        P = agg.pivot_table(index=["task", "setting", "item_key"], columns="variant", values="E_vox").reset_index()
        P["dE_dilate_vox"] = P["dilate"] - P["orig"]
        P["dE_erode_vox"] = P["erode"] - P["orig"]
        P["ok_dilate"] = P.dE_dilate_vox.between(GATE_LO, GATE_HI)
        P["ok_erode"] = (-P.dE_erode_vox).between(GATE_LO, GATE_HI)
        P["ok"] = P.ok_dilate & P.ok_erode
        P.to_csv(out / "gate.tsv", sep="\t", index=False)
        print(P.round(3).to_string())
        passed = bool(P.ok.all()) and len(P) > 0
        print("GATE:", "PASS" if passed else "FAIL", f"{int(P.ok.sum())}/{len(P)} items", flush=True)
        sys.exit(0 if passed else 3)


if __name__ == "__main__":
    main()
