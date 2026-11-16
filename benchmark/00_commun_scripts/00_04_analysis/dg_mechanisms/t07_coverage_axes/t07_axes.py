#!/usr/bin/env python3
"""T07 step 1 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T07.md, sha256 771c20db...):
appearance axes per OOD cell of T02, measured from images and labels only (never from results).

Reuses T02's cell -> image mapping (the manifest.json written by t02_cell_features/t02_features.py --map, scratch only);
no new mapping. Source images for A_res / A_noise = T02's in-domain (IND) pool of the setting (the same source T02 used
for its order / quantile features). Decided before any value was computed.

  A_res   |log2(t_test / t_source)|, t = median over cases of the largest voxel spacing (nibabel header zooms).
          Secondary: same with the smallest spacing (in-plane).
  A_noise |log2(n_test / n_source)|; per image n = MAD of the 3-D discrete Laplacian (scipy.ndimage.laplace) inside
          the body mask (Otsu threshold on the volume, largest connected component) / intensity IQR inside the mask;
          median over cases. The image is the one given to the network (before nnU-Net normalisation), 4-D -> first volume.
  A_merge (source side, one value per setting) on the first 20 cases of fold 0's training list (splits_final.json of
          the preprocessed dataset), read from nnU-Net's PREPROCESSED volumes (what the trainer feeds Palette: after
          nnU-Net normalisation and resampling). Palette's own partition is imported from AugLab
          (auglab.transforms.gpu.fromSeg._kmeans_1d) and its assignment is reproduced line by line (min-max to [0, 1],
          foreground = > dark_threshold 0.01, 40k random voxels -> foreground -> first 10k -> 1-D k-means with C clusters,
          midpoint thresholds, bucketize). Background voxels (<= 0.01, zeroed by Palette) form their own region (-1).
          For every C in 2..6 and every target class c (the eval classes of the setting's OOD cells, mapped to training
          label ids through the settings snapshot): merge = fraction of the class's voxels assigned to the region that
          holds the majority of its 3 mm ring (spacing-aware EDT, (0, 3] mm, not in c; as T02's ring). Case value =
          class-macro mean; setting value = mean over (case, C).
          Deviations from Palette (recorded): whole volume instead of a training patch; fixed seed per (case, C).

Modes (job wrapper t07_axes_job.sh, never on a login node):
  --images   per-image spacing + noise for every distinct image of the task's T02 manifest -> <scratch>/case_axes.tsv
  --gate     noise-response gate on 3 test images of the task -> <out>/gate_noise.tsv
  --train    sensitivity source = training images (raw imagesTr, first 20 of fold 0) -> <scratch>/train_axes.tsv
  --merge    A_merge per setting -> <scratch>/merge_cases.tsv, <out>/merge_settings.tsv
  --aggregate cell-level axes -> <out>/cells_axes.tsv, <out>/gate_counts.tsv (no case ids in <out>)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

R = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
SNAP = R / "benchmark/01_commun_results/gain_decomposition/v2/settings_snapshot.yaml"
T02_SCR = Path("/scratch/paulh/dg_mechanisms/T02")
T02_OUT = R / "benchmark/01_commun_results/dg_mechanisms/T02"
AUGLAB = R / "sub-workspaces/auglab_workspace/AugLab"
RING_MM = 3.0
C_RANGE = (2, 3, 4, 5, 6)
N_TRAIN = 20
DARK = 0.01            # Palette's dark_threshold
N_SAMP, N_SUB = 40_000, 10_000   # Palette's random voxel draw and k-means subsample


# ── per-image measures ───────────────────────────────────────────────────────
def load_img(path: str):
    import nibabel as nib
    im = nib.load(path)
    zooms = [float(z) for z in im.header.get_zooms()[:3]]
    return im, zooms


def noise_level(a: np.ndarray) -> tuple:
    """n = MAD(Laplacian | body) / IQR(intensity | body); body = Otsu + largest component."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    thr = threshold_otsu(a)
    m = a > thr
    lab, k = ndimage.label(m)
    if k == 0:
        return float("nan"), float("nan"), 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    m = lab == int(np.argmax(sizes))
    v = a[m]
    q25, q75 = np.percentile(v, [25, 75])
    iqr = float(q75 - q25)
    lap = ndimage.laplace(a)[m]
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return (mad / iqr if iqr > 0 else float("nan")), iqr, int(m.sum())


def image_row(path: str) -> dict:
    im, zooms = load_img(path)
    a = np.asanyarray(im.dataobj).astype(np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    n, iqr, nb = noise_level(a)
    return dict(img=path, sp_max=max(zooms), sp_min=min(zooms), noise=n, iqr=iqr, n_body=nb)


def _img_work(path):
    try:
        return path, image_row(path), None
    except Exception as e:  # noqa: BLE001
        return path, None, f"{type(e).__name__}: {e}"


def gate_row(path: str, seed: int) -> dict:
    im, _ = load_img(path)
    a = np.asanyarray(im.dataobj).astype(np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    n0, iqr, _ = noise_level(a)
    rng = np.random.default_rng(seed)
    b = a + rng.normal(0.0, 0.05 * iqr, a.shape).astype(np.float32)
    n1, _, _ = noise_level(b)
    return dict(n_clean=n0, n_noisy=n1, ratio=n1 / n0 if n0 > 0 else float("nan"), sigma=0.05 * iqr)


# ── Palette partition (imported k-means, assignment as in fromSeg.RandomV26_6_2ContrastGPU) ──
def palette_regions(img: np.ndarray, C: int, seed: int) -> np.ndarray:
    import torch
    torch.set_num_threads(1)
    if str(AUGLAB) not in sys.path:
        sys.path.insert(0, str(AUGLAB))
    from auglab.transforms.gpu.fromSeg import _kmeans_1d
    flat_all = torch.from_numpy(img.astype(np.float32).ravel())
    v_min, v_max = flat_all.min(), flat_all.max()
    flat = ((flat_all - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)
    fg = flat > DARK
    N = flat.numel()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, N, (min(N, N_SAMP),), generator=g)
    samp = flat[idx]
    sub_fg = samp[samp > DARK][:N_SUB]
    if sub_fg.numel() < 4:
        sub_fg = samp[:N_SUB]
    centroids = _kmeans_1d(sub_fg, C)
    sorted_c, sort_idx = torch.sort(centroids)
    boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
    lbl_s = torch.bucketize(flat, boundaries)
    lbl_l = sort_idx[lbl_s].long()
    lbl_l[~fg] = -1
    return lbl_l.numpy().reshape(img.shape)


def class_rings(seg: np.ndarray, classes: list, sp: np.ndarray) -> dict:
    """{name: (slices, target mask in crop, ring mask in crop)} for classes present."""
    from scipy.ndimage import distance_transform_edt
    out = {}
    for name, pid in classes:
        tgt = seg == pid
        if not tgt.any():
            continue
        idx = np.argwhere(tgt)
        m = np.ceil(RING_MM / sp).astype(int) + 1
        lo = np.maximum(idx.min(0) - m, 0)
        hi = np.minimum(idx.max(0) + m + 1, np.array(seg.shape))
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        st = tgt[sl]
        dt = distance_transform_edt(~st, sampling=sp)
        ring = (dt > 0) & (dt <= RING_MM)
        if ring.sum() == 0:
            continue
        out[name] = (sl, st, ring)
    return out


def merge_case(job) -> list:
    data_p, seg_p, classes, sp, case_i = job
    import blosc2
    blosc2.set_nthreads(1)
    img = np.asarray(blosc2.open(data_p, mode="r")[:])[0]
    seg = np.asarray(blosc2.open(seg_p, mode="r")[:])[0]
    rings = class_rings(seg, classes, np.asarray(sp, float))
    rows = []
    for C in C_RANGE:
        reg = palette_regions(img, C, seed=1000 * case_i + C)
        for name, (sl, st, ring) in rings.items():
            r = reg[sl]
            vals, cnt = np.unique(r[ring], return_counts=True)
            maj = vals[np.argmax(cnt)]
            rows.append(dict(C=C, label=name, merge=float((r[st] == maj).mean()), n_target=int(st.sum()),
                             ring_bg=float((r[ring] == -1).mean())))
    return rows


def _merge_work(job):
    try:
        return job, merge_case(job), None
    except Exception as e:  # noqa: BLE001
        return job, None, f"{type(e).__name__}: {e}"


# ── helpers ──────────────────────────────────────────────────────────────────
def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k)))
                              for k in keys) + "\n")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def setting_key(cell):
    return f"{cell['task']}_{cell['modality']}"


def training_info(snap_setting: dict) -> dict:
    """Dataset dirs and fold-0 training list of a setting (from the texture model's dataset name)."""
    md = Path(snap_setting["models"]["texture"]["model_dir"])
    dsname = md.parts[-2]
    base = R / str(md).split("/8_results_")[0]
    pre = next(base.glob(f"2_nnUNet_*/preprocessed/{dsname}"))
    raw = next(base.glob(f"2_nnUNet_*/raw/{dsname}"))
    plans = json.load(open(pre / "nnUNetPlans.json"))
    cfg = plans["configurations"]["3d_fullres"]
    # nnU-Net stores the preprocessed arrays transposed by transpose_forward and lists the 3d_fullres spacing in that
    # same (transposed) order, so cfg["spacing"] matches the array axes whatever the transpose (Brain DWI: [1, 0, 2]).
    splits = json.load(open(pre / "splits_final.json"))
    train = list(splits[0]["train"])
    return dict(dataset=dsname, pre=pre, raw=raw, spacing=cfg["spacing"], data_id=cfg.get("data_identifier", "nnUNetPlans_3d_fullres"),
                norm=cfg["normalization_schemes"], mask_norm=cfg["use_mask_for_norm"], train=train[:N_TRAIN],
                transpose=list(plans.get("transpose_forward", [0, 1, 2])),
                n_train_fold0=len(train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    for m in ("images", "gate", "train", "merge", "aggregate"):
        ap.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(T02_SCR / f"features_{a.task}" / "manifest.json"))
    snap = yaml.safe_load(SNAP.read_text())["settings"]
    settings = sorted({setting_key(c) for c in man.values()})
    P = lambda *x: print(*x, flush=True)  # noqa: E731

    if a.images:
        paths = sorted({c["img"] for cell in man.values() for c in cell["cases"]})
        P(f"{len(paths)} distinct images")
        rows, errs = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(_img_work, p) for p in paths]
            for i, f in enumerate(as_completed(futs)):
                p, r, e = f.result()
                (errs.append(f"{p}: {e}") if e else rows.append(r))
                if i % 200 == 0:
                    P(f"  {i}/{len(paths)}")
        write(a.scratch / "case_axes.tsv", rows)
        (a.scratch / "case_axes_errors.txt").write_text("\n".join(errs) + "\n")
        P(f"images: {len(rows)} rows, {len(errs)} errors")

    if a.gate:
        ood = sorted({c["img"] for cell in man.values() if cell["kind"] == "OOD" for c in cell["cases"]})
        pick = [ood[int(round(i * (len(ood) - 1) / 2))] for i in range(3)] if len(ood) >= 3 else ood
        rows = []
        for i, p in enumerate(pick):
            g = gate_row(p, seed=7 + i)
            cell = next(cl for cl in man.values() if any(c["img"] == p for c in cl["cases"]))
            rows.append(dict(task=a.task, image_index=i, setting=cell["setting"], group=cell["column"], **g,
                             ok=bool(g["ratio"] > 1)))
        write(a.out / "gate_noise.tsv", rows)
        P("noise gate:", [(r["group"], round(r["ratio"], 3)) for r in rows])

    if a.train:
        rows = []
        for s in settings:
            ti = training_info(snap[s])
            for case in ti["train"]:
                p = ti["raw"] / "imagesTr" / f"{case}_0000.nii.gz"
                try:
                    r = image_row(str(p))
                    rows.append(dict(setting_key=s, case=case, **{k: v for k, v in r.items() if k != "img"}))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(setting_key=s, case=case, error=f"{type(e).__name__}: {e}"))
        write(a.scratch / "train_axes.tsv", rows)
        P(f"train-source sensitivity: {len(rows)} rows")

    if a.merge:
        pcls = read(T02_OUT / "cells_per_class.tsv")
        jobs, meta = [], {}
        for s in settings:
            st = snap[s]
            setting_name = next(c["setting"] for c in man.values() if setting_key(c) == s)
            names = sorted({r["label"] for r in pcls if r["setting"] == setting_name})
            tri = {}
            for it in st["items"]:
                for t in it["triples"]:
                    tri[t[0]] = t[1]
            classes = [(n, int(tri[n])) for n in names]
            ti = training_info(st)
            meta[s] = dict(setting=setting_name, classes=classes, **{k: ti[k] for k in ("dataset", "spacing", "norm", "mask_norm", "n_train_fold0", "transpose")},
                           n_used=len(ti["train"]))
            for i, case in enumerate(ti["train"]):
                d = ti["pre"] / ti["data_id"]
                jobs.append((str(d / f"{case}.b2nd"), str(d / f"{case}_seg.b2nd"), classes, ti["spacing"], i))
        P(f"merge: {len(jobs)} training cases over {len(settings)} settings")
        case_rows, errs = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(_merge_work, j) for j in jobs]
            for f in as_completed(futs):
                job, rows, e = f.result()
                s = next(k for k, m in meta.items() if Path(job[0]).parent.parent.name == m["dataset"])
                if e:
                    errs.append(f"{s} case#{job[4]}: {e}")
                    continue
                for r in rows:
                    case_rows.append(dict(setting_key=s, case=Path(job[0]).stem, case_i=job[4], **r))
        write(a.scratch / "merge_cases.tsv", case_rows)
        (a.scratch / "merge_errors.txt").write_text("\n".join(errs) + "\n")
        out = []
        for s, m in meta.items():
            rs = [r for r in case_rows if r["setting_key"] == s]
            per = defaultdict(list)                       # (case, C) -> class merges
            for r in rs:
                per[(r["case_i"], r["C"])].append(r["merge"])
            vals = [float(np.mean(v)) for v in per.values()]
            per_c = {C: float(np.mean([float(np.mean(v)) for (ci, cc), v in per.items() if cc == C])) for C in C_RANGE
                     if any(cc == C for (_, cc) in per)}
            out.append(dict(setting_key=s, setting=m["setting"], dataset=m["dataset"], A_merge=float(np.mean(vals)) if vals else float("nan"),
                            n_cases=len({ci for ci, _ in per}), n_case_C=len(vals), n_train_fold0=m["n_train_fold0"],
                            n_classes=len(m["classes"]), spacing=str(m["spacing"]), transpose=str(m["transpose"]), norm=str(m["norm"]), mask_norm=str(m["mask_norm"]),
                            **{f"merge_C{C}": v for C, v in per_c.items()}))
        write(a.out / "merge_settings.tsv", out)
        P(f"merge: {len(case_rows)} rows, {len(errs)} errors;", [(o["setting"], round(o["A_merge"], 3)) for o in out])

    if a.aggregate:
        img = {r["img"]: r for r in read(a.scratch / "case_axes.tsv")}
        f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")  # noqa: E731
        cells_tsv = {(r["setting"], r["group"]): r for r in read(T02_OUT / "cells.tsv")}
        train = defaultdict(list)
        if (a.scratch / "train_axes.tsv").exists():
            for r in read(a.scratch / "train_axes.tsv"):
                if not r.get("error"):
                    train[r["setting_key"]].append(r)

        def pool(cases):
            rs = [img[c["img"]] for c in cases if c["img"] in img]
            return (float(np.median([f(r, "sp_max") for r in rs])) if rs else float("nan"),
                    float(np.median([f(r, "sp_min") for r in rs])) if rs else float("nan"),
                    float(np.nanmedian([f(r, "noise") for r in rs])) if rs else float("nan"), len(rs),
                    sum(np.isfinite(f(r, "noise")) for r in rs))

        def la(x, y):
            return abs(math.log2(x / y)) if x > 0 and y > 0 else float("nan")

        by_set = defaultdict(list)
        for cell in man.values():
            by_set[cell["setting"]].append(cell)
        out, gate = [], []
        for s, cs in sorted(by_set.items()):
            ind = [c for cell in cs if cell["kind"] == "IND" for c in cell["cases"]]
            ts, tsi, ns, nis, nns = pool(ind)
            sk = setting_key(cs[0])
            tr = train.get(sk, [])
            tr_s = float(np.median([f(r, "sp_max") for r in tr])) if tr else float("nan")
            tr_n = float(np.nanmedian([f(r, "noise") for r in tr])) if tr else float("nan")
            gate.append(dict(setting=s, group="(IND source pool)", n_manifest=len(ind), n_spacing_read=nis,
                             n_noise_ok=nns, n_cells_tsv="", ok=bool(nis == len(ind))))
            for cell in cs:
                if cell["kind"] != "OOD":
                    continue
                tt, tti, nt, nit, nnt = pool(cell["cases"])
                ref = cells_tsv.get((s, cell["column"]))
                ncell = int(ref["n_cases"]) if ref else -1
                per_item = Counter(f"{c['dataset']}:{c['group']}" for c in cell["cases"])
                read_item = Counter(f"{c['dataset']}:{c['group']}" for c in cell["cases"] if c["img"] in img)
                for it, n in sorted(per_item.items()):
                    gate.append(dict(setting=s, group=cell["column"], item=it, n_manifest=n, n_spacing_read=read_item[it],
                                     ok=bool(read_item[it] == n)))
                gate.append(dict(setting=s, group=cell["column"], item="(cell)", n_manifest=len(cell["cases"]),
                                 n_spacing_read=nit, n_noise_ok=nnt, n_cells_tsv=ncell,
                                 ok=bool(nit == len(cell["cases"]) == ncell)))
                out.append(dict(setting=s, task=cell["task"], setting_key=sk, group=cell["column"], n_cases=nit,
                                t_test=tt, t_source=ts, A_res=la(tt, ts), t_test_inplane=tti, t_source_inplane=tsi,
                                A_res_inplane=la(tti, tsi), n_test=nt, n_source=ns, A_noise=la(nt, ns),
                                t_source_train=tr_s, A_res_train=la(tt, tr_s), n_source_train=tr_n, A_noise_train=la(nt, tr_n),
                                n_train_imgs=len(tr)))
        write(a.out / "cells_axes.tsv", out)
        write(a.out / "gate_counts.tsv", gate)
        P(f"aggregate: {len(out)} OOD cells; gate rows {len(gate)}, failing {sum(not g['ok'] for g in gate)}")


if __name__ == "__main__":
    main()
