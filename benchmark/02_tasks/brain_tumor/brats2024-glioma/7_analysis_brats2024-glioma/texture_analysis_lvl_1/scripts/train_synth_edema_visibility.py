#!/usr/bin/env python3
"""Training-side test: is glioma edema VISIBLE in the synthesised training images of ladder rung 4.5 (flat fill) vs rung 5 (real fill)?

Hypothesis under test (from the output probe: flat-fill models call much of the edema background, mostly when T1n-trained, and their edema
call falls with distance from the tumour core): in T1n, edema falls in the SAME k-means bin / Voronoi region as the adjacent normal tissue,
so with a flat fill it gets the SAME value as that tissue (invisible) unless (a) the per-label remap hits the edema label, or (b) it is
recognised by proximity to the core. T2w edema is bright -> own bin -> visible. Rung 5 keeps within-region variation -> stays separable.

What is reproduced exactly: the two AugLab GPU transforms (RandomV26_6_2NoiseFillContrastGPU with the rung-4.5 config,
RandomV26_6_2ContrastGPU with the rung-5 config) on nnU-Net-preprocessed 3d_fullres training patches (z-scored, background 0, seg -1 outside
the nonzero region, padded data 0 / seg -1), patch 128x160x112. The transform bodies are copied here (B=1) only to RECORD the k-means bin,
Voronoi region id and per-label remap flags; `--verify` proves bit-identical outputs vs the real AugLab classes on CPU for many seeds.
Not reproduced: nnU-Net's CPU rotation/scaling (p=0.2 each), GPU flip, and the 67% random-location patches (all draws here are centred on a
random tumour voxel, i.e. the foreground-oversampled draws). Synthesis is applied to 100% of the analysed draws (in training: 50%).

Paired design: the same seed gives the same partition (bins, Voronoi cells, region means mu_c, first blur sigma) in both rungs (their RNG
streams only diverge after mu_c); the label step and second blur then differ.

Metrics per draw (masks from the full case volume, cropped to the patch):
  ring   = label 0 inside the brain, 2 < EDT(whole tumour) <= 5 voxels (1 mm iso)
  r(v)   = nearest ring voxel of edema voxel v;  block(v) = 8-voxel cube containing r(v)
  * same_bin / same_region : k-means bin / step-1 region id of v == that of r(v)  (rung-independent premise check)
  * identical (4.5)        : |s(v) - s(r(v))| < 1e-3 z (final flat image; same value as the adjacent normal tissue)
  * separability per block : |2*AUC - 1| of edema values (with r(v) in the block) vs ring values in the block (ties count 0.5 -> flat
                             same-value = 0); invisible = edema voxels in blocks with separability < 0.2
  * CNR per block          : |mean_E - mean_R| / max(pooled SD, 0.05), capped at 10
  all also per distance-to-core bin (core = NCR|ET|RC), and the same separability for the core (labels 1,3,4) vs ring.

Usage (run inside a Slurm job only):
  python train_synth_edema_visibility.py --verify --data-root R --cases-file F
  python train_synth_edema_visibility.py --run --contrast t1n --data-root R --cases-file F --case-slice 0:4 --draws 30 --out O
  python train_synth_edema_visibility.py --grid --data-root R --cases-file F --out O
  python train_synth_edema_visibility.py --report --out O
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from auglab.transforms.gpu.fromSeg import (RandomV26_6_2ContrastGPU, _kmeans_1d, _voronoi_region_ids, _gaussian_blur_3d)
from auglab.transforms.gpu.palette_noisefill import RandomV26_6_2NoiseFillContrastGPU

PATCH = (128, 160, 112)
DSETS = {"t1n": "Dataset051_BraTS2024GliomaT1n", "t2w": "Dataset052_BraTS2024GliomaT2w"}
CFG_FILES = {"flat": "transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json",
             "real": "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"}
CFG_KEYS = {"flat": "ImageContrastV26_6_2NoiseFillGPUTransform", "real": "ImageContrastV26_6_2GPUTransform"}
DIST_BINS = [0, 2, 5, 10, 15, 1e9]
DIST_NAMES = ["0-2", "2-5", "5-10", "10-15", ">15"]
BLOCK = 8
SEP_INVISIBLE = 0.2


# ───────────────────────── transforms ─────────────────────────
def load_cfg(auglab_root: Path, rung: str) -> dict:
    cfg = json.load(open(auglab_root / "configs" / CFG_FILES[rung]))[CFG_KEYS[rung]]
    return {k: v for k, v in cfg.items() if k != "probability"}


def make_real_class(rung, cfg):
    if rung == "flat":
        return RandomV26_6_2NoiseFillContrastGPU(p=1.0, **cfg)
    return RandomV26_6_2ContrastGPU(p=1.0, **cfg)


@torch.no_grad()
def synth_instrumented(T, rung: str, inp: torch.Tensor, seg: torch.Tensor):
    """Verbatim copy of the AugLab apply_transform bodies for B=1 (flat: label_fill_noise False; real: pv_prob 0) with recording.
    T = the real class instance (parameters read from it). Returns (out, info)."""
    labels = seg.long()
    B, _C, D, H, W = inp.shape
    N = D * H * W
    device = inp.device
    eps = 1e-7
    alpha_lo, alpha_hi = T.alpha_magnitude_range
    if rung == "flat":
        std_lo, std_hi = T.noise_std_range
        assert not T.label_fill_noise
    else:
        assert T.pv_prob == 0
    info = {"global": False, "C_k": 1, "lbl_l": None, "rid": None, "remap": {}}

    flat_all = inp[:, 0].float().reshape(B, N)
    v_min = flat_all.min(dim=1).values.view(B, 1)
    v_max = flat_all.max(dim=1).values.view(B, 1)
    images_01 = ((flat_all - v_min) / (v_max - v_min + eps)).clamp(0, 1)
    flat_m_all = (images_01 > T.dark_threshold).float()
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"), dim=-1).reshape(N, 3)

    flat = images_01[0]
    flat_m = flat_m_all[0]
    n_fg = flat_m.sum()
    if n_fg < 4 or torch.rand(1, device=device).item() < T.skip_parcellation_prob:
        info["global"] = True
        if rung == "flat":
            mu = torch.rand(1, device=device).item()
            sigma = torch.rand(1, device=device).item() * (std_hi - std_lo) + std_lo
            noise = torch.randn(N, device=device)
            synth_i = (mu + sigma * noise).clamp(0, 1) * flat_m
        else:
            b_mean_i = (flat * flat_m).sum() / n_fg.clamp(min=1)
            mu = torch.rand(1, device=device).item()
            sign = (torch.rand(1, device=device) > 0.5).float() * 2 - 1
            mag = torch.rand(1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
            alpha = (sign * mag).item()
            synth_i = (mu + alpha * (flat - b_mean_i)).clamp(0, 1) * flat_m
        info["lbl_l"] = torch.zeros(N, dtype=torch.long)
        info["rid"] = torch.zeros(N, dtype=torch.long)
    else:
        C_k = T.c_choices[int(torch.rand(1, device=device).item() * len(T.c_choices))]
        idx = torch.randint(0, N, (min(N, 40_000),), device=device)
        samp = flat[idx]
        sub_fg = samp[samp > T.dark_threshold][:T.n_kmeans_subsample]
        if sub_fg.numel() < 4:
            sub_fg = samp[:T.n_kmeans_subsample]
        centroids = _kmeans_1d(sub_fg, C_k)
        sorted_c, sort_idx = torch.sort(centroids)
        boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
        lbl_s = torch.bucketize(flat, boundaries)
        lbl_l = sort_idx[lbl_s].long()
        rid, R = _voronoi_region_ids(coords, lbl_l, flat_m, C_k, device, T.s_choices, T.skip_sub_parc_prob)
        s_c = torch.zeros(R, device=device).scatter_add_(0, rid, flat * flat_m)
        n_c = torch.zeros(R, device=device).scatter_add_(0, rid, flat_m)
        mean_c = s_c / n_c.clamp(min=eps)
        mu_c = torch.rand(R, device=device)
        if rung == "flat":
            sigma_c = torch.rand(R, device=device) * (std_hi - std_lo) + std_lo
            noise = torch.randn(N, device=device)
            synth_i = (mu_c[rid] + sigma_c[rid] * noise).clamp(0, 1) * flat_m
        else:
            mag_c = torch.rand(R, device=device) * (alpha_hi - alpha_lo) + alpha_lo
            sign_c = (torch.rand(R, device=device) > 0.5).float() * 2 - 1
            alp_c = mag_c * sign_c
            synth_i = (mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m
        info.update(C_k=C_k, lbl_l=lbl_s.long().cpu(), rid=rid.cpu())   # bin index in sorted-intensity order
    synth = synth_i.view(1, N)
    synth_01 = synth.reshape(B, 1, D, H, W)
    sigma_blur = random.choice(T.blur_sigmas)
    info["blur1"] = sigma_blur
    if sigma_blur > 0.0:
        synth_01 = _gaussian_blur_3d(synth_01, sigma_blur)
        synth = synth_01.reshape(B, N)

    if labels.shape[2:] != (D, H, W):
        labels = F.interpolate(labels.float(), size=(D, H, W), mode="nearest").long()
    lbl = labels[:, 0].reshape(B, N).clamp(min=0)
    unique_classes = lbl.unique()
    unique_classes = unique_classes[unique_classes > 0]
    if T.label_classes is not None:
        keep = torch.tensor(T.label_classes, device=device)
        unique_classes = unique_classes[torch.isin(unique_classes, keep)]
    for c in unique_classes:
        c_val = int(c.item())
        c_mask = (lbl == c_val).float()
        c_cnt = c_mask.sum(dim=1, keepdim=True)
        apply = ((torch.rand(B, 1, device=device) < T.label_remap_prob) & (c_cnt >= T.min_label_voxels)).float()
        info["remap"][c_val] = bool(apply.sum() > 0)
        if apply.sum() == 0:
            continue
        if rung == "flat":
            mu_c = torch.rand(B, 1, device=device)
            c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)
        else:
            c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)
            mu_c = torch.rand(B, 1, device=device)
        mag_c = torch.rand(B, 1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
        sign_c = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1
        alp_c = mag_c * sign_c
        new_vals = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)
        write_mask = c_mask * apply
        synth = synth * (1.0 - write_mask) + new_vals * write_mask

    synth_01 = synth.reshape(B, 1, D, H, W)
    sigma2 = random.choice(T.blur_sigmas)
    info["blur2"] = sigma2
    if sigma2 > 0.0:
        synth_01 = _gaussian_blur_3d(synth_01, sigma2)
        synth = synth_01.reshape(B, N)
    b_sum = (synth * flat_m_all).sum(dim=1, keepdim=True)
    b_cnt = flat_m_all.sum(dim=1, keepdim=True).clamp(min=1)
    b_mean = b_sum / b_cnt
    b_sq = ((synth - b_mean) * flat_m_all).pow(2).sum(dim=1, keepdim=True)
    b_std = (b_sq / b_cnt + eps).sqrt()
    synth_z = ((synth - b_mean) / b_std * flat_m_all).reshape(B, 1, D, H, W)
    out = inp.clone()
    out[:, 0:1] = synth_z.to(inp.dtype)
    return out, info


# ───────────────────────── data ─────────────────────────
def load_case(data_root: Path, contrast: str, case: str):
    import blosc2
    d = data_root / DSETS[contrast] / "nnUNetPlans_3d_fullres"
    img = blosc2.open(urlpath=str(d / f"{case}.b2nd"), mode="r")[:].astype(np.float32)[0]
    seg = blosc2.open(urlpath=str(d / f"{case}_seg.b2nd"), mode="r")[:].astype(np.int16)[0]
    return img, seg


def case_geometry(seg: np.ndarray):
    from scipy import ndimage as ndi
    wt = seg > 0
    core = np.isin(seg, (1, 3, 4))
    d_wt = ndi.distance_transform_edt(~wt)
    ring = (seg == 0) & (d_wt > 2) & (d_wt <= 5)
    _, ring_idx = ndi.distance_transform_edt(~ring, return_indices=True)
    d_core = ndi.distance_transform_edt(~core) if core.any() else np.full(seg.shape, np.inf, np.float32)
    return {"ring": ring, "ring_idx": ring_idx.astype(np.int32), "d_core": d_core.astype(np.float32), "core": core}


def patch_box(shape, centre):
    starts = []
    for s, p, c in zip(shape, PATCH, centre):
        lo = c - p // 2
        lo = int(np.clip(lo, min(0, s - p), max(0, s - p))) if s >= p else (s - p) // 2
        starts.append(lo)
    return starts


def crop(arr, starts, fill):
    out = np.full(PATCH, fill, dtype=arr.dtype)
    src, dst = [], []
    for s, p, lo in zip(arr.shape, PATCH, starts):
        a, b = max(lo, 0), min(lo + p, s)
        src.append(slice(a, b)); dst.append(slice(a - lo, b - lo))
    out[tuple(dst)] = arr[tuple(src)]
    return out


# ───────────────────────── metrics ─────────────────────────
def block_sep(vals_e, keys_e, vals_r, keys_r, min_n=10):
    """Per-group |2AUC-1| and CNR of edema values vs ring values sharing the same group key (group = block[, dist bin]).
    Ring groups are keyed by block only; keys_e = (block, sub) where sub is the dist bin (or 0). Returns arrays per (block, sub)."""
    if vals_e.size == 0 or vals_r.size == 0:
        return None
    rb, r_inv = np.unique(keys_r, return_inverse=True)
    nr = np.bincount(r_inv)
    order = np.lexsort((vals_r, r_inv))
    rs_vals = vals_r[order].astype(np.float64)
    rs_grp = r_inv[order]
    start = np.concatenate([[0], np.cumsum(nr)[:-1]])
    pos = np.searchsorted(rb, keys_e[:, 0])
    ok = (pos < rb.size) & (rb[np.minimum(pos, rb.size - 1)] == keys_e[:, 0])
    ok &= nr[np.minimum(pos, rb.size - 1)] >= min_n
    if not ok.any():
        return None
    g = pos[ok]; ve = vals_e[ok].astype(np.float64); sub = keys_e[ok, 1]
    # searchsorted within each ring group via a combined key (group offset >> value range)
    off = 1e3
    ck = rs_grp * off + np.clip(rs_vals, -400, 400)
    q_key = g * off + np.clip(ve, -400, 400)
    left = np.searchsorted(ck, q_key, side="left") - start[g]
    right = np.searchsorted(ck, q_key, side="right") - start[g]
    q = (left + right) / (2.0 * nr[g])
    # group (ring group, sub)
    gk = g.astype(np.int64) * 16 + sub
    ug, ginv = np.unique(gk, return_inverse=True)
    ne = np.bincount(ginv)
    mq = np.bincount(ginv, q) / ne
    sep = np.abs(2 * mq - 1)
    me = np.bincount(ginv, ve) / ne
    ve2 = np.bincount(ginv, ve * ve) / ne - me ** 2
    rg = ug // 16
    mr = np.bincount(rs_grp, rs_vals)[rg] / nr[rg]
    vr2 = np.bincount(rs_grp, rs_vals ** 2)[rg] / nr[rg] - mr ** 2
    sd = np.sqrt(np.maximum((np.maximum(ve2, 0) + np.maximum(vr2, 0)) / 2, 0.05 ** 2))
    cnr = np.minimum(np.abs(me - mr) / sd, 10.0)
    return {"sub": ug % 16, "n": ne, "sep": sep, "cnr": cnr}


def summarise(res, min_n=10, nsub=1):
    out = {}
    if res is None:
        return out
    for s in range(nsub):
        m = (res["sub"] == s) & (res["n"] >= min_n)
        if not m.any():
            continue
        w = res["n"][m]
        out[s] = {"n": int(w.sum()), "sep": float((res["sep"][m] * w).sum() / w.sum()),
                  "invis": float(w[res["sep"][m] < SEP_INVISIBLE].sum() / w.sum()),
                  "cnr": float((res["cnr"][m] * w).sum() / w.sum())}
    return out


def draw_metrics(vol, seg_p, geo_p, info=None, flat_vals=False):
    """vol: (D,H,W) image (z). geo_p: patch-cropped geometry. Returns dict of metrics."""
    ed = np.flatnonzero((seg_p == 2).ravel())
    ring_p = geo_p["ring"].ravel()
    ridx = geo_p["ring_idx_lin"][ed]          # nearest ring voxel, linear patch index (-1 if outside patch)
    keep = ridx >= 0
    ed, ridx = ed[keep], ridx[keep]
    v = vol.ravel()
    blk = geo_p["block"]
    dcore = geo_p["d_core"].ravel()[ed]
    dbin = np.clip(np.digitize(dcore, DIST_BINS) - 1, 0, len(DIST_NAMES) - 1)
    rr = np.flatnonzero(ring_p)
    out = {"n_edema": int(ed.size)}
    res_all = block_sep(v[ed], np.stack([blk[ridx], np.zeros_like(dbin)], 1), v[rr], blk[rr])
    res_d = block_sep(v[ed], np.stack([blk[ridx], dbin], 1), v[rr], blk[rr])
    out["all"] = summarise(res_all).get(0)
    out["dist"] = {DIST_NAMES[k]: val for k, val in summarise(res_d, nsub=len(DIST_NAMES)).items()}
    if flat_vals:
        same = np.abs(v[ed] - v[ridx]) < 1e-3
        out["identical"] = float(same.mean()) if ed.size else None
        out["identical_dist"] = {DIST_NAMES[k]: float(same[dbin == k].mean()) for k in range(len(DIST_NAMES)) if (dbin == k).sum() >= 20}
    if info is not None:
        lb, ri = info["lbl_l"].numpy(), info["rid"].numpy()
        sb = lb[ed] == lb[ridx]; sr = ri[ed] == ri[ridx]
        out["same_bin"] = float(sb.mean()); out["same_region"] = float(sr.mean())
        out["same_bin_dist"] = {DIST_NAMES[k]: float(sb[dbin == k].mean()) for k in range(len(DIST_NAMES)) if (dbin == k).sum() >= 20}
        out["same_region_dist"] = {DIST_NAMES[k]: float(sr[dbin == k].mean()) for k in range(len(DIST_NAMES)) if (dbin == k).sum() >= 20}
        # which bins (sorted by intensity, 0 = darkest) edema and ring fall in; background (seg -1 / brain 0-z) bin too
        out["edema_bin_mode"] = int(np.bincount(lb[ed]).argmax()) if ed.size else None
        out["ring_bin_mode"] = int(np.bincount(lb[rr]).argmax()) if rr.size else None
        bgv = np.flatnonzero((seg_p.ravel() < 0))
        out["bg_bin_mode"] = int(np.bincount(lb[bgv]).argmax()) if bgv.size else None
        out["C_k"] = info["C_k"]; out["global"] = info["global"]
        out["remap"] = {str(k): v_ for k, v_ in info["remap"].items()}
        out["blur"] = [info["blur1"], info["blur2"]]
    # core vs ring
    co = np.flatnonzero(geo_p["core"].ravel())
    cidx = geo_p["ring_idx_lin"][co]; k2 = cidx >= 0
    if k2.sum() > 0:
        r = summarise(block_sep(v[co[k2]], np.stack([blk[cidx[k2]], np.zeros(k2.sum(), int)], 1), v[rr], blk[rr])).get(0)
        out["core_sep"] = r["sep"] if r else None
    return out


def patch_geometry(geo, starts):
    ring_p = crop(geo["ring"], starts, False)
    dcore_p = crop(geo["d_core"], starts, np.float32(np.inf))
    core_p = crop(geo["core"], starts, False)
    # nearest ring index (full-volume coords) -> patch linear index, -1 if outside patch
    ri = np.stack([crop(geo["ring_idx"][a], starts, -10**6) for a in range(3)])
    loc = ri - np.array(starts).reshape(3, 1, 1, 1)
    inside = np.all((loc >= 0) & (loc < np.array(PATCH).reshape(3, 1, 1, 1)), axis=0)
    lin = np.where(inside, np.ravel_multi_index(np.clip(loc, 0, np.array(PATCH).reshape(3, 1, 1, 1) - 1), PATCH), -1)
    # block id of each patch voxel (in full-volume coords so blocks do not move with the patch)
    gz, gy, gx = np.meshgrid(*[np.arange(p) + s for p, s in zip(PATCH, starts)], indexing="ij")
    blk = ((gz // BLOCK + 50) * 200 + (gy // BLOCK + 50)) * 200 + (gx // BLOCK + 50)
    return {"ring": ring_p, "d_core": dcore_p, "core": core_p, "ring_idx_lin": lin.ravel(), "block": blk.ravel().astype(np.int64)}


# ───────────────────────── modes ─────────────────────────
def get_transforms(auglab_root):
    return {r: make_real_class(r, load_cfg(auglab_root, r)) for r in ("flat", "real")}


def do_verify(args, auglab_root):
    Ts = get_transforms(auglab_root)
    cases = [c.strip() for c in open(args.cases_file) if c.strip()][:2]
    lines = []
    cover = {r: {"global": 0, "blur": 0, "edema_remap": 0, "n": 0, "equal": 0} for r in Ts}
    for contrast in ("t1n", "t2w"):
        for case in cases:
            img, seg = load_case(args.data_root, contrast, case)
            fg = np.argwhere(seg > 0)
            rng = np.random.default_rng(0)
            for s in range(args.verify_seeds):
                starts = patch_box(seg.shape, fg[rng.integers(len(fg))])
                x = torch.from_numpy(crop(img, starts, np.float32(0)))[None, None]
                sg = torch.from_numpy(crop(seg, starts, np.int16(-1)).astype(np.float32))[None, None]
                for r, T in Ts.items():
                    torch.manual_seed(s); random.seed(s)
                    mine, info = synth_instrumented(T, r, x, sg)
                    torch.manual_seed(s); random.seed(s)
                    ref = T.apply_transform(x, {"seg": sg}, {})
                    eq = torch.equal(mine, ref)
                    c = cover[r]; c["n"] += 1; c["equal"] += int(eq); c["global"] += int(info["global"])
                    c["blur"] += int(info["blur1"] > 0 or info["blur2"] > 0); c["edema_remap"] += int(info["remap"].get(2, False))
                    if not eq:
                        lines.append(f"MISMATCH {contrast} {case} seed {s} rung {r}: max|d|={float((mine-ref).abs().max()):.3g}")
    lines.append(json.dumps(cover))
    txt = "\n".join(lines)
    print(txt)
    (args.out / "verify.txt").write_text(txt + "\n")
    ok = all(c["equal"] == c["n"] for c in cover.values())
    print("VERIFY", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def do_run(args, auglab_root):
    Ts = get_transforms(auglab_root)
    cases = [c.strip() for c in open(args.cases_file) if c.strip()]
    if args.case_slice:
        a, b = (int(x) for x in args.case_slice.split(":")); cases = cases[a:b]
    od = args.out / "draws"; od.mkdir(parents=True, exist_ok=True)
    for case in cases:
        of = od / f"{args.contrast}_{case}.jsonl"
        if of.exists():
            print("skip", of); continue
        t0 = time.time()
        img, seg = load_case(args.data_root, args.contrast, case)
        n_ed = int((seg == 2).sum()); n_core = int(np.isin(seg, (1, 3, 4)).sum())
        if n_ed < 10_000 or n_core < 500:
            of.write_text(json.dumps({"case": case, "excluded": True, "n_edema": n_ed, "n_core": n_core}) + "\n")
            print("exclude", case, n_ed, n_core); continue
        geo = case_geometry(seg)
        fg = np.argwhere(seg > 0)
        rng = np.random.default_rng(int(case[-8:]))
        recs = []
        for d in range(args.draws):
            starts = patch_box(seg.shape, fg[rng.integers(len(fg))])
            xp = crop(img, starts, np.float32(0)); sp = crop(seg, starts, np.int16(-1))
            gp = patch_geometry(geo, starts)
            x = torch.from_numpy(xp)[None, None]; sg = torch.from_numpy(sp.astype(np.float32))[None, None]
            rec = {"case": case, "contrast": args.contrast, "draw": d, "starts": starts,
                   "real_img": draw_metrics(xp, sp, gp)}
            seed = 1000 * int(case[-8:]) % 2**31 + d
            for r, T in Ts.items():
                torch.manual_seed(seed); random.seed(seed)
                out, info = synth_instrumented(T, r, x, sg)
                rec[r] = draw_metrics(out[0, 0].numpy(), sp, gp, info=info, flat_vals=(r == "flat"))
            recs.append(rec)
        with open(of, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{case} {args.contrast}: {args.draws} draws in {time.time()-t0:.0f}s", flush=True)


def do_grid(args, auglab_root):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ts = get_transforms(auglab_root)
    case = args.grid_case or [c.strip() for c in open(args.cases_file) if c.strip()][0]
    nd = 4
    fig, axes = plt.subplots(4, 1 + nd, figsize=(2.3 * (1 + nd), 2.5 * 4))
    for row0, contrast in zip((0, 2), ("t1n", "t2w")):
        img, seg = load_case(args.data_root, contrast, case)
        ed_slices = (seg == 2).sum(axis=(1, 2))
        fg = np.argwhere(seg > 0)
        centre = fg.mean(0).astype(int)
        starts = patch_box(seg.shape, centre)
        xp = crop(img, starts, np.float32(0)); sp = crop(seg, starts, np.int16(-1))
        z = int(np.argmax((sp == 2).sum(axis=(1, 2))))
        x = torch.from_numpy(xp)[None, None]; sg = torch.from_numpy(sp.astype(np.float32))[None, None]
        gp = None   # set on the first draw (k == 0, d == 0) and reused for the others
        for k, r in enumerate(("flat", "real")):
            row = row0 + k
            ax = axes[row, 0]
            ax.imshow(xp[z].T, cmap="gray", origin="lower", vmin=np.percentile(xp[z], 1), vmax=np.percentile(xp[z], 99.5))
            ax.contour((sp[z] == 2).T, levels=[0.5], colors="#E69F00", linewidths=0.6, origin="lower")
            ax.contour(np.isin(sp[z], (1, 3, 4)).T, levels=[0.5], colors="#56B4E9", linewidths=0.6, origin="lower")
            ax.set_title(f"{contrast} real (training image)" if k == 0 else "", fontsize=8)
            ax.set_ylabel(f"{contrast}: {'rung 4.5 flat' if r == 'flat' else 'rung 5 real'}", fontsize=8)
            for d in range(nd):
                seed = 777 + d
                torch.manual_seed(seed); random.seed(seed)
                out, info = synth_instrumented(Ts[r], r, x, sg)
                o = out[0, 0, z].numpy()
                gp = patch_geometry(case_geometry(seg), starts) if d == 0 and k == 0 else gp
                m = draw_metrics(out[0, 0].numpy(), sp, gp, info=info, flat_vals=(r == "flat"))
                a = axes[row, 1 + d]
                a.imshow(o.T, cmap="gray", origin="lower", vmin=np.percentile(o, 1), vmax=np.percentile(o, 99.5))
                a.contour((sp[z] == 2).T, levels=[0.5], colors="#E69F00", linewidths=0.6, origin="lower")
                sep = m["all"]["sep"] if m.get("all") else float("nan")
                a.set_title(f"seed {seed}: sep {sep:.2f}, same-reg {m['same_region']:.2f}\nedema remap {'yes' if m['remap'].get('2') else 'no'}",
                            fontsize=6.5)
        for a in axes[row0:row0 + 2].ravel():
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{case}: same seed = same partition in both rungs (orange = edema, blue = core)", fontsize=9)
    fig.tight_layout()
    p = args.out / f"qualitative_grid_{case}.png"
    fig.savefig(p, dpi=150); print("wrote", p)


def do_report(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    recs, excluded = [], []
    for f in sorted((args.out / "draws").glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            (excluded if r.get("excluded") else recs).append(r)
    COLS = {"real_img": "#7F7F7F", "flat": "#D55E00", "real": "#0072B2"}
    NAMES = {"real_img": "real training image", "flat": "rung 4.5 flat fill", "real": "rung 5 real fill"}

    def per_case(contrast, cond, getter, subset=None):
        vals = {}
        for r in recs:
            if r["contrast"] != contrast:
                continue
            if subset is not None and not subset(r):
                continue
            g = getter(r[cond])
            if g is None:
                continue
            vals.setdefault(r["case"], []).append(g)
        return np.array([np.mean(v) for v in vals.values()]), len(vals)

    def fmt(arr):
        return "n/a" if arr.size == 0 else f"{arr.mean():.2f} ± {arr.std(ddof=1) if arr.size > 1 else 0:.2f}"

    L = ["# Training-side test: is edema visible in rung-4.5 (flat) vs rung-5 (real) synthesised training images?", ""]
    ip = args.out / "interpretation.md"
    if ip.exists():
        L += [ip.read_text().strip(), "", "---", ""]
    nc = {c: len({r['case'] for r in recs if r['contrast'] == c}) for c in DSETS}
    nd = {c: sum(1 for r in recs if r['contrast'] == c) for c in DSETS}
    L += [f"Cases analysed: T1n {nc['t1n']} ({nd['t1n']} draws), T2w {nc['t2w']} ({nd['t2w']} draws); excluded (edema < 10 mL or core < 0.5 mL): "
          f"{sorted({e['case'] for e in excluded})}. Values = mean ± SD across cases of per-case means over draws (draws are paired by seed "
          "across rungs: same partition and region means).", ""]
    L += ["## 1. Premise: does edema share its k-means bin / step-1 region with the adjacent normal tissue? (rung-independent partition)", "",
          "| contrast | same k-means bin as nearest ring voxel | same step-1 region (bin x Voronoi) | global (no parcellation) draws |", "|---|---|---|---|"]
    for c in DSETS:
        sb, _ = per_case(c, "flat", lambda m: m.get("same_bin"))
        sr, _ = per_case(c, "flat", lambda m: m.get("same_region"))
        gl, _ = per_case(c, "flat", lambda m: float(m.get("global", False)))
        L.append(f"| {c} | {fmt(sb)} | {fmt(sr)} | {fmt(gl)} |")
    L += ["", "Bin of edema vs ring vs background-0 (mode bin index in intensity order; fraction of draws where edema's modal bin == ring's / == background's):", ""]
    for c in DSETS:
        rr = [r["flat"] for r in recs if r["contrast"] == c and not r["flat"].get("global")]
        if rr:
            er = np.mean([m["edema_bin_mode"] == m["ring_bin_mode"] for m in rr])
            eb = np.mean([m["edema_bin_mode"] == m["bg_bin_mode"] for m in rr if m["bg_bin_mode"] is not None])
            L.append(f"- {c}: edema modal bin == ring modal bin in {er:.2f} of parcellated draws; == background(0 z) modal bin in {eb:.2f}")
    L += ["", "## 2. Visibility of edema vs adjacent ring (2-5 vox outside the lesion)", "",
          "Separability = |2 AUC - 1| per 8-voxel block of the nearest ring voxel (0 = indistinguishable, ties count as equal); "
          f"invisible = fraction of edema voxels in blocks with separability < {SEP_INVISIBLE}; CNR = |dmean| / max(pooled SD, 0.05), capped 10.", "",
          "| contrast | condition | separability | invisible fraction | CNR | identical value to nearest ring voxel |", "|---|---|---|---|---|---|"]
    for c in DSETS:
        for cond in ("real_img", "flat", "real"):
            s, _ = per_case(c, cond, lambda m: m["all"]["sep"] if m.get("all") else None)
            iv, _ = per_case(c, cond, lambda m: m["all"]["invis"] if m.get("all") else None)
            cn, _ = per_case(c, cond, lambda m: m["all"]["cnr"] if m.get("all") else None)
            idt, _ = per_case(c, cond, lambda m: m.get("identical"))
            L.append(f"| {c} | {NAMES[cond]} | {fmt(s)} | {fmt(iv)} | {fmt(cn)} | {fmt(idt) if cond == 'flat' else '-'} |")
    L += ["", "## 3. Visibility vs distance to the tumour core (NCR|ET|RC), separability / invisible fraction", ""]
    hdr = "| contrast | condition | " + " | ".join(DIST_NAMES) + " |"
    L += [hdr, "|---|---|" + "---|" * len(DIST_NAMES)]
    for c in DSETS:
        for cond in ("real_img", "flat", "real"):
            cells = []
            for dn in DIST_NAMES:
                s, _ = per_case(c, cond, lambda m, dn=dn: m["dist"][dn]["sep"] if dn in m.get("dist", {}) else None)
                iv, _ = per_case(c, cond, lambda m, dn=dn: m["dist"][dn]["invis"] if dn in m.get("dist", {}) else None)
                cells.append("n/a" if s.size == 0 else f"{s.mean():.2f} / {iv.mean():.2f} (n={s.size})")
            L.append(f"| {c} | {NAMES[cond]} | " + " | ".join(cells) + " |")
    L += ["", "Rung 4.5 premise by distance (same step-1 region as nearest ring voxel / identical final value):", ""]
    L += [hdr, "|---|---|" + "---|" * len(DIST_NAMES)]
    for c in DSETS:
        for key, nm in (("same_region_dist", "same region"), ("identical_dist", "identical value")):
            cells = []
            for dn in DIST_NAMES:
                s, _ = per_case(c, "flat", lambda m, dn=dn, key=key: m.get(key, {}).get(dn))
                cells.append("n/a" if s.size == 0 else f"{s.mean():.2f}")
            L.append(f"| {c} | {nm} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Rung 4.5: what makes edema visible? split by whether the edema label (2) was remapped in the label step", "",
          "| contrast | edema remapped? | fraction of draws | separability | invisible fraction | identical value |", "|---|---|---|---|---|---|"]
    for c in DSETS:
        allr = [r for r in recs if r["contrast"] == c]
        for flag in (True, False):
            sub = lambda r, flag=flag: bool(r["flat"].get("remap", {}).get("2", False)) == flag
            frac = np.mean([sub(r) for r in allr]) if allr else float("nan")
            s, _ = per_case(c, "flat", lambda m: m["all"]["sep"] if m.get("all") else None, sub)
            iv, _ = per_case(c, "flat", lambda m: m["all"]["invis"] if m.get("all") else None, sub)
            idt, _ = per_case(c, "flat", lambda m: m.get("identical"), sub)
            L.append(f"| {c} | {'yes' if flag else 'no'} | {frac:.2f} | {fmt(s)} | {fmt(iv)} | {fmt(idt)} |")
    L += ["", "Same split for rung 5 (real fill):", "", "| contrast | edema remapped? | separability | invisible fraction |", "|---|---|---|---|"]
    for c in DSETS:
        for flag in (True, False):
            sub = lambda r, flag=flag: bool(r["real"].get("remap", {}).get("2", False)) == flag
            s, _ = per_case(c, "real", lambda m: m["all"]["sep"] if m.get("all") else None, sub)
            iv, _ = per_case(c, "real", lambda m: m["all"]["invis"] if m.get("all") else None, sub)
            L.append(f"| {c} | {'yes' if flag else 'no'} | {fmt(s)} | {fmt(iv)} |")
    L += ["", "Core (NCR|ET|RC) vs ring separability (is the core conspicuous?):", "", "| contrast | real image | rung 4.5 | rung 5 |", "|---|---|---|---|"]
    for c in DSETS:
        cells = [fmt(per_case(c, cond, lambda m: m.get("core_sep"))[0]) for cond in ("real_img", "flat", "real")]
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    # per-case paired table (flat vs real separability)
    L += ["", "Per-case paired separability (real image / flat / real fill):", "", "| contrast | case | real image | 4.5 flat | 5 real | same region |", "|---|---|---|---|---|---|"]
    for c in DSETS:
        for case in sorted({r["case"] for r in recs if r["contrast"] == c}):
            rr = [r for r in recs if r["contrast"] == c and r["case"] == case]
            f_ = lambda cond: np.mean([r[cond]["all"]["sep"] for r in rr if r[cond].get("all")])
            sr = np.mean([r["flat"]["same_region"] for r in rr])
            L.append(f"| {c} | {case} | {f_('real_img'):.2f} | {f_('flat'):.2f} | {f_('real'):.2f} | {sr:.2f} |")

    # ── figures ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, c in zip(axes, DSETS):
        for cond in ("real_img", "flat", "real"):
            ys = [per_case(c, cond, lambda m, dn=dn: m["dist"][dn]["sep"] if dn in m.get("dist", {}) else None)[0] for dn in DIST_NAMES]
            mean = [y.mean() if y.size else np.nan for y in ys]
            ax.plot(range(len(DIST_NAMES)), mean, "-o", color=COLS[cond], lw=2, ms=6, label=NAMES[cond])
        sr = [per_case(c, "flat", lambda m, dn=dn: m.get("same_region_dist", {}).get(dn))[0] for dn in DIST_NAMES]
        ax.plot(range(len(DIST_NAMES)), [y.mean() if y.size else np.nan for y in sr], "--", color="#555555", lw=1.2,
                label="frac. edema in same step-1 region as ring")
        ax.set_xticks(range(len(DIST_NAMES))); ax.set_xticklabels(DIST_NAMES); ax.set_xlabel("distance to tumour core (mm)")
        ax.set_title(f"{c.upper()}-source training images", fontsize=10); ax.grid(alpha=0.25); ax.set_ylim(0, 1)
    axes[0].set_ylabel("edema vs adjacent-ring separability |2AUC-1|")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(args.out / "separability_vs_core_distance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, c in zip(axes, DSETS):
        xs = 0
        labels = []
        for cond, sub, nm in (("real_img", None, "real image"), ("flat", lambda r: r["flat"].get("remap", {}).get("2", False), "4.5, edema remapped"),
                              ("flat", lambda r: not r["flat"].get("remap", {}).get("2", False), "4.5, not remapped"),
                              ("real", lambda r: r["real"].get("remap", {}).get("2", False), "5, edema remapped"),
                              ("real", lambda r: not r["real"].get("remap", {}).get("2", False), "5, not remapped")):
            v, _ = per_case(c, cond, lambda m: m["all"]["invis"] if m.get("all") else None, sub)
            ax.bar(xs, v.mean() if v.size else 0, width=0.7, color=COLS[cond], alpha=0.85)
            ax.scatter(np.full(v.size, xs) + np.linspace(-0.2, 0.2, v.size), v, s=10, color="#222222", zorder=3)
            labels.append(nm); xs += 1
        ax.set_xticks(range(xs)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{c.upper()}-source", fontsize=10); ax.grid(axis="y", alpha=0.25); ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction of edema invisible\n(block separability < 0.2)")
    fig.tight_layout(); fig.savefig(args.out / "invisible_fraction_by_remap.png", dpi=150); plt.close(fig)

    L += ["", "Figures: `separability_vs_core_distance.png`, `invisible_fraction_by_remap.png`, `qualitative_grid_*.png`.", ""]
    vt = args.out / "verify.txt"
    if vt.exists():
        L += ["## Verification (instrumented copy vs the real AugLab classes, CPU, bitwise)", "", "```", vt.read_text().strip(), "```", ""]
    mf = args.out / "provenance.txt"
    if mf.exists():
        L += ["## Provenance", "", "```", mf.read_text().strip(), "```", ""]
    (args.out / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("verify", "run", "grid", "report"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--auglab-root", type=Path, default=None, help="auglab package dir (default: imported module's)")
    ap.add_argument("--cases-file")
    ap.add_argument("--case-slice", default="")
    ap.add_argument("--contrast", choices=list(DSETS))
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--verify-seeds", type=int, default=12)
    ap.add_argument("--grid-case", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    import auglab.transforms.gpu.fromSeg as _fs, auglab.transforms.gpu.palette_noisefill as _nf
    auglab_root = args.auglab_root or Path(_fs.__file__).resolve().parents[2]
    print("auglab modules from", _fs.__file__, _nf.__file__, flush=True)
    if not args.report:
        with open(args.out / "provenance.txt", "a") as f:
            f.write(f"imported: {_fs.__file__} {_nf.__file__}\n")
    if args.verify:
        do_verify(args, auglab_root)
    elif args.run:
        do_run(args, auglab_root)
    elif args.grid:
        do_grid(args, auglab_root)
    else:
        do_report(args)


if __name__ == "__main__":
    main()
