#!/usr/bin/env python3
"""Texture-reliance experiment, perturbation T (see README.md): label-free fine+mid band phase scrambling.

Per image x (the nnU-Net test input, its own voxel grid, SimpleITK = nnU-Net's reader):
  low  = G_s(x), band = x - low            (Gaussian, sigma s = 2 voxels per axis: the detail above the sigma-2 scale)
  scr  = phase-randomised copy of band: same Fourier amplitude spectrum, random Hermitian phases (phases of the rfftn
         of real white noise, global DC bin kept), multiplied by the local-energy gain
         g = sqrt((G_2(band^2) + eps) / (G_2(scr^2) + eps)) re-estimated and applied twice (eps = 1e-3 mean(band^2) in
         the body), masked to the body, rescaled so its energy in the body equals the band's
         (verbatim the "allfine_le" variant of FINDINGS 41: brats2024-glioma/7_analysis_brats2024-glioma/
         texture_analysis_lvl_1/scripts/visual_proof_scramble.py, phase_randomise + scramble(local=True))
  x'   = x - body * band + scr             (outside the body x is untouched)
body = label-free foreground of x alone: v = min-max of x clipped to its own [0.5, 99.5] percentiles, v > 0.05,
       closed (2 iterations), holes filled, connected components smaller than 1% of the mask dropped. No GT anywhere.
Seed: numpy default_rng(crc32("<task>/<dataset>/<item>/<case>")), one perturbed image per case, shared by every model.
Output: float32 NIfTI, same file name and geometry as the input (geometry re-read and checked).
QC row per case -> --stats TSV (scratch only: case ids), columns:
  body_frac, scr_corr = corr(band, scr) in the body (structure destroyed: ~0), le_corr = corr of the log local-energy
  maps G_2(band^2) and G_2(scr^2) in the body (local energy restored: high), fine_corr  = corr(band(x), band(x')) in the body (band(.) = . - G_s(.)), fine_ratio = rms band(x') / rms band(x),
  lp4_frac  = rms(G_4(x') - G_4(x)) / rms(x' - x) in the body  (share of the change that survives a sigma-4 low-pass),
  lp8_frac  = same with sigma 8, lp4_rel = rms(G_4(x') - G_4(x)) / std(x) in the body.
--mode control_all scrambles x - mean (every scale) instead: the QC positive control (must FAIL the low-pass gate).

  python perturb.py --in-dir <imagesTs_x> --out-dir <dir> --salt <task/dataset/item> --stats <tsv> [--workers 6]
                    [--mode T|control_all] [--cases a,b] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

SIGMA = 2.0                      # band = x - G_2(x)   (voxels)
LE_SIGMA, LE_EPS, LE_ITERS = 2.0, 1e-3, 2
BODY_T, BODY_CLOSE, BODY_MINFRAC = 0.05, 2, 0.01


def body_mask(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [0.5, 99.5])
    v = np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)
    m = ndi.binary_closing(v > BODY_T, iterations=BODY_CLOSE)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        keep = np.flatnonzero(sz >= BODY_MINFRAC * sz.sum())
        m = np.isin(lab, keep)
    return m


def phase_randomise(band, rng):
    """Real array with exactly the amplitude spectrum of `band` and random Hermitian-consistent phases (DC kept)."""
    ax = tuple(range(band.ndim))
    F = np.fft.rfftn(band.astype(np.float64), axes=ax)
    N = np.fft.rfftn(rng.standard_normal(band.shape), axes=ax)
    S = np.abs(F) * N / np.maximum(np.abs(N), 1e-12)
    S[(0,) * band.ndim] = F[(0,) * band.ndim]
    return np.fft.irfftn(S, s=band.shape, axes=ax)


def scramble(band, body, rng, local=True):
    s = phase_randomise(band, rng)
    if local:
        b2 = band.astype(np.float64) ** 2
        eps = LE_EPS * float(b2[body].mean())
        G_b = ndi.gaussian_filter(b2, LE_SIGMA)
        for _ in range(LE_ITERS):
            s = s * np.sqrt((G_b + eps) / (ndi.gaussian_filter(s ** 2, LE_SIGMA) + eps))
    s = s * body
    e0 = float((band[body].astype(np.float64) ** 2).sum())
    e1 = float((s[body] ** 2).sum())
    return (s * np.sqrt(e0 / e1) if e1 > 0 else s).astype(np.float32)


def perturb(x: np.ndarray, rng, mode="T"):
    body = body_mask(x)
    if mode == "T":
        band = x - ndi.gaussian_filter(x, SIGMA)
    elif mode == "control_all":
        band = x - float(x[body].mean())
    else:
        raise ValueError(mode)
    scr = scramble(band, body, rng, local=True)
    xp = x - band * body + scr
    return xp.astype(np.float32), body, band, scr


def _rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if a.size else float("nan")


def qc_stats(x, xp, body, band, scr):
    bb, ss = band[body].astype(np.float64), scr[body].astype(np.float64)
    e_b = ndi.gaussian_filter(band.astype(np.float64) ** 2, LE_SIGMA)[body]
    e_s = ndi.gaussian_filter(scr.astype(np.float64) ** 2, LE_SIGMA)[body]
    b0 = (x - ndi.gaussian_filter(x, SIGMA))[body]
    b1 = (xp - ndi.gaussian_filter(xp, SIGMA))[body]
    d = (xp - x)[body]
    out = dict(body_frac=float(body.mean()),
               scr_corr=float(np.corrcoef(bb, ss)[0, 1]) if bb.std() > 0 and ss.std() > 0 else float("nan"),
               le_corr=float(np.corrcoef(np.log(e_b + 1e-12), np.log(e_s + 1e-12))[0, 1]),
               fine_corr=float(np.corrcoef(b0, b1)[0, 1]) if b0.std() > 0 and b1.std() > 0 else float("nan"),
               fine_ratio=_rms(b1) / max(_rms(b0), 1e-12))
    for s in (4, 8):
        dl = (ndi.gaussian_filter(xp, s) - ndi.gaussian_filter(x, s))[body]
        out[f"lp{s}_frac"] = _rms(dl) / max(_rms(d), 1e-12)
        if s == 4:
            out["lp4_rel"] = _rms(dl) / max(float(x[body].std()), 1e-12)
    return out


def case_id(name: str) -> str:
    return name[: -len("_0000.nii.gz")] if name.endswith("_0000.nii.gz") else name[: -len(".nii.gz")]


def one(args):
    src, dst, salt, mode, overwrite = args
    cid = case_id(src.name)
    if dst.exists() and not overwrite:
        return None
    img = sitk.ReadImage(str(src))
    x = sitk.GetArrayFromImage(img).astype(np.float32)
    rng = np.random.default_rng(zlib.crc32(f"{salt}/{cid}".encode()))
    xp, body, band, scr = perturb(x, rng, mode)
    out = sitk.GetImageFromArray(xp)
    out.CopyInformation(img)
    tmp = dst.with_name(dst.name + ".tmp.nii.gz")
    sitk.WriteImage(out, str(tmp), useCompression=True)
    chk = sitk.ReadImage(str(tmp))
    geom_ok = (chk.GetSize() == img.GetSize() and np.allclose(chk.GetSpacing(), img.GetSpacing(), rtol=1e-5, atol=1e-5)
               and np.allclose(chk.GetOrigin(), img.GetOrigin(), rtol=1e-5, atol=1e-3)
               and np.allclose(chk.GetDirection(), img.GetDirection(), atol=1e-5))   # float32 header precision
    if not geom_ok:
        tmp.unlink()
        raise RuntimeError(f"geometry changed for {src}")
    tmp.rename(dst)
    st = qc_stats(x, xp, body, band, scr)
    return dict(case=cid, mode=mode, shape="x".join(map(str, x.shape)), **{k: f"{v:.5g}" for k, v in st.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--mode", default="T", choices=("T", "control_all"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(a.in_dir.glob("*_0000.nii.gz"))
    if a.cases:
        want = set(a.cases.split(","))
        srcs = [s for s in srcs if case_id(s.name) in want]
    if not srcs:
        sys.exit(f"ERROR: no *_0000.nii.gz in {a.in_dir}")
    jobs = [(s, a.out_dir / s.name, a.salt, a.mode, a.overwrite) for s in srcs]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs):
            if r:
                rows.append(r)
                print(f"[perturb] {r['case']} " + " ".join(f"{k}={r[k]}" for k in ("scr_corr", "le_corr", "fine_corr", "fine_ratio", "lp4_frac", "lp4_rel")), flush=True)
    if rows:
        a.stats.parent.mkdir(parents=True, exist_ok=True)
        new = not a.stats.exists()
        with a.stats.open("a") as f:
            if new:
                f.write("\t".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write("\t".join(str(v) for v in r.values()) + "\n")
    n = len(list(a.out_dir.glob("*_0000.nii.gz")))
    print(f"[perturb] {a.in_dir} -> {a.out_dir}: {len(rows)} written now, {n}/{len(srcs)} present")
    if n < len(srcs) and not a.cases:
        sys.exit(1)


if __name__ == "__main__":
    main()
