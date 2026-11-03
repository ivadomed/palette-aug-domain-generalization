#!/usr/bin/env python3
"""Texture-removed copies of nnU-Net test inputs ("texture removed at test time" experiment, 2026-11-02).

Each image is partitioned WITHOUT labels exactly like step A of PA's transform minus the label regions (1-D k-means
intensity regions, then a Voronoi sub-partition of every region), and every sub-region is replaced by its OWN mean
intensity: PA's per-region remap y = mu_R + alpha_R * (x - mean_R) with alpha_R = 0 and mu_R = mean_R. Contrast is
unchanged, no blur, no label remap. Voxels below PA's dark threshold (background) are left untouched.

Partition source (READ ONLY, nothing imported at run time so the generator does not depend on an AugLab checkout):
sub-workspaces/auglab_workspace/AugLab @ f018536, auglab/transforms/gpu/fromSeg.py
  - _kmeans_1d                       -> kmeans_1d below (same init, iterations, empty-cluster rule, stopping rule)
  - RandomV26_6_2ContrastGPU.apply_transform step 1 (min-max to [0, 1], fg = > dark_threshold 0.01, 40k random voxels
    -> first 10k fg values for k-means, midpoint thresholds + bucketize)   -> partition() below
  - _voronoi_region_ids              -> voronoi_cells() below (seeds drawn uniformly without replacement among the
    cluster's fg voxels, nearest-seed assignment)
Fixed parameters instead of PA's random draws (documented choice): k = 4 clusters (median of c_choices [2..6]),
S = 6 Voronoi cells per cluster (median of s_choices [2..10]), skip_parcellation_prob = skip_sub_parc_prob = 0 (always
the full partition). Deterministic: one numpy Generator per case, seeded from crc32("<salt>|<case id>").

Two deliberate adaptations to whole test images (PA runs on training patches after nnU-Net preprocessing):
  * intensities used for the PARTITION follow the predicting model's normalization: CTNormalization clips to the plans'
    foreground percentiles [0.5, 99.5] first (what PA saw in training), ZScore is affine so nothing to do. Region
    MEANS are computed on the original, unclipped intensities.
  * Voronoi distances in millimetres (voxel index x spacing): PA's patches are on nnU-Net's resampled grid, so
    index distances there are approximately physical distances.

Writes float32 NIfTI with the input's origin / spacing / direction (SimpleITK, nnU-Net's reader), same file name.
Per-case QC row (geometry re-read, region count, region-mean preservation) -> --stats TSV (scratch only: case ids).

  python texture_removal.py --in-dir <imagesTs_x> --out-dir <dir> [--ct-window LO HI] [--k 4] [--cells 6]
                            [--salt task/item] [--workers 4] --stats <file.tsv>
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DARK_THRESHOLD = 0.01        # fromSeg.py RandomV26_6_2ContrastGPU default / every PA config
N_SAMPLE = 40_000            # random voxels drawn before the fg filter (fromSeg.py: min(N, 40_000))
N_KMEANS_SUBSAMPLE = 10_000  # n_kmeans_subsample
EPS = 1e-7


def kmeans_1d(values: np.ndarray, C: int, n_iter: int = 10) -> np.ndarray:
    """numpy port of fromSeg._kmeans_1d: linspace init over [min, max], argmin assignment, an empty cluster keeps its
    centroid, stop when centroids are allclose (torch.allclose defaults rtol 1e-5, atol 1e-8)."""
    values = values.astype(np.float32)
    centroids = np.linspace(values.min(), values.max(), C, dtype=np.float32)
    for _ in range(n_iter):
        lbl = np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)
        s = np.bincount(lbl, weights=values, minlength=C).astype(np.float32)
        n = np.bincount(lbl, minlength=C).astype(np.float32)
        new_c = np.where(n > 0, s / np.maximum(n, 1), centroids).astype(np.float32)
        if np.allclose(centroids, new_c, rtol=1e-5, atol=1e-8):
            break
        centroids = new_c
    return centroids


def partition(v: np.ndarray, fg: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster index per voxel (flat arrays), PA step-1 k-means on min-max intensities v in [0, 1]."""
    N = v.size
    idx = rng.integers(0, N, size=min(N, N_SAMPLE))
    samp = v[idx]
    sub_fg = samp[samp > DARK_THRESHOLD][:N_KMEANS_SUBSAMPLE]
    if sub_fg.size < 4:
        sub_fg = samp[:N_KMEANS_SUBSAMPLE]
    centroids = kmeans_1d(sub_fg, k)
    sorted_c = np.sort(centroids)
    boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
    # torch.bucketize(right=False) == np.searchsorted(side="left"); the cluster id relabelling by sort_idx in fromSeg
    # is a permutation of ids and changes no region, so the sorted index is used directly.
    return np.searchsorted(boundaries, v, side="left")


def voronoi_cells(coords_mm: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
    """Nearest-seed cell id for each row of coords_mm (one cluster's fg voxels); seeds = S of those voxels drawn
    uniformly without replacement (fromSeg._voronoi_region_ids: torch.multinomial over the cluster's fg mask)."""
    n = coords_mm.shape[0]
    S = min(S, n)
    if n < 2 or S <= 1:
        return np.zeros(n, dtype=np.int64)
    seeds = coords_mm[rng.choice(n, size=S, replace=False)]
    best = np.full(n, np.inf, dtype=np.float32)
    cell = np.zeros(n, dtype=np.int64)
    for j in range(S):
        d = ((coords_mm - seeds[j]) ** 2).sum(axis=1)
        better = d < best
        best[better] = d[better]
        cell[better] = j
    return cell


def case_id_of(name: str) -> str:
    for suf in ("_0000.nii.gz", ".nii.gz"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def remove_texture(img: sitk.Image, case_id: str, salt: str, k: int, cells: int, ct_window=None):
    """-> (output image, stats dict)."""
    x = sitk.GetArrayFromImage(img).astype(np.float32)          # (z, y, x)
    shape = x.shape
    sp = np.array(img.GetSpacing()[::-1], dtype=np.float32)     # array-axis order
    flat = x.reshape(-1)
    xp = np.clip(flat, ct_window[0], ct_window[1]) if ct_window is not None else flat
    lo, hi = float(xp.min()), float(xp.max())
    v = np.clip((xp - lo) / (hi - lo + EPS), 0.0, 1.0)
    fg = v > DARK_THRESHOLD
    rng = np.random.default_rng(zlib.crc32(f"{salt}|{case_id}".encode()))
    out = flat.copy()
    n_regions = 0
    max_mean_err = 0.0
    if fg.sum() >= 4:
        lbl = partition(v, fg, k, rng)
        for c in range(k):
            ids = np.flatnonzero(fg & (lbl == c))
            if ids.size == 0:
                continue
            zyx = np.stack(np.unravel_index(ids, shape), axis=1).astype(np.float32) * sp
            cell = voronoi_cells(zyx, cells, rng)
            ncell = int(cell.max()) + 1
            sums = np.bincount(cell, weights=flat[ids].astype(np.float64), minlength=ncell)
            cnts = np.bincount(cell, minlength=ncell)
            means = sums / np.maximum(cnts, 1)
            out[ids] = means[cell].astype(np.float32)
            # region-mean preservation, measured on what is written
            got = np.bincount(cell, weights=out[ids].astype(np.float64), minlength=ncell) / np.maximum(cnts, 1)
            max_mean_err = max(max_mean_err, float(np.abs(got - means)[cnts > 0].max()))
            n_regions += int((cnts > 0).sum())
    o = sitk.GetImageFromArray(out.reshape(shape))
    o.CopyInformation(img)
    stats = dict(case=case_id, n_vox=int(flat.size), n_fg=int(fg.sum()), n_regions=n_regions,
                 frac_changed=float(np.mean(out != flat)), max_region_mean_err=max_mean_err,
                 in_std_fg=float(flat[fg].std()) if fg.any() else 0.0,
                 out_std_fg=float(out[fg].std()) if fg.any() else 0.0)
    return o, stats


def _geom(img):
    return (img.GetSize(), np.round(img.GetSpacing(), 6).tolist(), np.round(img.GetOrigin(), 4).tolist(),
            np.round(img.GetDirection(), 6).tolist())


def process_one(args):
    src, dst, salt, k, cells, ct_window, overwrite = args
    if dst.exists() and not overwrite:
        return None
    img = sitk.ReadImage(str(src))
    if img.GetNumberOfComponentsPerPixel() != 1:
        raise SystemExit(f"multi-component image not supported: {src}")
    out, st = remove_texture(img, case_id_of(src.name), salt, k, cells, ct_window)
    tmp = dst.with_name(dst.name + ".tmp.nii.gz")
    sitk.WriteImage(out, str(tmp), useCompression=True)
    back = sitk.ReadImage(str(tmp))
    st["geometry_ok"] = int(_geom(back) == _geom(img))
    if not st["geometry_ok"]:
        raise SystemExit(f"geometry changed on write: {src}")
    os.replace(tmp, dst)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--ct-window", nargs=2, type=float, default=None, metavar=("LO", "HI"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cells", type=int, default=6)
    ap.add_argument("--salt", default="texrm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (QC)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    files = sorted(p for p in a.in_dir.iterdir() if p.name.endswith(".nii.gz"))
    if any(not p.name.endswith("_0000.nii.gz") for p in files):
        sys.exit(f"unexpected non-_0000 file in {a.in_dir} (single-channel inputs expected)")
    if a.limit:
        files = files[: a.limit]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, a.out_dir / p.name, a.salt, a.k, a.cells, a.ct_window, a.overwrite) for p in files]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for st in ex.map(process_one, jobs):
            if st is not None:
                rows.append(st)
    a.stats.parent.mkdir(parents=True, exist_ok=True)
    new = not a.stats.exists()
    with a.stats.open("a") as f:
        if new:
            f.write("\t".join(["in_dir"] + list(rows[0].keys() if rows else [])) + "\n")
        for r in rows:
            f.write("\t".join([str(a.in_dir)] + [str(r[c]) for c in r]) + "\n")
    n_out = len([p for p in a.out_dir.iterdir() if p.name.endswith("_0000.nii.gz")])
    print(f"[texrm] {a.in_dir} -> {a.out_dir}: {len(rows)} written, {n_out}/{len(files)} present; "
          f"k={a.k} cells={a.cells} ct_window={a.ct_window}")
    if rows:
        err = max(r["max_region_mean_err"] for r in rows)
        print(f"[texrm] regions/case min {min(r['n_regions'] for r in rows)} max {max(r['n_regions'] for r in rows)}; "
              f"max region-mean error {err:.3g}; geometry_ok {sum(r['geometry_ok'] for r in rows)}/{len(rows)}")
    if n_out != len(files):
        sys.exit("ERROR: missing outputs")


if __name__ == "__main__":
    main()
