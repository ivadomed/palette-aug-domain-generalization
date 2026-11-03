#!/usr/bin/env python3
"""QC of the texture-removed inputs: summary of the generator's per-case rows (geometry re-read identical, regions per
case, region means preserved, fraction of voxels changed) for every (task, item), and a small PNG of a few cases of
PUBLIC datasets only (ON-Harmony, Open-MS, CHAOS; never LLD-MMRI / restricted cohorts) written to --png-dir (scratch,
never git). Prints aggregates only (no case ids).

  python qc_texture_removal.py --stats-dir <gen_stats> --png-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PUBLIC = [("Brain", "on-harmony__T2w"), ("MS", "open-ms__t1w"), ("Abdomen", "chaos__ct")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", type=Path, required=True)
    ap.add_argument("--png-dir", type=Path, required=True)
    ap.add_argument("--gen-jobs", type=Path, required=True)
    a = ap.parse_args()
    jobs = {(r["task"], r["key"]): r for r in csv.DictReader(a.gen_jobs.open(), delimiter="\t")}
    bad = 0
    print(f"{'task/item':45s} {'n':>4s} {'geom':>5s} {'regions min-max':>15s} {'mean err':>9s} {'changed':>8s} {'std out/in':>10s}")
    for (task, key), j in jobs.items():
        f = a.stats_dir / f"{task}__{key}.tsv"
        rows = list(csv.DictReader(f.open(), delimiter="\t")) if f.exists() else []
        n_in = len(list(Path(j["input"]).glob("*_0000.nii.gz")))
        n_out = len(list(Path(j["output"]).glob("*_0000.nii.gz")))
        if not rows:
            print(f"{task + '/' + key:45s} no stats ({n_out}/{n_in} outputs)")
            bad += 1
            continue
        g = sum(int(r["geometry_ok"]) for r in rows)
        reg = [int(r["n_regions"]) for r in rows]
        err = max(float(r["max_region_mean_err"]) for r in rows)
        ch = np.median([float(r["frac_changed"]) for r in rows])
        sr = np.median([float(r["out_std_fg"]) / max(float(r["in_std_fg"]), 1e-9) for r in rows])
        ok = g == len(rows) and n_out == n_in and len(rows) >= n_in
        bad += not ok
        print(f"{task + '/' + key:45s} {n_out:4d} {g:3d}/{len(rows):<3d} {min(reg):7d}-{max(reg):<7d} {err:9.2g} "
              f"{ch:8.2f} {sr:10.2f}{'' if ok else '  <-- CHECK'}")
    a.png_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(PUBLIC), 2, figsize=(7, 3.4 * len(PUBLIC)))
    for i, (task, key) in enumerate(PUBLIC):
        j = jobs[(task, key)]
        src = sorted(Path(j["input"]).glob("*_0000.nii.gz"))[0]
        x = sitk.GetArrayFromImage(sitk.ReadImage(str(src))).astype(np.float32)
        y = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(j["output"]) / src.name))).astype(np.float32)
        ax_ = int(np.argmax(x.shape[::-1][:3][::-1]))       # slice through the thickest-sampled axis' middle
        z = x.shape[0] // 2
        lo, hi = np.percentile(x, [1, 99])
        for c, (im, t) in enumerate(((x, "original"), (y, "texture removed"))):
            ax[i, c].imshow(im[z], cmap="gray", vmin=lo, vmax=hi)
            ax[i, c].set_title(f"{task} {key.split('__')[1]}: {t}", fontsize=9)
            ax[i, c].axis("off")
    fig.tight_layout()
    out = a.png_dir / "texture_removal_qc.png"
    fig.savefig(out, dpi=110)
    print(f"QC png: {out}")
    print("ALL OK" if bad == 0 else f"{bad} item(s) to check")


if __name__ == "__main__":
    main()
