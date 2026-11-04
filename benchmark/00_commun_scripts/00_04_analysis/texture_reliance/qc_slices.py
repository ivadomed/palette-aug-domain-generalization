#!/usr/bin/env python3
"""Texture-reliance experiment: QC slice images of perturbation T, ONLY from openly licensed cohorts
(build_manifest.OPEN_IMAGE_COHORTS: open-ms, chaos, ispy2, totalseg-pelvic, on-harmony; never LLD-MMRI, ISLES, HaN-Seg,
or any other restricted / DUA / NC-ND cohort). Per open input dir, 2 cases: the axial slice through the body centroid,
panels original | perturbed T | fine band of the original (x - G_2 x) | fine band of T | T - original | low-pass G_4
difference (same colour scale as T - original), plus the positive control (all-scale scramble) when present.
Output: <work>/qc_png/*.png (scratch only, never git).

  .venv/bin/python qc_slices.py [--work ...]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from perturb import SIGMA, body_mask  # noqa: E402


def arr(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/scratch/paulh/texture_reliance"))
    a = ap.parse_args()
    od = a.work / "qc_png"
    od.mkdir(parents=True, exist_ok=True)
    with (a.work / "gen_jobs.tsv").open() as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["open_license"] == "1"]
    for r in rows:
        src, out = Path(r["input"]), Path(r["output"])
        for p in sorted(src.glob("*_0000.nii.gz"))[:2]:
            x, xp = arr(p), arr(out / p.name)
            ctrl = a.work / "control_all" / out.name / p.name
            xc = arr(ctrl) if ctrl.exists() else None
            body = body_mask(x)
            z = int(round(ndi.center_of_mass(body)[0])) if body.any() else x.shape[0] // 2
            b0, b1 = x - ndi.gaussian_filter(x, SIGMA), xp - ndi.gaussian_filter(xp, SIGMA)
            d = xp - x
            dl = ndi.gaussian_filter(xp, 4) - ndi.gaussian_filter(x, 4)
            lo, hi = np.percentile(x[body], [1, 99]) if body.any() else (x.min(), x.max())
            bl = np.percentile(np.abs(b0[body]), 99) if body.any() else 1
            panels = [("original", x, "gray", lo, hi), ("perturbed T", xp, "gray", lo, hi),
                      ("fine band, original", b0, "gray", -bl, bl), ("fine band, T", b1, "gray", -bl, bl),
                      ("T - original", d, "coolwarm", -bl, bl), ("G4(T) - G4(original)", dl, "coolwarm", -bl, bl)]
            if xc is not None:
                panels.append(("control: all scales", xc, "gray", lo, hi))
            fig, axs = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.9))
            for ax, (t, im, cm, vmin, vmax) in zip(axs, panels):
                ax.imshow(im[z], cmap=cm, vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(t, fontsize=7)
                ax.axis("off")
            fig.suptitle(f"{r['task']} {r['dataset']}/{r['item']}", fontsize=8)
            fig.tight_layout()
            fn = od / f"{r['task']}__{r['dataset']}__{r['item']}__{p.name.replace('_0000.nii.gz', '')}.png"
            fig.savefig(fn, dpi=110)
            plt.close(fig)
            print(f"[qc_slices] {fn}")


if __name__ == "__main__":
    main()
