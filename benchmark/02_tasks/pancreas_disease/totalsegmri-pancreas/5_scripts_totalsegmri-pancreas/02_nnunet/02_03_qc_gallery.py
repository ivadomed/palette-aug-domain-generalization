#!/usr/bin/env python3
"""QC gallery: axial slice through the mask centroid + coronal slice, mask contour overlaid, for converted test cases and a few source TRAINING cases (aff2axcodes is blind to 180-degree flips: look).
Usage: 02_03_qc_gallery.py <raw_dir_of_this_companion> <source_raw_Dataset150_dir> <out_dir>   (writes PNGs; delete old ones first)"""
import sys
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

raw, src, out = map(Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.png"): f.unlink()
def panel(ax, im, m, plane):
    c = [int(round(v)) for v in np.argwhere(m).mean(0)]
    sl = (lambda a: a[:, :, c[2]]) if plane == "ax" else (lambda a: a[:, c[1], :])
    i, mm = sl(im), sl(m)
    ax.imshow(np.rot90(i), cmap="gray", vmin=np.percentile(i, 1), vmax=np.percentile(i, 99.5), aspect="auto" if plane == "ax" else "auto")
    ax.contour(np.rot90(mm), levels=[0.5], colors="r", linewidths=0.7); ax.axis("off")
def sheet(cases, name):
    n = len(cases); fig, axs = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    axs = np.atleast_2d(axs).reshape(2, -1)
    for j, (img, lab, title) in enumerate(cases):
        I = nib.load(img); I = nib.as_closest_canonical(I) if False else I
        a = np.asanyarray(I.dataobj).astype(np.float32); m = np.asanyarray(nib.load(lab).dataobj) > 0
        panel(axs[0, j], a, m, "ax"); panel(axs[1, j], a, m, "cor"); axs[0, j].set_title(title, fontsize=6)
    fig.tight_layout(); fig.savefig(out / name, dpi=110); plt.close(fig)
for it in ("t1gre", "t2like"):
    fs = sorted((raw / f"imagesTs_{it}").glob("*_0000.nii.gz"))
    for k in range(0, len(fs), 8):
        sheet([(f, raw / f"labelsTs_{it}" / f.name.replace("_0000", ""), f.name.split("_0000")[0].split("_")[-1]) for f in fs[k:k + 8]], f"test_{it}_{k//8}.png")
fs = sorted((src / "imagesTr").glob("*_0000.nii.gz"))[::28][:8]
sheet([(f, src / "labelsTr" / f.name.replace("_0000", ""), "SRC " + f.name.split("_0000")[0][-8:]) for f in fs], "source_train.png")
print("pngs", len(list(out.glob("*.png"))))
