#!/usr/bin/env python3
"""Contact sheets (axial slice through the pancreas centroid + contour, 5x4 per sheet) of the AMOS-MRI cases, for classifying contrast by eye.
Usage: 02_00b_contact_sheets.py <amos_bids_leaf> <audit.tsv> <out_dir>   (via run_job)"""
import sys, csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
leaf, tsv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(tsv), delimiter="\t"))
def win(x): lo, hi = np.percentile(x, [1, 99.5]); return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
for si in range(0, len(rows), 20):
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    for a, r in zip(axs.ravel(), rows[si:si + 20]):
        s = r["case"]
        x = np.asanyarray(nib.load(leaf / s / "anat" / f"{s}_T2w.nii.gz").dataobj).astype(np.float32)
        p = np.asanyarray(nib.load(leaf / "derivatives/labels" / s / "anat" / f"{s}_T2w_label-organs_dseg.nii.gz").dataobj) == 10
        sh = x.shape; thick = int(np.argmax([float(v) for v in r["spacing"].split("x")]))  # slice axis = coarsest spacing
        c = [int(round(v)) for v in ndimage.center_of_mass(p)]
        sl = [slice(None)] * 3; sl[thick] = c[thick]
        a.imshow(win(x[tuple(sl)]).T, cmap="gray", origin="lower"); a.contour(p[tuple(sl)].T, [0.5], colors="r", linewidths=.7)
        a.set_title(f"{s[-4:]} thick-axis={'xyz'[thick]} {r['spacing']}", fontsize=9); a.axis("off")
    for a in axs.ravel()[len(rows[si:si + 20]):]: a.axis("off")
    fig.savefig(out / f"sheet_{si // 20}.png", dpi=55, bbox_inches="tight"); plt.close(fig)
