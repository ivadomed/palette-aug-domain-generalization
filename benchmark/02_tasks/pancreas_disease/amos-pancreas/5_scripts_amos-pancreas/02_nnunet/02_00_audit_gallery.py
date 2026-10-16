#!/usr/bin/env python3
"""Audit + visual gallery of the 60 labeled AMOS-MRI cases (BIDS tree of abdomen_healthy/amos, read-only) for a PANCREAS eval companion.
Per case: shape/spacing/axcodes, pancreas (AMOS label 10) volume, S-I border touch, organ-intensity ratios (spleen/liver, aorta/muscle-free proxies) for a
crude T1-vs-T2 call, and a PNG with an axial + coronal slice through the pancreas centroid (look at it: BIDS suffix says T2w for ALL cases, which cannot be true).
Usage: 02_00_audit_gallery.py <amos_bids_leaf> <out_dir>   (via run_job)"""
import sys, csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

leaf, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "png").mkdir(parents=True, exist_ok=True)
subs = [l.split("\t") for l in (leaf / "participants.tsv").read_text().splitlines()[1:]]
subs = [s for s in subs if s[1] == "MRI"]
rows = []
def win(x): lo, hi = np.percentile(x, [1, 99.5]); return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
for sub, mod, aid, split in subs:
    img = nib.load(leaf / sub / "anat" / f"{sub}_T2w.nii.gz")
    lab = nib.load(leaf / "derivatives/labels" / sub / "anat" / f"{sub}_T2w_label-organs_dseg.nii.gz")
    x = np.asanyarray(img.dataobj).astype(np.float32); l = np.asanyarray(lab.dataobj).astype(np.int16)
    sp = np.array(img.header.get_zooms()[:3]); p = l == 10
    nv = int(p.sum()); ncc = int(ndimage.label(p)[1]) if nv else 0
    zs = np.where(p.any((0, 1)))[0]
    si_touch = int(nv > 0 and (zs.min() == 0 or zs.max() == x.shape[2] - 1))
    med = lambda k: float(np.median(x[l == k])) if (l == k).any() else float("nan")
    liv, spl, kid, aor, pan = med(6), med(1), (med(2) + med(3)) / 2, med(8), med(10)
    rows.append(dict(case=sub, amos_id=aid, split=split, shape="x".join(map(str, x.shape)), spacing="x".join(f"{s:.2f}" for s in sp),
                     axcodes="".join(nib.aff2axcodes(img.affine)), pancreas_ml=round(nv * float(np.prod(sp)) / 1000, 1), n_cc=ncc, si_touch=si_touch,
                     z_extent_mm=round(float(x.shape[2] * sp[2])), spleen_over_liver=round(spl / liv, 2), pancreas_over_liver=round(pan / liv, 2),
                     aorta_over_liver=round(aor / liv, 2), kidney_over_liver=round(kid / liv, 2)))
    if nv:
        c = [int(round(v)) for v in ndimage.center_of_mass(p)]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(win(x[:, :, c[2]]).T, cmap="gray", origin="lower"); ax[0].contour(p[:, :, c[2]].T, [0.5], colors="r", linewidths=.6); ax[0].set_title(f"{sub} axial")
        ax[1].imshow(win(x[:, c[1], :]).T, cmap="gray", origin="lower", aspect=sp[2] / sp[0]); ax[1].contour(p[:, c[1], :].T, [0.5], colors="r", linewidths=.6); ax[1].set_title("coronal")
        ax[2].imshow(win(x[c[0], :, :]).T, cmap="gray", origin="lower", aspect=sp[2] / sp[1]); ax[2].contour(p[c[0], :, :].T, [0.5], colors="r", linewidths=.6); ax[2].set_title("sagittal")
        for a in ax: a.axis("off")
        fig.savefig(out / "png" / f"{sub}.png", dpi=60, bbox_inches="tight"); plt.close(fig)
with open(out / "audit.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(len(rows), "MRI cases;", sum(r["pancreas_ml"] > 0 for r in rows), "with pancreas")
