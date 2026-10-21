#!/usr/bin/env python3
"""Look at one BIDS case: image + its pancreas label overlaid (3 orthogonal planes through the label centroid), plus numbers
(axcodes, shape, label bbox/centroid in voxel and mm, label volume, fraction of the label inside the body, label-vs-image bounding-box fit).
Usage: 02_12_inspect_case.py <out_dir> <sub-id> [<sub-id> ...]    (T1 venous + T2w of each subject, each with its own label)"""
import sys, json
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = Path(__file__).resolve().parents[2] / "1_BIDS_pansegdata/onc-pancreas-pansegdata"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
def win(a): lo, hi = np.percentile(a, [1, 99]); return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
for sub in sys.argv[2:]:
    for tag, img_n, lab_n in (("T1v", f"{sub}_acq-venous_T1w", f"{sub}_acq-venous_label-pancreas_seg"), ("T2", f"{sub}_T2w", f"{sub}_label-pancreas_seg")):
        ip, lp = B / sub / "anat" / f"{img_n}.nii.gz", B / "derivatives/labels" / sub / "anat" / f"{lab_n}.nii.gz"
        if not ip.exists() or not lp.exists(): print(sub, tag, "missing", ip.exists(), lp.exists()); continue
        im, lb = nib.load(ip), nib.load(lp); a, m = np.asarray(im.dataobj).astype(float), np.asarray(lb.dataobj) > 0
        same = im.shape == lb.shape and np.allclose(im.affine, lb.affine, atol=1e-3)
        idx = np.argwhere(m)
        if not len(idx): print(sub, tag, "EMPTY LABEL"); continue
        c = idx.mean(0); lo, hi = idx.min(0), idx.max(0)
        body = a > np.percentile(a, 60) * 0.5            # crude body mask: well above background
        vox = np.abs(np.linalg.det(im.affine[:3, :3]))
        print(f"{sub} {tag}: img axcodes={''.join(nib.aff2axcodes(im.affine))} shape={a.shape} zooms={np.round(im.header.get_zooms()[:3],2)} | same grid as label={same} (lab axcodes={''.join(nib.aff2axcodes(lb.affine))}, shape={m.shape})")
        print(f"    label: vol={m.sum()*vox/1000:.1f} ml  bbox_vox={lo.tolist()}..{hi.tolist()}  centroid_vox={np.round(c,1).tolist()}  centroid/shape={np.round(c/np.array(m.shape),2).tolist()}  label mean intensity / image median={a[m].mean()/np.median(a):.2f}  frac label voxels in 'body'={body[m].mean():.2f}")
        ci = np.round(c).astype(int); fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        for k, (sl_a, sl_m, nm) in enumerate(((a[:, :, ci[2]], m[:, :, ci[2]], "axial"), (a[:, ci[1], :], m[:, ci[1], :], "coronal"), (a[ci[0], :, :], m[ci[0], :, :], "sagittal"))):
            ax[k].imshow(win(sl_a).T, cmap="gray", origin="lower"); ax[k].contour(sl_m.T, levels=[0.5], colors="r", linewidths=1.2); ax[k].set_title(f"{sub} {tag} {nm}"); ax[k].axis("off")
        fig.tight_layout(); fig.savefig(out / f"{sub}_{tag}.png", dpi=80); plt.close(fig)
