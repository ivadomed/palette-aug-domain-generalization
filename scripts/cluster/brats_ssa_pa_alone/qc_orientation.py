#!/usr/bin/env python3
"""
Orientation check of the BraTS-SSA 2024 test inputs against brats2024-glioma's own test set
(run inside a CPU job, never on a login node).

Three tiers (axis codes alone cannot see a 180-degree flip, see the onboarding notes):
  1. headers, every file: shape, spacing, aff2axcodes, image/label affine equality, label values;
  2. sign-sensitive tests on the voxel arrays as nnU-Net reads them:
       a. cohort-mean brain mask (t1n > 0) and cohort-mean z-scored t1n intensity of SSA, compared with
          the glioma test-set means under all 8 axis-sign flips; the identity must win clearly
          (A-P and S-I asymmetries of the brain decide 6 of the 8 hypotheses);
       b. label inside brain: fraction of each case's tumour voxels with t1n > 0 (a label/image flip
          would push tumour voxels outside the head);
  3. a gallery of mid-slices through each case's tumour centroid (t1n, t1c, t2w, t2f + GT contour),
     SSA cases next to glioma cases with the identical display transform.

Usage: qc_orientation.py <ssa_raw_dir> <glioma_raw_dir> <glioma_test_cases.json> <out_dir>
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

ITEMS = ("t1n", "t1c", "t2w", "t2f")


def header_rows(img_paths, lab_path):
    lab = nib.load(str(lab_path))
    rows = []
    for p in img_paths:
        im = nib.load(str(p))
        rows.append(dict(shape=tuple(im.shape), zooms=tuple(round(float(z), 3) for z in im.header.get_zooms()[:3]),
                         ax="".join(nib.aff2axcodes(im.affine)),
                         same_affine_as_label=bool(np.allclose(im.affine, lab.affine, atol=1e-3))))
    return rows, lab


def zscore_brain(a):
    m = a > 0
    if m.sum() < 1000:
        return np.zeros_like(a, dtype=np.float32)
    v = a[m]
    out = np.zeros_like(a, dtype=np.float32)
    out[m] = (a[m] - v.mean()) / (v.std() + 1e-6)
    return out


def axis_profiles(mask):
    """Per axis: brain-mask area per slice, resampled over the brain's own extent to 100 bins
    (grid- and template-independent; A-P and S-I profiles of a brain are strongly asymmetric)."""
    out = []
    for ax in range(3):
        prof = mask.sum(axis=tuple(a for a in range(3) if a != ax)).astype(np.float64)
        nz = np.nonzero(prof)[0]
        prof = prof[nz[0]:nz[-1] + 1] if len(nz) else prof
        x = np.linspace(0, len(prof) - 1, 100)
        out.append(np.interp(x, np.arange(len(prof)), prof) / (prof.max() + 1e-9))
    return out


def flips():
    for s in itertools.product((False, True), repeat=3):
        yield s


def apply_flip(a, s):
    for ax, f in enumerate(s):
        if f:
            a = np.flip(a, axis=ax)
    return a


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def main():
    ssa, gli, tc_json, out = (Path(x) for x in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    rep = []
    # ---- SSA headers ----
    ssa_cases = sorted(p.name[:-len("_0000.nii.gz")] for p in (ssa / "imagesTs_t1n").glob("*_0000.nii.gz"))
    hdr = Counter(); labvals = Counter(); aff_mismatch = []; inside = {}; ssa_prof = []; g_prof = []
    mean_mask = mean_int = None
    for c in ssa_cases:
        imgs = [ssa / f"imagesTs_{it}" / f"{c}_0000.nii.gz" for it in ITEMS]
        rows, lab = header_rows(imgs, ssa / "labelsTs_t1n" / f"{c}.nii.gz")
        for it, r in zip(ITEMS, rows):
            hdr[("ssa", it, r["shape"], r["zooms"], r["ax"])] += 1
            if not r["same_affine_as_label"]:
                aff_mismatch.append(f"ssa {c} {it}")
        L = np.asarray(lab.dataobj)
        labvals[tuple(sorted(int(v) for v in np.unique(L)))] += 1
        hdr[("ssa", "label", tuple(lab.shape), tuple(round(float(z), 3) for z in lab.header.get_zooms()[:3]),
             "".join(nib.aff2axcodes(lab.affine)))] += 1
        # all four labelsTs_<item> must be the same mask
        for it in ITEMS[1:]:
            L2 = np.asarray(nib.load(str(ssa / f"labelsTs_{it}" / f"{c}.nii.gz")).dataobj)
            if not np.array_equal(L, L2):
                aff_mismatch.append(f"ssa {c} labelsTs_{it} differs from labelsTs_t1n")
        a = np.asarray(nib.load(str(imgs[0])).dataobj).astype(np.float32)
        inside[c] = float(((a > 0) & (L > 0)).sum() / max((L > 0).sum(), 1))
        mm, zi = (a > 0).astype(np.float32), zscore_brain(a)
        ssa_prof.append(axis_profiles(a > 0))
        mean_mask = mm if mean_mask is None else mean_mask + mm
        mean_int = zi if mean_int is None else mean_int + zi
    mean_mask /= len(ssa_cases); mean_int /= len(ssa_cases)
    # ---- glioma test set ----
    test = json.loads(Path(tc_json).read_text())
    d51 = gli / "Dataset051_BraTS2024GliomaT1n"
    g_mask = g_int = None; g_inside = {}; n_g = 0
    for c in test:
        imgs = [gli / f"Dataset05{i}_BraTS2024Glioma{n}" / f"imagesTs_{it}" / f"{c}_0000.nii.gz"
                for i, n, it in ((1, "T1n", "t1n"), (1, "T1n", "t1c"), (1, "T1n", "t2w"), (1, "T1n", "t2f"))]
        lab_p = d51 / "labelsTr" / f"{c}.nii.gz"
        if not lab_p.exists() or not imgs[0].exists():
            continue
        rows, lab = header_rows(imgs, lab_p)
        for it, r in zip(ITEMS, rows):
            hdr[("glioma", it, r["shape"], r["zooms"], r["ax"])] += 1
        L = np.asarray(lab.dataobj)
        a = np.asarray(nib.load(str(imgs[0])).dataobj).astype(np.float32)
        g_inside[c] = float(((a > 0) & (L > 0)).sum() / max((L > 0).sum(), 1))
        mm, zi = (a > 0).astype(np.float32), zscore_brain(a)
        g_prof.append(axis_profiles(a > 0))
        g_mask = mm if g_mask is None else g_mask + mm
        g_int = zi if g_int is None else g_int + zi
        n_g += 1
    g_mask /= n_g; g_int /= n_g

    rep += ["# BraTS-SSA 2024 orientation QC", "",
            f"SSA cases: {len(ssa_cases)}; glioma test cases read: {n_g}", "",
            "## 1. Headers (count per (cohort, item, shape, spacing, axcodes))", ""]
    rep += [f"- {k}: {v}" for k, v in sorted(hdr.items(), key=str)]
    rep += ["", f"SSA label value sets: {dict(labvals)}",
            f"SSA image/label affine or mask mismatches: {aff_mismatch if aff_mismatch else 'none'}", ""]
    # ---- sign test ----
    rep += ["## 2a. Sign test: SSA cohort mean vs glioma test-set mean under the 8 axis-sign flips", "",
            "| flip (ax0,ax1,ax2) | corr(brain-mask mean) | corr(z-scored t1n mean) |", "|---|---|---|"]
    res = []
    for s in flips():
        r1 = corr(apply_flip(mean_mask, s), g_mask) if mean_mask.shape == g_mask.shape else float("nan")
        r2 = corr(apply_flip(mean_int, s), g_int) if mean_int.shape == g_int.shape else float("nan")
        res.append((s, r1, r2))
        rep.append(f"| {tuple(int(x) for x in s)} | {r1:.4f} | {r2:.4f} |")
    best = max(res, key=lambda x: x[2])
    rep += ["", f"best flip by intensity correlation: {tuple(int(x) for x in best[0])} "
                f"(identity = (0, 0, 0)); identity r = {res[0][2]:.4f}, best non-identity r = "
                f"{max(r[2] for r in res[1:]):.4f}", ""]
    ok = best[0] == (False, False, False)
    rep += ["## 2a'. Grid-independent sign test: cohort-mean brain-extent profile per array axis", "",
            "corr(SSA profile, glioma profile) as stored vs with the SSA axis reversed. A-P (axis 1) and S-I",
            "(axis 2) profiles are asymmetric, so the higher of the two decides; L-R (axis 0) is near-symmetric.", "",
            "| array axis | corr as stored | corr SSA axis reversed | verdict |", "|---|---|---|---|"]
    prof_ok = True
    for ax in range(3):
        sp = np.mean([p[ax] for p in ssa_prof], 0); gp = np.mean([p[ax] for p in g_prof], 0)
        r_id, r_fl = corr(sp, gp), corr(sp[::-1], gp)
        v = "same direction" if r_id > r_fl else "REVERSED vs training arrays"
        if ax > 0 and r_fl > r_id:
            prof_ok = False
        rep.append(f"| {ax} | {r_id:.4f} | {r_fl:.4f} | {v} |")
    rep.append("")
    if mean_mask.shape != g_mask.shape:
        rep += [f"(2a mean-image test not applicable: SSA grid {mean_mask.shape} != glioma grid {g_mask.shape}; "
                "2a' decides)", ""]
        ok = prof_ok
    np.save(out / "profiles_ssa.npy", np.array([[p[a] for a in range(3)] for p in ssa_prof]).mean(0))
    np.save(out / "profiles_glioma.npy", np.array([[p[a] for a in range(3)] for p in g_prof]).mean(0))
    # ---- label inside brain ----
    v = np.array(list(inside.values())); gv = np.array(list(g_inside.values()))
    rep += ["## 2b. Tumour voxels inside the brain (t1n > 0)", "",
            f"SSA: median {np.median(v):.4f}, min {v.min():.4f} ({min(inside, key=inside.get)}); "
            f"glioma test: median {np.median(gv):.4f}, min {gv.min():.4f}", ""]
    rep += [f"VERDICT (array-level): {'identity orientation, no flip' if ok else 'FLIP SUSPECTED'}", ""]
    # ---- gallery ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick_ssa = ssa_cases[:: max(1, len(ssa_cases) // 4)][:4]
    pick_g = [c for c in test if (d51 / "labelsTr" / f"{c}.nii.gz").exists()][:2]
    rowsets = [("SSA", c, [ssa / f"imagesTs_{it}" / f"{c}_0000.nii.gz" for it in ITEMS],
                ssa / "labelsTs_t1n" / f"{c}.nii.gz") for c in pick_ssa]
    rowsets += [("GLI", c, [d51 / f"imagesTs_{it}" / f"{c}_0000.nii.gz" for it in ITEMS],
                 d51 / "labelsTr" / f"{c}.nii.gz") for c in pick_g]
    views = ("axial", "coronal", "sagittal")
    fig, axes = plt.subplots(len(rowsets), 4 * 3, figsize=(4 * 3 * 1.6, len(rowsets) * 1.8))
    for i, (tag, c, ims, lp) in enumerate(rowsets):
        L = np.asarray(nib.load(str(lp)).dataobj) > 0
        cen = [int(x) for x in np.round(np.argwhere(L).mean(0))] if L.any() else [s // 2 for s in L.shape]
        for j, p in enumerate(ims):
            a = np.asarray(nib.load(str(p)).dataobj).astype(np.float32)
            hi = np.percentile(a[a > 0], 99) if (a > 0).any() else 1
            for k, vname in enumerate(views):
                sl = [slice(None)] * 3; sl[2 - k if k else 2] = cen[2 - k if k else 2]
                # axial: fix ax2; coronal: fix ax1; sagittal: fix ax0 -- same transform for both cohorts
                fixed = {0: 2, 1: 1, 2: 0}[k]
                sl = [slice(None)] * 3; sl[fixed] = cen[fixed]
                img2, lab2 = np.rot90(a[tuple(sl)]), np.rot90(L[tuple(sl)])
                ax = axes[i, j * 3 + k]
                ax.imshow(img2, cmap="gray", vmin=0, vmax=hi)
                if lab2.any():
                    ax.contour(lab2, levels=[0.5], colors="r", linewidths=0.6)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{ITEMS[j]} {vname}", fontsize=7)
            axes[i, 0].set_ylabel(f"{tag}\n{c}", fontsize=6)
    fig.suptitle("BraTS-SSA (rows 1-4) vs brats2024-glioma test (rows 5-6), voxel arrays as nnU-Net reads them "
                 "(rot90 of the array slice, identical for both cohorts); red = GT tumour", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gallery_ssa_vs_glioma.png", dpi=110)
    # mean-image panel (cohort means, identity orientation)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, (name, m) in enumerate((("SSA mean z-t1n", mean_int), ("glioma test mean z-t1n", g_int))):
        for k in range(3):
            fixed = {0: 2, 1: 1, 2: 0}[k]
            sl = [slice(None)] * 3; sl[fixed] = m.shape[fixed] // 2
            axes[i, k].imshow(np.rot90(m[tuple(sl)]), cmap="gray"); axes[i, k].set_xticks([]); axes[i, k].set_yticks([])
            axes[i, k].set_title(f"{name} {views[k]}", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "cohort_means.png", dpi=110)
    (out / "orientation_qc.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    sys.exit(0 if ok and not aff_mismatch else 3)


if __name__ == "__main__":
    main()
