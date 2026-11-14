#!/usr/bin/env python3
"""T08: mask agreement across contrasts (registration plans/2026-11-14-PREDICTION_T08.md, sha256 023b9d1c...).

Per subject, organ and direction A->B: d_eq = (V_B - V_A) / S_A (mm), V = voxel count x voxel volume, S = marching-cubes surface
area of A's mask in physical spacing. Statistics per (dataset, organ, direction, subset): median d_eq, percentile bootstrap
95% CI (10,000 resamples, seed 0), fraction of subjects with the median's sign. Secondary: Dice between the two masks in
scanner space (nearest-neighbour resampling by headers) and after a centroid-only translation. Rules: see registration.

Surface: skimage.measure.marching_cubes(level=0.5) on the organ mask cropped to its bounding box and zero-padded by 1 voxel
(a mask touching the image border is therefore closed there), spacing = header zooms; area = mesh_surface_area.

  t08_agreement.py --root REPO --out DIR [--max-subjects N]
Writes DIR/per_subject.tsv (case ids, untracked dir), DIR/summary.tsv, DIR/gates.json, DIR/tables.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.measure import marching_cubes, mesh_surface_area

N_BOOT, SEED = 10_000, 0
PAN = "benchmark/02_tasks/pancreas_disease/pansegdata"
CHAOS = "benchmark/02_tasks/abdomen_healthy/chaos"
DATASETS = {
    "Pancreas": dict(
        A="t1wce", B="t2w", organs={"pancreas": 1},
        dirs={"t1wce": f"{PAN}/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE",
              "t2w": f"{PAN}/2_nnUNet_pansegdata/raw/Dataset151_PanSegData_T2W"},
        test_dir={"t1wce": "labelsTs_t1wce", "t2w": "labelsTs_t2w"}),
    "CHAOS": dict(
        A="t1in", B="t2spir", organs={"liver": 1, "right_kidney": 2, "left_kidney": 3, "spleen": 4},
        dirs={"t1in": f"{CHAOS}/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in",
              "t2spir": f"{CHAOS}/2_nnUNet_chaos/raw/Dataset061_CHAOS_MR_T2spir"},
        test_dir={"t1in": "labelsTs_t1in", "t2spir": "labelsTs_t2spir"}),
}


def list_cases(d: Path):
    return sorted(p.name[:-7] for p in d.glob("*.nii.gz"))


def surface_area(mask: np.ndarray, zooms) -> float:
    idx = np.nonzero(mask)
    lo = [max(a.min() - 1, 0) for a in idx]
    hi = [a.max() + 2 for a in idx]
    sub = mask[tuple(slice(l, h) for l, h in zip(lo, hi))]
    sub = np.pad(sub.astype(np.uint8), 1)
    verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=tuple(float(z) for z in zooms))
    return float(mesh_surface_area(verts, faces))


def sitk_stats(path: Path, label: int):
    img = sitk.ReadImage(str(path))
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sitk.Cast(img, sitk.sitkUInt8))
    if label not in f.GetLabels():
        return img, 0, None
    return img, int(f.GetNumberOfPixels(label)), f.GetCentroid(label)


def dice_pair(imgA, imgB, label, cA, cB):
    """Dice of B resampled onto A's grid (identity) and after centroid translation. Returns dice0, dice_c, kept0, kept_c."""
    mA = sitk.Cast(imgA == label, sitk.sitkUInt8)
    mB = sitk.Cast(imgB == label, sitk.sitkUInt8)
    nA = int(sitk.GetArrayViewFromImage(mA).sum())
    out = []
    for tr in (sitk.Transform(), sitk.TranslationTransform(3, [b - a for a, b in zip(cA, cB)])):
        rb = sitk.Resample(mB, mA, tr, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        a = sitk.GetArrayViewFromImage(mA); b = sitk.GetArrayViewFromImage(rb)
        nB, inter = int(b.sum()), int((a & b).sum())
        out.append((2 * inter / (nA + nB) if nA + nB else np.nan, nB))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def boot_ci(x, rng):
    n = len(x)
    med = np.median(x[rng.integers(0, n, (N_BOOT, n))], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-subjects", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gates = dict(volume_count_mismatch=0, spacing_mismatch=0, n_checked=0, labels_unexpected=0)
    pairing, rows = {}, []

    for dname, cfg in DATASETS.items():
        A, B = cfg["A"], cfg["B"]
        src = {}  # contrast -> {case: (path, split)}
        for c in (A, B):
            d = root / cfg["dirs"][c]
            m = {k: (d / "labelsTr" / f"{k}.nii.gz", "train_pool") for k in list_cases(d / "labelsTr")}
            m.update({k: (d / cfg["test_dir"][c] / f"{k}.nii.gz", "test") for k in list_cases(d / cfg["test_dir"][c])})
            src[c] = m
        both = sorted(set(src[A]) & set(src[B]))
        pairing[dname] = dict(n_A=len(src[A]), n_B=len(src[B]), n_paired=len(both),
                              only_A=sorted(set(src[A]) - set(src[B])), only_B=sorted(set(src[B]) - set(src[A])),
                              split_disagree=[k for k in both if src[A][k][1] != src[B][k][1]],
                              n_test_paired=sum(src[A][k][1] == "test" for k in both))
        if a.max_subjects:
            both = both[: a.max_subjects]
        for case in both:
            pa, split = src[A][case]; pb, _ = src[B][case]
            na, nb = nib.load(str(pa)), nib.load(str(pb))
            arrA, arrB = np.asarray(na.dataobj), np.asarray(nb.dataobj)
            zA, zB = na.header.get_zooms()[:3], nb.header.get_zooms()[:3]
            for arr in (arrA, arrB):
                if not set(np.unique(arr)).issubset({0, *cfg["organs"].values()}):
                    gates["labels_unexpected"] += 1
            for oname, lab in cfg["organs"].items():
                r = dict(dataset=dname, organ=oname, case=case, split=split, status="ok")
                mA, mB = arrA == lab, arrB == lab
                vA, vB = int(mA.sum()), int(mB.sum())
                if vA == 0 or vB == 0:
                    r["status"] = "empty_" + ("A" if vA == 0 else "") + ("B" if vB == 0 else "")
                    rows.append(r); continue
                # gate: second count via SimpleITK
                imgA, sA, cA = sitk_stats(pa, lab); imgB, sB, cB = sitk_stats(pb, lab)
                gates["n_checked"] += 1
                if sA != vA or sB != vB:
                    gates["volume_count_mismatch"] += 1
                if not (np.allclose(imgA.GetSpacing(), zA, atol=1e-6) and np.allclose(imgB.GetSpacing(), zB, atol=1e-6)):
                    gates["spacing_mismatch"] += 1
                VA, VB = vA * float(np.prod(zA)), vB * float(np.prod(zB))
                SA, SB = surface_area(mA, zA), surface_area(mB, zB)
                d0, dc, k0, kc = dice_pair(imgA, imgB, lab, cA, cB)
                r.update(vox_A=vA, vox_B=vB, V_A=VA, V_B=VB, S_A=SA, S_B=SB,
                         d_eq_AB=(VB - VA) / SA, d_eq_BA=(VA - VB) / SB,
                         dice_scanner=d0, dice_centroid=dc, B_kept_frac_scanner=k0 * float(np.prod(zA)) / VB,
                         B_kept_frac_centroid=kc * float(np.prod(zA)) / VB)
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_subject.tsv", sep="\t", index=False)
    ok = df[df.status == "ok"]
    pairing["dropped_empty"] = df[df.status != "ok"].groupby(["dataset", "organ", "status"]).size().reset_index(name="n").to_dict("records")
    srows = []
    for (dname, organ), g in ok.groupby(["dataset", "organ"], sort=False):
        A, B = DATASETS[dname]["A"], DATASETS[dname]["B"]
        for subset, gg in (("pool", g), ("test", g[g.split == "test"])):
            if len(gg) == 0:
                continue
            for dirn, col in ((f"{A}->{B}", "d_eq_AB"), (f"{B}->{A}", "d_eq_BA")):
                x = gg[col].to_numpy(float)
                med = float(np.median(x))
                lo, hi = boot_ci(x, np.random.default_rng(SEED))
                sgn = np.sign(med)
                srows.append(dict(dataset=dname, organ=organ, subset=subset, direction=dirn, n=len(x), median_d_eq=med,
                                  ci_lo=lo, ci_hi=hi, sign_frac=float((np.sign(x) == sgn).mean()) if sgn != 0 else 0.0,
                                  median_V_A=float(gg.V_A.median()), median_V_B=float(gg.V_B.median()),
                                  median_dice_scanner=float(gg.dice_scanner.median()),
                                  q25_dice_scanner=float(gg.dice_scanner.quantile(.25)), q75_dice_scanner=float(gg.dice_scanner.quantile(.75)),
                                  median_dice_centroid=float(gg.dice_centroid.median()),
                                  q25_dice_centroid=float(gg.dice_centroid.quantile(.25)), q75_dice_centroid=float(gg.dice_centroid.quantile(.75)),
                                  median_B_kept_scanner=float(gg.B_kept_frac_scanner.median())))
    sm = pd.DataFrame(srows)
    sm.to_csv(out / "summary.tsv", sep="\t", index=False)

    # Pancreas rules vs T04 reference (median over scored arms of per_fold mean_asd)
    t04 = pd.read_csv(root / "benchmark/01_commun_results/dg_mechanisms/T04/per_setting.tsv", sep="\t")
    rules = {}
    for setting, dirn in (("Pancreas_t1wce", "t1wce->t2w"), ("Pancreas_t2w", "t2w->t1wce")):
        t = t04[(t04.setting == setting) & (t04.set == "per_fold") & (t04.role == "scored")]
        ref = float(t.mean_asd.median())
        rules[dirn] = dict(t04_setting=setting, t04_arm_values={r.arm: float(r.mean_asd) for r in t.itertuples()}, t04_ref_median=ref)
        for subset in ("pool", "test"):
            ss = sm[(sm.dataset == "Pancreas") & (sm.subset == subset) & (sm.direction == dirn)]
            if ss.empty:  # only in --max-subjects smoke runs (no test subject among the first N)
                continue
            s = ss.iloc[0]
            c = dict(median=s.median_d_eq, ci_lo=s.ci_lo, ci_hi=s.ci_hi, sign_frac=s.sign_frac,
                     c1_median_pos=bool(s.median_d_eq > 0), c2_ci_excludes0=bool(s.ci_lo > 0), c3_sign70=bool(s.sign_frac >= 0.7),
                     c4_ge_half_ref=bool(s.median_d_eq >= 0.5 * ref))
            c["explains"] = all(c[k] for k in ("c1_median_pos", "c2_ci_excludes0", "c3_sign70", "c4_ge_half_ref"))
            rules[dirn][subset] = c
    for subset in ("pool", "test"):
        m = [abs(rules[d][subset]["median"]) for d in ("t1wce->t2w", "t2w->t1wce") if subset in rules[d]]
        rules[f"ruled_out_{subset}"] = bool(len(m) == 2 and all(v < 0.3 for v in m))
    gates["pairing"] = pairing
    gates["rules"] = rules
    gates["gate_pass"] = bool(gates["volume_count_mismatch"] == 0 and gates["spacing_mismatch"] == 0 and gates["labels_unexpected"] == 0)
    (out / "gates.json").write_text(json.dumps(gates, indent=1, default=str))

    f = lambda v: f"{v:.3f}"
    L = ["| dataset | organ | subset | direction | n | median d_eq (mm) | 95% CI | sign frac | median Dice scanner [IQR] | median Dice centroid [IQR] |", "|" + "---|" * 10]
    for r in sm.itertuples():
        L.append(f"| {r.dataset} | {r.organ} | {r.subset} | {r.direction} | {r.n} | {f(r.median_d_eq)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] | {r.sign_frac:.2f} | "
                 f"{f(r.median_dice_scanner)} [{f(r.q25_dice_scanner)}, {f(r.q75_dice_scanner)}] | {f(r.median_dice_centroid)} [{f(r.q25_dice_centroid)}, {f(r.q75_dice_centroid)}] |")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(json.dumps(gates, indent=1, default=str))
    return 0 if gates["gate_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
