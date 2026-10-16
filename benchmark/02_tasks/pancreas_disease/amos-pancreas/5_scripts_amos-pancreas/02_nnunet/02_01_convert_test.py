#!/usr/bin/env python3
"""TEMPLATE -- complete every TODO. BIDS (1_BIDS_amos-pancreas/<leaf>/) -> flat nnU-Net TEST set 2_nnUNet_amos-pancreas/raw/{imagesTs,labelsTs}_<item>/  for items: mixed_t1.

Contract (each point cost this project a real debugging session):
  * case ids IDENTICAL across items and carry NO contrast suffix ("amos-pancreas_<id>"), so the patient-level significance merge pairs them;
  * pathology/stage/timepoint EXCLUSIONS (bilateral disease, implants, empty or garbage masks, label-side disagreement, wrong timepoint ...) are decided BEFORE any prediction,
    applied uniformly and recorded in the manifest with the reason per case;
  * orientation: reorient to LPS with shared orient.reorient_file (records the raw axcodes); look at a rendered PNG afterwards (aff2axcodes is blind to 180-degree flips);
  * GEOMETRY MATCH: run 02_02_fov_audit.sh first. If test FOV/spacing differ from the source's training data, crop/resample HERE (planned, documented, applied before
    prediction -- crop-before-predict, never mask-at-eval): shared helpers unilateral_crop.py (lesion_side_half, ap_skin_window), derive_ap_crop.py (breast), fov.py +
    00_02_predict/fov_crop_predict_evaluate.sh (CHAOS-style S-I slab), or a documented local rule. Name cropped items distinctly (e.g. <item>_crop) and re-audit;
  * assertions per case: image/label grids equal, labels binary/expected values, mask non-empty, and an EMPIRICAL channel/physics check (lesion mean vs brain in the right direction);
  * write amos-pancreas_test_manifest.json: per-case shape/spacing/lesion volume, exclusions + reasons, and the lesion-size distribution next to the SOURCE's labelsTr distribution.
"""
from __future__ import annotations
import json, sys, shutil
from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DATASET_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402,F401
# from unilateral_crop import crop_axis0, lesion_side_half, crop_axis1, ap_skin_window  # noqa: E402  (breast crop helpers)

BIDS = DATASET_ROOT / "1_BIDS_amos-pancreas" / "abdomen-amos"
OUT = DATASET_ROOT / "2_nnUNet_amos-pancreas" / "raw"
ITEMS = {"mixed_t1": "T2w"}   # BIDS suffix is "T2w" for ALL 60 AMOS MRI cases (wrong: sheets show T1-GRE/CE/in-opposed phase mostly) -> item is a MIXED clinical set, own named column
SOURCE_LABELS_TR = REPO_ROOT / "benchmark/02_tasks/pancreas_disease/pansegdata/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE/labelsTr"
AUDIT = DATASET_ROOT / "0_raw_amos-pancreas" / "audit" / "audit.tsv"   # from 02_00_audit_gallery.sh (spacing/axcodes/volumes/S-I touch per case)
PANCREAS_ID = 10                                                         # AMOS label id (verified: mean volume 79 mL over 60 cases = PanSegData's ~80 mL median)


def _audit():
    rows = [l.split("\t") for l in AUDIT.read_text().splitlines()]
    return {r[0]: dict(zip(rows[0], r)) for r in rows[1:]}


def iter_cases():
    for sub, a in sorted(_audit().items()):
        img = BIDS / sub / "anat" / f"{sub}_T2w.nii.gz"
        msk = BIDS / "derivatives/labels" / sub / "anat" / f"{sub}_T2w_label-organs_dseg.nii.gz"
        yield f"amos-pancreas_{sub.replace('sub-', '')}", {"mixed_t1": img}, msk, a


def exclude(case_id, meta):
    sp = [float(v) for v in meta["spacing"].split("x")]
    if int(max(range(3), key=lambda i: sp[i])) != 2:
        return "coronal/sagittal acquisition (thick slice axis is not S-I): plane mismatch vs the axial PanSegData training scans; resampling a 3 mm coronal stack to axial would be a different experiment"
    if int(meta["si_touch"]):
        return "pancreas mask touches the S-I image border (truncated by the FOV)"
    if float(meta["pancreas_ml"]) < 20:
        return "pancreas mask < 20 mL (truncated or garbage)"
    return None


def main():
    manifest = {"cases": {}, "excluded": {}}
    for cid, imgs, mask, meta in iter_cases():
        why = exclude(cid, meta)
        if why:
            manifest["excluded"][cid] = why; continue
        meta = dict(meta); meta["orig_axcodes"] = meta["axcodes"]
        for item, ip in imgs.items():
            idir = OUT / f"imagesTs_{item}"; ldir = OUT / f"labelsTs_{item}"; idir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)
            ti, tl = idir / f"{cid}_0000.nii.gz", ldir / f"{cid}.nii.gz"
            shutil.copy(ip, ti)
            lab = nib.load(mask); arr = (np.asanyarray(lab.dataobj) == PANCREAS_ID).astype(np.uint8)
            nib.save(nib.Nifti1Image(arr, lab.affine), tl)
            reorient_file(ti); reorient_file(tl)                      # in place, to LPS (a no-op here: BIDS already LPS)
            im, lb = nib.load(ti), nib.load(tl)
            assert im.shape == lb.shape and np.allclose(im.affine, lb.affine, atol=1e-3), f"{cid}: grid mismatch"
            la = np.asanyarray(lb.dataobj); assert set(np.unique(la)) <= {0, 1} and la.sum() > 0, f"{cid}: bad mask"
            assert "".join(nib.aff2axcodes(im.affine)) == "LPS", cid
            meta["vol_ml"] = round(float(la.sum() * np.prod(im.header.get_zooms()[:3]) / 1000), 1)
        manifest["cases"][cid] = meta
    v = np.array([c["vol_ml"] for c in manifest["cases"].values()])
    src = np.array([float(np.asanyarray(nib.load(f).dataobj).sum() * np.prod(nib.load(f).header.get_zooms()[:3]) / 1000) for f in sorted(SOURCE_LABELS_TR.glob("*.nii.gz"))])
    manifest["volume_ml_summary"] = {"companion_median": float(np.median(v)), "companion_p10_p90": [float(np.percentile(v, 10)), float(np.percentile(v, 90))],
                                      "source_labelsTr_median": float(np.median(src)), "source_labelsTr_p10_p90": [float(np.percentile(src, 10)), float(np.percentile(src, 90))]}
    (OUT / "amos-pancreas_test_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("cases:", len(manifest["cases"]), "excluded:", len(manifest["excluded"]))


if __name__ == "__main__":
    main()
