#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the LLD-MMRI-MALIGNANT evaluation set.

Eight items, one per BIDS-ified phase -- identical item names to lld-mmri-hcc's
own builder so a config can pool the two without a translation table. 109
patients (intrahepatic cholangiocarcinoma + hepatic metastasis) vs
lld-mmri-hcc's 157 HCC patients: same source repo and cohort, different lesion
types. atlas-liver-hcc's models are single-channel, so each volume is fed as
channel _0000.

Reads:  ../../1_BIDS_lld-mmri-malignant/lld-mmri-malignant/sub-lldmalNNN/anat/ (+ derivatives)
Writes: ../../2_nnUNet_lld-mmri-malignant/raw/imagesTs_<item>/{case}_0000.nii.gz
                                             /labelsTs_<item>/{case}.nii.gz

Case id = participant label without the sub- prefix (e.g. lldmal000) -- the
lldmal prefix keeps these distinguishable from lld-mmri-hcc's lldhcc* case ids
if the two are ever pooled into one table.

    python 05_00_build_test_inputs.py
"""
import gzip
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_lld-mmri-malignant" / "lld-mmri-malignant"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_lld-mmri-malignant" / "raw"

PHASES = {
    "t2wi": "T2w", "dwi": "dwi",
    "ce-pre": "ce-pre_T1w", "ce-art": "ce-art_T1w", "ce-pv": "ce-pv_T1w", "ce-del": "ce-del_T1w",
    "inphase": "inphase_T1w", "outphase": "outphase_T1w",
}   # item name -> BIDS suffix (must stay identical to lld-mmri-hcc's mapping)


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out)


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-lldmal*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-lldmal* under {BIDS_ROOT} -- run 00_utils/00_01_bidsify.py first.")

    for item, suffix in PHASES.items():
        img_dir = NNUNET_RAW / f"imagesTs_{item}"
        lab_dir = NNUNET_RAW / f"labelsTs_{item}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for sub in subs:
            case = sub[len("sub-"):]
            img = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
            seg = DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.nii.gz"
            if not img.exists() or not seg.exists():
                missing.append(case)
                continue
            gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
            gzip_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(subs)} -> {img_dir.name} (+labels)"
        if missing:
            status += f"  MISSING: {missing}"
        print(status)


if __name__ == "__main__":
    main()
