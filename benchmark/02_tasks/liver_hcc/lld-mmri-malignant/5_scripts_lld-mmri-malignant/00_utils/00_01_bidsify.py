#!/usr/bin/env python3
"""
BIDSify the MALIGNANT NON-HCC subset of LLD-MMRI-MedSAM2
(0_raw_lld-mmri-malignant/LLD-MMRI-MedSAM2/) -> 1_BIDS_lld-mmri-malignant/lld-mmri-malignant/.

109 patients: intrahepatic cholangiocarcinoma (category 1, n=58) + hepatic
metastasis (category 3, n=51), all 8 phases. Sibling of benchmark/02_tasks/liver_hcc/lld-mmri-hcc,
which holds the SAME source repo's 157 HCC patients (category 6) -- the two are
deliberately separate datasets so lld-mmri-hcc's existing HCC-only numbers can
never silently change, and so this subset can be reported on its own.

⚠️ SAME COHORT, NOT AN INDEPENDENT ONE. Same source repo, same acquisition
cohort, same 8 phases, same MedSAM2 human-in-the-loop mask provenance as
lld-mmri-hcc. Added 2026-09-01 to raise CASE COUNT on atlas-liver-hcc's
unambiguously-OOD contrasts (T2WI, DWI) after the ablation ladder was found to
flip sign depending on which contrasts were pooled. It does NOT buy cohort
independence -- never describe it as a second independent cohort.

Disease differs from lld-mmri-hcc (ICC/metastasis vs HCC), so pooling the two
changes the estimand from "HCC tumour" to "malignant liver lesion". That is a
scope change to state explicitly wherever the pooled number is reported.

Subject ids: sequential lldmal000..lldmal108 (raw ids mix hyphenated and
non-hyphenated forms, and BIDS entity labels must be alphanumeric). Mapping
preserved in participants.tsv (source_patient_id, lesion_category) and
case_id_map.json.

Segmentation is LESION ONLY (binary {0,1}) -- as in lld-mmri-hcc, cross-dataset
evaluation must remap the atlas model's tumour class (2) onto this dataset's
lesion class (1) via evaluate.py's --label_map.

Orientation/spacing are NOT touched here. Re-verify nib.aff2axcodes on every
case of every phase before trusting anything downstream -- do NOT assume the
lld-mmri-hcc result carries over just because it is the same source repo
(this project has been bitten by that assumption twice; see
feedback_verify_orientation_on_onboarding).

Reads:   0_raw_lld-mmri-malignant/LLD-MMRI-MedSAM2/{images,labels}/<PID>_<cat>_<PHASE>[_0000].nii.gz
         0_raw_lld-mmri-malignant/malignant_patient_categories.json
Writes:  1_BIDS_lld-mmri-malignant/lld-mmri-malignant/

Usage:  python 00_01_bidsify.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]        # benchmark/02_tasks/liver_hcc/lld-mmri-malignant
RAW_ROOT = DATASET_ROOT / "0_raw_lld-mmri-malignant"
RAW = RAW_ROOT / "LLD-MMRI-MedSAM2"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_lld-mmri-malignant" / "lld-mmri-malignant"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

PHASES = {
    "T2WI": "T2w", "DWI": "dwi",
    "C-pre": "ce-pre_T1w", "C+A": "ce-art_T1w", "C+V": "ce-pv_T1w", "C+Delay": "ce-del_T1w",
    "InPhase": "inphase_T1w", "OutPhase": "outphase_T1w",
}   # identical mapping to lld-mmri-hcc so both datasets' item names line up
    # exactly and a config can pool them without a translation table.

CATEGORY_NAMES = {1: "Intrahepatic_cholangiocarcinoma", 3: "Hepatic_metastasis"}


def _link(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)
    return True


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bidsify() -> None:
    cats = json.loads((RAW_ROOT / "malignant_patient_categories.json").read_text())
    pids = sorted(cats)
    if len(pids) != 109:
        raise SystemExit(f"expected 109 malignant non-HCC patient ids, found {len(pids)}")

    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "LLD-MMRI-MALIGNANT -- intrahepatic cholangiocarcinoma + hepatic metastasis "
                "subset of LLD-MMRI-MedSAM2 (8 phases), cross-contrast eval-only set for "
                "atlas-liver-hcc. Same cohort as lld-mmri-hcc, different lesion types.",
        "BIDSVersion": "1.9.0",
        "License": "CC BY-NC 4.0 (research use only, no commercial use)",
        "Authors": ["Lou et al. (2025, LLD-MMRI)", "Ma, Yang et al. (2025, MedSAM2 annotation)"],
        "ReferencesAndLinks": [
            "https://github.com/LMMMEng/LLD-MMRI-Dataset",
            "https://huggingface.co/benchmark/wanglab/LLD-MMRI-MedSAM2",
            "https://arxiv.org/abs/2504.03600",
        ],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "LLD-MMRI-MALIGNANT lesion segmentation masks (MedSAM2, human-in-the-loop)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "MedSAM2 human-in-the-loop annotation (Ma, Yang et al. 2025)"}],
    })

    case_id_map: dict[str, dict] = {}
    rows = ["participant_id\tsource_patient_id\tlesion_category\tlesion_category_name"]
    missing: list[str] = []
    for i, pid in enumerate(pids):
        sub = f"sub-lldmal{i:03d}"
        cat = cats[pid]
        case_id_map[sub] = {"source_patient_id": pid, "lesion_category": cat,
                            "lesion_category_name": CATEGORY_NAMES.get(cat, str(cat))}
        for raw_phase, suffix in PHASES.items():
            img_ok = _link(RAW / "images" / f"{pid}_{cat}_{raw_phase}_0000.nii.gz",
                           BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz")
            lab_ok = _link(RAW / "labels" / f"{pid}_{cat}_{raw_phase}.nii.gz",
                           DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.nii.gz")
            if not (img_ok and lab_ok):
                missing.append(f"{pid}_{cat}_{raw_phase}")
                continue
            _json(BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.json",
                  {"Modality": "MR",
                   "Description": f"{raw_phase}, {CATEGORY_NAMES.get(cat, cat)} (category {cat})"})
            _json(DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.json",
                  {"Manual": False, "GeneratedBy": "MedSAM2 human-in-the-loop",
                   "Labels": {"1": "lesion"}})
        rows.append(f"{sub}\t{pid}\t{cat}\t{CATEGORY_NAMES.get(cat, cat)}")

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    _json(BIDS_ROOT / "case_id_map.json", case_id_map)
    print(f"BIDSified {len(pids)} malignant non-HCC patients -> {BIDS_ROOT}")
    if missing:
        (RAW_ROOT / "missing_phase_pairs.txt").write_text("\n".join(missing) + "\n")
        print(f"WARNING: {len(missing)} patient/phase pairs missing an image or label "
              f"-> {RAW_ROOT/'missing_phase_pairs.txt'} (per-item case counts will differ; "
              f"check them before interpreting any pooled number)")


if __name__ == "__main__":
    bidsify()
