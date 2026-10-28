#!/usr/bin/env python3
"""
Build the BIDS tree 1_BIDS_duke-breast-mri/onc-breast-dukebreastmri from the MAMA-MIA download
(Duke-Breast-Cancer-MRI subset, Synapse syn60868042) staged by 0_raw_duke-breast-mri/download_duke.py.

Source (DUKE_RAW_DIR, default $SCRATCH/duke_download/raw):
  DUKE_XXX/DUKE_XXX_0000.nii.gz        pre-contrast DCE phase
  DUKE_XXX/DUKE_XXX_000{1..4}.nii.gz   post-contrast DCE phases (2 to 4 per case, as released)
  DUKE_XXX/DUKE_XXX_mask.nii.gz        MAMA-MIA expert primary-tumour segmentation (0/1)

Output (images and masks are byte-identical copies, only renamed):
  sub-dukeXXX/anat/sub-dukeXXX_acq-pre_T1w.nii.gz, ..._acq-post<k>_T1w.nii.gz (+ .json)
  derivatives/labels/sub-dukeXXX/anat/sub-dukeXXX_acq-post1_T1w_label-tumor_seg.nii.gz
  participants.tsv / .json (source id, number of post-contrast phases, enhancement check)
README, LICENSE and dataset_description.json are written by hand next to this script's output.

Checks per case, before anything is written: every file passes a full gzip read, all phases and the
mask share one voxel grid (shape + affine), the mask is binary and non-empty. A case failing a check
is reported and not written. Idempotent: an existing identical file is left alone; writes go through
a .tmp file and os.replace. The existing nnU-Net test sets (2_nnUNet_duke-breast-mri) are not read or
changed by this script.

Run:  bash 5_scripts_duke-breast-mri/00_utils/00_01_bidsify.sh   (Slurm CPU job via run_job)
"""
from __future__ import annotations

import filecmp
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("DUKE_RAW_DIR", Path(os.environ.get("SCRATCH", "")) / "duke_download" / "raw"))
LEAF = DATASET_ROOT / "1_BIDS_duke-breast-mri" / "onc-breast-dukebreastmri"
DERIV = LEAF / "derivatives" / "labels"

PHASE_ACQ = {"0000": "pre", "0001": "post1", "0002": "post2", "0003": "post3", "0004": "post4"}
PHASE_TEXT = {"0000": "pre-contrast phase", "0001": "first post-contrast phase",
              "0002": "second post-contrast phase", "0003": "third post-contrast phase",
              "0004": "fourth post-contrast phase"}


def _gzip_ok(p: Path) -> bool:
    try:
        with gzip.open(p, "rb") as fh:
            while fh.read(1 << 24):
                pass
        return True
    except (OSError, EOFError):
        return False


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return "same"
    tmp = dst.with_name(".tmp_" + dst.name)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return "written"


def _write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".tmp_" + dst.name)
    tmp.write_text(text)
    os.replace(tmp, dst)


def check_case(cdir: Path):
    pid = cdir.name
    phases = sorted(p.name[len(pid) + 1:len(pid) + 5] for p in cdir.glob(f"{pid}_000?.nii.gz"))
    mask_p = cdir / f"{pid}_mask.nii.gz"
    problems = []
    if not phases or phases[0] != "0000" or "0001" not in phases:
        problems.append(f"phases {phases}")
    if phases != [f"{i:04d}" for i in range(len(phases))]:
        problems.append(f"phase numbering not contiguous: {phases}")
    if not mask_p.exists():
        problems.append("no mask")
    files = [cdir / f"{pid}_{ph}.nii.gz" for ph in phases] + ([mask_p] if mask_p.exists() else [])
    for f in files:
        if not _gzip_ok(f):
            problems.append(f"gzip broken: {f.name}")
    if problems:
        return pid, phases, None, problems
    imgs = {ph: nib.load(str(cdir / f"{pid}_{ph}.nii.gz")) for ph in phases}
    mask = nib.load(str(mask_p))
    ref = imgs["0000"]
    for ph, im in list(imgs.items()) + [("mask", mask)]:
        if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-4):
            problems.append(f"grid of {ph} differs from pre-contrast")
    m = np.asanyarray(mask.dataobj)
    vals = np.unique(m)
    if not set(vals.tolist()) <= {0, 1}:
        problems.append(f"mask values {vals[:10]}")
    if not m.any():
        problems.append("mask empty")
    info = None
    if not problems:
        inside = m > 0
        pre = float(np.asanyarray(imgs["0000"].dataobj)[inside].mean())
        post1 = float(np.asanyarray(imgs["0001"].dataobj)[inside].mean())
        info = {"post1_enhances": "yes" if post1 > pre else "no", "mask_voxels": int(inside.sum())}
    return pid, phases, info, problems


def main() -> None:
    cases = sorted(d for d in RAW.glob("DUKE_*") if d.is_dir())
    if len(cases) == 0:
        sys.exit(f"no DUKE_* cases under {RAW}")
    print(f"[bidsify] {len(cases)} cases under {RAW} -> {LEAF}")
    rows, failed, stats = [], [], {"written": 0, "same": 0}
    for cdir in cases:
        pid, phases, info, problems = check_case(cdir)
        if problems:
            failed.append((pid, problems))
            print(f"[bidsify] SKIP {pid}: {problems}")
            continue
        sub = "sub-duke" + pid.split("_")[1]
        for ph in phases:
            acq = PHASE_ACQ[ph]
            dst = LEAF / sub / "anat" / f"{sub}_acq-{acq}_T1w.nii.gz"
            stats[_copy(cdir / f"{pid}_{ph}.nii.gz", dst)] += 1
            side = {"Description": f"Dynamic contrast-enhanced T1-weighted breast MRI, {PHASE_TEXT[ph]}",
                    "MAMAMIAChannel": f"{pid}_{ph}.nii.gz"}
            _write_text(dst.with_name(dst.name.replace(".nii.gz", ".json")), json.dumps(side, indent=2) + "\n")
        dst = DERIV / sub / "anat" / f"{sub}_acq-post1_T1w_label-tumor_seg.nii.gz"
        stats[_copy(cdir / f"{pid}_mask.nii.gz", dst)] += 1
        rows.append((sub, pid, len(phases) - 1, info["post1_enhances"]))
    tsv = "participant_id\tsource_id\tn_postcontrast_phases\tpost1_enhances\n" + "".join(
        f"{a}\t{b}\t{c}\t{d}\n" for a, b, c, d in rows)
    _write_text(LEAF / "participants.tsv", tsv)
    pjson = {
        "participant_id": {"Description": "BIDS subject id, sub-duke<NNN> from the source id DUKE_<NNN>"},
        "source_id": {"Description": "Patient id in the MAMA-MIA release (DUKE_<NNN>)"},
        "n_postcontrast_phases": {"Description": "Number of post-contrast DCE phases released by MAMA-MIA for this patient (2 to 4)"},
        "post1_enhances": {"Description": "Whether the mean intensity inside the tumour mask is higher in the first post-contrast phase than in the pre-contrast phase (computed by the conversion script)",
                           "Levels": {"yes": "higher after contrast", "no": "not higher"}},
    }
    _write_text(LEAF / "participants.json", json.dumps(pjson, indent=2) + "\n")
    deriv_dd = {
        "Name": "MAMA-MIA expert primary-tumour segmentations, Duke-Breast-Cancer-MRI subset",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "MAMA-MIA expert segmentation (Garrucho et al. 2025)",
                         "Description": "Manual expert segmentations distributed by MAMA-MIA; copied unchanged, only renamed",
                         "CodeURL": "https://github.com/LidiaGarrucho/MAMA-MIA"}],
        "SourceDatasets": [{"URL": "https://www.synapse.org/Synapse:syn60868042"}],
    }
    _write_text(DERIV / "dataset_description.json", json.dumps(deriv_dd, indent=2) + "\n")
    code = LEAF / "code"
    code.mkdir(exist_ok=True)
    for src in [Path(__file__), Path(__file__).with_suffix(".sh"),
                DATASET_ROOT / "0_raw_duke-breast-mri" / "download_duke.py"]:
        if src.exists():
            shutil.copyfile(src, code / src.name)
    n_no = sum(1 for r in rows if r[3] == "no")
    print(f"[bidsify] subjects written: {len(rows)}, failed: {len(failed)}, files written: {stats['written']}, "
          f"already identical: {stats['same']}, post1 not enhancing: {n_no}")
    print("[bidsify] phase counts:", {k: sum(1 for r in rows if r[2] == k) for k in sorted({r[2] for r in rows})})
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
