#!/usr/bin/env python3
"""Add the MRI acquisition fields released in the TotalSegmentator MRI meta.csv to participants.tsv.

Columns added (verbatim from meta.csv, "n/a" for CT subjects and for empty values):
  mri_scanning_sequence   DICOM ScanningSequence (GR, SE, IR, EP, RM; several joined with "\\")
  mri_repetition_time     as released; the release mixes seconds and milliseconds, and rounds short values to 0.00
  mri_echo_time           as released; same caveat
  mri_field_strength      tesla
These are what the release provides; they are not enough to name a BIDS contrast (T1w/T2w/...) per scan,
which is why the MRI images keep the generic _MRI suffix (see README.md).

  TOTALSEG_MRI_ZIP=<path to the MRI release zip> python 00_03_add_mri_sequence_columns.py
Idempotent: existing mri_* columns are rewritten. Light (reads one small csv member): fine on a login node.
"""
from __future__ import annotations

import ast
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = DATASET_ROOT / "1_BIDS_totalseg-pelvic" / "pelvis-totalsegpelvic"
MRI_ZIP = Path(os.environ.get("TOTALSEG_MRI_ZIP", "mri.zip"))
NEW = ["mri_scanning_sequence", "mri_repetition_time", "mri_echo_time", "mri_field_strength"]
SRC = {"mri_scanning_sequence": "scanning_sequence", "mri_repetition_time": "repetition_time",
       "mri_echo_time": "echo_time", "mri_field_strength": "magnetic_field_strength"}


def _seq(v: str) -> str:
    v = v.strip()
    if v.startswith("["):
        v = "\\".join(ast.literal_eval(v))
    return v


def main() -> None:
    if not MRI_ZIP.exists():
        sys.exit(f"MRI zip not found: {MRI_ZIP} (set TOTALSEG_MRI_ZIP)")
    with zipfile.ZipFile(MRI_ZIP) as zf:
        text = zf.read("meta.csv").decode("utf-8-sig")
    meta = {r["image_id"]: r for r in csv.DictReader(io.StringIO(text), delimiter=";")}
    ptsv = BIDS_ROOT / "participants.tsv"
    rows = list(csv.DictReader(ptsv.open(), delimiter="\t"))
    cols = [c for c in rows[0].keys() if c not in NEW] + NEW
    missing = []
    for r in rows:
        m = meta.get(r["source_case_id"]) if r["modality"] == "mri" else None
        if r["modality"] == "mri" and m is None:
            missing.append(r["source_case_id"])
        for c in NEW:
            v = (_seq(m[SRC[c]]) if c == "mri_scanning_sequence" else m[SRC[c]].strip()) if m else ""
            r[c] = v if v else "n/a"
    if missing:
        sys.exit(f"MRI cases absent from meta.csv: {missing[:10]}")
    tmp = ptsv.with_name(".tmp_participants.tsv")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ptsv)
    n_mri = sum(r["modality"] == "mri" for r in rows)
    print(f"participants.tsv: {len(rows)} rows, MRI acquisition columns filled for {n_mri} MRI subjects")


if __name__ == "__main__":
    main()
