#!/usr/bin/env python3
"""Write a copy of the BraTS-SSA nnU-Net test inputs (imagesTs_<c>/labelsTs_<c>, c in t1n t1c t2w t2f)
reoriented to the given axis codes (lossless nibabel as_reoriented: permute/flip, no resampling).
brats2024-glioma's own nnU-Net arrays (train + test) are LAS (182x218x182, 1 mm; measured 2026-10-20
by qc_orientation.py), while the archived SSA pipeline wrote LPS (it assumed the glioma data were LPS).
Usage: reorient_inputs.py <in_raw_dir> <out_raw_dir> <AXCODES e.g. LAS>"""
import sys
from pathlib import Path

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

src, dst, codes = Path(sys.argv[1]), Path(sys.argv[2]), tuple(sys.argv[3])
n = 0
for sub in [f"{k}Ts_{c}" for k in ("images", "labels") for c in ("t1n", "t1c", "t2w", "t2f")]:
    (dst / sub).mkdir(parents=True, exist_ok=True)
    for f in sorted((src / sub).glob("*.nii.gz")):
        o = dst / sub / f.name
        if o.exists():
            continue
        im = nib.load(str(f))
        t = ornt_transform(io_orientation(im.affine), axcodes2ornt(codes))
        r = im.as_reoriented(t)
        if r.get_data_dtype() != im.get_data_dtype():
            r.set_data_dtype(im.get_data_dtype())
        nib.save(r, str(o))
        assert nib.aff2axcodes(nib.load(str(o)).affine) == codes, o
        n += 1
print(f"wrote {n} files -> {dst} ({''.join(codes)})")
