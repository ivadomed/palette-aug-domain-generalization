"""CT value audit before the CT re-send (2026-10-28). Read-only on every dataset.

  pelvis <bids_leaf> <source_dir>   value profile of every *_CT image; for each case fetched from the
                                    TotalSegmentator zip into <source_dir>/sNNNN/ct.nii.gz, compare with ours
  chaos  <bids_leaf> <raw_ct_dir>   value profile of every chaos CT; for subjects with values > 2000, read the
                                    DICOM slices with pydicom and report where the high values are in the source
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def profile(arr):
    return dict(dtype=str(arr.dtype), min=float(arr.min()), max=float(arr.max()),
                n_m8192=int((arr == -8192).sum()), n_m3024=int((arr == -3024).sum()),
                n_lt_m1024=int((arr < -1024).sum()), n_gt_3071=int((arr > 3071).sum()))


def pelvis(leaf: Path, src: Path):
    for img_p in sorted(leaf.glob("sub-ct*/*/sub-ct*_CT.nii.gz")) + sorted(leaf.glob("sub-ct*/*/sub-ct*_ct.nii.gz")):
        img = nib.load(str(img_p))
        arr = np.asanyarray(img.dataobj)
        p = profile(arr)
        flag = p["n_m8192"] or p["n_m3024"] or p["max"] > 5000
        if flag:
            print(f"OURS {img_p.name}: {p} on-disk dtype {img.get_data_dtype()}")
        case = "s" + img_p.name.split("_")[0][len("sub-ct"):]
        s = src / case / "ct.nii.gz"
        if s.exists():
            simg = nib.as_closest_canonical(nib.load(str(s)))
            sarr = np.asanyarray(simg.dataobj)
            same = sarr.shape == arr.shape and np.array_equal(sarr, arr)
            print(f"SOURCE {case}: {profile(sarr)} source dtype {nib.load(str(s)).get_data_dtype()} "
                  f"| voxel data identical to ours after canonical reorientation: {same}")


def chaos(leaf: Path, raw: Path):
    import pydicom
    for img_p in sorted(leaf.glob("sub-CT*/*/sub-CT*_CT.nii.gz")) + sorted(leaf.glob("sub-CT*/*/sub-CT*_ct.nii.gz")):
        img = nib.load(str(img_p))
        arr = np.asanyarray(img.dataobj)
        p = profile(arr)
        hi = np.argwhere(arr > 2000)
        print(f"OURS {img_p.name}: {p} on-disk dtype {img.get_data_dtype()} n>2000={len(hi)}"
              + (f" slices(k) {sorted(set(hi[:, 2].tolist()))[:20]}" if len(hi) else ""))
        if not len(hi):
            continue
        cid = str(int(img_p.name.split("_")[0][len("sub-CT"):]))
        files = sorted((raw / cid / "DICOM_anon").glob("*.dcm"))
        rows = []
        for f in files:
            d = pydicom.dcmread(str(f))
            px = d.pixel_array.astype(np.float64) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
            if px.max() > 2000:
                rows.append((f.name, int(getattr(d, "InstanceNumber", -1)), float(d.ImagePositionPatient[2]),
                             int((px > 2000).sum()), float(px.max()), int(d.BitsStored), int(d.PixelRepresentation),
                             str(getattr(d, "RescaleIntercept", "")), str(getattr(d, "RescaleSlope", "")),
                             "overlay" if any(0x6000 <= t.group <= 0x60FF for t in d.keys()) else ""))
        d0 = pydicom.dcmread(str(files[0]))
        print(f"  DICOM CT{cid}: {len(files)} files, BitsAllocated {d0.BitsAllocated} BitsStored {d0.BitsStored} "
              f"PixelRepresentation {d0.PixelRepresentation} slope {getattr(d0,'RescaleSlope','')} intercept {getattr(d0,'RescaleIntercept','')}; "
              f"slices with values > 2000 in the SOURCE DICOM: {len(rows)}")
        for r in rows[:12]:
            print("   ", r)


if __name__ == "__main__":
    {"pelvis": pelvis, "chaos": chaos}[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3]))
