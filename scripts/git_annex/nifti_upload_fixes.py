"""Small, verified in-place fixes for BIDS leaves before a git-annex upload (round 2, 2026-10-27).

  units <leaf>        set the spatial unit of every NIfTI-1 file whose xyzt_units says "unknown" to mm.
                      Patches the one header byte only (offset 123), re-gzips, and checks before replacing
                      that data, affine and every other header field are unchanged.
  regzip <file> <bak> rewrite a .nii.gz whose valid gzip stream is followed by trailing bytes: keeps the first
                      stream, checks it holds a complete image, backs the original up to <bak>, replaces.
  empty <leaf>        list derivative masks with no nonzero voxel (tsv on stdout, summary on stderr).

Run inside a job (see round2_vulcan_fixes.sh), never on the login node.
"""
import gzip
import io
import os
import shutil
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np


def _gunzip_first(path: Path):
    blob = path.read_bytes()
    d = zlib.decompressobj(wbits=31)
    raw = d.decompress(blob) + d.flush()
    return raw, d.unused_data


def _check_same(old: nib.Nifti1Image, new: nib.Nifti1Image, allow=("xyzt_units",)):
    if not np.array_equal(np.asanyarray(old.dataobj), np.asanyarray(new.dataobj)):
        raise RuntimeError("data differ")
    if not np.array_equal(old.affine, new.affine):
        raise RuntimeError("affine differ")
    ho, hn = old.header, new.header
    for k in ho.keys():
        if k in allow:
            continue
        try:
            same = np.array_equal(ho[k], hn[k], equal_nan=True)   # scl_slope/inter are often NaN
        except TypeError:                                         # non-numeric fields (descrip, magic)
            same = np.array_equal(ho[k], hn[k])
        if not same:
            raise RuntimeError(f"header field {k} differs: {ho[k]} vs {hn[k]}")


def fix_units(leaf: Path):
    n_seen = n_fixed = 0
    by_kind = {}
    for f in sorted(leaf.rglob("*.nii.gz")):
        n_seen += 1
        raw, tail = _gunzip_first(f)
        if tail:
            print(f"SKIP trailing data: {f}", file=sys.stderr)
            continue
        if int.from_bytes(raw[0:4], "little") != 348 and int.from_bytes(raw[0:4], "big") != 348:
            print(f"SKIP not NIfTI-1: {f}", file=sys.stderr)
            continue
        b = raw[123]
        if b & 0x07:          # spatial unit already set
            continue
        new_raw = bytearray(raw)
        new_raw[123] = (b & ~0x07) | 0x02   # NIFTI_UNITS_MM, time bits kept
        tmp = f.with_name(".unitfix_tmp_" + f.name)   # must end in .nii.gz for nibabel
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            fh.write(bytes(new_raw))
        old, new = nib.load(str(f)), nib.load(str(tmp))
        _check_same(old, new)
        if new.header.get_xyzt_units()[0] != "mm":
            raise RuntimeError(f"unit not set: {tmp}")
        os.replace(tmp, f)
        n_fixed += 1
        kind = "derivatives" if "derivatives" in f.parts else "raw"
        by_kind[kind] = by_kind.get(kind, 0) + 1
        print(f"FIXED\t{f.relative_to(leaf)}")
    print(f"units: {n_seen} files seen, {n_fixed} fixed {by_kind}", file=sys.stderr)


def regzip(f: Path, bak_dir: Path):
    raw, tail = _gunzip_first(f)
    magic = tail[:2] == b"\x1f\x8b"
    print(f"{f}: first stream {len(raw)} B decompressed, trailing {len(tail)} B, "
          f"trailing starts with gzip magic: {magic}", file=sys.stderr)
    if not tail:
        print("nothing to do", file=sys.stderr)
        return
    hdr = nib.Nifti1Header.from_fileobj(io.BytesIO(raw[:348]))
    need = int(hdr["vox_offset"]) + int(np.prod(hdr.get_data_shape())) * hdr.get_data_dtype().itemsize
    if len(raw) < need:
        raise RuntimeError(f"first stream too short: {len(raw)} < {need}")
    print(f"image needs {need} B, first stream has {len(raw)} B: complete", file=sys.stderr)
    bak_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, bak_dir / f.name)
    tmp = f.with_name(".regzip_tmp_" + f.name)
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(raw)
    old, new = nib.load(str(bak_dir / f.name)), nib.load(str(tmp))
    _check_same(old, new, allow=())
    raw2, tail2 = _gunzip_first(tmp)
    assert raw2 == raw and not tail2
    os.replace(tmp, f)
    print(f"rewritten: {f} ({f.stat().st_size} B), original in {bak_dir}", file=sys.stderr)


def empty_masks(leaf: Path):
    counts = {}
    total = 0
    for f in sorted((leaf / "derivatives").rglob("*.nii.gz")):
        total += 1
        name = f.name.split(".nii")[0]
        mod = name.split("_")[1]
        lab = [p for p in name.split("_") if p.startswith("label-")][0][6:]
        key = (mod, lab)
        c = counts.setdefault(key, [0, 0])
        c[1] += 1
        if not np.any(np.asanyarray(nib.load(str(f)).dataobj)):
            c[0] += 1
            print(f"{f.relative_to(leaf)}")
    print(f"empty masks: {sum(v[0] for v in counts.values())} of {total}", file=sys.stderr)
    for (mod, lab), (e, n) in sorted(counts.items()):
        print(f"  {mod}\t{lab}\t{e}/{n} empty", file=sys.stderr)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "units":
        fix_units(Path(sys.argv[2]))
    elif cmd == "regzip":
        regzip(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "empty":
        empty_masks(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
