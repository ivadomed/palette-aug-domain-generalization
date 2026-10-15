#!/usr/bin/env python
"""
texture_rule_hand.py -- the xds_full_hand.py hand features (same unit(), pooled foreground label, 1 mm resampling, <= 20 cases)
for the contrast keys the 2026-10-02/07 extraction never covered (pancreas, pelvis, HCC, amos, sliver07), so the frozen percentile
rules (frozen_rule_*.json) can be scored on every cell of the real-vs-flat table (FINDINGS 48). Features only, no outcomes.
AMOS labels are first remapped to the CHAOS 4-organ set (amos ids 6/2/3/1 -> 1/2/3/4, 06_00_evaluate_amos.py), written to
--tmp, so the pooled target matches what the chaos models are scored on.
  python texture_rule_hand.py --nproc 32 --tmp <dir>   ->  outputs/data/xds_full_hand_extra.csv
"""
from __future__ import annotations
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import xds_full_hand as H  # noqa: E402
import texture_rule_polarity as T  # noqa: E402
KEYS = ["pansegdata:t1wce", "pansegdata:t2w", "totalsegmri-pancreas:t1gre", "totalsegmri-pancreas:t2like", "msd-pancreas:ct",
        "totalseg-pelvic:ct", "totalseg-pelvic:mri", "atlas:t1w", "lld-mmri-hcc:t2wi", "lld-mmri-hcc:dwi", "lld-mmri-hcc:inphase",
        "lld-mmri-hcc:outphase", "amos:ct", "amos:mri", "sliver07:ct"]
MAXC = 20


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--tmp", required=True); a = ap.parse_args()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True); man = T.manifest(); units = []
    for k in KEYS:
        mode, fn = man[k]
        for c, i, l in fn()[:MAXC]:
            if mode == "amos":
                lp = tmp / f"{k.replace(':', '_')}_{c}.nii.gz"
                if not lp.exists():
                    im = nib.load(str(l)); lb = np.asarray(im.dataobj).round().astype(int)
                    h = im.header.copy(); h.set_data_dtype(np.uint8); h.set_slope_inter(1, 0)
                    nib.save(nib.Nifti1Image(sum((lb == s) * d for s, d in T.AMOS_MAP.items()).astype(np.uint8), im.affine, h), str(lp))
                l = lp
            units.append((k, c, i, l, mode is True))
    out = T.DATA / "xds_full_hand_extra.csv"
    if out.exists():
        print("exists, nothing to do:", out); return
    part = T.DATA / "xds_full_hand_extra.partial.csv"   # appended per unit, so a killed job resumes
    done = set(map(tuple, pd.read_csv(part, usecols=["key", "case"]).astype(str).values)) if part.exists() else set()
    todo = [u for u in units if (u[0], str(u[1])) not in done]
    print(len(units), "units,", len(todo), "todo", flush=True)
    with Pool(a.nproc) as pool:
        for (uid, rs) in pool.imap_unordered(H.unit, todo, chunksize=1):
            if rs:
                pd.DataFrame(rs).to_csv(part, mode="a", header=not part.exists(), index=False)
            print("done", uid, flush=True)
    part.rename(out); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
