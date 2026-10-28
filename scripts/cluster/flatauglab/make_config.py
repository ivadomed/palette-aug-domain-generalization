#!/usr/bin/env python
"""Make + prove the AugLab configs of the "flat fill + Auglab" arm (flatauglab, 2026-10-28).

Arm = the HEADLINE PA+Auglab training with PA's real fill replaced by the FLAT fill of the paper's texture ablation (ladder rung 4.5),
everything else identical. Two configs (AugLab auglab/configs/):
  TRAIN  transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json
         = HEADLINE transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json (md5 b5015cfe..., the file
           saved in every trained headline PA+Auglab run) with the block "ImageContrastV26_6_2GPUTransform" renamed to
           "ImageContrastV26_6_2NoiseFillGPUTransform" (the operator class the rung-4.5 flat config uses) and the three flat-fill keys of
           rung 4.5 (label_fill_noise false, label_voronoi false, noise_std_range [0, 0]) added. Every other block byte-identical.
           AugLab builds the chain in a FIXED code order (transforms.py _build_transforms): the NoiseFill slot is the very next one after
           the PA slot, nothing between them, so the operator sits at the same position in the chain.
  VAL    transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json   (synthetic validation of the DualVal trainers,
         only used for the checkpoint_best_val100 mirror, never for the val000 checkpoint the arm reports)
         = headline VAL config transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json with the same edit.
Proofs (exit 1 on any failure): (a) text diff = exactly the renamed key line + the label_classes line gaining a comma + 3 added lines;
(b) semantic: same top-level blocks except the rename, every other block deep-equal; (c) the new flat block is deep-equal to the
rung-4.5 flat block of transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json (TRAIN: exactly; VAL:
except probability 1.0, which is the headline VAL config's PA probability); (d) the renamed block minus the three flat keys equals the
headline PA block.
  python scripts/cluster/flatauglab/make_config.py [--write]     (--write creates the two files if absent, refuses to overwrite a different file)
"""
import difflib, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "sub-workspaces/auglab_workspace/AugLab/auglab/configs"
PAIRS = [("TRAIN", "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json",
          "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json", None),
         ("VAL", "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json",
          "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json", 1.0)]
HEAD_MD5 = "b5015cfed815f28c4a2e354b6c12c609"
R45 = CFG / "transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json"
OLDK, NEWK = "ImageContrastV26_6_2GPUTransform", "ImageContrastV26_6_2NoiseFillGPUTransform"
FLATKEYS = {"label_fill_noise": False, "label_voronoi": False, "noise_std_range": [0.0, 0.0]}


def make(text):
    t = text.replace(f'"{OLDK}": {{', f'"{NEWK}": {{', 1)
    assert t.count(f'"{NEWK}"') == 1 and text.count(f'"{OLDK}"') == 1
    i = t.index(f'"{NEWK}"'); j = t.index('"label_classes": null', i); e = j + len('"label_classes": null')
    ind = t[t.rindex("\n", 0, j) + 1:j]
    assert t[e:].lstrip(" ").startswith("\n") and t[e:].lstrip().startswith("}"), "label_classes is not the last key of the PA block"
    add = f',\n{ind}"label_fill_noise": false,\n{ind}"label_voronoi": false,\n{ind}"noise_std_range": [0.0, 0.0]'
    return t[:e] + add + t[e:]


def main():
    ok = True; write = "--write" in sys.argv
    r45 = json.loads(R45.read_text())[NEWK]
    for tag, hn, nn, p in PAIRS:
        hp, np_ = CFG / hn, CFG / nn
        ht = hp.read_text(); want = make(ht)
        if write:
            if np_.exists() and np_.read_text() != want:
                print(f"REFUSE: {np_.name} exists with different content"); sys.exit(1)
            np_.write_text(want)
        nt = np_.read_text()
        print(f"== {tag}\nheadline {hn} md5 {hashlib.md5(ht.encode()).hexdigest()}\nnew      {nn} md5 {hashlib.md5(nt.encode()).hexdigest()}")
        if tag == "TRAIN":
            ok &= hashlib.md5(ht.encode()).hexdigest() == HEAD_MD5
        ok &= nt == want
        d = list(difflib.unified_diff(ht.splitlines(True), nt.splitlines(True), hn, nn, n=0)); sys.stdout.writelines(d)
        minus = [l[1:].strip() for l in d if l.startswith("-") and not l.startswith("---")]
        plus = [l[1:].strip() for l in d if l.startswith("+") and not l.startswith("+++")]
        a = minus == [f'"{OLDK}": {{', '"label_classes": null'] and plus == [f'"{NEWK}": {{', '"label_classes": null,',
             '"label_fill_noise": false,', '"label_voronoi": false,', '"noise_std_range": [0.0, 0.0]']
        print(f"(a) text diff = rename + 3 flat keys only: {a}"); ok &= a
        h, n = json.loads(ht), json.loads(nt)
        b = list(n) == [NEWK if k == OLDK else k for k in h] and all(n[k] == h[k] for k in h if k != OLDK)
        print(f"(b) same blocks in the same order, all except the PA block deep-equal: {b}"); ok &= b
        exp = dict(r45) if p is None else {**r45, "probability": p}
        c = n[NEWK] == exp
        print(f"(c) flat block == rung-4.5 flat block{'' if p is None else f' (probability {p})'}: {c}"); ok &= c
        dd = {k: v for k, v in n[NEWK].items() if k not in FLATKEYS} == h[OLDK] and all(n[NEWK][k] == v for k, v in FLATKEYS.items())
        print(f"(d) flat block minus {sorted(FLATKEYS)} == headline PA block: {dd}"); ok &= dd
        print(f"    PA block probability headline {h[OLDK]['probability']} / new {n[NEWK]['probability']}")
    print("ALL PROOFS PASSED" if ok else "PROOF FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
