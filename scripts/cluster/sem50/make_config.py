#!/usr/bin/env python
"""Prove the AugLab config of the "SynthSeg-EM at 50%" arm (sem50, 2026-10-25; not a paper arm).

transform_params_gpu_default01-23_Synthseg_EM_train050.json (AugLab d0f2648) was made by copying the HEADLINE SynthSeg-EM config
transform_params_gpu_default01-23_Synthseg_EM.json (md5 59f7e8c6 = the file saved in every trained headline synthseg_EM run) and
replacing its first "probability": 1.0 (the SynthSeg block) by 0.5 with sed. It is NOT derived from the ssem50 config
(..._auglabAug_Synthseg_EM_train050.json, which also switches the six Auglab operators on). This script only checks:
  (a) the text diff headline -> new is exactly one changed line, inside the SynthSeg block
  (b) the semantic diff headline -> new is exactly [SynthSeg.probability 1.0 -> 0.5]
  (c) the headline file md5 is 59f7e8c6..., the six Auglab operators (Scharr, UnsharpMask, RandomConv, RedistributeSeg, Inverse,
      HistogramEqualization) have probability 0 in both files
  python scripts/cluster/sem50/make_config.py      (run inside a Slurm job; prints ALL PROOFS PASSED or exits 1)
"""
import difflib, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "sub-workspaces/auglab_workspace/AugLab/auglab/configs"
HEAD = CFG / "transform_params_gpu_default01-23_Synthseg_EM.json"
NEW = CFG / "transform_params_gpu_default01-23_Synthseg_EM_train050.json"
SIX = ["ScharrTransform", "UnsharpMaskTransform", "RandomConvTransform", "RedistributeSegTransform", "InverseTransform",
       "HistogramEqualizationTransform"]


def flat(d, p=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + (k,))
        else:
            yield p + (k,), v


def main():
    ok = True
    ht, nt = HEAD.read_text(), NEW.read_text()
    md5h, md5n = hashlib.md5(ht.encode()).hexdigest(), hashlib.md5(nt.encode()).hexdigest()
    print(f"headline {HEAD.name} md5 {md5h}\nnew      {NEW.name} md5 {md5n}")
    ok &= md5h == "59f7e8c62e174e7dbfd7f60f385cc969"
    d = [l for l in difflib.unified_diff(ht.splitlines(True), nt.splitlines(True), HEAD.name, NEW.name, n=0)]
    sys.stdout.writelines(d)
    changed = [l for l in d if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    print(f"(a) changed text lines: {len(changed)}")
    ok &= len(changed) == 2 and changed[0][1:].strip() == '"probability": 1.0,' and changed[1][1:].strip() == '"probability": 0.5,'
    syn_end = nt.index("\n    },")   # end of the first (SynthSeg) block
    ok &= nt.startswith('{\n    "SynthSeg"') and nt.index('"probability": 0.5') < syn_end
    h, n = json.loads(ht), json.loads(nt)
    fh, fn = dict(flat(h)), dict(flat(n))
    sd = sorted((".".join(k), fh.get(k, "<absent>"), fn.get(k, "<absent>")) for k in set(fh) | set(fn)
                if fh.get(k, "<absent>") != fn.get(k, "<absent>"))
    print(f"(b) semantic diff: {sd}")
    ok &= sd == [("SynthSeg.probability", 1.0, 0.5)]
    six = {b: (h[b]["probability"], n[b]["probability"]) for b in SIX}
    print(f"(c) six Auglab operators (headline, new) probability: {six}")
    ok &= all(v == (0.0, 0.0) for v in six.values())
    print("ALL PROOFS PASSED" if ok else "PROOF FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
