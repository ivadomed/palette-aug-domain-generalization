#!/usr/bin/env python
"""PROOF (no write) of the AugLab config of the "PA_EM + Auglab" arm (pa_em, 2026-10-22). The config itself is AugLab e0d8422.

  EM  = transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2EM_GPUTransform_train050.json
  PA  = transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json   (headline PA + Auglab)
  SS  = transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json                       (the ssem50 arm)
Asserts: (a) EM vs PA: the ONLY differences are ImageContrastV26_6_2GPUTransform.partition = "em" and .em_n_iters = 20 (both absent in PA:
the transform's default is partition="kmeans"); (b) every non-synthesis block of EM == SS's (so pa_em and ssem50 differ only in the
synthesis block: PA-with-EM-partition at p 0.5 vs SynthSeg-EM at p 0.5); (c) synthesis probability EM == PA == SS == 0.5. Prints md5s.
  python scripts/cluster/pa_em/make_config.py
"""
import difflib, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "sub-workspaces/auglab_workspace/AugLab/auglab/configs"
EM = CFG / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2EM_GPUTransform_train050.json"
PA = CFG / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
SS = CFG / "transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json"
SYN = {"SynthSeg", "ImageContrastV26_6_2GPUTransform"}


def flat(d, p=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + (k,))
        else:
            yield p + (k,), v


def diff(a, b):
    fa, fb = dict(flat(a)), dict(flat(b))
    return sorted((".".join(k), fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in set(fa) | set(fb)
                  if fa.get(k, "<absent>") != fb.get(k, "<absent>"))


em, pa, ss = (json.loads(p.read_text()) for p in (EM, PA, SS))
for p in (EM, PA, SS):
    print(f"md5 {hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}")
print("\n== unified text diff PA -> EM ==")
sys.stdout.writelines(difflib.unified_diff(PA.read_text().splitlines(True), EM.read_text().splitlines(True), PA.name, EM.name, n=1))
ok = True
d = diff(pa, em); print(f"\n(a) semantic diff PA -> EM: {d}")
ok &= d == [("ImageContrastV26_6_2GPUTransform.em_n_iters", "<absent>", 20), ("ImageContrastV26_6_2GPUTransform.partition", "<absent>", "em")]
d = diff({k: v for k, v in ss.items() if k not in SYN}, {k: v for k, v in em.items() if k not in SYN})
print(f"(b) non-synthesis blocks SS -> EM: same block set {set(ss) - SYN == set(em) - SYN}, semantic diff {d}")
ok &= d == [] and set(ss) - SYN == set(em) - SYN and set(em) & SYN == {"ImageContrastV26_6_2GPUTransform"} and set(ss) & SYN == {"SynthSeg"}
pr = (em["ImageContrastV26_6_2GPUTransform"]["probability"], pa["ImageContrastV26_6_2GPUTransform"]["probability"], ss["SynthSeg"]["probability"])
print(f"(c) synthesis probability EM / PA / SS: {pr}")
ok &= pr == (0.5, 0.5, 0.5)
print("ALL PROOFS PASSED" if ok else "PROOF FAILED"); sys.exit(0 if ok else 1)
