#!/usr/bin/env python
"""Functional check of the ssem50 config (SynthSeg-EM + Auglab, SynthSeg p=0.5), CPU, on a REAL training batch.

Builds the trainer exactly as training does (nnunetv2 get_trainer_from_args -> the dataset's AugLabDefault trainer class -> nnUNetTrainerDAExtGPU.__init__ ->
AugTransformsGPU(json_path=$AUGLAB_PARAMS_GPU_JSON, num_labels=...)), pulls real (data, target) batches from the trainer's
own nnU-Net training dataloader (fold 0, CPU-side transforms as in training), and applies trainer.transforms N times.

Detectors of "this sample's image was replaced by SynthSeg":
  D1 (all configs): the SynthSeg child's own kornia params of that forward (`batch_prob` per sample, read from the
      sequential's `_params`).
  D2 (SynthSeg-only configs, everything else removed): out[i] differs from in[i] (torch.equal). With no other operator in
      the pipeline, a changed image can only come from SynthSeg, so D2 is ground truth for D1; D1 must agree with D2
      sample by sample (that is the proof that D1 responds).
Arms: SynthSeg-only at p = 0 / 0.5 / 1 (D1 + D2 + agreement); full ssem50 config as is (p 0.5), and with SynthSeg p
forced to 0 and 1 (D1); PA headline config (D1 on its PALETTE child, same pipeline slot) for comparison. Also prints the
instantiated child list (class, p) of the ssem50 and PA pipelines: identical except the synthesis class.
  python functional_check.py --cfg-dir <AugLab configs> --n <forwards per arm> --out <json>
"""
import argparse, copy, json, os, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cfg-dir", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--trainer", default="nnUNetTrainerOpenMSAugLabDefault")
ap.add_argument("--dataset", default="070")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
CFG = Path(a.cfg_dir)
NEW = CFG / "transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json"
PA = CFG / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
tmpd = Path(tempfile.mkdtemp(dir=os.environ.get("SLURM_TMPDIR") or os.environ["SCRATCH"]))

def variant(base, p=None, only_synthseg=False, name="x"):
    cfg = json.loads(Path(base).read_text())
    if only_synthseg:
        cfg = {"SynthSeg": cfg["SynthSeg"]}
    if p is not None:
        cfg["SynthSeg"]["probability"] = p
    f = tmpd / f"{name}.json"; f.write_text(json.dumps(cfg)); return str(f)

# ---- trainer, built as in training ------------------------------------------------------------------------------------
os.environ["AUGLAB_PARAMS_GPU_JSON"] = str(NEW)
from nnunetv2.run.run_training import get_trainer_from_args   # the exact entry point nnUNetv2_train uses
t0 = time.time()
tr = get_trainer_from_args(a.dataset, "3d_fullres", 0, a.trainer, "nnUNetPlans", False, device=torch.device("cpu"))
tr.initialize()
dl_tr, _ = tr.get_dataloaders()
batches = []
for _ in range(3):
    b = next(dl_tr)
    batches.append((b["data"].float(), b["target"] if not isinstance(b["target"], list) else b["target"][0]))
print(f"[setup] trainer {a.trainer} built + 3 real batches in {time.time()-t0:.0f}s; data {tuple(batches[0][0].shape)} "
      f"target {tuple(batches[0][1].shape)} labels {sorted(torch.unique(batches[0][1]).tolist())}", flush=True)
lm = tr.label_manager
n_fg = len(lm.foreground_regions) if lm.has_regions else len(lm.foreground_labels)
from auglab.transforms.gpu.transforms import AugTransformsGPU

def children(seq):
    return [(n, type(m).__name__, float(getattr(m, "p", float("nan")))) for n, m in seq.named_children()]

def synth_child_name(seq, cls):
    names = [n for n, m in seq.named_children() if type(m).__name__ == cls]
    assert len(names) == 1, (cls, children(seq)); return names[0]

def d1_flags(seq, name, B):
    for it in seq._params:
        if it.name == name:
            bp = it.data.get("batch_prob") if isinstance(it.data, dict) else None
            if bp is None:
                return [False] * B
            return [bool(x) for x in bp.reshape(-1).tolist()]
    raise RuntimeError(f"{name} not in _params")

def run(json_path, cls, n, d2=False):
    seq = AugTransformsGPU(json_path=json_path, num_labels=1 + n_fg)
    name = synth_child_name(seq, cls)
    d1, d2v, agree, conc = [], [], 0, 0
    k = 0
    while len(d1) < n:
        x0, y0 = batches[k % len(batches)]; k += 1
        x, y = x0.clone(), y0.clone()
        xo, yo = seq(x, y)
        f = d1_flags(seq, name, x.shape[0]); d1 += f
        conc += int(len(set(f)) == 1)
        if d2:
            # SynthSeg-only pipeline: any change of the image comes from SynthSeg (mask must stay identical)
            ch = [not torch.equal(xo[i], x0[i]) for i in range(x.shape[0])]
            d2v += ch; agree += sum(int(p == q) for p, q in zip(f, ch))
            assert torch.equal(yo, y0) or yo.shape != y0.shape, "SynthSeg-only pipeline changed the mask"
    r = {"n_samples": len(d1), "n_forwards": k, "D1_fraction": float(np.mean(d1)),
         "within_batch_concordant_forwards": conc}
    if d2:
        r.update(D2_fraction=float(np.mean(d2v)), D1_D2_agreement=agree / len(d2v))
    return r, children(seq)

res = {}
for p in (0.0, 0.5, 1.0):
    r, _ = run(variant(NEW, p, True, f"only_{p}"), "RandomSynthSegGPU", a.n, d2=True); res[f"synthseg_only_p{p}"] = r
    print(f"[synthseg-only p={p}] {r}", flush=True)
for p in (0.0, None, 1.0):
    tag = "as_is_p0.5" if p is None else f"forced_p{p}"
    r, ch = run(str(NEW) if p is None else variant(NEW, p, False, f"full_{p}"), "RandomSynthSegGPU", a.n)
    res[f"ssem50_full_{tag}"] = r; print(f"[ssem50 full {tag}] {r}", flush=True)
    if p is None:
        res["ssem50_children"] = ch
r, ch_pa = run(str(PA), "RandomV26_6_2ContrastGPU", a.n)
res["PA_full"] = r; res["PA_children"] = ch_pa; print(f"[PA full] {r}", flush=True)
# child lists: identical except the synthesis class (same slot)
A = [(c, p) for _, c, p in res["ssem50_children"]]; B = [(c, p) for _, c, p in ch_pa]
same_slot = len(A) == len(B) and all((x == y) or ({x[0], y[0]} == {"RandomSynthSegGPU", "RandomV26_6_2ContrastGPU"} and x[1] == y[1])
                                      for x, y in zip(A, B))
res["children_identical_except_synthesis_class"] = same_slot
print("\n[children] ssem50 | PA")
for x, y in zip(res["ssem50_children"], ch_pa):
    print(f"   {x[1]:<40} p={x[2]:<6} | {y[1]:<40} p={y[2]}")
print(f"[children] identical except synthesis class (same slot, same p): {same_slot}")
Path(a.out).write_text(json.dumps(res, indent=1)); print(f"[done] {a.out} in {time.time()-t0:.0f}s")
