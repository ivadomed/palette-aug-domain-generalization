#!/usr/bin/env python
"""Functional check of the flatauglab configs (headline PA+Auglab with PA's real fill -> rung-4.5 flat fill), CPU, on REAL training batches.

Builds the trainer exactly as training does (nnunetv2 get_trainer_from_args -> the dataset trainer -> AugTransformsGPU(json_path=
$AUGLAB_PARAMS_GPU_JSON)), pulls real (data, target) batches from its own nnU-Net training dataloader (open-ms FLAIR fold 0) and checks:
  F1 child lists: the flat pipeline and the headline PA+Auglab pipeline have the same transforms, probabilities and order, except ONE
     slot: RandomV26_6_2ContrastGPU (headline) vs RandomV26_6_2NoiseFillContrastGPU (flat), same p 0.5, same position.
  F2 the flat operator fires on ~50 % of the samples INSIDE the full Auglab chain: D1 = that child's kornia batch_prob per forward (N
     samples); forced p 0 / 1 give 0 / 1. Detector validity: a flat-operator-only pipeline at p 0.5 -- D2 = output differs from input,
     D1 must agree with D2 sample by sample. Draws are per batch (same_on_batch), as for PA.
  F3 flat regions: the flat operator instantiated FROM THE NEW CONFIG, synthesis forced, its internal blur off, on each real sample:
     output CONSTANT (zero within-cell spread) on every (step-1 PALETTE region x label) cell in every draw; discrimination: the headline
     PA operator (real fill) fails the same test. Also counts the label-step firing and the whole-foreground-constant edge case
     (skip-parcellation branch + no label step, known ~1/40 on MS). With the configured blur the cells are no longer constant (blur is
     part of the operator in both PA and flat, applied after the fill) -- reported, not tested.
  F4 VAL config (DualVal val100 mirror): same child check vs the headline VAL config, flat operator at p 1.0.
  F5 AUGLAB_ALLOW_LEGACY_NOISEFILL is unset and the operator builds (the lblvor guard only refuses label_fill_noise without label_voronoi).
  python functional_check.py --cfg-dir <AugLab configs> --n <samples> --out <json>
"""
import argparse, json, os, random, tempfile, time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cfg-dir", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--draws", type=int, default=30)
ap.add_argument("--trainer", default="nnUNetTrainerOpenMSAugLabDefault")
ap.add_argument("--dataset", default="070")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0); random.seed(0)
CFG = Path(a.cfg_dir)
NEW = CFG / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json"
HEAD = CFG / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
VNEW = CFG / "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json"
VHEAD = CFG / "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
FLATCLS, PACLS, KEY = "RandomV26_6_2NoiseFillContrastGPU", "RandomV26_6_2ContrastGPU", "ImageContrastV26_6_2NoiseFillGPUTransform"
tmpd = Path(tempfile.mkdtemp(dir=os.environ.get("SLURM_TMPDIR") or os.environ["SCRATCH"]))
FAILS = []
def check(c, msg):
    print(("  ok    " if c else "  FAIL  ") + msg, flush=True)
    if not c: FAILS.append(msg)

def variant(base, p=None, only=False, name="x"):
    cfg = json.loads(Path(base).read_text())
    if only: cfg = {KEY: cfg[KEY]}
    if p is not None: cfg[KEY]["probability"] = p
    f = tmpd / f"{name}.json"; f.write_text(json.dumps(cfg)); return str(f)

check("AUGLAB_ALLOW_LEGACY_NOISEFILL" not in os.environ, "F5 AUGLAB_ALLOW_LEGACY_NOISEFILL unset")
os.environ["AUGLAB_PARAMS_GPU_JSON"] = str(NEW)
from nnunetv2.run.run_training import get_trainer_from_args
t0 = time.time()
tr = get_trainer_from_args(a.dataset, "3d_fullres", 0, a.trainer, "nnUNetPlans", False, device=torch.device("cpu"))
tr.initialize()
dl_tr, _ = tr.get_dataloaders()
batches = []
for _ in range(3):
    b = next(dl_tr)
    batches.append((b["data"].float(), b["target"] if not isinstance(b["target"], list) else b["target"][0]))
print(f"[setup] {a.trainer} + 3 real batches in {time.time()-t0:.0f}s; data {tuple(batches[0][0].shape)} labels "
      f"{sorted(torch.unique(batches[0][1]).tolist())}", flush=True)
lm = tr.label_manager
n_fg = len(lm.foreground_regions) if lm.has_regions else len(lm.foreground_labels)
from auglab.transforms.gpu.transforms import AugTransformsGPU
import auglab.transforms.gpu.fromSeg as fs
import auglab.transforms.gpu.palette_noisefill as pn

def children(seq):
    return [(n, type(m).__name__, float(getattr(m, "p", float("nan")))) for n, m in seq.named_children()]

def child(seq, cls):
    c = [(n, m) for n, m in seq.named_children() if type(m).__name__ == cls]
    assert len(c) == 1, (cls, children(seq)); return c[0]

def d1_flags(seq, name, B):
    for it in seq._params:
        if it.name == name:
            bp = it.data.get("batch_prob") if isinstance(it.data, dict) else None
            return [False] * B if bp is None else [bool(x) for x in bp.reshape(-1).tolist()]
    raise RuntimeError(f"{name} not in _params")

res = {}
# ---- F1 / F4 child lists
def same_but_slot(A, B):
    A = [(c, p) for _, c, p in A]; B = [(c, p) for _, c, p in B]
    diff = [i for i, (x, y) in enumerate(zip(A, B)) if x != y]
    return len(A) == len(B) and len(diff) == 1 and A[diff[0]][0] == FLATCLS and B[diff[0]][0] == PACLS and A[diff[0]][1] == B[diff[0]][1], diff
print("== F1 child lists (flat | headline)")
cn, ch = children(AugTransformsGPU(json_path=str(NEW), num_labels=1 + n_fg)), children(AugTransformsGPU(json_path=str(HEAD), num_labels=1 + n_fg))
for x, y in zip(cn, ch): print(f"   {x[1]:<36} p={x[2]:<5} | {y[1]:<36} p={y[2]}")
ok, d = same_but_slot(cn, ch); res["F1_children_flat"] = cn; res["F1_children_headline"] = ch
check(ok, f"F1 identical chains except slot {d}: {FLATCLS} vs {PACLS}, same p")
print("== F4 VAL config child lists")
vn, vh = children(AugTransformsGPU(json_path=str(VNEW), num_labels=1 + n_fg)), children(AugTransformsGPU(json_path=str(VHEAD), num_labels=1 + n_fg))
ok, d = same_but_slot(vn, vh); res["F4_val_children"] = vn
check(ok and dict((c, p) for _, c, p in vn)[FLATCLS] == 1.0, f"F4 VAL chain identical to headline VAL except slot {d}; flat p 1.0")

# ---- F2 firing rate inside the full chain
def run(json_path, n, d2=False):
    seq = AugTransformsGPU(json_path=json_path, num_labels=1 + n_fg); name, _ = child(seq, FLATCLS)
    d1, d2v, agree, conc, k = [], [], 0, 0, 0
    while len(d1) < n:
        x0, y0 = batches[k % len(batches)]; k += 1
        xo, yo = seq(x0.clone(), y0.clone())
        f = d1_flags(seq, name, x0.shape[0]); d1 += f; conc += int(len(set(f)) == 1)
        if d2:
            c = [not torch.equal(xo[i], x0[i]) for i in range(x0.shape[0])]; d2v += c; agree += sum(int(p == q) for p, q in zip(f, c))
    r = {"n_samples": len(d1), "n_forwards": k, "D1_fraction": float(np.mean(d1)), "within_batch_concordant_forwards": conc}
    if d2: r.update(D2_fraction=float(np.mean(d2v)), D1_D2_agreement=agree / len(d2v))
    return r
print("== F2 firing rate")
for p in (0.0, 0.5, 1.0):
    r = run(variant(NEW, p, True, f"only_{p}"), a.n, d2=True); res[f"F2_flat_only_p{p}"] = r; print(f"  [flat-only p={p}] {r}", flush=True)
    check(abs(r["D2_fraction"] - p) < 0.15 and r["D1_D2_agreement"] == 1.0, f"F2 detector: flat-only p={p} D2 {r['D2_fraction']:.2f}, D1==D2")
for p in (0.0, None, 1.0):
    tag = "as_is" if p is None else f"forced_p{p}"
    r = run(str(NEW) if p is None else variant(NEW, p, False, f"full_{p}"), a.n); res[f"F2_full_{tag}"] = r; print(f"  [full chain {tag}] {r}", flush=True)
    want = 0.5 if p is None else p
    check(abs(r["D1_fraction"] - want) < (0.12 if p is None else 1e-9), f"F2 full chain {tag}: flat operator fired on {r['D1_fraction']:.2f} of samples (want {want})")

# ---- F3 flat regions (operator from the NEW config vs headline PA operator)
class Recorder:
    def __init__(self, mod): self.mod, self.orig, self.rid = mod, mod._voronoi_region_ids, None
    def __enter__(self):
        def w(*x, **k):
            r = self.orig(*x, **k)
            if self.rid is None: self.rid = r[0].clone()
            return r
        self.mod._voronoi_region_ids = w; return self
    def __exit__(self, *e): self.mod._voronoi_region_ids = self.orig

def cell_test(t, mod, img, lbl, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    with Recorder(mod) as rec:
        out = t.apply_transform(img.clone(), {"seg": lbl}, {})
    o = out[0, 0].reshape(-1).double(); fl = img[0, 0].reshape(-1).float()
    v01 = (fl - fl.min()) / (fl.max() - fl.min() + 1e-7); fg = v01 > t.dark_threshold
    rid = rec.rid if rec.rid is not None else torch.zeros_like(fl, dtype=torch.long)
    l = lbl[0, 0].reshape(-1).long(); key = (rid.long() * 64 + l)[fg]; vals = o[fg]
    uk, inv = torch.unique(key, return_inverse=True)
    mx = torch.full((len(uk),), -1e30, dtype=torch.float64).scatter_reduce(0, inv, vals, "amax")
    mn = torch.full((len(uk),), 1e30, dtype=torch.float64).scatter_reduce(0, inv, vals, "amin")
    var = torch.zeros(len(uk), dtype=torch.float64).scatter_add(0, inv, vals ** 2) / torch.bincount(inv).double() - \
          (torch.zeros(len(uk), dtype=torch.float64).scatter_add(0, inv, vals) / torch.bincount(inv).double()) ** 2
    r_fg, l_fg = rid[fg].long(), l[fg]; fired = False
    for r in torch.unique(r_fg[l_fg > 0])[:200]:
        inr = r_fg == r; x, y = vals[inr & (l_fg > 0)], vals[inr & (l_fg == 0)]
        if len(x) and len(y) and abs(float(x.mean() - y.mean())) > 1e-6: fired = True; break
    return float((mx - mn).max()), float(var.clamp(min=0).max()), fired, float(vals.max() - vals.min()) < 1e-9

print("== F3 flat regions (synthesis forced)")
fseq, hseq = AugTransformsGPU(json_path=str(NEW), num_labels=1 + n_fg), AugTransformsGPU(json_path=str(HEAD), num_labels=1 + n_fg)
_, ft = child(fseq, FLATCLS); _, ht = child(hseq, PACLS)
check(isinstance(ft, pn.RandomV26_6_2NoiseFillContrastGPU) and list(ft.noise_std_range) == [0.0, 0.0] and ft.label_fill_noise is False
      and ft.label_voronoi is False and float(ft.p) == 0.5, f"F3 operator from the new config: noise_std_range {ft.noise_std_range}, "
      f"label_fill_noise {ft.label_fill_noise}, label_voronoi {ft.label_voronoi}, p {float(ft.p)}")
for att in ("c_choices", "s_choices", "blur_sigmas", "dark_threshold", "n_kmeans_subsample", "skip_parcellation_prob", "skip_sub_parc_prob",
            "alpha_magnitude_range", "label_remap_prob", "min_label_voxels", "label_classes"):
    check(getattr(ft, att) == getattr(ht, att), f"F3 flat.{att} == headline PA.{att} ({getattr(ft, att)})")
blur_cfg = list(ft.blur_sigmas)
samples = [(x[i:i + 1], y[i:i + 1].long()) for x, y in batches for i in range(x.shape[0])]
out = {}
for blur_tag, bl in (("blur_off", [0.0]), ("blur_as_configured", blur_cfg)):
    ft.blur_sigmas = bl; ht.blur_sigmas = bl
    for tag, t, mod in (("flat", ft, pn), ("headline_PA", ht, fs)):
        rs = [cell_test(t, mod, img, lbl, 100 * si + s) for si, (img, lbl) in enumerate(samples) for s in range(a.draws // len(samples) + 1)]
        sp, vr, fi, ac = zip(*rs)
        out[f"{tag}_{blur_tag}"] = dict(draws=len(rs), max_within_cell_spread=max(sp), max_within_cell_variance=max(vr),
                                       n_nonconstant=sum(x > 1e-6 for x in sp), label_step_fired=sum(fi), whole_fg_constant=sum(ac))
        print(f"  {tag:12s} {blur_tag:19s} {out[f'{tag}_{blur_tag}']}", flush=True)
ft.blur_sigmas = blur_cfg; ht.blur_sigmas = blur_cfg
res["F3"] = out
check(out["flat_blur_off"]["max_within_cell_spread"] < 1e-6, f"F3 flat (blur off): zero within-region variance in all {out['flat_blur_off']['draws']} draws")
check(out["flat_blur_off"]["label_step_fired"] > 0, "F3 flat: PALETTE's label step fires")
check(out["headline_PA_blur_off"]["n_nonconstant"] >= 0.8 * out["headline_PA_blur_off"]["draws"], "F3 discrimination: headline PA (real fill) fails the invariant")
res["FAILS"] = FAILS
Path(a.out).write_text(json.dumps(res, indent=1, default=str))
print(f"[done] {a.out} in {time.time()-t0:.0f}s"); print("RESULT:", "ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
