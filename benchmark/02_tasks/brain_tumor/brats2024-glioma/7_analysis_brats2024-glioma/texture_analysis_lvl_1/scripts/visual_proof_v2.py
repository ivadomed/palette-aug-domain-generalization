#!/usr/bin/env python
"""
Visual proof v2 (Paul's instructions, 2026-10-11): UNMASKED HiResCAM maps of the flat-fill (rung 4.5) and real-fill (rung 5) models on
the same test images, 3 folds per arm (fold-MEAN displayed; per-fold maps + a fold-agreement map saved), per-LESION statistics shown as
distributions, and a pelvis CT cell as the interface-defined control.

Per case and arm, per fold k (checkpoint_best): (1) standard sliding-window prediction (no mirroring) -> the arm's 3-fold MAJORITY
prediction; (2) HiResCAM on decoder stages[-3] (1/4-res, "s4") and stages[-4] (1/8-res, "s8") accumulated over the same sliding-window
tiles (uniform tile average), target = sum over the fold's OWN predicted target voxels in the tile of (logit_target - logit_bg); the
full-res stage is dropped (its map equals the target set: 1x1x1 seg head). Each fold map is normalised to unit mass over the volume,
the fold-mean is the displayed map, the agreement map = 1 - CV across folds (where the mean > 0).
Lesions = connected components (26-conn) of the UNION of the two arms' majority predictions (>= min_cc voxels), so both arms are read on
the same voxel sets without any GT. Per lesion x arm x stage: interior share (mass in the component's interior, d >= 1.5 vox, over the
mass within the component dilated by 3), border share (|d| <= 1), outside share, interface share (border-band mass on voxels within 2 vox
of an intensity-defined interface mask -- CSF bright on T2w / dark on T1w, T1n, FLAIR; bone bright on CT -- over all border mass) with
its voxel-fraction baseline, GT/flat/real overlap of the component, and the same shares on a sham component shifted 10 vox in-plane.
GT only enters as contour + overlap columns. Outputs under outputs/plots/visual_proof_v2/:
  <cell>/<case>_s4.png, _s8.png  sheets: 3 slices x [image+GT | flat pred+heat | real pred+heat | real-flat]
  <cell>/<case>_maps.npz           fold-mean maps (full res, float16), per-fold maps at 1/4 res, agreement maps, majority preds
  <cell>/<case>_lesions.csv        per-lesion rows;  lesions_all.csv, lesion_stats.md, violin_*.png, scatter_*.png from --stats-only
GPU: .venv/bin/python visual_proof_v2.py --cells openms_t2w,openms_t1w      CPU: --stats-only
Env overrides (TamIA keeps bulk data on $SCRATCH): VP_OPENMS_ROOT, VP_PELVIS_ROOT, VP_BRATS_RAW, VP_BRATS_PRED.
BraTS uses the rung-4.5 FLAT-fill run too (brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190652, 3 folds
present), not rung 4 lblvor, so all cells compare the same two rungs.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"
OUT = Path(os.environ.get("VP_OUT") or (THIS.parent / "outputs/plots/visual_proof_v2")); OUT.mkdir(parents=True, exist_ok=True)   # VP_OUT: e.g. .../visual_proof_v2_brats_full
os.environ.setdefault("nnUNet_compile", "false"); os.environ.setdefault("NNUNET_PROJECT_ROOT", str(REPO))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor  # noqa: E402
from acvl_utils.cropping_and_padding.padding import pad_nd_image  # noqa: E402

O = Path(os.environ.get("VP_OPENMS_ROOT", T2 / "brain_ms/open-ms")); RO = O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
PO = O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet"
P = Path(os.environ.get("VP_PELVIS_ROOT", T2 / "pelvis_healthy/totalseg-pelvic")); RPC = P / "2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT"
PP = P / "8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab"
RB = Path(os.environ.get("VP_BRATS_RAW", T2 / "brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w"))
PB = Path(os.environ.get("VP_BRATS_PRED", T2 / "brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab"))
OM = {"flat": PO / "open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190831/Dataset070_OpenMS_FLAIR/nnUNetTrainerOpenMSAugLabDefault__nnUNetPlans__3d_fullres",
      "real": PO / "open-ms_flair_v26_6_2_train050_val000_20260711_062946/Dataset070_OpenMS_FLAIR/nnUNetTrainerOpenMSAugLabDefault__nnUNetPlans__3d_fullres"}
BM = {"flat": PB / "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190652/Dataset052_BraTS2024GliomaT2w/nnUNetTrainerBraTS2024GliomaT2wAugLabDefault__nnUNetPlans__3d_fullres",
      "real": PB / "brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300/Dataset052_BraTS2024GliomaT2w/nnUNetTrainerBraTS2024GliomaT2wAugLabDefault__nnUNetPlans__3d_fullres"}
PB1 = Path(os.environ.get("VP_BRATS_PRED_T1N", T2 / "brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t1n/auglab"))
BM1 = {"flat": PB1 / "brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190618/Dataset051_BraTS2024GliomaT1n/nnUNetTrainerBraTS2024GliomaAugLabDefault__nnUNetPlans__3d_fullres",
       "real": PB1 / "brats2024-glioma_t1n_v26_6_2_train050_val000_20261005_222227/Dataset051_BraTS2024GliomaT1n/nnUNetTrainerBraTS2024GliomaAugLabDefault__nnUNetPlans__3d_fullres"}
BRATS_ALL = sorted(p.name.replace("_0000.nii.gz", "") for p in (RB / "imagesTs_t1n").glob("*_0000.nii.gz"))   # the 70 held-out cases (test images + labelsTr are voxel-identical across Dataset051/052)
PM = {"flat": PP / "totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257/Dataset131_TotalsegPelvic_MRI/nnUNetTrainerTotalsegPelvicAugLabDefault__nnUNetPlans__3d_fullres",
      "real": PP / "totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058/Dataset131_TotalsegPelvic_MRI/nnUNetTrainerTotalsegPelvicAugLabDefault__nnUNetPlans__3d_fullres"}
OPENMS_CASES = ["patient10", "patient12", "patient21", "patient22", "patient24", "patient27", "patient28", "patient29"]
CELLS = {
    "openms_t2w": dict(title="Open-MS FLAIR-trained -> T2w, lesion (texture HELPS, +10.6 Dice flat->real)", models=OM, cases=OPENMS_CASES, lids={1: "lesion"}, min_cc=10,
                       img=lambda c: RO / "imagesTs_t2w" / f"{c}_0000.nii.gz", gt=lambda c: RO / "labelsTs_t2w" / f"{c}.nii.gz", interface="bright", iface_name="CSF (bright on T2w)"),
    "openms_t1w": dict(title="Open-MS FLAIR-trained -> T1w, lesion (texture HELPS, +3.6 Dice flat->real)", models=OM, cases=OPENMS_CASES, lids={1: "lesion"}, min_cc=10,
                       img=lambda c: RO / "imagesTs_t1w" / f"{c}_0000.nii.gz", gt=lambda c: RO / "labelsTs_t1w" / f"{c}.nii.gz", interface="dark", iface_name="CSF (dark on T1w)"),
    "brats_t1n": dict(title="BraTS T2w-trained -> T1n, edema (3 largest fold-0 losses + 2 largest gains flat->real)", models=BM, lids={2: "edema"}, min_cc=50,
                      cases=["BraTSGLI00008100", "BraTSGLI02238102", "BraTSGLI02108101", "BraTSGLI02335101", "BraTSGLI02178102"],
                      img=lambda c: RB / "imagesTs_t1n" / f"{c}_0000.nii.gz", gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz", interface="dark", iface_name="CSF (dark on T1n)"),
    "brats_t2f": dict(title="BraTS T2w-trained -> FLAIR, edema (3 largest fold-0 gains + 2 largest losses flat->real)", models=BM, lids={2: "edema"}, min_cc=50,
                      cases=["BraTSGLI02205101", "BraTSGLI02195101", "BraTSGLI02236100", "BraTSGLI02292100", "BraTSGLI02078102"],
                      img=lambda c: RB / "imagesTs_t2f" / f"{c}_0000.nii.gz", gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz", interface="dark", iface_name="CSF (dark on FLAIR)"),
    "bratsfull_t2w_t1n": dict(title="BraTS T2w-trained -> T1n, edema, ALL 70 test cases (texture HURTS in the 4->5 ladder)", models=BM, cases=BRATS_ALL, lids={2: "edema"}, min_cc=50,
                              img=lambda c: RB / "imagesTs_t1n" / f"{c}_0000.nii.gz", gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz", interface="dark", iface_name="CSF (dark on T1n)"),
    "bratsfull_t2w_t2f": dict(title="BraTS T2w-trained -> FLAIR, edema, ALL 70 test cases (texture HELPS)", models=BM, cases=BRATS_ALL, lids={2: "edema"}, min_cc=50,
                              img=lambda c: RB / "imagesTs_t2f" / f"{c}_0000.nii.gz", gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz", interface="dark", iface_name="CSF (dark on FLAIR)"),
    "bratsfull_t1n_t2f": dict(title="BraTS T1n-trained -> FLAIR, edema, ALL 70 test cases (largest texture gain)", models=BM1, cases=BRATS_ALL, lids={2: "edema"}, min_cc=50,
                              img=lambda c: RB / "imagesTs_t2f" / f"{c}_0000.nii.gz", gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz", interface="dark", iface_name="CSF (dark on FLAIR)"),
    "bratsfull_t1n_t2w": dict(title="BraTS T1n-trained -> T2w, edema, ALL 70 test cases (texture HELPS)", models=BM1, cases=BRATS_ALL, lids={2: "edema"}, min_cc=50,
                              img=lambda c: RB / "imagesTs_t2w" / f"{c}_0000.nii.gz", gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz", interface="bright", iface_name="CSF (bright on T2w)"),
    "pelvis_ct": dict(title="Pelvis MRI-trained -> CT, hips + gluteus maximus (interface-defined CONTROL)", models=PM, cases=["s0015", "s0072", "s0193"], min_cc=500,
                      lids={1: "hip_left", 2: "hip_right", 4: "gluteus_maximus_left", 5: "gluteus_maximus_right"},
                      img=lambda c: RPC / "imagesTs_ct" / f"{c}_0000.nii.gz", gt=lambda c: RPC / "labelsTs_ct" / f"{c}.nii.gz", interface="bright", iface_name="bone (bright on CT)"),
}
ARM_LABEL = {"flat": "flat-fill model (rung 4.5)", "real": "real-fill model (rung 5)"}; STAGES = {"s4": -3, "s8": -4}
FOLDS = (0, 1, 2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_predictor(model_dir: Path, dev):
    pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True, device=dev, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pr.initialize_from_trained_model_folder(str(model_dir), use_folds=FOLDS, checkpoint_name="checkpoint_best.pth")
    pr.network.to(dev).eval(); return pr


def preprocess(pr, img_path: Path, gt_path: Path):
    rw = pr.plans_manager.image_reader_writer_class(); data, props = rw.read_images([str(img_path)]); seg, _ = rw.read_seg(str(gt_path))
    data, seg, _ = DefaultPreprocessor(verbose=False).run_case_npy(data, seg, props, pr.plans_manager, pr.configuration_manager, pr.dataset_json)
    return data.astype(np.float32), seg[0].astype(np.int16)


def fold_predict_and_attribute(pr, k: int, data: np.ndarray, lids, dev):
    """Fold k of the loaded predictor: full-volume argmax prediction + HiResCAM maps (stages s4/s8, upsampled to full res, tile-averaged)."""
    net = pr.network; net.load_state_dict(pr.list_of_parameters[k]); net.eval()
    x = torch.from_numpy(data)
    logits = pr.predict_sliding_window_return_logits(x)            # (C, Z, Y, X) on device, no grad, gaussian-weighted tiles
    pred = logits.argmax(0).to(torch.int16).cpu().numpy(); del logits
    xp, revert = pad_nd_image(x, pr.configuration_manager.patch_size, "constant", {"value": 0}, True, None)
    predp = torch.from_numpy(np.pad(pred, [(s.start, d - s.stop) for s, d in zip(revert[1:], xp.shape[1:])])).long()
    slicers = pr._internal_get_sliding_window_slicers(xp.shape[1:])
    acc = {s: torch.zeros(xp.shape[1:], device=dev) for s in STAGES}; cnt = torch.zeros(xp.shape[1:], device=dev); acc_g = torch.zeros(xp.shape[1:], device=dev)
    taps = {s: net.decoder.stages[i] for s, i in STAGES.items()}
    lid_t = torch.tensor(list(lids), device=dev)
    for sl in slicers:
        tile = xp[sl][None].to(dev).float(); M = torch.isin(predp[sl[1:]].to(dev), lid_t)
        cnt[sl[1:]] += 1
        if not M.any():
            continue
        acts, grads, hs = {}, {}, []
        for s, m in taps.items():
            def fh(mod, inp, out, s=s):
                acts[s] = out; out.register_hook(lambda g, s=s: grads.__setitem__(s, g))
            hs.append(m.register_forward_hook(fh))
        tile.requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            out = net(tile); out = out[0] if isinstance(out, (list, tuple)) else out
            score = 0.
            for lid in lids:
                Ml = M & (predp[sl[1:]].to(dev) == lid)
                if Ml.any():
                    score = score + (out[0, lid] - out[0, 0])[Ml].sum()
            score.backward()
        for h in hs:
            h.remove()
        for s in STAGES:
            mp = F.relu((acts[s].detach().float() * grads[s].detach().float()).sum(1, keepdim=True))
            mp = F.interpolate(mp, size=tile.shape[2:], mode="trilinear", align_corners=False)[0, 0]
            acc[s][sl[1:]] += mp
        acc_g[sl[1:]] += tile.grad[0, 0].detach().float()
        del acts, grads, out, tile
    maps = {}
    for s in STAGES:
        m = (acc[s] / cnt.clamp(min=1))[revert[1:]].cpu().numpy().astype(np.float32); tot = m.sum()
        maps[s] = m / tot if tot > 0 else m
    maps["grad"] = (acc_g / cnt.clamp(min=1))[revert[1:]].cpu().numpy().astype(np.float32)   # signed input gradient, tile-averaged (NOT normalised)
    torch.cuda.empty_cache()
    return pred, maps


def block_mean(a, f=4):
    z, y, x = [(n // f) * f for n in a.shape]; a = a[:z, :y, :x]
    return a.reshape(z // f, f, y // f, f, x // f, f).mean((1, 3, 5)).astype(np.float16)


def interface_mask(img, body, mode):
    v = img[body]
    if mode == "bright":
        return body & (img > np.percentile(v, 97))
    return body & (img < np.percentile(v, 3))


class Zones:
    """Distance-based zones of one component, computed ONCE on a cropped box (pad 6) and reused for every arm / stage."""
    def __init__(self, C, iface, body, pad=6):
        idx = np.argwhere(C); lo = np.maximum(idx.min(0) - pad, 0); hi = np.minimum(idx.max(0) + pad + 1, C.shape)
        self.box = tuple(slice(a, b) for a, b in zip(lo, hi)); Cb = C[self.box]; bb = body[self.box]
        din = ndi.distance_transform_edt(Cb); dout = ndi.distance_transform_edt(~Cb)
        self.interior = Cb & (din >= 1.5) & bb; self.border = (((din <= 1.0) & Cb) | ((dout <= 1.0) & ~Cb)) & bb
        self.reg = (dout <= 3) & bb; self.outside = self.reg & ~Cb
        self.ib = self.border & ndi.binary_dilation(iface[self.box], iterations=2)
        self.interface_voxel_frac = float(self.ib.sum() / max(self.border.sum(), 1))


def shares(mp, zn: Zones):
    m = mp[zn.box]; tot = m[zn.reg].sum()
    if tot <= 0:
        return None
    bm = m[zn.border].sum(); ib_ = m[zn.interior].mean() if zn.interior.any() else np.nan; bb_ = m[zn.border].mean() if zn.border.any() else np.nan
    return dict(interior_share=float(m[zn.interior].sum() / tot), border_share=float(bm / tot), outside_share=float(m[zn.outside].sum() / tot),
                interface_share=float(m[zn.ib].sum() / bm) if bm > 0 else np.nan, interface_voxel_frac=zn.interface_voxel_frac,
                interior_density_ratio=float(ib_ / bb_) if bb_ and bb_ > 0 else np.nan)


def sham_component(C, body, avoid_dil, shift=10):
    """C shifted `shift` vox in-plane, kept inside the body and off the (pre-dilated) union of predictions; computed on a cropped box."""
    idx = np.argwhere(C); lo = np.maximum(idx.min(0) - shift - 1, 0); hi = np.minimum(idx.max(0) + shift + 2, C.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi)); Cb = C[box]; best = None
    for dy, dx in ((shift, 0), (-shift, 0), (0, shift), (0, -shift)):
        Sb = np.roll(np.roll(Cb, dy, axis=1), dx, axis=2) & body[box] & ~avoid_dil[box]
        S = np.zeros(C.shape, bool); S[box] = Sb
        if best is None or S.sum() > best.sum():
            best = S
    return best


def run_cell(key, cell, dev, only_cases=None):
    od = OUT / key; od.mkdir(exist_ok=True); lids = list(cell["lids"])
    prs = {arm: load_predictor(d, dev) for arm, d in cell["models"].items()}; pr0 = prs["flat"]
    rows, crows = [], []
    for case in (only_cases or cell["cases"]):
        if (od / f"{case}_maps.npz").exists() and (od / f"{case}_lesions.csv").exists() and (od / f"{case}_case.csv").exists():
            log(f"[{key}] {case}: outputs exist, skipped (resume)"); continue   # resumable: short queue-friendly jobs can be resubmitted
        t0 = time.time(); data, gt = preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]
        body = img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool)
        preds, fmaps = {}, {}
        for arm, pr in prs.items():
            pk, mk, gk = [], {s: [] for s in STAGES}, []
            for k in FOLDS:
                p, m = fold_predict_and_attribute(pr, k, data, lids, dev); pk.append(p); gk.append(m["grad"])
                for s in STAGES:
                    mk[s].append(m[s])
            preds[arm] = dict(folds=np.stack(pk), major=None)
            tgt = np.stack([np.isin(p, lids) for p in pk]); preds[arm]["major_any"] = tgt.sum(0) >= 2
            maj = np.zeros(img.shape, np.int16)
            for lid in lids:
                maj[(np.stack([p == lid for p in pk]).sum(0) >= 2)] = lid
            preds[arm]["major"] = maj
            fmaps[arm] = {"grad": np.mean(gk, 0)}
            for s in STAGES:
                st = np.stack(mk[s]); mean = st.mean(0); sd = st.std(0)
                fmaps[arm][s] = dict(mean=mean, agree=np.where(mean > 0, 1 - sd / np.maximum(mean, 1e-12), 0).astype(np.float32), folds=[block_mean(f) for f in mk[s]])
        np.savez_compressed(od / f"{case}_maps.npz", img=img.astype(np.float16), gt=gt, body=body,
                            **{f"major_{a}": preds[a]["major"] for a in prs}, **{f"folds_pred_{a}": preds[a]["folds"].astype(np.int8) for a in prs}, **{f"grad_{a}": fmaps[a]["grad"].astype(np.float16) for a in prs},
                            **{f"mean_{a}_{s}": fmaps[a][s]["mean"].astype(np.float16) for a in prs for s in STAGES},
                            **{f"agree_{a}_{s}": fmaps[a][s]["agree"].astype(np.float16) for a in prs for s in STAGES},
                            **{f"fold{k}_{a}_{s}": fmaps[a][s]["folds"][k] for a in prs for s in STAGES for k in FOLDS})
        major_any = {a: preds[a]["major_any"] for a in prs}; means = {a: {st: fmaps[a][st]["mean"] for st in STAGES} for a in prs}; gt_t = np.isin(gt, lids)
        rws, _ = lesion_rows(key, cell, case, img, gt, body, major_any, means); rows += rws; pd.DataFrame(rws).to_csv(od / f"{case}_lesions.csv", index=False)
        crows.append(case_row(key, cell, case, gt_t, major_any, od, hp_shares(img, gt_t, major_any, {a: fmaps[a]["grad"] for a in prs})))
        n = len(rws)
        for st in STAGES:
            sheet(key, cell, case, img, gt_t, major_any, means, st, od / f"{case}_{st}.png")
        log(f"[{key}] {case}: {n} lesions kept; majority Dice vs GT flat "
            f"{2 * (gt_t & preds['flat']['major_any']).sum() / max(gt_t.sum() + preds['flat']['major_any'].sum(), 1):.3f} real "
            f"{2 * (gt_t & preds['real']['major_any']).sum() / max(gt_t.sum() + preds['real']['major_any'].sum(), 1):.3f}; {time.time() - t0:.0f} s")
    gather_cell(od)


def gather_cell(od: Path):
    """Combine the per-case CSVs (written by possibly several chunked jobs) into lesions.csv / cases.csv."""
    L = [pd.read_csv(f) for f in sorted(od.glob("*_lesions.csv")) if f.stat().st_size > 0]; C = [pd.read_csv(f) for f in sorted(od.glob("*_case.csv")) if f.stat().st_size > 0]
    (pd.concat(L, ignore_index=True) if L else pd.DataFrame()).to_csv(od / "lesions.csv", index=False); (pd.concat(C, ignore_index=True) if C else pd.DataFrame()).to_csv(od / "cases.csv", index=False)


REFS = ("union", "gt", "own", "both")


def lesion_rows(key, cell, case, img, gt, body, major_any, means):
    """Per lesion (26-conn component of the union of both majority predictions, >= min_cc vox): shares of each arm's fold-mean map
    with zones defined on FOUR references: the union component itself, the GT inside the dilated component, the arm's OWN prediction
    inside it, and the intersection of both predictions inside it (so an arm that predicts a smaller mask is not automatically
    'interior'). Interface share and sham on the union reference only."""
    lids = list(cell["lids"]); gt_t = np.isin(gt, lids); union = major_any["flat"] | major_any["real"]
    lab, n = ndi.label(union, structure=np.ones((3, 3, 3))); iface = interface_mask(img, body, cell["interface"])
    sizes = ndi.sum(union, lab, range(1, n + 1)); avoid_dil = ndi.binary_dilation(union, iterations=3); rows = []
    for ci in range(1, n + 1):
        if sizes[ci - 1] < cell["min_cc"]:
            continue
        C = lab == ci; dil = ndi.binary_dilation(C, iterations=3); S = sham_component(C, body, avoid_dil)
        z, y, x = np.argwhere(C).mean(0)
        pf, pr_ = major_any["flat"] & dil, major_any["real"] & dil; g_ = gt_t & dil
        dsc = lambda a_, b_: float(2 * (a_ & b_).sum() / max(a_.sum() + b_.sum(), 1))
        base = dict(cell=key, case=case, lesion=ci, vol_vox=int(C.sum()), cz=float(z), cy=float(y), cx=float(x), gt_frac=float(gt_t[C].mean()),
                    flat_frac=float(major_any["flat"][C].mean()), real_frac=float(major_any["real"][C].mean()),
                    gt_vox_in=int(g_.sum()), both_vox_in=int((pf & pr_).sum()), agree_dice=dsc(pf, pr_), gt_dice_flat=dsc(pf, g_), gt_dice_real=dsc(pr_, g_))
        base["matched_07"] = bool(base["agree_dice"] >= 0.7 and abs(base["gt_dice_flat"] - base["gt_dice_real"]) <= 0.15)
        refs = {"union": C, "gt": gt_t & dil, "both": major_any["flat"] & major_any["real"] & dil, "own_flat": major_any["flat"] & dil, "own_real": major_any["real"] & dil}
        zones = {r: (Zones(m, iface, body) if m.sum() >= max(5, cell["min_cc"] // 4) else None) for r, m in refs.items()}
        zs_ = Zones(S, iface, body) if (S is not None and S.sum() >= cell["min_cc"] // 2) else None
        for a in major_any:
            for st in STAGES:
                for r in REFS:
                    zn = zones[f"own_{a}"] if r == "own" else zones[r]
                    if zn is None:
                        continue
                    sh = shares(means[a][st], zn)
                    if sh:
                        if r == "union":
                            base.update({f"{a}_{st}_{k}": v for k, v in sh.items()})
                        base.update({f"{a}_{st}_{r}_{k}": v for k, v in sh.items() if k in ("interior_share", "border_share", "interior_density_ratio")})
                if zs_ is not None:
                    shs = shares(means[a][st], zs_)
                    if shs:
                        base.update({f"{a}_{st}_sham_{k}": v for k, v in shs.items() if k in ("interior_share", "border_share", "interior_density_ratio")})
        rows.append(base)
    return rows, lab


def hp_shares(img, gt_t, major_any, grads, sigma=2.0):
    """Input x Gradient split: I = L + H (Gaussian low-pass sigma 2 and its residual); share_H = sum|g*H| / (sum|g*L| + sum|g*H|) over a region,
    from the fold-mean signed input gradient g. Regions: GT target, each arm's own prediction, body outside the GT (control)."""
    L = ndi.gaussian_filter(img, sigma); H = img - L; body = img != 0; out = {}
    for a, g in grads.items():
        gl, gh = np.abs(g * L), np.abs(g * H)
        for name, R in (("gt", gt_t), ("own", major_any[a]), ("bg", body & ~ndi.binary_dilation(gt_t, iterations=5))):
            den = gl[R].sum() + gh[R].sum(); out[f"hp_share_{name}_{a}"] = float(gh[R].sum() / den) if den > 0 else np.nan
    return out


def case_row(key, cell, case, gt_t, major_any, od, extra=None):
    d = {"cell": key, "case": case, "gt_vox": int(gt_t.sum())}
    if extra:
        d.update(extra)
    for a, m in major_any.items():
        inter = (m & gt_t).sum(); d[f"dice_{a}"] = float(2 * inter / max(m.sum() + gt_t.sum(), 1)); d[f"recall_{a}"] = float(inter / max(gt_t.sum(), 1))
        d[f"precision_{a}"] = float(inter / max(m.sum(), 1)); d[f"pred_vox_{a}"] = int(m.sum())
    d["delta_dice"] = d["dice_real"] - d["dice_flat"]; d["outcome"] = "GAIN" if d["delta_dice"] > 0.02 else ("LOSS" if d["delta_dice"] < -0.02 else "TIE")
    dr = d["recall_real"] - d["recall_flat"]; d["coverage"] = "real under-segments" if dr < -0.10 else ("real covers more" if dr > 0.10 else "similar coverage")
    pd.DataFrame([d]).to_csv(od / f"{case}_case.csv", index=False); return d


def rerender_cell(key, cell):
    """CPU pass: rebuild sheets, per-lesion rows (4 references) and per-case rows from the saved <case>_maps.npz."""
    od = OUT / key; rows, crows = [], []
    for f in sorted(od.glob("*_maps.npz")):
        case = f.name[:-9]; z = np.load(f); img = z["img"].astype(np.float32); gt = z["gt"]; body = z["body"]; lids = list(cell["lids"]); gt_t = np.isin(gt, lids)
        major_any = {a: np.isin(z[f"major_{a}"], lids) for a in ("flat", "real")}; means = {a: {st: z[f"mean_{a}_{st}"].astype(np.float32) for st in STAGES} for a in ("flat", "real")}
        rws, lab = lesion_rows(key, cell, case, img, gt, body, major_any, means); rows += rws; pd.DataFrame(rws).to_csv(od / f"{case}_lesions.csv", index=False)
        grads = {a: z[f"grad_{a}"].astype(np.float32) for a in ("flat", "real") if f"grad_{a}" in z}
        crows.append(case_row(key, cell, case, gt_t, major_any, od, hp_shares(img, gt_t, major_any, grads) if grads else None))
        for st in STAGES:
            sheet(key, cell, case, img, gt_t, major_any, means, st, od / f"{case}_{st}.png")
        matched = [r for r in rws if r["matched_07"] and r["gt_dice_flat"] >= 0.5 and r["gt_dice_real"] >= 0.5]
        if matched:
            mmask = np.isin(lab, [r["lesion"] for r in matched])
            for st in STAGES:
                sheet(key, cell, case, img, gt_t, major_any, means, st, od / f"{case}_matched_{st}.png", focus=mmask,
                      tag=f"MATCHED lesions only ({len(matched)}: arm agreement Dice >= 0.7, |Dice vs GT flat - real| <= 0.15, both >= 0.5); slices through them")
        log(f"[{key}] re-rendered {case}: {sum(r['case'] == case for r in rows)} lesions; Dice flat {crows[-1]['dice_flat']:.3f} real {crows[-1]['dice_real']:.3f} ({crows[-1]['outcome']}, {crows[-1]['coverage']})")
    gather_cell(od)


def pick_slices(mask, k=3, min_gap=8):
    area = mask.sum((1, 2)); order = np.argsort(area)[::-1]; out = []
    for z in order:
        if area[z] == 0:
            break
        if all(abs(z - o) >= min_gap for o in out):
            out.append(int(z))
        if len(out) == k:
            break
    return sorted(out) or [int(np.argmax(area))]


def sheet(key, cell, case, img, gt_t, major_any, means, s, path, focus=None, tag=None):
    """Rows = 3 lesion-rich slices; columns = flat pred (fill, no heat) | flat heat + flat contour | real pred (fill, no heat) |
    real heat + real contour | real-flat with both contours. GT = green contour everywhere; flat = blue, real = orange.
    Short panel titles; everything else once, in the wrapped suptitle."""
    import textwrap
    union = major_any["flat"] | major_any["real"]; zs = pick_slices(union if focus is None else focus)
    mf, mr = means["flat"][s], means["real"][s]; vmax = max(np.percentile(mf, 99.7), np.percentile(mr, 99.7)) or 1
    diff = mr - mf; dmax = np.percentile(np.abs(diff), 99.7) or 1
    fg = img[img != 0]; lo, hi = np.percentile(fg, [1, 99])
    dice = {a: 2 * (m & gt_t).sum() / max(m.sum() + gt_t.sum(), 1) for a, m in major_any.items()}; rec = {a: (m & gt_t).sum() / max(gt_t.sum(), 1) for a, m in major_any.items()}
    dd = dice["real"] - dice["flat"]; outcome = "GAIN" if dd > 0.02 else ("LOSS" if dd < -0.02 else "TIE")
    COL = {"flat": "#2a78d6", "real": "#eb6834"}
    fig, axes = plt.subplots(len(zs), 5, figsize=(26, 5.6 * len(zs) + 1.2), facecolor="#fcfcfb", squeeze=False, constrained_layout=True)
    titles = {0: "flat-fill pred (fill)", 1: f"flat-fill HiResCAM {s}", 2: "real-fill pred (fill)", 3: f"real-fill HiResCAM {s}", 4: "real - flat"}
    for r, z in enumerate(zs):
        g = img[z]
        for c in range(5):
            ax = axes[r, c]; ax.imshow(g.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest"); ax.axis("off")
            ax.set_title((f"z={z}  " if c == 0 else "") + titles[c], fontsize=11, loc="left", pad=4)
        for c, arm in ((0, "flat"), (2, "real")):
            pm = major_any[arm][z]; rgba = np.zeros(pm.T.shape + (4,)); rgba[pm.T] = matplotlib.colors.to_rgba(COL[arm], 0.45); axes[r, c].imshow(rgba, origin="lower", interpolation="nearest")
        for c, arm in ((1, "flat"), (3, "real")):
            mp = means[arm][s][z]; axes[r, c].imshow(np.ma.masked_less(mp.T, vmax * 0.03), cmap="inferno", vmin=0, vmax=vmax, alpha=0.75, origin="lower", interpolation="nearest")
            if major_any[arm][z].any():
                axes[r, c].contour(major_any[arm][z].T.astype(float), levels=[0.5], colors=[COL[arm]], linewidths=2.0)
        axes[r, 4].imshow(diff[z].T, cmap="RdBu_r", vmin=-dmax, vmax=dmax, alpha=0.8, origin="lower", interpolation="nearest")
        for arm in ("flat", "real"):
            if major_any[arm][z].any():
                axes[r, 4].contour(major_any[arm][z].T.astype(float), levels=[0.5], colors=[COL[arm]], linewidths=1.5)
        for c in range(5):
            if gt_t[z].any():
                axes[r, c].contour(gt_t[z].T.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=1.3)
    head = (f"{case}  [{outcome}]  majority Dice flat {dice['flat']:.2f} -> real {dice['real']:.2f}, GT recall {rec['flat']:.2f} -> {rec['real']:.2f}   |   {cell['title']}" + (f"   |   {tag}" if tag else ""))
    if focus is not None:
        for r in range(len(zs)):
            for c in range(5):
                if focus[zs[r]].any():
                    axes[r, c].contour(focus[zs[r]].T.astype(float), levels=[0.5], colors=["#ffffff"], linewidths=0.8, linestyles="dotted")
    body = (f"Columns: each arm's 3-fold majority prediction as a filled overlay (no heat), then its fold-mean HiResCAM at stage {s} (decoder stages[{STAGES[s]}], "
            f"{'1/4' if s == 's4' else '1/8'} res; each fold normalised to unit mass, averaged over folds 0-2; unmasked; colour scale shared by both arms) with its prediction contour, "
            "then real - flat (red = real-fill model attends more). Contours: GT green, flat-fill blue, real-fill orange. Rows: three lesion-rich slices.")
    fig.suptitle(textwrap.fill(head, 190) + "\n" + textwrap.fill(body, 190), fontsize=11.5, x=0.01, ha="left")
    fig.savefig(path, dpi=100, facecolor="#fcfcfb"); plt.close(fig)


def stats_only():
    dfs = [pd.read_csv(f) for f in sorted(OUT.glob("*/lesions.csv")) if f.stat().st_size > 0]
    cfs = [pd.read_csv(f) for f in sorted(OUT.glob("*/cases.csv")) if f.stat().st_size > 0]
    if not dfs:
        log("no lesions.csv found"); return
    df = pd.concat(dfs, ignore_index=True); df.to_csv(OUT / "lesions_all.csv", index=False)
    cf = pd.concat(cfs, ignore_index=True) if cfs else pd.DataFrame(); cf.to_csv(OUT / "cases_all.csv", index=False)
    cells = [c for c in CELLS if c in set(df.cell)]
    q = lambda x: f"{x.median():.3f} [{x.quantile(.25):.3f}-{x.quantile(.75):.3f}]"
    lines = ["# Visual proof v2: per-lesion attribution shares (fold-mean HiResCAM, unmasked), flat-fill vs real-fill model", "",
             "One row per lesion = 26-conn component of the union of the two arms' 3-fold majority predictions (>= min_cc vox). Interior share = attribution mass in the interior (d >= 1.5 vox) of the REFERENCE mask / mass within the reference dilated by 3 vox. References: `union` = the component itself; `gt` = GT voxels inside the dilated component; `own` = that arm's own prediction inside it; `both` = intersection of the two predictions inside it (the gt/own/both references separate 'attends inside' from 'predicts a smaller mask'). Paired Wilcoxon real vs flat per cell (lesions with both values). Sham = union zones shifted 10 vox in-plane.", ""]
    if len(cf):
        lines += ["## Per-case coverage vs attention (s4, gt reference; lesion-level median per case)", "",
                  "| cell | case | outcome | Dice flat -> real | GT recall flat -> real | gt-ref interior share flat / real | coverage class | attention (gt-ref) |", "|---|---|---|---|---|---|---|---|"]
        summary = {}
        for _, r in cf.sort_values(["cell", "case"]).iterrows():
            d = df[(df.cell == r.cell) & (df.case == r.case)].dropna(subset=["flat_s4_gt_interior_share", "real_s4_gt_interior_share"])
            fi, ri = (d["flat_s4_gt_interior_share"].median(), d["real_s4_gt_interior_share"].median()) if len(d) else (np.nan, np.nan)
            att = "real attends inside more" if ri > fi + 0.02 else ("flat attends inside more" if fi > ri + 0.02 else "similar")
            lines.append(f"| {r.cell} | {r.case} | {r.outcome} | {r.dice_flat:.2f} -> {r.dice_real:.2f} | {r.recall_flat:.2f} -> {r.recall_real:.2f} | {fi:.3f} / {ri:.3f} | {r.coverage} | {att} |")
            summary.setdefault(r.cell, []).append((r.coverage, att))
        lines += ["", "### Counts per cell: coverage class x attention", ""]
        for c in cells:
            if c in summary:
                from collections import Counter
                cnt = Counter(summary[c]); lines.append(f"- **{c}** ({len(summary[c])} cases): " + "; ".join(f"{k[0]} & {k[1]}: {v}" for k, v in sorted(cnt.items())))
        lines.append("")
    if "agree_dice" in df:
        lines += ["## PRIMARY: matched lesions (arm agreement Dice >= 0.7 and |Dice vs GT flat - real| <= 0.15), interior share, GT reference", "",
                  "| cell | stage | n all | n matched | flat (median [IQR]) | real | real > flat | Wilcoxon p | n @0.6 | real>flat @0.6 | p @0.6 | n @0.8 | real>flat @0.8 | p @0.8 | n @0.5 | real>flat @0.5 | p @0.5 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            for s in STAGES:
                fc, rc = f"flat_{s}_gt_interior_share", f"real_{s}_gt_interior_share"; d0 = df[df.cell == c].dropna(subset=[fc, rc, "agree_dice"])
                out = []
                for thr in (0.7, 0.6, 0.8, 0.5):
                    d = d0[(d0.agree_dice >= thr) & ((d0.gt_dice_flat - d0.gt_dice_real).abs() <= 0.15)]; f_, r_ = d[fc], d[rc]
                    p = wilcoxon(r_, f_).pvalue if len(d) >= 5 and (r_ != f_).any() else np.nan
                    out.append((len(d), q(f_) if len(d) else "-", q(r_) if len(d) else "-", (r_ > f_).mean() if len(d) else np.nan, p))
                m7, m6, m8, m5 = out
                lines.append(f"| {c} | {s} | {len(d0)} | {m7[0]} | {m7[1]} | {m7[2]} | {m7[3]:.2f} | {m7[4]:.2g} | {m6[0]} | {m6[3]:.2f} | {m6[4]:.2g} | {m8[0]} | {m8[3]:.2f} | {m8[4]:.2g} | {m5[0]} | {m5[3]:.2f} | {m5[4]:.2g} |")
        lines.append("")
        lines += ["## Volume strata (GT-reference interior share, s4): tertiles of lesion volume within each cell, all lesions and matched (>= 0.5 agreement, |dDice| <= 0.15)", "",
                  "| cell | tertile | vol range (vox) | n all | real > flat | p | n matched@0.5 | real > flat | p |", "|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            fc, rc = "flat_s4_gt_interior_share", "real_s4_gt_interior_share"; d0 = df[df.cell == c].dropna(subset=[fc, rc, "agree_dice"])
            if len(d0) < 6:
                continue
            edges = np.quantile(d0.vol_vox, [0, 1 / 3, 2 / 3, 1])
            for t in range(3):
                dt = d0[(d0.vol_vox >= edges[t]) & (d0.vol_vox <= edges[t + 1])]; dm = dt[(dt.agree_dice >= 0.5) & ((dt.gt_dice_flat - dt.gt_dice_real).abs() <= 0.15)]
                pa = wilcoxon(dt[rc], dt[fc]).pvalue if len(dt) >= 5 and (dt[rc] != dt[fc]).any() else np.nan; pm = wilcoxon(dm[rc], dm[fc]).pvalue if len(dm) >= 5 and (dm[rc] != dm[fc]).any() else np.nan
                lines.append(f"| {c} | T{t + 1} | {int(edges[t])}-{int(edges[t + 1])} | {len(dt)} | {(dt[rc] > dt[fc]).mean():.2f} | {pa:.2g} | {len(dm)} | {(dm[rc] > dm[fc]).mean() if len(dm) else np.nan:.2f} | {pm:.2g} |")
        lines.append("")
        if len(cf) and "hp_share_gt_flat" in cf:
            lines += ["## High-pass share of |Input x Gradient| (sigma 2 split; per case, fold-mean input gradient): does the real-fill model's input sensitivity sit more on the texture band inside the GT target?", "",
                      "| cell | n cases | region | flat (median) | real | real > flat | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
            for c in cells:
                d = cf[cf.cell == c]
                for reg in ("gt", "own", "bg"):
                    dd = d.dropna(subset=[f"hp_share_{reg}_flat", f"hp_share_{reg}_real"])
                    if len(dd) == 0:
                        continue
                    f_, r_ = dd[f"hp_share_{reg}_flat"], dd[f"hp_share_{reg}_real"]; p = wilcoxon(r_, f_).pvalue if len(dd) >= 5 and (r_ != f_).any() else np.nan
                    lines.append(f"| {c} | {len(dd)} | {reg} | {f_.median():.3f} | {r_.median():.3f} | {(r_ > f_).mean():.2f} | {p:.2g} |")
            lines.append("")
        for s in STAGES:
            fig, axes = plt.subplots(2, len(cells), figsize=(4.6 * len(cells), 9), facecolor="#fcfcfb", squeeze=False)
            for j_, c in enumerate(cells):
                fc, rc = f"flat_{s}_gt_interior_share", f"real_{s}_gt_interior_share"; d0 = df[df.cell == c].dropna(subset=[fc, rc, "agree_dice"])
                d = d0[(d0.agree_dice >= 0.7) & ((d0.gt_dice_flat - d0.gt_dice_real).abs() <= 0.15)]; f_, r_ = d[fc].values, d[rc].values; ax = axes[0, j_]
                if len(d):
                    vp = ax.violinplot([f_, r_], positions=[0, 1], showmedians=True, widths=0.8)
                    for b, col in zip(vp["bodies"], ("#2a78d6", "#eb6834")):
                        b.set_facecolor(col); b.set_alpha(0.35)
                    rng = np.random.default_rng(0); ax.scatter(rng.normal(0, 0.05, len(f_)), f_, s=12, c="#2a78d6", alpha=0.7); ax.scatter(1 + rng.normal(0, 0.05, len(r_)), r_, s=12, c="#eb6834", alpha=0.7)
                    for a_, b_ in zip(f_, r_):
                        ax.plot([0, 1], [a_, b_], color="#999", lw=0.4, alpha=0.5)
                    p = wilcoxon(r_, f_).pvalue if len(d) >= 5 and (r_ != f_).any() else np.nan
                    ax.set_title(f"{c}: matched n={len(d)}/{len(d0)}, real>flat {(r_ > f_).mean():.2f}, p={p:.2g}", fontsize=9)
                ax.set_xticks([0, 1]); ax.set_xticklabels(["flat-fill", "real-fill"]); ax.set_ylim(0, 1); ax.set_ylabel("interior share (GT ref)" if j_ == 0 else "")
                ax = axes[1, j_]
                if len(d):
                    sc = ax.scatter(f_, r_, c=d.agree_dice, cmap="viridis", vmin=0.7, vmax=1, s=18, alpha=0.85); ax.plot([0, 1], [0, 1], "k--", lw=0.7)
                    if j_ == len(cells) - 1:
                        plt.colorbar(sc, ax=ax, label="arm agreement Dice")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("flat-fill interior share"); ax.set_ylabel("real-fill interior share" if j_ == 0 else ""); ax.set_title("matched lesions, paired", fontsize=9)
            fig.suptitle(f"PRIMARY: matched lesions (agreement >= 0.7, |dDice| <= 0.15), interior share of the fold-mean HiResCAM ({s}), GT reference", fontsize=11, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / f"matched_violin_scatter_{s}.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    # interface-adjacent lesions: does the flat-fill model put more of its border attribution on the CSF/bone side than the real-fill model?
    lines += ["## Interface-adjacent lesions (>= 10 % of the border band within 2 vox of the intensity-defined interface): share of border attribution ON the interface, and its enrichment over the voxel baseline", "",
              "| cell | stage | subset | n | interface share flat (median) | real | enrichment flat | real | flat > real (frac) | Wilcoxon p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for s in STAGES:
            fc, rc, bc = f"flat_{s}_interface_share", f"real_{s}_interface_share", f"flat_{s}_interface_voxel_frac"
            d0 = df[(df.cell == c)].dropna(subset=[fc, rc, bc]); d0 = d0[d0[bc] >= 0.10]
            for name, d in (("all", d0), ("matched", d0[d0.get("matched_07", False) == True] if "matched_07" in d0 else d0.iloc[0:0])):
                if len(d) == 0:
                    lines.append(f"| {c} | {s} | {name} | 0 | - | - | - | - | - | - |"); continue
                f_, r_ = d[fc], d[rc]; ef, er = (d[fc] / d[bc]).median(), (d[rc] / d[bc]).median()
                p = wilcoxon(f_, r_).pvalue if len(d) >= 5 and (r_ != f_).any() else np.nan
                lines.append(f"| {c} | {s} | {name} | {len(d)} | {f_.median():.3f} | {r_.median():.3f} | {ef:.2f} | {er:.2f} | {(f_ > r_).mean():.2f} | {p:.2g} |")
    lines.append("")
    for s in STAGES:
        for ref in REFS:
            fc, rc = f"flat_{s}_{ref}_interior_share", f"real_{s}_{ref}_interior_share"
            lines += [f"## SECONDARY (all lesions): stage {s}, reference `{ref}`", "", "| cell | n lesions | interior share flat (median [IQR]) | real | real > flat (frac) | Wilcoxon p | border share flat | real | sham interior flat / real |", "|---|---|---|---|---|---|---|---|---|"]
            for c in cells:
                d = df[df.cell == c].dropna(subset=[fc, rc])
                if len(d) == 0:
                    continue
                f_, r_ = d[fc], d[rc]; p = wilcoxon(r_, f_).pvalue if len(d) >= 5 and (r_ != f_).any() else np.nan
                sh = (d.get(f"flat_{s}_sham_interior_share", pd.Series(dtype=float)).median(), d.get(f"real_{s}_sham_interior_share", pd.Series(dtype=float)).median())
                lines.append(f"| {c} | {len(d)} | {q(f_)} | {q(r_)} | {(r_ > f_).mean():.2f} | {p:.2g} | {q(d[f'flat_{s}_{ref}_border_share'])} | {q(d[f'real_{s}_{ref}_border_share'])} | {sh[0]:.3f} / {sh[1]:.3f} |")
            lines.append("")
        # interface (union ref)
        lines += [f"## stage {s}: interface share of the border mass (union reference; baseline = fraction of border voxels on the interface)", "", "| cell | n | flat | real | baseline |", "|---|---|---|---|---|"]
        for c in cells:
            d = df[df.cell == c].dropna(subset=[f"flat_{s}_interface_share", f"real_{s}_interface_share"])
            if len(d):
                lines.append(f"| {c} | {len(d)} | {d[f'flat_{s}_interface_share'].median():.3f} | {d[f'real_{s}_interface_share'].median():.3f} | {d[f'flat_{s}_interface_voxel_frac'].median():.3f} |")
        lines.append("")
        # violins: rows = references, cols = cells
        fig, axes = plt.subplots(len(REFS), len(cells), figsize=(4.6 * len(cells), 4.2 * len(REFS)), facecolor="#fcfcfb", squeeze=False)
        for i_, ref in enumerate(REFS):
            for j_, c in enumerate(cells):
                fc, rc = f"flat_{s}_{ref}_interior_share", f"real_{s}_{ref}_interior_share"; d = df[df.cell == c].dropna(subset=[fc, rc]); f_, r_ = d[fc].values, d[rc].values; ax = axes[i_, j_]
                if len(d):
                    vp = ax.violinplot([f_, r_], positions=[0, 1], showmedians=True, widths=0.8)
                    for b, col in zip(vp["bodies"], ("#2a78d6", "#eb6834")):
                        b.set_facecolor(col); b.set_alpha(0.35)
                    rng = np.random.default_rng(0); ax.scatter(rng.normal(0, 0.05, len(f_)), f_, s=12, c="#2a78d6", alpha=0.7); ax.scatter(1 + rng.normal(0, 0.05, len(r_)), r_, s=12, c="#eb6834", alpha=0.7)
                    for a_, b_ in zip(f_, r_):
                        ax.plot([0, 1], [a_, b_], color="#999", lw=0.4, alpha=0.5)
                    p = wilcoxon(r_, f_).pvalue if len(d) >= 5 and (r_ != f_).any() else np.nan
                    ax.set_title(f"{c}: ref {ref}, n={len(d)}, real>flat {(r_ > f_).mean():.2f}, p={p:.2g}", fontsize=9)
                ax.set_xticks([0, 1]); ax.set_xticklabels(["flat-fill", "real-fill"]); ax.set_ylim(0, 1); ax.set_ylabel("interior share" if j_ == 0 else "")
        fig.suptitle(f"Per-lesion interior share of the fold-mean HiResCAM ({s}); rows = reference mask for the zones (union / GT / own prediction / both predictions)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"violin_interior_{s}.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
        fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.8), facecolor="#fcfcfb", squeeze=False)
        for j_, c in enumerate(cells):
            fc, rc = f"flat_{s}_gt_interior_share", f"real_{s}_gt_interior_share"; d = df[df.cell == c].dropna(subset=[fc, rc]); ax = axes[0, j_]
            if len(d):
                sc = ax.scatter(d[fc], d[rc], c=np.log10(d.vol_vox), cmap="viridis", s=18, alpha=0.8); ax.plot([0, 1], [0, 1], "k--", lw=0.7)
                if j_ == len(cells) - 1:
                    plt.colorbar(sc, ax=ax, label="log10 lesion volume (vox)")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("flat-fill interior share (GT reference)"); ax.set_ylabel("real-fill" if j_ == 0 else ""); ax.set_title(f"{c}, n={len(d)}", fontsize=9)
        fig.suptitle(f"Paired per-lesion interior share, GT reference ({s})", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / f"scatter_gt_interior_{s}.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    (OUT / "lesion_stats.md").write_text("\n".join(lines)); log("wrote lesions_all.csv, cases_all.csv, lesion_stats.md, violin/scatter PNGs"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELLS)); ap.add_argument("--cases", default=None); ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--rerender", action="store_true", help="CPU: rebuild sheets + per-lesion/per-case tables from <case>_maps.npz for --cells")
    ap.add_argument("--case-slice", default=None, help="i:j -> only cases[i:j] of each cell (chunked short jobs)")
    a = ap.parse_args()
    if a.rerender:
        for key in a.cells.split(","):
            rerender_cell(key, CELLS[key])
        return
    if a.stats_only:
        stats_only(); return
    dev = torch.device("cuda")
    for key in a.cells.split(","):
        try:
            cases = a.cases.split(",") if a.cases else None
            if a.case_slice:
                i_, j_ = (int(x) if x else None for x in a.case_slice.split(":")); cases = (cases or CELLS[key]["cases"])[i_:j_]
            run_cell(key, CELLS[key], dev, cases)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"[{key}] FAILED: {e}")


if __name__ == "__main__":
    main()
