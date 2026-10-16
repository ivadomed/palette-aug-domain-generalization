#!/usr/bin/env python
"""
Visual-proof PILOT (2026-10-10): where do the rung-4.5 (FLAT fill) and rung-5 (REAL fill) models look on the SAME test image?
Seg-Grad-CAM-style attribution on nnU-Net v2.7 checkpoints (fold 0, checkpoint_best), one plan-sized patch per case, no
mirroring, fp32, gradients of a class-logit sum over a voxel set M:
  M_shared = union of the two models' own predicted target masks (same target for both models -> paired comparison),
  M_own    = each model's own predicted mask (what the model itself "defends").
Attribution variants (all from the same forward/backward): HiResCAM (sum_c grad*act, no pooling) on the full-res decoder stage
and on the 1/4-res stage, Grad-CAM (GAP weights) on the same stages, Input x Gradient on the input patch.
GT is used ONLY for the Dice numbers and the green contour; every zone/patch/target below is defined from the PREDICTED masks
(no GT leakage, cf. FINDINGS section 12).

Quantification per model (JSON next to the PNGs):
  zones from the signed Euclidean distance to the boundary of M_shared (voxels, preprocessed grid):
     interior (inside, d >= 1.5 vox), boundary (|d| <= 1 vox), ring (outside, 1 < d <= 4 vox), far ring (4 < d <= 8 vox);
  attribution DENSITY per zone (mean of the map normalised to unit mass over the dilated region d <= 8) -> interior/boundary ratio;
  mask-free: Spearman(attribution, |grad I|) and Spearman(attribution, high-pass energy) over the dilated region d <= 8;
  sham: the same zones from M_shared translated by 10 voxels in-plane into tissue (metric-responds control).
Predicted direction (frozen before looking): the REAL-fill model shows a higher interior/boundary density ratio and a higher
texture correlation than the FLAT-fill model on Open-MS (texture helps, +10.6 Dice flair->t2w), and an equal or reversed
ordering on the pelvis MRI->CT gluteus maximus (texture hurts, -5.2 Dice).

Cells (fold 0): open-ms FLAIR-trained -> T2w patient12 (flat 0.385 / real 0.554 case Dice);
                totalseg-pelvic MRI-trained -> CT s0015 gluteus_maximus_left (flat 0.893 / real 0.492).
Output: outputs/plots/visual_proof_pilot/{visual_proof_pilot_2x3.png, <cell>_detail.png, <cell>_metrics.json}.
GPU job: bash scripts/run_visual_proof_pilot.sh   (one L40S, < 1 h).  Options: --cells openms,pelvis  --case <id>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"
OUT = THIS.parent / "outputs/plots/visual_proof_pilot"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("nnUNet_compile", "false")   # a compiled module would break the hooks
os.environ.setdefault("NNUNET_PROJECT_ROOT", str(REPO))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor  # noqa: E402

# dataset roots are overridable for clusters that keep bulk data outside the repo (TamIA: $SCRATCH/<dataset>/...)
O = Path(os.environ.get("VP_OPENMS_ROOT", T2 / "brain_ms/open-ms")); RO = O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
PO = O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet"
P = Path(os.environ.get("VP_PELVIS_ROOT", T2 / "pelvis_healthy/totalseg-pelvic")); RPC = P / "2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT"
PP = P / "8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab"
CELLS = {
    "openms": dict(title="Open-MS, FLAIR-trained -> T2w, lesion (texture HELPS: +10.6 Dice flat -> real)", case="patient12", lid=1, target="lesion",
                   img=lambda c: RO / "imagesTs_t2w" / f"{c}_0000.nii.gz", gt=lambda c: RO / "labelsTs_t2w" / f"{c}.nii.gz",
                   models={"flat": PO / "open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190831/Dataset070_OpenMS_FLAIR/nnUNetTrainerOpenMSAugLabDefault__nnUNetPlans__3d_fullres",
                           "real": PO / "open-ms_flair_v26_6_2_train050_val000_20260711_062946/Dataset070_OpenMS_FLAIR/nnUNetTrainerOpenMSAugLabDefault__nnUNetPlans__3d_fullres"}),
    "pelvis": dict(title="Pelvis, MRI-trained -> CT, gluteus maximus L (texture HURTS: -5.2 Dice flat -> real)", case="s0015", lid=4, target="gluteus_maximus_left",
                   img=lambda c: RPC / "imagesTs_ct" / f"{c}_0000.nii.gz", gt=lambda c: RPC / "labelsTs_ct" / f"{c}.nii.gz",
                   models={"flat": PP / "totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257/Dataset131_TotalsegPelvic_MRI/nnUNetTrainerTotalsegPelvicAugLabDefault__nnUNetPlans__3d_fullres",
                           "real": PP / "totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058/Dataset131_TotalsegPelvic_MRI/nnUNetTrainerTotalsegPelvicAugLabDefault__nnUNetPlans__3d_fullres"}),
}
ARM_LABEL = {"flat": "flat-fill model (rung 4.5)", "real": "real-fill model (rung 5)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_predictor(model_dir: Path, dev):
    pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                         device=dev, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pr.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,), checkpoint_name="checkpoint_best.pth")
    pr.network.load_state_dict(pr.list_of_parameters[0]); pr.network.to(dev).eval()
    return pr


def preprocess(pr, img_path: Path, gt_path: Path):
    """Exactly inference's preprocessing (reader, transpose, resampling, masked z-score); GT resampled alongside (nearest)."""
    rw = pr.plans_manager.image_reader_writer_class()
    data, props = rw.read_images([str(img_path)])
    seg, _ = rw.read_seg(str(gt_path))
    data, seg, _ = DefaultPreprocessor(verbose=False).run_case_npy(data, seg, props, pr.plans_manager, pr.configuration_manager, pr.dataset_json)   # crop_to_nonzero marks outside as -1 -> masked z-score identical to inference
    return data.astype(np.float32), seg[0].astype(np.int16), props


@torch.no_grad()
def full_prediction(pr, data: np.ndarray, dev):
    logits = pr.predict_logits_from_preprocessed_data(torch.from_numpy(data))   # (C, Z, Y, X) on cpu, preprocessed grid
    return logits.argmax(0).numpy().astype(np.int16)


def patch_box(center, size, shape):
    size = np.minimum(np.array(size), np.array(shape)); lo = np.clip(np.array(center) - size // 2, 0, np.array(shape) - size)
    return tuple(slice(int(a), int(a + s)) for a, s in zip(lo, size))


def attribution(net, x: torch.Tensor, M: torch.Tensor, lid: int, dev):
    """x (1,1,Z,Y,X) fp32; M bool (Z,Y,X) target voxel set. Returns dict of maps (Z,Y,X) numpy, all non-negative."""
    stages = net.decoder.stages; taps = {"full": stages[-1], "quarter": stages[-3]}
    acts, grads = {}, {}
    hs = []
    for k, m in taps.items():
        def fh(mod, inp, out, k=k):
            acts[k] = out; out.register_hook(lambda g, k=k: grads.__setitem__(k, g))
        hs.append(m.register_forward_hook(fh))
    x = x.clone().requires_grad_(True)
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        out = net(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        score = (out[0, lid] - out[0, 0])[M].sum()
        score.backward()
    for h in hs:
        h.remove()
    maps = {}
    size = x.shape[2:]
    for k in taps:
        a, g = acts[k].detach().float(), grads[k].detach().float()
        hires = F.relu((a * g).sum(1, keepdim=True)); gcam = F.relu((a * g.mean(dim=(2, 3, 4), keepdim=True)).sum(1, keepdim=True))
        for name, mp in (("hirescam", hires), ("gradcam", gcam)):
            if tuple(mp.shape[2:]) != tuple(size):
                mp = F.interpolate(mp, size=size, mode="trilinear", align_corners=False)
            maps[f"{name}_{k}"] = mp[0, 0].cpu().numpy()
    maps["inputxgrad"] = (x.grad[0, 0] * x[0, 0]).abs().detach().cpu().numpy()
    maps["_logit_margin"] = (out[0, lid] - out[0, 0]).detach().cpu().numpy(); maps["_pred"] = out[0].argmax(0).detach().cpu().numpy()
    return maps


def zones(M: np.ndarray):
    """Signed distance (vox) to the boundary of M: >0 inside, <0 outside."""
    din = ndi.distance_transform_edt(M); dout = ndi.distance_transform_edt(~M)
    d = np.where(M, din, -dout)
    return {"interior": M & (din >= 1.5), "boundary": np.abs(d) <= 1.0, "ring": (~M) & (dout > 1) & (dout <= 4), "far_ring": (~M) & (dout > 4) & (dout <= 8), "_region": d >= -8}


def texture_edge_maps(img: np.ndarray):
    g = np.sqrt(sum(ndi.sobel(img, axis=i) ** 2 for i in range(3)))
    hp = img - ndi.gaussian_filter(img, 1.5); tex = ndi.gaussian_filter(hp ** 2, 1.5)
    return g, tex


def quantify(mp: np.ndarray, M: np.ndarray, grad: np.ndarray, tex: np.ndarray, body: np.ndarray):
    z = zones(M); reg = z["_region"] & body
    tot = mp[reg].sum()
    if tot <= 0 or M.sum() == 0:
        return None
    n = mp / tot
    out = {k: float(n[v & body].mean()) if (v & body).any() else float("nan") for k, v in z.items() if not k.startswith("_")}
    out["n_vox"] = {k: int((v & body).sum()) for k, v in z.items() if not k.startswith("_")}
    out["interior_over_boundary"] = out["interior"] / out["boundary"] if out["boundary"] > 0 else float("nan")
    out["mass_inside_M"] = float(n[M & body].sum())
    idx = np.flatnonzero(reg)
    if idx.size > 60000:
        idx = np.random.default_rng(0).choice(idx, 60000, replace=False)
    out["spearman_edge"] = float(spearmanr(mp.ravel()[idx], grad.ravel()[idx]).statistic)
    out["spearman_texture"] = float(spearmanr(mp.ravel()[idx], tex.ravel()[idx]).statistic)
    return out


def sham_mask(M: np.ndarray, body: np.ndarray, shift=10):
    """M translated in-plane by `shift` voxels, in the direction that keeps most of it inside the body and off M."""
    best = None
    for dy, dx in ((shift, 0), (-shift, 0), (0, shift), (0, -shift)):
        S = np.roll(np.roll(M, dy, axis=1), dx, axis=2); S &= body & ~ndi.binary_dilation(M, iterations=3)
        if best is None or S.sum() > best.sum():
            best = S
    return best


def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else float("nan")


def show(ax, g, title, cmap="gray", vlim=None, alpha_map=None, contours=()):
    if vlim is None:
        fg = g[g != 0]; vlim = np.percentile(fg, [1, 99]) if fg.size else (0, 1)
    ax.imshow(g, cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin="lower", interpolation="nearest")
    if alpha_map is not None:
        ax.imshow(alpha_map, cmap="inferno", vmin=0, vmax=np.percentile(alpha_map, 99.5) or 1, alpha=0.6, origin="lower", interpolation="nearest")
    for m, col, lw in contours:
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=lw)
    ax.set_title(title, fontsize=8.5, loc="left"); ax.axis("off")


def run_cell(key, cell, dev, case=None):
    case = case or cell["case"]; lid = cell["lid"]
    log(f"[{key}] case {case}: loading models")
    prs = {arm: load_predictor(d, dev) for arm, d in cell["models"].items()}
    pr0 = prs["flat"]
    data, gt, props = preprocess(pr0, cell["img"](case), cell["gt"](case))
    log(f"[{key}] preprocessed grid {data.shape}, spacing(transposed) {pr0.configuration_manager.spacing}")
    body = data[0] != 0 if (data[0] == 0).mean() > 0.05 else np.ones(data.shape[1:], bool)   # masked z-score leaves exactly 0 outside the body
    preds = {}
    for arm, pr in prs.items():
        preds[arm] = full_prediction(pr, data, dev) == lid
        log(f"[{key}] {arm}: full-volume pred {int(preds[arm].sum())} vox, Dice vs GT (preprocessed grid) {dice(preds[arm], gt == lid):.3f}")
    M_shared = preds["flat"] | preds["real"]
    if M_shared.sum() == 0:
        log(f"[{key}] no predicted target at all -> cannot place a patch from predictions; falling back to GT centroid ONLY for placement")
        M_shared = gt == lid
    ps = pr0.configuration_manager.patch_size
    z = int(np.argmax(M_shared.sum((1, 2)))); cyx = np.argwhere(M_shared[z]).mean(0)
    box = patch_box((z, cyx[0], cyx[1]), ps, data.shape[1:])
    log(f"[{key}] patch {box} (plan patch {ps}); shared target {int(M_shared[box].sum())} vox in patch")
    xp = torch.from_numpy(data[0][box])[None, None].to(dev); Mp = torch.from_numpy(M_shared[box]).to(dev)
    img_p = data[0][box]; body_p = body[box]; grad_p, tex_p = texture_edge_maps(img_p)
    gt_p = gt[box] == lid
    res = {"cell": key, "case": case, "target": cell["target"], "patch_box": [[s.start, s.stop] for s in box], "patch_size": list(map(int, ps)),
           "dice_fullvolume_preprocessed_grid": {arm: dice(preds[arm], gt == lid) for arm in prs}, "models": {arm: str(d) for arm, d in cell["models"].items()},
           "n_shared_target_vox_patch": int(M_shared[box].sum()), "arms": {}}
    maps_all = {}
    for arm, pr in prs.items():
        net = pr.network
        maps = attribution(net, xp, Mp, lid, dev)
        own = torch.from_numpy(preds[arm][box]).to(dev)
        maps_own = attribution(net, xp, own, lid, dev) if own.any() else None
        maps_all[arm] = maps
        pred_patch = maps["_pred"] == lid
        r = {"patch_pred_vox": int(pred_patch.sum()), "patch_pred_dice_vs_gt": dice(pred_patch, gt_p), "patch_pred_dice_vs_fullvolume_pred": dice(pred_patch, preds[arm][box])}
        for name in ("hirescam_full", "hirescam_quarter", "gradcam_full", "gradcam_quarter", "inputxgrad"):
            r[f"{name}__target_shared"] = quantify(maps[name], M_shared[box], grad_p, tex_p, body_p)
            if maps_own is not None:
                r[f"{name}__target_own"] = quantify(maps_own[name], preds[arm][box], grad_p, tex_p, body_p)
            S = sham_mask(M_shared[box], body_p)
            r[f"{name}__sham_shifted_zones"] = quantify(maps[name], S, grad_p, tex_p, body_p) if S.sum() > 0 else None
        res["arms"][arm] = r
        q = r["hirescam_full__target_shared"]
        log(f"[{key}] {arm}: HiResCAM(full) interior/boundary density {q['interior_over_boundary']:.3f}, mass in M {q['mass_inside_M']:.3f}, "
            f"rho(edge) {q['spearman_edge']:.3f}, rho(texture) {q['spearman_texture']:.3f}; sham i/b "
            f"{(r['hirescam_full__sham_shifted_zones'] or {}).get('interior_over_boundary', float('nan')):.3f}")
    with open(OUT / f"{key}_metrics.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    # ---- detail figure: rows = attribution variants, cols = image | flat | real
    zs = int(np.argmax(M_shared[box].sum((1, 2))))
    sl = lambda a: a[zs]
    names = [("hirescam_full", "HiResCAM, full-res decoder stage"), ("hirescam_quarter", "HiResCAM, 1/4-res decoder stage"),
             ("gradcam_full", "Grad-CAM, full-res stage"), ("inputxgrad", "|Input x Gradient|")]
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 4.2 * len(names)), facecolor="#fcfcfb")
    for r_, (name, lab) in enumerate(names):
        show(axes[r_, 0], sl(img_p), f"{case}: test image (preprocessed grid), GT {cell['target']} green" if r_ == 0 else "test image",
             contours=[(sl(gt_p), "#1baf7a", 0.9)])
        for c_, arm in enumerate(("flat", "real")):
            mp = maps_all[arm][name]; q = res["arms"][arm][f"{name}__target_shared"]
            show(axes[r_, 1 + c_], sl(img_p), f"{ARM_LABEL[arm]}: {lab}\nown pred blue, shared target orange | int/bnd density {q['interior_over_boundary']:.2f}, rho_tex {q['spearman_texture']:.2f}, rho_edge {q['spearman_edge']:.2f}",
                 alpha_map=sl(mp), contours=[(sl(maps_all[arm]["_pred"] == lid), "#2a78d6", 0.8), (sl(M_shared[box]), "#eb6834", 0.5)])
    fig.suptitle(f"{cell['title']}  --  fold 0, case Dice (full volume, preprocessed grid): flat {res['dice_fullvolume_preprocessed_grid']['flat']:.2f} / real {res['dice_fullvolume_preprocessed_grid']['real']:.2f}; "
                 "target = sum of (logit target - logit bg) over the SHARED predicted mask; no GT in the attribution", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"{key}_detail.png", dpi=120, facecolor="#fcfcfb"); plt.close(fig)
    log(f"[{key}] wrote {key}_detail.png, {key}_metrics.json")
    row = dict(img=sl(img_p), gt=sl(gt_p), M=sl(M_shared[box]), maps={arm: sl(maps_all[arm]["hirescam_full"]) for arm in prs},
               preds={arm: sl(maps_all[arm]["_pred"] == lid) for arm in prs}, res=res, title=cell["title"])
    np.savez_compressed(OUT / f"{key}_slices.npz", img=row["img"], gt=row["gt"], M=row["M"], map_flat=row["maps"]["flat"], map_real=row["maps"]["real"],
                        pred_flat=row["preds"]["flat"], pred_real=row["preds"]["real"])   # so the summary figure can be rebuilt without a GPU
    return row


def load_row(key):
    f = OUT / f"{key}_slices.npz"
    if not f.exists():
        return None
    z = np.load(f); res = json.load(open(OUT / f"{key}_metrics.json"))
    return dict(img=z["img"], gt=z["gt"], M=z["M"], maps={"flat": z["map_flat"], "real": z["map_real"]}, preds={"flat": z["pred_flat"], "real": z["pred_real"]}, res=res, title=CELLS[key]["title"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default="openms,pelvis"); ap.add_argument("--case", default=None)
    ap.add_argument("--summary-only", action="store_true", help="rebuild the 2x3 figure from the saved <cell>_slices.npz (no GPU)")
    a = ap.parse_args()
    rows = []
    if a.summary_only:
        rows = [r for r in (load_row(k) for k in CELLS) if r is not None]
    dev = torch.device("cuda") if not a.summary_only else None
    for key in ([] if a.summary_only else a.cells.split(",")):
        try:
            rows.append(run_cell(key, CELLS[key], dev, a.case if len(a.cells.split(",")) == 1 else None))
        except Exception as e:   # one failing cell must not lose the other
            import traceback; traceback.print_exc(); log(f"[{key}] FAILED: {e}")
    if not rows:
        sys.exit(1)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 4.6 * len(rows)), facecolor="#fcfcfb", squeeze=False)
    for r_, row in enumerate(rows):
        res = row["res"]
        show(axes[r_, 0], row["img"], f"{res['case']}: test image, GT {res['target']} (green)", contours=[(row["gt"], "#1baf7a", 0.9)])
        for c_, arm in enumerate(("flat", "real")):
            q = res["arms"][arm]["hirescam_full__target_shared"]
            show(axes[r_, 1 + c_], row["img"], f"{ARM_LABEL[arm]}, HiResCAM (full-res), case Dice {res['dice_fullvolume_preprocessed_grid'][arm]:.2f}\n"
                 f"interior/boundary density {q['interior_over_boundary']:.2f} | rho(texture) {q['spearman_texture']:.2f} | rho(edge) {q['spearman_edge']:.2f}",
                 alpha_map=row["maps"][arm], contours=[(row["preds"][arm], "#2a78d6", 0.8), (row["M"], "#eb6834", 0.5)])
        axes[r_, 0].text(0.0, -0.04, row["title"], transform=axes[r_, 0].transAxes, fontsize=9, va="top")
    fig.suptitle("PILOT: attribution of the same test image under the flat-fill (rung 4.5) and real-fill (rung 5) models. Blue = model's own prediction, orange = shared target (union), "
                 "zones from the predicted masks, GT only drawn.", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "visual_proof_pilot_2x3.png", dpi=130, facecolor="#fcfcfb"); plt.close(fig)
    log("wrote visual_proof_pilot_2x3.png")


if __name__ == "__main__":
    main()
