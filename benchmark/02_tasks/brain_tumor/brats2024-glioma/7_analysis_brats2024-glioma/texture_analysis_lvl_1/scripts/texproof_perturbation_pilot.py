#!/usr/bin/env python
"""
texproof pilot: GT-FREE global texture removal as a visual/quantitative proof of WHEN a model uses texture.

Compares the two arms of the fill-swap decomposition (FINDINGS §25) on the SAME test images, perturbed WITHOUT the ground truth:
  rung 4.5 = FLAT fill (region means)   vs   rung 5 = REAL fill ("PALETTE alone")      (fold 0, checkpoint_best, no mirroring TTA)
Perturbations (all defined on the whole image, the GT is used only for SCORING):
  gauss_s<σ mm>   Gaussian smoothing, normalised convolution inside the image's own nonzero mask (skull-stripped MRI: no halo at the brain edge,
                  nnU-Net's nonzero normalisation mask unchanged); plain Gaussian on CT (no zero mask).
  noise_k<k>      additive white Gaussian noise, std = k x std of the nonzero voxels (control: a perturbation that ADDS high frequencies).
Readouts per case x label x perturbation: Dice vs GT, predicted volume; plot = absolute Dice vs σ per arm, Dice retention Dice(σ)/Dice(0),
Δ = real − flat vs σ. Prediction: where texture addition HELPS (Open-MS lesions on T2w, §25) the real-fill model loses more under smoothing;
where it HURTS (pelvis MRI-trained -> CT gluteal muscles/hips) it should not.
Optional (--occlusion CASE): flat-fill OCCLUSION map on one case: each window (win³ voxels, stride) inside the nonzero mask on an axial band
around the slice with the most PREDICTED lesion (union of both arms' unperturbed predictions, GT not used) is replaced by its local mean;
readout per window = ΔDice and Δ predicted lesion volume per arm -> a gradient-free "where texture matters" map.
Self-check: the unperturbed re-prediction must reproduce the stored fold-0 eval_all.csv Dice to < 0.01 (printed + in the CSV).
Usage (GPU job via texproof_run_pilot.sh):
  .venv/bin/python texproof_perturbation_pilot.py --dataset openms [--cases ...] [--sigmas 0.5 1 1.5 2.5] [--noise 0.25 0.5] [--occlusion patient12]
  .venv/bin/python texproof_perturbation_pilot.py --dataset pelvis --cases s0066 s0193 s0261 --sigmas 1 2 3.5 5
Outputs: outputs/plots/texproof_perturbation_pilot/{<dataset>_curves.png,<dataset>_examples.png,<dataset>_occlusion_<case>.png},
         outputs/tables/texproof_perturbation_pilot_<dataset>.csv ; scratch inputs/predictions under $SCRATCH/texproof_pilot/<dataset>/.
"""
from __future__ import annotations
import argparse, os, time, zlib
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; LVL1 = THIS.parent; REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"
OUT_PLOTS = LVL1 / "outputs/plots/texproof_perturbation_pilot"; OUT_TABLES = LVL1 / "outputs/tables"
SCR = Path(os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'paulh')}")) / "texproof_pilot"
TRN = "__nnUNetPlans__3d_fullres"

# Data roots: Vulcan = the repo tree; TamIA = the scratch mirror (same layout below the dataset dir). Override with TEXPROOF_OPENMS_ROOT / TEXPROOF_PELVIS_ROOT.
O = Path(os.environ.get("TEXPROOF_OPENMS_ROOT", T2 / "brain_ms/open-ms")); P = Path(os.environ.get("TEXPROOF_PELVIS_ROOT", T2 / "pelvis_healthy/totalseg-pelvic"))
STORED = LVL1 / "outputs/data/texproof_stored_metrics"   # copies of the fold-0 eval_all.csv of each arm (self-check), shipped with the scripts
DATASETS = {
    "openms": dict(
        title="Open-MS, FLAIR-trained, tested on T2w (texture addition HELPS: +10.6 Dice pooled, §25)",
        img_dir=O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/imagesTs_t2w", gt_dir=O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/labelsTs_t2w",
        labels={1: "lesion"}, focus=[1], group="t2w", mask_norm=True,
        models={"flat": O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet/open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190831/Dataset070_OpenMS_FLAIR" / ("nnUNetTrainerOpenMSAugLabDefault" + TRN),
                "real": O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet/open-ms_flair_v26_6_2_train050_val000_20260711_062946/Dataset070_OpenMS_FLAIR" / ("nnUNetTrainerOpenMSAugLabDefault" + TRN)},
        stored={"flat": STORED / "openms_flat_fold0_eval_all.csv", "real": STORED / "openms_real_fold0_eval_all.csv"},
        default_cases=None, default_sigmas=[0.5, 1.0, 1.5, 2.5], default_noise=[0.25, 0.5]),
    "pelvis": dict(
        title="TotalSegmentator pelvis, MRI-trained, tested on CT (texture addition HURTS: hips −3.9/−3.6, gluteus maximus L −5.2, §25)",
        img_dir=P / "2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT/imagesTs_ct", gt_dir=P / "2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT/labelsTs_ct",
        labels={1: "hip_left", 2: "hip_right", 3: "sacrum", 4: "gluteus_maximus_left", 5: "gluteus_maximus_right", 6: "gluteus_medius_left", 7: "gluteus_medius_right",
                8: "gluteus_minimus_left", 9: "gluteus_minimus_right", 10: "iliopsoas_left", 11: "iliopsoas_right"},
        focus=[1, 2, 4, 5], group="ct", mask_norm=False,
        models={"flat": P / "8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab/totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257/Dataset131_TotalsegPelvic_MRI" / ("nnUNetTrainerTotalsegPelvicAugLabDefault" + TRN),
                "real": P / "8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab/totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058/Dataset131_TotalsegPelvic_MRI" / ("nnUNetTrainerTotalsegPelvicAugLabDefault" + TRN)},
        stored={"flat": STORED / "pelvis_flat_fold0_eval_all.csv", "real": STORED / "pelvis_real_fold0_eval_all.csv"},
        default_cases=["s0066", "s0193", "s0261"], default_sigmas=[1.0, 2.0, 3.5, 5.0], default_noise=[]),
}
COL = {"flat": "#2a78d6", "real": "#eb6834"}


def md_table(df):
    """Markdown table without the optional `tabulate` dependency (absent on TamIA's venv)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + ["" if pd.isna(v) else str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def arr(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))


def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else float("nan")


# ----------------------------------------------------------------------------- perturbations (GT never touched)
def smooth(img, sigma_vox, mask):
    """Gaussian smoothing; normalised convolution inside `mask` (None -> plain)."""
    if mask is None:
        return gaussian_filter(img, sigma_vox)
    m = mask.astype(np.float32)
    num = gaussian_filter(img * m, sigma_vox); den = gaussian_filter(m, sigma_vox)
    out = np.where(mask, num / np.maximum(den, 1e-3), 0.0).astype(np.float32)
    return out


def build_inputs(D, cases, sigmas, noises):
    """Write <SCR>/<ds>/inputs/<pert>/<case>_0000.nii.gz with the original affine/header. Returns {pert: dir}."""
    perts = {}
    for case in cases:
        src = D["img_dir"] / f"{case}_0000.nii.gz"; ni = nib.load(str(src)); img = np.asarray(ni.dataobj).astype(np.float32)
        zoom = np.asarray(ni.header.get_zooms()[:3], dtype=np.float32)
        mask = (img != 0) if D["mask_norm"] else None
        fg = img[mask] if mask is not None else img
        rng = np.random.default_rng(zlib.crc32(case.encode()))
        variants = {"orig": img}
        for s in sigmas:
            variants[f"gauss_s{s:g}"] = smooth(img, s / zoom, mask)   # σ in mm -> voxels per axis
        for k in noises:
            n = rng.normal(0.0, k * float(fg.std()), img.shape).astype(np.float32)
            variants[f"noise_k{k:g}"] = np.where(mask, img + n, 0.0).astype(np.float32) if mask is not None else img + n
        for name, v in variants.items():
            d = SCR / D["name"] / "inputs" / name; d.mkdir(parents=True, exist_ok=True); perts[name] = d
            out = d / f"{case}_0000.nii.gz"
            if not out.exists():
                nib.save(nib.Nifti1Image(v.astype(np.float32), ni.affine, ni.header), str(out))
        print(f"inputs {case}: shape {img.shape} zooms {zoom.round(2).tolist()} -> {len(variants)} variants", flush=True)
    return perts


# ----------------------------------------------------------------------------- inference (nnU-Net v2.7 nnUNetPredictor)
def make_predictor(model_dir):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                         device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pr.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,), checkpoint_name="checkpoint_best.pth")
    return pr


def predict_dir(pr, in_dir, out_dir, cases, probs_for=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if not (out_dir / f"{c}.nii.gz").exists()]
    if todo:
        pr.predict_from_files([[str(in_dir / f"{c}_0000.nii.gz")] for c in todo], [str(out_dir / c) for c in todo], save_probabilities=False,
                              overwrite=True, num_processes_preprocessing=2, num_processes_segmentation_export=2)
    need = [c for c in probs_for if c in cases and not (out_dir / f"{c}.npz").exists()]
    if need:
        pr.predict_from_files([[str(in_dir / f"{c}_0000.nii.gz")] for c in need], [str(out_dir / c) for c in need], save_probabilities=True,
                              overwrite=True, num_processes_preprocessing=2, num_processes_segmentation_export=2)


# ----------------------------------------------------------------------------- scoring
def score(D, cases, perts, rows_csv):
    gts = {c: arr(D["gt_dir"] / f"{c}.nii.gz").round().astype(np.int16) for c in cases}
    stored = {}
    for arm, f in D["stored"].items():
        df = pd.read_csv(f); df = df[df.group == D["group"]]; stored[arm] = {(r.case, r.label): float(r.dice) for r in df.itertuples()}
    rows = []
    for arm in D["models"]:
        for pert in perts:
            for c in cases:
                seg = arr(SCR / D["name"] / "preds" / arm / pert / f"{c}.nii.gz").round().astype(np.int16)
                for lid, lname in D["labels"].items():
                    g = gts[c] == lid; p = seg == lid
                    if not g.any():
                        continue
                    rows.append(dict(dataset=D["name"], arm=arm, pert=pert, kind=pert.split("_")[0], strength=float(pert.split("_")[1][1:]) if "_" in pert else 0.0,
                                     case=c, label=lname, dice=dice(p, g), pred_vox=int(p.sum()), gt_vox=int(g.sum()),
                                     stored_dice=stored[arm].get((c, lname), np.nan)))
    df = pd.DataFrame(rows); df.to_csv(rows_csv, index=False)
    chk = df[df.pert == "orig"].assign(absdiff=lambda d: (d.dice - d.stored_dice).abs())
    print("SELF-CHECK orig vs stored eval_all.csv (fold 0): max |Δ| = %.4f over %d rows; mean |Δ| = %.4f" % (chk.absdiff.max(), len(chk), chk.absdiff.mean()), flush=True)
    print(chk[chk.label.isin([D["labels"][l] for l in D["focus"]])][["arm", "case", "label", "dice", "stored_dice"]].round(3).to_string(index=False), flush=True)
    return df


# ----------------------------------------------------------------------------- figures
def plot_curves(D, df, out):
    foc = [D["labels"][l] for l in D["focus"]]
    d = df[df.label.isin(foc)].copy()
    for kind, xlabel in (("gauss", "Gaussian σ (mm)"), ("noise", "noise std / foreground std")):
        sub = d[(d.kind == kind) | (d.pert == "orig")]
        if (sub.kind == kind).sum() == 0:
            continue
        xs = sorted(sub.strength.unique())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor="#fcfcfb")
        base = sub[sub.pert == "orig"].set_index(["arm", "case", "label"]).dice
        sub = sub.assign(retention=[r.dice / base.get((r.arm, r.case, r.label), np.nan) if base.get((r.arm, r.case, r.label), 0) > 0 else np.nan for r in sub.itertuples()])
        for arm in ("flat", "real"):
            a = sub[sub.arm == arm]
            for (c, l), g in a.groupby(["case", "label"]):
                g = g.sort_values("strength"); axes[0].plot(g.strength, g.dice, color=COL[arm], alpha=0.25, lw=0.8); axes[1].plot(g.strength, g.retention, color=COL[arm], alpha=0.25, lw=0.8)
            m = a.groupby("strength").agg(dice=("dice", "mean"), ret=("retention", "mean"), n=("dice", "size")).reset_index()
            axes[0].plot(m.strength, m.dice, "o-", color=COL[arm], lw=2.2, label=f"{arm}-fill model (mean of {int(m.n.max())} case×label)")
            axes[1].plot(m.strength, m.ret, "o-", color=COL[arm], lw=2.2, label=f"{arm}-fill")
        piv = sub.pivot_table(index=["case", "label", "strength"], columns="arm", values="dice").dropna()
        piv["delta"] = piv.real - piv.flat; piv = piv.reset_index()
        for (c, l), g in piv.groupby(["case", "label"]):
            g = g.sort_values("strength"); axes[2].plot(g.strength, g.delta, color="#555", alpha=0.25, lw=0.8)
        m = piv.groupby("strength").delta.agg(["mean", "sem"]).reset_index()
        axes[2].errorbar(m.strength, m["mean"], yerr=m["sem"], fmt="o-", color="#1baf7a", lw=2.2, capsize=3, label="Δ Dice = real − flat (mean ± SEM)")
        axes[2].axhline(0, color="#999", lw=0.8, ls="--")
        axes[0].set_title("Dice vs GT (GT used for scoring only)", loc="left", fontsize=10); axes[0].set_ylabel("Dice"); axes[0].legend(fontsize=8)
        axes[1].set_title("Dice retention  Dice(σ) / Dice(0)", loc="left", fontsize=10); axes[1].axhline(1, color="#999", lw=0.8, ls="--"); axes[1].legend(fontsize=8)
        axes[2].set_title("real − flat (texture-addition effect) vs perturbation", loc="left", fontsize=10); axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel(xlabel); ax.set_xticks(xs); ax.grid(alpha=0.25)
        fig.suptitle(f"{D['title']}\nlabels {', '.join(foc)}; fold 0, checkpoint_best, no TTA; perturbation = {'Gaussian smoothing of the whole image (normalised inside the nonzero mask)' if kind == 'gauss' else 'additive white noise on the whole image'}", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9)); f = out / f"{D['name']}_curves_{kind}.png"; fig.savefig(f, dpi=130, facecolor="#fcfcfb"); plt.close(fig); print("wrote", f, flush=True)


def _show(ax, g, m=None, col="#eb6834", title="", vlim=None):
    fgm = g != 0; lo, hi = vlim if vlim is not None else (np.percentile(g[fgm], [1, 99]) if fgm.any() else (0, 1))
    ax.imshow(g, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    if m is not None and m.any():
        rgba = np.zeros(m.shape + (4,)); rgba[m] = matplotlib.colors.to_rgba(col, 0.5); ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=8.5, loc="left"); ax.axis("off")


def plot_examples(D, df, cases, pert_hi, out, lid):
    """Rows = cases; columns: image | smoothed image | GT | flat pred (orig) | flat pred (smoothed) | real pred (orig) | real pred (smoothed) | real-model P(label) change."""
    lname = D["labels"][lid]
    fig, axes = plt.subplots(len(cases), 8, figsize=(25, 3.2 * len(cases)), facecolor="#fcfcfb")
    axes = np.atleast_2d(axes)
    for r, c in enumerate(cases):
        img = arr(SCR / D["name"] / "inputs/orig" / f"{c}_0000.nii.gz"); sm = arr(SCR / D["name"] / "inputs" / pert_hi / f"{c}_0000.nii.gz")
        g = arr(D["gt_dir"] / f"{c}.nii.gz").round() == lid
        pr = {(arm, p): arr(SCR / D["name"] / "preds" / arm / p / f"{c}.nii.gz").round() == lid for arm in ("flat", "real") for p in ("orig", pert_hi)}
        z = int(np.argmax(g.sum((1, 2))))   # slice with most GT: display choice only
        dz = lambda arm, p: df[(df.case == c) & (df.label == lname) & (df.arm == arm) & (df.pert == p)].dice.iloc[0]
        vl = np.percentile(img[img != 0], [1, 99]) if (img != 0).any() else None
        _show(axes[r, 0], img[z], None, title=f"{c}: test image", vlim=vl); _show(axes[r, 1], sm[z], None, title=f"{pert_hi} (whole image, GT-free)", vlim=vl)
        _show(axes[r, 2], img[z], g[z], title=f"GT {lname} ({int(g.sum())} vox)")
        _show(axes[r, 3], img[z], pr[("flat", "orig")][z], COL["flat"], title=f"flat-fill model, orig: Dice {dz('flat', 'orig'):.2f}")
        _show(axes[r, 4], sm[z], pr[("flat", pert_hi)][z], COL["flat"], title=f"flat-fill model, smoothed: Dice {dz('flat', pert_hi):.2f}", vlim=vl)
        _show(axes[r, 5], img[z], pr[("real", "orig")][z], COL["real"], title=f"real-fill model, orig: Dice {dz('real', 'orig'):.2f}")
        _show(axes[r, 6], sm[z], pr[("real", pert_hi)][z], COL["real"], title=f"real-fill model, smoothed: Dice {dz('real', pert_hi):.2f}", vlim=vl)
        npz0 = SCR / D["name"] / "preds/real/orig" / f"{c}.npz"; npz1 = SCR / D["name"] / "preds/real" / pert_hi / f"{c}.npz"
        if npz0.exists() and npz1.exists():
            p0 = np.load(npz0)["probabilities"][lid]; p1 = np.load(npz1)["probabilities"][lid]
            axes[r, 7].imshow(img[z], cmap="gray", vmin=vl[0] if vl is not None else None, vmax=vl[1] if vl is not None else None)
            im = axes[r, 7].imshow(p1[z] - p0[z], cmap="RdBu_r", vmin=-1, vmax=1, alpha=np.clip(np.abs(p1[z] - p0[z]) * 2, 0, 0.9))
            axes[r, 7].set_title(f"real-fill P({lname}): smoothed − orig  (red = gained, blue = lost)", fontsize=8.5, loc="left"); axes[r, 7].axis("off")
        else:
            axes[r, 7].axis("off")
    fig.suptitle(f"{D['title']} — example slices (slice = largest GT section, display only)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); f = out / f"{D['name']}_examples_{pert_hi}.png"; fig.savefig(f, dpi=110, facecolor="#fcfcfb"); plt.close(fig); print("wrote", f, flush=True)


# ----------------------------------------------------------------------------- flat-fill occlusion map (one case)
def occlusion(D, predictors, case, lid, win, stride, out, max_windows=80, max_seconds=900):
    """Replace each window inside the nonzero mask by its local mean (GT-free), re-predict the FULL volume, record ΔDice and Δ predicted volume."""
    lname = D["labels"][lid]
    src = D["img_dir"] / f"{case}_0000.nii.gz"; itk = sitk.ReadImage(str(src)); img = sitk.GetArrayFromImage(itk).astype(np.float32)
    props = {"spacing": list(itk.GetSpacing())[::-1]}
    g = arr(D["gt_dir"] / f"{case}.nii.gz").round() == lid
    mask = (img != 0) if D["mask_norm"] else np.ones_like(img, dtype=bool)
    base = {arm: predictors[arm].predict_single_npy_array(img[None], props) == lid for arm in predictors}
    union = base["flat"] | base["real"]
    z0 = int(np.argmax(union.sum((1, 2)))) if union.any() else img.shape[0] // 2      # band from the PREDICTIONS, not the GT
    zs = slice(max(0, z0 - win // 2), min(img.shape[0], z0 - win // 2 + win))
    ys = np.arange(0, img.shape[1] - win + 1, stride); xs = np.arange(0, img.shape[2] - win + 1, stride)
    wins = [(y, x) for y in ys for x in xs if mask[zs, y:y + win, x:x + win].mean() > 0.3]
    if len(wins) > max_windows:   # thin the grid uniformly
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, max_windows).round().astype(int)]
    print(f"occlusion {case}: band z={zs.start}:{zs.stop}, {len(wins)} windows of {win}^3 (stride {stride}), baseline Dice flat {dice(base['flat'], g):.3f} real {dice(base['real'], g):.3f}", flush=True)
    rows = []; t0 = time.time()
    for i, (y, x) in enumerate(wins):
        if time.time() - t0 > max_seconds:
            print(f"occlusion: time cap reached after {i} windows", flush=True); break
        occ = img.copy(); sl = (zs, slice(y, y + win), slice(x, x + win)); m = mask[sl]
        if m.any():
            occ[sl][m] = img[sl][m].mean()
        row = dict(case=case, label=lname, z=zs.start, y=y, x=x, win=win, pred_overlap_flat=int(base["flat"][sl].sum()), pred_overlap_real=int(base["real"][sl].sum()), gt_in_window=int(g[sl].sum()))
        for arm, pr in predictors.items():
            seg = pr.predict_single_npy_array(occ[None], props) == lid
            row[f"dice_{arm}"] = dice(seg, g); row[f"ddice_{arm}"] = dice(seg, g) - dice(base[arm], g)
            row[f"dvol_{arm}"] = int(seg.sum()) - int(base[arm].sum()); row[f"dvol_in_win_{arm}"] = int(seg[sl].sum()) - int(base[arm][sl].sum())
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(OUT_TABLES / f"texproof_perturbation_pilot_{D['name']}_occlusion_{case}.csv", index=False)
    # figure: slice z0, image + predicted contours, then ΔDice maps per arm and their difference
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#fcfcfb")
    _show(axes[0], img[z0], None, title=f"{case} slice {z0}: image; contours = unperturbed predictions (blue flat, orange real), GT not drawn")
    for arm in ("flat", "real"):
        axes[0].contour(base[arm][z0].astype(float), levels=[0.5], colors=[COL[arm]], linewidths=0.9)
    vmax = max(1e-3, float(np.nanmax(np.abs(df[["ddice_flat", "ddice_real"]].values))))
    for k, (key, ttl) in enumerate((("ddice_flat", "ΔDice flat-fill model"), ("ddice_real", "ΔDice real-fill model"))):
        _show(axes[1 + k], img[z0], None, title=f"{ttl} when the window is flat-filled (blue = Dice drops)")
        for r in df.itertuples():
            v = getattr(r, key); axes[1 + k].add_patch(plt.Rectangle((r.x, r.y), win, win, color=plt.cm.RdBu_r((v / vmax + 1) / 2), alpha=0.55, lw=0))
    _show(axes[3], img[z0], None, title="ΔDice(real) − ΔDice(flat): blue = real-fill model relies more on this window's texture")
    dd = df.ddice_real - df.ddice_flat; v2 = max(1e-3, float(np.nanmax(np.abs(dd))))
    for r, v in zip(df.itertuples(), dd):
        axes[3].add_patch(plt.Rectangle((r.x, r.y), win, win, color=plt.cm.RdBu_r((v / v2 + 1) / 2), alpha=0.55, lw=0))
    fig.suptitle(f"{D['title']} — flat-fill occlusion map, {len(df)} windows of {win}³ voxels on an axial band chosen from the predictions (GT-free); colour scale ±{vmax:.3f} (panels 2-3), ±{v2:.3f} (panel 4)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); f = out / f"{D['name']}_occlusion_{case}.png"; fig.savefig(f, dpi=120, facecolor="#fcfcfb"); plt.close(fig); print("wrote", f, flush=True)
    inside = df[(df.pred_overlap_flat + df.pred_overlap_real) > 0]; outside = df[(df.pred_overlap_flat + df.pred_overlap_real) == 0]
    print("occlusion summary: windows touching a predicted lesion: n=%d mean ΔDice flat %.4f real %.4f | windows away from predictions: n=%d flat %.4f real %.4f" % (
        len(inside), inside.ddice_flat.mean(), inside.ddice_real.mean(), len(outside), outside.ddice_flat.mean(), outside.ddice_real.mean()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS)); ap.add_argument("--cases", nargs="*"); ap.add_argument("--sigmas", nargs="*", type=float)
    ap.add_argument("--noise", nargs="*", type=float); ap.add_argument("--occlusion", default=None, help="case name for the flat-fill occlusion map")
    ap.add_argument("--occ-win", type=int, default=24); ap.add_argument("--occ-stride", type=int, default=24); ap.add_argument("--occ-max-windows", type=int, default=80); ap.add_argument("--occ-max-seconds", type=int, default=900)
    ap.add_argument("--fig-cases", nargs="*", help="cases for the example figure + probability maps (default: 2 with the largest real−flat Dice at σ=0)")
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    D = dict(DATASETS[a.dataset], name=a.dataset); OUT_PLOTS.mkdir(parents=True, exist_ok=True); OUT_TABLES.mkdir(parents=True, exist_ok=True)
    cases = a.cases or D["default_cases"] or sorted(p.name.replace("_0000.nii.gz", "") for p in D["img_dir"].glob("*_0000.nii.gz"))
    sigmas = a.sigmas if a.sigmas is not None else D["default_sigmas"]; noises = a.noise if a.noise is not None else D["default_noise"]
    t0 = time.time()
    perts = build_inputs(D, cases, sigmas, noises)
    order = ["orig"] + [f"gauss_s{s:g}" for s in sigmas] + [f"noise_k{k:g}" for k in noises]
    pert_hi = f"gauss_s{max(sigmas):g}"
    predictors = {}
    if not a.skip_predict:
        for arm, mdir in D["models"].items():
            predictors[arm] = make_predictor(mdir); print(f"[{time.time() - t0:.0f}s] loaded {arm}: {mdir}", flush=True)
            for p in order:
                predict_dir(predictors[arm], perts[p], SCR / D["name"] / "preds" / arm / p, cases)
                print(f"[{time.time() - t0:.0f}s] predicted {arm} / {p} ({len(cases)} cases)", flush=True)
    csvf = OUT_TABLES / f"texproof_perturbation_pilot_{a.dataset}.csv"
    df = score(D, cases, order, csvf); print("wrote", csvf, flush=True)
    plot_curves(D, df, OUT_PLOTS)
    lid = D["focus"][0]; lname = D["labels"][lid]
    if a.fig_cases:
        fig_cases = a.fig_cases
    else:
        piv = df[(df.pert == "orig") & (df.label == lname)].pivot(index="case", columns="arm", values="dice"); fig_cases = list((piv.real - piv.flat).sort_values(ascending=False).index[:2])
    if not a.skip_predict:
        for arm in predictors:
            for p in ("orig", pert_hi):
                predict_dir(predictors[arm], perts[p], SCR / D["name"] / "preds" / arm / p, cases, probs_for=fig_cases)
    plot_examples(D, df, fig_cases, pert_hi, OUT_PLOTS, lid)
    if a.occlusion and predictors:
        occlusion(D, predictors, a.occlusion, lid, a.occ_win, a.occ_stride, OUT_PLOTS, a.occ_max_windows, a.occ_max_seconds)
    # compact summary for the report
    foc = [D["labels"][l] for l in D["focus"]]; d = df[df.label.isin(foc)]
    piv = d.pivot_table(index=["pert"], columns="arm", values="dice", aggfunc="mean").reindex(order); piv["delta"] = piv.real - piv.flat
    base = d[d.pert == "orig"].groupby("arm").dice.mean(); piv["ret_flat"] = piv.flat / base["flat"]; piv["ret_real"] = piv.real / base["real"]
    print("\nSUMMARY (mean over cases x focus labels)\n" + piv.round(3).to_string(), flush=True)
    (OUT_TABLES / f"texproof_perturbation_pilot_{a.dataset}_summary.md").write_text(f"# texproof pilot summary: {a.dataset}\n\n{D['title']}\n\ncases: {cases}\nfocus labels: {foc}\n\n" + md_table(piv.round(3)) + "\n")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
