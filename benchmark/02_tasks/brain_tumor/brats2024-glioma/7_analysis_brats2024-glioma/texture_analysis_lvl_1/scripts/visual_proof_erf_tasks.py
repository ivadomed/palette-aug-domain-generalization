#!/usr/bin/env python
"""
Visual proof v8 (2026-10-12): the v7 seed-centred evidence map (visual_proof_erf.py) for EVERY paper task with a rung-4.5 flat-fill model,
one panel per task, restricted to structures on which the two models AGREE.
Same method as v7 (per-voxel gradient of the log-odds, fold-mean, unperturbed held-out-contrast test images; seeds = deep target voxels both
models segment correctly and confidently), plus:
  * per (case, class): in-patch Dice of each model's 3-fold-majority prediction is stored with every seed; the report keeps only the seeds of
    (case, class) pairs where |Dice_real - Dice_flat| <= AGREE (0.05): "both models agree on this structure" (set before looking at results);
  * seeds sampled per class (multi-structure tasks: brain 31, pelvis 11, abdomen 4 -> at least 3 per present class, <= 24 per case);
  * cells = every held-out-contrast cell of the task's own test set (and the mandible CT companions); a task pools its cells, one case = one
    unit (equal weight). Images on which a model was trained are never used.
Depth threshold d_min / local radius r_loc per task fixed by structure size before the run: 6 / 3 vox (glioma edema, breast tumour, abdomen
organs, pelvis bones/muscles), 4 / 2 (brain parcels: mostly thin structures), 3 / 2 (MS lesions, pancreas, mandible).
Outputs: outputs/plots/visual_proof_erf_tasks/<cell>/{seeds.csv, <case>_erf.npz}; --report: erf_tasks.png, erf_tasks_stats.md.
GPU: .venv/bin/python visual_proof_erf_tasks.py --cells <cell> [--case-slice i:j]      CPU: --report
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
os.environ.setdefault("VP_OUT", str(THIS.parent / "outputs/plots/visual_proof_v2"))
os.environ.setdefault("VP7_OUT", str(THIS.parent / "outputs/plots/visual_proof_erf"))
import visual_proof_v2 as v2  # noqa: E402
import visual_proof_erf as e7  # noqa: E402

OUT = Path(os.environ.get("VP8_OUT") or (THIS.parent / "outputs/plots/visual_proof_erf_tasks")); OUT.mkdir(parents=True, exist_ok=True)
T2 = v2.T2; AGREE = 0.05; MAX_CASES = int(os.environ.get("ERF_MAX_CASES", 40)); MAX_SEEDS = 24


def run(rid_glob):
    hits = sorted(T2.glob(rid_glob))
    if len(hits) != 1:
        raise FileNotFoundError(f"{rid_glob}: {len(hits)} matches")
    return hits[0]


def M(pred_root, flat, real):
    return {"flat": lambda: run(f"{pred_root}/{flat}/Dataset*/*__3d_fullres"), "real": lambda: run(f"{pred_root}/{real}/Dataset*/*__3d_fullres")}


def nn(img_dir, gt_dir, keep=None):
    """cases + path functions for an nnU-Net style test pair (images <case>_0000.nii.gz, labels <case>.nii.gz)."""
    I, G = T2 / img_dir, T2 / gt_dir
    cs = sorted(p.name[:-12] for p in I.glob("*_0000.nii.gz") if (G / f"{p.name[:-12]}.nii.gz").exists())
    cs = [c for c in cs if keep is None or keep(c)]
    return dict(cases=cs, img=lambda c: I / f"{c}_0000.nii.gz", gt=lambda c: G / f"{c}.nii.gz")


def onh(contrast):
    I = T2 / f"brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model/_test_set/{contrast}"
    cs = sorted(p.name[:-12] for p in (I / "images_ras").glob("*_0000.nii.gz") if (I / "gt_native" / f"{p.name[:-12]}.nii.gz").exists())
    return dict(cases=cs, img=lambda c: I / "images_ras" / f"{c}_0000.nii.gz", gt=lambda c: I / "gt_native" / f"{c}.nii.gz")


GL = "brain_tumor/brats2024-glioma"; MS = "brain_ms/open-ms"; ONH = "brain_healthy/on-harmony"; CH = "abdomen_healthy/chaos"; TF = "mandible_healthy"
BR = "breast_cancer/ispy2"; PE = "pelvis_healthy/totalseg-pelvic"; PA = "pancreas_disease/pansegdata"
GLP = f"{GL}/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model"; GLR = f"{GL}/2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w"
MSP = f"{MS}/8_results_open-ms/01_predictions/open_ms_model"; ONP = f"{ONH}/8_results_on-harmony/01_predictions/on_harmony_model"
CHP = f"{CH}/8_results_chaos/01_predictions/chaos_model"; CHR = f"{CH}/2_nnUNet_chaos/raw"; TFP = f"{TF}/toothfairy2/8_results_toothfairy2/01_predictions/toothfairy2_model/cbct/auglab"
BRP = f"{BR}/8_results_ispy2/01_predictions/ispy2_model"; BRR = f"{BR}/2_nnUNet_ispy2/raw"; PEP = f"{PE}/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model"
PER = f"{PE}/2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT"; PAP = f"{PA}/8_results_pansegdata/01_predictions/pansegdata_model"; PAR = f"{PA}/2_nnUNet_pansegdata/raw"
G_T2W = M(f"{GLP}/t2w/auglab", "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190652", "brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300")
G_T1N = M(f"{GLP}/t1n/auglab", "brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190618", "brats2024-glioma_t1n_v26_6_2_train050_val000_20261005_222227")
MS_FL = M(f"{MSP}/flair/nnUNet", "open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190831", "open-ms_flair_v26_6_2_train050_val000_20260711_062946")
MS_T1 = M(f"{MSP}/t1w/nnUNet", "open-ms_t1w_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190904", "open-ms_t1w_v26_6_2_train050_val000_20261005_222621")
ON_T1 = M(f"{ONP}/T1w/auglab", "on-harmony_T1w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_211657", "on-harmony_T1w_v26_6_2_train050_val000_20261005_222440")
ON_T2 = M(f"{ONP}/T2w/auglab", "on-harmony_T2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_211657", "on-harmony_T2w_v26_6_2_train050_val000_20261005_222513")
CH_T1 = M(f"{CHP}/t1in/nnUNet", "chaos_t1in_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190938", "chaos_t1in_v26_6_2_train050_val000_20260715_082040")
CH_T2 = M(f"{CHP}/t2spir/nnUNet", "chaos_t2spir_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_191011", "chaos_t2spir_v26_6_2_train050_val000_20261005_222654")
TF_CB = M(TFP, "toothfairy2_cbct_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191044", "toothfairy2_cbct_v26_6_2_train050_val000_20261005_222846")
BR_T1 = M(f"{BRP}/t1wce/auglab", "ispy2_t1wce_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191117", "ispy2_t1wce_v26_6_2_train050_val000_20261005_222919")
BR_T2 = M(f"{BRP}/t2w/auglab", "ispy2_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191151", "ispy2_t2w_v26_6_2_train050_val000_20261005_222952")
PE_MR = M(f"{PEP}/mri/auglab", "totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257", "totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058")
PE_CT = M(f"{PEP}/ct/auglab", "totalseg-pelvic_ct_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191224", "totalseg-pelvic_ct_v26_6_2_train050_val000_20261005_223025")
PA_T1 = M(f"{PAP}/t1wce/auglab", "pansegdata_t1wce_baseline_kmeans_label_remap_voronoi_flatfill_20261009_080433", "pansegdata_t1wce_v26_6_2_train050_val000_20261005_223237")
PA_T2 = M(f"{PAP}/t2w/auglab", "pansegdata_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261009_080507", "pansegdata_t2w_v26_6_2_train050_val000_20261005_223311")
UNI = lambda c: c.endswith("_uni")   # noqa: E731  (I-SPY2: one FOV variant per patient, the unilateral one = the training FOV)
S6, S4, S3 = dict(d_min=6, r_loc=3), dict(d_min=4, r_loc=2), dict(d_min=3, r_loc=2)
CELLS = {
    "glioma_t2w_t1n": dict(task="Glioma", models=G_T2W, target=[2], **S6, **nn(f"{GLR}/imagesTs_t1n", f"{GLR}/labelsTr")),
    "glioma_t2w_t2f": dict(task="Glioma", models=G_T2W, target=[2], **S6, **nn(f"{GLR}/imagesTs_t2f", f"{GLR}/labelsTr")),
    "glioma_t1n_t2f": dict(task="Glioma", models=G_T1N, target=[2], **S6, **nn(f"{GLR}/imagesTs_t2f", f"{GLR}/labelsTr")),
    "glioma_t1n_t2w": dict(task="Glioma", models=G_T1N, target=[2], **S6, **nn(f"{GLR}/imagesTs_t2w", f"{GLR}/labelsTr")),
    "ms_flair_t2w": dict(task="MS", models=MS_FL, target=[1], **S3, **nn(f"{MS}/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/imagesTs_t2w", f"{MS}/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/labelsTs_t2w")),
    "ms_flair_t1w": dict(task="MS", models=MS_FL, target=[1], **S3, **nn(f"{MS}/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/imagesTs_t1w", f"{MS}/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/labelsTs_t1w")),
    "ms_t1w_flair": dict(task="MS", models=MS_T1, target=[1], **S3, **nn(f"{MS}/2_nnUNet_open-ms/raw/Dataset071_OpenMS_T1W/imagesTs_flair", f"{MS}/2_nnUNet_open-ms/raw/Dataset071_OpenMS_T1W/labelsTs_flair")),
    "ms_t1w_t2w": dict(task="MS", models=MS_T1, target=[1], **S3, **nn(f"{MS}/2_nnUNet_open-ms/raw/Dataset071_OpenMS_T1W/imagesTs_t2w", f"{MS}/2_nnUNet_open-ms/raw/Dataset071_OpenMS_T1W/labelsTs_t2w")),
    "brain_t1w_t2w": dict(task="Brain", models=ON_T1, target=list(range(1, 32)), **S4, **onh("T2w")),
    "brain_t2w_t1w": dict(task="Brain", models=ON_T2, target=list(range(1, 32)), **S4, **onh("T1w")),
    "abd_t1in_t2spir": dict(task="Abdomen", models=CH_T1, target=[1, 2, 3, 4], **S6, **nn(f"{CHR}/Dataset060_CHAOS_MR_T1in/imagesTs_t2spir", f"{CHR}/Dataset060_CHAOS_MR_T1in/labelsTs_t2spir")),
    "abd_t1in_ct": dict(task="Abdomen", models=CH_T1, target=[1, 2, 3, 4], **S6, **nn(f"{CHR}/Dataset060_CHAOS_MR_T1in/imagesTs_ct", f"{CHR}/Dataset060_CHAOS_MR_T1in/labelsTs_ct")),
    "abd_t2spir_t1in": dict(task="Abdomen", models=CH_T2, target=[1, 2, 3, 4], **S6, **nn(f"{CHR}/Dataset061_CHAOS_MR_T2spir/imagesTs_t1in", f"{CHR}/Dataset061_CHAOS_MR_T2spir/labelsTs_t1in")),
    "abd_t2spir_t1out": dict(task="Abdomen", models=CH_T2, target=[1, 2, 3, 4], **S6, **nn(f"{CHR}/Dataset061_CHAOS_MR_T2spir/imagesTs_t1out", f"{CHR}/Dataset061_CHAOS_MR_T2spir/labelsTs_t1out")),
    "abd_t2spir_ct": dict(task="Abdomen", models=CH_T2, target=[1, 2, 3, 4], **S6, **nn(f"{CHR}/Dataset061_CHAOS_MR_T2spir/imagesTs_ct", f"{CHR}/Dataset061_CHAOS_MR_T2spir/labelsTs_ct")),
    "mand_cbct_hanseg_ct": dict(task="Mandible", models=TF_CB, target=[1], **S3, **nn(f"{TF}/hanseg/2_nnUNet_hanseg/raw/imagesTs_ct", f"{TF}/hanseg/2_nnUNet_hanseg/raw/labelsTs_ct")),
    "mand_cbct_pddca_ct": dict(task="Mandible", models=TF_CB, target=[1], **S3, **nn(f"{TF}/pddca/2_nnUNet_pddca/raw/imagesTs_ct", f"{TF}/pddca/2_nnUNet_pddca/raw/labelsTs_ct")),
    "breast_t1wce_t2w": dict(task="Breast", models=BR_T1, target=[1], **S6, **nn(f"{BRR}/Dataset100_ISPY2T1wce/imagesTs_t2w", f"{BRR}/Dataset100_ISPY2T1wce/labelsTs_t2w", UNI)),
    "breast_t2w_t1wce": dict(task="Breast", models=BR_T2, target=[1], **S6, **nn(f"{BRR}/Dataset101_ISPY2T2w/imagesTs_t1wce", f"{BRR}/Dataset101_ISPY2T2w/labelsTs_t1wce", UNI)),
    "pelvis_mri_ct": dict(task="Pelvis", models=PE_MR, target=list(range(1, 12)), **S6, **nn(f"{PER}/imagesTs_ct", f"{PER}/labelsTs_ct")),
    "pelvis_ct_mri": dict(task="Pelvis", models=PE_CT, target=list(range(1, 12)), **S6, **nn(f"{PER}/imagesTs_mri", f"{PER}/labelsTs_mri")),
    "pancreas_t1wce_t2w": dict(task="Pancreas", models=PA_T1, target=[1], **S3, **nn(f"{PAR}/Dataset150_PanSegData_T1WCE/imagesTs_t2w", f"{PAR}/Dataset150_PanSegData_T1WCE/labelsTs_t2w")),
    "pancreas_t2w_t1wce": dict(task="Pancreas", models=PA_T2, target=[1], **S3, **nn(f"{PAR}/Dataset151_PanSegData_T2W/imagesTs_t1wce", f"{PAR}/Dataset151_PanSegData_T2W/labelsTs_t1wce")),
}
TASKS = ["Glioma", "MS", "Brain", "Abdomen", "Mandible", "Breast", "Pelvis", "Pancreas"]
KIND = {"Glioma": "appearance", "MS": "appearance", "Breast": "appearance", "Brain": "interface", "Abdomen": "interface", "Mandible": "interface",
        "Pelvis": "interface", "Pancreas": "interface"}   # the paper's grouping (appearance-defined vs interface-defined target)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def run_cell(key, dev, sl=None):
    cell = CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True); T = cell["target"]; r_loc, d_min = cell["r_loc"], cell["d_min"]
    prs = {a: v2.load_predictor(f(), dev) for a, f in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"])[:MAX_CASES] if not key.startswith("glioma") else list(cell["cases"])
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "seeds.csv"; done = set(pd.read_csv(f_csv).case.astype(str)) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    f_skip = od / "skipped.csv"; rng = np.random.default_rng(0)
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        if tg.sum() < 50:
            pd.DataFrame([dict(case=case, why="target < 50 vox")]).to_csv(f_skip, mode="a", index=False, header=not f_skip.exists()); continue
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        lo, maj = {}, {}
        for a, pr in prs.items():
            net = pr.network; acc = 0; votes = []
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()
                acc = acc + o[0]; votes.append(o[0].argmax(0))
            lo[a] = (acc / len(v2.FOLDS)).cpu().numpy(); maj[a] = torch.mode(torch.stack(votes), 0).values.cpu().numpy()
        geo = {c: e7.region_geometry(gp, c) for c in T if (gp == c).sum() >= 50}
        dsc = {c: {a: dice(maj[a] == c, gp == c) for a in ARMS} for c in geo}
        inner = np.zeros(gp.shape, bool); inner[tuple(slice(e7.EDGE, s - e7.EDGE) for s in gp.shape)] = True
        per_c = max(3, int(np.ceil(e7.N_SEEDS / max(1, len(geo))))); seeds, cls, depth = [], [], []
        for c, (reg, din, rim) in geo.items():
            oth = [j for j in range(lo["flat"].shape[0]) if j != c]; ok = reg & inner & (din >= d_min) & bp
            for a in ARMS:
                mx = lo[a][oth].max(0); la = lo[a][c] - (np.log(np.exp(lo[a][oth] - mx).sum(0)) + mx); ok &= (maj[a] == c) & (la > e7.CONF)
            idx = np.argwhere(ok)
            if not len(idx):
                continue
            dep = din[tuple(idx.T)]; qs = np.quantile(dep, [0, 1 / 3, 2 / 3, 1]); pick = []
            for b in range(3):
                ib = np.where((dep >= qs[b]) & (dep <= qs[b + 1]))[0]
                pick += list(rng.choice(ib, min(len(ib), max(1, per_c // 3)), replace=False)) if len(ib) else []
            pick = list(dict.fromkeys(pick))
            seeds += [tuple(idx[i]) for i in pick]; cls += [int(c)] * len(pick); depth += [float(dep[i]) for i in pick]
        if len(seeds) > MAX_SEEDS:
            keep = sorted(rng.choice(len(seeds), MAX_SEEDS, replace=False)); seeds = [seeds[i] for i in keep]; cls = [cls[i] for i in keep]; depth = [depth[i] for i in keep]
        if not seeds:
            pd.DataFrame([dict(case=case, why=f"no seed (both correct + confident, depth >= {d_min})")]).to_csv(f_skip, mode="a", index=False, header=not f_skip.exists())
            log(f"[{key}] {case}: no seed"); continue
        Gs = {}
        for a, pr in prs.items():
            net = pr.network; acc = None
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                g = e7.seed_grads(net, ip, seeds, cls, dev); acc = g if acc is None else [x + y for x, y in zip(acc, g)]
            Gs[a] = [(x / len(v2.FOLDS)).abs().cpu().numpy() for x in acc]
        net = prs["flat"].network; torch.manual_seed(0)
        for m in net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        net.eval(); Gs["random"] = [x.abs().cpu().numpy() for x in e7.seed_grads(net, ip, seeds, cls, dev)]
        rows = []
        for i, (v, c, d) in enumerate(zip(seeds, cls, depth)):
            reg, din, rim = geo[c]
            row = dict(cell=key, task=cell["task"], case=case, seed=i, z=v[0], y=v[1], x=v[2], cls=c, depth=d, dice_flat=dsc[c]["flat"], dice_real=dsc[c]["real"])
            for a in ARMS + ("random",):
                for mk, mv in e7.seed_metrics(Gs[a][i], v, rim, bp, r_loc).items():
                    row[f"{a}_{mk}"] = mv
            rows.append(row)
        sv = {"img": [], "gt": [], "G_flat": [], "G_real": [], "G_random": []}
        for i, (v, c) in enumerate(zip(seeds, cls)):
            z = v[0]; sv["img"].append(ip[z]); sv["gt"].append(gp[z] == c)
            for a in ARMS + ("random",):
                sv[f"G_{a}"].append(Gs[a][i][z])
        np.savez_compressed(od / f"{case}_erf.npz", seeds=np.array(seeds), cls=np.array(cls), depth=np.array(depth),
                            **{k: np.array(vv).astype(np.float16 if k != "gt" else bool) for k, vv in sv.items()})
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)
        log(f"[{key}] {case}: {len(seeds)} seeds, classes {sorted(set(cls))}, Dice flat/real " +
            ", ".join(f"{c}:{dsc[c]['flat']:.2f}/{dsc[c]['real']:.2f}" for c in sorted(set(cls))) + f"; {time.time() - t0:.0f} s")


ARMS = e7.ARMS


def report(W=30, LIM=0.4, SMOOTH=1.0, SUPPORT=0.0005):
    from scipy.stats import wilcoxon
    rows = []
    for key in CELLS:
        f = OUT / key / "seeds.csv"
        if f.exists():
            rows.append(e7.num(pd.read_csv(f)))
    if not rows:
        return
    S = pd.concat(rows); S["agree"] = (S.dice_real - S.dice_flat).abs() <= AGREE
    L = ["# Visual proof v8: seed-centred evidence maps per task, structures where both models agree (|Dice_real - Dice_flat| <= 0.05)", "",
         "inside share = in-plane share of the evidence inside the target structure (body-masked, map normalised); per case = median over its agreeing seeds.", "",
         "| task | target kind | cells | cases (agreeing / with seeds) | seeds kept | median Dice flat / real (kept) | inside share flat | real | random | real > flat | p | local share flat | real | p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for task in TASKS:
        st = S[(S.task == task) & S.agree]
        if st.case.nunique() == 0:
            L.append(f"| {task} | {KIND[task]} | {S[S.task == task].cell.nunique()} | 0 / {S[S.task == task].case.nunique()} | 0 | | | | | | | | | |"); continue
        maps = {a: [] for a in ARMS}; gts = []; ins = {a: [] for a in ARMS + ("random",)}; loc = {a: [] for a in ARMS}
        for (cell, case), g in st.groupby(["cell", "case"]):
            z = np.load(OUT / cell / f"{case}_erf.npz"); acc = {a: [] for a in ARMS}; ga = []; ii = {a: [] for a in ins}
            for i in g.seed.astype(int):
                sd = z["seeds"][i]; body = z["img"][i] != 0; reg = z["gt"][i] & body
                for a in ins:
                    G = z[f"G_{a}"][i].astype(float) * body; G = G / (G.sum() + 1e-30); ii[a].append(G[reg].sum())
                    if a in ARMS:
                        acc[a].append(np.pad(G, W, constant_values=np.nan)[sd[1]:sd[1] + 2 * W + 1, sd[2]:sd[2] + 2 * W + 1])
                ga.append(np.pad(z["gt"][i].astype(float), W, constant_values=np.nan)[sd[1]:sd[1] + 2 * W + 1, sd[2]:sd[2] + 2 * W + 1])
            for a in ARMS:
                maps[a].append(np.nanmean(acc[a], 0)); loc[a].append(g[f"{a}_local_share"].median())
            for a in ins:
                ins[a].append(np.median(ii[a]))
            gts.append(np.nanmean(ga, 0))
        Mm = {a: ndi.gaussian_filter(np.nan_to_num(np.nanmean(v, 0)), SMOOTH) for a, v in maps.items()}
        d = np.array(ins["real"]) - np.array(ins["flat"]); dl = np.array(loc["real"]) - np.array(loc["flat"]); n = len(d)
        pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan   # noqa: E731
        sup = 0.5 * (Mm["real"] + Mm["flat"]); Lr = np.log2(Mm["real"] / Mm["flat"]); Lr[sup < SUPPORT * sup.max()] = np.nan   # no-evidence area (e.g. air): ratio undefined
        res[task] = (Lr, np.nanmean(gts, 0), n, d, pw(d))
        L.append(f"| {task} | {KIND[task]} | {st.cell.nunique()} | {n} / {S[S.task == task].groupby(['cell', 'case']).ngroups} | {len(st)} | "
                 f"{st.dice_flat.median():.2f} / {st.dice_real.median():.2f} | {np.median(ins['flat']):.3f} | {np.median(ins['real']):.3f} | {np.median(ins['random']):.3f} | "
                 f"{np.mean(d > 0) * 100:.0f}% | {pw(d):.2g} | {np.median(loc['flat']):.4f} | {np.median(loc['real']):.4f} | {pw(dl):.2g} |")
    (OUT / "erf_tasks_stats.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    report_cells(S); report_cells(S, core=3); report_ctrl()
    if not res:
        return
    ext = [-W - 0.5, W + 0.5, W + 0.5, -W - 0.5]; nc = 4; nr = int(np.ceil(len(res) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc + 0.9, 3.7 * nr), squeeze=False)
    for j, (task, (Lm, gt, n, d, p)) in enumerate(res.items()):
        ax = axs[j // nc, j % nc]; cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.75"); im = ax.imshow(Lm, cmap=cm, vmin=-LIM, vmax=LIM, extent=ext)
        ax.contour(np.arange(-W, W + 1), np.arange(-W, W + 1), gt, [0.5], colors="k", linewidths=1); ax.plot(0, 0, "+", color="lime", ms=9, mew=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{task} ({KIND[task]}-defined)\n{n} agreeing cases\nevidence in target: real > flat in {np.mean(d > 0) * 100:.0f}% (p {p:.1g})", fontsize=8)
    for j in range(len(res), nr * nc):
        axs[j // nc, j % nc].axis("off")
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.015, pad=0.01); cb.set_label("log2 evidence, real-fill / flat-fill", fontsize=8)
    fig.suptitle("Where each model takes its evidence for a deep target voxel (+), averaged over voxels of structures both models segment equally well "
                 f"(|dDice| <= {AGREE}); red = real-fill model relies more; black = target region; maps normalised; {SMOOTH:g} mm display smoothing; grey = no evidence in either model (< {SUPPORT * 100:g}% of peak, e.g. air)", fontsize=8)
    fig.savefig(OUT / "erf_tasks.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# texture step 4.5 -> 5 per TRAINING setting (Dice points, paper tab:flatfill, $SCRATCH/rung45_flat/paper_gen/flat_dice.txt, 2026-10-12)
TEX_STEP = {"glioma_t2w": 2.83, "glioma_t1n": 2.99, "ms_flair": 7.06, "ms_t1w": 10.41, "brain_t1w": -1.07, "brain_t2w": -0.70, "abd_t1in": 0.31,
            "abd_t2spir": 0.64, "mand_cbct": 0.48, "breast_t1wce": 0.60, "breast_t2w": 2.11, "pelvis_mri": -2.76, "pelvis_ct": 2.01,
            "pancreas_t1wce": -0.99, "pancreas_t2w": 0.17}
TCOL = dict(zip(TASKS, plt.get_cmap("tab10").colors))


def cell_maps(S, W, SMOOTH, core=0, tag="erf"):
    """per cell: per-case inside-share difference (real - flat), seed-centred mean maps, mean target mask, n voxels."""
    out = {}
    for cell, sc in S.groupby("cell", sort=False):
        maps = {a: [] for a in ARMS}; gts = []; dif = []; nv = 0
        for case, g in sc.groupby("case"):
            z = np.load(OUT / cell / f"{case}_{tag}.npz"); acc = {a: [] for a in ARMS}; ga = []; ii = {a: [] for a in ARMS}
            for i in g.seed.astype(int):
                sd = z["seeds"][i]; body = z["img"][i] != 0; nv += 1
                if core:   # drop the explained voxel's own neighbourhood (disc of radius `core` in-plane) before normalising
                    yy, xx = np.indices(body.shape); body = body & (np.hypot(yy - sd[1], xx - sd[2]) > core)
                reg = z["gt"][i] & body
                for a in ARMS:
                    G = z[f"G_{a}"][i].astype(float) * body; G = G / (G.sum() + 1e-30); ii[a].append(G[reg].sum())
                    acc[a].append(np.pad(G, W, constant_values=np.nan)[sd[1]:sd[1] + 2 * W + 1, sd[2]:sd[2] + 2 * W + 1])
                ga.append(np.pad(z["gt"][i].astype(float), W, constant_values=np.nan)[sd[1]:sd[1] + 2 * W + 1, sd[2]:sd[2] + 2 * W + 1])
            for a in ARMS:
                maps[a].append(np.nanmean(acc[a], 0))
            gts.append(np.nanmean(ga, 0)); dif.append(np.median(ii["real"]) - np.median(ii["flat"]))
        Mm = {a: ndi.gaussian_filter(np.nan_to_num(np.nanmean(v, 0)), SMOOTH) for a, v in maps.items()}
        out[cell] = dict(L=np.log2(Mm["real"] / Mm["flat"]), sup=0.5 * (Mm["real"] + Mm["flat"]), gt=np.nanmean(gts, 0), d=np.array(dif), n_case=len(dif), n_vox=nv)
    return out


def boot_ci(x, n=2000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x)
    if len(x) < 3:
        return np.nan, np.nan
    m = [np.median(r.choice(x, len(x))) for _ in range(n)]; return np.percentile(m, 2.5), np.percentile(m, 97.5)


def report_cells(S, W=30, LIM=0.4, SMOOTH=1.0, SUPPORT=0.0005, core=0):
    from scipy.stats import wilcoxon, spearmanr
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    A = cell_maps(S[S.agree], W, SMOOTH, core); U = cell_maps(S, W, SMOOTH, core); cells = [c for c in CELLS if c in A]
    suf = f"_nocore{core}" if core else ""; ctxt = f" EXCLUDING a {core} mm disc around the explained voxel (renormalised)." if core else ""
    title = lambda c: c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    # 1) one panel per cell
    nc = 5; nr = int(np.ceil(len(cells) / nc)); ext = [-W - 0.5, W + 0.5, W + 0.5, -W - 0.5]; xx = np.arange(-W, W + 1)
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc + 0.9, 3.5 * nr), squeeze=False)
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.75")
    for j, c in enumerate(cells):
        r = A[c]; L = r["L"].copy(); L[r["sup"] < SUPPORT * r["sup"].max()] = np.nan; ax = axs[j // nc, j % nc]
        if core:
            L[np.hypot(*np.meshgrid(xx, xx)) <= core] = np.nan
        im = ax.imshow(L, cmap=cm, vmin=-LIM, vmax=LIM, extent=ext)
        ax.contour(xx, xx, r["gt"], [0.25, 0.5, 0.75], colors="k", linewidths=[0.6, 1.1, 0.6], linestyles=["dotted", "solid", "dashed"])
        ax.plot(0, 0, "+", color="lime", ms=8, mew=2); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{CELLS[c]['task']}: {title(c)}\n{r['n_case']} cases, {r['n_vox']} voxels\nreal > flat in target: {np.mean(r['d'] > 0) * 100:.0f}% (p {pw(r['d']):.1g})", fontsize=7.5)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.012, pad=0.01); cb.set_label("log2 evidence, real-fill / flat-fill", fontsize=8)
    fig.suptitle("Per cell (training -> test contrast): evidence for a deep target voxel (+), averaged over all explained voxels of the cell (each crop centred on its voxel; "
                 f"structures both models segment equally well, |dDice| <= {AGREE}).\nRed = real-fill model relies more. Contours = fraction of crops in which the pixel is inside the target: "
                 "25 % dotted, 50 % solid, 75 % dashed. Grey = no evidence in either model." + ctxt, fontsize=8)
    fig.savefig(OUT / f"erf_cells{suf}.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # 2) summary: forest plot of the per-cell shift + scatter against the ladder's texture step
    rows = []
    for c in cells:
        for tag, R in (("agree", A), ("all", U)):
            if c in R:
                d = R[c]["d"]; lo, hi = boot_ci(d)
                rows.append(dict(cell=c, task=CELLS[c]["task"], set=tag, n=len(d), med=np.median(d), lo=lo, hi=hi, p=pw(d), pos=np.mean(d > 0),
                                 tex=TEX_STEP.get("_".join(c.split("_")[:2]), np.nan)))
    D = pd.DataFrame(rows); D.to_csv(OUT / f"erf_cells_summary{suf}.csv", index=False)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.32 * len(cells) + 2.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for k, c in enumerate(cells[::-1]):
        for tag, mk, off in (("agree", "o", 0.12), ("all", "o", -0.12)):
            r = D[(D.cell == c) & (D.set == tag)]
            if len(r):
                r = r.iloc[0]; col = TCOL[r.task]
                ax.errorbar(r.med * 100, k + off, xerr=[[max(0, (r.med - r.lo) * 100)], [max(0, (r.hi - r.med) * 100)]], fmt=mk, color=col, mfc=col if tag == "agree" else "white", ms=5, lw=1)
        ra = D[(D.cell == c) & (D.set == "agree")].iloc[0]
        ax.text(1.01, k, f"n={ra.n}  p={ra.p:.1g}", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(cells))); ax.set_yticklabels([f"{CELLS[c]['task']}: {title(c)}" for c in cells[::-1]], fontsize=7.5)
    ax.set_xlabel("evidence inside the target, real-fill minus flat-fill (percentage points; median over cases, 95% bootstrap CI)", fontsize=8)
    ax.set_title("filled = structures both models segment equally well (n, p shown); hollow = all structures" + ctxt, fontsize=8)
    Da = D[D.set == "agree"].dropna(subset=["tex"])
    for _, r in Da.iterrows():
        bx.errorbar(r.tex, r.med * 100, yerr=[[max(0, (r.med - r.lo) * 100)], [max(0, (r.hi - r.med) * 100)]], fmt="o", color=TCOL[r.task], ms=4 + 6 * min(1, r.n / 40), lw=0.8)
    for t in TASKS:
        bx.plot([], [], "o", color=TCOL[t], label=t)
    rho, pr = spearmanr(Da.tex, Da.med) if len(Da) >= 4 else (np.nan, np.nan)
    bx.axhline(0, color="k", lw=0.6); bx.axvline(0, color="k", lw=0.6); bx.legend(fontsize=7, ncol=2)
    bx.set_xlabel("ladder texture step 4.5 -> 5 of the training setting (Dice points)", fontsize=8); bx.set_ylabel("evidence-inside shift (pp)", fontsize=8)
    bx.set_title(f"per cell, agreeing structures: Spearman {rho:+.2f} (p {pr:.2g}, {len(Da)} cells; marker size ~ n cases)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"erf_cells_summary{suf}.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    print(D.round(4).to_string())


def report_ctrl(W=30, LIM=0.4, SMOOTH=1.0, SUPPORT=0.0005):
    """Matched-control voxels (run_ctrl), shown SEPARATELY from the target voxels: (1) per-cell seed-centred maps of the controls, same layout
    as erf_cells.png; (2) per cell, real - flat local share (3D ball r_loc around the explained voxel) for target voxels vs their matched controls,
    and the difference (target shift - control shift), Wilcoxon over cases."""
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    T_, C_ = [], []
    for key in CELLS:
        ft, fc = OUT / key / "seeds.csv", OUT / key / "seeds_ctrl.csv"
        if ft.exists() and fc.exists():
            T_.append(e7.num(pd.read_csv(ft))); C_.append(e7.num(pd.read_csv(fc)))
    if not C_:
        return
    St, Sc = pd.concat(T_), pd.concat(C_)
    for d in (St, Sc):
        d["case"] = d.case.astype(str); d["agree"] = (d.dice_real - d.dice_flat).abs() <= AGREE
    St = St.merge(Sc[["cell", "case", "pair"]].rename(columns={"pair": "seed"}), on=["cell", "case", "seed"])   # target seeds that got a control
    title = lambda c: c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    for sel, suf, lab in ((True, "", "agreeing structures"), (False, "_all", "all structures (no agreement filter)")):
        St_, Sc_ = (St[St.agree], Sc[Sc.agree]) if sel else (St, Sc)
        _ctrl_set(St_, Sc_, suf, lab, title, pw, W, LIM, SMOOTH, SUPPORT)


def _ctrl_set(St, Sc, suf, lab, title, pw, W, LIM, SMOOTH, SUPPORT):
    """One selection of report_ctrl (agreeing structures, or all): control maps erf_cells_ctrl{suf}.png and erf_ctrl_summary{suf}.png/.csv."""
    A = cell_maps(Sc, W, SMOOTH, tag="ctrl"); cells = [c for c in CELLS if c in A]
    nc = 5; nr = int(np.ceil(len(cells) / nc)); ext = [-W - 0.5, W + 0.5, W + 0.5, -W - 0.5]; xx = np.arange(-W, W + 1)
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc + 0.9, 3.5 * nr), squeeze=False); cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("0.75")
    for j, c in enumerate(cells):
        r = A[c]; L = r["L"].copy(); L[r["sup"] < SUPPORT * r["sup"].max()] = np.nan; ax = axs[j // nc, j % nc]
        im = ax.imshow(L, cmap=cm, vmin=-LIM, vmax=LIM, extent=ext)
        ax.contour(xx, xx, r["gt"], [0.25, 0.5, 0.75], colors="k", linewidths=[0.6, 1.1, 0.6], linestyles=["dotted", "solid", "dashed"])
        ax.plot(0, 0, "x", color="lime", ms=8, mew=2); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{CELLS[c]['task']}: {title(c)} (CONTROL)\n{r['n_case']} cases, {r['n_vox']} voxels", fontsize=7.5)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.012, pad=0.01); cb.set_label("log2 evidence, real-fill / flat-fill", fontsize=8)
    fig.suptitle("MATCHED CONTROL voxels (x): background voxels outside the target at the same distance from it as the target voxel is deep, "
                 "both models confident; same cases; " + lab + ".\nRed = real-fill model relies more. Contours = where the TARGET lies around the control voxel (25/50/75 %).", fontsize=8)
    fig.savefig(OUT / f"erf_cells_ctrl{suf}.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    rows = []
    for c in CELLS:
        t, k = St[St.cell == c], Sc[Sc.cell == c]
        if not len(k):
            continue
        dt = t.assign(d=t.real_local_share - t.flat_local_share).groupby("case").d.median()
        dc = k.assign(d=k.real_local_share - k.flat_local_share).groupby("case").d.median(); cs = dt.index.intersection(dc.index)
        dd = (dt[cs] - dc[cs]).values
        rows.append(dict(cell=c, task=CELLS[c]["task"], n=len(cs), target=np.median(dt[cs]), t_lo=boot_ci(dt[cs].values)[0], t_hi=boot_ci(dt[cs].values)[1], p_target=pw(dt[cs].values),
                         control=np.median(dc[cs]), c_lo=boot_ci(dc[cs].values)[0], c_hi=boot_ci(dc[cs].values)[1], p_control=pw(dc[cs].values),
                         diff=np.median(dd), p_diff=pw(dd), tex=TEX_STEP.get("_".join(c.split("_")[:2]), np.nan)))
    D = pd.DataFrame(rows); D.to_csv(OUT / f"erf_ctrl_summary{suf}.csv", index=False); print(D.round(5).to_string())
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(D) + 1.8)); cl = D.cell.tolist()[::-1]
    for k_, c in enumerate(cl):
        r = D[D.cell == c].iloc[0]; col = TCOL[r.task]
        ax.errorbar(r.target * 100, k_ + 0.14, xerr=[[max(0, (r.target - r.t_lo) * 100)], [max(0, (r.t_hi - r.target) * 100)]], fmt="o", color=col, ms=5, lw=1)
        ax.errorbar(r.control * 100, k_ - 0.14, xerr=[[max(0, (r.control - r.c_lo) * 100)], [max(0, (r.c_hi - r.control) * 100)]], fmt="s", color=col, mfc="white", ms=5, lw=1)
        ax.text(1.01, k_, f"n={r.n}  target-control p={r.p_diff:.1g}", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(cl))); ax.set_yticklabels([f"{CELLS[c]['task']}: {title(c)}" for c in cl], fontsize=7.5)
    ax.set_xlabel(f"evidence within r_loc of the explained voxel, real-fill minus flat-fill (pp; median over cases, 95% bootstrap CI)", fontsize=8)
    ax.set_title("circle = deep TARGET voxel; square = its MATCHED CONTROL (background at the same distance from the target); " + lab, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"erf_ctrl_summary{suf}.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def run_ctrl(key, dev, sl=None):
    """MATCHED CONTROL voxels for the target seeds of run_cell: for every target seed (depth d inside its class region), one random voxel
    OUTSIDE every target label, in the body, >= EDGE from the patch faces, at distance d (+-0.5, widened to +-1.5 if empty) from the same class
    region, predicted background by both models' 3-fold majority with background log-odds > CONF in both. Explained the same way (gradient of
    its own-class = background log-odds, fold-mean). Same case, same patch. Outputs <cell>/seeds_ctrl.csv, <case>_ctrl.npz (pair = target seed index)."""
    cell = CELLS[key]; od = OUT / key; T = cell["target"]; r_loc = cell["r_loc"]
    f_t = od / "seeds.csv"
    if not f_t.exists():
        return
    St = e7.num(pd.read_csv(f_t)); St["case"] = St.case.astype(str)
    prs = {a: v2.load_predictor(f(), dev) for a, f in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = [c for c in (list(cell["cases"])[:MAX_CASES] if not key.startswith("glioma") else list(cell["cases"])) if c in set(St.case)]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "seeds_ctrl.csv"; done = set(pd.read_csv(f_csv).case.astype(str)) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    rng = np.random.default_rng(1)
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        g = St[St.case == case].sort_values("seed")
        if not np.all(np.isin(g.cls.astype(int), T)) or any(not (gp[int(r.z), int(r.y), int(r.x)] == int(r.cls)) for r in g.itertuples()):
            log(f"[{key}] {case}: patch does not reproduce the target seeds, skipped"); continue
        lo, maj = {}, {}
        for a, pr in prs.items():
            net = pr.network; acc = 0; votes = []
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()
                acc = acc + o[0]; votes.append(o[0].argmax(0))
            lo[a] = (acc / len(v2.FOLDS)).cpu().numpy(); maj[a] = torch.mode(torch.stack(votes), 0).values.cpu().numpy()
        inner = np.zeros(gp.shape, bool); inner[tuple(slice(e7.EDGE, s - e7.EDGE) for s in gp.shape)] = True
        ok0 = (gp == 0) & bp & inner
        for a in ARMS:
            oth = lo[a][1:]; mx = oth.max(0); ok0 &= (maj[a] == 0) & (lo[a][0] - (np.log(np.exp(oth - mx).sum(0)) + mx) > e7.CONF)
        dout = {c: ndi.distance_transform_edt(gp != c) for c in set(g.cls.astype(int))}; geo = {c: e7.region_geometry(gp, c) for c in dout}
        seeds, pair, dist, cls_t = [], [], [], []
        for r in g.itertuples():
            c = int(r.cls)
            for tol in (0.5, 1.5):
                idx = np.argwhere(ok0 & (np.abs(dout[c] - r.depth) <= tol))
                if len(idx):
                    break
            if not len(idx):
                continue
            v = tuple(idx[rng.integers(len(idx))]); seeds.append(v); pair.append(int(r.seed)); dist.append(float(dout[c][v])); cls_t.append(c)
        if not seeds:
            log(f"[{key}] {case}: no matched control voxel"); continue
        Gs = {}
        for a, pr in prs.items():
            net = pr.network; acc = None
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                gg = e7.seed_grads(net, ip, seeds, [0] * len(seeds), dev); acc = gg if acc is None else [x + y for x, y in zip(acc, gg)]
            Gs[a] = [(x / len(v2.FOLDS)).abs().cpu().numpy() for x in acc]
        rows = []
        for i, (v, pi, d, c) in enumerate(zip(seeds, pair, dist, cls_t)):
            rt = g[g.seed == pi].iloc[0]
            row = dict(cell=key, task=cell["task"], case=case, seed=i, pair=pi, z=v[0], y=v[1], x=v[2], cls=c, dist_out=d, depth_target=float(rt.depth),
                       dice_flat=float(rt.dice_flat), dice_real=float(rt.dice_real))
            for a in ARMS:
                for mk, mv in e7.seed_metrics(Gs[a][i], v, geo[c][2], bp, r_loc).items():
                    row[f"{a}_{mk}"] = mv
            rows.append(row)
        sv = {"img": [], "gt": [], "G_flat": [], "G_real": []}
        for i, (v, c) in enumerate(zip(seeds, cls_t)):
            z = v[0]; sv["img"].append(ip[z]); sv["gt"].append(gp[z] == c)
            for a in ARMS:
                sv[f"G_{a}"].append(Gs[a][i][z])
        np.savez_compressed(od / f"{case}_ctrl.npz", seeds=np.array(seeds), pair=np.array(pair), **{k: np.array(vv).astype(np.float16 if k != "gt" else bool) for k, vv in sv.items()})
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)
        df = pd.DataFrame(rows); log(f"[{key}] {case}: {len(seeds)}/{len(g)} matched controls | local share ctrl flat {df.flat_local_share.median():.4f} real {df.real_local_share.median():.4f}; {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELLS)); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--list", action="store_true"); ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, c in CELLS.items():
            print(k, c["task"], len(c["cases"]), c["models"]["flat"](), c["models"]["real"](), sep="\t")
        return
    if a.report:
        report(); return
    for c in a.cells.split(","):
        (run_ctrl if a.control else run_cell)(c, torch.device("cuda"), a.case_slice)


if __name__ == "__main__":
    main()
