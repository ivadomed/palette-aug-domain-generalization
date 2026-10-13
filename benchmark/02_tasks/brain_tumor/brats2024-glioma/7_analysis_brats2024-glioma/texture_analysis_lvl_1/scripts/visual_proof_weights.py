#!/usr/bin/env python
"""
Model-weight / representation analysis (2026-10-12): how does the BraTS glioma model trained with FLAT-fill synthesis (ladder rung 4.5)
differ internally from the REAL-fill one (rung 5)? Ladder context: rung 0 (baseline, no synthesis) and rung 4 (Voronoi noise fill, lblvor);
random-init networks of the same architecture as a floor. Pre-registration + results: outputs/plots/weights_analysis/weights_analysis.md.
  A1 (CPU)  --weights        first-layer 3x3x3 filter spectra (HF fraction, mean-filter alignment, orientation FA) + effective / stable rank
                             of every conv weight matrix, per fold.
  A2 (GPU)  --acts --contrast <t1n|t1c|t2w|t2f> [--case-slice i:j]
                             per case: one edema-centred NATURAL test patch (no intervention), forward through all 27 networks (8 models x
                             3 folds + 3 random); hooks on the 6 encoder + 5 decoder stages store debiased linear HSIC matrices between all
                             network pairs on 4096 shared brain positions (minibatch CKA). (A3, texture-removed inputs, was dropped
                             2026-10-12 on request: no intervention-based analyses.)
                             -> $VPW_CACHE/<contrast>/<case>.npz
  report (CPU) --report      aggregates, tables (weights_tables.md) and figures (fig1..fig3, PDF + PNG) into outputs/plots/weights_analysis/.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"
GL = T2 / "brain_tumor/brats2024-glioma"; PRED = GL / "8_results_brats2024-glioma/01_predictions/brats2024_glioma_model"
RAW = GL / "2_nnUNet_brats2024-glioma/raw"
OUT = Path(os.environ.get("VPW_OUT") or (THIS.parent / "outputs/plots/weights_analysis")); OUT.mkdir(parents=True, exist_ok=True)
REGION = os.environ.get("VPW_REGION", "brain"); SUF = "" if REGION == "brain" else f"_{REGION}"   # Amendment 2: "edema" = lesion-restricted CKA
CACHE = Path(os.environ.get("VPW_CACHE") or (OUT / "_cache")); CACHE = CACHE.parent / (CACHE.name + SUF) if SUF else CACHE
RUNS = {
    ("t1n", "r0"): "t1n/nnUNet/brats2024-glioma_t1n_baseline_20260622_044535",
    ("t1n", "r4"): "t1n/auglab/brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_lblvor_20261007_170548",
    ("t1n", "r45"): "t1n/auglab/brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190618",
    ("t1n", "r5"): "t1n/auglab/brats2024-glioma_t1n_v26_6_2_train050_val000_20261005_222227",
    ("t2w", "r0"): "t2w/nnUNet/brats2024-glioma_t2w_baseline_20260620_125115",
    ("t2w", "r4"): "t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_lblvor_20261007_170621",
    ("t2w", "r45"): "t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190652",
    ("t2w", "r5"): "t2w/auglab/brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300",
}
RUNGS = ("r0", "r4", "r45", "r5"); TCS = ("t1n", "t2w"); FOLDS = (0, 1, 2)
SLAB = {"r0": "r0 baseline", "r4": "r4 noise fill", "r45": "r4.5 flat fill", "r5": "r5 real fill", "rand": "random init"}
RLAB = {"r0": "baseline (rung 0)", "r4": "noise fill (rung 4)", "r45": "flat fill (rung 4.5)", "r5": "real fill (rung 5)", "rand": "random init"}
COL = {"r45": "#1f77b4", "r5": "#d62728", "r4": "#2ca02c", "r0": "#7f7f7f", "rand": "#bdbdbd"}
NETS = [(tc, r, k) for tc in TCS for r in RUNGS for k in FOLDS] + [("rand", "rand", s) for s in FOLDS]   # 27 networks, fixed order
N_POS = 4096; N_CASES = 24
STAGES = ["E0", "E1", "E2", "E3", "E4", "E5", "D4", "D3", "D2", "D1", "D0"]   # D<j> = decoder stage at encoder-stage-j resolution
IMG = {c: RAW / ("Dataset051_BraTS2024GliomaT1n" if c == "t1c" else "Dataset052_BraTS2024GliomaT2w") / f"imagesTs_{c}" for c in ("t1n", "t1c", "t2w", "t2f")}
GT = RAW / "Dataset052_BraTS2024GliomaT2w/labelsTr"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def model_dir(tc, r):
    hits = sorted((PRED / RUNS[(tc, r)]).glob("Dataset*/*__3d_fullres"))
    assert len(hits) == 1, (tc, r, hits); return hits[0]


def load_sd(path):
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False); sd = ck["network_weights"]
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    return sd, int(ck.get("current_epoch", -1))


def build_net(md, seed=None):
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    plans = json.load(open(md / "plans.json")); dj = json.load(open(md / "dataset.json")); pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres"); lm = pm.get_label_manager(dj)
    if seed is not None:
        torch.manual_seed(seed)
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
                                 determine_num_input_channels(pm, cm, dj), lm.num_segmentation_heads, allow_init=True, deep_supervision=False)
    return net, pm, cm, dj   # allow_init=True: nnU-Net's own He init (seeded above for the random-init floor)


# ---------------------------------------------------------------------------------------------------------------- A1: weights (CPU)
def kernel_metrics(w):
    """w: (n_filters, 3, 3, 3) -> per-filter HF fraction, mean-filter alignment, orientation FA, radial power profile."""
    n = w.shape[0]; G = 16
    f1 = np.fft.fftfreq(G); FZ, FY, FX = np.meshgrid(f1, f1, f1, indexing="ij"); R = np.sqrt(FZ ** 2 + FY ** 2 + FX ** 2)
    edges = np.linspace(0, R.max() + 1e-9, 13); rb = np.digitize(R.ravel(), edges) - 1
    F = np.stack([FZ.ravel(), FY.ravel(), FX.ravel()], 1)
    out = []
    for i in range(n):
        k = w[i].astype(np.float64); k = k / np.sqrt((k ** 2).sum() + 1e-30)
        P = np.abs(np.fft.fftn(k, s=(G, G, G))) ** 2; P = P.ravel(); Pn = P / P.sum()
        hf = float(Pn[R.ravel() > 0.25].sum()); nyq = float(Pn[R.ravel() > 0.6].sum()); dc = float(k.sum() ** 2 / (27 * (k ** 2).sum()))
        S = (F * Pn[:, None]).T @ F; lam = np.sort(np.linalg.eigvalsh(S))[::-1]
        fa = float(np.sqrt(1.5) * np.linalg.norm(lam - lam.mean()) / (np.linalg.norm(lam) + 1e-30))
        prof = np.bincount(rb, weights=Pn, minlength=12)[:12] / np.maximum(np.bincount(rb, minlength=12)[:12], 1)
        out.append(dict(hf=hf, nyq=nyq, dc=dc, fa=fa, prof=prof))
    return out, 0.5 * (edges[1:] + edges[:-1])


def mat_ranks(W):
    s = np.linalg.svd(W.reshape(W.shape[0], -1).astype(np.float64), compute_uv=False); p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum())), float((s ** 2).sum() / s[0] ** 2), int(min(W.shape[0], np.prod(W.shape[1:])))


def run_weights():
    import pandas as pd
    rows_f, rows_l, profs, kerns = [], [], {}, {}
    jobs = [(tc, r, k) for tc in TCS for r in RUNGS for k in FOLDS] + [("rand", "rand", s) for s in FOLDS]
    md0 = model_dir("t1n", "r5")
    for tc, r, k in jobs:
        if r == "rand":
            net, *_ = build_net(md0, seed=k); sd = {n: v.detach() for n, v in net.state_dict().items()}; ep = 0
        else:
            sd, ep = load_sd(model_dir(tc, r) / f"fold_{k}" / "checkpoint_best.pth")
        convs = [n for n, v in sd.items() if n.endswith(".weight") and v.ndim == 5 and not n.startswith("decoder.encoder.") and ".all_modules." not in n]   # decoder holds an encoder alias
        first = [n for n in convs if tuple(sd[n].shape) == (32, 1, 3, 3, 3)]; assert len(first) == 1, first
        w = sd[first[0]].float().numpy()[:, 0]
        gname = first[0].replace("convs.0.conv.weight", "convs.0.norm.weight"); gam = sd[gname].float().numpy() if gname in sd else np.ones(32)
        km, rc = kernel_metrics(w)
        for i, m in enumerate(km):
            rows_f.append(dict(tc=tc, rung=r, fold=k, epoch=ep, filt=i, gamma=float(abs(gam[i])), hf=m["hf"], nyq=m["nyq"], dc=m["dc"], fa=m["fa"]))
        profs[f"{tc}_{r}_{k}"] = np.stack([m["prof"] for m in km]); kerns[f"{tc}_{r}_{k}"] = w
        for li, n in enumerate(convs):
            er, sr, full = mat_ranks(sd[n].float().numpy())
            rows_l.append(dict(tc=tc, rung=r, fold=k, epoch=ep, layer=li, name=n, out=sd[n].shape[0], fan_in=int(np.prod(sd[n].shape[1:])),
                               eff_rank=er, stable_rank=sr, eff_rank_frac=er / full))
        log(f"weights {tc} {r} fold{k}: epoch {ep}, {len(convs)} convs, first = {first[0]}")
    pd.DataFrame(rows_f).to_csv(OUT / "a1_first_layer_filters.csv", index=False)
    pd.DataFrame(rows_l).to_csv(OUT / "a1_layer_ranks.csv", index=False)
    np.savez_compressed(OUT / "a1_radial_profiles.npz", radius=rc, **profs); np.savez_compressed(OUT / "a1_first_layer_kernels.npz", **kerns)


# ---------------------------------------------------------------------------------------------------------------- A2+A3: activations (GPU)
def hsic_matrix(feats):
    """feats: list of (n, C) float64 CUDA tensors (centred). Debiased HSIC_1 (Song et al. 2012) between all pairs, linear kernel."""
    import torch
    n = feats[0].shape[0]; M = len(feats)
    K = torch.empty((M, n, n), dtype=feats[0].dtype, device=feats[0].device)
    for i, X in enumerate(feats):
        torch.mm(X, X.T, out=K[i])
    K.diagonal(dim1=1, dim2=2).zero_()
    r = K.sum(2); s = r.sum(1); Kf = K.reshape(M, -1); T = Kf @ Kf.T; RR = r @ r.T; del K, Kf
    H = (T + s[:, None] * s[None, :] / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * RR) / (n * (n - 3))
    return H.cpu().numpy()


def run_acts(contrast, sl=None):
    import torch, torch.nn.functional as Fnn
    from types import SimpleNamespace
    os.environ.setdefault("VP8_OUT", str(THIS.parent / "outputs/plots/visual_proof_erf_tasks"))
    import visual_proof_v2 as v2
    import visual_proof_erf as e7
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda"); od = CACHE / contrast; od.mkdir(parents=True, exist_ok=True)
    md0 = model_dir("t2w", "r5"); net, pm, cm, dj = build_net(md0); net.to(dev).eval(); ref_arch = cm.network_arch_init_kwargs
    pr0 = SimpleNamespace(plans_manager=pm, configuration_manager=cm, dataset_json=dj); ps = cm.patch_size
    sds, epochs = [], []
    for tc, r, k in NETS:
        if r == "rand":
            rn, *_ = build_net(md0, seed=k); sds.append({n: v.detach().clone() for n, v in rn.state_dict().items()}); epochs.append(0)
        else:
            md = model_dir(tc, r); _, _, cmx, _ = build_net(md)
            assert cmx.network_arch_init_kwargs == ref_arch and list(cmx.patch_size) == list(ps), (tc, r)
            sd, ep = load_sd(md / f"fold_{k}" / "checkpoint_best.pth"); net.load_state_dict(sd, strict=True); sds.append(sd); epochs.append(ep)
    log(f"loaded {len(sds)} networks; patch {ps}")
    mods = list(net.encoder.stages) + list(net.decoder.stages); assert len(mods) == len(STAGES)
    cases = sorted(p.name[:-12] for p in IMG[contrast].glob("*_0000.nii.gz") if (GT / f"{p.name[:-12]}.nii.gz").exists())[:N_CASES]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    state = {}

    def hook(si):
        def f(_m, _i, out):
            o = out.detach(); sh = tuple(o.shape[2:])
            if sh not in state["mask"]:
                fr = Fnn.interpolate(state["region"][None, None], size=sh, mode="area")[0, 0]; m = (fr > 0) if REGION == "edema" else (fr >= 0.5)
                idx = torch.nonzero(m.flatten()).flatten(); g = np.random.default_rng(state["seed"] * 100 + si)
                pick = idx[torch.from_numpy(np.sort(g.choice(len(idx), min(N_POS, len(idx)), replace=False))).to(dev)]
                state["mask"][sh] = (m.flatten(), pick)
            m, pick = state["mask"][sh]; x = o.flatten(2)                       # (1, C, V)
            fz = x[0][:, pick].double().T                                       # (n, C)
            state["feats"][si].append(fz - fz.mean(0, keepdim=True))
        return f
    hs = [m.register_forward_hook(hook(i)) for i, m in enumerate(mods)]
    for ci, case in enumerate(cases):
        f_out = od / f"{case}.npz"
        if f_out.exists():
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, IMG[contrast] / f"{case}_0000.nii.gz", GT / f"{case}.nii.gz"); img = data[0]
        tg = gt == 2
        if tg.sum() < 50:
            log(f"{case}: edema < 50 vox, skipped"); continue
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, img[box].shape)]
        X = np.pad(img[box].astype(np.float32), padw)[None]; bp = np.pad(body[box], padw); gp = np.pad(gt[box], padw)
        state.update(body=torch.from_numpy(bp.astype(np.float32)).to(dev), region=torch.from_numpy(((gp == 2) if REGION == "edema" else bp).astype(np.float32)).to(dev), mask={}, seed=sum(map(ord, case)),
                     feats=[[] for _ in STAGES])
        xin = torch.from_numpy(X[:, None]).to(dev); dice = np.zeros((len(NETS), 1)); ed = gp == 2
        for ni, sd in enumerate(sds):
            net.load_state_dict(sd, strict=True); net.eval()
            with torch.no_grad():
                o = net(xin)
            am = o.argmax(1).cpu().numpy() == 2
            for c in range(1):
                s_ = am[c].sum() + ed.sum(); dice[ni, c] = 2 * (am[c] & ed).sum() / s_ if s_ else np.nan
        H = np.stack([hsic_matrix(state["feats"][si]) if state["feats"][si][0].shape[0] >= 10 else np.full((len(NETS), len(NETS)), np.nan)
                      for si in range(len(STAGES))])     # (11, 27, 27); NaN = < 10 positions at this stage (lesion mode)
        npos = np.array([state["feats"][si][0].shape[0] for si in range(len(STAGES))])
        np.savez_compressed(f_out, H=H.astype(np.float64), npos=npos, dice=dice, epochs=np.array(epochs), n_edema=int(ed.sum()))
        state["feats"] = None; torch.cuda.empty_cache()
        log(f"[{contrast}] {case} ({ci + 1}/{len(cases)}): {time.time() - t0:.1f}s; edema dice orig r45/r5 t1n f0: {dice[6, 0]:.3f}/{dice[9, 0]:.3f}")
    for h in hs:
        h.remove()


# ---------------------------------------------------------------------------------------------------------------- report (CPU)
def gallery(F, plt, save, top=6):
    """Publication panel: (a) ECDF of per-filter HF fraction (one thin line per fold, pooled thick line); (b) the `top` highest-HF first-layer
    filters of each model, pooled over its 3 folds (selection rule fixed before plotting; illustrative - the test is the fold-level means
    in fig1). Each 3x3x3 kernel drawn as its three slices along array axis 0, colour scale symmetric per filter (red +, blue -)."""
    K = np.load(OUT / "a1_first_layer_kernels.npz"); rungs = ("r0", "r4", "r45", "r5")
    fig = plt.figure(figsize=(7.1, 3.9)); gs = fig.add_gridspec(2, 5, width_ratios=[1.15, 1, 1, 1, 1], wspace=0.25, hspace=0.55)
    for ri, tc in enumerate(TCS):
        ax = fig.add_subplot(gs[ri, 0])
        for r in rungs + ("rand",):
            t_ = "rand" if r == "rand" else tc; d = F[(F.tc == t_) & (F.rung == r)]
            for k in FOLDS:
                v = np.sort(d[d.fold == k].hf.values); ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[r], lw=0.4, alpha=0.6)
            v = np.sort(d.hf.values); ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[r], lw=1.3, label=SLAB[r])
        ax.set_xlim(0, 1); ax.set_xlabel("HF energy fraction per filter (|f| > 0.25)")
        ax.set_ylabel(f"{tc.upper()}-trained\ncumulative fraction of filters")
        if ri == 0:
            ax.legend(frameon=False, fontsize=5.0, loc="upper left", handlelength=1.0)
        for ci, r in enumerate(rungs):
            ax = fig.add_subplot(gs[ri, ci + 1]); d = F[(F.tc == tc) & (F.rung == r)].sort_values("hf", ascending=False).head(top)
            canvas = np.full((top * 4 - 1, 11), np.nan)
            for j, (_, row) in enumerate(d.iterrows()):
                w = K[f"{tc}_{r}_{int(row.fold)}"][int(row.filt)]; w = w / (np.abs(w).max() + 1e-12)
                canvas[j * 4:j * 4 + 3] = np.concatenate([w[0], np.full((3, 1), np.nan), w[1], np.full((3, 1), np.nan), w[2]], 1)
                ax.text(11.3, j * 4 + 1, f"{row.hf:.2f} (f{int(row.fold)})", va="center", fontsize=5.2)
            ax.imshow(np.ma.masked_invalid(canvas), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            ax.set_title(RLAB[r].replace(" (", "\n("), color=COL[r], fontsize=6.5)
    fig.text(0.6, 0.005, f"top-{top} first-layer filters by HF fraction (pooled over 3 folds; 3 slices each; label = HF fraction (fold))", ha="center", fontsize=5.8)
    save(fig, "fig1b_first_layer_filter_gallery")


def report():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0})
    md = []
    sep = lambda a, b: (min(a) > max(b)) or (min(b) > max(a))   # noqa: E731  (3 vs 3 complete separation = the pre-registered claim rule)

    def save(fig, name):
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    def dots(ax, xs, vals, col, label=None, mk="o"):
        ax.scatter(np.full(len(vals), xs) + np.linspace(-0.12, 0.12, len(vals)), vals, s=9, color=col, alpha=0.75, lw=0, marker=mk, zorder=3)
        ax.plot([xs - 0.22, xs + 0.22], [np.mean(vals)] * 2, color=col, lw=1.6, zorder=4, label=label)

    if REGION != "brain":
        return report_a2(pd, plt, md, sep, save)
    # ---------------- A1
    F = pd.read_csv(OUT / "a1_first_layer_filters.csv"); L = pd.read_csv(OUT / "a1_layer_ranks.csv"); PR = np.load(OUT / "a1_radial_profiles.npz")
    md += ["## A1 First-layer filter spectra and weight-matrix rank", "",
           "Per fold = mean over the 32 first-layer filters. Claim = complete 3-vs-3 separation of flat (r45) and real (r5) folds.", "",
           "| train | rung | epochs (folds 0/1/2) | HF fraction | mean-filter alignment | orientation FA | HF (|gamma|-weighted) | Nyquist band |f|>0.6 (post hoc) |", "|---|---|---|---|---|---|---|---|"]
    g = F.groupby(["tc", "rung", "fold"])
    per = g.apply(lambda d: pd.Series(dict(hf=d.hf.mean(), nyq=d.nyq.mean(), dc=d.dc.mean(), fa=d.fa.mean(), hfw=np.average(d.hf, weights=d.gamma + 1e-12),
                                            epoch=d.epoch.iloc[0]))).reset_index()
    for tc in TCS + ("rand",):
        for r in (RUNGS if tc != "rand" else ("rand",)):
            d = per[(per.tc == tc) & (per.rung == r)]
            if not len(d):
                continue
            fm = lambda c: f"{d[c].mean():.3f} [" + ", ".join(f"{v:.3f}" for v in d[c]) + "]"   # noqa: E731
            md.append(f"| {tc} | {RLAB[r]} | {'/'.join(str(int(e)) for e in d.epoch)} | {fm('hf')} | {fm('dc')} | {fm('fa')} | {fm('hfw')} | {fm('nyq')} |")
    md += [""]
    for tc in TCS:
        a, b = per[(per.tc == tc) & (per.rung == "r45")], per[(per.tc == tc) & (per.rung == "r5")]
        md.append(f"- {tc}-trained flat vs real: " + "; ".join(f"{m}: {a[m].mean():.3f} vs {b[m].mean():.3f} ({'SEPARATED' if sep(a[m], b[m]) else 'overlap'})"
                                                         for m in ("hf", "dc", "fa", "hfw", "nyq")))
    md += ["", "Layer-wise effective rank (fraction of full rank), mean over folds [min-max]; layers = all conv weights in state-dict order.", ""]
    Lm = L.groupby(["tc", "rung", "layer"]).eff_rank_frac.agg(["mean", "min", "max"]).reset_index()
    nsep = {}
    for tc in TCS:
        a = L[(L.tc == tc) & (L.rung == "r45")]; b = L[(L.tc == tc) & (L.rung == "r5")]
        s_ = [(li, a[a.layer == li].eff_rank_frac.mean() - b[b.layer == li].eff_rank_frac.mean()) for li in sorted(L.layer.unique())
              if sep(a[a.layer == li].eff_rank_frac.values, b[b.layer == li].eff_rank_frac.values)]
        nsep[tc] = s_
        md.append(f"- {tc}: {len(s_)}/{L.layer.nunique()} conv layers with flat/real effective-rank folds completely separated"
                  + (": " + ", ".join(f"L{li} ({dv:+.3f})" for li, dv in s_) if s_ else "") )
    sr = L.groupby(["tc", "rung", "fold"]).stable_rank.mean().reset_index()
    for tc in TCS:
        a, b = sr[(sr.tc == tc) & (sr.rung == "r45")].stable_rank, sr[(sr.tc == tc) & (sr.rung == "r5")].stable_rank
        md.append(f"- {tc}: mean stable rank over layers flat {a.mean():.2f} [{a.min():.2f}-{a.max():.2f}] vs real {b.mean():.2f} [{b.min():.2f}-{b.max():.2f}] ({'SEPARATED' if sep(a, b) else 'overlap'})")
    md += [""]
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    rc = PR["radius"]
    for ri, tc in enumerate(TCS):
        ax = axs[ri, 0]
        for r in ("r0", "r4", "r45", "r5", "rand"):
            P = np.stack([PR[f"{tc if r != 'rand' else 'rand'}_{r}_{k}"].mean(0) for k in FOLDS])
            for p_ in P:
                ax.plot(rc, p_, color=COL[r], lw=0.5, alpha=0.5)
            ax.plot(rc, P.mean(0), color=COL[r], lw=1.4, label=SLAB[r])
        ax.axvline(0.25, color="k", lw=0.5, ls=":"); ax.set_yscale("log"); ax.set_xlabel("spatial frequency |f| (cycles/voxel)")
        ax.set_ylabel(f"{tc.upper()}-trained\nnormalised power"); ax.set_xlim(0, 0.87)
        if ri == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=5.2, handlelength=1.2)
        for j, (m, lab) in enumerate((("hf", "HF fraction (|f|>0.25)"), ("dc", "mean-filter alignment"), ("fa", "orientation FA"))):
            ax = axs[ri, j + 1]
            for xi, r in enumerate(("r0", "r4", "r45", "r5", "rand")):
                d = per[(per.tc == (tc if r != "rand" else "rand")) & (per.rung == r)][m].values; dots(ax, xi, d, COL[r])
            ax.set_xticks(range(5)); ax.set_xticklabels(["r0", "r4", "r4.5\nflat", "r5\nreal", "rand"]); ax.set_title(lab if ri == 0 else "")
    fig.tight_layout(); save(fig, "fig1_first_layer_spectra")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.0), sharey=True)
    for ax, tc in zip(axs, TCS):
        for r in ("r0", "r4", "r45", "r5", "rand"):
            d = Lm[(Lm.tc == (tc if r != "rand" else "rand")) & (Lm.rung == r)].sort_values("layer")
            ax.fill_between(d.layer, d["min"], d["max"], color=COL[r], alpha=0.2, lw=0); ax.plot(d.layer, d["mean"], color=COL[r], lw=1.0, label=RLAB[r])
        nm = L[(L.tc == tc) & (L.rung == "r5") & (L.fold == 0)].sort_values("layer").name.tolist()
        ticks = [i for i, n in enumerate(nm) if (".convs.0." in n and ("encoder.stages" in n or "decoder.stages" in n)) or n.endswith("transpconvs.0.weight")
                 or n.endswith("seg_layers.0.weight")]
        lab_ = lambda n: ("E" + n.split("stages.")[1].split(".")[0]) if n.startswith("encoder") else ("up" if "transpconvs" in n else ("seg" if "seg_layers" in n else  # noqa: E731
                         "D" + str(4 - int(n.split("stages.")[1].split(".")[0]))))
        ax.set_xticks(ticks); ax.set_xticklabels([lab_(nm[i]) for i in ticks], fontsize=5.5)
        ax.set_title(f"{tc.upper()}-trained"); ax.set_xlabel("conv layer (up = transposed convs, seg = output heads)", fontsize=6)
    axs[0].set_ylabel("effective rank / full rank"); axs[0].legend(frameon=False, fontsize=5.5, ncol=1)
    fig.tight_layout(); save(fig, "fig2_layer_effective_rank")
    gallery(F, plt, save)

    report_a2(pd, plt, md, sep, save)


def report_a2(pd, plt, md, sep, save):
    CONTR = ("t1n", "t1c", "t2w", "t2f")
    idx = {n: i for i, n in enumerate(NETS)}; M = len(NETS)
    vi = lambda tc, r, k: idx[(tc, r, k)]   # noqa: E731
    Hs, dices = {}, {}
    for c in CONTR:
        fs = sorted((CACHE / c).glob("*.npz"))
        if not fs:
            continue
        Z = [np.load(f) for f in fs]; Hs[c] = np.stack([z["H"] for z in Z]); dices[c] = np.stack([z["dice"] for z in Z])
        log(f"{c}: {len(fs)} cases")
    md += ["## Sanity: edema Dice in the analysed patches (orig image; mean over cases, per fold)", "",
           "| test contrast | n | " + " | ".join(f"{tc} {r}" for tc in TCS for r in RUNGS) + " |", "|---|---|" + "---|" * 8]
    for c in Hs:
        D = np.nanmean(dices[c][:, :, 0], 0)
        md.append(f"| {c} | {len(dices[c])} | " + " | ".join("/".join(f"{D[idx[(tc, r, k)]]:.2f}" for k in FOLDS) for tc in TCS for r in RUNGS) + " |")
    md += [""]

    def cka(H, i, j):
        S = np.nansum(H, 0); return S[i, j] / np.sqrt(S[i, i] * S[j, j])

    rows = []
    for c in Hs:
        for si, st in enumerate(STAGES):
            H = Hs[c][:, si]
            for tc in TCS:
                for (ra, rb) in (("r45", "r5"), ("r4", "r5"), ("r0", "r5"), ("r45", "r4")):
                    cross = [cka(H, vi(tc, ra, a), vi(tc, rb, b)) for a in FOLDS for b in FOLDS]
                    null = [cka(H, vi(tc, rr, a), vi(tc, rr, b)) for rr in (ra, rb) for a in FOLDS for b in FOLDS if a < b]
                    rows.append(dict(contrast=c, stage=st, si=si, tc=tc, pair=f"{ra}-{rb}", domain="in" if c == tc else "held-out",
                                     cross_mean=np.mean(cross), cross_min=np.min(cross), cross_max=np.max(cross),
                                     null_mean=np.mean(null), null_min=np.min(null), null_max=np.max(null), gap=np.mean(null) - np.mean(cross),
                                     diverged=bool(np.max(cross) < np.min(null))))
                rows.append(dict(contrast=c, stage=st, si=si, tc=tc, pair="r45-rand", domain="in" if c == tc else "held-out",
                                 cross_mean=np.mean([cka(H, vi(tc, "r45", a), vi("rand", "rand", b)) for a in FOLDS for b in FOLDS])))
    C = pd.DataFrame(rows); C.to_csv(OUT / f"a2_cka{SUF}.csv", index=False)
    md += ["## A2 Linear CKA (debiased, minibatch over cases): flat (r45) vs real (r5)", "",
           "gap = mean(within-model fold CKA) - mean(flat-vs-real CKA); `*` = diverged (max cross < min within). Columns = stages.", ""]
    for tc in TCS:
        md += [f"**{tc.upper()}-trained** (in-domain = {tc})", "", "| test contrast | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
        for c in Hs:
            d = C[(C.contrast == c) & (C.tc == tc) & (C.pair == "r45-r5")].sort_values("si")
            md.append(f"| {c}{' (in)' if c == tc else ''} | " + " | ".join(f"{g_:.3f}{'*' if dv else ''}" for g_, dv in zip(d.gap, d.diverged)) + " |")
        md += ["", "cross CKA flat-vs-real (mean [min-max]) / within-model null (mean [min-max]):", "",
               "| test contrast | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
        for c in Hs:
            d = C[(C.contrast == c) & (C.tc == tc) & (C.pair == "r45-r5")].sort_values("si")
            md.append(f"| {c} | " + " | ".join(f"{a:.2f} [{b:.2f}-{e:.2f}] / {f:.2f} [{g_:.2f}-{h:.2f}]" for a, b, e, f, g_, h in
                                               zip(d.cross_mean, d.cross_min, d.cross_max, d.null_mean, d.null_min, d.null_max)) + " |")
        md += [""]
    md += ["Ladder context, gap per stage (pair: rows), held-out mean over the held-out contrasts vs in-domain:", "",
           "| train | pair | domain | " + " | ".join(STAGES) + " |", "|---|---|---|" + "---|" * len(STAGES)]
    for tc in TCS:
        for p in ("r45-r5", "r4-r5", "r0-r5", "r45-r4"):
            for dm in ("in", "held-out"):
                d = C[(C.tc == tc) & (C.pair == p) & (C.domain == dm)].groupby("si").gap.mean()
                md.append(f"| {tc} | {p} | {dm} | " + " | ".join(f"{v:.3f}" for v in d.values) + " |")
    md += [""]
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.4), sharey="row")
    for ri, tc in enumerate(TCS):
        for ci, c in enumerate(CONTR):
            ax = axs[ri, ci]
            if c not in Hs:
                ax.axis("off"); continue
            d = C[(C.contrast == c) & (C.tc == tc) & (C.pair == "r45-r5")].sort_values("si"); x = np.arange(len(STAGES))
            ax.fill_between(x, d.null_min, d.null_max, color="#999999", alpha=0.35, lw=0, label="within-model folds (null)")
            ax.fill_between(x, d.cross_min, d.cross_max, color="#6a3d9a", alpha=0.25, lw=0)
            ax.plot(x, d.cross_mean, color="#6a3d9a", lw=1.2, label="flat vs real (9 fold pairs)")
            ax.plot(x, d.null_mean, color="#555555", lw=0.8, ls="--")
            for xx, dv, cm_ in zip(x, d.diverged, d.cross_mean):
                if dv:
                    ax.plot(xx, cm_, "v", color="#6a3d9a", ms=3)
            dd = C[(C.contrast == c) & (C.tc == tc) & (C.pair == "r45-rand")].sort_values("si")
            ax.plot(x, dd.cross_mean, color=COL["rand"], lw=0.8, ls=":", label="flat vs random init")
            ax.set_xticks(x); ax.set_xticklabels(STAGES, rotation=90, fontsize=5.5); ax.axvline(5.5, color="k", lw=0.3)
            ax.set_title(f"{tc.upper()}-trained \u2192 {c.upper()}" + (" (in-domain)" if c == tc else " (held-out)"), fontsize=6.5)
            ax.set_ylim(0, 1.02)
        axs[ri, 0].set_ylabel("linear CKA (debiased)")
    axs[0, 0].legend(frameon=False, fontsize=5.3, loc="lower left")
    fig.tight_layout(); save(fig, f"fig3_cka_flat_vs_real{SUF}")

    (OUT / f"weights_tables{SUF}.md").write_text(f"# Weights analysis - generated tables (visual_proof_weights.py --report, CKA positions: {REGION})\n\n" + "\n".join(md) + "\n")
    log("report written")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", action="store_true"); ap.add_argument("--acts", action="store_true")
    ap.add_argument("--contrast"); ap.add_argument("--case-slice"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.weights:
        run_weights()
    if a.acts:
        run_acts(a.contrast, a.case_slice)
    if a.report:
        report()


if __name__ == "__main__":
    main()
