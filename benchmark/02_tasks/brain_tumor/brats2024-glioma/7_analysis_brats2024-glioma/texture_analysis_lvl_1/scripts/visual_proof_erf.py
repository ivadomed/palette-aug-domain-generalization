#!/usr/bin/env python
"""
Visual proof v7 (2026-10-12): WHERE DOES EACH MODEL LOOK TO DECIDE THAT A VOXEL DEEP INSIDE THE TARGET IS TARGET?  Per-voxel input gradients
(effective receptive field, Luo et al. NeurIPS 2016; saliency, Simonyan et al. 2013) on the UNPERTURBED real test image -- no blur, no fill,
no masking. Replaces the test-time blur strip (v5/v6) as the mechanism figure: a removal perturbation measures robustness to a shift that is
in-distribution for the flat-fill model (trained on piecewise-constant interiors) and out-of-distribution for the real-fill model, so it cannot
separate "uses texture" from "breaks without texture" (Hooker et al., ROAR, NeurIPS 2019). The causal evidence is the retrained ladder (4.5 -> 5).

Per case, one plan-sized patch around the target (as v6). Seeds = target voxels that BOTH models segment correctly and confidently (3-fold
majority = GT class, fold-mean log-odds > CONF for both arms), at depth >= D_MIN from the GT region boundary, >= EDGE vox from the patch faces,
sampled depth-stratified (N_SEEDS per case, rng 0). For each seed v of class c:
  f_v(x) = logit_c(v) - logsumexp_{j != c} logit_j(v)      (log-odds, pre-softmax: no saturation)
  G_v    = | mean over folds of d f_v / d x |              (ensemble gradient, 3 folds)
Metrics per seed (GT region = the seed's class; used only to read the metrics, never to change the input):
  local_share = sum G in the ball |u - v| <= R_LOC  /  sum G in the body                     ("decides from what is around the voxel")
  rim_excess  = sum G on the rim band (|signed dist| <= 2)  /  sum over the same rim voxels of E[G | distance from v] taken over non-rim body
                voxels   (> 1: the model looks at the contour more than the ERF fall-off alone predicts; = 1: no contour preference)
  rim_share   = sum G on the rim band / sum G in the body
PRE-REGISTERED PREDICTION (written before the first run): on appearance-defined targets (BraTS edema, Open-MS lesion) the flat-fill model
(rung 4.5) has HIGHER rim_excess and LOWER local_share than the real-fill model (rung 5): under contrast randomization a uniform interior carries
no class information, so the flat model must read the contour; the real-fill model can read the local texture. On the interface-defined control
(pelvis CT) both models read the contour (rim_excess > 1) and the arm difference is small. Controls: a randomly re-initialised network
(same architecture, same seeds) and the image's own edge map |grad G_1 x| get the same metrics.
Outputs: outputs/plots/visual_proof_erf/<cell>/{seeds.csv, <case>_erf.npz}; --report: violins, profiles, hero sheets, erf_stats.md.
GPU: .venv/bin/python visual_proof_erf.py --cells bratsfull_t1n_t2f --cases BraTSGLI02169102     CPU: --report
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
os.environ.setdefault("VP_OUT", str(THIS.parent / "outputs/plots/visual_proof_v2"))
import visual_proof_v2 as v2  # noqa: E402

OUT = Path(os.environ.get("VP7_OUT") or (THIS.parent / "outputs/plots/visual_proof_erf")); OUT.mkdir(parents=True, exist_ok=True)
PELVIS_CASES = sorted(p.name.replace("_0000.nii.gz", "") for p in (v2.RPC / "imagesTs_ct").glob("*_0000.nii.gz"))
CELLS = {k: dict(v2.CELLS[k], target=[2], r_loc=3, d_min=6) for k in ("bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w")}
CELLS.update({k: dict(v2.CELLS[k], target=[1], r_loc=2, d_min=4) for k in ("openms_t2w", "openms_t1w")})
CELLS["pelvis_ct_full"] = dict(v2.CELLS["pelvis_ct"], cases=PELVIS_CASES, target=list(range(1, 12)), r_loc=3, d_min=6)
TITLE = {"bratsfull_t2w_t1n": "BraTS T2w-trained -> T1n (edema)", "bratsfull_t2w_t2f": "BraTS T2w-trained -> FLAIR (edema)",
         "bratsfull_t1n_t2f": "BraTS T1n-trained -> FLAIR (edema)", "bratsfull_t1n_t2w": "BraTS T1n-trained -> T2w (edema)",
         "openms_t2w": "Open-MS FLAIR-trained -> T2w (lesion)", "openms_t1w": "Open-MS FLAIR-trained -> T1w (lesion)",
         "pelvis_ct_full": "Pelvis MRI-trained -> CT (bones/muscles, control)"}
ARMS = ("flat", "real"); N_SEEDS = int(os.environ.get("ERF_SEEDS", 12)); CONF = 2.0; EDGE = 20; RIM = 2; BATCH = int(os.environ.get("ERF_BATCH", 2)); R_MAX = 60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def patch_at(m, size, shape):
    c = np.argwhere(m).mean(0).astype(int) if m.any() else np.array(shape) // 2
    size = np.minimum(np.array(size), np.array(shape)); lo = np.clip(c - size // 2, 0, np.array(shape) - size)
    return tuple(slice(int(a), int(a + s)) for a, s in zip(lo, size))


def fwd(net, x):
    o = net(x); return o[0] if isinstance(o, (list, tuple)) else o


def log_odds(o, c):
    """o: (B, C, ...) logits -> log-odds of class c (B, ...)."""
    oth = torch.cat([o[:, :c], o[:, c + 1:]], 1)
    return o[:, c] - torch.logsumexp(oth, 1)


def seed_grads(net, x, seeds, cls, dev):
    """x: (Z,Y,X) float32 numpy; seeds: list of (z,y,x); cls: list of classes -> list of gradient arrays (same shape as x)."""
    out = []
    xt = torch.from_numpy(x).to(dev)
    for i in range(0, len(seeds), BATCH):
        sb, cb = seeds[i:i + BATCH], cls[i:i + BATCH]
        xb = xt[None, None].repeat(len(sb), 1, 1, 1, 1).clone().requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            o = fwd(net, xb)
            f = sum(log_odds(o[b:b + 1], c)[0][tuple(s)] for b, (s, c) in enumerate(zip(sb, cb)))
            g, = torch.autograd.grad(f, xb)
        out += [g[b, 0].detach() for b in range(len(sb))]
    return out


def region_geometry(gp, c):
    reg = gp == c; din = ndi.distance_transform_edt(reg); dout = ndi.distance_transform_edt(~reg)
    return reg, din, ((din <= RIM) & reg) | ((dout <= RIM) & ~reg)


def seed_metrics(G, v, rim, body, r_loc):
    """G: |grad| (numpy); v: seed; rim/body: bool masks -> dict of metrics + distance profile."""
    zz, yy, xx = np.indices(G.shape, sparse=True); r = np.sqrt((zz - v[0]) ** 2 + (yy - v[1]) ** 2 + (xx - v[2]) ** 2)
    tot = G[body].sum() + 1e-30; ball = r <= r_loc
    rb = np.minimum(np.round(r).astype(int), R_MAX)
    nr = body & ~rim
    num = np.bincount(rb[nr], G[nr], minlength=R_MAX + 1); den = np.bincount(rb[nr], minlength=R_MAX + 1)
    expect = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    rr = rim & body; e_rim = expect[rb[rr]]; ok = np.isfinite(e_rim)
    rim_excess = G[rr][ok].sum() / (e_rim[ok].sum() + 1e-30)
    return dict(local_share=float(G[ball & body].sum() / tot), rim_share=float(G[rr].sum() / tot), rim_excess=float(rim_excess))


def run_cell(key, dev, sl=None, only=None):
    cell = CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True); T = cell["target"]; r_loc, d_min = cell["r_loc"], cell["d_min"]
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]); cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    if only:
        cases = [c for c in cases if c in only]
    f_csv = od / "seeds.csv"; done = set(pd.read_csv(f_csv).case) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    rng = np.random.default_rng(0)
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        if tg.sum() < 50:
            continue
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        # fold-mean log-odds + 3-fold majority class per arm (no grad)
        lo, maj = {}, {}
        for a, pr in prs.items():
            net = pr.network; acc = 0; votes = []
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    o = fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()
                acc = acc + o[0]; votes.append(o[0].argmax(0))
            lo[a] = (acc / len(v2.FOLDS)).cpu().numpy(); V = torch.stack(votes)
            maj[a] = torch.mode(V, 0).values.cpu().numpy()
        # candidate seeds
        geo = {c: region_geometry(gp, c) for c in T if (gp == c).sum() >= 50}
        inner = np.zeros(gp.shape, bool); inner[tuple(slice(EDGE, s - EDGE) for s in gp.shape)] = True
        cand = []
        for c, (reg, din, rim) in geo.items():
            oth = [j for j in range(lo["flat"].shape[0]) if j != c]
            ok = reg & inner & (din >= d_min) & bp
            for a in ARMS:
                la = lo[a][c] - np.log(np.exp(lo[a][oth] - lo[a][oth].max(0)).sum(0)) - lo[a][oth].max(0)
                ok &= (maj[a] == c) & (la > CONF)
            idx = np.argwhere(ok); cand += [(tuple(i), c, float(din[tuple(i)])) for i in idx]
        n_cand = len(cand)
        if n_cand == 0:
            log(f"[{key}] {case}: no seed (both-correct, depth >= {d_min}), skipped"); continue
        dep = np.array([d for _, _, d in cand]); qs = np.quantile(dep, [0, 1 / 3, 2 / 3, 1]); pick = []
        for b in range(3):
            ib = np.where((dep >= qs[b]) & (dep <= qs[b + 1]))[0]
            pick += list(rng.choice(ib, min(len(ib), N_SEEDS // 3), replace=False)) if len(ib) else []
        seeds = [cand[i][0] for i in pick]; cls = [int(cand[i][1]) for i in pick]; depth = [cand[i][2] for i in pick]
        Gs = {}
        for a, pr in prs.items():
            net = pr.network; acc = None
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                g = seed_grads(net, ip, seeds, cls, dev); acc = g if acc is None else [x + y for x, y in zip(acc, g)]
            Gs[a] = [(x / len(v2.FOLDS)).abs().cpu().numpy() for x in acc]
        net = prs["flat"].network; torch.manual_seed(0)
        for m in net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        net.eval(); Gs["random"] = [x.abs().cpu().numpy() for x in seed_grads(net, ip, seeds, cls, dev)]
        edge = np.sqrt(sum(ndi.sobel(ndi.gaussian_filter(ip, 1.0), axis=ax) ** 2 for ax in range(3)))
        rows = []
        for i, (v, c, d) in enumerate(zip(seeds, cls, depth)):
            reg, din, rim = geo[c]; row = dict(cell=key, case=case, seed=i, z=v[0], y=v[1], x=v[2], cls=c, depth=d, n_cand=n_cand)
            for a in ARMS + ("random",):
                for mk, mv in seed_metrics(Gs[a][i], v, rim, bp, r_loc).items():
                    row[f"{a}_{mk}"] = mv
            for mk, mv in seed_metrics(edge, v, rim, bp, r_loc).items():
                row[f"edge_{mk}"] = mv
            rows.append(row)
        df = pd.DataFrame(rows)
        # save the slice through every seed (axis 0) for the sheets: image, GT class region, both arms' majority masks, |G| per arm
        sv = {"img": [], "gt": [], "pred_flat": [], "pred_real": [], "G_flat": [], "G_real": [], "G_random": []}
        for i, (v, c) in enumerate(zip(seeds, cls)):
            z = v[0]; sv["img"].append(ip[z]); sv["gt"].append(gp[z] == c); sv["pred_flat"].append(maj["flat"][z] == c); sv["pred_real"].append(maj["real"][z] == c)
            for a in ARMS + ("random",):
                sv[f"G_{a}"].append(Gs[a][i][z])
        np.savez_compressed(od / f"{case}_erf.npz", seeds=np.array(seeds), cls=np.array(cls), depth=np.array(depth),
                            **{k: np.array(vv).astype(np.float16 if k.startswith(("img", "G_")) else bool) for k, vv in sv.items()})
        df.to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)
        med = df.median(numeric_only=True)
        log(f"[{key}] {case}: {len(seeds)} seeds (of {n_cand}), depth {np.median(depth):.1f} | rim_excess flat {med.flat_rim_excess:.2f} real {med.real_rim_excess:.2f} "
            f"random {med.random_rim_excess:.2f} | local_share flat {med.flat_local_share:.3f} real {med.real_local_share:.3f} random {med.random_local_share:.3f}; "
            f"{time.time() - t0:.0f} s")
        # restore the trained weights for the next case (reset_parameters overwrote them)
        prs["flat"].network.load_state_dict(prs["flat"].list_of_parameters[0])


def num(d):
    d = d[d.case != "case"].copy()
    for c in d.columns:
        try:
            d[c] = pd.to_numeric(d[c])
        except (ValueError, TypeError):
            pass
    return d


def p_w(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan


def report():
    import matplotlib.colors as mcolors
    L = ["# Visual proof v7: where each model looks to call a deep target voxel (per-seed input gradients, unperturbed images)", "",
         "Per case: median over seeds (both models correct + confident, depth >= d_min). Paired Wilcoxon over cases, flat vs real.", "",
         "| cell | n cases | seeds | rim_excess flat | real | random | edge map | p | local_share flat | real | random | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per = {}
    for key in CELLS:
        f = OUT / key / "seeds.csv"
        if not f.exists():
            continue
        d = num(pd.read_csv(f))
        c = d.groupby("case").median(numeric_only=True); per[key] = c
        L.append(f"| {TITLE[key]} | {len(c)} | {len(d)} | {c.flat_rim_excess.median():.2f} | {c.real_rim_excess.median():.2f} | {c.random_rim_excess.median():.2f} | "
                 f"{c.edge_rim_excess.median():.2f} | {p_w(np.log(c.flat_rim_excess) - np.log(c.real_rim_excess)):.2g} | {c.flat_local_share.median():.3f} | "
                 f"{c.real_local_share.median():.3f} | {c.random_local_share.median():.3f} | {p_w(c.flat_local_share - c.real_local_share):.2g} |")
    (OUT / "erf_stats.md").write_text("\n".join(L)); print("\n".join(L))
    if not per:
        return
    # violins: per-case log2 rim_excess and local_share, flat vs real, paired lines
    fig, axs = plt.subplots(2, 1, figsize=(max(6, 2.2 * len(per)), 8))
    for ai, (m, lab) in enumerate((("rim_excess", "contour preference  log2(rim_excess)"), ("local_share", "evidence within the voxel's neighbourhood (share)"))):
        ax = axs[ai]
        for j, (key, c) in enumerate(per.items()):
            vals = [np.log2(c[f"{a}_{m}"]) if m == "rim_excess" else c[f"{a}_{m}"] for a in ARMS]
            for o, (vv, col) in enumerate(zip(vals, ("tab:blue", "tab:orange"))):
                vp = ax.violinplot([vv.dropna()], positions=[3 * j + o], widths=0.8, showmedians=True)
                for b in vp["bodies"]:
                    b.set_facecolor(col); b.set_alpha(0.4)
            for y0, y1 in zip(*vals):
                ax.plot([3 * j, 3 * j + 1], [y0, y1], color="grey", lw=0.4, alpha=0.5)
        ax.set_xticks([3 * j + 0.5 for j in range(len(per))]); ax.set_xticklabels([TITLE[k].replace(" (", "\n(") for k in per], fontsize=7)
        ax.set_ylabel(lab, fontsize=9)
        if m == "rim_excess":
            ax.axhline(0, color="k", lw=0.5)
    axs[0].set_title("blue = flat-fill model (rung 4.5), orange = real-fill model (rung 5); one line per case", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "erf_violins.png", dpi=150); plt.close(fig)
    # radial profile (in-plane, from the saved seed slices): evidence density vs distance from the explained voxel, real / flat, per cell
    fig, ax = plt.subplots(figsize=(7, 4.5)); RB = 30
    for key, c in per.items():
        curves = []
        for case in c.index:
            f = OUT / key / f"{case}_erf.npz"
            if not f.exists():
                continue
            z = np.load(f); acc = {a: np.zeros(RB + 1) for a in ARMS}
            for i, sd in enumerate(z["seeds"]):
                yy, xx = np.indices(z["img"][i].shape); r = np.minimum(np.round(np.hypot(yy - sd[1], xx - sd[2])).astype(int), RB); n = np.bincount(r.ravel(), minlength=RB + 1)
                for a in ARMS:
                    G = z[f"G_{a}"][i].astype(float); acc[a] += np.bincount(r.ravel(), G.ravel(), minlength=RB + 1) / np.maximum(n, 1) / (G.sum() + 1e-30)
            curves.append(np.log2(acc["real"] / acc["flat"]))
        if len(curves) >= 5:   # Open-MS: 3 cases with deep voxels per cell -- too few for a curve
            C = np.array(curves); m = np.median(C, 0); q1, q3 = np.percentile(C, [25, 75], 0)
            ls = "--" if key.startswith("pelvis") else ("-" if key.startswith("brats") else ":")
            ax.plot(np.arange(RB + 1), m, ls, label=f"{TITLE[key]} (n={len(C)})"); ax.fill_between(np.arange(RB + 1), q1, q3, alpha=0.08)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("in-plane distance from the explained voxel (vox = mm)"); ax.set_ylabel("log2 evidence density, real-fill / flat-fill")
    ax.set_title("Where the evidence for a deep target voxel comes from (normalised per map; median over cases, IQR shaded)", fontsize=8); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "erf_radial_ratio.png", dpi=150); plt.close(fig)
    seed_centred(per)
    # hero sheets: for each cell, the case closest to the cell median of the flat-real rim_excess difference; its seed closest to that case's median
    for key, c in per.items():
        dd = np.log(c.flat_rim_excess) - np.log(c.real_rim_excess); case = (dd - dd.median()).abs().idxmin()
        z = np.load(OUT / key / f"{case}_erf.npz"); d = num(pd.read_csv(OUT / key / "seeds.csv")); d = d[d.case == case]
        sd = np.log(d.flat_rim_excess) - np.log(d.real_rim_excess); i = int(d.seed.values[np.argmin(np.abs(sd.values - sd.median()))])
        sz = z["seeds"][i]; img = z["img"][i].astype(float); gt = z["gt"][i]
        fig, axs = plt.subplots(2, 3, figsize=(11, 7.6))
        for col, a in enumerate(("flat", "real", "random")):
            G = z[f"G_{a}"][i].astype(float); vmax = np.percentile(G, 99.9) + 1e-30
            ax = axs[0, col]; ax.imshow(img, cmap="gray"); ax.imshow(G, cmap="inferno", norm=mcolors.LogNorm(vmin=vmax / 300, vmax=vmax), alpha=0.75)
            ax.contour(gt, [0.5], colors="cyan", linewidths=0.7); ax.plot(sz[2], sz[1], "+", color="lime", ms=12, mew=2)
            r = d[d.seed == i].iloc[0]
            ax.set_title({"flat": "flat-fill model (rung 4.5)", "real": "real-fill model (rung 5)", "random": "random weights (control)"}[a]
                         + f"\nrim excess {r[f'{a}_rim_excess']:.2f}, local share {r[f'{a}_local_share']:.3f}", fontsize=9)
            ax.axis("off")
            ax = axs[1, col]; ax.imshow(img, cmap="gray"); ax.contour(gt, [0.5], colors="cyan", linewidths=0.7)
            if a in ARMS:
                ax.imshow(np.ma.masked_where(~z[f"pred_{a}"][i], z[f"pred_{a}"][i]), cmap="autumn", alpha=0.45); ax.set_title(f"{a}-fill model prediction (3-fold majority)", fontsize=9)
            else:
                ax.set_title("cyan = ground truth, + = the voxel being explained", fontsize=9)
            ax.plot(sz[2], sz[1], "+", color="lime", ms=12, mew=2); ax.axis("off")
        fig.suptitle(f"{TITLE[key]}: |d log-odds(voxel +) / d image|, fold-mean, {case} (median case, median seed, depth {z['depth'][i]:.0f} vox)", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"erf_hero_{key}_{case}.png", dpi=150); plt.close(fig)


LIM = 0.4; SMOOTH = 1.0   # display smoothing of the case-averaged maps (Gaussian, px = mm), applied identically to both arms before the ratio


def seed_centred(per, W=30):
    """Noise-reduced map: every seed's in-plane |G| slice, normalised to sum 1, cropped +-W around the seed (outside the patch = NaN),
    averaged over the seeds of a case, then over cases (equal weight per case). Also the fraction of seeds whose GT region covers each pixel."""
    import matplotlib.colors as mcolors
    keys = [k for k, c in per.items() if len(c) >= 5]; res = {}
    for key in keys:
        cs = {a: [] for a in ARMS + ("random",)}; gts = []; inside = {a: [] for a in cs}; rexc = {a: [] for a in cs}; cases_used = []
        for case in per[key].index:
            f = OUT / key / f"{case}_erf.npz"
            if not f.exists():
                continue
            z = np.load(f); acc = {a: [] for a in cs}; ga = []; ins = {a: [] for a in cs}; rex = {a: [] for a in cs}
            for i, sd in enumerate(z["seeds"]):
                body = z["img"][i] != 0; reg = z["gt"][i] & body
                yy, xx = np.indices(body.shape); rb = np.round(np.hypot(yy - sd[1], xx - sd[2])).astype(int)
                for a in cs:
                    G = z[f"G_{a}"][i].astype(float) * body; G = G / (G.sum() + 1e-30); ins[a].append(G[reg].sum())
                    # region excess: evidence inside the target vs what an isotropic fall-off with distance from the voxel predicts
                    e = np.bincount(rb[body], G[body]) / np.maximum(np.bincount(rb[body]), 1); rex[a].append(G[reg].sum() / (e[rb[reg]].sum() + 1e-30))
                    Gp = np.pad(G, W, constant_values=np.nan); acc[a].append(Gp[sd[1]:sd[1] + 2 * W + 1, sd[2]:sd[2] + 2 * W + 1])
                gp = np.pad(z["gt"][i].astype(float), W, constant_values=np.nan); ga.append(gp[sd[1]:sd[1] + 2 * W + 1, sd[2]:sd[2] + 2 * W + 1])
            for a in cs:
                cs[a].append(np.nanmean(acc[a], 0)); inside[a].append(np.median(ins[a])); rexc[a].append(np.median(rex[a]))
            cases_used.append(case)
            gts.append(np.nanmean(ga, 0))
        res[key] = ({a: ndi.gaussian_filter(np.nan_to_num(np.nanmean(v, 0)), SMOOTH) for a, v in cs.items()}, np.nanmean(gts, 0), len(gts),
                    np.array([np.log2(r / f) for r, f in zip(cs["real"], cs["flat"])]), {a: np.array(v) for a, v in inside.items()},
                    {a: np.array(v) for a, v in rexc.items()}, cases_used)
    if not res:
        return
    fig, axs = plt.subplots(len(res), 4, figsize=(13, 3.2 * len(res)), squeeze=False); ext = [-W - 0.5, W + 0.5, W + 0.5, -W - 0.5]
    for r_, (key, (M, gt, n, per_case, inside, rexc, cu)) in enumerate(res.items()):
        vmax = np.nanmax([M["flat"], M["real"]]); 
        for c_, a in enumerate(ARMS):
            ax = axs[r_, c_]; ax.imshow(M[a], cmap="inferno", norm=mcolors.LogNorm(vmin=vmax / 100, vmax=vmax), extent=ext)
            ax.contour(np.arange(-W, W + 1), np.arange(-W, W + 1), gt, [0.5], colors="cyan", linewidths=0.8)
            ax.set_title(("flat-fill model (rung 4.5)" if a == "flat" else "real-fill model (rung 5)") + ": mean evidence", fontsize=8)
        L = np.log2(M["real"] / M["flat"]); lim = LIM
        ax = axs[r_, 2]; im = ax.imshow(L, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext); plt.colorbar(im, ax=ax, fraction=0.046)
        ax.contour(np.arange(-W, W + 1), np.arange(-W, W + 1), gt, [0.5], colors="k", linewidths=0.8)
        ax.add_patch(plt.Circle((0, 0), 3, fill=False, color="k", lw=0.6, ls="--")); ax.set_title("log2 real / flat (red = real-fill relies more)", fontsize=8)
        # consistency: per-case sign of the core (r <= 3) minus the ring (6..20)
        yy, xx = np.mgrid[-W:W + 1, -W:W + 1]; rr = np.hypot(yy, xx); core = rr <= 3; ring = (rr >= 6) & (rr <= 20)
        dc = np.array([np.nanmean(pc[core]) - np.nanmean(pc[ring]) for pc in per_case]); pv = p_w(dc)
        ax = axs[r_, 3]; prof = [np.nanmean(L[(rr >= d - 0.5) & (rr < d + 0.5)]) for d in range(W + 1)]
        pcs = np.array([[np.nanmean(pc[(rr >= d - 0.5) & (rr < d + 0.5)]) for d in range(W + 1)] for pc in per_case])
        ax.fill_between(range(W + 1), *np.nanpercentile(pcs, [25, 75], 0), alpha=0.2); ax.plot(range(W + 1), prof, lw=1.5); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"core minus 6-20 mm ring > 0:\n{np.mean(dc > 0) * 100:.0f}% of cases, p {pv:.1g}", fontsize=8); ax.set_xlabel("distance from voxel (mm)", fontsize=8)
        for c_ in range(3):
            axs[r_, c_].plot(0, 0, "+", color="lime", ms=8, mew=1.5); axs[r_, c_].set_xticks([]); axs[r_, c_].set_yticks([])
        axs[r_, 0].set_ylabel(f"{TITLE[key]}\n{n} cases", fontsize=8)
    fig.suptitle("Evidence for a deep target voxel, centred on that voxel and averaged over all explained voxels (in-plane; cyan/black = target region in >= 50% of seeds)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "erf_seed_centred.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, len(res), figsize=(3.1 * len(res) + 0.8, 3.6), squeeze=False); axs = axs[0]
    lines = ["", "## Seed-centred maps (POST HOC metrics; in-plane slice through the voxel, body-masked, per case = median over seeds)", "",
             "inside share = share of the evidence inside the GT target region; region excess = inside share / inside share predicted by an isotropic",
             "fall-off with distance from the voxel (1 = 'just more local'; > 1 = follows the region).", "",
             "| cell | n cases | inside share flat | real | random | real > flat | p | region excess flat | real | random | real > flat | p |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    qlines = ["", "## Does it depend on prediction quality? (per-case Dice of each model, original image, same patch: visual_proof_strip2 g0)", "",
              "| cell | n | dDice real-flat (median) | Spearman(dDice, d inside) | p | Dice-matched (abs dDice <= 0.05) n | real > flat | p | d inside at dDice = 0 (OLS intercept) | p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    from scipy.stats import spearmanr, linregress, t as tdist
    for j, (key, (M, gt, n, per_case, inside, rexc, cu)) in enumerate(res.items()):
        L = np.log2(M["real"] / M["flat"]); ax = axs[j]
        im = ax.imshow(L, cmap="RdBu_r", vmin=-LIM, vmax=LIM, extent=ext); ax.contour(np.arange(-W, W + 1), np.arange(-W, W + 1), gt, [0.5], colors="k", linewidths=1)
        ax.plot(0, 0, "+", color="lime", ms=9, mew=2); ax.set_xticks([]); ax.set_yticks([])
        d = inside["real"] - inside["flat"]; pv = p_w(d)
        ax.set_title(f"{TITLE[key].replace(' (', chr(10) + '(')}\nevidence inside target: real > flat\nin {np.mean(d > 0) * 100:.0f}% of {n} cases (p {pv:.1g})", fontsize=8)
        dr = np.log(rexc["real"]) - np.log(rexc["flat"])
        lines.append(f"| {TITLE[key]} | {n} | {np.median(inside['flat']):.3f} | {np.median(inside['real']):.3f} | {np.median(inside['random']):.3f} | {np.mean(d > 0) * 100:.0f}% | {pv:.2g} | "
                     f"{np.median(rexc['flat']):.3f} | {np.median(rexc['real']):.3f} | {np.median(rexc['random']):.3f} | {np.mean(dr > 0) * 100:.0f}% | {p_w(dr):.2g} |")
        fq = OUT.parent / "visual_proof_strip2" / key / "cases.csv"
        if fq.exists():
            q = num(pd.read_csv(fq)).drop_duplicates("case").set_index("case")
            dd = np.array([q.real_g0_dice.get(c, np.nan) - q.flat_g0_dice.get(c, np.nan) for c in cu]); ok = np.isfinite(dd)
            if ok.sum() >= 5:
                rho, prho = spearmanr(dd[ok], d[ok]); mt = ok & (np.abs(np.nan_to_num(dd, nan=9)) <= 0.05); lr = linregress(dd[ok], d[ok])
                pint = 2 * tdist.sf(abs(lr.intercept / lr.intercept_stderr), ok.sum() - 2)
                qlines.append(f"| {TITLE[key]} | {ok.sum()} | {np.median(dd[ok]):+.3f} | {rho:+.2f} | {prho:.2g} | {mt.sum()} | {np.mean(d[mt] > 0) * 100:.0f}% | {p_w(d[mt]):.2g} | "
                              f"{lr.intercept:+.4f} | {pint:.2g} |")
    cb = fig.colorbar(im, ax=list(axs), fraction=0.015, pad=0.01); cb.set_label("log2 evidence, real-fill / flat-fill", fontsize=8)
    fig.suptitle("Where each model takes the evidence for a deep target voxel (+), averaged over all explained voxels (red = real-fill model relies more; "
                 f"black = target region; each map normalised; Gaussian {SMOOTH:g} mm display smoothing)", fontsize=8)
    fig.savefig(OUT / "erf_seed_centred_compact.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    with open(OUT / "erf_stats.md", "a") as fh:
        fh.write("\n".join(lines + qlines) + "\n")
    print("\n".join(lines + qlines))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELLS)); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--cases", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    for c in a.cells.split(","):
        run_cell(c, torch.device("cuda"), a.case_slice, set(a.cases.split(",")) if a.cases else None)


if __name__ == "__main__":
    main()
