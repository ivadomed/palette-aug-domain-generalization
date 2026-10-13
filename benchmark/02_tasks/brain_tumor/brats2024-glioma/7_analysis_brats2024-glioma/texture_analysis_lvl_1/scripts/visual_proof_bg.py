#!/usr/bin/env python
"""
Visual proof v8, output side (2026-10-12): HOW each model calls background vs target, from its own predictions on unperturbed held-out-contrast
test images (same cells, cases, patch and fold-mean logits as visual_proof_erf_tasks.py; no gradient, no intervention).
Per case and model: L = log-odds of "any target class" vs every other class, per voxel of the patch. Per voxel: signed distance s to the
ground-truth target boundary (> 0 inside), and image descriptors at a 2-voxel scale: local mean intensity (Gaussian sigma 2), fine texture
energy (RMS of img - Gaussian(img, 1), averaged at sigma 2), coarse edge strength (Gaussian gradient magnitude, sigma 2).
Questions:
  1. Profile: L against s, flat vs real (how sharp / how far from the lesion is background called confidently).
  2. Background decision rule: among background voxels (s < -3), how L depends on how lesion-like their INTENSITY is (|z| of the local mean
     against the case's own target voxels) and how lesion-like their TEXTURE is (|z| of log texture energy against the target voxels);
     per-case linear fit L ~ |z_int| + |z_tex| + log distance (standardised within case), coefficients compared between models.
  3. Hard negatives: background voxels whose local intensity matches the lesion (|z_int| < 0.5): false-positive rate (L > 0) of each model,
     split by texture match.
GPU: --cells <cell> [--case-slice i:j] -> VP8_OUT/bg/<cell>/<case>.npz      CPU: --report -> VP8_OUT/bg/bg_*.png, bg_summary.md
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402

e7, v2 = vt.e7, vt.v2; ARMS = vt.ARMS
OUT = vt.OUT / "bg"; OUT.mkdir(exist_ok=True)
N_BG, N_TG, N_OT = 16000, 4000, 4000
CELLS = ["glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w", "ms_flair_t2w", "ms_flair_t1w", "ms_t1w_flair", "ms_t1w_t2w",
         "breast_t1wce_t2w", "pancreas_t1wce_t2w", "mand_cbct_hanseg_ct", "mand_cbct_pddca_ct", "abd_t1in_ct", "pelvis_mri_ct", "pelvis_ct_mri", "brain_t2w_t1w"]


def lse(x, axis=0):
    m = x.max(axis); return m + np.log(np.exp(x - m).sum(axis))


def transform_img(img, body, cond):
    """Counterfactual test images (section 30 of FINDINGS): flat = 4-level 1-D k-means of the body intensities, each voxel -> its bin mean;
    inv = -x in the body; both re-z-scored over the body afterwards. Background (outside the body) untouched."""
    x = img.copy(); v = x[body].astype(np.float64)
    if "flat" in cond:
        c = np.quantile(v, [0.125, 0.375, 0.625, 0.875])
        for _ in range(30):
            lab = np.abs(v[:, None] - c[None]).argmin(1); c = np.array([v[lab == k].mean() if (lab == k).any() else c[k] for k in range(len(c))])
        v = c[np.abs(v[:, None] - c[None]).argmin(1)]
    if "inv" in cond:
        v = -v
    x[body] = ((v - v.mean()) / (v.std() + 1e-6)).astype(x.dtype); return x


def run_bg(key, dev, sl=None, cond="orig"):
    import torch
    cell = vt.CELLS[key]; od = (OUT if cond == "orig" else vt.OUT / "bg_cond" / cond) / key; od.mkdir(parents=True, exist_ok=True); T = cell["target"]
    prs = {a: v2.load_predictor(f(), dev) for a, f in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"])[:vt.MAX_CASES] if not key.startswith("glioma") else list(cell["cases"])
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    rng = np.random.default_rng(0)
    for case in cases:
        if (od / f"{case}.npz").exists():
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        if tg.sum() < 50:
            continue
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        if cond != "orig":
            img = transform_img(img, body, cond)
        box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        L, P, LO = {}, {}, {}
        for a, pr in prs.items():
            net = pr.network; acc = 0; votes = []
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()
                acc = acc + o[0]; votes.append(o[0].argmax(0))
            lo = (acc / len(v2.FOLDS)).cpu().numpy(); oth = [j for j in range(lo.shape[0]) if j not in T]
            LO[a] = lo; L[a] = (lse(lo[T]) - lse(lo[oth])).astype(np.float32); P[a] = np.isin(torch.mode(torch.stack(votes), 0).values.cpu().numpy(), T)
        tgp = np.isin(gp, T); s = (ndi.distance_transform_edt(tgp) - ndi.distance_transform_edt(~tgp)).astype(np.float32)
        olab = (gp != 0) & ~tgp   # other labels (glioma: tumour core NCR/ET/RC): distance to them = how far the voxel is from that context
        dcore = ndi.distance_transform_edt(~olab).astype(np.float32) if olab.any() else np.full(gp.shape, 999.0, np.float32)
        lm = ndi.gaussian_filter(ip, 2.0); hp = ip - ndi.gaussian_filter(ip, 1.0); te = np.sqrt(ndi.gaussian_filter(hp ** 2, 2.0)); ed = ndi.gaussian_gradient_magnitude(ip, 2.0)
        # pre-registered from the fill code (rung 4.5 = flat patches per k-means bin x Voronoi cell): heterogeneity at the patch scale
        # (SD of the 1-voxel-smoothed image in a sigma-3 window) and local contrast to the immediate surroundings (sigma 2 mean - sigma 6 mean)
        g1 = ndi.gaussian_filter(ip, 1.0); het = np.sqrt(np.maximum(ndi.gaussian_filter(g1 ** 2, 3.0) - ndi.gaussian_filter(g1, 3.0) ** 2, 0)); lc = lm - ndi.gaussian_filter(ip, 6.0)
        inner = np.zeros(gp.shape, bool); inner[tuple(slice(e7.EDGE, n - e7.EDGE) for n in gp.shape)] = True
        kinds = {0: (gp == 0) & bp & inner, 1: tgp & bp & inner, 2: (gp != 0) & ~tgp & bp & inner}
        idx, kk = [], []
        for k, n in ((0, N_BG), (1, N_TG), (2, N_OT)):
            w = np.flatnonzero(kinds[k])
            if len(w):
                w = rng.choice(w, min(n, len(w)), replace=False); idx.append(w); kk.append(np.full(len(w), k, np.int8))
        idx = np.concatenate(idx); kk = np.concatenate(kk); fl = lambda x: x.ravel()[idx]   # noqa: E731
        tv = tgp & bp
        stats = dict(mu_lm=float(lm[tv].mean()), sd_lm=float(lm[tv].std()), mu_lte=float(np.log(te[tv] + 1e-6).mean()), sd_lte=float(np.log(te[tv] + 1e-6).std()),
                     bg_mu_lm=float(lm[kinds[0]].mean()) if kinds[0].any() else np.nan, bg_sd_lm=float(lm[kinds[0]].std()) if kinds[0].any() else np.nan)
        bgall = kinds[0]; far = bgall & (s < -3)
        for a in ARMS:
            stats[f"fp_{a}"] = float((L[a][far] > 0).mean()) if far.any() else np.nan
            stats[f"dice_{a}"] = vt.dice(P[a] & inner, tgp & inner)
        np.savez_compressed(od / f"{case}.npz", kind=kk, s=fl(s).astype(np.float16), img=fl(ip).astype(np.float16), lm=fl(lm).astype(np.float16), te=fl(te).astype(np.float16),
                            ed=fl(ed).astype(np.float16), het=fl(het).astype(np.float16), lc=fl(lc).astype(np.float16), dcore=fl(dcore).astype(np.float16), L_flat=fl(L["flat"]).astype(np.float16), L_real=fl(L["real"]).astype(np.float16), lo_flat=LO["flat"].reshape(LO["flat"].shape[0], -1)[:, idx].astype(np.float16), lo_real=LO["real"].reshape(LO["real"].shape[0], -1)[:, idx].astype(np.float16), lab=fl(gp).astype(np.int16), **{k: np.array(v) for k, v in stats.items()})
        vt.log(f"[bg {key}] {case}: Dice flat/real {stats['dice_flat']:.2f}/{stats['dice_real']:.2f}, background FP (s<-3) flat/real "
               f"{stats['fp_flat']:.4f}/{stats['fp_real']:.4f}; {time.time() - t0:.0f} s")


def load(cell, base=None):
    rows = []
    for f in sorted(((base or OUT) / cell).glob("*.npz")):
        z = np.load(f); d = pd.DataFrame({k: z[k].astype(np.float32) for k in ("s", "img", "lm", "te", "ed", "het", "lc", "dcore", "L_flat", "L_real") if k in z}); d["kind"] = z["kind"]
        for k in ("mu_lm", "sd_lm", "mu_lte", "sd_lte", "fp_flat", "fp_real", "dice_flat", "dice_real"):
            d[k] = float(z[k])
        d["case"] = f.stem; rows.append(d)
    if not rows:
        return None
    D = pd.concat(rows, ignore_index=True)
    D["zi"] = (D.lm - D.mu_lm) / (D.sd_lm + 1e-6); D["zt"] = (np.log(D.te + 1e-6) - D.mu_lte) / (D.sd_lte + 1e-6)
    return D


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    data = {c: load(c) for c in CELLS}; data = {c: d for c, d in data.items() if d is not None}; cells = list(data)
    L = ["# How each model calls background vs target (fold-mean log-odds of the target on unperturbed held-out-contrast images)", "",
         "per case: median / fit over its sampled voxels; then median over cases; p = Wilcoxon over cases, real vs flat. AGREE = |Dice_real - Dice_flat| <= 0.05 on the patch.", "",
         "## 1. Confidence across the boundary and background false positives", "",
         "| cell | n cases | Dice flat / real | bg FP rate (s < -3) flat / real | p | median L at s=-10 flat / real | s=-3 | s=+3 | s=+8 | L=0 crossing s (flat / real) |", "|---|---|---|---|---|---|---|---|---|---|"]
    bins = np.arange(-25, 16); prof = {}
    for c in cells:
        D = data[c]; cs = D.groupby("case").first()
        pr = {}
        for a in ARMS:
            m = D[D.kind != 2].assign(b=np.clip(np.round(D.s), -25, 15)).groupby(["case", "b"])[f"L_{a}"].median().unstack()
            pr[a] = m.reindex(columns=bins).median(0)
        prof[c] = pr
        def cross(p):
            v = p.values; i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]; return float(bins[i[0]] + (-v[i[0]]) / (v[i[0] + 1] - v[i[0]])) if len(i) else np.nan
        g = lambda a, b: f"{pr['flat'].get(b, np.nan):+.1f} / {pr['real'].get(b, np.nan):+.1f}"   # noqa: E731
        L.append(f"| {title(c)} | {len(cs)} | {cs.dice_flat.median():.2f} / {cs.dice_real.median():.2f} | {cs.fp_flat.median():.4f} / {cs.fp_real.median():.4f} | {pw(cs.fp_real - cs.fp_flat):.1g} | "
                 f"{g(0, -10)} | {g(0, -3)} | {g(0, 3)} | {g(0, 8)} | {cross(pr['flat']):+.1f} / {cross(pr['real']):+.1f} |")
    # 2. background decision rule
    L += ["", "## 2. What makes a background voxel look like target (background voxels, s < -3)", "",
          "Per case: OLS L ~ |z_int| + |z_tex| + log(-s) on standardised predictors (L in log-odds units per SD). z_int = local mean intensity vs the case's target voxels, "
          "z_tex = log texture energy vs the target voxels. Negative coefficient = the less lesion-like, the more confidently background.", "",
          "| cell | n | b intensity flat | real | p | b texture flat | real | p | b distance flat | real | p | texture / intensity weight flat | real | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    coef = {}
    for c in cells:
        D = data[c]; B = D[(D.kind == 0) & (D.s < -3)]; rows = []
        for case, w in B.groupby("case"):
            if len(w) < 200:
                continue
            X = np.c_[np.abs(w.zi), np.abs(w.zt), np.log(-w.s)]; X = (X - X.mean(0)) / (X.std(0) + 1e-9); X = np.c_[np.ones(len(w)), X]
            r = dict(case=case)
            for a in ARMS:
                b = np.linalg.lstsq(X, w[f"L_{a}"].values.astype(float), rcond=None)[0]; r.update({f"{a}_bi": b[1], f"{a}_bt": b[2], f"{a}_bd": b[3]})
                r[f"{a}_rat"] = abs(b[2]) / (abs(b[1]) + abs(b[2]) + 1e-9)
            rows.append(r)
        C = pd.DataFrame(rows); coef[c] = C
        if not len(C):
            continue
        f = lambda m: f"{C[f'flat_{m}'].median():+.2f} | {C[f'real_{m}'].median():+.2f} | {pw(C[f'real_{m}'] - C[f'flat_{m}']):.1g}"   # noqa: E731
        L.append(f"| {title(c)} | {len(C)} | {f('bi')} | {f('bt')} | {f('bd')} | {f('rat')} |")
    # 3. hard negatives
    L += ["", "## 3. Hard negatives: background voxels (s < -3) whose local intensity matches the target (|z_int| < 0.5)", "",
          "| cell | n | share of bg voxels | FP rate flat | real | p | FP rate when texture also matches (|z_tex| < 0.5) flat | real | when texture differs (|z_tex| > 1.5) flat | real |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        D = data[c]; B = D[(D.kind == 0) & (D.s < -3)]; rows = []
        for case, w in B.groupby("case"):
            h = w[np.abs(w.zi) < 0.5]
            if len(h) < 20:
                continue
            ht, hd = h[np.abs(h.zt) < 0.5], h[np.abs(h.zt) > 1.5]
            rows.append(dict(case=case, share=len(h) / len(w), **{f"{a}_fp": (h[f"L_{a}"] > 0).mean() for a in ARMS},
                             **{f"{a}_fpt": (ht[f"L_{a}"] > 0).mean() if len(ht) >= 10 else np.nan for a in ARMS},
                             **{f"{a}_fpd": (hd[f"L_{a}"] > 0).mean() if len(hd) >= 10 else np.nan for a in ARMS}))
        H = pd.DataFrame(rows)
        if not len(H):
            continue
        L.append(f"| {title(c)} | {len(H)} | {H.share.median():.3f} | {H.flat_fp.median():.3f} | {H.real_fp.median():.3f} | {pw(H.real_fp - H.flat_fp):.1g} | "
                 f"{H.flat_fpt.median():.3f} | {H.real_fpt.median():.3f} | {H.flat_fpd.median():.3f} | {H.real_fpd.median():.3f} |")
    (OUT / "bg_summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # figures: profiles; L heatmaps over (|z_int|, |z_tex|) for background voxels
    nc = 4; nr = int(np.ceil(len(cells) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.6 * nr), squeeze=False)
    for j, c in enumerate(cells):
        ax = axs[j // nc, j % nc]
        for a, col in (("flat", "C0"), ("real", "C3")):
            ax.plot(bins, prof[c][a].values, color=col, lw=1.4, label=a)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5, ls=":"); ax.set_title(title(c), fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.supxlabel("signed distance to the ground-truth target boundary (voxels; < 0 = background)", fontsize=9); fig.supylabel("target log-odds (median over cases)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "bg_profiles.png", dpi=170); plt.close(fig)
    eb = np.array([0, 0.5, 1, 1.5, 2, 3, 5, 100])
    fig, axs = plt.subplots(len(cells), 3, figsize=(10, 2.6 * len(cells)), squeeze=False)
    for j, c in enumerate(cells):
        D = data[c]; B = D[(D.kind == 0) & (D.s < -3)].copy(); B["bi"] = np.digitize(np.abs(B.zi), eb) - 1; B["bt"] = np.digitize(np.abs(B.zt), eb) - 1
        M = {a: B.groupby(["case", "bt", "bi"])[f"L_{a}"].median().groupby(["bt", "bi"]).median().unstack().reindex(index=range(len(eb) - 1), columns=range(len(eb) - 1)) for a in ARMS}
        for k, (a, Z, cm, lim) in enumerate((("flat", M["flat"], "viridis", None), ("real", M["real"], "viridis", None), ("real - flat", M["real"] - M["flat"], "RdBu_r", 3))):
            ax = axs[j, k]; vmin, vmax = (np.nanpercentile(np.c_[M["flat"].values, M["real"].values], [2, 98]) if lim is None else (-lim, lim))
            im = ax.imshow(Z.values, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, aspect="auto"); fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks(range(len(eb) - 1)); ax.set_xticklabels([f"{x:g}" for x in eb[:-1]], fontsize=6); ax.set_yticks(range(len(eb) - 1)); ax.set_yticklabels([f"{x:g}" for x in eb[:-1]], fontsize=6)
            ax.set_title(f"{title(c)}\n{a}: target log-odds of background voxels", fontsize=7)
            if k == 0:
                ax.set_ylabel("|z texture| vs target (bin start)", fontsize=7)
            ax.set_xlabel("|z intensity| vs target (bin start)", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "bg_rule.png", dpi=150); plt.close(fig)


def report_target():
    """Inside the target: which lesion voxels does each model call target? Features relative to the case's own BACKGROUND (s < -3 samples):
    zb_int = (local mean - background mean) / background SD (signed: > 0 brighter than background), zb_tex = log texture energy vs background.
    Outcome per voxel and model: hit = L > 0. 'rescued' = real hits and flat misses; 'lost' = flat hits and real misses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    L = ["# Inside the target: which lesion voxels each model calls target (L > 0), and what distinguishes the voxels only the real-fill model recovers", "",
         "Features vs the case's own background (s < -3): zb_int = (local mean - bg mean) / bg SD; zb_tex = (log texture energy - bg mean) / bg SD; depth = s.",
         "Per case: shares over its sampled target voxels; medians of features within each outcome group; then median over cases; p = Wilcoxon over cases.", "",
         "| cell | n | hit flat | hit real | rescued (real only) | lost (flat only) | p (hit real - flat) | zb_int both-hit / rescued | p | zb_tex both-hit / rescued | p | depth both-hit / rescued | p | zb_int both-miss | zb_tex both-miss |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    maps = {}; eb_i = np.array([-100, -1, 0, 1, 2, 3, 4, 6, 100]); eb_t = np.array([-100, -1, -0.5, 0, 0.5, 1, 1.5, 100])
    for c in CELLS:
        D = load(c)
        if D is None:
            continue
        rows = []
        bgs = D[(D.kind == 0) & (D.s < -3)].groupby("case").agg(bm=("lm", "mean"), bs=("lm", "std"), tm=("te", lambda x: np.log(x + 1e-6).mean()), ts=("te", lambda x: np.log(x + 1e-6).std()))
        T = D[(D.kind == 1) & (D.s >= 1)].merge(bgs, left_on="case", right_index=True).reset_index(drop=True)
        T["zbi"] = (T.lm - T.bm) / (T.bs + 1e-6); T["zbt"] = (np.log(T.te + 1e-6) - T.tm) / (T.ts + 1e-6)
        T["hf"] = T.L_flat > 0; T["hr"] = T.L_real > 0
        T["grp"] = np.select([T.hf & T.hr, ~T.hf & T.hr, T.hf & ~T.hr], ["both", "rescued", "lost"], "none")
        for case, w in T.groupby("case"):
            if len(w) < 50:
                continue
            r = dict(case=case, hf=w.hf.mean(), hr=w.hr.mean(), resc=(w.grp == "rescued").mean(), lost=(w.grp == "lost").mean())
            for g in ("both", "rescued", "none"):
                ww = w[w.grp == g]; ok = len(ww) >= 10
                r.update({f"{g}_zbi": ww.zbi.median() if ok else np.nan, f"{g}_zbt": ww.zbt.median() if ok else np.nan, f"{g}_s": ww.s.median() if ok else np.nan})
            rows.append(r)
        R = pd.DataFrame(rows)
        if not len(R):
            continue
        pr = lambda m: pw((R[f"rescued_{m}"] - R[f"both_{m}"]).dropna())   # noqa: E731
        L.append(f"| {title(c)} | {len(R)} | {R.hf.median():.2f} | {R.hr.median():.2f} | {R.resc.median():.3f} | {R.lost.median():.3f} | {pw(R.hr - R.hf):.1g} | "
                 f"{R.both_zbi.median():+.2f} / {R.rescued_zbi.median():+.2f} | {pr('zbi'):.1g} | {R.both_zbt.median():+.2f} / {R.rescued_zbt.median():+.2f} | {pr('zbt'):.1g} | "
                 f"{R.both_s.median():.1f} / {R.rescued_s.median():.1f} | {pr('s'):.1g} | {R.none_zbi.median():+.2f} | {R.none_zbt.median():+.2f} |")
        T["bi"] = np.digitize(T.zbi, eb_i) - 1; T["bt"] = np.digitize(T.zbt, eb_t) - 1
        def rate(col):
            g = T.groupby(["case", "bt", "bi"])[col].agg(["mean", "size"]); g = g[g["size"] >= 10]["mean"]
            return g.groupby(["bt", "bi"]).mean().unstack().reindex(index=range(len(eb_t) - 1), columns=range(len(eb_i) - 1))
        cnt = T.groupby(["bt", "bi"]).size().unstack().reindex(index=range(len(eb_t) - 1), columns=range(len(eb_i) - 1)).fillna(0)
        maps[c] = (rate("hf"), rate("hr"), cnt / cnt.values.sum())
    (OUT / "bg_target_summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    cells = list(maps); xl = ["<-1", "-1..0", "0..1", "1..2", "2..3", "3..4", "4..6", ">6"]; yl = ["<-1", "-1..-.5", "-.5..0", "0..0.5", "0.5..1", "1..1.5", ">1.5"]
    fig, axs = plt.subplots(len(cells), 4, figsize=(15, 2.7 * len(cells)), squeeze=False)
    for j, c in enumerate(cells):
        F, Rr, N = maps[c]
        for k, (Z, nm, cm, vr) in enumerate(((N, "share of target voxels", "Greys", (0, None)), (F, "flat-fill: called target", "viridis", (0, 1)), (Rr, "real-fill: called target", "viridis", (0, 1)),
                                             (Rr - F, "real - flat", "RdBu_r", (-0.6, 0.6)))):
            ax = axs[j, k]; im = ax.imshow(Z.values, origin="lower", cmap=cm, vmin=vr[0], vmax=vr[1], aspect="auto"); fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, fontsize=5.5, rotation=45); ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl, fontsize=5.5)
            ax.set_title(f"{title(c)}\n{nm}", fontsize=7)
            if k == 0:
                ax.set_ylabel("texture energy vs background (z, log)", fontsize=7)
            ax.set_xlabel("local intensity vs background (z)", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "bg_target_maps.png", dpi=140); plt.close(fig)


def report_interior():
    """Depth-controlled: target voxels at least D_MIN voxels inside (texture window no longer reaches the boundary). Per case: OLS of each model's
    target log-odds L on z(local intensity vs background), z(log texture energy vs background), log depth (standardised within case); and the hit
    rate (L > 0) per within-case texture tertile. Texture use predicts: flat-fill calls the lesion less where its interior is more textured,
    real-fill does not (b_tex flat < b_tex real; hit-rate gap grows with texture)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    L = ["# Lesion interior at fixed depth: does each model's target call depend on the local texture? (target voxels, depth >= D)", "",
         "Per case OLS: L ~ z_int + z_tex + log depth (standardised within case; L in log-odds per SD); hit = L > 0 by within-case texture tertile (low / mid / high).", ""]
    curves = {}
    for dmin in (3, 5):
        L += [f"## depth >= {dmin}", "", "| cell | n | b texture flat | real | p | b intensity flat | real | p | hit flat low/mid/high tex | hit real low/mid/high | gap (real-flat) high - low tex | p |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in CELLS:
            D = load(c)
            if D is None:
                continue
            bgs = D[(D.kind == 0) & (D.s < -3)].groupby("case").agg(bm=("lm", "mean"), bs=("lm", "std"), tm=("te", lambda x: np.log(x + 1e-6).mean()), ts=("te", lambda x: np.log(x + 1e-6).std()))
            T = D[(D.kind == 1) & (D.s >= dmin)].merge(bgs, left_on="case", right_index=True).reset_index(drop=True)
            T["zbi"] = (T.lm - T.bm) / (T.bs + 1e-6); T["zbt"] = (np.log(T.te + 1e-6) - T.tm) / (T.ts + 1e-6)
            rows = []
            for case, w in T.groupby("case"):
                if len(w) < 60:
                    continue
                X = np.c_[w.zbi, w.zbt, np.log(w.s)]; X = (X - X.mean(0)) / (X.std(0) + 1e-9); X = np.c_[np.ones(len(w)), X]
                r = dict(case=case); q = pd.qcut(w.zbt.rank(method="first"), 3, labels=False)
                for a in ARMS:
                    b = np.linalg.lstsq(X, w[f"L_{a}"].values.astype(float), rcond=None)[0]; r[f"{a}_bi"], r[f"{a}_bt"] = b[1], b[2]
                    for t in range(3):
                        r[f"{a}_h{t}"] = float((w[f"L_{a}"][q == t] > 0).mean())
                rows.append(r)
            R = pd.DataFrame(rows)
            if not len(R):
                continue
            R["gap"] = (R.real_h2 - R.flat_h2) - (R.real_h0 - R.flat_h0); curves[(c, dmin)] = R
            h = lambda a: "/".join(f"{R[f'{a}_h{t}'].median():.2f}" for t in range(3))   # noqa: E731
            L.append(f"| {title(c)} | {len(R)} | {R.flat_bt.median():+.2f} | {R.real_bt.median():+.2f} | {pw(R.real_bt - R.flat_bt):.1g} | {R.flat_bi.median():+.2f} | {R.real_bi.median():+.2f} | "
                     f"{pw(R.real_bi - R.flat_bi):.1g} | {h('flat')} | {h('real')} | {R.gap.median():+.3f} | {pw(R.gap):.1g} |")
        L.append("")
    (OUT / "bg_interior_summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    cells = [c for c in CELLS if (c, 3) in curves]; nc = 4; nr = int(np.ceil(len(cells) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.6 * nr), squeeze=False)
    for j, c in enumerate(cells):
        ax = axs[j // nc, j % nc]; R = curves[(c, 3)]
        for a, col in (("flat", "C0"), ("real", "C3")):
            m = [R[f"{a}_h{t}"].mean() for t in range(3)]; se = [R[f"{a}_h{t}"].std() / np.sqrt(len(R)) for t in range(3)]
            ax.errorbar([0, 1, 2], m, yerr=se, color=col, marker="o", ms=4, lw=1.4, capsize=2, label=a)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["smooth", "mid", "textured"], fontsize=7); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
        ax.set_title(f"{title(c)}  (n {len(R)}, gap p {pw(R.gap):.1g})", fontsize=7.5); ax.legend(fontsize=6)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.supxlabel("lesion-interior voxels (>= 3 voxels inside), within-case tertile of local texture energy", fontsize=9)
    fig.supylabel("share called target (mean over cases +- SE)", fontsize=9); fig.tight_layout(); fig.savefig(OUT / "bg_interior_texture.png", dpi=170); plt.close(fig)


GL_CLS = ["background", "NCR", "SNFH (edema)", "ET", "RC"]


def report_classes():
    """Glioma only (needs the all-class logits): what each model calls the ground-truth EDEMA voxels (argmax class) -- is a flat-fill miss
    'called background' (under-segmentation) or 'called tumour core' (class confusion)? Per case shares over its sampled edema voxels
    (depth >= 1), median over cases; plus the same restricted to the voxels only one model calls edema."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    L = ["# Glioma: what each model calls the ground-truth edema voxels (argmax of the fold-mean logits; shares per case, median over cases)", "",
         "| cell | n | model | " + " | ".join(GL_CLS) + " | tumour vs background log-odds < 0 |", "|---|---|---|" + "---|" * (len(GL_CLS) + 1)]
    L2 = ["", "## Voxels only one model calls edema: what the other model calls them (pooled over cases)", "",
          "| cell | rescued voxels (real edema, flat not) | flat calls them: " + " / ".join(GL_CLS) + " | lost voxels (flat edema, real not) | real calls them: " + " / ".join(GL_CLS) + " |",
          "|---|---|---|---|---|"]
    bars = {}
    for c in [k for k in CELLS if k.startswith("glioma")]:
        rows = []; resc = {a: [] for a in ARMS}; nres = {"flat": 0, "real": 0}
        for f in sorted((OUT / c).glob("*.npz")):
            z = np.load(f)
            if "lo_flat" not in z:
                continue
            m = (z["lab"] == 2) & (z["s"].astype(np.float32) >= 1)
            if m.sum() < 50:
                continue
            am = {a: z[f"lo_{a}"][:, m].astype(np.float32).argmax(0) for a in ARMS}
            r = dict(case=f.stem)
            for a in ARMS:
                lo = z[f"lo_{a}"][:, m].astype(np.float32); lt = lse(lo[1:]) - lo[0]
                for k in range(len(GL_CLS)):
                    r[f"{a}_{k}"] = float((am[a] == k).mean())
                r[f"{a}_bgL"] = float((lt < 0).mean())
            rows.append(r)
            only_r = (am["real"] == 2) & (am["flat"] != 2); only_f = (am["flat"] == 2) & (am["real"] != 2)
            resc["flat"].append(np.bincount(am["flat"][only_r], minlength=len(GL_CLS))); nres["flat"] += int(only_r.sum())
            resc["real"].append(np.bincount(am["real"][only_f], minlength=len(GL_CLS))); nres["real"] += int(only_f.sum())
        R = pd.DataFrame(rows)
        if not len(R):
            continue
        for a in ARMS:
            L.append(f"| {title(c)} | {len(R)} | {a} | " + " | ".join(f"{R[f'{a}_{k}'].median():.3f}" for k in range(len(GL_CLS))) + f" | {R[f'{a}_bgL'].median():.3f} |")
        L.append(f"| | | p real-flat | " + " | ".join(f"{pw(R[f'real_{k}'] - R[f'flat_{k}']):.1g}" for k in range(len(GL_CLS))) + f" | {pw(R.real_bgL - R.flat_bgL):.1g} |")
        sf = np.sum(resc["flat"], 0); sr = np.sum(resc["real"], 0)
        L2.append(f"| {title(c)} | {nres['flat']} | " + " / ".join(f"{x / max(1, sf.sum()):.2f}" for x in sf) + f" | {nres['real']} | " + " / ".join(f"{x / max(1, sr.sum()):.2f}" for x in sr) + " |")
        bars[c] = R
    (OUT / "bg_glioma_classes.md").write_text("\n".join(L + L2) + "\n"); print("\n".join(L + L2))
    if not bars:
        return
    fig, axs = plt.subplots(1, len(bars), figsize=(3.6 * len(bars), 3.2), squeeze=False); cols = ["0.6", "C4", "C2", "C1", "C5"]
    for j, (c, R) in enumerate(bars.items()):
        ax = axs[0, j]
        for i, a in enumerate(ARMS):
            left = 0
            for k in range(len(GL_CLS)):
                w = R[f"{a}_{k}"].mean(); ax.bar(i, w, 0.6, bottom=left, color=cols[k], label=GL_CLS[k] if (i == 0 and j == 0) else None); left += w
        ax.set_xticks([0, 1]); ax.set_xticklabels(["flat-fill", "real-fill"], fontsize=8); ax.set_ylim(0, 1); ax.set_title(title(c), fontsize=8); ax.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("ground-truth edema voxels: predicted class (mean over cases)", fontsize=7); fig.legend(fontsize=7, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); fig.savefig(OUT / "bg_glioma_classes.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def report_drivers():
    """Pre-registered (written before the run with het/lc existed; scales fixed from the fill code): which target voxels does each model call target?
    Per case, target voxels (depth >= 1 and >= 3); background reference = its sampled background voxels with s < -3; polarity = sign of (median
    target local mean - background mean), so every contrast feature is oriented 'more lesion-like = larger':
      zc = oriented (local mean - background mean) / background SD           (contrast against the whole background)
      zl = oriented (sigma-2 mean - sigma-6 mean) / background SD            (contrast against the immediate surroundings)
      zh = (log heterogeneity - background mean) / background SD             (patch-scale heterogeneity, the property rung 4.5 removes)
    OLS per case: L ~ zc + zl + zh + log depth (standardised within case). Prediction (texture/heterogeneity story): flat-fill calls the target
    less where zh is high and needs higher zc/zl; real-fill does not / needs less. Hit-rate curves (L > 0) against fixed zc bins and zh tertiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    eb = np.array([-100, 0, 0.5, 1, 1.5, 2, 3, 100]); xl = ["<0", "0-.5", ".5-1", "1-1.5", "1.5-2", "2-3", ">3"]
    L = ["# What decides whether a target voxel is called target: contrast vs patch-scale heterogeneity (pre-registered; see report_drivers docstring)", ""]
    curves = {}
    for dmin in (1, 3):
        L += [f"## target voxels at depth >= {dmin}", "",
              "| cell | n | b zc flat | real | p | b zl flat | real | p | b zh flat | real | p | hit flat by zh tertile | real | rescued vs both-hit: zc | zl | zh (median of per-case differences; p) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in CELLS:
            D = load(c)
            if D is None or "het" not in D:
                continue
            B = D[(D.kind == 0) & (D.s < -3)]
            bgs = B.groupby("case").agg(bm=("lm", "mean"), bs=("lm", "std"), hm=("het", lambda x: np.log(x + 1e-6).mean()), hs=("het", lambda x: np.log(x + 1e-6).std()))
            T = D[(D.kind == 1) & (D.s >= 1)].merge(bgs, left_on="case", right_index=True).reset_index(drop=True)
            pol = T.groupby("case").apply(lambda w: np.sign(w.lm.median() - w.bm.iloc[0]) or 1.0).rename("pol")
            T = T.merge(pol, left_on="case", right_index=True).reset_index(drop=True)
            T["zc"] = T.pol * (T.lm - T.bm) / (T.bs + 1e-6); T["zl"] = T.pol * T.lc / (T.bs + 1e-6); T["zh"] = (np.log(T.het + 1e-6) - T.hm) / (T.hs + 1e-6)
            T = T[T.s >= dmin]; rows = []
            for case, w in T.groupby("case"):
                if len(w) < 60:
                    continue
                X = np.c_[w.zc, w.zl, w.zh, np.log(w.s)]; X = (X - X.mean(0)) / (X.std(0) + 1e-9); X = np.c_[np.ones(len(w)), X]
                r = dict(case=case); q = pd.qcut(w.zh.rank(method="first"), 3, labels=False); bi = np.digitize(w.zc, eb) - 1
                for a in ARMS:
                    b = np.linalg.lstsq(X, w[f"L_{a}"].values.astype(float), rcond=None)[0]; r.update({f"{a}_bc": b[1], f"{a}_bl": b[2], f"{a}_bh": b[3]})
                    h = w[f"L_{a}"] > 0
                    for t in range(3):
                        r[f"{a}_h{t}"] = float(h[q == t].mean())
                    for k in range(len(eb) - 1):
                        r[f"{a}_c{k}"] = float(h[bi == k].mean()) if (bi == k).sum() >= 10 else np.nan
                hf, hr = w.L_flat > 0, w.L_real > 0; rs, bh = w[hr & ~hf], w[hr & hf]
                for m in ("zc", "zl", "zh"):
                    r[f"d_{m}"] = rs[m].median() - bh[m].median() if (len(rs) >= 10 and len(bh) >= 10) else np.nan
                rows.append(r)
            R = pd.DataFrame(rows)
            if not len(R):
                continue
            curves[(c, dmin)] = R
            f = lambda m: f"{R[f'flat_{m}'].median():+.2f} | {R[f'real_{m}'].median():+.2f} | {pw(R[f'real_{m}'] - R[f'flat_{m}']):.1g}"   # noqa: E731
            h = lambda a: "/".join(f"{R[f'{a}_h{t}'].median():.2f}" for t in range(3))   # noqa: E731
            d = lambda m: f"{R[f'd_{m}'].median():+.2f} (p {pw(R[f'd_{m}'].dropna()):.1g})"   # noqa: E731
            L.append(f"| {title(c)} | {len(R)} | {f('bc')} | {f('bl')} | {f('bh')} | {h('flat')} | {h('real')} | {d('zc')} | {d('zl')} | {d('zh')} |")
        L.append("")
    (OUT / "bg_drivers.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    cells = [c for c in CELLS if (c, 3) in curves]; nc = 4; nr = int(np.ceil(len(cells) / nc))
    for kind, cols, xt, xlab, fn in (("c", len(eb) - 1, xl, "contrast of the target voxel against the background (oriented z)", "bg_drivers_contrast.png"),
                                     ("h", 3, ["homogeneous", "mid", "heterogeneous"], "patch-scale heterogeneity, within-case tertile", "bg_drivers_heterogeneity.png")):
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.7 * nr), squeeze=False)
        for j, c in enumerate(cells):
            ax = axs[j // nc, j % nc]; R = curves[(c, 3)]
            for a, col in (("flat", "C0"), ("real", "C3")):
                m = np.array([np.nanmean(R[f"{a}_{kind}{k}"]) for k in range(cols)]); n = np.array([R[f"{a}_{kind}{k}"].notna().sum() for k in range(cols)])
                se = np.array([np.nanstd(R[f"{a}_{kind}{k}"]) for k in range(cols)]) / np.sqrt(np.maximum(n, 1)); m[n < 5] = np.nan
                ax.errorbar(range(cols), m, yerr=se, color=col, marker="o", ms=3.5, lw=1.3, capsize=2, label=f"{a}-fill")
            ax.set_xticks(range(cols)); ax.set_xticklabels(xt, fontsize=6, rotation=30); ax.set_ylim(0, 1.02); ax.tick_params(labelsize=7)
            ax.set_title(f"{title(c)} (n {len(R)})", fontsize=7.5); ax.legend(fontsize=6)
        for j in range(len(cells), nr * nc):
            axs[j // nc, j % nc].axis("off")
        fig.supxlabel(f"target voxels >= 3 voxels inside the boundary: {xlab}", fontsize=9); fig.supylabel("share called target (mean over cases +- SE)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / fn, dpi=170); plt.close(fig)


def run_examples(key, dev, cases):
    """Qualitative panels: for the given cases, the in-patch axial slice with the most ground-truth target voxels: image, labels, each model's
    target log-odds and argmax class (fold mean). -> OUT/examples/<cell>/<case>.npz"""
    import torch
    cell = vt.CELLS[key]; od = OUT / "examples" / key; od.mkdir(parents=True, exist_ok=True); T = cell["target"]
    prs = {a: v2.load_predictor(f(), dev) for a, f in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    for case in cases:
        data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        box = e7.patch_at(tg, ps, img.shape); ip, gp = img[box].astype(np.float32), gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, gp = np.pad(ip, padw), np.pad(gp, padw)
        z = int(np.argmax(np.isin(gp, T).sum((1, 2)))); out = dict(img=ip[z].astype(np.float16), gt=gp[z].astype(np.int16), z=z)
        for a, pr in prs.items():
            net = pr.network; acc = 0
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    acc = acc + e7.fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()[0]
            lo = (acc / len(v2.FOLDS)).cpu().numpy()[:, z]; oth = [j for j in range(lo.shape[0]) if j not in T]
            out[f"L_{a}"] = (lse(lo[T]) - lse(lo[oth])).astype(np.float16); out[f"am_{a}"] = lo.argmax(0).astype(np.int16)
        np.savez_compressed(od / f"{case}.npz", **out); vt.log(f"[examples {key}] {case}: slice {z}")


def pick_example_cases(key, q=(0.25, 0.5, 0.75)):
    """Fixed rule (no hand-picking): cases at the 25th / 50th / 75th percentile of the per-case share of target voxels only real-fill calls."""
    D = load(key); T = D[(D.kind == 1) & (D.s >= 1)]
    sh = T.groupby("case").apply(lambda w: ((w.L_real > 0) & (w.L_flat <= 0)).mean()).sort_values()
    return [sh.index[int(round(x * (len(sh) - 1)))] for x in q], sh


def report_examples(cells=("glioma_t1n_t2f", "glioma_t2w_t1n")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rows = []
    for c in cells:
        cs, sh = pick_example_cases(c)
        rows += [(c, k, sh[k]) for k in cs if (OUT / "examples" / c / f"{k}.npz").exists()]
    if not rows:
        return
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    fig, axs = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)), squeeze=False)
    for i, (c, k, s_) in enumerate(rows):
        z = np.load(OUT / "examples" / c / f"{k}.npz"); img = z["img"].astype(np.float32); T = vt.CELLS[c]["target"]; g = np.isin(z["gt"], T)
        lo, hi = np.percentile(img[img != 0], [1, 99]) if (img != 0).any() else (img.min(), img.max())
        # argmax class per model (glioma: background / NCR / edema / ET / RC); target = the cell's target classes
        amf, amr = z["am_flat"].astype(int), z["am_real"].astype(int); hf, hr = np.isin(amf, T), np.isin(amr, T); gl = z["gt"].astype(int)
        ccol = ListedColormap(["#000000", "#9b30ff", "#ffd400", "#ff3030", "#00e5ff"]); ncls = 5
        for j, (nm, am) in enumerate((("image + ground truth (target contour; core labels filled)", None), ("flat-fill (rung 4.5): predicted class", amf),
                                       ("real-fill (rung 5): predicted class", amr), ("difference in target calls", None))):
            ax = axs[i, j]; ax.imshow(img, cmap="gray", vmin=lo, vmax=hi); ax.set_xticks([]); ax.set_yticks([])
            if j == 0 and gl.max() >= 1:
                core = np.where(np.isin(gl, T) | (gl == 0), 0, gl)
                ax.imshow(np.ma.masked_where(core == 0, core), cmap=ccol, vmin=0, vmax=ncls - 1, alpha=0.5)
            if am is not None:
                ax.imshow(np.ma.masked_where(am == 0, am), cmap=ccol, vmin=0, vmax=ncls - 1, alpha=0.45)
            if j == 3:
                ov = np.zeros(img.shape, int); ov[hr & ~hf] = 1; ov[hf & ~hr] = 2
                ax.imshow(np.ma.masked_where(ov == 0, ov), cmap=ListedColormap(["#00e05a", "#ff2bd6"]), vmin=1, vmax=2, alpha=0.8)
            ax.contour(g, [0.5], colors="#00b4ff", linewidths=0.9)
            ax.set_title((f"{title(c)}  {k}\n" if j == 0 else "") + nm + (f"  (rescued share {s_:.2f})" if j == 0 else ""), fontsize=7.5)
    fig.suptitle("Blue contour = ground-truth target (edema). Predicted class: yellow = edema, purple = NCR, red = ET, cyan = RC, none = background. "
                 "Difference: green = only real-fill calls edema, magenta = only flat-fill.\nCases at the 25th/50th/75th percentile of the per-case rescued share "
                 "(fixed rule, no hand-picking).", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(OUT / "bg_examples.png", dpi=150); plt.close(fig)


def report_cases():
    """Lesion-level (between cases; written before looking): does a case's whole-target heterogeneity predict how much more of the target the
    real-fill model calls? Per case: gain = share of target voxels (depth >= 1) called target by real - by flat, and mean L difference;
    lesion features: median zh (patch-scale heterogeneity vs background), within-target spread of the local mean (SD / background SD),
    median oriented contrast zc, log target size (sampled-count-free proxy: max depth). Spearman per cell and OLS gain ~ z(features)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    L = ["# Between cases: does the target's own heterogeneity predict the real-fill gain? (per case; Spearman over cases; OLS on z-scored features)", "",
         "| cell | n | median gain (hit real - flat) | rho(gain, heterogeneity) | p | rho(gain, intensity spread) | p | rho(gain, contrast) | p | rho(gain, size) | p | "
         "OLS b heterogeneity | b spread | b contrast | b size (gain in pp per SD) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keep = {}
    for c in CELLS:
        D = load(c)
        if D is None or "het" not in D:
            continue
        B = D[(D.kind == 0) & (D.s < -3)]
        bgs = B.groupby("case").agg(bm=("lm", "mean"), bs=("lm", "std"), hm=("het", lambda x: np.log(x + 1e-6).mean()), hs=("het", lambda x: np.log(x + 1e-6).std()))
        rows = []
        for case, w in D[(D.kind == 1) & (D.s >= 1)].groupby("case"):
            if len(w) < 60 or case not in bgs.index:
                continue
            b = bgs.loc[case]; pol = np.sign(w.lm.median() - b.bm) or 1.0
            rows.append(dict(case=case, gain=(w.L_real > 0).mean() - (w.L_flat > 0).mean(), dL=(w.L_real - w.L_flat).mean(),
                             het=((np.log(w.het + 1e-6) - b.hm) / (b.hs + 1e-6)).median(), spread=w.lm.std() / (b.bs + 1e-6),
                             con=(pol * (w.lm - b.bm) / (b.bs + 1e-6)).median(), size=np.log(w.s.max())))
        R = pd.DataFrame(rows)
        if len(R) < 8:
            continue
        keep[c] = R; sp = {m: spearmanr(R.gain, R[m]) for m in ("het", "spread", "con", "size")}
        X = R[["het", "spread", "con", "size"]].values; X = (X - X.mean(0)) / (X.std(0) + 1e-9); bb = np.linalg.lstsq(np.c_[np.ones(len(R)), X], R.gain.values * 100, rcond=None)[0]
        L.append(f"| {title(c)} | {len(R)} | {R.gain.median() * 100:+.1f} pp | " + " | ".join(f"{sp[m].correlation:+.2f} | {sp[m].pvalue:.1g}" for m in ("het", "spread", "con", "size"))
                 + f" | {bb[1]:+.1f} | {bb[2]:+.1f} | {bb[3]:+.1f} | {bb[4]:+.1f} |")
    (OUT / "bg_cases.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    cells = list(keep); nc = 4; nr = int(np.ceil(len(cells) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.8 * nr), squeeze=False)
    for j, c in enumerate(cells):
        ax = axs[j // nc, j % nc]; R = keep[c]; r = spearmanr(R.het, R.gain)
        ax.scatter(R.het, R.gain * 100, s=12, c=R.con, cmap="viridis"); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{title(c)}\nrho {r.correlation:+.2f} (p {r.pvalue:.1g}, n {len(R)})", fontsize=7.5); ax.tick_params(labelsize=7)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.supxlabel("target heterogeneity at the patch scale (median z vs the case's background); colour = target contrast", fontsize=9)
    fig.supylabel("real-fill gain: share of target called, real - flat (pp)", fontsize=9); fig.tight_layout(); fig.savefig(OUT / "bg_cases.png", dpi=170); plt.close(fig)


def report_context(cells=("glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w")):
    """Glioma, written before looking: does each model's edema call depend on the distance to the tumour core (NCR/ET/RC labels)?
    Edema voxels >= 3 inside; hit rate (L > 0) per distance-to-core bin, overall and within the case's contrast tertile; per case OLS
    L ~ zc + log(1 + dcore) + log depth. Prediction if flat-fill relies on that context: its hit rate falls with dcore, real-fill's much less."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    eb = np.array([0, 3, 6, 10, 15, 25, 1000]); xl = ["<3", "3-6", "6-10", "10-15", "15-25", ">25"]
    L = ["# Glioma: edema call vs distance to the tumour core (edema voxels >= 3 inside; distance in voxels to the nearest NCR/ET/RC voxel)", "",
         "| cell | n | b log(1+dcore) flat | real | p | b contrast flat | real | p | hit flat by dcore bin " + "/".join(xl) + " | real | gap far - near (real-flat, pp) | p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for c in cells:
        D = load(c)
        if D is None or "dcore" not in D:
            continue
        B = D[(D.kind == 0) & (D.s < -3)]; bgs = B.groupby("case").agg(bm=("lm", "mean"), bs=("lm", "std"))
        T = D[(D.kind == 1) & (D.s >= 1)].merge(bgs, left_on="case", right_index=True).reset_index(drop=True)
        pol = T.groupby("case").apply(lambda w: np.sign(w.lm.median() - w.bm.iloc[0]) or 1.0).rename("pol")
        T = T.merge(pol, left_on="case", right_index=True).reset_index(drop=True); T["zc"] = T.pol * (T.lm - T.bm) / (T.bs + 1e-6)
        T = T[(T.s >= 3) & (T.dcore < 900)]; rows = []
        for case, w in T.groupby("case"):
            if len(w) < 60:
                continue
            X = np.c_[w.zc, np.log1p(w.dcore), np.log(w.s)]; X = (X - X.mean(0)) / (X.std(0) + 1e-9); X = np.c_[np.ones(len(w)), X]; bi = np.digitize(w.dcore, eb) - 1
            r = dict(case=case)
            for a in ARMS:
                b = np.linalg.lstsq(X, w[f"L_{a}"].values.astype(float), rcond=None)[0]; r[f"{a}_bc"], r[f"{a}_bd"] = b[1], b[2]; h = w[f"L_{a}"] > 0
                for k in range(len(eb) - 1):
                    r[f"{a}_d{k}"] = float(h[bi == k].mean()) if (bi == k).sum() >= 10 else np.nan
            near = [r[f"real_d{k}"] - r[f"flat_d{k}"] for k in (0, 1)]; far = [r[f"real_d{k}"] - r[f"flat_d{k}"] for k in (3, 4, 5)]
            r["gap"] = np.nanmean(far) - np.nanmean(near) if np.isfinite(near).any() and np.isfinite(far).any() else np.nan
            rows.append(r)
        R = pd.DataFrame(rows)
        if not len(R):
            continue
        res[c] = R; f = lambda m: f"{R[f'flat_{m}'].median():+.2f} | {R[f'real_{m}'].median():+.2f} | {pw(R[f'real_{m}'] - R[f'flat_{m}']):.1g}"   # noqa: E731
        h = lambda a: "/".join(f"{np.nanmean(R[f'{a}_d{k}']):.2f}" for k in range(len(eb) - 1))   # noqa: E731
        L.append(f"| {title(c)} | {len(R)} | {f('bd')} | {f('bc')} | {h('flat')} | {h('real')} | {np.nanmedian(R.gap) * 100:+.1f} | {pw(R.gap.dropna()):.1g} |")
    (OUT / "bg_context.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    if not res:
        return
    fig, axs = plt.subplots(1, len(res), figsize=(3.6 * len(res), 3.0), squeeze=False)
    for j, (c, R) in enumerate(res.items()):
        ax = axs[0, j]
        for a, col in (("flat", "C0"), ("real", "C3")):
            m = np.array([np.nanmean(R[f"{a}_d{k}"]) for k in range(len(eb) - 1)]); n = np.array([R[f"{a}_d{k}"].notna().sum() for k in range(len(eb) - 1)])
            se = np.array([np.nanstd(R[f"{a}_d{k}"]) for k in range(len(eb) - 1)]) / np.sqrt(np.maximum(n, 1)); m[n < 5] = np.nan
            ax.errorbar(range(len(xl)), m, yerr=se, color=col, marker="o", ms=3.5, lw=1.3, capsize=2, label=f"{a}-fill")
        ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, fontsize=7); ax.set_ylim(0, 1.02); ax.set_title(f"{title(c)} (n {len(R)})", fontsize=8); ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    fig.supxlabel("edema voxels >= 3 inside: distance to the nearest tumour-core voxel (NCR/ET/RC, voxels)", fontsize=8.5)
    fig.supylabel("share called edema", fontsize=8.5); fig.tight_layout(); fig.savefig(OUT / "bg_context.png", dpi=170); plt.close(fig)


def report_cond(cells=("glioma_t1n_t2f", "glioma_t2w_t2f", "glioma_t1n_t2w", "glioma_t2w_t1n"), conds=("orig", "inv", "flat", "invflat")):
    """Crossed counterfactual (FINDINGS section 30, pre-registered): per case edema hit (L > 0, depth >= 1), background FP (s < -3), patch Dice,
    flat vs real, for the original test image and its inverted / flattened / inverted+flattened versions. Interaction = change of flat minus change
    of real vs the original (Wilcoxon over cases)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    L = ["# Crossed counterfactual: inverted / flattened test images (pre-registered predictions in FINDINGS section 30)", "",
         "Per case; mean over cases (median in brackets for hit). d = change vs original; interaction = d flat - d real, Wilcoxon over cases.", "",
         "| cell | condition | n | edema hit flat | real | Dice flat | real | bg FP flat | real | d hit flat | p | d hit real | p | interaction | p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for c in cells:
        per = {}
        for k in conds:
            base = OUT if k == "orig" else vt.OUT / "bg_cond" / k; rows = []
            for f in sorted((base / c).glob("*.npz")):
                z = np.load(f); m = (z["kind"] == 1) & (z["s"].astype(np.float32) >= 1)
                if m.sum() < 50:
                    continue
                rows.append(dict(case=f.stem, **{f"hit_{a}": float((z[f"L_{a}"][m].astype(np.float32) > 0).mean()) for a in ARMS},
                                 **{f"fp_{a}": float(z[f"fp_{a}"]) for a in ARMS}, **{f"dice_{a}": float(z[f"dice_{a}"]) for a in ARMS}))
            if rows:
                per[k] = pd.DataFrame(rows).set_index("case")
        if "orig" not in per:
            continue
        res[c] = per; o = per["orig"]
        for k, P in per.items():
            cs = P.index.intersection(o.index); P = P.loc[cs]; oo = o.loc[cs]
            df_, dr_ = P.hit_flat - oo.hit_flat, P.hit_real - oo.hit_real
            L.append(f"| {title(c)} | {k} | {len(cs)} | {P.hit_flat.mean():.3f} ({P.hit_flat.median():.2f}) | {P.hit_real.mean():.3f} ({P.hit_real.median():.2f}) | "
                     f"{P.dice_flat.mean():.3f} | {P.dice_real.mean():.3f} | {P.fp_flat.mean():.4f} | {P.fp_real.mean():.4f} | "
                     + ("| | | | | |" if k == "orig" else f"{df_.mean():+.3f} | {pw(df_):.1g} | {dr_.mean():+.3f} | {pw(dr_):.1g} | {(df_ - dr_).mean():+.3f} | {pw(df_ - dr_):.1g} |"))
    (OUT / "bg_cond.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    if not res:
        return
    fig, axs = plt.subplots(2, len(res), figsize=(3.6 * len(res), 5.6), squeeze=False)
    for j, (c, per) in enumerate(res.items()):
        ks = [k for k in conds if k in per]
        for r, (met, lab) in enumerate((("hit", "edema voxels called edema"), ("dice", "patch Dice (edema)"))):
            ax = axs[r, j]
            for a, col, off in (("flat", "C0", -0.12), ("real", "C3", 0.12)):
                m = [per[k][f"{met}_{a}"].mean() for k in ks]; se = [per[k][f"{met}_{a}"].std() / np.sqrt(len(per[k])) for k in ks]
                ax.errorbar(np.arange(len(ks)) + off, m, yerr=se, fmt="o-", color=col, ms=4, lw=1.2, capsize=2, label=f"{a}-fill")
            ax.set_xticks(range(len(ks))); ax.set_xticklabels(["original", "inverted", "flattened", "inv + flat"][:len(ks)], fontsize=7); ax.set_ylim(0, 1)
            ax.set_ylabel(lab, fontsize=7); ax.tick_params(labelsize=7); ax.legend(fontsize=6)
            if r == 0:
                ax.set_title(title(c), fontsize=8)
    fig.suptitle("Same test cases, test image inverted (-x) and/or flattened (4-level k-means, texture removed) after preprocessing; mean over cases +- SE", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / "bg_cond.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELLS)); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true"); ap.add_argument("--target", action="store_true"); ap.add_argument("--classes", action="store_true"); ap.add_argument("--drivers", action="store_true"); ap.add_argument("--examples", default=None); ap.add_argument("--examples-report", action="store_true"); ap.add_argument("--cases", action="store_true"); ap.add_argument("--context", action="store_true"); ap.add_argument("--cond", default="orig"); ap.add_argument("--cond-report", action="store_true")
    a = ap.parse_args()
    if a.cond_report:
        report_cond(); return
    if a.context:
        report_context(); return
    if a.cases:
        report_cases(); return
    if a.examples_report:
        report_examples(); return
    if a.examples:
        import torch
        run_examples(a.cells, torch.device("cuda"), pick_example_cases(a.cells)[0] if a.examples == "auto" else a.examples.split(",")); return
    if a.drivers:
        report_drivers(); return
    if a.classes:
        report_classes(); return
    if a.target:
        report_target(); report_interior(); return
    if a.report:
        report(); return
    import torch
    for c in a.cells.split(","):
        run_bg(c, torch.device("cuda"), a.case_slice, a.cond)


if __name__ == "__main__":
    main()
