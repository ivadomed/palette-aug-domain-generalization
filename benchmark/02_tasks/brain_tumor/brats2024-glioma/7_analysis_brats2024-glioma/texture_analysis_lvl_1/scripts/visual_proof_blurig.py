#!/usr/bin/env python
"""
Visual proof v4 (2026-10-11): BLUR INTEGRATED GRADIENTS (Xu, Venugopalan, Sundararajan, "Attribution in Scale and Space", CVPR 2020) as a
direct "where is texture used" map, flat-fill (rung 4.5) vs real-fill (rung 5), same checkpoints / 70 BraTS test cases / 4 cells as v2-v3.

Per case, on one plan-sized patch centred on the GT edema (GT used for patch placement, the target region and the zones; never to edit the image):
  path      x(s) = G_s * x inside the body (s = Gaussian sigma in voxels, 1 mm iso), s from S_MAX down to 0 in K steps; x(S_MAX) = the image with
            its texture removed, x(0) = the real image. Same baseline for both arms.
  target    f(x) = mean over GT edema voxels of (logit_edema - logit_background)  [identical region for both arms]
  map       A = sum_k grad f(x_mid_k) * (x(s_{k+1}) - x(s_k))   (midpoint Riemann), so sum(A) ~= f(x) - f(x(S_MAX)) = TEXTURE CONTRIBUTION
            to the edema evidence (completeness error reported). A > 0: texture at that voxel raises the edema evidence; A < 0: lowers it.
  per arm, per fold, fold-mean reported. Zones from the GT: interior (d >= 3 vox inside), border (|d| <= 2), near (3..15 vox outside), far.
Outputs: outputs/plots/visual_proof_blurig/<cell>/cases.csv (resumable, appended per case), <cell>/<case>_blurig.npz (fold-mean maps, central
slabs only, float16), plots + blurig_stats.md with --report (CPU).
GPU: .venv/bin/python visual_proof_blurig.py --cells bratsfull_t2w_t1n [--case-slice 0:24]      CPU: --report
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

OUT = Path(os.environ.get("VP4_OUT") or (THIS.parent / "outputs/plots/visual_proof_blurig")); OUT.mkdir(parents=True, exist_ok=True)
CELL_KEYS = ["bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w"]
EFFECT = {"bratsfull_t2w_t1n": "Dice flat->real 0.38->0.44", "bratsfull_t2w_t2f": "Dice 0.69->0.73", "bratsfull_t1n_t2f": "Dice 0.40->0.57", "bratsfull_t1n_t2w": "Dice 0.64->0.70"}   # 3-fold majority edema Dice, 70 cases (v2 full)
FINE = 1.0   # sigma below which a path step counts as the fine (texture) band
ARMS = ("flat", "real"); S_MAX = float(os.environ.get("BLURIG_SMAX", 4.0)); K = int(os.environ.get("BLURIG_K", 32)); EDEMA = 2
ZONES = ("interior", "border", "near", "far")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def patch_at(gt_t, size, shape):
    c = np.argwhere(gt_t).mean(0).astype(int) if gt_t.any() else np.array(shape) // 2
    size = np.minimum(np.array(size), np.array(shape)); lo = np.clip(c - size // 2, 0, np.array(shape) - size)
    return tuple(slice(int(a), int(a + s)) for a, s in zip(lo, size))


def blur_path(ip, body):
    sig = np.linspace(S_MAX, 0.0, K + 1)
    return [np.where(body, ndi.gaussian_filter(ip, s), ip).astype(np.float32) if s > 0 else ip.astype(np.float32) for s in sig]


def target(out, M):
    return (out[0, EDEMA] - out[0, 0])[M].mean()


SIG = None


def blur_ig(net, path, M, dev, zmasks):
    """Returns (A map, A fine-band map, f(x), f(baseline), per-step zone contributions [K x zones], argmax edema mask of the real image)."""
    sig = np.linspace(S_MAX, 0.0, K + 1); A = torch.zeros(path[0].shape, device=dev); Af = torch.zeros_like(A); steps = []
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        fb = target(net(torch.from_numpy(path[0])[None, None].to(dev)), M).item()
        o = net(torch.from_numpy(path[-1])[None, None].to(dev)); fx = target(o, M).item(); pred = (o[0].argmax(0) == EDEMA)
    for k in range(K):
        a, b = torch.from_numpy(path[k]).to(dev), torch.from_numpy(path[k + 1]).to(dev)
        xm = (0.5 * (a + b))[None, None].clone().requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            out = net(xm); out = out[0] if isinstance(out, (list, tuple)) else out
            g, = torch.autograd.grad(target(out, M), xm)
        c = g[0, 0] * (b - a); A += c
        if 0.5 * (sig[k] + sig[k + 1]) < FINE:
            Af += c
        steps.append([c.sum().item()] + [c[zm].sum().item() for zm in zmasks])
    return A, Af, fx, fb, np.array(steps), pred


def zones(gp, body):
    din = ndi.distance_transform_edt(gp); dout = ndi.distance_transform_edt(~gp)
    return {"interior": gp & (din >= 3), "border": ((din <= 2) & gp) | ((dout <= 2) & ~gp), "near": (~gp) & (dout > 2) & (dout <= 15) & body,
            "far": (~gp) & (dout > 15) & body}


def run_cell(key, dev, sl=None):
    cell = v2.CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        dec = pr.network.decoder
        if hasattr(dec, "deep_supervision"):
            dec.deep_supervision = False
    cases = list(cell["cases"]); cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "cases.csv"; done = set(pd.read_csv(f_csv).case) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; gt_t = gt == EDEMA
        if gt_t.sum() < 50:
            log(f"[{key}] {case}: GT edema < 50 vox, skipped"); continue
        body = img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool)
        box = patch_at(gt_t, ps, img.shape); ip, bp, gp = img[box], body[box], gt_t[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]   # images smaller than the plan patch (e.g. 159 < 160): zero-pad, mask out
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        path = blur_path(ip, bp); M = torch.from_numpy(gp).to(dev); Z = zones(gp, bp); zt = [torch.from_numpy(Z[z]).to(dev) for z in ZONES]
        row = dict(cell=key, case=case, gt_vox=int(gt_t.sum()), patch_gt_vox=int(gp.sum())); maps = {}; fmaps = {}; preds = {}; srows = []
        sig = np.linspace(S_MAX, 0.0, K + 1); smid = 0.5 * (sig[:-1] + sig[1:])
        for a, pr in prs.items():
            net = pr.network; acc = None; accf = None; fxs, fbs, sums, stp, pv = [], [], [], [], 0
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                A, Af, fx, fb, st, pd_ = blur_ig(net, path, M, dev, zt); fxs.append(fx); fbs.append(fb); sums.append(A.sum().item()); stp.append(st); pv = pv + pd_.int()
                acc = A if acc is None else acc + A; accf = Af if accf is None else accf + Af
            Am = (acc / len(v2.FOLDS)).cpu().numpy(); maps[a] = Am; fmaps[a] = (accf / len(v2.FOLDS)).cpu().numpy(); preds[a] = (pv >= 2).cpu().numpy()
            st = np.mean(stp, 0)
            for kk in range(K):
                for zi, zn in enumerate(("total",) + ZONES):
                    srows.append(dict(cell=key, case=case, arm=a, step=kk, sigma=float(smid[kk]), zone=zn, contrib=float(st[kk, zi])))
            row[f"{a}_fine_contrib"] = float(st[smid < FINE, 0].sum()); row[f"{a}_coarse_contrib"] = float(st[smid >= FINE, 0].sum())
            row[f"{a}_f_real"] = float(np.mean(fxs)); row[f"{a}_f_blur"] = float(np.mean(fbs)); row[f"{a}_texture_contrib"] = float(np.mean(fxs) - np.mean(fbs))
            row[f"{a}_sumA"] = float(np.mean(sums)); row[f"{a}_completeness_err"] = float(np.mean(np.abs(np.array(sums) - (np.array(fxs) - np.array(fbs)))))
            tot_abs = np.abs(Am[bp]).sum() + 1e-12
            for zn, zm in Z.items():
                row[f"{a}_sum_{zn}"] = float(Am[zm].sum()); row[f"{a}_pos_{zn}"] = float(np.clip(Am[zm], 0, None).sum()); row[f"{a}_neg_{zn}"] = float(np.clip(Am[zm], None, 0).sum())
                row[f"{a}_absshare_{zn}"] = float(np.abs(Am[zm]).sum() / tot_abs); row[f"{a}_n_{zn}"] = int(zm.sum())
        # save 3 axial slices (axis 0, as in v2's sheets), >= 8 apart, with the most GT, + maps (float16)
        zs = np.array(v2.pick_slices(gp, 3, 8))
        np.savez_compressed(od / f"{case}_blurig.npz", z=zs, img=ip[zs].astype(np.float16), blur=path[0][zs].astype(np.float16), gt=gp[zs],
                            **{f"A_{a}": m[zs].astype(np.float16) for a, m in maps.items()}, **{f"Af_{a}": m[zs].astype(np.float16) for a, m in fmaps.items()},
                            **{f"pred_{a}": m[zs] for a, m in preds.items()})
        f_s = od / "scales.csv"; pd.DataFrame(srows).to_csv(f_s, mode="a", index=False, header=not (f_s.exists() and f_s.stat().st_size > 0))
        log(f"[{key}] {case}: texture contribution to edema logit margin flat {row['flat_texture_contrib']:+.3f} real {row['real_texture_contrib']:+.3f} "
            f"(compl. err {row['flat_completeness_err']:.3f}/{row['real_completeness_err']:.3f}); interior sum flat {row['flat_sum_interior']:+.3f} real {row['real_sum_interior']:+.3f}; {time.time() - t0:.0f} s")
        pd.DataFrame([row]).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)


def read_clean(f):
    d = pd.read_csv(f); d = d[d.case != "case"]
    for c in d.columns:
        if c not in ("cell", "case"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates("case", keep="last")


def report():
    wp = lambda x, y: wilcoxon(x, y).pvalue if len(x) >= 5 and (np.asarray(x) != np.asarray(y)).any() else np.nan
    cells = [c for c in CELL_KEYS if (OUT / c / "cases.csv").exists()]; df = pd.concat([read_clean(OUT / c / "cases.csv") for c in cells], ignore_index=True)
    df.to_csv(OUT / "cases_all.csv", index=False)
    L = ["# Visual proof v4: blur integrated gradients (texture contribution to the edema evidence), flat-fill vs real-fill", "",
         f"Path: Gaussian blur sigma {S_MAX} -> 0 vox in {K} steps inside the body; target = mean (logit_edema - logit_bg) over GT edema; fold mean over 3 folds; "
         "texture contribution = f(real image) - f(blurred image) = sum of the map (completeness). Paired Wilcoxon real vs flat over cases.", "",
         "## Texture contribution per arm (logit-margin units)", "",
         "| cell | ladder | n | f(real) flat / real | f(blur) flat / real | texture contrib flat (median) | real | real > flat | p | frac cases contrib > 0 flat / real | max completeness err |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = df[df.cell == c]; tf, tr = d.flat_texture_contrib, d.real_texture_contrib
        L.append(f"| {c} | {EFFECT[c]} | {len(d)} | {d.flat_f_real.median():.2f} / {d.real_f_real.median():.2f} | {d.flat_f_blur.median():.2f} / {d.real_f_blur.median():.2f} | "
                 f"{tf.median():+.3f} | {tr.median():+.3f} | {(tr > tf).mean():.2f} | {wp(tr, tf):.2g} | {(tf > 0).mean():.2f} / {(tr > 0).mean():.2f} | "
                 f"{max(d.flat_completeness_err.max(), d.real_completeness_err.max()):.3f} |")
    L += ["", "## Where the texture contribution sits: signed sum of the map per zone (interior = GT edema >= 3 vox deep, border = |d| <= 2, near = 3..15 vox outside, far)", "",
          "| cell | zone | sum flat (median) | real | real > flat | p | |A| share flat | real |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = df[df.cell == c]
        for z in ZONES:
            f_, r_ = d[f"flat_sum_{z}"], d[f"real_sum_{z}"]
            L.append(f"| {c} | {z} | {f_.median():+.3f} | {r_.median():+.3f} | {(r_ > f_).mean():.2f} | {wp(r_, f_):.2g} | {d[f'flat_absshare_{z}'].median():.2f} | {d[f'real_absshare_{z}'].median():.2f} |")
    (OUT / "blurig_stats.md").write_text("\n".join(L)); print("\n".join(L))
    # figure 1: paired texture contribution per cell
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 4.2), facecolor="#fcfcfb", squeeze=False)
    for j, c in enumerate(cells):
        d = df[df.cell == c]; ax = axes[0, j]; tf, tr = d.flat_texture_contrib.values, d.real_texture_contrib.values
        parts = ax.violinplot([tf, tr], positions=[0, 1], showmedians=True)
        for pc, col in zip(parts["bodies"], ("#2a78d6", "#eb6834")):
            pc.set_facecolor(col); pc.set_alpha(0.35)
        for a_, b_ in zip(tf, tr):
            ax.plot([0, 1], [a_, b_], color="0.6", lw=0.4, alpha=0.6)
        ax.axhline(0, color="k", lw=0.7); ax.set_xticks([0, 1]); ax.set_xticklabels(["flat-fill", "real-fill"])
        ax.set_title(f"{c.replace('bratsfull_', '')}\n{EFFECT[c]}: real>flat {(tr > tf).mean():.0%}, p={wp(tr, tf):.1g}", fontsize=9)
        if j == 0:
            ax.set_ylabel("texture contribution to edema logit margin\nf(image) - f(blurred image)")
    fig.suptitle("Blur integrated gradients: how much of each model's edema evidence comes from texture (70 BraTS cases per cell, 3-fold mean)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "texture_contribution.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    # figure 2: zone decomposition (median signed sums)
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.8), facecolor="#fcfcfb", squeeze=False)
    for j, c in enumerate(cells):
        d = df[df.cell == c]; ax = axes[0, j]; x = np.arange(len(ZONES))
        ax.bar(x - 0.18, [d[f"flat_sum_{z}"].median() for z in ZONES], 0.36, color="#2a78d6", label="flat-fill")
        ax.bar(x + 0.18, [d[f"real_sum_{z}"].median() for z in ZONES], 0.36, color="#eb6834", label="real-fill")
        ax.axhline(0, color="k", lw=0.7); ax.set_xticks(x); ax.set_xticklabels(ZONES); ax.set_title(f"{c.replace('bratsfull_', '')} ({EFFECT[c]})", fontsize=9)
        if j == 0:
            ax.set_ylabel("median signed sum of blur-IG map"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "texture_contribution_zones.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    # fine (texture, sigma < FINE) vs coarse band table
    L2 = ["", f"## Scale split: evidence gained in the fine band (sigma < {FINE:g} vox = texture) vs the coarse band (sigma {FINE:g}..{S_MAX:g})", "",
          "| cell | fine flat | fine real | real > flat | p | coarse flat | coarse real | real > flat | p |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = df[df.cell == c]
        L2.append(f"| {c} | {d.flat_fine_contrib.median():+.3f} | {d.real_fine_contrib.median():+.3f} | {(d.real_fine_contrib > d.flat_fine_contrib).mean():.2f} | {wp(d.real_fine_contrib, d.flat_fine_contrib):.2g} | "
                  f"{d.flat_coarse_contrib.median():+.3f} | {d.real_coarse_contrib.median():+.3f} | {(d.real_coarse_contrib > d.flat_coarse_contrib).mean():.2f} | {wp(d.real_coarse_contrib, d.flat_coarse_contrib):.2g} |")
    with open(OUT / "blurig_stats.md", "a") as fh:
        fh.write("\n".join(L2) + "\n")
    print("\n".join(L2))
    # figure 3: cumulative evidence along the scale path (coarse -> fine), per arm, per zone
    sc = pd.concat([pd.read_csv(OUT / c / "scales.csv") for c in cells if (OUT / c / "scales.csv").exists()], ignore_index=True)
    sc = sc[sc.case != "case"]; sc["contrib"] = pd.to_numeric(sc.contrib); sc["sigma"] = pd.to_numeric(sc.sigma); sc["step"] = pd.to_numeric(sc.step)
    zl = ("total",) + ZONES
    fig, axes = plt.subplots(len(zl), len(cells), figsize=(4.3 * len(cells), 2.7 * len(zl)), facecolor="#fcfcfb", squeeze=False)
    for j_, c in enumerate(cells):
        for i_, z in enumerate(zl):
            ax = axes[i_, j_]
            for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
                t = sc[(sc.cell == c) & (sc.arm == a) & (sc.zone == z)].pivot_table(index="case", columns="step", values="contrib").sort_index(axis=1).cumsum(1)
                sig = sc[(sc.cell == c) & (sc.zone == z) & (sc.arm == a)].groupby("step").sigma.first().sort_index().values
                ax.plot(sig, t.median().values, color=col, lw=1.8, label=f"{a}-fill"); ax.fill_between(sig, t.quantile(.25).values, t.quantile(.75).values, color=col, alpha=0.15)
            ax.invert_xaxis(); ax.axhline(0, color="k", lw=0.6); ax.axvline(FINE, color="0.5", lw=0.6, ls="--")
            ax.set_title(f"{c.replace('bratsfull_', '')}: {z}", fontsize=9)
            if j_ == 0:
                ax.set_ylabel("cumulative evidence\n(logit margin)")
            if i_ == len(zl) - 1:
                ax.set_xlabel("blur sigma along the path (vox), coarse -> fine")
            if i_ == 0 and j_ == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Blur IG in scale: edema evidence accumulated as the image is sharpened from sigma 4 to 0 (median, IQR over 70 cases); right of the dashed line = fine scales (texture)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "scale_curves.png", dpi=105, facecolor="#fcfcfb"); plt.close(fig)
    # sheets: per cell 2 cases with the largest real - flat fine-band contribution, 1 with the smallest, 1 median
    for c in cells:
        d = df[df.cell == c].assign(diff=lambda t: t.real_fine_contrib - t.flat_fine_contrib).sort_values("diff").reset_index(drop=True)
        for case in list(dict.fromkeys([d.case.iloc[-1], d.case.iloc[-2], d.case.iloc[len(d) // 2], d.case.iloc[0]])):
            f = OUT / c / f"{case}_blurig.npz"
            if not f.exists():
                continue
            z = np.load(f); r = d[d.case == case].iloc[0]; n = z["img"].shape[0]
            sm = lambda m: ndi.gaussian_filter(m.astype(np.float32), 1.0)
            Fs = {a: [sm(z[f"Af_{a}"][i]) for i in range(n)] for a in ARMS}; As = {a: [sm(z[f"A_{a}"][i]) for i in range(n)] for a in ARMS}
            vf = np.percentile(np.abs(np.concatenate([np.ravel(m) for a in ARMS for m in Fs[a]])), 99.5) + 1e-9
            va = np.percentile(np.abs(np.concatenate([np.ravel(m) for a in ARMS for m in As[a]])), 99.5) + 1e-9
            fig, axes = plt.subplots(n, 6, figsize=(21, 3.9 * n), facecolor="#fcfcfb", squeeze=False)
            for i in range(n):
                im, g = z["img"][i].astype(np.float32), z["gt"][i]; lo, hi = np.percentile(im[im != 0], [1, 99.5]) if (im != 0).any() else (0, 1)
                pf, pr_ = z["pred_flat"][i], z["pred_real"][i]
                pans = [(im, "image  (GT edema green)", "gray", (lo, hi)), (As["flat"][i], "flat-fill: all scales", "RdBu_r", (-va, va)), (As["real"][i], "real-fill: all scales", "RdBu_r", (-va, va)),
                        (Fs["flat"][i], f"flat-fill: fine scales (sigma<{FINE:g})", "RdBu_r", (-vf, vf)), (Fs["real"][i], f"real-fill: fine scales (sigma<{FINE:g})", "RdBu_r", (-vf, vf)),
                        (Fs["real"][i] - Fs["flat"][i], "fine scales: real - flat", "PuOr_r", (-vf, vf))]
                for jj, (pan, ttl, cm, (v0, v1)) in enumerate(pans):
                    ax = axes[i, jj]; ax.imshow(pan.T, cmap=cm, vmin=v0, vmax=v1, origin="lower", interpolation="nearest")
                    ax.contour(g.T, [0.5], colors="#1ea672", linewidths=1.0, origin="lower")
                    if jj in (1, 3):
                        ax.contour(pf.T, [0.5], colors="#2a78d6", linewidths=0.9, linestyles="dotted", origin="lower")
                    if jj in (2, 4):
                        ax.contour(pr_.T, [0.5], colors="#eb6834", linewidths=0.9, linestyles="dotted", origin="lower")
                    ax.set_title(ttl if i == 0 else "", fontsize=9); ax.axis("off")
            fig.suptitle(f"{c} ({EFFECT[c]})  {case}: texture contribution to the edema logit margin, flat {r.flat_texture_contrib:+.2f} / real {r.real_texture_contrib:+.2f}; "
                         f"fine-scale part flat {r.flat_fine_contrib:+.2f} / real {r.real_fine_contrib:+.2f}.\nBlur IG (sigma 4 -> 0), 3-fold mean, smoothed 1 vox for display. Red = this voxel's detail raises the edema evidence, blue = lowers it. "
                         "Green = GT edema, dotted = that model's predicted edema (3-fold majority).", fontsize=10, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / c / f"{case}_blurig.png", dpi=90, facecolor="#fcfcfb"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELL_KEYS)); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    dev = torch.device("cuda")
    for c in a.cells.split(","):
        run_cell(c, dev, a.case_slice)


if __name__ == "__main__":
    main()
