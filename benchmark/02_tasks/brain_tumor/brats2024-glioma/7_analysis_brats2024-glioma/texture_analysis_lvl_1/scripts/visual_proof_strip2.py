#!/usr/bin/env python
"""
Visual proof v6 (2026-10-12): scale-space / texture-removal PREDICTION STRIP with an EDGE-PRESERVING path and the controls.
Why: v5 sharpened the image along a Gaussian path, which removes texture AND blurs the target boundary, so "the real-fill model gains more from
fine detail" could be edges. Here two paths are run on the same patch:
  gauss  : Gaussian blur, sigma 4, 3, 2, 1.5, 1, 0.5, 0 vox (as v5)
  diff   : Perona-Malik anisotropic diffusion (6-neighbour, conductance exp(-(g/kappa)^2), lambda 0.14), 100, 50, 25, 12, 6, 3, 0 iterations;
           kappa = 60th percentile of |grad| over the body, so small (texture) gradients are smoothed and strong (boundary) gradients kept.
  noise  : the original + i.i.d. Gaussian noise 0.05 / 0.10 x body std (control for generic fragility).
QC per level (GT used ONLY to read these numbers, never to build an input): texture remaining = high-pass energy (x - G_1 x)^2 inside the
target / original; boundary remaining = mean |grad G_1 x| on the target boundary band (|d| <= 1 vox) / original. Both arms see the same inputs.
Score: fold-mean softmax, per-label Dice in a plan-sized patch around the target, mean over the cell's target labels present in the GT.
Cells: the 4 BraTS cells (edema), Open-MS FLAIR-trained -> T2w / T1w (lesion), pelvis MRI-trained -> CT (all 11 structures; the cell where
texture HURT in the 4.5 -> 5 step = the control).
GPU: .venv/bin/python visual_proof_strip2.py --cells pelvis_ct_full [--case-slice 0:15]      CPU: --report
Output: outputs/plots/visual_proof_strip2/<cell>/{cases.csv, <case>_strip.npz}, curves + strips + strip2_stats.md
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
os.environ.setdefault("VP_OUT", str(THIS.parent / "outputs/plots/visual_proof_v2"))
import visual_proof_v2 as v2  # noqa: E402

OUT = Path(os.environ.get("VP6_OUT") or (THIS.parent / "outputs/plots/visual_proof_strip2")); OUT.mkdir(parents=True, exist_ok=True)
PELVIS_CASES = sorted(p.name.replace("_0000.nii.gz", "") for p in (v2.RPC / "imagesTs_ct").glob("*_0000.nii.gz"))
CELLS = {k: dict(v2.CELLS[k], target=[2]) for k in ("bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w")}
CELLS.update({k: dict(v2.CELLS[k], target=[1]) for k in ("openms_t2w", "openms_t1w")})
CELLS["pelvis_ct_full"] = dict(v2.CELLS["pelvis_ct"], cases=PELVIS_CASES, target=list(range(1, 12)))
TITLE = {"bratsfull_t2w_t1n": "BraTS T2w-trained -> T1n (edema)", "bratsfull_t2w_t2f": "BraTS T2w-trained -> FLAIR (edema)",
         "bratsfull_t1n_t2f": "BraTS T1n-trained -> FLAIR (edema)", "bratsfull_t1n_t2w": "BraTS T1n-trained -> T2w (edema)",
         "openms_t2w": "Open-MS FLAIR-trained -> T2w (lesions)", "openms_t1w": "Open-MS FLAIR-trained -> T1w (lesions)",
         "pelvis_ct_full": "Pelvis MRI-trained -> CT (11 structures; CONTROL)"}
ARMS = ("flat", "real"); SIGMAS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0]; ITERS = [25, 10, 0]; NOISE = [0.05, 0.10]   # ITERS: 2026-10-12 calibration (visual_proof_strip2_calibrate.py) -- Perona-Malik gives NO better texture/contrast trade-off than Gaussian here; kept as 2 reference points
LEVELS = [f"g{s:g}" for s in SIGMAS] + [f"d{n}" for n in ITERS] + [f"n{n:g}" for n in NOISE]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def patch_at(m, size, shape):
    c = np.argwhere(m).mean(0).astype(int) if m.any() else np.array(shape) // 2
    size = np.minimum(np.array(size), np.array(shape)); lo = np.clip(c - size // 2, 0, np.array(shape) - size)
    return tuple(slice(int(a), int(a + s)) for a, s in zip(lo, size))


def perona_malik(x: torch.Tensor, body: torch.Tensor, iters: list[int], kappa: float, lam: float = 0.14):
    """x (Z,Y,X) on device; returns {n_iter: field} for the requested iteration counts (body voxels only evolve)."""
    out = {0: x.clone()}; u = x.clone(); todo = sorted(i for i in iters if i > 0); k2 = kappa ** 2
    for it in range(1, (todo[-1] if todo else 0) + 1):
        up = F.pad(u[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]; flux = torch.zeros_like(u)
        for ax in range(3):
            for sh in (1, -1):
                nb = torch.roll(up, shifts=sh, dims=ax)[1:-1, 1:-1, 1:-1]; d = nb - u; flux += torch.exp(-(d * d) / k2) * d
        u = torch.where(body, u + lam * flux, u)
        if it in todo:
            out[it] = u.clone()
    return out


def inputs(ip, body, dev):
    X = {}
    for s in SIGMAS:
        X[f"g{s:g}"] = np.where(body, ndi.gaussian_filter(ip, s), ip).astype(np.float32) if s > 0 else ip.astype(np.float32)
    g = np.sqrt(sum(ndi.sobel(ip, axis=a) ** 2 for a in range(3))) / 8.0; kappa = float(np.percentile(g[body], 50)) if body.any() else 1.0
    pm = perona_malik(torch.from_numpy(ip.astype(np.float32)).to(dev), torch.from_numpy(body).to(dev), ITERS, kappa)
    for n in ITERS:
        X[f"d{n}"] = pm[n].cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(0); sd = ip[body].std() if body.any() else 1.0
    for n in NOISE:
        X[f"n{n:g}"] = np.where(body, ip + rng.normal(0, n * sd, ip.shape), ip).astype(np.float32)
    return X, kappa


def qc(X, tgt):
    din = ndi.distance_transform_edt(tgt); dout = ndi.distance_transform_edt(~tgt); band = ((din <= 1) & tgt) | ((dout <= 1) & ~tgt)
    res = {}
    for lv, x in X.items():
        hp = x - ndi.gaussian_filter(x, 1.0); g = np.sqrt(sum(ndi.sobel(ndi.gaussian_filter(x, 1.0), axis=a) ** 2 for a in range(3)))
        res[lv] = (float((hp[tgt] ** 2).mean()) if tgt.any() else np.nan, float(g[band].mean()) if band.any() else np.nan)
    t0, b0 = res["g0"]
    return {lv: (t / t0 if t0 else np.nan, b / b0 if b0 else np.nan) for lv, (t, b) in res.items()}


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


@torch.no_grad()
def run_cell(key, dev, sl=None):
    cell = CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True); T = cell["target"]
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]); cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "cases.csv"; done = set(pd.read_csv(f_csv).case) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        if tg.sum() < 50:
            continue
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = patch_at(tg, ps, img.shape); ip, bp, gp = img[box], body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        tgp = np.isin(gp, T); X, kappa = inputs(ip, bp, dev); Q = qc(X, tgp); z = int(np.argmax(tgp.sum((1, 2))))
        present = [l for l in T if (gp == l).any()]
        row = dict(cell=key, case=case, target_vox=int(tg.sum()), kappa=kappa, n_labels=len(present))
        for lv in LEVELS:
            row[f"qc_tex_{lv}"], row[f"qc_bnd_{lv}"] = Q[lv]
        probs = {}
        for a, pr in prs.items():
            net = pr.network; acc = {lv: 0 for lv in LEVELS}
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                for lv in LEVELS:
                    with torch.autocast("cuda", enabled=False):
                        o = net(torch.from_numpy(X[lv])[None, None].to(dev)); o = o[0] if isinstance(o, (list, tuple)) else o
                    acc[lv] = acc[lv] + torch.softmax(o[0].float(), 0)
            for lv in LEVELS:
                p = acc[lv] / len(v2.FOLDS); am = p.argmax(0).cpu().numpy()
                ds = [dice(am == l, gp == l) for l in present]; row[f"{a}_{lv}_dice"] = float(np.nanmean(ds)) if ds else np.nan
                for l, dv in zip(present, ds):
                    row[f"{a}_{lv}_dice_l{l}"] = float(dv)
                probs[f"p_{a}_{lv}"] = p[T].sum(0)[z].cpu().numpy().astype(np.float16)
        np.savez_compressed(od / f"{case}_strip.npz", z=z, gt=tgp[z], **{f"img_{lv}": X[lv][z].astype(np.float16) for lv in LEVELS}, **probs)
        log(f"[{key}] {case}: Dice orig flat {row['flat_g0_dice']:.2f} real {row['real_g0_dice']:.2f} | blur1 flat {row['flat_g1_dice']:.2f} real {row['real_g1_dice']:.2f} "
            f"(texture left {row['qc_tex_g1']:.2f}, edge left {row['qc_bnd_g1']:.2f}) | gauss2 flat {row['flat_g2_dice']:.2f} real {row['real_g2_dice']:.2f} "
            f"(tex {row['qc_tex_g2']:.2f}, bnd {row['qc_bnd_g2']:.2f}); {time.time() - t0:.0f} s")
        pd.DataFrame([row]).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)


def read_clean(f):
    d = pd.read_csv(f); d = d[d.case != "case"]
    for c in d.columns:
        if c not in ("cell", "case"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates("case", keep="last")


def report():
    def wp(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float); ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
        return wilcoxon(x, y).pvalue if len(x) >= 5 and (x != y).any() else np.nan
    cells = [c for c in CELLS if (OUT / c / "cases.csv").exists()]; df = pd.concat([read_clean(OUT / c / "cases.csv") for c in cells], ignore_index=True)
    df.to_csv(OUT / "cases_all.csv", index=False)
    L = ["# Visual proof v6: prediction under texture removal -- Gaussian path vs EDGE-PRESERVING (Perona-Malik) path, + noise control", "",
         "Fold-mean softmax; Dice per target label in a plan-sized patch, averaged over labels present. 'loss' = Dice(original) - Dice(perturbed). "
         "QC (median over cases): texture left = high-pass energy in the target relative to the original; boundary left = gradient on the target boundary band relative to the original. "
         "Paired Wilcoxon real vs flat over cases.", "",
         "## QC: what each path removes (median over all cases of each cell)", "", "| cell | level | texture left | boundary left |", "|---|---|---|---|"]
    for c in cells:
        d = df[df.cell == c]
        for lv in ("g2", "g1", "g0.5", "d25", "d10"):
            L.append(f"| {c} | {lv} | {d[f'qc_tex_{lv}'].median():.2f} | {d[f'qc_bnd_{lv}'].median():.2f} |")
    L += ["", "## Loss of Dice when texture is removed (per arm) and the real - flat difference", "",
          "| cell | n | Dice orig flat / real | GAUSS sigma 1 (texture ~25% left, contrast ~80%): loss flat / real | real loses more | p | GAUSS sigma 2: loss flat / real | real loses more | p | NOISE 0.10: loss flat / real | real loses more | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = df[df.cell == c]; o = {a: d[f"{a}_g0_dice"] for a in ARMS}
        def part(lv):
            ls = {a: o[a] - d[f"{a}_{lv}_dice"] for a in ARMS}
            return f"{ls['flat'].mean():+.3f} / {ls['real'].mean():+.3f} | {(ls['real'] > ls['flat']).mean():.2f} | {wp(ls['real'], ls['flat']):.2g}"
        L.append(f"| {c} | {len(d)} | {o['flat'].mean():.3f} / {o['real'].mean():.3f} | {part('g1')} | {part('g2')} | {part('n0.1')} |")
    # curves: Dice vs texture left (x, measured), both paths
    fig, axes = plt.subplots(1, len(cells), figsize=(4.0 * len(cells), 4.0), facecolor="#fcfcfb", squeeze=False)
    for j, c in enumerate(cells):
        d = df[df.cell == c]; ax = axes[0, j]
        for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
            for path, lvs, ls_, mk in (("edge-preserving", [f"d{n}" for n in ITERS], "-", "o"), ("Gaussian", [f"g{s:g}" for s in SIGMAS], ":", "s")):
                xs = [d[f"qc_tex_{lv}"].median() for lv in lvs]; ys = [d[f"{a}_{lv}_dice"].mean() for lv in lvs]
                ax.plot(xs, ys, ls_, marker=mk, color=col, lw=1.6, ms=3.5, label=f"{a}-fill, {path}")
            ax.plot([1.0], [d[f"{a}_n0.1_dice"].mean()], "^", color=col, mfc="white", ms=8)
        ax.set_xscale("symlog", linthresh=0.05); ax.set_xlabel("texture left in the target (measured, 1 = original)"); ax.set_title(TITLE[c], fontsize=9)
        if j == 0:
            ax.set_ylabel("Dice (mean over cases)"); ax.legend(fontsize=7)
    fig.suptitle("Dice vs measured texture left: dotted = Gaussian blur path, solid = Perona-Malik (2 points); open triangle = original + noise control", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "dice_vs_texture_left.png", dpi=105, facecolor="#fcfcfb"); plt.close(fig)
    # summary bar: real - flat difference in loss under edge-preserving removal, per cell
    fig, ax = plt.subplots(figsize=(1.3 * len(cells) + 2, 4.0), facecolor="#fcfcfb")
    vals, errs, labs = [], [], []
    for c in cells:
        d = df[df.cell == c]; dd = (d.real_g0_dice - d.real_g1_dice) - (d.flat_g0_dice - d.flat_g1_dice); vals.append(dd.mean()); errs.append(dd.std() / np.sqrt(len(d))); labs.append(TITLE[c].replace(" (", "\n("))
    ax.bar(range(len(cells)), vals, yerr=errs, color=["#eb6834" if v > 0 else "#2a78d6" for v in vals], capsize=3); ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(labs, fontsize=7, rotation=0); ax.set_ylabel("extra Dice lost by the real-fill model\nwhen fine detail is removed (Gaussian sigma 1)")
    fig.tight_layout(); fig.savefig(OUT / "texture_dependence_by_cell.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    # strips: per cell, the case with the largest real - flat extra loss under edge-preserving removal (among real Dice orig >= 0.5) and the median case
    sel = ["", "## Strip cases (rule: largest real - flat extra Dice loss when fine detail is removed (sigma 1) among cases with real Dice(original) >= 0.5; and the median case)", ""]
    for c in cells:
        d = df[df.cell == c].copy(); d["x"] = (d.real_g0_dice - d.real_g1_dice) - (d.flat_g0_dice - d.flat_g1_dice)
        cand = d[d.real_g0_dice >= 0.5].sort_values("x", ascending=False); med = d.iloc[(d.x - d.x.median()).abs().argsort().iloc[0]].case
        picks = list(dict.fromkeys(list(cand.case.head(1)) + [med])); sel.append(f"- {c}: " + ", ".join(f"{p} ({'top' if i == 0 and p != med else 'median'}, extra loss {d[d.case == p].x.iloc[0]:+.2f})" for i, p in enumerate(picks)))
        for tag, case in zip(("top", "median") if len(picks) == 2 else ("median",), picks):
            f = OUT / c / f"{case}_strip.npz"
            if not f.exists():
                continue
            zf = np.load(f); r = d[d.case == case].iloc[0]; g = zf["gt"]; ys, xs_ = np.nonzero(g)
            y0, y1, x0, x1 = (max(0, ys.min() - 25), min(g.shape[0], ys.max() + 26), max(0, xs_.min() - 25), min(g.shape[1], xs_.max() + 26)) if ys.size else (0, g.shape[0], 0, g.shape[1])
            lvs = ["g4", "g2", "g1.5", "g1", "g0.5", "g0", "n0.1"]; nm = {"g4": "blur sigma 4", "g2": "blur sigma 2", "g1.5": "blur sigma 1.5", "g1": "blur sigma 1", "g0.5": "blur sigma 0.5", "g0": "original", "n0.1": "original + noise"}
            fig, axes = plt.subplots(2, len(lvs), figsize=(2.75 * len(lvs), 6.2), facecolor="#fcfcfb")
            for i, a in enumerate(ARMS):
                for jj, lv in enumerate(lvs):
                    ax = axes[i, jj]; im = zf[f"img_{lv}"].astype(np.float32)[y0:y1, x0:x1]; p = zf[f"p_{a}_{lv}"].astype(np.float32)[y0:y1, x0:x1]
                    lo, hi = np.percentile(im, [1, 99.5]); ax.imshow(im.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
                    ax.imshow(np.ma.masked_less(p.T, 0.5), cmap="autumn_r" if a == "real" else "winter", vmin=0.5, vmax=1.0, alpha=0.5, origin="lower", interpolation="nearest")
                    ax.contour(g[y0:y1, x0:x1].T, [0.5], colors="#1ea672", linewidths=1.1, origin="lower")
                    ttl = f"{nm[lv]}\nDice {r[f'{a}_{lv}_dice']:.2f}" + (f"  (texture {r[f'qc_tex_{lv}']:.2f}, edge {r[f'qc_bnd_{lv}']:.2f})" if i == 0 and lv not in ("g0", "n0.1") else "")
                    ax.set_title(ttl, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
                    if jj == 0:
                        ax.set_ylabel(v2.ARM_LABEL[a], fontsize=9)
            fig.suptitle(f"{TITLE[c]}, {case} ({tag} case by the stated rule): prediction (coloured, p > 0.5, fold mean) as the image is sharpened from blur sigma 4 to the original "
                         "(texture / contrast left measured per level in the top row titles), last column = noise control. Green = GT.", fontsize=9, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / f"strip_{c}_{tag}_{case}.png", dpi=105, facecolor="#fcfcfb"); plt.close(fig)
    # gap vs scale: mean paired (real - flat) Dice per blur level, one curve per cell -- where does the fill-swap effect live?
    G = ["", "## Real - flat Dice gap per blur level (mean, paired); the fill-swap effect at each scale", "", "| cell | " + " | ".join(f"sigma {s:g}" for s in SIGMAS) + " | noise 0.10 | p (sigma 0 vs sigma 2 gap) |",
         "|---|" + "---|" * (len(SIGMAS) + 2)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor="#fcfcfb"); cols = plt.cm.tab10.colors
    for i, c in enumerate(cells):
        d = df[df.cell == c]; gaps = [(d[f"real_g{s:g}_dice"] - d[f"flat_g{s:g}_dice"]) for s in SIGMAS]; m = np.array([g.mean() for g in gaps]); se = np.array([g.std() / np.sqrt(g.notna().sum()) for g in gaps])
        ctl = c.startswith("pelvis"); ax.plot(SIGMAS, m, "--" if ctl else "-", marker="o", ms=3.5, lw=2.2 if ctl else 1.6, color="k" if ctl else cols[i], label=TITLE[c])
        ax.fill_between(SIGMAS, m - se, m + se, color="k" if ctl else cols[i], alpha=0.12)
        gn = d["real_n0.1_dice"] - d["flat_n0.1_dice"]
        G.append(f"| {c} | " + " | ".join(f"{v:+.3f}" for v in m) + f" | {gn.mean():+.3f} | {wp(gaps[-1], gaps[2]):.2g} |")
    ax.axhline(0, color="k", lw=0.7); ax.invert_xaxis(); ax.set_xlabel("blur sigma applied to the test image (vox = mm), coarse -> original"); ax.set_ylabel("Dice: real-fill minus flat-fill (mean +- s.e.)")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("The fill-swap effect appears only when fine-scale detail is present", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "gap_vs_scale.png", dpi=120, facecolor="#fcfcfb"); plt.close(fig)
    (OUT / "strip2_stats.md").write_text("\n".join(L + G + sel)); print("\n".join(L + G + sel))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELLS)); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    for c in a.cells.split(","):
        run_cell(c, torch.device("cuda"), a.case_slice)


if __name__ == "__main__":
    main()
