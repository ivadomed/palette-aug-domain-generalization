#!/usr/bin/env python
"""
Visual proof v5 (2026-10-11): SCALE-SPACE PREDICTION STRIP. Same checkpoints (flat-fill rung 4.5 vs real-fill rung 5, 3 folds), same 70 BraTS test
cases and 4 cells as v2-v4, edema. The image is sharpened from a heavy blur to the original along the Gaussian scale-space path used by Blur IG
(Xu et al., CVPR 2020); at each level both models predict; the question "at which scale does each model recognise the edema" becomes visible.
Control: the original image with added i.i.d. Gaussian noise (std 0.05 / 0.10 of the z-scored body intensities), so a stronger blur effect for
the real-fill model can be told apart from generic fragility to any input change.
Per case: one plan-sized patch centred on the GT edema (GT for placement and scoring only); fold-mean softmax; edema = argmax == 2; Dice vs GT in
the patch; edema probability on the axial slice with the most GT saved at every level for the strip figure.
GPU: .venv/bin/python visual_proof_strip.py --cells bratsfull_t2w_t2f [--case-slice 0:35]      CPU: --report
Output: outputs/plots/visual_proof_strip/<cell>/{cases.csv, <case>_strip.npz}, dice_vs_scale.png, strip_<cell>_<case>.png, strip_stats.md
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

OUT = Path(os.environ.get("VP5_OUT") or (THIS.parent / "outputs/plots/visual_proof_strip")); OUT.mkdir(parents=True, exist_ok=True)
CELL_KEYS = ["bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w"]
TITLE = {"bratsfull_t2w_t1n": "T2w-trained -> T1n", "bratsfull_t2w_t2f": "T2w-trained -> FLAIR", "bratsfull_t1n_t2f": "T1n-trained -> FLAIR", "bratsfull_t1n_t2w": "T1n-trained -> T2w"}
ARMS = ("flat", "real"); EDEMA = 2
SIGMAS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0]; NOISE = [0.05, 0.10]
LEVELS = [f"blur{s:g}" for s in SIGMAS] + [f"noise{n:g}" for n in NOISE]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def patch_at(gt_t, size, shape):
    c = np.argwhere(gt_t).mean(0).astype(int) if gt_t.any() else np.array(shape) // 2
    size = np.minimum(np.array(size), np.array(shape)); lo = np.clip(c - size // 2, 0, np.array(shape) - size)
    return tuple(slice(int(a), int(a + s)) for a, s in zip(lo, size))


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def inputs(ip, body):
    rng = np.random.default_rng(0); out = {}
    for s in SIGMAS:
        out[f"blur{s:g}"] = np.where(body, ndi.gaussian_filter(ip, s), ip).astype(np.float32) if s > 0 else ip.astype(np.float32)
    sd = ip[body].std() if body.any() else 1.0
    for n in NOISE:
        out[f"noise{n:g}"] = np.where(body, ip + rng.normal(0, n * sd, ip.shape), ip).astype(np.float32)
    return out


@torch.no_grad()
def run_cell(key, dev, sl=None):
    cell = v2.CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]); cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "cases.csv"; done = set(pd.read_csv(f_csv).case) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; gt_t = gt == EDEMA
        if gt_t.sum() < 50:
            continue
        body = img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool)
        box = patch_at(gt_t, ps, img.shape); ip, bp, gp = img[box], body[box], gt_t[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        X = inputs(ip, bp); z = int(np.argmax(gp.sum((1, 2))))
        row = dict(cell=key, case=case, gt_vox=int(gt_t.sum())); probs = {}
        for a, pr in prs.items():
            net = pr.network; acc = {lv: 0 for lv in LEVELS}
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                for lv in LEVELS:
                    with torch.autocast("cuda", enabled=False):
                        o = net(torch.from_numpy(X[lv])[None, None].to(dev)); o = o[0] if isinstance(o, (list, tuple)) else o
                    acc[lv] = acc[lv] + torch.softmax(o[0].float(), 0)
            for lv in LEVELS:
                p = (acc[lv] / len(v2.FOLDS)); pe = (p.argmax(0) == EDEMA).cpu().numpy()
                row[f"{a}_{lv}_dice"] = float(dice(pe, gp)); row[f"{a}_{lv}_vol"] = int(pe.sum()); probs[f"p_{a}_{lv}"] = p[EDEMA][z].cpu().numpy().astype(np.float16)
        np.savez_compressed(od / f"{case}_strip.npz", z=z, gt=gp[z], **{f"img_{lv}": X[lv][z].astype(np.float16) for lv in LEVELS}, **probs)
        log(f"[{key}] {case}: Dice flat blur4/1/0 {row['flat_blur4_dice']:.2f}/{row['flat_blur1_dice']:.2f}/{row['flat_blur0_dice']:.2f}  "
            f"real {row['real_blur4_dice']:.2f}/{row['real_blur1_dice']:.2f}/{row['real_blur0_dice']:.2f}; {time.time() - t0:.0f} s")
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
    # population curves
    fig, axes = plt.subplots(1, len(cells), figsize=(4.4 * len(cells), 4.0), facecolor="#fcfcfb", squeeze=False)
    L = ["# Visual proof v5: scale-space prediction strip -- edema Dice as the image is sharpened (blur sigma 4 -> 0) and under added noise", "",
         "Fold-mean softmax, 70 BraTS test cases per cell, Dice in a plan-sized patch around the GT edema. 'gain from sharpening' = Dice(sigma 0) - Dice(sigma 2); "
         "'noise loss' = Dice(sigma 0) - Dice(noise 0.10). Paired Wilcoxon real vs flat.", "",
         "| cell | Dice sigma 4 flat / real | sigma 2 | sigma 1 | sigma 0 | gain sigma2->0 flat / real | real > flat | p | noise-0.10 loss flat / real | real > flat | p |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for j, c in enumerate(cells):
        d = df[df.cell == c]; ax = axes[0, j]; xs = SIGMAS
        for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
            m = np.array([d[f"{a}_blur{s:g}_dice"].mean() for s in xs]); se = np.array([d[f"{a}_blur{s:g}_dice"].std() / np.sqrt(len(d)) for s in xs])
            ax.plot(xs, m, "-o", color=col, lw=1.8, ms=4, label=f"{a}-fill"); ax.fill_between(xs, m - se, m + se, color=col, alpha=0.2)
            for n, mk in zip(NOISE, ("^", "s")):
                ax.plot([-0.35 - 0.15 * NOISE.index(n)], [d[f"{a}_noise{n:g}_dice"].mean()], mk, color=col, ms=6, mfc="white")
        ax.invert_xaxis(); ax.set_xlabel("blur sigma (vox), coarse -> sharp  |  open markers: + noise 0.05 / 0.10"); ax.set_title(TITLE[c], fontsize=10)
        if j == 0:
            ax.set_ylabel("edema Dice (mean, +- s.e.)"); ax.legend(fontsize=8)
        g = {a: d[f"{a}_blur0_dice"] - d[f"{a}_blur2_dice"] for a in ARMS}; nl = {a: d[f"{a}_blur0_dice"] - d[f"{a}_noise0.1_dice"] for a in ARMS}
        q = lambda s: f"{d[f'flat_blur{s:g}_dice'].mean():.2f} / {d[f'real_blur{s:g}_dice'].mean():.2f}"
        L.append(f"| {c} | {q(4)} | {q(2)} | {q(1)} | {q(0)} | {g['flat'].mean():+.3f} / {g['real'].mean():+.3f} | {(g['real'] > g['flat']).mean():.2f} | {wp(g['real'], g['flat']):.2g} | "
                 f"{nl['flat'].mean():+.3f} / {nl['real'].mean():+.3f} | {(nl['real'] > nl['flat']).mean():.2f} | {wp(nl['real'], nl['flat']):.2g} |")
    fig.suptitle("At which scale does each model find the edema? Dice as the test image is sharpened from blur sigma 4 to the original (70 BraTS cases per cell, 3-fold mean)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "dice_vs_scale.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    # strip figures: per cell, the 2 cases with the largest (real - flat) sharpening gain among cases where the real model reaches Dice >= 0.5, plus the median case
    sel = ["", "## Strip cases (selection rule: largest real - flat gain from sharpening sigma 2 -> 0, among cases with real Dice(sigma 0) >= 0.5; plus the cell's median case)", ""]
    for c in cells:
        d = df[df.cell == c].copy(); d["gdiff"] = (d.real_blur0_dice - d.real_blur2_dice) - (d.flat_blur0_dice - d.flat_blur2_dice)
        cand = d[d.real_blur0_dice >= 0.5].sort_values("gdiff", ascending=False); med = d.iloc[(d.gdiff - d.gdiff.median()).abs().argsort().iloc[0]]
        picks = list(dict.fromkeys(list(cand.case.head(2)) + [med.case]))
        sel.append(f"- {c}: " + ", ".join(f"{p} (gain diff {d[d.case == p].gdiff.iloc[0]:+.2f})" for p in picks))
        for case in picks:
            f = OUT / c / f"{case}_strip.npz"
            if not f.exists():
                continue
            zf = np.load(f); r = d[d.case == case].iloc[0]; g = zf["gt"]
            ys, xs_ = np.nonzero(g)
            if ys.size:
                y0, y1 = max(0, ys.min() - 25), min(g.shape[0], ys.max() + 26); x0, x1 = max(0, xs_.min() - 25), min(g.shape[1], xs_.max() + 26)
            else:
                y0, y1, x0, x1 = 0, g.shape[0], 0, g.shape[1]
            lv_show = [f"blur{s:g}" for s in SIGMAS] + ["noise0.1"]
            fig, axes = plt.subplots(2, len(lv_show), figsize=(2.6 * len(lv_show), 5.9), facecolor="#fcfcfb")
            for i, a in enumerate(ARMS):
                for jj, lv in enumerate(lv_show):
                    ax = axes[i, jj]; im = zf[f"img_{lv}"].astype(np.float32)[y0:y1, x0:x1]; p = zf[f"p_{a}_{lv}"].astype(np.float32)[y0:y1, x0:x1]
                    lo, hi = np.percentile(im, [1, 99.5]); ax.imshow(im.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
                    ax.imshow(np.ma.masked_less(p.T, 0.5), cmap="autumn_r" if a == "real" else "winter", vmin=0.5, vmax=1.0, alpha=0.55, origin="lower", interpolation="nearest")
                    ax.contour(g[y0:y1, x0:x1].T, [0.5], colors="#1ea672", linewidths=1.1, origin="lower")
                    dval = r[f"{a}_{lv}_dice"]
                    ttl = (f"sigma {lv[4:]}" if lv.startswith("blur") else "+ noise 0.10") + f"\nDice {dval:.2f}"
                    ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                    if jj == 0:
                        ax.set_ylabel(v2.ARM_LABEL[a], fontsize=9)
            fig.suptitle(f"{TITLE[c]}, {case}: edema predicted as the image is sharpened (left: blur sigma 4, right: original; last column = original + noise control). "
                         "Coloured = predicted edema (p > 0.5, fold mean), green = GT edema.", fontsize=9.5, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / f"strip_{c}_{case}.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    (OUT / "strip_stats.md").write_text("\n".join(L + sel)); print("\n".join(L + sel))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELL_KEYS)); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    for c in a.cells.split(","):
        run_cell(c, torch.device("cuda"), a.case_slice)


if __name__ == "__main__":
    main()
