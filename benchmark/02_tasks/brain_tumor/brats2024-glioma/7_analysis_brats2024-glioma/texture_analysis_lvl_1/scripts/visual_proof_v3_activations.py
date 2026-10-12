#!/usr/bin/env python
"""
Visual proof v3 (2026-10-11): REPRESENTATION-level evidence on the same checkpoints as v2 (BraTS T2w-/T1n-trained, flat-fill rung 4.5 vs
real-fill rung 5, folds 0-2) and the same 70 BraTS test cases, edema. Three measurements, GPU pass per cell, then --plots on CPU:
 (1) first-layer kernel spectra per fold: fraction of each 3x3x3 kernel's energy at high spatial frequency (|f| >= 0.25 cycles/vox after
     zero-padding to 16^3), distribution over folds x channels, flat vs real (case-independent).
 (2) encoder feature maps at the full-res (stage 0, 32 ch) and half-res (stage 1, 64 ch) stages on one plan-sized patch centred on the GT
     edema: per channel Spearman correlation of |activation| with local high-pass energy (sigma 2) and with gradient magnitude over body
     voxels (subsampled); a channel is texture-selective if rho_tex > rho_edge + 0.05 and rho_tex > 0.10, edge-selective if the reverse;
     counts per model (fold-mean); activation variance inside the GT edema vs its border band (|d| <= 2 vox), channel-mean log ratio.
 (3) sensitivity to texture removal: the encoder sees the real patch, Gaussian-smoothed copies (sigma 1 and 2 mm = voxels here, body only)
     and a GT-FREE flat-filled copy (k-means k=6 on body intensities x random Voronoi cells, ~1 seed / 1500 vox, each cell filled with its
     mean, like PALETTE's partition); per encoder stage s: sum_GT ||f_s(x') - f_s(x)||^2 / sum_GT ||f_s(x)||^2 (GT max-pooled to the stage
     grid), flat vs real arm, fold mean.
GT enters only for patch placement and as the region of the measurements (no image edit uses it). Visual (--plots): for the cases listed in
SHOW, the top-4 texture-selective channels of each arm as full-res activation maps side by side; per-cell sensitivity curves; kernel-spectrum
and selectivity-count distributions. Output: outputs/plots/visual_proof_v3_activations/ (<cell>/<case>_feat.npz, <cell>/cases.csv,
<cell>/channels.csv, kernels_<arm>_<cell>.json, plots).
GPU: .venv/bin/python visual_proof_v3_activations.py --cells bratsfull_t2w_t1n     CPU: --plots
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
os.environ.setdefault("VP_OUT", str(THIS.parent / "outputs/plots/visual_proof_v2"))   # v2's OUT is irrelevant here, keep it default
import visual_proof_v2 as v2  # noqa: E402  (load_predictor, preprocess, CELLS, FOLDS)

OUT = Path(os.environ.get("VP3_OUT") or (THIS.parent / "outputs/plots/visual_proof_v3_activations")); OUT.mkdir(parents=True, exist_ok=True)
CELL_KEYS = ["bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w"]
SHOW = {"bratsfull_t2w_t1n": ["BraTSGLI02335101", "BraTSGLI00008100"], "bratsfull_t2w_t2f": ["BraTSGLI02205101", "BraTSGLI02292100"],
        "bratsfull_t1n_t2f": ["BraTSGLI02205101"], "bratsfull_t1n_t2w": ["BraTSGLI02335101"]}
ARMS = ("flat", "real"); STAGES_FEAT = (0, 1); ARM_LABEL = v2.ARM_LABEL


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kernel_hf_fraction(w: np.ndarray, pad=16, fcut=0.25):
    """w: (C_out, C_in, 3, 3, 3). Fraction of spectral energy at radial frequency >= fcut cycles/vox (zero-padded FFT)."""
    C = w.shape[0]; k = np.zeros((C, pad, pad, pad), np.float32); k[:, :3, :3, :3] = w[:, 0]
    P = np.abs(np.fft.fftn(k, axes=(1, 2, 3))) ** 2; f = np.fft.fftfreq(pad); r = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
    hf = P[:, r >= fcut].sum(1); tot = P.reshape(C, -1).sum(1)
    return hf / np.maximum(tot, 1e-12)


def partition_flat_fill(img, body, k=6, seed=0, vox_per_seed=1500):
    """GT-free PALETTE-like flat fill: k-means (k) on body intensities x random Voronoi cells; each (cluster, cell) region -> its mean."""
    rng = np.random.default_rng(seed); v = img[body]
    cent = np.quantile(v, np.linspace(0.05, 0.95, k))
    for _ in range(15):
        lab = np.abs(v[:, None] - cent[None]).argmin(1)
        cent = np.array([v[lab == j].mean() if (lab == j).any() else cent[j] for j in range(k)])
    km = np.zeros(img.shape, np.int32); km[body] = np.abs(v[:, None] - cent[None]).argmin(1) + 1
    n_seed = max(8, int(body.sum() / vox_per_seed)); idx = np.argwhere(body); seeds = idx[rng.choice(len(idx), n_seed, replace=False)]
    seedvol = np.ones(img.shape, bool); seedvol[tuple(seeds.T)] = False
    _, ind = ndi.distance_transform_edt(seedvol, return_indices=True); vor = np.ravel_multi_index(tuple(ind), img.shape)
    _, vor = np.unique(vor, return_inverse=True); vor = vor.reshape(img.shape)
    region = (km.astype(np.int64) * (vor.max() + 1) + vor); region[~body] = -1
    out = img.copy(); ids, inv = np.unique(region[body], return_inverse=True); means = np.bincount(inv, weights=v) / np.bincount(inv)
    out[body] = means[inv]; return out


def spearman_rows(X: np.ndarray, y: np.ndarray):
    """Spearman rho of every row of X (C, N) with y (N,), vectorised (ranks via argsort, Pearson on ranks)."""
    rx = np.argsort(np.argsort(X, axis=1), axis=1).astype(np.float32); ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean()
    return (rx @ ry) / (np.sqrt((rx ** 2).sum(1)) * np.sqrt((ry ** 2).sum()) + 1e-12)


def to_grid(a, shape, order):
    return a if a.shape == tuple(shape) else ndi.zoom(a.astype(np.float32), np.array(shape) / np.array(a.shape), order=order)


def patch_at(gt_t, size, shape):
    c = np.argwhere(gt_t).mean(0).astype(int) if gt_t.any() else np.array(shape) // 2
    size = np.minimum(np.array(size), np.array(shape)); lo = np.clip(c - size // 2, 0, np.array(shape) - size)
    return tuple(slice(int(a), int(a + s)) for a, s in zip(lo, size))


@torch.no_grad()
def encoder_feats(net, x: torch.Tensor):
    f = net.encoder(x)
    return f if isinstance(f, (list, tuple)) else [f]


def run_cell(key, dev, cases=None):
    cell = v2.CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True); lids = list(cell["lids"])
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]
    # (1) kernel spectra, case independent
    for a, pr in prs.items():
        rows = []
        for k in v2.FOLDS:
            pr.network.load_state_dict(pr.list_of_parameters[k]); w = next(m for m in pr.network.encoder.stages[0].modules() if isinstance(m, torch.nn.Conv3d)).weight.detach().cpu().numpy()   # first 3x3x3 conv (stage 0 is a Sequential)
            hf = kernel_hf_fraction(w); rows += [dict(arm=a, fold=k, channel=c, hf_fraction=float(hf[c]), l2=float(np.sqrt((w[c] ** 2).sum()))) for c in range(w.shape[0])]
        pd.DataFrame(rows).to_csv(od / f"kernels_{a}.csv", index=False)
        log(f"[{key}] {a}: first-layer HF fraction median over folds x channels {np.median([r['hf_fraction'] for r in rows]):.3f}")
    ps = pr0.configuration_manager.patch_size
    done = set(pd.read_csv(od / "cases.csv").case) if (od / "cases.csv").exists() and (od / "cases.csv").stat().st_size > 0 else set()
    for case in (cases or cell["cases"]):
        if case in done:
            log(f"[{key}] {case}: already in cases.csv, skipped (resume)"); continue
        crows, chrows = [], []
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; gt_t = np.isin(gt, lids)
        if gt_t.sum() < 50:
            log(f"[{key}] {case}: GT edema < 50 vox, skipped"); continue
        body = img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool)
        box = patch_at(gt_t, ps, img.shape); ip = img[box]; bp = body[box]; gp = gt_t[box]
        # image-side reference maps (patch)
        L = ndi.gaussian_filter(ip, 2.0); hp = ip - L; tex = ndi.gaussian_filter(hp ** 2, 2.0); grad = np.sqrt(sum(ndi.sobel(ip, axis=i) ** 2 for i in range(3)))
        din = ndi.distance_transform_edt(gp); dout = ndi.distance_transform_edt(~gp); border = ((din <= 2) & gp) | ((dout <= 2) & ~gp); inner = gp & (din > 2)
        sm1 = np.where(bp, ndi.gaussian_filter(ip, 1.0), ip); sm2 = np.where(bp, ndi.gaussian_filter(ip, 2.0), ip); ff = partition_flat_fill(ip, bp)
        inputs = {"real": ip, "smooth1": sm1, "smooth2": sm2, "flatfill": ff}
        ref_s = {}; feats_show = {}
        for a, pr in prs.items():
            net = pr.network; sens = {}; sel = {s: [] for s in STAGES_FEAT}; varr = {s: [] for s in STAGES_FEAT}
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                F = {n: encoder_feats(net, torch.from_numpy(v)[None, None].to(dev).float()) for n, v in inputs.items()}
                for s_i, fs in enumerate(F["real"]):
                    g_s = torch.from_numpy(gp.astype(np.float32))[None, None].to(dev)
                    g_s = torch.nn.functional.adaptive_max_pool3d(g_s, fs.shape[2:])[0, 0] > 0.5 if fs.shape[2:] != gp.shape else torch.from_numpy(gp).to(dev)
                    base = (fs[0] ** 2).sum(0)[g_s].sum().item()
                    for n in ("smooth1", "smooth2", "flatfill"):
                        d = ((F[n][s_i][0] - fs[0]) ** 2).sum(0)[g_s].sum().item(); sens.setdefault((n, s_i), []).append(d / max(base, 1e-12))
                for s in STAGES_FEAT:
                    fs = F["real"][s][0].abs().cpu().numpy()   # (C, z, y, x) at the stage grid; reference maps go DOWN to that grid
                    if s not in ref_s:
                        shp = fs.shape[1:]; bs = to_grid(bp.astype(np.float32), shp, 1) > 0.5
                        subs = np.flatnonzero(bs); subs = np.random.default_rng(0).choice(subs, min(40000, subs.size), replace=False)
                        ref_s[s] = dict(sub=subs, t=to_grid(tex, shp, 1).ravel()[subs], g=to_grid(grad, shp, 1).ravel()[subs],
                                        inner=to_grid(inner.astype(np.float32), shp, 0) > 0.5, border=to_grid(border.astype(np.float32), shp, 0) > 0.5)
                    R = ref_s[s]; X = fs.reshape(fs.shape[0], -1)[:, R["sub"]]
                    rts = spearman_rows(X, R["t"]); rgs = spearman_rows(X, R["g"])
                    vis = fs[:, R["inner"]].var(1) if R["inner"].any() else np.full(fs.shape[0], np.nan); vbs = fs[:, R["border"]].var(1) if R["border"].any() else np.full(fs.shape[0], np.nan)
                    for c in range(fs.shape[0]):
                        rt, rg, vi, vb = float(rts[c]), float(rgs[c]), float(vis[c]), float(vbs[c])
                        sel[s].append((k, c, rt, rg)); varr[s].append((k, c, vi, vb))
                        chrows.append(dict(cell=key, case=case, arm=a, stage=s, fold=k, channel=c, rho_tex=rt, rho_edge=rg, var_inner=vi, var_border=vb,
                                           texture_sel=bool(rt > rg + 0.05 and rt > 0.10), edge_sel=bool(rg > rt + 0.05 and rg > 0.10)))
                    if k == 0 and case in SHOW.get(key, []):
                        feats_show[(a, s)] = fs.astype(np.float16)
            row = dict(cell=key, case=case, arm=a, gt_vox=int(gt_t.sum()), patch_gt_vox=int(gp.sum()))
            for (n, s_i), vals in sens.items():
                row[f"sens_{n}_s{s_i}"] = float(np.mean(vals))
            for s in STAGES_FEAT:
                arr = np.array([(rt, rg) for _, _, rt, rg in sel[s]], float); row[f"n_texture_sel_s{s}"] = float(((arr[:, 0] > arr[:, 1] + 0.05) & (arr[:, 0] > 0.10)).sum() / 3)
                row[f"n_edge_sel_s{s}"] = float(((arr[:, 1] > arr[:, 0] + 0.05) & (arr[:, 1] > 0.10)).sum() / 3); row[f"rho_tex_mean_s{s}"] = float(np.nanmean(arr[:, 0])); row[f"rho_edge_mean_s{s}"] = float(np.nanmean(arr[:, 1]))
                va = np.array([(vi, vb) for _, _, vi, vb in varr[s]], float); row[f"log_var_inner_over_border_s{s}"] = float(np.nanmean(np.log((va[:, 0] + 1e-12) / (va[:, 1] + 1e-12))))
            crows.append(row)
        if case in SHOW.get(key, []):
            np.savez_compressed(od / f"{case}_feat.npz", img=ip.astype(np.float16), gt=gp, flatfill=ff.astype(np.float16), smooth2=sm2.astype(np.float16),
                                **{f"feat_{a}_s{s}": v for (a, s), v in feats_show.items()})
        r_ = {a: [r for r in crows if r["case"] == case and r["arm"] == a][0] for a in ARMS}
        log(f"[{key}] {case}: texture-sel ch s0 flat {r_['flat']['n_texture_sel_s0']:.1f} real {r_['real']['n_texture_sel_s0']:.1f}; flatfill sensitivity s0 flat {r_['flat']['sens_flatfill_s0']:.3f} real {r_['real']['sens_flatfill_s0']:.3f}; {time.time() - t0:.0f} s")
        for name, rows_ in (("cases.csv", crows), ("channels.csv", chrows)):   # append per case: resumable + chunk-safe
            f = od / name; pd.DataFrame(rows_).to_csv(f, mode="a", index=False, header=not (f.exists() and f.stat().st_size > 0))
        done.add(case)


def read_csv_clean(f):
    """Chunked jobs append to the same CSV; two chunks can both write the header -> drop repeated header rows, coerce numerics, de-duplicate."""
    d = pd.read_csv(f, dtype=str); d = d[d.iloc[:, 0] != d.columns[0]]
    for c in d.columns:
        if c not in ("cell", "case", "arm"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    keys = [k for k in ("cell", "case", "arm", "stage", "fold", "channel") if k in d.columns]
    return d.drop_duplicates(subset=keys).reset_index(drop=True)


def plots():
    cells = [c for c in CELL_KEYS if (OUT / c / "cases.csv").exists()]
    if not cells:
        log("nothing to plot"); return
    cf = pd.concat([read_csv_clean(OUT / c / "cases.csv") for c in cells], ignore_index=True); cf.to_csv(OUT / "cases_all.csv", index=False)
    lines = ["# Visual proof v3: representation-level measurements, flat-fill (rung 4.5) vs real-fill (rung 5), BraTS 70 test cases, edema", ""]
    # (1) kernels
    lines += ["## (1) First-layer 3x3x3 kernel spectra: fraction of energy at |f| >= 0.25 cycles/vox (folds x 32 channels = 96 per arm)", "", "| cell | flat median [IQR] | real median [IQR] | Mann-Whitney p |", "|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.8), facecolor="#fcfcfb", squeeze=False)
    for j, c in enumerate(cells):
        kf = {a: read_csv_clean(OUT / c / f"kernels_{a}.csv") for a in ARMS}; q = lambda x: f"{x.median():.3f} [{x.quantile(.25):.3f}-{x.quantile(.75):.3f}]"
        p = mannwhitneyu(kf["flat"].hf_fraction, kf["real"].hf_fraction).pvalue
        lines.append(f"| {c} | {q(kf['flat'].hf_fraction)} | {q(kf['real'].hf_fraction)} | {p:.2g} |")
        ax = axes[0, j]; ax.hist([kf["flat"].hf_fraction, kf["real"].hf_fraction], bins=20, color=["#2a78d6", "#eb6834"], alpha=0.6, label=["flat-fill", "real-fill"]); ax.set_title(f"{c}\nfirst-layer kernel HF fraction, p={p:.2g}", fontsize=9); ax.set_xlabel("HF energy fraction"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "kernel_spectra.png", dpi=120, facecolor="#fcfcfb"); plt.close(fig); lines.append("")
    # (2) selectivity + variance
    lines += ["## (2) Encoder channels: texture- vs edge-selective counts (fold-mean per case), mean rho, log(var inside edema / var border band)", "",
              "| cell | stage | n cases | texture-sel flat (median) | real | real > flat | p | edge-sel flat | real | p | log var ratio flat | real | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for s in STAGES_FEAT:
            d = cf[cf.cell == c].pivot(index="case", columns="arm", values=[f"n_texture_sel_s{s}", f"n_edge_sel_s{s}", f"log_var_inner_over_border_s{s}"]).dropna()
            if len(d) == 0:
                continue
            t = d[f"n_texture_sel_s{s}"]; e = d[f"n_edge_sel_s{s}"]; v = d[f"log_var_inner_over_border_s{s}"]
            wp = lambda x: wilcoxon(x["real"], x["flat"]).pvalue if len(x) >= 5 and (x["real"] != x["flat"]).any() else np.nan
            lines.append(f"| {c} | s{s} | {len(d)} | {t['flat'].median():.1f} | {t['real'].median():.1f} | {(t['real'] > t['flat']).mean():.2f} | {wp(t):.2g} | {e['flat'].median():.1f} | {e['real'].median():.1f} | {wp(e):.2g} | {v['flat'].median():.2f} | {v['real'].median():.2f} | {wp(v):.2g} |")
    lines.append("")
    # (3) sensitivity curves
    lines += ["## (3) Sensitivity of encoder features inside the GT edema to texture removal: sum_GT ||f(x') - f(x)||^2 / sum_GT ||f(x)||^2, per stage (fold mean, median over cases)", "",
              "| cell | perturbation | stage | flat | real | real > flat | p |", "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(3, len(cells), figsize=(4.4 * len(cells), 10), facecolor="#fcfcfb", squeeze=False)
    for j, c in enumerate(cells):
        d = cf[cf.cell == c]
        for i, n in enumerate(("smooth1", "smooth2", "flatfill")):
            ax = axes[i, j]
            for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
                da = d[d.arm == a]; cols = [k for k in da.columns if k.startswith(f"sens_{n}_s")]; cols = sorted(cols, key=lambda k: int(k.split("_s")[-1]))
                med = da[cols].median(); q1 = da[cols].quantile(.25); q3 = da[cols].quantile(.75); xs = range(len(cols))
                ax.plot(xs, med.values, "-o", color=col, label=ARM_LABEL[a]); ax.fill_between(xs, q1.values, q3.values, color=col, alpha=0.15)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"s{k}" for k in range(len(cols))]); ax.set_title(f"{c}: {n}", fontsize=9); ax.set_ylabel("relative feature change in GT edema" if j == 0 else "")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            for k_, col_ in enumerate(cols):
                dd = d.pivot(index="case", columns="arm", values=col_).dropna(); p = wilcoxon(dd["real"], dd["flat"]).pvalue if len(dd) >= 5 else np.nan
                lines.append(f"| {c} | {n} | s{k_} | {dd['flat'].median():.3f} | {dd['real'].median():.3f} | {(dd['real'] > dd['flat']).mean():.2f} | {p:.2g} |")
    fig.suptitle("Encoder sensitivity to texture removal inside the GT edema (median, IQR over cases); rows: Gaussian sigma 1, sigma 2, GT-free flat fill", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "sensitivity_curves.png", dpi=120, facecolor="#fcfcfb"); plt.close(fig); lines.append("")
    # visual: top-4 texture-selective channels per arm
    for c in cells:
        ch = read_csv_clean(OUT / c / "channels.csv") if (OUT / c / "channels.csv").exists() else None
        for case in SHOW.get(c, []):
            f = OUT / c / f"{case}_feat.npz"
            if not f.exists() or ch is None:
                continue
            z = np.load(f); img = z["img"].astype(np.float32); gp = z["gt"]; zs = int(np.argmax(gp.sum((1, 2))))
            fig, axes = plt.subplots(2, 6, figsize=(26, 9), facecolor="#fcfcfb")
            for r, a in enumerate(ARMS):
                d = ch[(ch.cell == c) & (ch.case == case) & (ch.arm == a) & (ch.stage == 0) & (ch.fold == 0)].copy()
                if len(d) == 0:
                    continue
                d["margin"] = d.rho_tex - d.rho_edge; top = d.sort_values("margin", ascending=False).head(4)
                fe = z[f"feat_{a}_s0"].astype(np.float32)
                axes[r, 0].imshow(img[zs].T, cmap="gray", origin="lower"); axes[r, 0].contour(gp[zs].T.astype(float), levels=[0.5], colors=["#1baf7a"]); axes[r, 0].set_title(f"{case} z={zs}: image, GT edema", fontsize=9, loc="left"); axes[r, 0].axis("off")
                axes[r, 1].imshow(z["flatfill"].astype(np.float32)[zs].T, cmap="gray", origin="lower"); axes[r, 1].set_title("GT-free flat fill (k-means x Voronoi)", fontsize=9, loc="left"); axes[r, 1].axis("off")
                for jj, (_, t) in enumerate(top.iterrows()):
                    ax = axes[r, 2 + jj]; m = fe[int(t.channel)][zs]; ax.imshow(m.T, cmap="inferno", vmin=0, vmax=np.percentile(m, 99.5) or 1, origin="lower"); ax.contour(gp[zs].T.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=0.8)
                    ax.set_title(f"{ARM_LABEL[a]}: ch {int(t.channel)}\nrho_tex {t.rho_tex:.2f}, rho_edge {t.rho_edge:.2f}", fontsize=9, loc="left"); ax.axis("off")
            fig.suptitle(f"{c} {case}: top-4 texture-selective full-res encoder channels (fold 0) per arm; |activation|", fontsize=10.5, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / f"{c}_{case}_texture_channels.png", dpi=100, facecolor="#fcfcfb"); plt.close(fig)
    (OUT / "v3_stats.md").write_text("\n".join(lines)); print("\n".join(lines)); log("wrote v3_stats.md + plots")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELL_KEYS)); ap.add_argument("--plots", action="store_true"); ap.add_argument("--case-slice", default=None, help="i:j -> only cases[i:j] of each cell"); a = ap.parse_args()
    if a.plots:
        plots(); return
    dev = torch.device("cuda")
    for key in a.cells.split(","):
        try:
            cases = None
            if a.case_slice:
                i_, j_ = (int(x) if x else None for x in a.case_slice.split(":")); cases = v2.CELLS[key]["cases"][i_:j_]
            run_cell(key, dev, cases)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"[{key}] FAILED: {e}")


if __name__ == "__main__":
    main()
