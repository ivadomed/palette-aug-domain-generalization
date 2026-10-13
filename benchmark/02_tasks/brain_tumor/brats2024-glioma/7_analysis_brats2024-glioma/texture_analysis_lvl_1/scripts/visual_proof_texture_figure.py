#!/usr/bin/env python3
"""Visual proof, texture figure (CPU only, FINDINGS section 35): does the real-fill model rely more on fine-scale image detail than the
flat-fill model? No lesion-specificity, no per-cell coherence with the Dice step: one claim, every cell, from the arrays already extracted by
visual_proof_spectral.py (radial profiles, per-case fine-detail share) and visual_proof_hires3.py (per-case input-gradient slices + image).

Three outputs
  texture_figure.png/.pdf     committable (no patient image): (a) per-case scatter of the fine-detail share of the input gradient, flat (x) vs real (y),
                              one dot per case (fold-mean, median over structures), coloured by task; (b) log10(real / flat) radial power ratio of
                              the structure gradient vs frequency, one line per cell; (c) per-cell paired flat -> real fine share.
  texture_alignment.csv/.png  committable: co-location of the FINE component of the gradient with the FINE component of the image (Spearman over
                              body voxels of |hp(G)| vs |hp(img)|, hp = x - gaussian(x, 1)) per case and arm, from the hires3 slices. This is the
                              discriminator between "uses fine image detail" and "noisier gradients": real > flat in alignment means the extra
                              fine-scale sensitivity sits where the image's fine detail is.
  texture_examples_<cell>.png NOT committable (patient slices): zoomed crop, image | |hp(img)| | |G| flat | |G| real | |hp(G)| flat | |hp(G)| real.
  texture_figure.md           the numbers.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_proof_erf_tasks as vt  # noqa: E402

ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
SPEC = Path(os.environ.get("VP_SPECTRAL_OUT") or (vt.OUT.parent / "spectral")); HIRES = Path(os.environ.get("VP_HIRES3_OUT") or (vt.OUT.parent / "hires3"))
OUT = Path(os.environ.get("VP_TEXFIG_OUT") or (vt.OUT.parent / "texture_figure")); OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t2f", "glioma_t2w_t1n", "ms_flair_t2w", "ms_flair_t1w", "ms_t1w_flair", "ms_t1w_t2w", "breast_t1wce_t2w", "breast_t2w_t1wce",
         "brain_t1w_t2w", "brain_t2w_t1w", "abd_t1in_t2spir", "abd_t1in_ct", "abd_t2spir_t1in", "abd_t2spir_t1out", "abd_t2spir_ct", "mand_cbct_hanseg_ct", "mand_cbct_pddca_ct",
         "pelvis_mri_ct", "pelvis_ct_mri", "pancreas_t1wce_t2w", "pancreas_t2w_t1wce"]
EXAMPLES = ["glioma_t1n_t2w", "glioma_t2w_t2f", "breast_t2w_t1wce", "mand_cbct_hanseg_ct", "pancreas_t1wce_t2w", "pelvis_ct_mri"]
NB = 16; RMAX = 0.5 * np.sqrt(3); EDGES = np.linspace(0, RMAX, NB + 1); CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
FC = [f"fc_{k:02d}" for k in range(NB)]


def hp(x):
    return x - ndi.gaussian_filter(x, 1.0)


def body_of(img):
    return img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool)


def wtest(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 5 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d).pvalue)


def per_case_share(key):
    """fold-mean, then median over structures, per case and arm: g_fine and hf_c from the spectral metrics."""
    f = SPEC / key / "metrics.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype=str); d = d[d.case != "case"]; d[["g_fine", "hf_c"]] = d[["g_fine", "hf_c"]].apply(pd.to_numeric, errors="coerce")   # parallel chunks may repeat the header
    d = d.groupby(["case", "struct", "arm"])[["g_fine", "hf_c"]].mean().reset_index()
    return d.groupby(["case", "arm"])[["g_fine", "hf_c"]].median().reset_index().assign(cell=key, task=vt.CELLS[key]["task"])


def ratio_profile(key):
    """cell-level log10(real / flat) of the structure-gradient radial power, from radial.npz (per case x struct x arm, fold-mean): mean over cases of the per-case log ratio."""
    f = SPEC / key / "radial.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True); P = z["fc"]  # (rows, NB): structure-gradient radial profile per (case, struct, arm)
    d = pd.DataFrame({"case": z["case"], "struct": z["struct"], "arm": z["arm"]}); d[FC] = P
    d = d.groupby(["case", "arm"])[FC].mean().reset_index()
    r = d[d.arm == "real"].set_index("case")[FC]; fl = d[d.arm == "flat"].set_index("case")[FC]; cs = r.index.intersection(fl.index)
    lr = np.log10((r.loc[cs].values + 1e-12) / (fl.loc[cs].values + 1e-12))
    return lr.mean(0), np.std(lr, 0) / np.sqrt(max(1, len(cs))), len(cs)


def alignment(key):
    """per case and arm: Spearman over body voxels of |hp(G)| vs |hp(img)| on the saved hires3 slice; also with |G| unfiltered, as a control."""
    rows = []
    for f in sorted((HIRES / key).glob("*.npz")):
        z = np.load(f); img = z["img"].astype(np.float32); body = body_of(img); hi = np.abs(hp(img))[body]
        for a in ARMS:
            G = z[f"{a}_G"].astype(np.float32)
            rows.append(dict(cell=key, task=vt.CELLS[key]["task"], case=f.stem, arm=a,
                             align_hp=spearmanr(np.abs(hp(G))[body], hi).correlation, align_raw=spearmanr(np.abs(G)[body], hi).correlation))
    return pd.DataFrame(rows)


def examples(key, A):
    fs = {f.stem: f for f in (HIRES / key).glob("*.npz")}
    if not fs:
        return
    vol = {c: int((np.load(f)["gt"] == np.load(f)["c0"]).sum()) for c, f in fs.items()}; order = sorted(vol, key=vol.get)
    pick = [order[int(round(q * (len(order) - 1)))] for q in (0.5, 0.8)]
    cols = ["image", "|hp(image)|", "flat: |G|", "real: |G|", "flat: |hp(G)|", "real: |hp(G)|"]
    fig, axs = plt.subplots(len(pick), len(cols), figsize=(2.3 * len(cols), 2.5 * len(pick)), squeeze=False)
    for i, c in enumerate(pick):
        z = np.load(fs[c]); img = z["img"].astype(np.float32); g = z["gt"] == int(z["c0"]); body = body_of(img)
        ys, xs = np.where(g); pad = 24
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        sl = (slice(y0, y1), slice(x0, x1)); lo_, hi_ = np.percentile(img[body], [1, 99])
        G = {a: np.abs(z[f"{a}_G"].astype(np.float32)) / (np.abs(z[f"{a}_G"])[body].sum() + 1e-30) for a in ("flat", "real")}
        H = {a: np.abs(hp(z[f"{a}_G"].astype(np.float32))) / (np.abs(z[f"{a}_G"])[body].sum() + 1e-30) for a in ("flat", "real")}
        vg = np.percentile(np.concatenate([G[a][sl][body[sl]] for a in G]), 99); vh = np.percentile(np.concatenate([H[a][sl][body[sl]] for a in H]), 99)
        hi = np.abs(hp(img)); vi = np.percentile(hi[body], 99)
        al = A[(A.cell == key) & (A.case == c)].set_index("arm").align_hp
        for j, nm in enumerate(cols):
            ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.imshow(img[sl], cmap="gray", vmin=lo_, vmax=hi_); ax.set_ylabel(f"{c}\nstructure {int(z['c0'])}", fontsize=6)
            elif j == 1:
                ax.imshow(np.where(body, hi, np.nan)[sl], cmap="gray", vmin=0, vmax=vi)
            else:
                a = nm.split(":")[0]; M = G[a] if "hp" not in nm else H[a]; v = vg if "hp" not in nm else vh
                ax.imshow(np.where(body, M, np.nan)[sl], cmap="magma", vmin=0, vmax=v)
                if "hp" in nm:
                    ax.set_xlabel(f"align(|hp G|, |hp img|) = {al.get(a, np.nan):+.2f}", fontsize=6)
            ax.contour(g[sl], [0.5], colors="#00b4ff" if j == 0 else "white", linewidths=0.5)
            if i == 0:
                ax.set_title(nm, fontsize=7, color=PAL.get(nm.split(":")[0], "k"))
    fig.suptitle(f"{key}: input gradient of the structure's decision (fold-mean, unit mass, shared scale per row), full resolution, crop around the structure; hp = x - gaussian(x, 1). Contour = ground truth.", fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"texture_examples_{key}.png", dpi=170); plt.close(fig)


def main():
    cells = [k for k in ORDER if (SPEC / k / "metrics.csv").exists()]
    S = pd.concat([per_case_share(k) for k in cells], ignore_index=True)
    W = S.pivot_table(index=["cell", "task", "case"], columns="arm", values="g_fine").reset_index().dropna(subset=["flat", "real"])
    WH = S.pivot_table(index=["cell", "task", "case"], columns="arm", values="hf_c").reset_index().dropna(subset=["flat", "real"])
    R = {k: ratio_profile(k) for k in cells}
    A = pd.concat([alignment(k) for k in cells if (HIRES / k).exists()], ignore_index=True)
    A.to_csv(OUT / "texture_alignment.csv", index=False)
    # ---- per-cell table
    rows = []
    for k in cells:
        w = W[W.cell == k]; wh = WH[WH.cell == k]; a = A[A.cell == k].pivot_table(index="case", columns="arm", values="align_hp")
        rows.append(dict(cell=k, task=vt.CELLS[k]["task"], n=len(w), fine_flat=w.flat.median(), fine_real=w.real.median(), fine_noise=w.noise.median() if "noise" in w else np.nan,
                         fine_d=(w.real - w.flat).median(), fine_frac=(w.real > w.flat).mean(), fine_p=wtest(w.real - w.flat),
                         hf_flat=wh.flat.median(), hf_real=wh.real.median(), hf_d=(wh.real - wh.flat).median(), hf_frac=(wh.real > wh.flat).mean(), hf_p=wtest(wh.real - wh.flat),
                         al_flat=a.flat.median() if len(a) else np.nan, al_real=a.real.median() if len(a) else np.nan, al_noise=a.noise.median() if len(a) else np.nan,
                         al_d=(a.real - a.flat).median() if len(a) else np.nan, al_frac=(a.real > a.flat).mean() if len(a) else np.nan, al_p=wtest(a.real - a.flat) if len(a) else np.nan,
                         al_dn=(a.real - a.noise).median() if len(a) else np.nan, al_pn=wtest(a.real - a.noise) if len(a) else np.nan))
    T = pd.DataFrame(rows); T.to_csv(OUT / "texture_figure_cells.csv", index=False)
    # ---- figure: scatter + ratio spectra + paired per-cell
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axs[0]
    for t in vt.TASKS:
        w = W[W.task == t]
        if len(w):
            ax.scatter(w.flat, w.real, s=9, color=vt.TCOL[t], alpha=0.75, lw=0, label=f"{t} (n={len(w)})")
    lim = [min(W.flat.min(), W.real.min()) - 0.02, max(W.flat.max(), W.real.max()) + 0.02]; ax.plot(lim, lim, "k-", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("flat-fill model: fine-detail share of the input gradient", fontsize=8); ax.set_ylabel("real-fill model: fine-detail share", fontsize=8)
    ax.set_title(f"per case, all cells: real > flat in {(W.real > W.flat).mean() * 100:.0f} % of {len(W)} cases", fontsize=8.5); ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=7)
    ax = axs[1]
    for k in cells:
        if R[k] is None:
            continue
        m, se, n = R[k]; sel = CENTRES < 0.5
        ax.plot(CENTRES[sel], m[sel], color=vt.TCOL[vt.CELLS[k]["task"]], lw=1.0, alpha=0.85)
    ax.axhline(0, color="k", lw=0.7); ax.set_xlabel("radial frequency (cycles / voxel)", fontsize=8); ax.set_ylabel("log10 power ratio, real / flat (structure gradient)", fontsize=8)
    ax.set_title("radial power spectrum of the input gradient: real-fill / flat-fill, one line per cell (23)", fontsize=8.5); ax.tick_params(labelsize=7)
    ax = axs[2]
    for i, r in T.iterrows():
        col = vt.TCOL[r.task]; ax.plot([0, 1], [r.fine_flat, r.fine_real], "-", color=col, lw=0.9, alpha=0.8); ax.plot([0], [r.fine_flat], "o", color=col, ms=3.5); ax.plot([1], [r.fine_real], "o", color=col, ms=3.5)
        ax.annotate(r.cell, (1.03, r.fine_real), fontsize=4.5, va="center")
    ax.set_xlim(-0.3, 1.9); ax.set_xticks([0, 1]); ax.set_xticklabels(["flat fill", "real fill"], fontsize=8); ax.set_ylabel("fine-detail share, cell median", fontsize=8)
    ax.set_title(f"per cell: real > flat in {(T.fine_real > T.fine_flat).sum()} / {len(T)} cells", fontsize=8.5); ax.tick_params(labelsize=7)
    fig.suptitle("Real-fill models are more sensitive to fine-scale image detail than flat-fill models (input gradient dJ/dx of the structure's decision; fine-detail share = energy of G - gaussian(G, 1) / energy of G; "
                 "every held-out cell of every task, noise-fill arm omitted)", fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "texture_figure.png", dpi=180); fig.savefig(OUT / "texture_figure.pdf"); plt.close(fig)
    # ---- alignment figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axs[0]; P = A.pivot_table(index=["cell", "task", "case"], columns="arm", values="align_hp").reset_index()
    for t in vt.TASKS:
        p = P[P.task == t]
        if len(p):
            ax.scatter(p.flat, p.real, s=9, color=vt.TCOL[t], alpha=0.75, lw=0, label=f"{t} (n={len(p)})")
    lim = [P[["flat", "real"]].min().min() - 0.02, P[["flat", "real"]].max().max() + 0.02]; ax.plot(lim, lim, "k-", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("flat-fill: Spearman(|hp G|, |hp image|) over body voxels", fontsize=8); ax.set_ylabel("real-fill: same", fontsize=8); ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=7)
    ax.set_title(f"co-location of the gradient's fine component with the image's fine component: real > flat in {(P.real > P.flat).mean() * 100:.0f} % of {len(P)} cases", fontsize=7.5)
    ax = axs[1]
    for i, r in T.iterrows():
        col = vt.TCOL[r.task]; ax.plot([0, 1, 2], [r.al_flat, r.al_real, r.al_noise], "-", color=col, lw=0.9, alpha=0.8); ax.plot([0, 1, 2], [r.al_flat, r.al_real, r.al_noise], "o", color=col, ms=3.5)
        ax.annotate(r.cell, (2.05, r.al_noise), fontsize=4.5, va="center")
    ax.set_xlim(-0.3, 3.0); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["flat", "real", "noise"], fontsize=8); ax.set_ylabel("alignment, cell median", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title(f"per cell: real > flat in {(T.al_real > T.al_flat).sum()} / {T.al_real.notna().sum()} cells", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "texture_alignment.png", dpi=180); plt.close(fig)
    # ---- examples (patient slices: NOT committed)
    for k in EXAMPLES:
        if (HIRES / k).exists():
            examples(k, A)
    # ---- numbers
    L = ["# Texture figure (FINDINGS section 35): real-fill vs flat-fill fine-scale sensitivity of the input gradient, all cells", "",
         f"Per case = fold-mean, median over structures (spectral metrics). Fine-detail share: real > flat in {(W.real > W.flat).mean() * 100:.1f} % of {len(W)} cases; "
         f"per cell real > flat in {(T.fine_real > T.fine_flat).sum()} / {len(T)} cells, Wilcoxon p < 0.05 in {(T.fine_p < 0.05).sum()} cells.",
         f"HF share (r >= 0.25) of the structure gradient: real > flat in {(WH.real > WH.flat).mean() * 100:.1f} % of {len(WH)} cases; per cell {(T.hf_real > T.hf_flat).sum()} / {len(T)}, p < 0.05 in {(T.hf_p < 0.05).sum()}.",
         f"Alignment (Spearman over body voxels of |hp(G)| with |hp(img)|, hires3 slices): real > flat in {(P.real > P.flat).mean() * 100:.1f} % of {len(P)} cases; per cell "
         f"{(T.al_real > T.al_flat).sum()} / {T.al_real.notna().sum()}, p < 0.05 in {(T.al_p < 0.05).sum()}; real > noise per cell {(T.al_real > T.al_noise).sum()} / {T.al_real.notna().sum()}, p < 0.05 in {(T.al_pn < 0.05).sum()}.", "",
         "| cell | task | n | fine flat | fine real | d | frac>0 | p | HF flat | HF real | d | p | align flat | align real | align noise | d (real-flat) | frac>0 | p | d (real-noise) | p |", "|" + "---|" * 20]
    for _, r in T.iterrows():
        L.append(f"| {r.cell} | {r.task} | {r.n} | {r.fine_flat:.3f} | {r.fine_real:.3f} | {r.fine_d:+.3f} | {r.fine_frac:.2f} | {r.fine_p:.2g} | {r.hf_flat:.3f} | {r.hf_real:.3f} | {r.hf_d:+.3f} | {r.hf_p:.2g} | "
                 f"{r.al_flat:.3f} | {r.al_real:.3f} | {r.al_noise:.3f} | {r.al_d:+.3f} | {r.al_frac:.2f} | {r.al_p:.2g} | {r.al_dn:+.3f} | {r.al_pn:.2g} |")
    (OUT / "texture_figure.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:6])); print("TEXFIG_DONE")


if __name__ == "__main__":
    main()
