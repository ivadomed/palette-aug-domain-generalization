#!/usr/bin/env python
"""
Visual proof v8 diagnostics (2026-10-12, CPU): WHY the real-fill model's evidence is more local, from the stored in-plane |gradient| slices of
visual_proof_erf_tasks.py (target seeds <case>_erf.npz, matched background controls <case>_ctrl.npz). No new forward pass.
Per explained voxel (in-plane slice through it, body = image != 0), per model (flat-fill rung 4.5, real-fill rung 5):
  1. ERF shape vs magnitude: r50 / r90 (radius holding 50 / 90 % of the in-plane evidence), absolute mean |G| in the r_loc disc (centre),
     in 5 < r <= 15 (mid) and 15 < r <= 40 (far). A share can rise because the centre grows or because the periphery shrinks.
  2. Where the evidence comes from (answers "how is background predicted"): disjoint shares local (r <= r_loc) / inside the target structure /
     its rim (+-2 px of the boundary) / other background, and each one's EXCESS = observed / expected under an isotropic map with the same
     radial profile (1 = the region gets what its distance alone predicts).
  3. Image covariates around the voxel (disc r <= r_loc + 2): fine texture energy (RMS of img - Gaussian(img, 1 px)) and coarse edge strength
     (Gaussian gradient magnitude, sigma 2 px), each relative to the slice's body median.
  4. What the evidence follows: Spearman, over pixels in r_loc < r <= 15, between the radially de-trended evidence (|G| / ring mean) and the
     image's fine texture |hp| or coarse edge magnitude ("does the model look at fine texture or at edges around the voxel").
Outputs (VP8_OUT/diag/): diag_voxels.csv, diag_summary.md, diag_radial.png, diag_sources.png, diag_texture.png, diag_alignment.png.
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

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402

OUT = vt.OUT; D_OUT = OUT / "diag"; D_OUT.mkdir(exist_ok=True)
ARMS = vt.ARMS; RB = 40; RIM = 2; R_ALIGN = 15
KINDS = (("target", "erf", "seeds.csv"), ("control", "ctrl", "seeds_ctrl.csv"))


def pw(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan


def voxel_metrics(img, gt, Gd, v, r_loc):
    body = img != 0; yy, xx = np.indices(img.shape); r = np.hypot(yy - v[1], xx - v[2]); rb = np.minimum(np.round(r).astype(int), RB)
    reg = gt & body; din = ndi.distance_transform_edt(reg); dout = ndi.distance_transform_edt(~reg)
    rim = body & (((din <= RIM) & reg) | ((dout <= RIM) & ~reg)) if reg.any() else np.zeros_like(body)
    loc = body & (r <= r_loc); cat = {"local": loc, "inside": reg & ~rim & ~loc, "rim": rim & ~loc, "other": body & ~reg & ~rim & ~loc}
    hp = img - ndi.gaussian_filter(img, 1.0); em = ndi.gaussian_gradient_magnitude(img, 2.0)
    near = body & (r <= r_loc + 2); bmed_t = np.median(np.abs(hp[body])) + 1e-6; bmed_e = np.median(em[body]) + 1e-6
    row = dict(tex=float(np.sqrt(np.mean(hp[near] ** 2)) / bmed_t) if near.any() else np.nan, edge=float(em[near].mean() / bmed_e) if near.any() else np.nan,
               reg_px=int(reg.sum()))
    ann = body & (r > r_loc) & (r <= R_ALIGN)
    for a, G in Gd.items():
        G = G.astype(np.float64) * body; tot = G.sum() + 1e-30
        num = np.bincount(rb[body], G[body], minlength=RB + 1); den = np.bincount(rb[body], minlength=RB + 1); ring = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        # in-plane mass within radius (rings up to RB; the rest is beyond RB)
        cum = np.cumsum(num) / tot
        row[f"{a}_r50"] = float(np.searchsorted(cum, 0.5)) if cum[-2] >= 0.5 else np.nan
        row[f"{a}_r90"] = float(np.searchsorted(cum, 0.9)) if cum[-2] >= 0.9 else np.nan
        row[f"{a}_g_c"] = float(G[loc].mean()) if loc.any() else np.nan
        mid, far = body & (r > 5) & (r <= 15), body & (r > 15) & (r <= 40)
        row[f"{a}_g_mid"] = float(G[mid].mean()) if mid.any() else np.nan
        row[f"{a}_g_far"] = float(G[far].mean()) if far.any() else np.nan
        row[f"{a}_zero"] = float((G[body] == 0).mean())
        exp = np.nan_to_num(ring[rb]) * body; etot = exp.sum() + 1e-30
        for k, m in cat.items():
            row[f"{a}_sh_{k}"] = float(G[m].sum() / tot)
            row[f"{a}_ex_{k}"] = float((G[m].sum() / tot) / (exp[m].sum() / etot)) if exp[m].sum() > 0 else np.nan
        if ann.sum() > 30:
            q = G[ann] / np.maximum(ring[rb[ann]], 1e-30)
            row[f"{a}_al_tex"] = float(spearmanr(q, np.abs(hp[ann])).correlation); row[f"{a}_al_edge"] = float(spearmanr(q, em[ann]).correlation)
        else:
            row[f"{a}_al_tex"] = row[f"{a}_al_edge"] = np.nan
        row[f"{a}_prof"] = ring[:RB]
    return row


def collect():
    rows, prof = [], []
    for cell in vt.CELLS:
        for kind, tag, csv in KINDS:
            f = OUT / cell / csv
            if not f.exists():
                continue
            S = vt.e7.num(pd.read_csv(f)); S["case"] = S.case.astype(str); r_loc = vt.CELLS[cell]["r_loc"]
            for case, g in S.groupby("case"):
                fz = OUT / cell / f"{case}_{tag}.npz"
                if not fz.exists():
                    continue
                z = np.load(fz)
                for rr in g.itertuples():
                    i = int(rr.seed); v = z["seeds"][i]
                    m = voxel_metrics(z["img"][i].astype(np.float64), z["gt"][i], {a: z[f"G_{a}"][i] for a in ARMS}, v, r_loc)
                    for a in ARMS:
                        prof.append(dict(cell=cell, kind=kind, case=case, seed=i, arm=a, **{f"p{k}": x for k, x in enumerate(m.pop(f"{a}_prof"))}))
                    rows.append(dict(cell=cell, task=vt.CELLS[cell]["task"], kind=kind, case=case, seed=i, pair=int(getattr(rr, "pair", i)), cls=int(rr.cls),
                                     agree=abs(rr.dice_real - rr.dice_flat) <= vt.AGREE, loc3d_flat=rr.flat_local_share, loc3d_real=rr.real_local_share, **m))
            vt.log(f"[diag] {cell} {kind}: {len(S)} voxels")
    V = pd.DataFrame(rows); P = pd.DataFrame(prof)
    V.to_csv(D_OUT / "diag_voxels.csv", index=False); P.to_csv(D_OUT / "diag_profiles.csv", index=False)
    return V, P


def case_med(V, col):
    return V.groupby(["cell", "kind", "case"])[col].median()


def report(V, P):
    cells = [c for c in vt.CELLS if c in set(V.cell)]
    title = lambda c: f"{vt.CELLS[c]['task']}: " + c.split("_", 1)[1].replace("_", " -> ", 1).replace("_", " ")   # noqa: E731
    V = V.copy()
    for m in ("r50", "r90", "g_c", "g_mid", "g_far"):
        V[f"lr_{m}"] = np.log2(V[f"real_{m}"] / V[f"flat_{m}"])
    for m in ("sh_local", "sh_inside", "sh_rim", "sh_other", "ex_inside", "ex_rim", "al_tex", "al_edge"):
        V[f"d_{m}"] = V[f"real_{m}"] - V[f"flat_{m}"]
    V["d_loc3d"] = V.loc3d_real - V.loc3d_flat; V["rel_loc3d"] = V.d_loc3d / V.loc3d_flat
    L = ["# Visual proof v8 diagnostics: why is the real-fill evidence more local? (in-plane, all structures; per case = median over its voxels, "
         "then median over cases; p = Wilcoxon over cases, real vs flat)", "",
         f"float16 storage check: median share of exactly-zero |G| pixels in the body: flat {V.flat_zero.median():.3f}, real {V.real_zero.median():.3f}.", "",
         "## 1. Shape vs magnitude (log2 real/flat; < 0 for r50 = real-fill evidence is more compact)", "",
         "| cell | kind | n | local share 3D flat | real | rel. change | p | log2 r50 | p | log2 centre |G| | p | log2 mid |G| | p | log2 far |G| | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for k in ("target", "control"):
            v = V[(V.cell == c) & (V.kind == k)]
            if not len(v):
                continue
            cm = v.groupby("case").median(numeric_only=True)
            L.append(f"| {title(c)} | {k} | {len(cm)} | {cm.loc3d_flat.median():.4f} | {cm.loc3d_real.median():.4f} | {cm.rel_loc3d.median() * 100:+.0f}% | {pw(cm.d_loc3d):.1g} | "
                     + " | ".join(f"{cm[f'lr_{m}'].median():+.2f} | {pw(cm[f'lr_{m}']):.1g}" for m in ("r50", "g_c", "g_mid", "g_far")) + " |")
    L += ["", "## 2. Where the evidence comes from (shares, flat -> real; excess = observed / isotropic expectation)", "",
          "| cell | kind | n | local | inside target | rim | other bg | inside excess | rim excess | p (inside+rim share, real vs flat) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for k in ("target", "control"):
            v = V[(V.cell == c) & (V.kind == k)]
            if not len(v):
                continue
            cm = v.groupby("case").median(numeric_only=True)
            f = lambda m: f"{cm[f'flat_{m}'].median():.3f} -> {cm[f'real_{m}'].median():.3f}"   # noqa: E731
            L.append(f"| {title(c)} | {k} | {len(cm)} | {f('sh_local')} | {f('sh_inside')} | {f('sh_rim')} | {f('sh_other')} | {f('ex_inside')} | {f('ex_rim')} | "
                     f"{pw((cm.real_sh_inside + cm.real_sh_rim) - (cm.flat_sh_inside + cm.flat_sh_rim)):.1g} |")
    L += ["", "## 3. Does the locality shift follow the local image texture? (voxels of both kinds pooled within a cell)", "",
          "OLS per cell: d_local3d ~ 1 + is_target + z(log texture) + z(log edge); 95% CI by case bootstrap (1000).", "",
          "| cell | n vox | texture target / control (median, rel. to body) | edge target / control | b texture | CI | b edge | CI | b is_target | CI | Spearman(d, texture) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    coefs = {}
    rng = np.random.default_rng(0)
    for c in cells:
        v = V[(V.cell == c)].dropna(subset=["d_loc3d", "tex", "edge"]).copy()
        if v.kind.nunique() < 2 or v.case.nunique() < 5:
            continue
        v["zlt"] = np.log(v.tex.clip(1e-3)); v["zle"] = np.log(v.edge.clip(1e-3)); v["zlt"] = (v.zlt - v.zlt.mean()) / v.zlt.std(); v["zle"] = (v.zle - v.zle.mean()) / v.zle.std()
        def fit(w):
            X = np.c_[np.ones(len(w)), (w.kind == "target").astype(float), w.zlt, w.zle]; return np.linalg.lstsq(X, w.d_loc3d.values * 100, rcond=None)[0]
        b = fit(v); cs = v.case.unique(); bs = []; grp = {p: w for p, w in v.groupby("case")}
        for _ in range(1000):
            pick = rng.choice(cs, len(cs)); bs.append(fit(pd.concat([grp[p] for p in pick])))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); coefs[c] = (b, lo, hi)
        tt, tc = v[v.kind == "target"], v[v.kind == "control"]
        L.append(f"| {title(c)} | {len(v)} | {tt.tex.median():.2f} / {tc.tex.median():.2f} | {tt.edge.median():.2f} / {tc.edge.median():.2f} | "
                 f"{b[2]:+.2f} | [{lo[2]:+.2f}, {hi[2]:+.2f}] | {b[3]:+.2f} | [{lo[3]:+.2f}, {hi[3]:+.2f}] | {b[1]:+.2f} | [{lo[1]:+.2f}, {hi[1]:+.2f}] | "
                 f"{spearmanr(v.d_loc3d, v.tex).correlation:+.2f} |")
    L += ["", "(coefficients in pp of 3D local share per SD of log texture / log edge; b is_target = target-vs-control difference left after the covariates)", "",
          "## 4. What the de-trended evidence follows around the voxel (Spearman with fine texture |hp| and coarse edge magnitude; flat -> real)", "",
          "| cell | kind | n | texture alignment | p | edge alignment | p |", "|---|---|---|---|---|---|---|"]
    for c in cells:
        for k in ("target", "control"):
            v = V[(V.cell == c) & (V.kind == k)]
            if not len(v):
                continue
            cm = v.groupby("case").median(numeric_only=True)
            L.append(f"| {title(c)} | {k} | {len(cm)} | {cm.flat_al_tex.median():+.2f} -> {cm.real_al_tex.median():+.2f} | {pw(cm.d_al_tex):.1g} | "
                     f"{cm.flat_al_edge.median():+.2f} -> {cm.real_al_edge.median():+.2f} | {pw(cm.d_al_edge):.1g} |")
    (D_OUT / "diag_summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # --- figures -------------------------------------------------------------------------------------------------------------------------
    nc = 5; nr = int(np.ceil(len(cells) / nc)); pc = [f"p{k}" for k in range(RB)]; rr = np.arange(RB)
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
    for j, c in enumerate(cells):
        ax = axs[j // nc, j % nc]
        for k, ls in (("target", "-"), ("control", "--")):
            p = P[(P.cell == c) & (P.kind == k)]
            if not len(p):
                continue
            cmed = {a: p[p.arm == a].groupby("case")[pc].mean() for a in ARMS}; cs = cmed["flat"].index.intersection(cmed["real"].index)
            lr = np.log2(cmed["real"].loc[cs].values / cmed["flat"].loc[cs].values); m = np.nanmedian(lr, 0)
            q1, q3 = np.nanpercentile(lr, [25, 75], axis=0)
            ax.plot(rr, m, ls, color="C3" if k == "target" else "C0", lw=1.4, label=f"{k} ({len(cs)})"); ax.fill_between(rr, q1, q3, color="C3" if k == "target" else "C0", alpha=0.12)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(vt.CELLS[c]["r_loc"], color="0.5", lw=0.5, ls=":"); ax.set_title(title(c), fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6); ax.set_ylim(-1.5, 1.5)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.supxlabel("in-plane distance from the explained voxel (px)", fontsize=9); fig.supylabel("log2 (real-fill |G| / flat-fill |G|), ring mean", fontsize=9)
    fig.suptitle("Absolute evidence per ring, real-fill vs flat-fill (median over cases, IQR shaded). Solid = target voxel, dashed = matched background control; dotted line = r_loc", fontsize=9)
    fig.tight_layout(); fig.savefig(D_OUT / "diag_radial.png", dpi=170); plt.close(fig)
    # sources: stacked shares for target and control, flat vs real
    comp = ("local", "inside", "rim", "other"); col = dict(local="C3", inside="C1", rim="C2", other="0.7")
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(cells) * 2 + 1.5)); yl, yt = [], []; y = 0
    for c in cells[::-1]:
        for k in ("control", "target"):
            v = V[(V.cell == c) & (V.kind == k)]
            if not len(v):
                continue
            cm = v.groupby("case").median(numeric_only=True)
            for a, dy in (("real", -0.18), ("flat", 0.18)):
                left = 0
                for comp_ in comp:
                    w = cm[f"{a}_sh_{comp_}"].median(); ax.barh(y + dy, w, 0.34, left=left, color=col[comp_], edgecolor="k" if a == "real" else "none", lw=0.4); left += w
                ax.text(1.01, y + dy, f"{a}  inside x{cm[f'{a}_ex_inside'].median():.2f}  rim x{cm[f'{a}_ex_rim'].median():.2f}", fontsize=6, va="center", transform=ax.get_yaxis_transform())
            yt.append(y); yl.append(f"{title(c)}  [{k}]"); y += 1
        y += 0.4
    for comp_ in comp:
        ax.barh([], [], color=col[comp_], label={"local": "own neighbourhood (r <= r_loc)", "inside": "inside the target structure", "rim": "target boundary (+-2 px)", "other": "other background"}[comp_])
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7); ax.set_xlim(0, 1); ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    ax.set_xlabel("share of the in-plane evidence (median over cases); upper bar flat-fill, lower bar (outlined) real-fill; x = excess over an isotropic map", fontsize=8)
    fig.tight_layout(); fig.savefig(D_OUT / "diag_sources.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # texture: d_local3d vs texture energy, per cell
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.8 * nr), squeeze=False)
    for j, c in enumerate(cells):
        ax = axs[j // nc, j % nc]; v = V[V.cell == c]
        for k, cc in (("control", "C0"), ("target", "C3")):
            w = v[v.kind == k]; ax.scatter(np.log2(w.tex.clip(1e-3)), w.d_loc3d * 100, s=3, color=cc, alpha=0.35, label=k)
        if c in coefs:
            b, lo, hi = coefs[c]; ax.set_title(f"{title(c)}\nb_tex {b[2]:+.2f} [{lo[2]:+.2f},{hi[2]:+.2f}]  b_target {b[1]:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]", fontsize=6.5)
        else:
            ax.set_title(title(c), fontsize=7)
        ax.axhline(0, color="k", lw=0.5); ax.tick_params(labelsize=7); ax.legend(fontsize=6, markerscale=3)
        lim = np.nanpercentile(np.abs(v.d_loc3d * 100), 98) if len(v) else 1; ax.set_ylim(-lim, lim)
    for j in range(len(cells), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.supxlabel("log2 local fine-texture energy around the voxel (relative to the slice's body median)", fontsize=9)
    fig.supylabel("local share 3D, real - flat (pp)", fontsize=9); fig.tight_layout(); fig.savefig(D_OUT / "diag_texture.png", dpi=170); plt.close(fig)
    # alignment
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(cells) * 2 + 1.5), sharey=True); yl = []
    for t_, (mk, lab) in enumerate((("al_tex", "fine texture |hp|"), ("al_edge", "coarse edge magnitude"))):
        y = 0; yl = []; yt = []
        for c in cells[::-1]:
            for k, cc in (("control", "C0"), ("target", "C3")):
                v = V[(V.cell == c) & (V.kind == k)]
                if not len(v):
                    continue
                cm = v.groupby("case").median(numeric_only=True)
                ax[t_].plot([cm[f"flat_{mk}"].median(), cm[f"real_{mk}"].median()], [y, y], "-", color=cc, lw=1)
                ax[t_].plot(cm[f"flat_{mk}"].median(), y, "o", mfc="white", color=cc, ms=4); ax[t_].plot(cm[f"real_{mk}"].median(), y, "o", color=cc, ms=4)
                yl.append(f"{title(c)} [{k}]"); yt.append(y); y += 1
            y += 0.5
        ax[t_].axvline(0, color="k", lw=0.5); ax[t_].set_xlabel(f"Spearman(de-trended |G|, {lab}), r_loc < r <= {R_ALIGN} px", fontsize=8)
    ytk = yt
    ax[0].set_yticks(ytk); ax[0].set_yticklabels(yl, fontsize=6.5)
    fig.suptitle("What the evidence around the voxel follows: hollow = flat-fill, filled = real-fill (median over cases); red = target, blue = matched control", fontsize=9)
    fig.tight_layout(); fig.savefig(D_OUT / "diag_alignment.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    f = D_OUT / "diag_voxels.csv"
    if "--reuse" in sys.argv and f.exists():
        V = pd.read_csv(f); V["case"] = V.case.astype(str); P = pd.read_csv(D_OUT / "diag_profiles.csv"); P["case"] = P.case.astype(str)
    else:
        V, P = collect()
    report(V, P)


if __name__ == "__main__":
    main()
