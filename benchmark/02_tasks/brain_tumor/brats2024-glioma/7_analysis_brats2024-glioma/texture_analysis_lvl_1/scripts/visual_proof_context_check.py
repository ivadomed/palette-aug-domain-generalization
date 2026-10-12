#!/usr/bin/env python
"""
Stress test of bg_context.md (2026-10-12): "flat-fill's edema call falls with distance to the tumour core, real-fill's does not (T1n-trained)".
CPU only, reads the sampled-voxel npz files of visual_proof_bg.py (VP8_OUT/bg/<cell>/<case>.npz); writes VP8_OUT/bg/context_check/.

Structural fact: s = EDT distance to the nearest NON-edema voxel, and core voxels (NCR/ET/RC) are non-edema, so s <= dcore always.
  core-side voxels  : dcore - s <= 1  (nearest non-edema IS the core; distance to the outer edema boundary unknown, >= s)
  outer-side voxels : dcore - s >  1  (nearest non-edema is outside the lesion, so s == distance to the outer edema boundary, exactly)
Only the outer-side subset can hold the depth to the outer boundary fixed while dcore varies.

Main statistic (per case, per model, outer-side edema voxels with s >= 3, dcore < 900): slope of y on z(log1p dcore) with stratum fixed
effects for depth bin (s 3-5 / 5-8 / 8-12 / 12+) x within-case contrast tertile (zc as in report_context: polarity-oriented z of the
sigma-2 local mean vs the case's own s<-3 background), covariates lc, het, log te (z-scored within case) demeaned within stratum too.
y on three scales: raw L, L / sd_case(L), hit = 1[L > 0] (linear probability). Case-level Wilcoxon: b_flat vs 0, b_real - b_flat vs 0.
Second parameterisation: all edema voxels s >= 3, y on 1[core-side] with the same strata/covariates (a binary "touching-core" effect).

Pre-registered verdict rule (written before running):
  SURVIVES  : in both T1n-trained cells, within-stratum outer-side b_flat < 0 (p<0.01) and b_real - b_flat > 0 (p<0.01) on all three scales,
              holds in the |dDice| <= 0.05 case subset, and the T2w-trained cells show no flat-real difference with the same code.
  WEAKENED  : holds but the slope attenuates > 50 % vs the unstratified per-case slope, or fails on one scale or one subset.
  DIES      : within-stratum slopes indistinguishable between models.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_bg as vb  # noqa: E402

OUT = vb.OUT / "context_check"; OUT.mkdir(exist_ok=True)
CELLS = ["glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t1n", "glioma_t2w_t2f"]
TTL = {"glioma_t1n_t2f": "T1n -> FLAIR", "glioma_t1n_t2w": "T1n -> T2w", "glioma_t2w_t1n": "T2w -> T1n", "glioma_t2w_t2f": "T2w -> FLAIR"}
SB = np.array([3, 5, 8, 12, 1e9]); SBL = ["3-5", "5-8", "8-12", "12+"]
DB = np.array([0, 7, 10, 15, 25, 1e9]); DBL = ["<7", "7-10", "10-15", "15-25", ">25"]
TB = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]); TBL = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1"]
ARMS = ("flat", "real"); SCALES = ("L", "Lsd", "hit")
MIN_N = 60


def wil(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan


def load_cell(c):
    rows = []
    for f in sorted((vb.OUT / c).glob("*.npz")):
        z = np.load(f)
        d = pd.DataFrame({k: z[k].astype(np.float32) for k in ("s", "lm", "te", "het", "lc", "dcore", "L_flat", "L_real")}); d["kind"] = z["kind"]
        d["dice_flat"], d["dice_real"] = float(z["dice_flat"]), float(z["dice_real"]); d["case"] = f.stem; rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    bgs = D[(D.kind == 0) & (D.s < -3)].groupby("case").agg(bm=("lm", "mean"), bs=("lm", "std"))
    T = D[(D.kind == 1) & (D.s >= 3) & (D.dcore < 900)].merge(bgs, left_on="case", right_index=True).reset_index(drop=True)
    pol = T.groupby("case").apply(lambda w: np.sign(w.lm.median() - w.bm.iloc[0]) or 1.0).rename("pol")
    T = T.merge(pol, left_on="case", right_index=True).reset_index(drop=True)
    T["zc"] = T.pol * (T.lm - T.bm) / (T.bs + 1e-6)
    T["outer"] = (T.dcore - T.s) > 1.0
    T["sbin"] = np.digitize(T.s, SB) - 1
    T["lte"] = np.log(T.te + 1e-6)
    T["ldc"] = np.log1p(T.dcore)
    T["t"] = T.dcore / (T.dcore + T.s)
    T["ctert"] = T.groupby("case").zc.transform(lambda x: np.floor(3 * (x.rank(method="first") - 1) / len(x))).astype(int)
    T["stratum"] = T.sbin * 3 + T.ctert
    for a in ARMS:
        T[f"hit_{a}"] = (T[f"L_{a}"] > 0).astype(np.float32)
    return T


def demean(w, cols, g):
    return w[cols] - w.groupby(g)[cols].transform("mean")


def fe_slope(w, xcol, ycols, covs=("lc", "het", "lte"), strata="stratum"):
    """within-stratum OLS slope of each y on xcol (z-scored within case) with z-scored covariates; returns dict y -> slope."""
    w = w.copy(); cs = [xcol] + list(covs)
    for k in cs:
        sd = w[k].std(); w[k] = (w[k] - w[k].mean()) / (sd if sd > 0 else 1.0)
    X = demean(w, cs, strata).values; Y = demean(w, list(ycols), strata).values
    if X[:, 0].std() < 1e-6:
        return {y: np.nan for y in ycols}
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    return {y: b[0, j] for j, y in enumerate(ycols)}


def per_case(T):
    rows = []
    for case, w in T.groupby("case"):
        r = dict(case=case, dice_flat=w.dice_flat.iloc[0], dice_real=w.dice_real.iloc[0], n_all=len(w))
        for a in ARMS:
            sd = w[f"L_{a}"].std(); w = w.assign(**{f"Lsd_{a}": w[f"L_{a}"] / (sd if sd > 0 else 1.0)})
        ys = [f"{sc}_{a}" for sc in SCALES for a in ARMS]
        # (0) unstratified slope (all s>=3 edema; covariates only zc + log s, like bg_context) -- the reference to measure attenuation
        if len(w) >= MIN_N:
            w0 = w.assign(one=0, ls=np.log(w.s))
            b0 = fe_slope(w0, "ldc", ys, covs=("zc", "ls"), strata="one")
            r.update({f"uns_{k}": v for k, v in b0.items()})
        O = w[w.outer]; r["n_outer"] = len(O)
        if len(O) >= MIN_N and O.stratum.nunique() >= 2:
            b = fe_slope(O, "ldc", ys); r.update({f"fe_{k}": v for k, v in b.items()})
            bt = fe_slope(O.assign(one=0), "t", ys, covs=("zc", "lc", "het", "lte"), strata="stratum"); r.update({f"fet_{k}": v for k, v in bt.items()})
            # hit per dcore bin within the populated depth bins (for the figure)
            for sb in range(3):
                for k in range(len(DB) - 1):
                    m = (O.sbin == sb) & (O.dcore >= DB[k]) & (O.dcore < DB[k + 1])
                    for a in ARMS:
                        r[f"h_{a}_s{sb}_d{k}"] = O[f"hit_{a}"][m].mean() if m.sum() >= 5 else np.nan
            for k in range(len(TB) - 1):
                m = (O.t >= TB[k]) & (O.t < TB[k + 1])
                for a in ARMS:
                    r[f"ht_{a}_{k}"] = O[f"hit_{a}"][m].mean() if m.sum() >= 5 else np.nan
        C = w[~w.outer]
        for a in ARMS:
            r[f"ht_{a}_core"] = C[f"hit_{a}"].mean() if len(C) >= 5 else np.nan
        # (2) binary core-side indicator at matched depth bin x contrast tertile (all s>=3 edema)
        if len(w) >= MIN_N and w.outer.any() and (~w.outer).any():
            # only strata containing both kinds identify the contrast
            ok = w.groupby("stratum").outer.transform(lambda x: x.any() and (~x).any())
            ww = w[ok].assign(cs=(~w.outer[ok]).astype(float))
            if len(ww) >= 30:
                bc = fe_slope(ww, "cs", ys); sdc = ww.cs.std(); r.update({f"cs_{k}": v / sdc for k, v in bc.items()})
                r["n_cs"] = len(ww)
        rows.append(r)
    R = pd.DataFrame(rows); R["ddice"] = R.dice_real - R.dice_flat
    return R


def summarise(R, pre, label):
    out = dict(subset=label)
    for sc in SCALES:
        bf, br = R.get(f"{pre}_{sc}_flat"), R.get(f"{pre}_{sc}_real")
        if bf is None:
            continue
        ok = bf.notna() & br.notna(); bf, br = bf[ok], br[ok]
        out[f"{sc}_n"] = int(ok.sum()); out[f"{sc}_flat"] = bf.median(); out[f"{sc}_real"] = br.median()
        out[f"{sc}_pflat"] = wil(bf); out[f"{sc}_preal"] = wil(br); out[f"{sc}_pdiff"] = wil(br - bf); out[f"{sc}_diff"] = (br - bf).median()
    return out


def fmt_row(cell, s, sc):
    if f"{sc}_n" not in s:
        return f"| {cell} | {s['subset']} | {sc} | - |"
    return (f"| {cell} | {s['subset']} | {sc} | {s[f'{sc}_n']} | {s[f'{sc}_flat']:+.3f} ({s[f'{sc}_pflat']:.1g}) | {s[f'{sc}_real']:+.3f} ({s[f'{sc}_preal']:.1g}) | "
            f"{s[f'{sc}_diff']:+.3f} | {s[f'{sc}_pdiff']:.1g} |")


def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    bs = rng.choice(x, (n, len(x))).mean(1); return x.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hdr = "| cell | case subset | scale | n cases | median b flat (p vs 0) | median b real (p vs 0) | median real - flat | p (Wilcoxon) |\n|---|---|---|---|---|---|---|---|"
    secs = {k: [hdr] for k in ("uns", "fe", "fet", "cs")}
    desc, res = [], {}
    for c in CELLS:
        T = load_cell(c); R = per_case(T); res[c] = R; R.to_csv(OUT / f"per_case_{c}.csv", index=False)
        O = T[T.outer]
        desc.append(f"| {TTL[c]} | {T.case.nunique()} | {len(T)} | {O.case.nunique() if len(O) else 0} | {T.outer.mean() * 100:.0f} % | "
                    f"{np.corrcoef(T.ldc, np.log(T.s))[0, 1]:+.2f} | {np.corrcoef(O.ldc, np.log(O.s))[0, 1]:+.2f} | {np.corrcoef(O.ldc, O.zc)[0, 1]:+.2f} | "
                    f"{(R.n_outer >= MIN_N).sum()} | {R.dice_flat.median():.2f} / {R.dice_real.median():.2f} | "
                    f"{T.L_flat.std():.2f} / {T.L_real.std():.2f} |")
        subsets = [("all", R), ("|dDice|<=0.05", R[R.ddice.abs() <= 0.05]), ("flat Dice >= median", R[R.dice_flat >= R.dice_flat.median()])]
        for pre in secs:
            for lab, RR in subsets:
                s = summarise(RR, pre, lab)
                for sc in SCALES:
                    secs[pre].append(fmt_row(TTL[c], s, sc))
    md = ["# Stress test: does flat-fill's edema call depend on distance to the tumour core at matched depth and contrast?", "",
          "Data: sampled edema voxels (s >= 3, case has a core) of visual_proof_bg.py, fold-mean logits; per-case statistics, case-level Wilcoxon.",
          "s = distance to the nearest NON-edema voxel, so s <= dcore always. Outer-side voxels (dcore - s > 1) have s = distance to the outer edema",
          "boundary exactly; core-side voxels (dcore - s <= 1) touch the core in the EDT sense and their outer distance is unknown.", "",
          "Pre-registered verdict rule (in the script header, written before running): SURVIVES if in both T1n-trained cells the within-stratum",
          "outer-side slope is < 0 for flat (p<0.01) and real - flat > 0 (p<0.01) on all three scales, holds for |dDice| <= 0.05, and the T2w-trained",
          "cells show no flat-real difference; WEAKENED if it attenuates > 50 % vs the unstratified slope or fails one scale/subset; DIES otherwise.", "",
          "## Descriptives", "",
          "| cell | cases | edema voxels | cases with outer-side voxels | outer-side share | corr(log1p dcore, log s) all | outer-side | corr(log1p dcore, zc) outer-side | cases with >= 60 outer-side | median patch Dice flat / real | SD of L flat / real |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + desc
    md += ["", "Scales: L = edema log-odds; Lsd = L / its within-case SD; hit = 1[L > 0] (linear probability). Slopes are per within-case SD of the regressor.", "",
           "## (0) Reference: unstratified per-case slope on log1p(dcore) (all s >= 3 edema; covariates zc, log s; as in bg_context.md)", ""] + secs["uns"]
    md += ["", "## (1) MAIN: outer-side voxels, slope on log1p(dcore) within depth bin (s 3-5/5-8/8-12/12+) x within-case contrast tertile, covariates lc, het, log te", ""] + secs["fe"]
    md += ["", "## (1b) same, regressor = relative position t = dcore / (dcore + s) (0.5 = equidistant, 1 = at the outer boundary), covariates zc, lc, het, log te", ""] + secs["fet"]
    md += ["", "## (2) Binary: core-side vs outer-side voxels at matched depth bin x contrast tertile (b = core-side minus outer-side difference in y units, within stratum; > 0 = touching the core raises the edema call)", ""] + secs["cs"]
    # figure 1: hit vs dcore per depth bin, outer-side
    fig, axs = plt.subplots(len(CELLS), 3, figsize=(10.5, 2.6 * len(CELLS)), squeeze=False)
    for i, c in enumerate(CELLS):
        R = res[c]
        for sb in range(3):
            ax = axs[i, sb]
            for a, col in (("flat", "C0"), ("real", "C3")):
                st = [boot_ci(R[f"h_{a}_s{sb}_d{k}"]) if f"h_{a}_s{sb}_d{k}" in R else (np.nan,) * 3 for k in range(len(DB) - 1)]
                m, lo, hi = map(np.array, zip(*st)); n = [R[f"h_{a}_s{sb}_d{k}"].notna().sum() if f"h_{a}_s{sb}_d{k}" in R else 0 for k in range(len(DB) - 1)]
                m = np.where(np.array(n) >= 8, m, np.nan)
                ax.errorbar(np.arange(len(DBL)) + (0.06 if a == "real" else -0.06), m, yerr=[m - lo, hi - m], color=col, marker="o", ms=3, lw=1.2, capsize=2, label=f"{a}-fill")
            ax.set_ylim(0, 1.02); ax.set_xticks(range(len(DBL))); ax.set_xticklabels(DBL, fontsize=7); ax.tick_params(labelsize=7)
            ax.set_title(f"{TTL[c]}: depth to outer boundary {SBL[sb]} vox", fontsize=8)
            if sb == 0:
                ax.legend(fontsize=6)
    fig.supxlabel("outer-side edema voxels: distance to the nearest tumour-core voxel (vox); points need >= 8 cases", fontsize=8.5)
    fig.supylabel("share called edema (case mean, 95% bootstrap CI over cases)", fontsize=8.5); fig.tight_layout(); fig.savefig(OUT / "context_check_depth_matched.png", dpi=170); plt.close(fig)
    # figure 2: hit vs relative position t
    fig, axs = plt.subplots(1, len(CELLS), figsize=(3.4 * len(CELLS), 2.9), squeeze=False)
    xl = ["core-side\n(t<=0.5)"] + TBL
    for j, c in enumerate(CELLS):
        R = res[c]; ax = axs[0, j]
        for a, col in (("flat", "C0"), ("real", "C3")):
            st = [boot_ci(R[f"ht_{a}_core"])] + [boot_ci(R[f"ht_{a}_{k}"]) if f"ht_{a}_{k}" in R else (np.nan,) * 3 for k in range(len(TB) - 1)]
            m, lo, hi = map(np.array, zip(*st))
            ax.errorbar(np.arange(len(xl)) + (0.06 if a == "real" else -0.06), m, yerr=[m - lo, hi - m], color=col, marker="o", ms=3, lw=1.2, capsize=2, label=f"{a}-fill")
        ax.set_ylim(0, 1.02); ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, fontsize=6, rotation=30); ax.tick_params(labelsize=7); ax.set_title(TTL[c], fontsize=8.5); ax.legend(fontsize=6)
    fig.supxlabel("relative position t = dcore / (dcore + s), outer-side voxels only (t is undefined on the core side)", fontsize=8.5)
    fig.supylabel("share called edema", fontsize=8.5); fig.tight_layout(); fig.savefig(OUT / "context_check_relpos.png", dpi=170); plt.close(fig)
    (OUT / "context_check.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
