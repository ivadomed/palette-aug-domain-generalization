#!/usr/bin/env python
"""
Visual proof v3b (Paul, 2026-10-11): activation maps on REAL, untouched test images -- "do I see activation in texture all across the map,
or mostly at boundaries, and how do the two arms differ?". Same checkpoints/cases as v2/v3 (BraTS 70 test cases, edema, flat-fill rung 4.5
vs real-fill rung 5, folds 0-2). Encoder run on the WHOLE preprocessed volume (fully convolutional; padded to a multiple of 32), per fold:
  energy map  E_s = RMS over channels of the stage-s output (s0 full-res, s1 half-res, upsampled to full res);
  local-variance map V_s = std of E_s over a 3x3x3 window (how structured the activation is locally);
fold-mean of both; saved as float16 npz (aggregated maps only). GT is used only for the ZONES of the statistics (never for any input):
  interior = edema with d >= 2 vox, border = |d| <= 2 vox, outside = body minus any tumour label dilated by 2 vox.
Per case x arm x stage: density (mean per voxel) of E and V in the three zones, interior/border ratio, share of total E/V mass per zone,
and the whole-volume Spearman of E with local high-pass energy (sigma 2) vs gradient magnitude. Paired Wilcoxon real vs flat over cases.
Sheets (display cases = highest arm agreement from v2's majority predictions, GT edema >= 2000 vox): 3 lesion-rich slices x
[image + GT | flat E_s0 | real E_s0 | real - flat | flat E_s1 | real E_s1 | real - flat], GT green, predictions thin blue/orange.
Output: VP3B_OUT (default outputs/plots/visual_proof_v3b_realmaps/): <cell>/<case>_realmaps.npz, <cell>/cases.csv, sheets, violins, v3b_stats.md.
GPU: --cells <cell> [--case-slice i:j]   (resumable per case)        CPU: --plots
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
os.environ.setdefault("VP_OUT", str(THIS.parent / "outputs/plots/visual_proof_v2_brats_full"))
import visual_proof_v2 as v2  # noqa: E402
from acvl_utils.cropping_and_padding.padding import pad_nd_image  # noqa: E402
from visual_proof_v3_activations import spearman_rows, read_csv_clean  # noqa: E402

OUT = Path(os.environ.get("VP3B_OUT") or (THIS.parent / "outputs/plots/visual_proof_v3b_realmaps")); OUT.mkdir(parents=True, exist_ok=True)
V2OUT = Path(os.environ.get("VP_OUT"))   # v2 full-BraTS outputs (majority predictions for contours / display-case choice)
CELL_KEYS = ["bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w"]
ARMS = ("flat", "real"); STG = (0, 1); ARM_LABEL = v2.ARM_LABEL; log = v2.log


@torch.no_grad()
def agg_maps(net, img: np.ndarray, dev):
    x = torch.from_numpy(img)[None, None]; xp, revert = pad_nd_image(x, None, "constant", {"value": 0}, True, [32, 32, 32])   # spatial dims only (an int would also pad batch/channel)
    feats = net.encoder(xp.to(dev).float()); feats = feats if isinstance(feats, (list, tuple)) else [feats]
    out = {}
    for s in STG:
        e = feats[s][0].float().pow(2).mean(0, keepdim=True).sqrt()[None]          # (1,1,z,y,x) RMS over channels
        if tuple(e.shape[2:]) != tuple(xp.shape[2:]):
            e = F.interpolate(e, size=xp.shape[2:], mode="trilinear", align_corners=False)
        e = e[0, 0][revert[2:]].cpu().numpy().astype(np.float32)
        lv = np.sqrt(np.maximum(ndi.uniform_filter(e ** 2, 3) - ndi.uniform_filter(e, 3) ** 2, 0))
        out[s] = (e, lv)
    return out


def zones_from_gt(gt, lids, body):
    ed = np.isin(gt, lids); tum = gt > 0
    din = ndi.distance_transform_edt(ed); dout = ndi.distance_transform_edt(~ed)
    return dict(interior=ed & (din >= 2), border=((din <= 2) & ed) | ((dout <= 2) & ~ed), outside=body & ~ndi.binary_dilation(tum, iterations=2)) | {"_edema": ed}


def run_cell(key, dev, cases=None):
    cell = v2.CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True); lids = list(cell["lids"])
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]
    for case in (cases or cell["cases"]):
        if (od / f"{case}_realmaps.npz").exists() and (od / f"{case}_case.csv").exists():
            log(f"[{key}] {case}: exists, skipped"); continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]
        body = img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool); Z = zones_from_gt(gt, lids, body)
        if Z["_edema"].sum() < 50:
            log(f"[{key}] {case}: edema < 50 vox, skipped"); continue
        L = ndi.gaussian_filter(img, 2.0); tex = ndi.gaussian_filter((img - L) ** 2, 2.0); grad = np.sqrt(sum(ndi.sobel(img, axis=i) ** 2 for i in range(3)))
        sub = np.flatnonzero(body); sub = np.random.default_rng(0).choice(sub, min(100000, sub.size), replace=False); t_, g_ = tex.ravel()[sub], grad.ravel()[sub]
        row = dict(cell=key, case=case, edema_vox=int(Z["_edema"].sum()), n_interior=int(Z["interior"].sum()), n_border=int(Z["border"].sum()), n_outside=int(Z["outside"].sum())); save = {"img": img.astype(np.float16), "gt": gt.astype(np.int8)}
        for a, pr in prs.items():
            acc = {s: [np.zeros(img.shape, np.float32), np.zeros(img.shape, np.float32)] for s in STG}
            for k in v2.FOLDS:
                pr.network.load_state_dict(pr.list_of_parameters[k]); pr.network.eval(); m = agg_maps(pr.network, img, dev)
                for s in STG:
                    acc[s][0] += m[s][0] / len(v2.FOLDS); acc[s][1] += m[s][1] / len(v2.FOLDS)
            for s in STG:
                e, lv = acc[s]; save[f"energy_{a}_s{s}"] = e.astype(np.float16); save[f"lvar_{a}_s{s}"] = lv.astype(np.float16)
                for name, mp in (("energy", e), ("lvar", lv)):
                    tot = mp[body].sum()
                    for zn in ("interior", "border", "outside"):
                        zm = Z[zn] & body
                        row[f"{a}_s{s}_{name}_density_{zn}"] = float(mp[zm].mean()) if zm.any() else np.nan
                        row[f"{a}_s{s}_{name}_share_{zn}"] = float(mp[zm].sum() / tot) if tot > 0 else np.nan
                    row[f"{a}_s{s}_{name}_int_over_border"] = row[f"{a}_s{s}_{name}_density_interior"] / row[f"{a}_s{s}_{name}_density_border"] if row[f"{a}_s{s}_{name}_density_border"] else np.nan
                    row[f"{a}_s{s}_{name}_int_over_outside"] = row[f"{a}_s{s}_{name}_density_interior"] / row[f"{a}_s{s}_{name}_density_outside"] if row[f"{a}_s{s}_{name}_density_outside"] else np.nan
                X = e.ravel()[sub][None]; row[f"{a}_s{s}_rho_energy_tex"] = float(spearman_rows(X, t_)[0]); row[f"{a}_s{s}_rho_energy_edge"] = float(spearman_rows(X, g_)[0])
                X = lv.ravel()[sub][None]; row[f"{a}_s{s}_rho_lvar_tex"] = float(spearman_rows(X, t_)[0]); row[f"{a}_s{s}_rho_lvar_edge"] = float(spearman_rows(X, g_)[0])
        np.savez_compressed(od / f"{case}_realmaps.npz", **save); pd.DataFrame([row]).to_csv(od / f"{case}_case.csv", index=False)
        log(f"[{key}] {case}: E s0 interior/border flat {row['flat_s0_energy_int_over_border']:.2f} real {row['real_s0_energy_int_over_border']:.2f}; rho(E,tex) flat {row['flat_s0_rho_energy_tex']:.2f} real {row['real_s0_rho_energy_tex']:.2f}; {time.time() - t0:.0f} s")
    C = [pd.read_csv(f) for f in sorted(od.glob("*_case.csv"))]; (pd.concat(C, ignore_index=True) if C else pd.DataFrame()).to_csv(od / "cases.csv", index=False)


def sheet(key, case, z, major, path):
    img = z["img"].astype(np.float32); gt = z["gt"]; ed = np.isin(gt, list(v2.CELLS[key]["lids"]))
    zs = v2.pick_slices(ed); fg = img[img != 0]; lo, hi = np.percentile(fg, [1, 99])
    fig, axes = plt.subplots(len(zs), 7, figsize=(31, 4.8 * len(zs) + 1.0), facecolor="#fcfcfb", squeeze=False, constrained_layout=True)
    for r, zz in enumerate(zs):
        axes[r, 0].imshow(img[zz].T, cmap="gray", vmin=lo, vmax=hi, origin="lower"); axes[r, 0].set_title(f"z={zz}  image + GT edema", fontsize=11, loc="left")
        for j, s in enumerate(STG):
            ef, er = z[f"energy_flat_s{s}"].astype(np.float32), z[f"energy_real_s{s}"].astype(np.float32); vmax = max(np.percentile(ef, 99.5), np.percentile(er, 99.5)) or 1
            d = er - ef; dmax = np.percentile(np.abs(d), 99.5) or 1; c0 = 1 + 3 * j
            axes[r, c0].imshow(ef[zz].T, cmap="inferno", vmin=0, vmax=vmax, origin="lower"); axes[r, c0].set_title(f"flat-fill energy s{s}", fontsize=11, loc="left")
            axes[r, c0 + 1].imshow(er[zz].T, cmap="inferno", vmin=0, vmax=vmax, origin="lower"); axes[r, c0 + 1].set_title(f"real-fill energy s{s}", fontsize=11, loc="left")
            axes[r, c0 + 2].imshow(d[zz].T, cmap="RdBu_r", vmin=-dmax, vmax=dmax, origin="lower"); axes[r, c0 + 2].set_title(f"real - flat s{s}", fontsize=11, loc="left")
        for c in range(7):
            ax = axes[r, c]; ax.axis("off")
            if ed[zz].any():
                ax.contour(ed[zz].T.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=1.2)
            for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
                if major is not None and f"major_{a}" in major and np.isin(major[f"major_{a}"][zz], list(v2.CELLS[key]["lids"])).any():
                    ax.contour(np.isin(major[f"major_{a}"][zz], list(v2.CELLS[key]["lids"])).T.astype(float), levels=[0.5], colors=[col], linewidths=0.7, linestyles="dotted")
    fig.suptitle(f"{case}  |  {v2.CELLS[key]['title']}  |  channel-RMS activation energy of encoder stage s0 (full-res) and s1 (half-res) on the UNTOUCHED test image, fold-mean over folds 0-2; "
                 "GT edema green, flat-fill / real-fill majority predictions dotted blue / orange; colour scale shared by both arms per stage", fontsize=11, x=0.01, ha="left", wrap=True)
    fig.savefig(path, dpi=90, facecolor="#fcfcfb"); plt.close(fig)


def plots():
    cells = [c for c in CELL_KEYS if (OUT / c / "cases.csv").exists()]
    cf = pd.concat([read_csv_clean(OUT / c / "cases.csv") for c in cells], ignore_index=True); cf.to_csv(OUT / "cases_all.csv", index=False)
    lines = ["# Visual proof v3b: channel-aggregated encoder activation on REAL test images (no perturbation), flat-fill vs real-fill", "",
             "Zones from the GT edema only: interior (d >= 2 vox), border (|d| <= 2), outside (body minus any tumour label dilated by 2). Density = mean map value per voxel in the zone; E = channel-RMS energy, V = local std of E (3-vox window). Paired Wilcoxon real vs flat over cases.", ""]
    wp = lambda x, y: wilcoxon(x, y).pvalue if len(x) >= 5 and (x != y).any() else np.nan
    for name in ("energy", "lvar"):
        lines += [f"## {name}: density per zone and ratios", "", "| cell | stage | n | interior flat / real | border flat / real | outside flat / real | int/border flat | real | real > flat | p | int/outside flat | real | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            d = cf[cf.cell == c]
            for s in STG:
                k = lambda a, q: d[f"{a}_s{s}_{name}_{q}"]
                lines.append(f"| {c} | s{s} | {len(d)} | {k('flat','density_interior').median():.3f} / {k('real','density_interior').median():.3f} | {k('flat','density_border').median():.3f} / {k('real','density_border').median():.3f} | "
                             f"{k('flat','density_outside').median():.3f} / {k('real','density_outside').median():.3f} | {k('flat','int_over_border').median():.2f} | {k('real','int_over_border').median():.2f} | {(k('real','int_over_border') > k('flat','int_over_border')).mean():.2f} | {wp(k('real','int_over_border'), k('flat','int_over_border')):.2g} | "
                             f"{k('flat','int_over_outside').median():.2f} | {k('real','int_over_outside').median():.2f} | {wp(k('real','int_over_outside'), k('flat','int_over_outside')):.2g} |")
        lines.append("")
    lines += ["## Whole-volume Spearman of the maps with local high-pass energy (texture) vs gradient magnitude (edges), median over cases", "", "| cell | stage | map | rho_tex flat / real | rho_edge flat / real | real > flat rho_tex (frac) | p |", "|---|---|---|---|---|---|---|"]
    for c in cells:
        d = cf[cf.cell == c]
        for s in STG:
            for name in ("energy", "lvar"):
                ft, rt = d[f"flat_s{s}_rho_{name}_tex"], d[f"real_s{s}_rho_{name}_tex"]; fe, re = d[f"flat_s{s}_rho_{name}_edge"], d[f"real_s{s}_rho_{name}_edge"]
                lines.append(f"| {c} | s{s} | {name} | {ft.median():.2f} / {rt.median():.2f} | {fe.median():.2f} / {re.median():.2f} | {(rt > ft).mean():.2f} | {wp(rt, ft):.2g} |")
    lines.append("")
    # violins: density interior / border / outside per arm, per stage, energy and lvar
    for name in ("energy", "lvar"):
        fig, axes = plt.subplots(len(STG), len(cells), figsize=(4.8 * len(cells), 4.4 * len(STG)), facecolor="#fcfcfb", squeeze=False)
        for i, s in enumerate(STG):
            for j, c in enumerate(cells):
                d = cf[cf.cell == c]; ax = axes[i, j]; pos = 0; ticks = []
                for zn in ("interior", "border", "outside"):
                    for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
                        v = d[f"{a}_s{s}_{name}_density_{zn}"].dropna().values
                        if len(v):
                            vp = ax.violinplot([v], positions=[pos], showmedians=True, widths=0.8); vp["bodies"][0].set_facecolor(col); vp["bodies"][0].set_alpha(0.4)
                            ax.scatter(pos + np.random.default_rng(0).normal(0, 0.05, len(v)), v, s=6, c=col, alpha=0.5)
                        ticks.append((pos, f"{zn}\n{a}")); pos += 1
                    pos += 0.5
                ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=7); ax.set_title(f"{c}: s{s} {name} density", fontsize=9); ax.set_ylabel("mean per voxel" if j == 0 else "")
        fig.suptitle(f"{name}: activation {'energy (channel RMS)' if name == 'energy' else 'local variance (3-vox std of energy)'} density in GT-defined zones on real images; blue flat-fill, orange real-fill", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / f"violin_{name}_density.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)
    # sheets: 3 display cases per cell with highest arm agreement (from v2 majors) and edema >= 2000
    for c in cells:
        d = cf[cf.cell == c]; scored = []
        for case in d.case:
            f2 = V2OUT / c / f"{case}_maps.npz"
            if not f2.exists() or int(d[d.case == case].edema_vox.iloc[0]) < 2000:
                continue
            m = np.load(f2); lids = list(v2.CELLS[c]["lids"]); pf, pr_ = np.isin(m["major_flat"], lids), np.isin(m["major_real"], lids)
            scored.append((2 * (pf & pr_).sum() / max(pf.sum() + pr_.sum(), 1), case))
        for agree, case in sorted(scored, reverse=True)[:3]:
            z = np.load(OUT / c / f"{case}_realmaps.npz"); m = np.load(V2OUT / c / f"{case}_maps.npz")
            sheet(c, f"{case} (arm agreement Dice {agree:.2f})", z, m, OUT / f"{c}_{case}_realmaps.png"); log(f"sheet {c} {case} agreement {agree:.2f}")
    (OUT / "v3b_stats.md").write_text("\n".join(lines)); print("\n".join(lines)); log("wrote v3b_stats.md + plots")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELL_KEYS)); ap.add_argument("--plots", action="store_true"); ap.add_argument("--case-slice", default=None); a = ap.parse_args()
    if a.plots:
        plots(); return
    dev = torch.device("cuda")
    for key in a.cells.split(","):
        cases = None
        if a.case_slice:
            i_, j_ = (int(x) if x else None for x in a.case_slice.split(":")); cases = v2.CELLS[key]["cases"][i_:j_]
        try:
            run_cell(key, dev, cases)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"[{key}] FAILED: {e}")


if __name__ == "__main__":
    main()
