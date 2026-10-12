#!/usr/bin/env python
"""
Visual proof v3c (Paul, 2026-10-11): does the flat-fill arm concentrate activation energy on the lesion CONTOUR while the real-fill arm
spreads it over the INTERIOR once integrated over area / at deeper stages? Builds on v3b (s0/s1 energy + local-variance maps on real images)
and adds, per case / arm:
  GPU pass (--gpu): encoder stages s2 (1/4 res) and s3 (1/8 res) channel-RMS energy + 3-vox local-std maps, fold-mean, saved at native stage
    resolution (float16); per-channel interior/border mean-|activation| ratios at s0 and s1 (GT zones: interior d <= -2, border |d| <= 2),
    fold-mean, saved per arm; for the display cases the top-3 interior-dominated s0 channels' maps (float16).
  CPU pass (--analyze, on the cluster that holds the v3b/v2 npz):
    1. radial profiles: mean energy (and local variance) vs signed distance to the GT edema boundary (-12 inside .. +12 outside, 1-vox bins),
       each map normalised by its whole-body mean; same vs the distance to the PREDICTED boundary (union of both arms' majority predictions
       from the v2 full-BraTS npz); mean +- IQR curves per arm and the real - flat curve, per cell and stage (s0-s3).
    2. integrated quantities: SUM (not density) of normalised energy in the interior (d <= -2) and the border band (|d| <= 2) per arm; the
       real - flat difference of each; interior_sum / border_sum per arm; paired Wilcoxon over cases.
    4. per-channel: distribution over channels of each channel's interior/border ratio, per arm (pooled over cases), tail fraction (> 1.2) per
       case with Wilcoxon; histograms; top-3 interior-dominated channels per arm as maps for one high-agreement case per cell.
    5. sheets for 2 high-agreement cases per cell: zoomed real - flat maps s0..s3 with GT + prediction contours, energy maps, top channels.
Output: VP3C_OUT (default outputs/plots/visual_proof_v3c_profiles/).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
os.environ.setdefault("VP_OUT", str(THIS.parent / "outputs/plots/visual_proof_v2_brats_full"))
import visual_proof_v2 as v2  # noqa: E402
from acvl_utils.cropping_and_padding.padding import pad_nd_image  # noqa: E402
from visual_proof_v3_activations import read_csv_clean  # noqa: E402

OUT = Path(os.environ.get("VP3C_OUT") or (THIS.parent / "outputs/plots/visual_proof_v3c_profiles")); OUT.mkdir(parents=True, exist_ok=True)
V2OUT = Path(os.environ.get("VP_OUT")); V3B = Path(os.environ.get("VP3B_OUT") or (THIS.parent / "outputs/plots/visual_proof_v3b_realmaps"))
CELL_KEYS = ["bratsfull_t2w_t1n", "bratsfull_t2w_t2f", "bratsfull_t1n_t2f", "bratsfull_t1n_t2w"]
ARMS = ("flat", "real"); ALL_STG = (0, 1, 2, 3); NEW_STG = (2, 3); CH_STG = (0, 1); BINS = np.arange(-12, 13); log = v2.log
N_SHOW = 2


def gt_zones(gt, lids):
    ed = np.isin(gt, lids); din = ndi.distance_transform_edt(ed); dout = ndi.distance_transform_edt(~ed); d = np.where(ed, -din, dout)
    return ed, d, ed & (din >= 2), np.abs(d) <= 2


def display_cases(key):
    """Highest arm-agreement cases (v2 majority predictions), GT edema >= 2000 vox."""
    cf = read_csv_clean(V3B / key / "cases.csv") if (V3B / key / "cases.csv").exists() else None
    scored = []
    for f2 in sorted((V2OUT / key).glob("*_maps.npz")):
        case = f2.name[:-9]
        if cf is not None and case in set(cf.case) and int(cf[cf.case == case].edema_vox.iloc[0]) < 2000:
            continue
        m = np.load(f2); lids = list(v2.CELLS[key]["lids"]); pf, pr_ = np.isin(m["major_flat"], lids), np.isin(m["major_real"], lids)
        if pf.sum() + pr_.sum() == 0:
            continue
        scored.append((2 * (pf & pr_).sum() / (pf.sum() + pr_.sum()), case))
    return [c for _, c in sorted(scored, reverse=True)[:N_SHOW]]


@torch.no_grad()
def run_gpu_cell(key, dev, cases=None):
    cell = v2.CELLS[key]; od = OUT / key; od.mkdir(exist_ok=True); lids = list(cell["lids"])
    prs = {a: v2.load_predictor(d, dev) for a, d in cell["models"].items()}; pr0 = prs["flat"]; show = set(display_cases(key))
    for case in (cases or cell["cases"]):
        if (od / f"{case}_deep.npz").exists():
            log(f"[{key}] {case}: exists, skipped"); continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]
        ed, dsg, inner, border = gt_zones(gt, lids)
        if ed.sum() < 50:
            log(f"[{key}] {case}: edema < 50 vox, skipped"); continue
        x = torch.from_numpy(img)[None, None]; xp, revert = pad_nd_image(x, None, "constant", {"value": 0}, True, [32, 32, 32])
        save = {}
        for a, pr in prs.items():
            acc = {}; chan = {s: [] for s in CH_STG}; topmaps = {}
            for k in v2.FOLDS:
                pr.network.load_state_dict(pr.list_of_parameters[k]); pr.network.eval()
                feats = pr.network.encoder(xp.to(dev).float()); feats = feats if isinstance(feats, (list, tuple)) else [feats]
                for s in NEW_STG:
                    fs = feats[s][0].float(); e = fs.pow(2).mean(0).sqrt()
                    # crop the padding at the stage grid
                    sc = [fs.shape[1 + i] / xp.shape[2 + i] for i in range(3)]
                    sl = tuple(slice(int(round(r.start * c)), int(round(r.stop * c))) for r, c in zip(revert[2:], sc)); e = e[sl].cpu().numpy()
                    lv = np.sqrt(np.maximum(ndi.uniform_filter(e ** 2, 3) - ndi.uniform_filter(e, 3) ** 2, 0))
                    acc.setdefault(f"energy_s{s}", np.zeros_like(e)); acc[f"energy_s{s}"] += e / 3; acc.setdefault(f"lvar_s{s}", np.zeros_like(e)); acc[f"lvar_s{s}"] += lv / 3
                for s in CH_STG:
                    fs = feats[s][0].abs().float()
                    if tuple(fs.shape[1:]) != tuple(xp.shape[2:]):
                        fs = F.interpolate(fs[None], size=xp.shape[2:], mode="trilinear", align_corners=False)[0]
                    fs = fs[(slice(None), *revert[2:])]
                    In = torch.from_numpy(inner).to(dev); Bo = torch.from_numpy(border).to(dev)
                    mi = fs[:, In].mean(1) if In.any() else torch.full((fs.shape[0],), float("nan"), device=dev); mb = fs[:, Bo].mean(1)
                    chan[s].append((mi / mb.clamp(min=1e-12)).cpu().numpy())
                    if s == 0 and k == 0 and case in show:
                        r = (mi / mb.clamp(min=1e-12)).cpu().numpy(); top = np.argsort(np.nan_to_num(r, nan=-1))[::-1][:3]
                        for c in top:
                            topmaps[f"topch_{a}_s0_ch{int(c)}"] = fs[c].cpu().numpy().astype(np.float16)
            for n, m in acc.items():
                save[f"{n.split('_')[0]}_{a}_{n.split('_')[1]}"] = m.astype(np.float16)
            for s in CH_STG:
                save[f"chratio_{a}_s{s}"] = np.nanmean(np.stack(chan[s]), 0).astype(np.float32)
            save.update(topmaps)
        np.savez_compressed(od / f"{case}_deep.npz", **save)
        log(f"[{key}] {case}: done ({time.time() - t0:.0f} s)")


class Binner:
    """Signed-distance bins computed once per (case, reference); profile(map) is a single bincount pass."""
    def __init__(self, d, body, bins=BINS):
        idx = np.digitize(d, bins) - 1; valid = body & (d >= bins[0]) & (d < bins[-1]); self.sel = np.flatnonzero(valid); self.idx = idx.ravel()[self.sel]
        self.n = len(bins) - 1; self.cnt = np.bincount(self.idx, minlength=self.n).astype(np.float64)
    def profile(self, mp):
        s_ = np.bincount(self.idx, weights=mp.ravel()[self.sel], minlength=self.n); return np.where(self.cnt > 0, s_ / np.maximum(self.cnt, 1), np.nan)


def profile(mp, d, body, bins=BINS):
    return Binner(d, body, bins).profile(mp)


def up(mp, shape):
    return mp if mp.shape == tuple(shape) else ndi.zoom(mp.astype(np.float32), np.array(shape) / np.array(mp.shape), order=1)


def analyze(only_cells=None):
    cells = [c for c in (only_cells or CELL_KEYS) if (V3B / c / "cases.csv").exists()]
    for key in cells:
        rows, profs, chans = [], [], []; od = OUT / key; od.mkdir(exist_ok=True)
        n_cases = len(list((V3B / key).glob("*_realmaps.npz")))
        if (od / "cases_v3c.csv").exists() and len(pd.read_csv(od / "cases_v3c.csv")) >= n_cases:
            log(f"[{key}] cases_v3c.csv complete, skipped"); continue
        lids = list(v2.CELLS[key]["lids"]); show = display_cases(key)
        for f in sorted((V3B / key).glob("*_realmaps.npz")):
            case = f.name[:-13]; zb = np.load(f); zd = np.load(od / f"{case}_deep.npz") if (od / f"{case}_deep.npz").exists() else None
            img = zb["img"].astype(np.float32); gt = zb["gt"]; body = img != 0; ed, dsg, inner, border = gt_zones(gt, lids); bin_gt = Binner(dsg, body)
            m2 = V2OUT / key / f"{case}_maps.npz"; dpred = None; bin_pred = None
            if m2.exists():
                mm = np.load(m2); un = np.isin(mm["major_flat"], lids) | np.isin(mm["major_real"], lids)
                if un.any():
                    dpred = np.where(un, -ndi.distance_transform_edt(un), ndi.distance_transform_edt(~un)); bin_pred = Binner(dpred, body)
            row = dict(cell=key, case=case, edema_vox=int(ed.sum()), n_inner=int(inner.sum()), n_border=int(border.sum()))
            for s in ALL_STG:
                for name in ("energy", "lvar"):
                    maps = {}
                    for a in ARMS:
                        k_ = f"{name}_{a}_s{s}"
                        src = zb if s in (0, 1) else zd
                        if src is None or k_ not in src:
                            continue
                        mp = up(src[k_].astype(np.float32), img.shape); mp = mp / max(mp[body].mean(), 1e-12); maps[a] = mp
                        si, sb = mp[inner].sum(), mp[border].sum()
                        row[f"{a}_s{s}_{name}_sum_interior"] = float(si); row[f"{a}_s{s}_{name}_sum_border"] = float(sb); row[f"{a}_s{s}_{name}_int_over_border_sum"] = float(si / sb) if sb > 0 else np.nan
                        row[f"{a}_s{s}_{name}_mean_interior"] = float(mp[inner].mean()) if inner.any() else np.nan; row[f"{a}_s{s}_{name}_mean_border"] = float(mp[border].mean())
                        profs.append(dict(cell=key, case=case, arm=a, stage=s, map=name, ref="gt", **{f"b{b}": v for b, v in zip(BINS[:-1], bin_gt.profile(mp))}))
                        if bin_pred is not None:
                            profs.append(dict(cell=key, case=case, arm=a, stage=s, map=name, ref="pred", **{f"b{b}": v for b, v in zip(BINS[:-1], bin_pred.profile(mp))}))
                    if len(maps) == 2:
                        row[f"diff_s{s}_{name}_sum_interior"] = row[f"real_s{s}_{name}_sum_interior"] - row[f"flat_s{s}_{name}_sum_interior"]
                        row[f"diff_s{s}_{name}_sum_border"] = row[f"real_s{s}_{name}_sum_border"] - row[f"flat_s{s}_{name}_sum_border"]
            if zd is not None:
                for s in CH_STG:
                    for a in ARMS:
                        r = zd[f"chratio_{a}_s{s}"]; chans += [dict(cell=key, case=case, arm=a, stage=s, channel=i, ratio=float(v)) for i, v in enumerate(r)]
                        row[f"{a}_s{s}_chfrac_gt1.2"] = float(np.nanmean(r > 1.2)); row[f"{a}_s{s}_chratio_median"] = float(np.nanmedian(r))
            rows.append(row)
            if case in show:
                try:
                    sheet(key, case, img, ed, dsg, zb, zd, V2OUT / key / f"{case}_maps.npz", od / f"{case}_v3c_sheet.png")
                except Exception as e:
                    log(f"[{key}] sheet {case} failed: {e}")
            if len(rows) % 10 == 0:
                log(f"[{key}] {len(rows)} cases...")
        pd.DataFrame(rows).to_csv(od / "cases_v3c.csv", index=False); pd.DataFrame(profs).to_csv(od / "profiles_v3c.csv", index=False); pd.DataFrame(chans).to_csv(od / "channels_v3c.csv", index=False)
        log(f"[{key}] analysed {len(rows)} cases")


def report_only():
    cells = [c for c in CELL_KEYS if (OUT / c / "cases_v3c.csv").exists()]
    cf = pd.concat([pd.read_csv(OUT / c / "cases_v3c.csv") for c in cells], ignore_index=True); cf.to_csv(OUT / "cases_all.csv", index=False)
    pf = pd.concat([pd.read_csv(OUT / c / "profiles_v3c.csv") for c in cells], ignore_index=True); pf.to_csv(OUT / "profiles_all.csv", index=False)
    ch = pd.concat([pd.read_csv(OUT / c / "channels_v3c.csv") for c in cells if (OUT / c / "channels_v3c.csv").stat().st_size > 1], ignore_index=True); ch.to_csv(OUT / "channels_all.csv", index=False)
    report(cf, pf, ch, cells)


def sheet(key, case, img, ed, dsg, zb, zd, m2path, path):
    z = int(np.argmax(ed.sum((1, 2)))); idx = np.argwhere(ed[z]); lo = np.maximum(idx.min(0) - 20, 0); hi = np.minimum(idx.max(0) + 21, ed.shape[1:])
    win = (slice(lo[0], hi[0]), slice(lo[1], hi[1])); body = img != 0; preds = {}
    if m2path.exists():
        mm = np.load(m2path); preds = {a: np.isin(mm[f"major_{a}"], list(v2.CELLS[key]["lids"]))[z][win] for a in ARMS}
    def norm(a, s, name):
        src = zb if s in (0, 1) else zd
        if src is None or f"{name}_{a}_s{s}" not in src:
            return None
        mp = up(src[f"{name}_{a}_s{s}"].astype(np.float32), img.shape); return mp / max(mp[body].mean(), 1e-12)
    fig, axes = plt.subplots(3, 5, figsize=(24, 14), facecolor="#fcfcfb", constrained_layout=True)
    fg = img[img != 0]; vlo, vhi = np.percentile(fg, [1, 99])
    def contours(ax):
        ax.contour(ed[z][win].T.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=1.3)
        for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
            if a in preds and preds[a].any():
                ax.contour(preds[a].T.astype(float), levels=[0.5], colors=[col], linewidths=0.9, linestyles="dotted")
        ax.axis("off")
    axes[0, 0].imshow(img[z][win].T, cmap="gray", vmin=vlo, vmax=vhi, origin="lower"); contours(axes[0, 0]); axes[0, 0].set_title(f"{case} z={z}: image (zoom), GT green, preds dotted", fontsize=10, loc="left")
    for j, s in enumerate(ALL_STG):
        ef, er = norm("flat", s, "energy"), norm("real", s, "energy")
        if ef is None:
            axes[0, 1 + j].axis("off"); continue
        d = (er - ef)[z][win]; dmax = np.percentile(np.abs(er - ef), 99.5) or 1
        axes[0, 1 + j].imshow(d.T, cmap="RdBu_r", vmin=-dmax, vmax=dmax, origin="lower"); contours(axes[0, 1 + j]); axes[0, 1 + j].set_title(f"real - flat energy s{s} (normalised)", fontsize=10, loc="left")
        vmax = max(np.percentile(ef, 99.5), np.percentile(er, 99.5))
        if j < 2:
            axes[1, 1 + 2 * j].imshow(ef[z][win].T, cmap="inferno", vmin=0, vmax=vmax, origin="lower"); contours(axes[1, 1 + 2 * j]); axes[1, 1 + 2 * j].set_title(f"flat-fill energy s{s}", fontsize=10, loc="left")
            axes[1, 2 + 2 * j].imshow(er[z][win].T, cmap="inferno", vmin=0, vmax=vmax, origin="lower"); contours(axes[1, 2 + 2 * j]); axes[1, 2 + 2 * j].set_title(f"real-fill energy s{s}", fontsize=10, loc="left")
    # profile for this case (s0 energy)
    ax = axes[1, 0]
    for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
        mp = norm(a, 0, "energy")
        if mp is not None:
            ax.plot(BINS[:-1] + 0.5, profile(mp, dsg, body), "-o", ms=3, color=col, label=a)
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("signed distance to GT edema boundary (vox; <0 inside)"); ax.set_ylabel("energy / whole-brain mean"); ax.set_title("radial profile, s0 energy", fontsize=10, loc="left"); ax.legend(fontsize=8)
    # top interior-dominated channels
    for r, a in enumerate(ARMS):
        keys = sorted([k for k in (zd.files if zd is not None else []) if k.startswith(f"topch_{a}_s0_")])[:3]
        axes[2, 0 + 2 * r if r == 0 else 2].axis("off")
        for jj, k in enumerate(keys):
            ax = axes[2, (0 if r == 0 else 3) + jj] if (0 if r == 0 else 3) + jj < 5 else None
            if ax is None:
                break
            m = zd[k].astype(np.float32)[z][win]; ax.imshow(m.T, cmap="inferno", vmin=0, vmax=np.percentile(m, 99.5) or 1, origin="lower"); contours(ax); ax.set_title(f"{a}: top interior-dominated s0 channel {k.split('ch')[-1]}", fontsize=9, loc="left")
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")
    fig.suptitle(f"{v2.CELLS[key]['title']}  |  {case}: zoomed real - flat energy at s0..s3 (each map / its whole-brain mean), energy maps s0/s1, s0 radial profile, top interior-dominated channels", fontsize=11, x=0.01, ha="left", wrap=True)
    fig.savefig(path, dpi=90, facecolor="#fcfcfb"); plt.close(fig)


def report(cf, pf, ch, cells):
    wp = lambda x, y: wilcoxon(x, y).pvalue if len(x) >= 5 and (np.asarray(x) != np.asarray(y)).any() else np.nan
    lines = ["# Visual proof v3c: energy around the lesion boundary -- radial profiles, integrated sums, deeper stages, per-channel", "",
             "Maps normalised per arm by their whole-body mean. Zones from the GT edema: interior d <= -2 vox, border |d| <= 2 vox. SUM = total normalised energy in the zone (area-integrated). Paired Wilcoxon real vs flat over cases.", ""]
    lines += ["## 2. Integrated energy: real - flat of the zone SUMS, and interior_sum / border_sum per arm", "", "| cell | stage | map | n | sum interior flat / real | real > flat | p | sum border flat / real | real > flat | p | int/border sum flat | real | real > flat | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = cf[cf.cell == c]
        for s in ALL_STG:
            for name in ("energy", "lvar"):
                fi, ri = d.get(f"flat_s{s}_{name}_sum_interior"), d.get(f"real_s{s}_{name}_sum_interior")
                if fi is None or ri is None or fi.isna().all():
                    continue
                fb, rb = d[f"flat_s{s}_{name}_sum_border"], d[f"real_s{s}_{name}_sum_border"]; fr, rr = d[f"flat_s{s}_{name}_int_over_border_sum"], d[f"real_s{s}_{name}_int_over_border_sum"]
                lines.append(f"| {c} | s{s} | {name} | {len(d)} | {fi.median():.0f} / {ri.median():.0f} | {(ri > fi).mean():.2f} | {wp(ri, fi):.2g} | {fb.median():.0f} / {rb.median():.0f} | {(rb > fb).mean():.2f} | {wp(rb, fb):.2g} | {fr.median():.2f} | {rr.median():.2f} | {(rr > fr).mean():.2f} | {wp(rr, fr):.2g} |")
    lines.append("")
    lines += ["## 1. Radial profiles (energy / whole-brain mean) at the GT boundary: value at d = -6..-3 (interior), d = -1..+1 (contour), d = +3..+6 (outside), median over cases", "",
              "| cell | stage | ref | interior flat / real | contour flat / real | outside flat / real | contour - interior flat / real | real > flat on (contour - interior) | p |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for s in ALL_STG:
            for ref in ("gt", "pred"):
                p_ = pf[(pf.cell == c) & (pf.stage == s) & (pf["map"] == "energy") & (pf.ref == ref)]
                if len(p_) == 0:
                    continue
                g = lambda a, lo, hi: p_[p_.arm == a][[f"b{b}" for b in range(lo, hi + 1)]].mean(1)
                ci = {a: (g(a, -1, 0) - g(a, -6, -3)) for a in ARMS}
                lines.append(f"| {c} | s{s} | {ref} | {g('flat', -6, -3).median():.3f} / {g('real', -6, -3).median():.3f} | {g('flat', -1, 0).median():.3f} / {g('real', -1, 0).median():.3f} | {g('flat', 3, 6).median():.3f} / {g('real', 3, 6).median():.3f} | "
                             f"{ci['flat'].median():.3f} / {ci['real'].median():.3f} | {(ci['real'].values > ci['flat'].values).mean():.2f} | {wp(ci['real'].values, ci['flat'].values):.2g} |")
    lines.append("")
    if len(ch):
        lines += ["## 4. Per-channel interior/border ratio (mean |activation| inside / border band), distribution over channels per arm", "", "| cell | stage | n channels x cases | flat median [IQR] | real median [IQR] | frac > 1.2 flat / real (per-case median) | real > flat | p (Wilcoxon per case) | Mann-Whitney pooled p |", "|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            d = cf[cf.cell == c]
            for s in CH_STG:
                x = ch[(ch.cell == c) & (ch.stage == s)]; f_, r_ = x[x.arm == "flat"].ratio.dropna(), x[x.arm == "real"].ratio.dropna()
                if len(f_) == 0:
                    continue
                q = lambda v: f"{v.median():.2f} [{v.quantile(.25):.2f}-{v.quantile(.75):.2f}]"; ff, rf = d[f"flat_s{s}_chfrac_gt1.2"], d[f"real_s{s}_chfrac_gt1.2"]
                lines.append(f"| {c} | s{s} | {len(f_)} | {q(f_)} | {q(r_)} | {ff.median():.2f} / {rf.median():.2f} | {(rf > ff).mean():.2f} | {wp(rf, ff):.2g} | {mannwhitneyu(f_, r_).pvalue:.2g} |")
        lines.append("")
    (OUT / "v3c_stats.md").write_text("\n".join(lines)); print("\n".join(lines))
    # figures: profiles
    for name in ("energy", "lvar"):
        for ref in ("gt", "pred"):
            fig, axes = plt.subplots(2 * len(ALL_STG), len(cells), figsize=(4.6 * len(cells), 3.0 * 2 * len(ALL_STG)), facecolor="#fcfcfb", squeeze=False)
            for j, c in enumerate(cells):
                for i, s in enumerate(ALL_STG):
                    p_ = pf[(pf.cell == c) & (pf.stage == s) & (pf["map"] == name) & (pf.ref == ref)]; ax = axes[2 * i, j]; axd = axes[2 * i + 1, j]
                    if len(p_) == 0:
                        ax.axis("off"); axd.axis("off"); continue
                    cols = [f"b{b}" for b in BINS[:-1]]; x = BINS[:-1] + 0.5; curves = {}
                    for a, col in (("flat", "#2a78d6"), ("real", "#eb6834")):
                        v = p_[p_.arm == a].set_index("case")[cols]; curves[a] = v
                        ax.plot(x, v.median(), color=col, lw=1.8, label=f"{a}-fill"); ax.fill_between(x, v.quantile(.25), v.quantile(.75), color=col, alpha=0.15)
                    ax.axvline(0, color="k", lw=0.6); ax.set_title(f"{c}: s{s} {name} vs distance to {ref.upper()} boundary", fontsize=9); ax.set_ylabel("map / whole-brain mean" if j == 0 else "")
                    if i == 0 and j == 0:
                        ax.legend(fontsize=8)
                    common = curves["flat"].index.intersection(curves["real"].index); dd = curves["real"].loc[common] - curves["flat"].loc[common]
                    axd.plot(x, dd.median(), color="#4a3aa7", lw=1.8); axd.fill_between(x, dd.quantile(.25), dd.quantile(.75), color="#4a3aa7", alpha=0.15); axd.axhline(0, color="k", lw=0.6); axd.axvline(0, color="k", lw=0.6)
                    axd.set_title("real - flat", fontsize=9); axd.set_xlabel("signed distance (vox; < 0 inside)")
            fig.suptitle(f"Radial profiles of {name} ({'channel-RMS energy' if name == 'energy' else 'local std of energy'}), normalised per arm by whole-brain mean; median + IQR over 70 cases; boundary = {ref.upper()}", fontsize=11, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"profiles_{name}_{ref}.png", dpi=100, facecolor="#fcfcfb"); plt.close(fig)
    # figure: integrated sums (interior/border sum ratio per arm, per stage)
    fig, axes = plt.subplots(len(ALL_STG), len(cells), figsize=(4.4 * len(cells), 3.4 * len(ALL_STG)), facecolor="#fcfcfb", squeeze=False)
    for j, c in enumerate(cells):
        d = cf[cf.cell == c]
        for i, s in enumerate(ALL_STG):
            ax = axes[i, j]; fr, rr = d.get(f"flat_s{s}_energy_int_over_border_sum"), d.get(f"real_s{s}_energy_int_over_border_sum")
            if fr is None or fr.isna().all():
                ax.axis("off"); continue
            vp = ax.violinplot([fr.dropna(), rr.dropna()], positions=[0, 1], showmedians=True, widths=0.8)
            for b, col in zip(vp["bodies"], ("#2a78d6", "#eb6834")):
                b.set_facecolor(col); b.set_alpha(0.4)
            for a_, b_ in zip(fr, rr):
                ax.plot([0, 1], [a_, b_], color="#999", lw=0.4, alpha=0.5)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["flat-fill", "real-fill"]); ax.set_title(f"{c}: s{s} interior SUM / border SUM, p={wp(rr, fr):.2g}", fontsize=9); ax.set_ylabel("energy sum ratio" if j == 0 else "")
    fig.suptitle("Area-integrated energy: interior sum / border-band sum per arm (normalised maps), paired over 70 cases", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "integrated_int_over_border.png", dpi=100, facecolor="#fcfcfb"); plt.close(fig)
    # figure: channel ratio histograms
    if len(ch):
        fig, axes = plt.subplots(len(CH_STG), len(cells), figsize=(4.4 * len(cells), 3.4 * len(CH_STG)), facecolor="#fcfcfb", squeeze=False)
        for j, c in enumerate(cells):
            for i, s in enumerate(CH_STG):
                x = ch[(ch.cell == c) & (ch.stage == s)]; ax = axes[i, j]
                ax.hist([x[x.arm == "flat"].ratio.dropna(), x[x.arm == "real"].ratio.dropna()], bins=np.linspace(0, 3, 40), color=["#2a78d6", "#eb6834"], alpha=0.6, label=["flat-fill", "real-fill"], density=True)
                ax.axvline(1, color="k", lw=0.6); ax.set_title(f"{c}: s{s} channel interior/border ratio", fontsize=9); ax.set_xlabel("mean |act| interior / border")
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.suptitle("Per-channel interior/border activation ratio, pooled over 70 cases (fold-mean); > 1 = interior-dominated channel", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "channel_ratio_hist.png", dpi=100, facecolor="#fcfcfb"); plt.close(fig)
    log("wrote v3c_stats.md + figures")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=",".join(CELL_KEYS)); ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--report", action="store_true"); ap.add_argument("--case-slice", default=None); a = ap.parse_args()
    if a.analyze:
        analyze(a.cells.split(",")); return
    if a.report:
        report_only(); return
    dev = torch.device("cuda")
    for key in a.cells.split(","):
        cases = None
        if a.case_slice:
            i_, j_ = (int(x) if x else None for x in a.case_slice.split(":")); cases = v2.CELLS[key]["cases"][i_:j_]
        try:
            run_gpu_cell(key, dev, cases)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"[{key}] FAILED: {e}")


if __name__ == "__main__":
    main()
