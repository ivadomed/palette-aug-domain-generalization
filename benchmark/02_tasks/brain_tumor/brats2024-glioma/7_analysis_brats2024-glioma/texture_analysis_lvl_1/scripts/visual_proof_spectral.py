#!/usr/bin/env python
"""
Fourier-sensitivity radial profile (Krishnamachari, Ng & Foo, TMLR 2022) and wavelet scale attribution (WCAM, Kasmi et al. 2023 / WAM, ICML 2025)
of the input gradient, noise fill (rung 4) vs flat fill (rung 4.5) vs real fill (rung 5), every held-out-contrast cell of every paper task
(FINDINGS section 34, pre-registered; LITERATURE_GRADIENT_TEXTURE.md section 3 items 1-2). Natural test images, no intervention.
Same cells, patches, structures, models, folds and pairwise matching as visual_proof_hires3.py (section 33).
Per case: one plan-sized patch centred on the target classes; CONTROL region = body voxels with GT label 0 at distance >= 5 vox from every labelled
voxel (all target classes and any other labelled class, e.g. the glioma core), one per case. Per model and fold, one forward pass:
  G_bg = d J_bg / dx, J_bg = mean log-odds of class 0 over the control region (NaN if the control region is empty);
  G_c  = d J_c / dx,  J_c  = mean log-odds of class c over its GT in the patch, per present target class c (>= 100 vox, the 12 largest).
(a) radial profile: |FFT3((G - mean_body G) * body)|^2 binned by r = sqrt(fx^2+fy^2+fz^2) (cycles/voxel), 16 equal bins over [0, 0.5*sqrt(3)],
    normalised to sum 1 -> columns fc_00..15 (G_c), fbg_00..15 (G_bg); image profiles of (img - mean_Mc) * M_c and (img - mean_ctrl) * M_ctrl ->
    ic_00..15, ictrl_00..15. HF share = spectral mass at r >= 0.25 cyc/vox (hf_c, hf_bg, hf_ex = hf_c - hf_bg).
(b) wavelet: 3-level orthonormal 3D Haar DWT of G * body (numpy; exact 2^l block support, so the nearest-neighbour localisation of each level's
    detail energy is exact; pywt is not installed in Killarney's venv). Per level l (1 = finest): detail energy of each 2^l block (7 sub-bands)
    spread evenly over its voxels. wf_in = level-1 energy in M_c / all-level energy in M_c; wf_ctrl the same in the control; wf_ex = wf_in - wf_ctrl;
    w{1,2,3,A}_in = share of that level's energy located in M_c; w{..}_ctrl in the control; bg_* = the same for G_bg.
Rows: (cell, task, setting, case, struct, vox, arm, fold, dice, hit, J, J_bg, ...). The report builds the band-restricted excess, radial.npz per cell,
the tables and the figures. No image is written by the extraction.
  GPU: --cells <cell> [--case-slice i:j]       CPU: --report
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402

e7, v2 = vt.e7, vt.v2; FOLDS = (0, 1, 2)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
LAB = {"noise": "noise fill (rung 4)", "flat": "flat fill (rung 4.5)", "real": "real fill (rung 5)"}
OUT = Path(os.environ.get("VP_SPECTRAL_OUT") or (vt.OUT.parent / "spectral")); OUT.mkdir(parents=True, exist_ok=True)
MAX_CASES = int(os.environ.get("SPECTRAL_MAX_CASES", 40)); MAX_STRUCT = 12; MIN_VOX = 100; MATCH = 0.05
CTRL_D = 5; NB = 16; RMAX = 0.5 * np.sqrt(3); HF_R = 0.25; LEVELS = 3
EDGES = np.linspace(0, RMAX, NB + 1); CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
BAND_BINS = [b for b in range(1, NB) if CENTRES[b] < 0.5]; BAND_K = 4   # band rule as AMENDED after the 2-case smoke test, before extraction (section 34): bins 1-8 (inside the Nyquist sphere)
BAND_BINS_ORIG = list(range(1, NB))   # band rule as first pre-registered (bin 0 excluded, all other bins): reported alongside as band_ex_orig
PCOLS = {p: [f"{p}_{k:02d}" for k in range(NB)] for p in ("fc", "fbg", "ic", "ictrl")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setting(key):
    return "_".join(key.split("_")[:2])   # glioma_t1n_t2f -> glioma_t1n (TEX_STEP key)


def models(key):
    """flat / real from vt.CELLS; noise = the single *lblvor* run next to them (same prediction root)."""
    cell = vt.CELLS[key]; flat, real = cell["models"]["flat"](), cell["models"]["real"]()
    runs = sorted(p for p in flat.parents[2].glob("*lblvor*") if p.is_dir()); assert len(runs) == 1, (key, runs)
    noise = sorted(runs[0].glob("Dataset*/*__3d_fullres")); assert len(noise) == 1, (key, noise)
    return {"noise": noise[0], "flat": flat, "real": real}


def fine_share(G, body):
    hp = G - ndi.gaussian_filter(G, 1.0); return float((hp[body] ** 2).sum() / ((G[body] ** 2).sum() + 1e-30))


# ------------------------------------------------------------------------------------------------------------------------------ spectra
_RCACHE = {}


def rgrid(shape):
    """radial frequency (cycles/voxel) on the rfftn grid, bin index, rfft double-count weights, HF mask; cached per shape."""
    if shape not in _RCACHE:
        f = [np.fft.fftfreq(n) for n in shape[:-1]] + [np.fft.rfftfreq(shape[-1])]
        g = np.meshgrid(*f, indexing="ij"); r = np.sqrt(sum(x ** 2 for x in g))
        w = np.full(len(f[-1]), 2.0); w[0] = 1.0
        if shape[-1] % 2 == 0:
            w[-1] = 1.0
        W = np.broadcast_to(w, r.shape)
        idx = np.minimum((r / (RMAX / NB)).astype(int), NB - 1)
        _RCACHE[shape] = (idx.ravel(), W.ravel().astype(np.float64), (r >= HF_R).ravel())
    return _RCACHE[shape]


def radial(x, mask):
    """x restricted to mask, mean over the mask removed -> (16-bin profile normalised to 1, HF share). NaN if the mask is empty or x is ~0."""
    if mask.sum() == 0:
        return np.full(NB, np.nan), np.nan
    y = np.where(mask, x - x[mask].mean(), 0.0).astype(np.float32)
    from scipy import fft as sfft
    P = np.abs(sfft.rfftn(y, workers=4)) ** 2; idx, W, hf = rgrid(y.shape); Pw = P.ravel().astype(np.float64) * W
    tot = Pw.sum()
    if not tot > 0:
        return np.full(NB, np.nan), np.nan
    prof = np.bincount(idx, weights=Pw, minlength=NB)[:NB] / tot
    return prof, float(Pw[hf].sum() / tot)


# ------------------------------------------------------------------------------------------------------------------------------ wavelet
def haar_levels(x):
    """orthonormal 3D Haar DWT, LEVELS levels, periodic-free (x padded to a multiple of 2^LEVELS by the caller).
    returns [detail-energy per 2^l block for l = 1..LEVELS] + [approximation energy per 2^LEVELS block]; Parseval: sum of all = sum x^2."""
    out, a = [], x.astype(np.float64)
    for _ in range(LEVELS):
        s = a.shape; b = a.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2)
        ssq = (b ** 2).sum((1, 3, 5)); a = b.sum((1, 3, 5)) / np.sqrt(8.0)
        out.append(ssq - a ** 2)   # the 7 detail sub-bands of each block = block energy - approximation energy
    out.append(a ** 2)
    return out


def block_count(m, l):
    s = m.shape; f = 2 ** l
    return m.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).sum((1, 3, 5)).astype(np.float64) / f ** 3   # share of each block's voxels in m


def pad8(x):
    f = 2 ** LEVELS; pw = [(0, (-n) % f) for n in x.shape]
    return np.pad(x, pw) if any(w for _, w in pw) else x


def mask_counts(m):
    """padded bool mask -> [share of each 2^l block in the mask, l = 1..LEVELS, then LEVELS again for the approximation]; None if empty."""
    if not m.any():
        return None
    bc = [block_count(m, l) for l in range(1, LEVELS + 1)]
    return bc + [bc[-1]]


def wav_stats(Es, masks):
    """Es: haar_levels output; masks: {name: mask_counts output}. Returns per mask: energy in the mask per level (l1, l2, l3, A)."""
    out = {}
    for nm, bc in masks.items():
        if bc is None:
            out[nm] = np.full(LEVELS + 1, np.nan); continue
        out[nm] = np.array([(E * b).sum() for E, b in zip(Es, bc)])
    out["_tot"] = np.array([E.sum() for E in Es])
    return out


def wav_row(Es, Ein, Ectrl, pre=""):
    tot = Es["_tot"]; r = {}
    sin, sct = Ein.sum(), Ectrl.sum()
    r[f"{pre}wf_in"] = float(Ein[0] / sin) if sin > 0 else np.nan
    r[f"{pre}wf_ctrl"] = float(Ectrl[0] / sct) if sct > 0 else np.nan
    r[f"{pre}wf_ex"] = r[f"{pre}wf_in"] - r[f"{pre}wf_ctrl"]
    for i, nm in enumerate(("1", "2", "3", "A")):
        r[f"{pre}w{nm}_in"] = float(Ein[i] / tot[i]) if tot[i] > 0 else np.nan
        r[f"{pre}w{nm}_ctrl"] = float(Ectrl[i] / tot[i]) if tot[i] > 0 else np.nan
    return r


# ------------------------------------------------------------------------------------------------------------------------------ extraction
def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in models(key).items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]) if key.startswith("glioma") else list(cell["cases"])[:MAX_CASES]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "metrics.csv"; done = set(pd.read_csv(f_csv).case.astype(str)) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    f_skip = od / "skipped.csv"
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tgt = np.isin(gt, T)
        if tgt.sum() < MIN_VOX:
            pd.DataFrame([dict(case=case, why="target < MIN_VOX")]).to_csv(f_skip, mode="a", index=False, header=not f_skip.exists()); continue
        box = e7.patch_at(tgt, ps, img.shape); ip, gp = img[box].astype(np.float32), gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, gp = np.pad(ip, padw), np.pad(gp, padw)
        body = ip != 0 if (ip == 0).mean() > 0.05 else np.ones(ip.shape, bool)
        sizes = {int(c): int((gp == c).sum()) for c in np.unique(gp) if c in T and (gp == c).sum() >= MIN_VOX}
        structs = sorted(sizes, key=sizes.get, reverse=True)[:MAX_STRUCT]
        if not structs:
            pd.DataFrame([dict(case=case, why="no structure >= MIN_VOX in the patch")]).to_csv(f_skip, mode="a", index=False, header=not f_skip.exists()); continue
        lab = gp > 0; ctrl = body & ~lab & (ndi.distance_transform_edt(~lab) >= CTRL_D) if lab.any() else body.copy()
        nctrl = int(ctrl.sum())
        Mc = {c: (gp == c) & body for c in structs}
        img_c = {c: radial(ip, Mc[c])[0] for c in structs}; img_ctrl = radial(ip, ctrl)[0]
        P = {nm: mask_counts(pad8(m)) for nm, m in [("ctrl", ctrl)] + [(c, Mc[c]) for c in structs]}
        rows = []; parse = []
        for a, pr in prs.items():
            net = pr.network
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                x = torch.from_numpy(ip)[None, None].to(dev).requires_grad_(True)
                with torch.enable_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, x).float(); pred = o.argmax(1)[0].cpu().numpy()
                    if nctrl > 0:
                        Jbg = e7.log_odds(o, 0)[0][torch.from_numpy(ctrl).to(dev)].mean()
                        Gbg = torch.autograd.grad(Jbg, x, retain_graph=True)[0][0, 0].detach().cpu().numpy(); Jbg = float(Jbg.detach())
                        fbg, hfbg = radial(Gbg, body); Ebg = haar_levels(pad8(Gbg * body)); Wbg = wav_stats(Ebg, P)   # all structures + control at once
                    else:
                        Jbg, fbg, hfbg, Wbg = np.nan, np.full(NB, np.nan), np.nan, None
                    for c in structs:
                        m = gp == c; lo = e7.log_odds(o, c)[0]; J = lo[torch.from_numpy(m).to(dev)].mean()
                        G = torch.autograd.grad(J, x, retain_graph=True)[0][0, 0].detach().cpu().numpy(); lov = lo.detach().cpu().numpy()
                        fc, hfc = radial(G, body); Gm = pad8(G * body); Es = haar_levels(Gm); W = wav_stats(Es, {c: P[c], "ctrl": P["ctrl"]})
                        parse.append(abs(W["_tot"].sum() - float((Gm.astype(np.float64) ** 2).sum())) / (float((Gm.astype(np.float64) ** 2).sum()) + 1e-30))
                        row = dict(cell=key, task=cell["task"], setting=setting(key), case=case, struct=c, vox=sizes[c], ctrl_vox=nctrl, arm=a, fold=k,
                                   dice=vt.dice(pred == c, m), hit=float((lov[m] > 0).mean()), J=float(J), J_bg=Jbg,
                                   hf_c=hfc, hf_bg=hfbg, hf_ex=hfc - hfbg, g_fine=fine_share(G, body))
                        row.update(wav_row(W, W[c], W["ctrl"]))
                        if Wbg is not None:
                            row.update(wav_row(Wbg, Wbg[c], Wbg["ctrl"], "bg_"))
                        else:
                            row.update({kk: np.nan for kk in wav_row(W, W[c], W["ctrl"], "bg_")})
                        row.update(dict(zip(PCOLS["fc"], fc))); row.update(dict(zip(PCOLS["fbg"], fbg)))
                        row.update(dict(zip(PCOLS["ic"], img_c[c]))); row.update(dict(zip(PCOLS["ictrl"], img_ctrl)))
                        rows.append(row)
                del o, x; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not f_csv.exists()); done.add(case)
        r = pd.DataFrame(rows).groupby("arm")[["dice", "hf_c", "hf_bg", "wf_ex"]].mean()
        log(f"[{key}] {case} ({len(structs)} structures, ctrl {nctrl} vox): dice " + " / ".join(f"{r.loc[a, 'dice']:.2f}" for a in ARMS)
            + " | HF(Gc) " + " / ".join(f"{r.loc[a, 'hf_c']:.3f}" for a in ARMS) + " | HF(Gbg) " + " / ".join(f"{r.loc[a, 'hf_bg']:.3f}" for a in ARMS)
            + " | wf_in-wf_ctrl " + " / ".join(f"{r.loc[a, 'wf_ex']:+.3f}" for a in ARMS) + f" | Parseval max rel. err {max(parse):.1e}; {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------------------- report
METS = [("hf_c", "HF share (r >= 0.25 cyc/vox) of G_c"), ("hf_bg", "HF share of G_bg (control-region background decision)"),
        ("hf_ex", "HF(G_c) - HF(G_bg): structure-specific HF excess"), ("band_c", "G_c spectral mass in the structure-vs-control image band"),
        ("band_ex", "band mass G_c - G_bg: band-restricted structure-specific excess (amended band, bins 1-8)"),
        ("band_ex_orig", "band mass G_c - G_bg with the band as first pre-registered (bins 1-15; sensitivity)"),
        ("wf_in", "wavelet: level-1 (finest) fraction of G_c energy inside the structure"), ("wf_ctrl", "wavelet: level-1 fraction of G_c energy inside the control"),
        ("wf_ex", "wavelet: fine_in - fine_ctrl of G_c (space x scale interaction)"), ("w1_in", "wavelet: share of the level-1 energy of G_c located in the structure"),
        ("bg_wf_ex", "wavelet: fine_in - fine_ctrl of G_bg"), ("g_fine", "fine-detail share of G_c (section 32/33 scalar, check)"),
        ("dice", "in-patch Dice"), ("hit", "structure called (share)")]
ORDER = ["glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t2f", "glioma_t2w_t1n", "ms_flair_t2w", "ms_flair_t1w", "ms_t1w_flair", "ms_t1w_t2w", "breast_t1wce_t2w", "breast_t2w_t1wce",
         "brain_t1w_t2w", "brain_t2w_t1w", "abd_t1in_t2spir", "abd_t1in_ct", "abd_t2spir_t1in", "abd_t2spir_t1out", "abd_t2spir_ct", "mand_cbct_hanseg_ct", "mand_cbct_pddca_ct",
         "pelvis_mri_ct", "pelvis_ct_mri", "pancreas_t1wce_t2w", "pancreas_t2w_t1wce"]
HELPS = ["glioma_t1n_t2w", "glioma_t1n_t2f", "glioma_t2w_t2f", "breast_t2w_t1wce", "pelvis_ct_mri", "abd_t2spir_t1in", "abd_t2spir_t1out", "abd_t2spir_ct"]
HURTS = ["glioma_t2w_t1n", "brain_t1w_t2w", "brain_t2w_t1w", "pelvis_mri_ct", "pancreas_t1wce_t2w"]
UNUSABLE = ["ms_flair_t2w", "ms_flair_t1w", "ms_t1w_flair", "ms_t1w_t2w", "abd_t1in_t2spir", "abd_t2spir_t1in", "abd_t2spir_t1out"]


def group(key):
    return "unusable" if key in UNUSABLE else "helps" if key in HELPS else "hurts/nil" if key in HURTS else "other"


def load_all():
    D = []
    for key in ORDER:
        f = OUT / key / "metrics.csv"
        if f.exists() and f.stat().st_size > 0:
            d = pd.read_csv(f, low_memory=False); d = d[d.case.astype(str) != "case"]   # guard against a duplicated header from concurrent chunk appends
            d["case"] = d.case.astype(str)
            for c in d.columns:
                if c not in ("cell", "task", "setting", "case", "arm"):
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            D.append(d.drop_duplicates(subset=["case", "struct", "arm", "fold"], keep="last"))
    return pd.concat(D, ignore_index=True) if D else pd.DataFrame()


def bands(D, cand_bins):
    """per cell: pre-registered band = the BAND_K bins (bin 0 excluded) with the largest median over (case, structure) of log(ic / ictrl)."""
    out = {}
    for key, d in D.groupby("cell"):
        u = d.drop_duplicates(["case", "struct"]); ic, ik = u[PCOLS["ic"]].values, u[PCOLS["ictrl"]].values
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.nanmedian(np.log(ic / ik), 0)
        cand = [b for b in cand_bins if np.isfinite(lr[b])]
        out[key] = (sorted(sorted(cand, key=lambda b: -lr[b])[:BAND_K]), lr)
    return out


def per_case(D, m, ref, other, matched):
    """per (cell, case): median over structures of metric m for arms ref and other; matched = keep structures with |dice_ref - dice_other| <= MATCH."""
    S = D.groupby(["cell", "case", "struct", "arm"]).mean(numeric_only=True).unstack("arm")
    ok = (S["dice"][ref] - S["dice"][other]).abs() <= MATCH if matched else pd.Series(True, index=S.index)
    S = S[ok & S[m][ref].notna() & S[m][other].notna()]
    C = S.groupby(level=["cell", "case"]).median(numeric_only=True)
    return C[m][ref], C[m][other], S.groupby(level=["cell", "case"]).size()


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon, spearmanr
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    D = load_all()
    if D.empty:
        print("no data"); return
    cells = [c for c in ORDER if c in set(D.cell)]
    B, Bo = bands(D, BAND_BINS), bands(D, BAND_BINS_ORIG)
    for sfx, BB in (("", B), ("_orig", Bo)):
        D["band_c" + sfx] = np.nan; D["band_bg" + sfx] = np.nan
        for key, (bb, _) in BB.items():
            s = D.cell == key; D.loc[s, "band_c" + sfx] = D.loc[s, [PCOLS["fc"][b] for b in bb]].sum(1, min_count=1); D.loc[s, "band_bg" + sfx] = D.loc[s, [PCOLS["fbg"][b] for b in bb]].sum(1, min_count=1)
        D["band_ex" + sfx] = D["band_c" + sfx] - D["band_bg" + sfx]
    # radial.npz per cell (fold-mean profiles per case, structure, arm)
    for key in cells:
        d = D[D.cell == key]; g = d.groupby(["case", "struct", "arm"]).mean(numeric_only=True).reset_index()
        np.savez_compressed(OUT / key / "radial.npz", bin_centres=CENTRES, bin_edges=EDGES, band_bins=np.array(B[key][0]), band_bins_orig=np.array(Bo[key][0]), band_logratio=B[key][1],
                            case=g.case.values.astype(str), struct=g.struct.values, arm=g.arm.values.astype(str),
                            **{p: g[PCOLS[p]].values for p in PCOLS})
    L = ["# Fourier-sensitivity radial profile and wavelet scale attribution of the input gradient, noise vs flat vs real fill, all tasks (FINDINGS section 34, pre-registered)", "",
         "Per (case, structure): fold-mean; per case: median over structures; differences: median of per-case differences, Wilcoxon over cases, share of cases > 0. "
         f"MATCHED = structures with |Dice_a - Dice_real| <= {MATCH} for the pair (a, real) (pairwise matching, pre-registered primary). Texture step 4.5 -> 5 = paper tab:flatfill (Dice points). "
         f"Group: helps / hurts/nil / other / unusable as pre-registered in section 34. Radial bins: {NB} equal bins over [0, {RMAX:.3f}] cycles/voxel, width {RMAX / NB:.4f}; HF = mass at r >= {HF_R}.", "",
         "## Pre-registered image band per cell (bins where the structure's image spectrum exceeds the control's the most; median log ratio)", "",
         "| cell | band bins (amended, 1-8) | band centres (cyc/vox) | median log(ic/ictrl) in band | band bins as first pre-registered (1-15) |", "|---|---|---|---|---|"]
    for key in cells:
        bb, lr = B[key]; L.append(f"| {key} | {bb} | {', '.join(f'{CENTRES[b]:.3f}' for b in bb)} | {', '.join(f'{lr[b]:+.2f}' for b in bb)} | {Bo[key][0]} |")
    L.append("")
    summ = []
    for m, nm in METS:
        L += [f"## {nm}", "", "| cell | group | tex. step | n (matched n-r / f-r) | noise | flat | real | real - noise MATCHED (p; >0) | real - flat MATCHED (p; >0) | real - noise ALL (p) | real - flat ALL (p) | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in cells:
            d = D[D.cell == key]; out = {}
            for other in ("noise", "flat"):
                for matched in (True, False):
                    r, o, n = per_case(d, m, "real", other, matched); diff = (r - o).dropna()
                    out[(other, matched)] = dict(med=diff.median() if len(diff) else np.nan, p=pw(diff.values) if len(diff) else np.nan, pos=(diff > 0).mean() if len(diff) else np.nan, n=len(diff), ref=r.median() if len(r) else np.nan, oth=o.median() if len(o) else np.nan)
            rn, rf = out[("noise", True)], out[("flat", True)]
            sig = lambda e: (not np.isnan(e["p"])) and e["p"] < 0.05   # noqa: E731
            if sig(rn) and sig(rf) and np.sign(rn["med"]) == np.sign(rf["med"]):
                verdict = "texture-specific (+)" if rn["med"] > 0 else "texture-specific (-)"
            elif sig(rf) and not sig(rn):
                verdict = "robustness-like (noise ~ real)"
            elif sig(rn) and not sig(rf):
                verdict = "noise differs only"
            else:
                verdict = "n.s."
            step = vt.TEX_STEP.get(setting(key), np.nan)
            L.append(f"| {key} | {group(key)} | {step:+.2f} | {rn['n']} / {rf['n']} | {rn['oth']:.3f} | {rf['oth']:.3f} | {rf['ref']:.3f} | {rn['med']:+.3f} ({rn['p']:.2g}; {rn['pos']:.0%}) | {rf['med']:+.3f} ({rf['p']:.2g}; {rf['pos']:.0%}) | "
                     f"{out[('noise', False)]['med']:+.3f} ({out[('noise', False)]['p']:.2g}) | {out[('flat', False)]['med']:+.3f} ({out[('flat', False)]['p']:.2g}) | {verdict} |")
            summ.append(dict(metric=m, cell=key, task=d.task.iloc[0], group=group(key), step=step, d_rn=rn["med"], p_rn=rn["p"], n_rn=rn["n"], d_rf=rf["med"], p_rf=rf["p"], n_rf=rf["n"],
                             d_rn_all=out[("noise", False)]["med"], p_rn_all=out[("noise", False)]["p"], d_rf_all=out[("flat", False)]["med"], p_rf_all=out[("flat", False)]["p"],
                             noise=rn["oth"], flat=rf["oth"], real=rf["ref"], verdict=verdict))
        S = pd.DataFrame([s for s in summ if s["metric"] == m]); S = S[~S.cell.isin(UNUSABLE)].dropna(subset=["d_rn", "step"])
        if len(S) >= 5:
            r1, p1 = spearmanr(S.d_rn, S.step); Sf = S.dropna(subset=["d_rf"]); r2, p2 = spearmanr(Sf.d_rf, Sf.step)
            L += ["", f"Across usable cells, Spearman(real - noise MATCHED, texture step) = {r1:+.2f} (p {p1:.2g}; n = {len(S)} cells); Spearman(real - flat MATCHED, texture step) = {r2:+.2f} (p {p2:.2g}; n = {len(Sf)} cells)."]
        L.append("")
    SM = pd.DataFrame(summ); SM.to_csv(OUT / "spectral_summary.csv", index=False)
    # ---- per-case Spearman: (wf_ex_real - wf_ex_flat) vs (dice_real - dice_flat), UNMATCHED; same for hf_ex and band_ex
    L += ["## Per-case Spearman of the real - flat metric shift with the real - flat Dice gain (UNMATCHED structures; per case median over structures)", "",
          "| cell | group | n | wf_ex: rho (p) | hf_ex: rho (p) | band_ex: rho (p) | band_ex_orig: rho (p) |", "|---|---|---|---|---|---|---|"]
    pcs = []
    for key in cells:
        d = D[D.cell == key]; res = {}
        for m in ("wf_ex", "hf_ex", "band_ex", "band_ex_orig"):
            r, f, _ = per_case(d, m, "real", "flat", False); dr, df_, _ = per_case(d, "dice", "real", "flat", False)
            j = pd.concat([(r - f).rename("x"), (dr - df_).rename("y")], axis=1).dropna()
            res[m] = spearmanr(j.x, j.y) if len(j) >= 5 and j.x.nunique() > 1 and j.y.nunique() > 1 else (np.nan, np.nan); res["n"] = len(j)
        pcs.append(dict(cell=key, group=group(key), n=res["n"], **{f"rho_{m}": res[m][0] for m in ("wf_ex", "hf_ex", "band_ex", "band_ex_orig")}, **{f"p_{m}": res[m][1] for m in ("wf_ex", "hf_ex", "band_ex", "band_ex_orig")}))
        L.append(f"| {key} | {group(key)} | {res['n']} | " + " | ".join(f"{res[m][0]:+.2f} ({res[m][1]:.2g})" for m in ("wf_ex", "hf_ex", "band_ex", "band_ex_orig")) + " |")
    pd.DataFrame(pcs).to_csv(OUT / "spectral_percase_spearman.csv", index=False)
    # ---- D-alignment (literature review #1 extension, secondary): per (case, struct) Spearman over bins 1..15 of (fc_real - fc_flat) with log(ic / ictrl)
    L += ["", "## Secondary: D-alignment per case = Spearman over bins 1-15 of (G_c profile real - flat) with log(image structure / control profile); median over structures; Wilcoxon vs 0", "",
          "| cell | group | n | median rho | p | share > 0 |", "|---|---|---|---|---|---|"]
    for key in cells:
        d = D[D.cell == key]; g = d.groupby(["case", "struct", "arm"]).mean(numeric_only=True)
        vals = {}
        for (case, st), gg in g.groupby(level=["case", "struct"]):
            arms_ = gg.index.get_level_values("arm")
            if "real" not in arms_ or "flat" not in arms_:
                continue
            dr = gg.xs("real", level="arm")[PCOLS["fc"]].values[0] - gg.xs("flat", level="arm")[PCOLS["fc"]].values[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                Dk = np.log(gg[PCOLS["ic"]].values[0] / gg[PCOLS["ictrl"]].values[0])
            ok = np.isfinite(dr) & np.isfinite(Dk); ok[0] = False
            if ok.sum() >= 5:
                vals.setdefault(case, []).append(spearmanr(dr[ok], Dk[ok])[0])
        v = np.array([np.nanmedian(x) for x in vals.values()]); v = v[np.isfinite(v)]
        L.append(f"| {key} | {group(key)} | {len(v)} | {np.median(v) if len(v) else np.nan:+.2f} | {pw(v) if len(v) else np.nan:.2g} | {(v > 0).mean() if len(v) else np.nan:.0%} |")
    # ---- coverage
    L += ["", "## Coverage", "", "| cell | cases with data | structures (case-level mean) | rows | cases with empty control | skipped |", "|---|---|---|---|---|---|"]
    for key in cells:
        d = D[D.cell == key]; sk = OUT / key / "skipped.csv"; ns = len(pd.read_csv(sk)) if sk.exists() and sk.stat().st_size > 0 else 0
        L.append(f"| {key} | {d.case.nunique()} | {d.groupby('case').struct.nunique().mean():.1f} | {len(d)} | {int((d.groupby('case').ctrl_vox.first() == 0).sum())} | {ns} |")
    (OUT / "spectral.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # ---- figure 1: mean radial profiles per cell
    nc = 5; nr = int(np.ceil(len(cells) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.6 * nr), squeeze=False)
    for ax, key in zip(axs.ravel(), cells):
        d = D[D.cell == key]; g = d.groupby(["case", "struct", "arm"]).mean(numeric_only=True).reset_index()
        for a in ARMS:
            ga = g[g.arm == a]
            ax.plot(CENTRES, np.nanmean(ga[PCOLS["fc"]].values, 0), "-", color=PAL[a], lw=1.1, label=f"{a}: G_c")
            ax.plot(CENTRES, np.nanmean(ga[PCOLS["fbg"]].values, 0), "--", color=PAL[a], lw=0.9, label=f"{a}: G_bg")
        u = g.drop_duplicates(["case", "struct"])
        ax.plot(CENTRES, np.nanmean(u[PCOLS["ic"]].values, 0), "-", color="0.45", lw=1.0, label="image: structure")
        ax.plot(CENTRES, np.nanmean(u[PCOLS["ictrl"]].values, 0), "--", color="0.45", lw=1.0, label="image: control")
        for b in B[key][0]:
            ax.axvspan(EDGES[b], EDGES[b + 1], color="#fff2b3", zorder=-10)
        ax.axvline(HF_R, color="0.7", lw=0.6, ls=":"); ax.set_yscale("log"); ax.tick_params(labelsize=6)
        ax.set_title(f"{key} ({group(key)}, step {vt.TEX_STEP.get(setting(key), np.nan):+.2f})", fontsize=7)
    for ax in axs.ravel()[len(cells):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=5, ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("radial frequency (cycles/voxel)", fontsize=7)
    fig.suptitle("Normalised radial power spectrum of the input gradient (mean over case x structure, fold-mean): noise (green) / flat (blue) / real (red); G_c solid, G_bg dashed; "
                 "image structure / control grey; yellow = pre-registered image band; dotted = HF threshold 0.25", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "spectral_profiles.png", dpi=150); plt.close(fig)
    # ---- figure 2: per-cell three-arm medians (matched structures for the pair with real)
    KEYM = [("hf_c", "HF share of G_c"), ("hf_bg", "HF share of G_bg"), ("hf_ex", "HF(G_c) - HF(G_bg)"), ("band_ex", "band excess\nG_c - G_bg"),
            ("wf_ex", "wavelet fine_in -\nfine_ctrl (G_c)"), ("w1_in", "level-1 energy\nshare in structure"), ("dice", "in-patch Dice")]
    fig, axs = plt.subplots(len(KEYM), 1, figsize=(0.62 * len(cells) + 2, 2.1 * len(KEYM)), sharex=True)
    for ax, (m, nm) in zip(axs, KEYM):
        for i, key in enumerate(cells):
            d = D[D.cell == key]; s = SM[(SM.metric == m) & (SM.cell == key)].iloc[0]
            for xi, a in enumerate(ARMS):
                r, o, _ = per_case(d, m, "real", a, True) if a != "real" else per_case(d, m, "real", "flat", True)
                v = (o if a != "real" else r).dropna().values
                if len(v) == 0:
                    continue
                ax.scatter(np.full(len(v), i + (xi - 1) * 0.25) + np.random.default_rng(xi).uniform(-0.06, 0.06, len(v)), v, s=3, color=PAL[a], alpha=0.3, lw=0)
                ax.plot(i + (xi - 1) * 0.25, np.median(v), "o", color="k", mfc=PAL[a], ms=4)
            stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""   # noqa: E731
            y = ax.get_ylim()[1]
            ax.text(i, y, f"{stars(s.p_rn)}\n{stars(s.p_rf)}", ha="center", va="top", fontsize=5.5, color="k")
        ax.set_ylabel(nm, fontsize=7); ax.tick_params(labelsize=7)
        for i, key in enumerate(cells):
            st = vt.TEX_STEP.get(setting(key), 0)
            ax.axvspan(i - 0.5, i + 0.5, color=("#ffe8e8" if st > 1 else "#e8f0ff" if st < -0.5 else "#f4f4f4"), zorder=-10)
    axs[-1].set_xticks(range(len(cells))); axs[-1].set_xticklabels(cells, rotation=90, fontsize=6.5)
    fig.suptitle("noise (green) / flat (blue) / real (red) fill models; matched structures; stars = Wilcoxon real vs noise (top) and real vs flat (bottom). Background: red = texture step > +1 Dice, blue = < -0.5, grey = between.", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "spectral_numbers.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        import torch
        for c in a.cells:
            run_cell(c, torch.device("cuda"), a.case_slice)


if __name__ == "__main__":
    main()
