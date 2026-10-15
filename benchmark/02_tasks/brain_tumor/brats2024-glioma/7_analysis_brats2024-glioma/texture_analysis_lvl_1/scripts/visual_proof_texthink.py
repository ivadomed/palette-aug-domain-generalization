#!/usr/bin/env python
"""Visual proof v15 (FINDINGS 42, pre-registered in outputs/plots/texthink/PREREG.md): "what the model thinks about texture" -- texture-evidence maps
by activation patching, BraTS cells, arms noise (rung 4, lblvor) / flat (4.5) / real (5), folds 0-2 (fold-mean maps).

Counterfactuals (mask-free, label-free, on the v10 plan patch; fine = (x - G1(x)) * body, Emap = G2(fine^2), m = body mean of fine^2, eps = 1e-3 m):
  x_eq  texture-energy map FLATTENED, local patterns kept: fine_eq = fine * g, g = sqrt((m + eps) / (G2(fine_eq^2) + eps)) re-estimated and applied
        twice, body-masked; x_eq = x - fine + fine_eq (no global rescale).
  x_le  fine STRUCTURE scrambled, local energy map kept: phase-randomised fine (rfftn amplitude kept, phases from the rfftn of real white noise so the
        inverse is real, DC kept, seed from the case name), local gain matched to the ORIGINAL Emap (sigma 2, two passes), body mask, body energy
        rescaled to the original; x_le = x - fine + fine_le.
Per case, arm, fold: activations of x, x_eq, x_le; per-voxel target log-odds J(x); E_full = J(x) - J(x_eq); E_L = J(x) - J(x with the activation at
L in LOCS (skips s0..s5, decoder stages d0..d4) replaced by its x_eq-run value) (positive = the texture-energy map, as seen at L, pushes the belief UP).
Same with x_le -> S (structure evidence). All maps fold-mean; scalars (target / ring 1-10 vox / far background means, fraction of target voxels with
E > 1 nat) are read on the fold-mean maps. Units at s2 / s3: per-channel relative displacement under x_eq (fold mean), top-5 channels and their
Spearman (fold 0, stage resolution) with Emap residualised on G2(x) and |grad G2(x)| over body voxels.
Added after the first full run (coordinator request): x_scr = plain global-energy fine scrambling (phase-randomised fine, same phases as x_le, body mask,
body energy rescaled, no local gain, not iterated) -> E_scr at full and every loc, second pass --scr writing <case>_texthink_scr.csv + maps npz keys.
TT_ITER (env, default 0 = pre-registered) = opt-in image-level re-flattening passes for x_eq / x_le (see make_counterfactuals).
Outputs per cell under OUT/<cell>/: <case>_texthink.csv (written last), <case>_units.npz, <case>_maps.npz (float16, one axial slice, patient data:
never commit). --report: texthink.md, texthink_cells.csv, texthink_summary.png/pdf. --figure: texthink_examples_<cell>.png (patient slices: never commit).
OUT = env VP_TEXTHINK_OUT or <THIS.parent>/outputs/plots/texthink.
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_erf_outcome as vo  # noqa: E402
import visual_proof_v2 as v2  # noqa: E402
import visual_proof_erf as e7  # noqa: E402
from visual_proof_switch import load_case, activations, decoder_fwd, models_for, wp, FOLDS, ARMS, PAL, LOCS  # noqa: E402

OUT = Path(os.environ.get("VP_TEXTHINK_OUT") or (THIS.parent / "outputs/plots/texthink")); OUT.mkdir(parents=True, exist_ok=True)
CELLS4 = ("glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w")
SIG_FINE, SIG_E, EPS_REL, N_PASS = 1.0, 2.0, 1e-3, 2
RING, ONE_NAT, DISP_THR, NTOP = 10, 1.0, 0.2, 5
ALLLOC = ["full"] + LOCS
UNIT_LOCS = ("s2", "s3")
TT_ITER = int(os.environ.get("TT_ITER", 0))   # opt-in image-level re-flattening passes (see make_counterfactuals); 0 = pre-registered construction


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def G(a, s):
    return ndi.gaussian_filter(a, s)


# ----------------------------------------------------------------------------------------------------------------------------------- counterfactuals
def phase_randomise(band, rng):
    """Real array with exactly the rfftn amplitude of `band`, random Hermitian-consistent phases (from the rfftn of real white noise), DC kept."""
    ax = tuple(range(band.ndim)); F = np.fft.rfftn(band, axes=ax); N = np.fft.rfftn(rng.standard_normal(band.shape), axes=ax)
    S = np.abs(F) * N / np.maximum(np.abs(N), 1e-12); S[(0,) * band.ndim] = F[(0,) * band.ndim]
    return np.fft.irfftn(S, s=band.shape, axes=ax)


def make_counterfactuals(ip, bp, hp, case):
    """-> x_eq, x_le (float32), fine_eq, fine_le (float64), Emap (float64) of the original fine band."""
    f = hp.astype(np.float64); b = bp.astype(bool); f2 = f ** 2; m = float(f2[b].mean()); eps = EPS_REL * m; Emap = G(f2, SIG_E)
    s = f.copy()
    for _ in range(N_PASS):
        s = s * np.sqrt((m + eps) / (G(s ** 2, SIG_E) + eps))
    fine_eq = s * b
    seed = int(hashlib.md5(case.encode()).hexdigest()[:8], 16); rng = np.random.default_rng(seed)
    s = phase_randomise(f, rng)
    for _ in range(N_PASS):
        s = s * np.sqrt((Emap + eps) / (G(s ** 2, SIG_E) + eps))
    s = s * b; e0 = float(f2[b].sum()); e1 = float((s[b] ** 2).sum()); fine_le = s * np.sqrt(e0 / e1) if e1 > 0 else s
    x_eq = ip - f + fine_eq; x_le = ip - f + fine_le
    # Opt-in (env TT_ITER = n outer passes, default 0 = the pre-registered construction): the sigma-1 highpass is not idempotent, so the image's own fine
    # band x' - G1(x') = G1(fine) + hp(fine_replacement) carries part of the original map back in. Each outer pass recomputes r = (x' - G1 x') * body
    # from the current image, re-applies the same gain rule to r (eq: toward m; le: toward the ORIGINAL Emap), and reinserts it: x' <- x' - r + r'.
    for _ in range(TT_ITER):
        for nm in ("eq", "le"):
            xc = x_eq if nm == "eq" else x_le; r = (xc - G(xc, SIG_FINE)) * b; s = r.copy()
            for _ in range(N_PASS):
                s = s * np.sqrt(((m if nm == "eq" else Emap) + eps) / (G(s ** 2, SIG_E) + eps))
            xc = xc - r + s * b
            if nm == "eq":
                x_eq = xc
            else:
                x_le = xc
    if TT_ITER:
        fine_eq = (x_eq - G(x_eq, SIG_FINE)) * b; fine_le = (x_le - G(x_le, SIG_FINE)) * b
    return x_eq.astype(np.float32), x_le.astype(np.float32), fine_eq, fine_le, Emap


def qc_stats(ip, bp, gp, T, x_eq, x_le, fine_eq, fine_le, Emap):
    """Target/body contrast of the fine-energy map (recomputed from each image, and directly from the replacement band), edge ratio on the GT
    boundary shell (dilate 2 minus erode 2, |grad G2|), body fine-energy ratio."""
    b = bp.astype(bool); tg = np.isin(gp, T); shell = ndi.binary_dilation(tg, iterations=2) & ~ndi.binary_erosion(tg, iterations=2)

    def ctr(E):
        return float(E[tg].mean() / (E[b].mean() + 1e-12)) if tg.any() else np.nan

    def fine(x):
        return (x.astype(np.float64) - G(x.astype(np.float64), SIG_FINE)) * b

    def edge(x):
        g = ndi.gaussian_gradient_magnitude(x.astype(np.float64), SIG_E); return float(g[shell].mean()) if shell.any() else np.nan
    e0 = edge(ip); f0 = fine(ip); en0 = float((f0[b] ** 2).sum())
    out = dict(ctr_orig=ctr(Emap), ctr_orig_recomp=ctr(G(f0 ** 2, SIG_E)))
    for cf, x, fb in (("eq", x_eq, fine_eq), ("le", x_le, fine_le)):
        fx = fine(x); out[f"ctr_{cf}"] = ctr(G(fx ** 2, SIG_E)); out[f"ctr_{cf}_band"] = ctr(G(fb ** 2, SIG_E))
        out[f"edge_{cf}"] = edge(x) / (e0 + 1e-12); out[f"energy_{cf}"] = float((fx[b] ** 2).sum()) / (en0 + 1e-12)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- per case
def run_cell(key, dev, sl=None):
    import torch
    import torch.nn.functional as Fn
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in models_for(key).items()}; pr0 = prs["flat"]
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    allc = sorted(cell["cases"]); cases = allc[slice(*[int(v) for v in sl.split(":")])] if sl else allc
    c0 = int(T[0])
    for case in cases:
        f_out = od / f"{case}_texthink.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, hp = load_case(pr0, cell, case, T)
        b = bp.astype(bool); tg = np.isin(gp, T); dil = ndi.binary_dilation(tg, iterations=RING); ring = dil & ~tg & b; bg = b & ~dil
        x_eq, x_le, fine_eq, fine_le, Emap = make_counterfactuals(ip, bp, hp, case)
        q = qc_stats(ip, bp, gp, T, x_eq, x_le, fine_eq, fine_le, Emap)
        z = int(np.argmax(tg.sum((1, 2)))) if tg.any() else ip.shape[0] // 2
        base = dict(cell=key, task=cell["task"], case=case, n_target=int(tg.sum()), n_ring=int(ring.sum()), n_bg=int(bg.sum()), z=z)
        log(f"[{key}] {case}: QC target/body fine-energy contrast orig {q['ctr_orig']:.3f} eq {q['ctr_eq']:.3f} (band {q['ctr_eq_band']:.3f}) "
            f"le {q['ctr_le']:.3f} (band {q['ctr_le_band']:.3f}); edge ratio eq {q['edge_eq']:.3f} le {q['edge_le']:.3f}; body energy eq {q['energy_eq']:.3f} le {q['energy_le']:.3f}")
        rows = [dict(base, kind="qc", **q)]
        # regressors for the unit Spearman, full resolution (downsampled per stage below)
        g2 = G(ip.astype(np.float64), SIG_E); gr = ndi.gaussian_gradient_magnitude(ip.astype(np.float64), SIG_E)
        reg_full = torch.from_numpy(np.stack([Emap, g2, gr]).astype(np.float32))[None]
        X = {nm: torch.from_numpy(a)[None, None].to(dev) for nm, a in (("x", ip), ("eq", x_eq), ("le", x_le))}
        tg_t = torch.from_numpy(tg).to(dev); body_t = torch.from_numpy(b).to(dev)[None, None].float()
        maps = dict(z=np.int32(z), ip=ip[z], emap=Emap[z], gt=tg[z]); units = {}
        for arm, pr in prs.items():
            net = pr.network; accJ = None; acc = {}; disp = {}; A0 = {}
            with torch.no_grad():
                for k in FOLDS:
                    net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                    A, lg = activations(net, X["x"]); Jx = e7.log_odds(lg.float(), c0)[0]; accJ = Jx if accJ is None else accJ + Jx
                    skips = [A[f"s{j}"] for j in range(6)]
                    for cf in ("eq", "le"):
                        Ac, lgc = activations(net, X[cf]); e = Jx - e7.log_odds(lgc.float(), c0)[0]; acc[(cf, "full")] = acc.get((cf, "full"), 0) + e
                        for loc in LOCS:
                            lgm = decoder_fwd(net, skips, swap={loc: Ac[loc]}); e = Jx - e7.log_odds(lgm.float(), c0)[0]; acc[(cf, loc)] = acc.get((cf, loc), 0) + e
                        if cf == "eq":
                            for loc in UNIT_LOCS:
                                h0 = A[loc][0].float(); h1 = Ac[loc][0].float(); C = h0.shape[0]
                                mb = Fn.interpolate(body_t, size=h0.shape[1:], mode="nearest")[0, 0] > 0.5
                                d = (h1 - h0)[:, mb].norm(dim=1) / (h0[:, mb].norm(dim=1) + 1e-6); disp[loc] = disp.get(loc, 0) + d / len(FOLDS)
                        del Ac, lgc
                    if k == FOLDS[0]:
                        A0 = {loc: A[loc][0].float().clone() for loc in UNIT_LOCS}
                    del A, skips, lg
                    torch.cuda.empty_cache()
            nf = float(len(FOLDS)); Jm = (accJ / nf).cpu().numpy()
            rows.append(dict(base, kind="nat", arm=arm, cf="nat", loc="", J_target=float(Jm[tg].mean()) if tg.any() else np.nan,
                             J_ring=float(Jm[ring].mean()) if ring.any() else np.nan, J_bg=float(Jm[bg].mean()) if bg.any() else np.nan))
            maps[f"{arm}_J"] = Jm[z]
            for (cf, loc), v in acc.items():
                E = (v / nf).cpu().numpy()
                rows.append(dict(base, kind="E", arm=arm, cf=cf, loc=loc, E_target=float(E[tg].mean()) if tg.any() else np.nan,
                                 E_ring=float(E[ring].mean()) if ring.any() else np.nan, E_bg=float(E[bg].mean()) if bg.any() else np.nan,
                                 frac_gt1=float((E[tg] > ONE_NAT).mean()) if tg.any() else np.nan))
                if (cf, loc) in (("eq", "full"), ("eq", "s2"), ("eq", "s3"), ("le", "full"), ("le", "s3")):
                    maps[f"{arm}_E_{loc}_{cf}"] = E[z]
            # units
            for loc in UNIT_LOCS:
                d = disp[loc].cpu().numpy(); units[f"{arm}_{loc}"] = d; top = np.argsort(d)[::-1][:NTOP]
                h0 = A0[loc]; shp = h0.shape[1:]
                mb = (Fn.interpolate(body_t, size=shp, mode="nearest")[0, 0] > 0.5).cpu().numpy()
                R = Fn.interpolate(reg_full, size=shp, mode="area")[0].numpy()     # Emap, G2(x), |grad G2(x)| at stage resolution
                y = R[0][mb].astype(np.float64); Xr = np.stack([np.ones_like(y), R[1][mb], R[2][mb]], 1).astype(np.float64)
                beta, *_ = np.linalg.lstsq(Xr, y, rcond=None); resid = y - Xr @ beta
                hm = h0.cpu().numpy(); rhos = []; rraw = []
                for c in top:
                    a = hm[c][mb]; ok = np.std(a) > 0
                    rhos.append(float(spearmanr(a, resid).correlation) if ok else np.nan); rraw.append(float(spearmanr(a, y).correlation) if ok else np.nan)
                row = dict(base, kind="unit", arm=arm, cf="eq", loc=loc, n_ch=int(len(d)), frac_disp_gt=float((d > DISP_THR).mean()), disp_median=float(np.median(d)),
                           disp_top1=float(d[top[0]]), top5=";".join(str(int(c)) for c in top), rho_mean=float(np.nanmean(rhos)), rho_absmean=float(np.nanmean(np.abs(rhos))),
                           rho_raw_mean=float(np.nanmean(rraw)))
                for i, (r_, rr_) in enumerate(zip(rhos, rraw)):
                    row[f"rho{i + 1}"] = r_; row[f"rho_raw{i + 1}"] = rr_
                rows.append(row)
                if loc == "s2":
                    u = Fn.interpolate(A0["s2"][int(top[0])][None, None], size=ip.shape, mode="nearest")[0, 0, z].cpu().numpy()
                    maps[f"{arm}_unit_s2"] = u; maps[f"{arm}_unit_s2_idx"] = np.int32(top[0])
            del accJ, acc, A0; torch.cuda.empty_cache()
        np.savez_compressed(od / f"{case}_units.npz", **units)
        np.savez_compressed(od / f"{case}_maps.npz", **{k_: (v_.astype(np.float16) if isinstance(v_, np.ndarray) and v_.dtype.kind == "f" else v_) for k_, v_ in maps.items()})
        pd.DataFrame(rows).to_csv(f_out, index=False)
        nat = {r["arm"]: r["J_target"] for r in rows if r["kind"] == "nat"}
        log(f"[{key}] {case}: {len(rows)} rows, J_target " + " ".join(f"{a} {v:.5f}" for a, v in nat.items()) + f", {time.time() - t0:.0f} s")
    print("TEXTHINK_DONE", flush=True)



# ----------------------------------------------------------------------------------------------------------------------------------- x_scr second pass
def make_scr(ip, bp, hp, case):
    """x_scr (added after the first full run, coordinator request; the Dice follow-up's plain global-energy fine scrambling): phase-randomised fine band
    (same seed / phase field as x_le, from the case name), body-masked, body energy rescaled to the original, NO local gain, no image-level iteration
    (identical under every TT_ITER); x_scr = x - fine + fine_scr."""
    f = hp.astype(np.float64); b = bp.astype(bool); f2 = f ** 2
    rng = np.random.default_rng(int(hashlib.md5(case.encode()).hexdigest()[:8], 16)); s = phase_randomise(f, rng) * b
    e0 = float(f2[b].sum()); e1 = float((s[b] ** 2).sum()); s = s * np.sqrt(e0 / e1) if e1 > 0 else s
    return (ip - f + s).astype(np.float32), s


def run_cell_scr(key, dev, sl=None):
    """Second pass: E_scr at full and every loc, written to <case>_texthink_scr.csv (kind E, cf scr; kind qc_scr), maps npz gains <arm>_E_full_scr /
    <arm>_E_s3_scr. Skips cases whose _scr csv exists."""
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in models_for(key).items()}; pr0 = prs["flat"]
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    allc = sorted(cell["cases"]); cases = allc[slice(*[int(v) for v in sl.split(":")])] if sl else allc; c0 = int(T[0])
    for case in cases:
        f_out = od / f"{case}_texthink_scr.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, hp = load_case(pr0, cell, case, T)
        b = bp.astype(bool); tg = np.isin(gp, T); dil = ndi.binary_dilation(tg, iterations=RING); ring = dil & ~tg & b; bg = b & ~dil
        x_scr, fine_scr = make_scr(ip, bp, hp, case); Emap = G(hp.astype(np.float64) ** 2, SIG_E)
        shell = ndi.binary_dilation(tg, iterations=2) & ~ndi.binary_erosion(tg, iterations=2)

        def ctr(E):
            return float(E[tg].mean() / (E[b].mean() + 1e-12)) if tg.any() else np.nan

        def fine(x):
            return (x.astype(np.float64) - G(x.astype(np.float64), SIG_FINE)) * b

        def edge(x):
            g = ndi.gaussian_gradient_magnitude(x.astype(np.float64), SIG_E); return float(g[shell].mean()) if shell.any() else np.nan
        f0 = fine(ip); fx = fine(x_scr)
        q = dict(ctr_orig=ctr(Emap), ctr_scr=ctr(G(fx ** 2, SIG_E)), ctr_scr_band=ctr(G(fine_scr ** 2, SIG_E)), edge_scr=edge(x_scr) / (edge(ip) + 1e-12),
                 energy_scr=float((fx[b] ** 2).sum()) / (float((f0[b] ** 2).sum()) + 1e-12))
        z = int(np.argmax(tg.sum((1, 2)))) if tg.any() else ip.shape[0] // 2
        base = dict(cell=key, task=cell["task"], case=case, n_target=int(tg.sum()), n_ring=int(ring.sum()), n_bg=int(bg.sum()), z=z)
        log(f"[{key}] {case}: QC scr: target/body contrast orig {q['ctr_orig']:.3f} scr {q['ctr_scr']:.3f} (band {q['ctr_scr_band']:.3f}); edge ratio {q['edge_scr']:.3f}; body energy {q['energy_scr']:.3f}")
        rows = [dict(base, kind="qc_scr", **q)]; X = {nm: torch.from_numpy(a)[None, None].to(dev) for nm, a in (("x", ip), ("scr", x_scr))}; newmaps = {}
        for arm, pr in prs.items():
            net = pr.network; acc = {}
            with torch.no_grad():
                for k in FOLDS:
                    net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                    A, lg = activations(net, X["x"]); Jx = e7.log_odds(lg.float(), c0)[0]; skips = [A[f"s{j}"] for j in range(6)]
                    Ac, lgc = activations(net, X["scr"]); acc["full"] = acc.get("full", 0) + (Jx - e7.log_odds(lgc.float(), c0)[0])
                    for loc in LOCS:
                        lgm = decoder_fwd(net, skips, swap={loc: Ac[loc]}); acc[loc] = acc.get(loc, 0) + (Jx - e7.log_odds(lgm.float(), c0)[0])
                    del A, Ac, lg, lgc, skips; torch.cuda.empty_cache()
            for loc, v in acc.items():
                E = (v / float(len(FOLDS))).cpu().numpy()
                rows.append(dict(base, kind="E", arm=arm, cf="scr", loc=loc, E_target=float(E[tg].mean()) if tg.any() else np.nan,
                                 E_ring=float(E[ring].mean()) if ring.any() else np.nan, E_bg=float(E[bg].mean()) if bg.any() else np.nan,
                                 frac_gt1=float((E[tg] > ONE_NAT).mean()) if tg.any() else np.nan))
                if loc in ("full", "s3"):
                    newmaps[f"{arm}_E_{loc}_scr"] = E[z].astype(np.float16)
            del acc; torch.cuda.empty_cache()
        fm = od / f"{case}_maps.npz"
        if fm.exists():
            M = dict(np.load(fm)); M.update(newmaps); tmp = od / f"{case}_maps.tmp.npz"; np.savez_compressed(tmp, **M); os.replace(tmp, fm)
        pd.DataFrame(rows).to_csv(f_out, index=False)
        log(f"[{key}] {case}: scr {len(rows)} rows, {time.time() - t0:.0f} s")
    print("TEXTHINK_SCR_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
CFS = ("eq", "le", "scr")
CFN = {"eq": "E (x_eq: energy map flattened, pattern kept)", "le": "S (x_le: pattern scrambled, local energy map kept)",
       "scr": "E_scr (x_scr: pattern scrambled, global body energy only)"}
HATCH = {"eq": None, "le": "////", "scr": "...."}


def fp(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.1g}"


def load_all():
    cells = [k for k in CELLS4 if (OUT / k).exists() and list((OUT / k).glob("*_texthink.csv"))]
    fs = [f for k in cells for f in sorted((OUT / k).glob("*_texthink.csv"))] + [f for k in cells for f in sorted((OUT / k).glob("*_texthink_scr.csv"))]
    S = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return cells, S


def piv(S, k, arm, cf, loc, col="E_target"):
    d = S[(S.kind == "E") & (S.cell == k) & (S.arm == arm) & (S.cf == cf) & (S["loc"] == loc)]
    return d.set_index("case")[col]


def med_iqr(s):
    s = s.dropna()
    return f"{s.median():+.3f} [{s.quantile(0.25):+.3f}, {s.quantile(0.75):+.3f}]" if len(s) else "n/a"


def ratio_med(a, b):
    a, b = a.median(), b.median()
    return a / b if np.isfinite(b) and abs(b) > 1e-6 else np.nan


def report():
    cells, S = load_all()
    if not cells:
        print("no csv"); return
    cfs = [cf for cf in CFS if (S.get("cf") is not None) and (S.cf == cf).any()]
    nsc = {k: S[(S.cell == k) & (S.kind == "qc_scr")].case.nunique() if "qc_scr" in set(S.kind) else 0 for k in cells}
    L = ["# What the model thinks about texture (FINDINGS 42, pre-registered: PREREG.md): texture-evidence maps by activation patching", "",
         "Cells: " + ", ".join(f"{k} (n = {S[(S.cell == k) & (S.kind == 'qc')].case.nunique()}; x_scr n = {nsc[k]})" for k in cells) + ". Arms noise / flat / real, folds 0-2. "
         "E_L = J(x) - J(x with the activation at L taken from the counterfactual run), J = per-voxel log-odds of the target class (edema), maps fold-mean, "
         "scalars read on the fold-mean maps; E_full = J(x) - J(x_cf). Three counterfactuals: x_eq (texture-energy map flattened, local patterns kept) -> E; "
         "x_le (fine structure phase-scrambled, local energy map kept) -> S; x_scr (added after the first full run on request: fine structure phase-scrambled, "
         "body-masked, global body energy rescaled, no local gain; same phase field as x_le; not iterated) -> E_scr. "
         "Positive = removing that property LOWERS the target belief, i.e. it raised it. Target = GT target voxels in the patch; ring = "
         f"dilation {RING} minus target, inside the body; background = body minus dilation {RING}. d3 / d4 equal or approach E_full by construction and are not read.",
         f"Counterfactual construction (x_eq, x_le): TT_ITER = {TT_ITER} " + ("(pre-registered: x - fine + fine_eq / fine_le, band gain only)." if TT_ITER == 0 else
         f"(image-level re-flattening: after the pre-registered construction, {TT_ITER} outer passes recompute the image's own fine band (x' - G1 x') * body "
         "and re-apply the same gain rule to it, because the sigma-1 highpass is not idempotent and the pre-registered x_eq left part of the original "
         "energy map in the image's fine band; chosen on the smoke QC before any evidence value was read)."), ""]
    rows = []
    # ---- QC
    Q = S[S.kind == "qc"]; Qs = S[S.kind == "qc_scr"] if "qc_scr" in set(S.kind) else pd.DataFrame(columns=["cell"])
    L += ["## QC (medians over cases; out = cases outside the smoke gates: edge ratio [0.8, 1.2], x_eq contrast [0.8, 1.25])", "",
          "| cell | Emap target/body contrast orig | after x_eq | after x_le | after x_scr | edge ratio eq [min-max, out] | le [min-max, out] | scr [min-max, out] | x_eq contrast out | body fine energy eq | le | scr |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def er(s):
        s = s.dropna(); return f"{s.median():.3f} [{s.min():.2f}-{s.max():.2f}, {int(((s < 0.8) | (s > 1.2)).sum())}]" if len(s) else "n/a"
    for k in cells:
        q = Q[Q.cell == k]; qs = Qs[Qs.cell == k]
        L.append(f"| {k} | {q.ctr_orig.median():.3f} | {q.ctr_eq.median():.3f} | {q.ctr_le.median():.3f} | {qs.ctr_scr.median() if len(qs) else np.nan:.3f} | {er(q.edge_eq)} | {er(q.edge_le)} | "
                 f"{er(qs.edge_scr) if len(qs) else 'n/a'} | {int(((q.ctr_eq < 0.8) | (q.ctr_eq > 1.25)).sum())} | {q.energy_eq.median():.3f} | {q.energy_le.median():.3f} | {qs.energy_scr.median() if len(qs) else np.nan:.3f} |")
    L.append("")
    # ---- per loc tables
    for cf in cfs:
        L += [f"## {CFN[cf]}: target mean, median over cases per location", "", "| cell | arm | " + " | ".join(ALLLOC) + " |", "|" + "---|" * (2 + len(ALLLOC))]
        for k in cells:
            for arm in ARMS:
                L.append(f"| {k} | {arm} | " + " | ".join(f"{piv(S, k, arm, cf, l).median():+.3f}" for l in ALLLOC) + " |")
        L.append("")
    # ---- per cell x arm summary per counterfactual
    for cf in cfs:
        L += [f"## {CFN[cf]}: per cell and arm, median [IQR] over cases", "",
              "| cell | arm | target full | target s2 | target s3 | ring full | background full | frac > 1 nat (target) | share (s2+s3)/full | ratio to S_target full (x_le) | ratio s3 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in cells:
            for arm in ARMS:
                Ef, E2, E3 = piv(S, k, arm, cf, "full"), piv(S, k, arm, cf, "s2"), piv(S, k, arm, cf, "s3")
                Sf, S3 = piv(S, k, arm, "le", "full"), piv(S, k, arm, "le", "s3")
                ring, bgr, fr = piv(S, k, arm, cf, "full", "E_ring"), piv(S, k, arm, cf, "full", "E_bg"), piv(S, k, arm, cf, "full", "frac_gt1")
                share = ((E2 + E3) / Ef.where(Ef.abs() > 1e-3)).median(); rf, r3 = ratio_med(Ef, Sf), ratio_med(E3, S3)
                pcr = (Ef / Sf.where(Sf.abs() > 1e-3)).median()
                L.append(f"| {k} | {arm} | {med_iqr(Ef)} | {med_iqr(E2)} | {med_iqr(E3)} | {med_iqr(ring)} | {med_iqr(bgr)} | {fr.median():.3f} | {share:.2f} | {rf:.2f} | {r3:.2f} |")
                rows.append(dict(cell=k, arm=arm, cf=cf, n=int(Ef.notna().sum()), X_full=Ef.median(), X_full_q25=Ef.quantile(0.25), X_full_q75=Ef.quantile(0.75), X_ring=ring.median(), X_bg=bgr.median(),
                                 frac_gt1=fr.median(), share_s2s3=share, ratio_to_S_full=rf, ratio_to_S_s3=r3, ratio_to_S_full_percase=pcr,
                                 **{f"X_{l}": piv(S, k, arm, cf, l).median() for l in LOCS}))
        L += ["", "Ratio = ratio of the medians over cases (primary for P2; per-case ratios with |S| > 1e-3 in texthink_cells.csv). Share = median over cases of the per-case (X_s2 + X_s3) / X_full (|X_full| > 1e-3).", ""]
    # ---- contrasts
    con = {}
    L += ["## Arm contrasts on the target mean (paired over cases, two-sided Wilcoxon)", "",
          "| counterfactual | cell | loc | real - flat median | p | real > flat (% cases) | real - noise median | p | flat - noise median | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cf in cfs:
        for k in cells:
            for l in ("full", "s2", "s3"):
                r, f, n = piv(S, k, "real", cf, l), piv(S, k, "flat", cf, l), piv(S, k, "noise", cf, l)
                drf, drn, dfn = (r - f).dropna(), (r - n).dropna(), (f - n).dropna(); con[(cf, k, l)] = (drf, drn)
                L.append(f"| {cf} | {k} | {l} | {drf.median():+.3f} | {fp(wp(drf))} | {100 * (drf > 0).mean() if len(drf) else np.nan:.0f} | {drn.median():+.3f} | {fp(wp(drn))} | {dfn.median():+.3f} | {fp(wp(dfn))} |")
                for row in rows:
                    if row["cell"] == k and row["arm"] == "real" and row["cf"] == cf:
                        row[f"rf_{l}"], row[f"p_rf_{l}"], row[f"rn_{l}"], row[f"p_rn_{l}"] = drf.median(), wp(drf), drn.median(), wp(drn)
    L.append("")
    # ---- units
    U = S[S.kind == "unit"]
    L += ["## Units (x_eq displacement, s2 / s3): fraction of channels with relative displacement > 0.2 (median over cases), median displacement, "
          "top-5 Spearman with the residualised fine-energy map (fold 0; mean of the 5 per case, then median; raw = not residualised)", "",
          "| cell | arm | loc | n ch | frac > 0.2 | median disp | top-1 disp | rho top-5 (resid) | |rho| top-5 | rho top-1 | rho top-5 raw |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    ufr = {}
    for k in cells:
        for loc in UNIT_LOCS:
            for arm in ARMS:
                u = U[(U.cell == k) & (U.arm == arm) & (U["loc"] == loc)].set_index("case"); ufr[(k, loc, arm)] = u.frac_disp_gt
                L.append(f"| {k} | {arm} | {loc} | {int(u.n_ch.median()) if len(u) else 0} | {u.frac_disp_gt.median():.3f} | {u.disp_median.median():.3f} | {u.disp_top1.median():.3f} | "
                         f"{u.rho_mean.median():+.3f} | {u.rho_absmean.median():.3f} | {u.rho1.median():+.3f} | {u.rho_raw_mean.median():+.3f} |")
                for row in rows:
                    if row["cell"] == k and row["arm"] == arm:
                        row[f"unit_frac_{loc}"], row[f"unit_rho_{loc}"], row[f"unit_absrho_{loc}"] = u.frac_disp_gt.median(), u.rho_mean.median(), u.rho_absmean.median()
    L.append("")
    R = pd.DataFrame(rows); R.to_csv(OUT / "texthink_cells.csv", index=False); R = R.set_index(["cell", "arm", "cf"])
    # ---- verdicts
    nc = len(cells); V = []
    for cf in cfs:
        rf_ = sum(1 for k in cells if con[(cf, k, "full")][0].median() > 0 and wp(con[(cf, k, "full")][0]) < 0.01)
        rn_ = sum(1 for k in cells if con[(cf, k, "full")][1].median() > 0 and wp(con[(cf, k, "full")][1]) < 0.01)
        le_ = sum(1 for k in cells if not (con[(cf, k, "full")][0].median() > 0))
        det = ", ".join(f"{k} {con[(cf, k, 'full')][0].median():+.3f} (p {fp(wp(con[(cf, k, 'full')][0]))})" for k in cells)
        if le_ >= 2:
            v = f"KILLED: real <= flat (median paired difference) in {le_}/{nc} cells"
        elif rf_ >= 3 and rn_ >= 3:
            v = f"SUPPORTED: real > flat (p < 0.01) in {rf_}/{nc} cells, real > noise in {rn_}/{nc}"
        else:
            v = f"NOT SUPPORTED: real > flat (p < 0.01) in {rf_}/{nc} cells, real > noise in {rn_}/{nc} (need >= 3 each); real <= flat in {le_}/{nc}"
        V.append(f"- P1 on {cf}{' (PRE-REGISTERED reading)' if cf == 'eq' else (' (expected to show the ordering, per the Dice follow-up)' if cf == 'scr' else '')}: {v}. real - flat full: {det}.")
    for cf in [c for c in cfs if c != "le"]:
        n2 = sum(1 for k in cells if np.isfinite(R.loc[(k, "real", cf), "ratio_to_S_full"]) and R.loc[(k, "real", cf), "ratio_to_S_full"] > 2)
        V.append(f"- P2 on {cf} (real-arm X_target / S_target full, ratio of medians > 2){' (PRE-REGISTERED reading)' if cf == 'eq' else ''}: {'SUPPORTED' if n2 >= 3 else 'NOT SUPPORTED'} in {n2}/{nc} cells ("
                 + ", ".join(f"{k} {R.loc[(k, 'real', cf), 'ratio_to_S_full']:.2f}" for k in cells) + ").")
    for cf in cfs:
        n3 = sum(1 for k in cells if R.loc[(k, "real", cf), "share_s2s3"] > 0.5)
        V.append(f"- P3 on {cf} (real-arm median (X_s2 + X_s3) / X_full > 0.5){' (PRE-REGISTERED reading)' if cf == 'eq' else ''}: {'SUPPORTED' if n3 >= 3 else 'NOT SUPPORTED'} in {n3}/{nc} cells ("
                 + ", ".join(f"{k} {R.loc[(k, 'real', cf), 'share_s2s3']:.2f}" for k in cells) + ").")
    sim = more = 0; det = []
    for k in cells:
        ds = [(ufr[(k, loc, "real")] - ufr[(k, loc, "flat")]).dropna().median() for loc in UNIT_LOCS]
        det.append(f"{k} s2 {ds[0]:+.3f}, s3 {ds[1]:+.3f}"); sim += all(abs(d) < 0.1 for d in ds); more += all(d >= 0.1 for d in ds)
    V.append(f"- P4 (units, x_eq): {'SUPPORTED' if sim >= 3 else ('NOT SUPPORTED: real has MORE displaced units (reported as the mechanism)' if more >= 3 else 'NOT SUPPORTED (mixed)')}: "
             f"|real - flat| fraction of channels with displacement > 0.2 < 0.1 at both s2 and s3 in {sim}/{nc} cells; real >= flat + 0.1 at both in {more}/{nc} "
             f"(median paired difference: " + "; ".join(det) + ").")
    L += ["## Verdicts (pre-registered P1-P4, read on x_eq as pre-registered; P1-P3 also read on x_le and x_scr on request. Operational rules fixed before reading the full-run "
          "numbers: P1 paired real - flat and real - noise median > 0 with two-sided Wilcoxon p < 0.01, kill if real <= flat in >= 2 cells; P2 ratio of medians; "
          "P3 median per-case share; P4 'similar' = |median paired difference of the fraction| < 0.1 at s2 and s3)", ""] + V + [""]
    (OUT / "texthink.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
    # ---- figure
    fig, axs = plt.subplots(1, nc, figsize=(6.0 * nc, 4.4), squeeze=False); nb = len(ARMS) * len(cfs); w = 0.85 / nb
    for c, k in enumerate(cells):
        ax = axs[0, c]
        for i, arm in enumerate(ARMS):
            for j, cf in enumerate(cfs):
                v = [piv(S, k, arm, cf, l).median() for l in ALLLOC]; off = (i * len(cfs) + j - (nb - 1) / 2) * w
                ax.bar(np.arange(len(ALLLOC)) + off, v, w, color=PAL[arm] if cf == "eq" else "white", edgecolor=PAL[arm], hatch=HATCH[cf], lw=0.7, label=f"{arm} {cf}")
        ax.set_xticks(range(len(ALLLOC))); ax.set_xticklabels(ALLLOC, fontsize=7); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{k} (n = {S[(S.cell == k) & (S.kind == 'qc')].case.nunique()})\ntarget evidence through one location (median over cases); solid eq, //// le, .... scr", fontsize=8)
        if c == 0:
            ax.set_ylabel("mean log-odds change over target voxels (nat)"); ax.legend(fontsize=5.5, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "texthink_summary.png", dpi=150); fig.savefig(OUT / "texthink_summary.pdf"); plt.close(fig)
    print("TEXTHINK_REPORT_DONE", flush=True)


def figure():
    cells, S = load_all(); VL = 3.0
    for k in cells:
        q = S[(S.cell == k) & (S.kind == "qc")].sort_values("n_target"); q = q[[(OUT / k / f"{c}_maps.npz").exists() for c in q.case]]
        if not len(q):
            continue
        picks = []
        for pc in (50, 75):
            tgt = np.percentile(q.n_target, pc); order = (q.n_target - tgt).abs().sort_values().index
            for i in order:
                if q.loc[i, "case"] not in picks:
                    picks.append(q.loc[i, "case"]); break
        M0 = np.load(OUT / k / f"{picks[0]}_maps.npz"); has_scr = "real_E_full_scr" in M0.files
        cols = [("image", None), ("fine-energy map G2(fine^2)", None)] + [(f"E_full eq {a}", f"{a}_E_full_eq") for a in ("real", "flat", "noise")] + \
               [(f"E_s3 eq {a}", f"{a}_E_s3_eq") for a in ("real", "flat", "noise")] + [(f"S_full le {a}", f"{a}_E_full_le") for a in ("real", "flat", "noise")] + \
               ([(f"E_full scr {a}", f"{a}_E_full_scr") for a in ("real", "flat", "noise")] if has_scr else []) + \
               [(f"top s2 texture unit {a}", f"{a}_unit_s2") for a in ("real", "flat")]
        fig, axs = plt.subplots(len(picks), len(cols), figsize=(1.9 * len(cols), 2.1 * len(picks) + 0.6), squeeze=False)
        for r, case in enumerate(picks):
            M = np.load(OUT / k / f"{case}_maps.npz"); gt = M["gt"].astype(float).T; nt = int(q[q.case == case].n_target.iloc[0])
            for c, (title, key) in enumerate(cols):
                ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    im = M["ip"].astype(float).T; ax.imshow(im, cmap="gray", origin="lower", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
                    ax.set_ylabel(f"{case}\nz={int(M['z'])}, n_target={nt}", fontsize=6)
                elif c == 1:
                    em = M["emap"].astype(float).T; ax.imshow(em, cmap="magma", origin="lower", vmin=0, vmax=np.percentile(em, 99.5))
                elif "unit" in key:
                    u = M[key].astype(float).T; ax.imshow(u, cmap="viridis", origin="lower", vmin=np.percentile(u, 1), vmax=np.percentile(u, 99.5))
                    title = f"{title} (ch {int(M[key + '_idx'])})"
                elif key in M.files:
                    ax.imshow(M[key].astype(float).T, cmap="RdBu_r", origin="lower", vmin=-VL, vmax=VL)
                ax.contour(gt, [0.5], colors="lime", linewidths=0.6)
                if r == 0 or "unit" in str(key):
                    ax.set_title(title, fontsize=6)
        fig.suptitle(f"{k}: texture-evidence maps (red = removing the texture-energy map / fine structure LOWERS the edema belief, i.e. it raised it; "
                     f"RdBu_r +-{VL:g} nat; lime = GT edema)", fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / f"texthink_examples_{k}.png", dpi=140); plt.close(fig)
    print("TEXTHINK_FIGURE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--figure", action="store_true"); ap.add_argument("--scr", action="store_true", help="second pass: x_scr only")
    args = ap.parse_args()
    if args.report or args.figure:
        if args.report:
            report()
        if args.figure:
            figure()
        return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.cells:
        (run_cell_scr if args.scr else run_cell)(key, dev, args.case_slice)


if __name__ == "__main__":
    main()
