#!/usr/bin/env python
"""Visual proof v14 (FINDINGS 41): scale-band scrambling, the reliance spectrum. Design: outputs/plots/scramble/PREREG.md (written before extraction).

Per case (v10 plan patch around the target, reused from visual_proof_switch.load_case), the patch image x is split into Gaussian scale bands
(sigma in voxels):  fine = x - G1(x),  mid = G1 - G2,  coarse = G2 - G4,  base = G4;  "allfine" = fine + mid  (the four bands sum to x exactly).
For band b, x_b is REPLACED INSIDE THE BODY by a phase-randomised copy:
    x^(-b) = x - body * x_b + scr(x_b),
    scr(x_b) = body * irfftn(|rfftn(x_b)| * exp(i phi)) rescaled so that sum(scr^2 over body) = sum(x_b^2 over body).
The random phases phi are the phases of rfftn(white Gaussian noise of the patch shape): a real source makes phi exactly Hermitian-symmetric in every
plane of the rfft half-spectrum (DC / Nyquist bins come out real), so the inverse is exactly real and the amplitude spectrum is exactly preserved
before masking. The global DC bin keeps its original value (no sign flip of the band mean). Outside the body x is untouched (background stays 0).
Local-energy-matched variants fine_le / mid_le / allfine_le (added after the smoke test, before the full run): before masking, the scrambled band is
multiplied by g = sqrt((G_2(x_b^2) + eps) / (G_2(scr^2) + eps)) (applied twice), eps = 1e-3 mean(x_b^2 in the body), then masked and globally rescaled as above, so the
LOCAL fine-energy map is kept and only the structure is destroyed (the global variants spread the energy evenly over the body). They reuse the phases of
their global version (same rng). P1-P3 are read on both, fine_le primary.
One scrambled image per (band, seed) is shared by all three arms (paired comparisons); rng = default_rng([seed, band index, crc32(case)]).
Arms noise (rung 4, lblvor) / flat (4.5) / real (5), folds 0-2: Dice from the fold-mean softmax, J = fold mean of the mean target log-odds over GT
target voxels (visual_proof_switch.dice_and_j). Gradient link: G = fold-mean dJ/dx on the natural patch; pred_dJ = <G, x^(-b) - x>.
QC per (band, seed): edge_ratio = mean |grad G2| on the GT boundary shell (dilate 2 minus erode 2), x^(-b) / x; tex_ratio = RMS(x - G1 x) inside the
target, x^(-b) / x.
Outputs: <OUT>/<cell>/<case>_scramble.csv (cell, task, case, arm, band, seed, dice, J, dJ, pred_dJ, edge_ratio, tex_ratio; seed -1 = seed mean, band
"none" = natural parent); --report: scramble.md, scramble_cells.csv, scramble_summary.png/pdf.
Addendum (PREREG_addendum.md): --set q runs the energy-map flattening variants fine_eq / fine_eq_scr / allfine_eq / allfine_eq_scr into
<case>_scrambleq.csv (extra columns tc0 / tc1 = target texture contrast before / after); --report merges both files and adds P4 / P5.
OUT = env VP_SCRAMBLE_OUT or <scripts/..>/outputs/plots/scramble. CLI: --cells, --case-slice a:b, --set main|q, --report; env SC_SEEDS (default 3).
"""
from __future__ import annotations
import argparse, os, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_switch as sw  # noqa: E402   (load_case, encoder_fwd, decoder_fwd, dice_and_j, dice_prob, models_for, wp, FOLDS, ARMS, PAL)
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_erf_outcome as vo  # noqa: E402
import visual_proof_v2 as v2  # noqa: E402
import visual_proof_erf as e7  # noqa: E402

OUT = Path(os.environ.get("VP_SCRAMBLE_OUT") or (THIS.parent / "outputs/plots/scramble")); OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = int(os.environ.get("SC_SEEDS", 3))
BANDS = ("fine", "fine_le", "mid", "mid_le", "coarse", "base", "allfine", "allfine_le")
# PREREG_addendum.md (written before running them): energy-map flattening. _eq = original band, local energy made uniform over the body (structure kept,
# map flattened; deterministic, 1 seed); _eq_scr = the same flattening applied to the scrambled band (structure destroyed, map flattened; same phases as fine).
QBANDS = ("fine_eq", "fine_eq_scr", "allfine_eq", "allfine_eq_scr")
ORDER_ALL = ("fine", "fine_le", "fine_eq", "fine_eq_scr", "mid", "mid_le", "coarse", "base", "allfine", "allfine_le", "allfine_eq", "allfine_eq_scr")
SRC = {b: b.replace("_le", "").replace("_eq_scr", "").replace("_eq", "") for b in BANDS + QBANDS}   # band image each variant starts from
RNG_IDX = {"fine": 0, "mid": 1, "coarse": 2, "base": 3, "allfine": 4}               # _le variants reuse the phases of their global version (paired)
LE_SIGMA = 2.0; LE_EPS = 1e-3; LE_ITERS = 2; EQ_ITERS = int(os.environ.get("SC_EQ_ITERS", LE_ITERS))   # EQ_ITERS: flattening iterations of the _eq variants (default = LE_ITERS, as in the addendum)
  # local gain: Gaussian sigma 2, applied twice (sigma 3 applied once left a small-target smoke case at texture ratio 1.11)
ARMS, PAL, FOLDS, wp = sw.ARMS, sw.PAL, sw.FOLDS, sw.wp
ALPHA = 0.01; RHO_MIN = 0.5; QC_LO, QC_HI = 0.8, 1.2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- bands + scrambling
def bands(ip):
    g1, g2, g4 = (ndi.gaussian_filter(ip, s).astype(np.float32) for s in (1.0, 2.0, 4.0))
    b = dict(fine=ip - g1, mid=g1 - g2, coarse=g2 - g4, base=g4); b["allfine"] = b["fine"] + b["mid"]
    err = float(np.abs(b["fine"] + b["mid"] + b["coarse"] + b["base"] - ip).max())
    return b, err


def phase_randomise(band, rng):
    """Unmasked real array with exactly the amplitude spectrum of `band` and random Hermitian-consistent phases (global DC bin kept)."""
    ax = tuple(range(band.ndim)); F = np.fft.rfftn(band.astype(np.float64), axes=ax); N = np.fft.rfftn(rng.standard_normal(band.shape), axes=ax)
    S = np.abs(F) * N / np.maximum(np.abs(N), 1e-12); S[(0,) * band.ndim] = F[(0,) * band.ndim]
    return np.fft.irfftn(S, s=band.shape, axes=ax)


def check_scramble(band, rng):
    """Self-check: max relative error of the amplitude spectrum after phase randomisation (before masking), and the phase change."""
    F = np.fft.rfftn(band.astype(np.float64)); G = np.fft.rfftn(phase_randomise(band, rng)); a = np.abs(F)
    amp_err = float(np.abs(np.abs(G) - a).max() / (a.max() + 1e-30)); dphi = float(np.median(np.abs(np.angle(G * np.conj(F)))[a > 1e-6 * a.max()]))
    return amp_err, dphi


def scramble(band, body, rng, local=False):
    """Same Fourier amplitude as `band` (patch-sized FFT, no padding), uniformly random Hermitian-consistent phases (from the rfftn of real white noise,
    so DC / Nyquist bins are real and the inverse is exactly real), original global DC kept; masked to the body; energy inside the body rescaled to
    the band's energy inside the body. local=True (the *_le variants): before masking, multiply by the smooth local gain
    g = sqrt((G_2(band^2) + eps) / (G_2(scr^2) + eps)) (sigma LE_SIGMA = 2, re-estimated and applied LE_ITERS = 2 times), eps = LE_EPS * mean(band^2 in the body), so the LOCAL fine-energy map matches the original."""
    s = phase_randomise(band, rng)
    if local:
        b2 = band.astype(np.float64) ** 2; eps = LE_EPS * float(b2[body].mean())
        G_b = ndi.gaussian_filter(b2, LE_SIGMA)
        for _ in range(LE_ITERS):
            s = s * np.sqrt((G_b + eps) / (ndi.gaussian_filter(s ** 2, LE_SIGMA) + eps))
    s = s * body
    e0 = float((band[body].astype(np.float64) ** 2).sum()); e1 = float((s[body] ** 2).sum())
    return (s * np.sqrt(e0 / e1) if e1 > 0 else s).astype(np.float32)


def flatten(arr, band, body):
    """Energy-map flattening (_eq variants): arr * g, g = sqrt((m + eps) / (G_2(arr^2) + eps)) re-estimated and applied LE_ITERS times, m = mean(band^2 in
    the body), eps = LE_EPS * m: the local energy becomes uniform over the body, local patterns (phases) untouched. Then body mask and global rescale to
    the band's body energy, as for every other variant."""
    b2 = band.astype(np.float64) ** 2; m = float(b2[body].mean()); eps = LE_EPS * m; s = arr.astype(np.float64)
    for _ in range(EQ_ITERS):
        s = s * np.sqrt((m + eps) / (ndi.gaussian_filter(s ** 2, LE_SIGMA) + eps))
    s = s * body; e0 = float(b2[body].sum()); e1 = float((s[body] ** 2).sum())
    return (s * np.sqrt(e0 / e1) if e1 > 0 else s).astype(np.float32)


def replacement(b, B, body, rng):
    """The band that replaces B[SRC[b]] inside the body, for any variant name."""
    src = B[SRC[b]]
    if b.endswith("_eq_scr"):
        return flatten(phase_randomise(src, rng), src, body)
    if b.endswith("_eq"):
        return flatten(src, src, body)
    return scramble(src, body, rng, local=b.endswith("_le"))


def n_seeds(b):
    return 1 if b.endswith("_eq") else N_SEEDS


def tex_contrast(x, tg, body):
    """Mean fine energy (x - G1 x)^2 in the target / mean in the body (the target's 'texture contrast')."""
    h2 = (x - ndi.gaussian_filter(x, 1.0)) ** 2; return float(h2[tg].mean() / (h2[body].mean() + 1e-12)) if tg.any() else np.nan


def counterfactual(ip, band, scr, body):
    return (ip - band * body + scr).astype(np.float32)


def qc(ip, xc, gp, T):
    tg = np.isin(gp, T); bd = ndi.binary_dilation(tg, iterations=2) & ~ndi.binary_erosion(tg, iterations=2)

    def edge(x):
        g = ndi.gaussian_gradient_magnitude(x, 2.0); return float(g[bd].mean()) if bd.any() else np.nan

    def tex(x):
        h = x - ndi.gaussian_filter(x, 1.0); return float(np.sqrt((h[tg] ** 2).mean())) if tg.any() else np.nan
    return edge(xc) / (edge(ip) + 1e-9), tex(xc) / (tex(ip) + 1e-9)


# ----------------------------------------------------------------------------------------------------------------------------------- extraction
def grad_J(net, ip, tg, T, dev):
    """dJ/dx on the natural patch for the loaded fold; J = mean log-odds of T[0] over GT target voxels (same objective as dice_and_j)."""
    import torch
    x = torch.from_numpy(ip)[None, None].to(dev).clone().requires_grad_(True); tg_t = torch.from_numpy(tg).to(dev)
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        lg = sw.decoder_fwd(net, sw.encoder_fwd(net, x)); J = e7.log_odds(lg.float(), int(T[0]))[0][tg_t].mean()
        g, = torch.autograd.grad(J, x)
    return g[0, 0].detach().float().cpu().numpy()


def run_cell(key, dev, sl=None, qset=False):
    """qset=False: the PREREG bands -> <case>_scramble.csv; qset=True: the addendum bands (QBANDS) -> <case>_scrambleq.csv (no natural rows; report merges)."""
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in sw.models_for(key).items()}; pr0 = prs["flat"]
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    allc = sorted(cell["cases"]); cases = allc[slice(*[int(v) for v in sl.split(":")])] if sl else allc
    for case in cases:
        f_out = od / f"{case}_scramble{'q' if qset else ''}.csv"; bl = QBANDS if qset else BANDS
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, _ = sw.load_case(pr0, cell, case, T); tg = np.isin(gp, T)
        B, err = bands(ip); log(f"[{key}] {case}: patch {ip.shape}, body {bp.mean():.2f}, target {int(tg.sum())} vox, band-sum max|err| {err:.2e}")
        ae, dp = check_scramble(B["fine"], np.random.default_rng(12345)); log(f"[{key}] {case}: fine-band phase randomisation: amplitude max rel err {ae:.1e}, median |phase change| {dp:.2f} rad (uniform -> pi/2 = 1.57)")
        imgs = {("none", -1): ip}; q = {}
        tc0 = tex_contrast(ip, tg, bp)
        for b in bl:
            for s in range(n_seeds(b)):
                rng = np.random.default_rng([s, RNG_IDX[SRC[b]], zlib.crc32(case.encode())])
                xc = counterfactual(ip, B[SRC[b]], replacement(b, B, bp, rng), bp); imgs[(b, s)] = xc; q[(b, s)] = qc(ip, xc, gp, T) + (tex_contrast(xc, tg, bp),)
        log(f"[{key}] {case}: QC seed 0 (edge / tex ratio / texture contrast, natural {tc0:.2f}): " + ", ".join(f"{b} {q[(b, 0)][0]:.2f}/{q[(b, 0)][1]:.2f}/{q[(b, 0)][2]:.2f}" for b in bl))
        res = {}   # (arm, band, seed) -> dice, J, pred_dJ
        for arm, pr in prs.items():
            net = pr.network; P = {k_: 0 for k_ in imgs}; Js = {k_: [] for k_ in imgs}; G = 0
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                if tg.any():
                    G = G + grad_J(net, ip, tg, T, dev) / len(FOLDS)
                for k_, xi in imgs.items():
                    with torch.no_grad():
                        lg = sw.decoder_fwd(net, sw.encoder_fwd(net, torch.from_numpy(xi)[None, None].to(dev)))
                    _, j_, p_ = sw.dice_and_j(lg, gp, T); P[k_] = P[k_] + p_.cpu().numpy() / len(FOLDS); Js[k_].append(j_)
                torch.cuda.empty_cache()
            for k_, xi in imgs.items():
                pdj = float((G * (xi - ip)).sum()) if (tg.any() and k_[0] != "none") else np.nan
                res[(arm,) + k_] = (sw.dice_prob(P[k_], gp, T), float(np.mean(Js[k_])), pdj)
        base = dict(cell=key, task=cell["task"], case=case); rows = []
        for arm in prs:
            d0, j0, _ = res[(arm, "none", -1)]
            if not qset:
                rows.append(dict(base, arm=arm, band="none", seed=-1, dice=d0, J=j0, dJ=np.nan, pred_dJ=np.nan, edge_ratio=1.0, tex_ratio=1.0))
            for b in bl:
                per = []
                for s in range(n_seeds(b)):
                    d, j, pdj = res[(arm, b, s)]; er, tr, tc1 = q[(b, s)]
                    per.append(dict(base, arm=arm, band=b, seed=s, dice=d, J=j, dJ=j - j0, pred_dJ=pdj, edge_ratio=er, tex_ratio=tr, tc0=tc0, tc1=tc1))
                m = pd.DataFrame(per)[["dice", "J", "dJ", "pred_dJ", "edge_ratio", "tex_ratio", "tc0", "tc1"]].mean()
                rows.append(dict(base, arm=arm, band=b, seed=-1, **{c: float(m[c]) for c in m.index})); rows += per
        pd.DataFrame(rows).to_csv(f_out, index=False)
        nat = " ".join(f"{a} {res[(a, 'none', -1)][0]:.3f}" for a in prs)
        log(f"[{key}] {case}: {len(rows)} rows, {time.time() - t0:.0f} s; natural Dice {nat}")
    print("SCRAMBLE_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
def _p(p):
    return "n/a" if not np.isfinite(p) else f"{p:.1g}"


def _sig_pos(d, p):
    return bool(np.isfinite(p) and p < ALPHA and np.median(d) > 0)


def report():
    cells = [k for k in vo.ORDER if (OUT / k).exists() and list((OUT / k).glob("*_scramble.csv"))]
    S = pd.concat([pd.read_csv(f) for k in cells for pat in ("*_scramble.csv", "*_scrambleq.csv") for f in sorted((OUT / k).glob(pat))], ignore_index=True)
    RB = [b for b in ORDER_ALL if (S["band"] == b).any()]; has_q = all(b in RB for b in QBANDS)
    M = S[S["seed"] == -1]; nseed = int(S["seed"].max()) + 1
    rows, drops, verd = [], {}, {}
    for k in cells:
        Mk = M[M["cell"] == k]
        nat = {a: Mk[(Mk["arm"] == a) & (Mk["band"] == "none")].set_index("case")["dice"] for a in ARMS}
        for b in RB:
            for a in ARMS:
                sb = Mk[(Mk["arm"] == a) & (Mk["band"] == b)].set_index("case")
                r = (nat[a] - sb["dice"]).dropna(); drops[(k, b, a)] = r
                rel = (r / nat[a].reindex(r.index)).replace([np.inf, -np.inf], np.nan)
                rel = rel[nat[a].reindex(r.index) > 0.05]
                ok = sb[["pred_dJ", "dJ"]].dropna(); rho = spearmanr(ok["pred_dJ"], ok["dJ"])[0] if len(ok) >= 3 else np.nan
                rows.append(dict(cell=k, band=b, arm=a, n=len(r), natural_dice=float(nat[a].median()), r_b=float(r.median()), r_b_q25=float(r.quantile(0.25)), r_b_q75=float(r.quantile(0.75)),
                                 r_rel=float(rel.median()) if len(rel) else np.nan, dJ=float(sb["dJ"].median()), pred_dJ=float(sb["pred_dJ"].median()), spearman_pred_vs_dJ=float(rho),
                                 edge_ratio=float(sb["edge_ratio"].median()), tex_ratio=float(sb["tex_ratio"].median()), tc0=float(sb["tc0"].median()) if "tc0" in sb else np.nan, tc1=float(sb["tc1"].median()) if "tc1" in sb else np.nan))
            for x_, y_ in (("real", "flat"), ("real", "noise"), ("noise", "flat")):
                d = (drops[(k, b, x_)] - drops[(k, b, y_)]).dropna(); verd[(k, b, f"{x_}-{y_}")] = (d, wp(d))
    R = pd.DataFrame(rows)
    for (k, b, c), (d, p) in verd.items():
        R.loc[(R["cell"] == k) & (R["band"] == b), f"{c}_median"] = float(d.median()) if len(d) else np.nan
        R.loc[(R["cell"] == k) & (R["band"] == b), f"{c}_p"] = p
    R.to_csv(OUT / "scramble_cells.csv", index=False)

    L = ["# Scale-band scrambling: the reliance spectrum (visual proof v14, FINDINGS 41)", "",
         f"Design: PREREG.md (this directory). Cells: {', '.join(cells)}; cases: " + ", ".join(f"{k} {M[M['cell'] == k]['case'].nunique()}" for k in cells) + f"; seeds per band: {nseed}. "
         "Bands (Gaussian sigma in voxels): fine = x - G1, mid = G1 - G2, coarse = G2 - G4, base = G4, allfine = fine + mid; *_le = local fine-energy map matched (Gaussian sigma 2 gain, applied twice) before the global rescale. x^(-b) = x - body*x_b + scr(x_b): same amplitude spectrum, "
         "random Hermitian phases, body-masked, body energy restored. Dice = target Dice on the plan patch (fold-mean softmax); r_b = Dice(x) - Dice(x^(-b)) (seed mean, then median over cases); "
         "rel = r_b / Dice(x) (cases with Dice(x) > 0.05). Differences real - flat etc. are per-case differences of r_b, Wilcoxon over cases.", ""]
    for k in cells:
        Rk = R[R["cell"] == k]
        L += [f"## {k} (natural Dice noise {Rk[Rk['arm'] == 'noise']['natural_dice'].iloc[0]:.3f} / flat {Rk[Rk['arm'] == 'flat']['natural_dice'].iloc[0]:.3f} / real {Rk[Rk['arm'] == 'real']['natural_dice'].iloc[0]:.3f})", "",
              "| band | r_b noise | r_b flat | r_b real | rel noise | rel flat | rel real | real-flat (p) | real-noise (p) | noise-flat (p) | rho(pred,dJ) noise / flat / real | edge ratio | tex ratio | tex contrast before -> after |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in RB:
            r = {a: Rk[(Rk["band"] == b) & (Rk["arm"] == a)].iloc[0] for a in ARMS}
            def dc(c):
                d, p = verd[(k, b, c)]; return f"{d.median():+.3f} ({_p(p)})" if len(d) else "n/a"
            L.append(f"| {b} | " + " | ".join(f"{r[a]['r_b']:+.3f}" for a in ARMS) + " | " + " | ".join(f"{r[a]['r_rel']:+.2f}" for a in ARMS) + f" | {dc('real-flat')} | {dc('real-noise')} | {dc('noise-flat')} | "
                     + " / ".join(f"{r[a]['spearman_pred_vs_dJ']:.2f}" for a in ARMS) + f" | {r['real']['edge_ratio']:.2f} | {r['real']['tex_ratio']:.2f} | " + (f"{r['real']['tc0']:.2f} -> {r['real']['tc1']:.2f}" if np.isfinite(r['real']['tc1']) else "") + " |")
        L.append("")
    # ---- verdicts, read on both variants: fine_le (local-energy matched, PRIMARY = structure only) and fine (global energy)
    L += ["## Pre-registered verdicts (alpha 0.01, Wilcoxon over cases)", "",
          "Read on both scrambling variants: **fine_le** (local fine-energy map matched, primary: destroys structure only) and **fine** (global body energy only; also changes the local "
          "heterogeneity level). The specificity check uses coarse and base (one variant each).", ""]
    nc = len(cells)
    for fb, tag in (("fine_le", "PRIMARY, local-energy matched"), ("fine", "global-energy variant")):
        L += [f"### Band {fb} ({tag})", ""]
        n_fine_pos = n_fine_le = 0; spec = []
        for k in cells:
            d, p = verd[(k, fb, "real-flat")]; pos = _sig_pos(d, p); n_fine_pos += pos; n_fine_le += bool(len(d) and np.median(d) <= 0)
            cb = [b for b in ("coarse", "base") if _sig_pos(*verd[(k, b, "real-flat")])]; spec.append((k, pos, cb))
            L.append(f"- {k}: r_{fb} real - flat {np.median(d) if len(d) else np.nan:+.3f} (p {_p(p)}) -> {'real > flat' if pos else 'not significant / not positive'}; "
                     f"coarse/base real - flat significantly positive: {', '.join(cb) if cb else 'none'}")
        if n_fine_pos >= 3 and all(not cb for _, pos, cb in spec if pos):
            p1 = f"SUPPORTED: r_{fb} real > flat in {n_fine_pos}/{nc} cells and the ordering is specific to fine scale (coarse/base not significantly positive in those cells)."
        elif n_fine_pos >= 3:
            p1 = f"PARTLY: r_{fb} real > flat in {n_fine_pos}/{nc} cells, but coarse/base show the same ordering in some of them (not fine-scale specific)."
        elif n_fine_le >= 3:
            p1 = f"KILLED: r_{fb} real <= flat in {n_fine_le}/{nc} cells."
        else:
            p1 = f"NOT SUPPORTED (inconclusive): r_{fb} real > flat significant in {n_fine_pos}/{nc} cells (need >= 3); real <= flat in {n_fine_le}/{nc}."
        L += ["", f"**P1 ({fb}, reliance ordering):** {p1}", ""]
        p2c = []
        for k in cells:
            d, p = verd[(k, fb, "noise-flat")]; p2c.append((k, np.median(d) if len(d) else np.nan, p, _sig_pos(d, p)))
        n_viol = sum(v for *_, v in p2c)
        L.append(f"**P2 ({fb}, noise arm r <= flat):** {'HOLDS' if n_viol == 0 else 'VIOLATED'} (noise significantly greater than flat in {n_viol}/{nc} cells; 'not greater' is read as no "
                 "significant positive noise - flat difference). " + "; ".join(f"{k} {m:+.3f} (p {_p(p)})" for k, m, p, _ in p2c))
        L.append("")
        rr = [(k, R[(R["cell"] == k) & (R["band"] == fb) & (R["arm"] == "real")]["spearman_pred_vs_dJ"].iloc[0]) for k in cells]
        n_fin = sum(bool(np.isfinite(r)) for _, r in rr); n_ok = sum(bool(np.isfinite(r) and r > RHO_MIN) for _, r in rr)
        if n_fin == 0:
            p3 = "n/a (Spearman undefined: fewer than 3 cases per cell)"
        else:
            p3 = (f"{n_ok}/{n_fin} cells with a defined rho pass -> " + ("SUPPORTED" if n_ok == n_fin else ("PARTLY" if n_ok else
                  "NOT SUPPORTED: the gradient-based sensitivity results (sections 33-37) do not predict reliance")))
        L.append(f"**P3 ({fb}, gradient link, real arm, Spearman > {RHO_MIN}):** {p3}. " + "; ".join(f"{k} {r:.2f}" for k, r in rr))
        L.append("")
        qq = [(k, R[(R["cell"] == k) & (R["band"] == fb) & (R["arm"] == "real")]["edge_ratio"].iloc[0], R[(R["cell"] == k) & (R["band"] == fb) & (R["arm"] == "real")]["tex_ratio"].iloc[0]) for k in cells]
        q_ok = all(QC_LO <= v <= QC_HI for _, v, _ in qq)
        L.append(f"**QC ({fb}, GT-boundary edge ratio within [{QC_LO}, {QC_HI}]):** {'PASS' if q_ok else 'FAIL: this scrambling is also a boundary edit; read as fine-scale appearance (texture + fine boundary detail)'}. "
                 + "; ".join(f"{k} edge {v:.2f}, target texture ratio {t:.2f}" for k, v, t in qq))
        L.append("")
    # ---- addendum (PREREG_addendum.md): energy-map flattening, P4 / P5
    if has_q:
        L += ["## Addendum (PREREG_addendum.md, set after the first smoke tests / after the 4-cell result): structure vs energy map", "",
              "Crossed factors: fine = structure destroyed, energy spread evenly over the body; fine_le = structure destroyed, local energy map kept; "
              "fine_eq = structure kept, map flattened (deterministic, 1 seed); fine_eq_scr = structure destroyed AND map flattened. Same for allfine. "
              "Texture contrast = mean fine energy in the target / in the body (natural -> after).", ""]
        n4 = n4_kill = 0
        for k in cells:
            d, p = verd[(k, "fine_eq", "real-flat")]; dn, pn = verd[(k, "fine_eq", "real-noise")]; pos = _sig_pos(d, p); posn = _sig_pos(dn, pn)
            n4 += pos and posn; n4_kill += not pos
            L.append(f"- {k}: r_fine_eq real - flat {np.median(d):+.3f} (p {_p(p)}), real - noise {np.median(dn):+.3f} (p {_p(pn)})")
        p4 = (f"SUPPORTED: real > flat and real > noise on fine_eq in {n4}/{len(cells)} cells" if n4 >= 3 else
              (f"KILLED: real - flat on fine_eq not significantly positive in {n4_kill}/{len(cells)} cells" if n4_kill >= 2 else
               f"NOT SUPPORTED: real > flat and > noise in {n4}/{len(cells)} cells (need >= 3)"))
        L += ["", f"**P4 (structure kept, map flattened: r_fine_eq real > flat in >= 3/4 cells and real > noise):** {p4}.", ""]
        n5 = 0
        for k in cells:
            Mk = M[M["cell"] == k]; nat = {a: Mk[(Mk["arm"] == a) & (Mk["band"] == "none")].set_index("case")["dice"] for a in ARMS}
            r = {(b, a): drops[(k, b, a)] for b in ("fine", "fine_le", "fine_eq", "fine_eq_scr") for a in ARMS}
            c_eqs = (r[("fine_eq_scr", "real")] - r[("fine_eq_scr", "flat")]); c_f = (r[("fine", "real")] - r[("fine", "flat")])
            dd = (c_eqs - c_f).dropna(); pdd = wp(dd); ps = _sig_pos(c_eqs.dropna(), wp(c_eqs.dropna())); ok = ps and not (np.isfinite(pdd) and pdd < ALPHA)
            n5 += ok
            per_arm = "; ".join(f"{a}: r(eq_scr) - r(fine) {(r[('fine_eq_scr', a)] - r[('fine', a)]).median():+.3f} (p {_p(wp((r[('fine_eq_scr', a)] - r[('fine', a)]).dropna()))}), "
                                f"additivity residual r(eq_scr) - [r(le) + r(eq)] {(r[('fine_eq_scr', a)] - r[('fine_le', a)] - r[('fine_eq', a)]).median():+.3f}" for a in ARMS)
            L.append(f"- {k}: real - flat on fine_eq_scr {c_eqs.median():+.3f} (p {_p(wp(c_eqs.dropna()))}) vs on fine {c_f.median():+.3f}; difference {dd.median():+.3f} (p {_p(pdd)}) -> "
                     f"{'holds' if ok else 'does not hold'}. {per_arm}")
        L += ["", f"**P5 (fine_eq_scr ~ fine: flattening explains the global-scrambling effect):** {'SUPPORTED' if n5 >= 3 else 'NOT SUPPORTED'} ({n5}/{len(cells)} cells).", ""]
        L += ["**QC (addendum variants; real-arm rows, values are identical across arms):**", ""]
        for k in cells:
            parts = []
            for b in ("fine", "fine_le", "fine_eq", "fine_eq_scr", "allfine_eq", "allfine_eq_scr"):
                x = R[(R["cell"] == k) & (R["band"] == b) & (R["arm"] == "real")].iloc[0]
                parts.append(f"{b} edge {x['edge_ratio']:.2f} tex {x['tex_ratio']:.2f}" + (f" contrast {x['tc0']:.2f} -> {x['tc1']:.2f}" if np.isfinite(x["tc1"]) else ""))
            L.append(f"- {k}: " + "; ".join(parts))
        L.append("")
    (OUT / "scramble.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

    # ---- figure: bars of all bands, then predicted vs measured J change for fine and fine_le
    nc_ = max(1, len(cells)); fig, axs = plt.subplots(3 + int(has_q), nc_, figsize=(6.4 * nc_, 11.0 + 3.6 * int(has_q)), squeeze=False); w = 0.27
    for c, k in enumerate(cells):
        ax = axs[0, c]
        for i, a in enumerate(ARMS):
            med = [drops[(k, b, a)].median() for b in RB]; lo = [drops[(k, b, a)].quantile(0.25) for b in RB]; hi = [drops[(k, b, a)].quantile(0.75) for b in RB]
            ax.bar(np.arange(len(RB)) + (i - 1) * w, med, w, color=PAL[a], label=a, yerr=[np.subtract(med, lo), np.subtract(hi, med)], error_kw=dict(lw=0.7, capsize=1.5))
        ax.set_xticks(range(len(RB))); ax.set_xticklabels(RB, fontsize=7.5, rotation=30); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{k}\nDice drop when one band is phase-scrambled (_le = local energy matched)", fontsize=8)
        if c == 0:
            ax.set_ylabel("Dice(x) - Dice(x^(-b)) (median, IQR)"); ax.legend(fontsize=7, frameon=False)
        for row, fb in [(1, "fine"), (2, "fine_le")] + ([(3, "fine_eq")] if has_q else []):
            ax = axs[row, c]; F = M[(M["cell"] == k) & (M["band"] == fb)]
            for a in ARMS:
                f = F[F["arm"] == a]; rho = R[(R["cell"] == k) & (R["band"] == fb) & (R["arm"] == a)]["spearman_pred_vs_dJ"].iloc[0]
                ax.scatter(f["pred_dJ"], f["dJ"], s=9, color=PAL[a], alpha=0.7, label=f"{a} (rho {rho:.2f})")
            lim = np.nanmax(np.abs(F[["pred_dJ", "dJ"]].values)) if len(F) else 1
            ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.6); ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
            ax.set_xlabel(f"first-order prediction <G, x^(-{fb}) - x>"); ax.set_title(f"{fb}: predicted vs measured change in J", fontsize=8); ax.legend(fontsize=6.5, frameon=False)
            if c == 0:
                ax.set_ylabel(f"measured J(x^(-{fb})) - J(x)")
    fig.tight_layout(); fig.savefig(OUT / "scramble_summary.png", dpi=150); fig.savefig(OUT / "scramble_summary.pdf"); plt.close(fig); print("SCRAMBLE_REPORT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true"); ap.add_argument("--set", default="main", choices=["main", "q"])
    args = ap.parse_args()
    if args.report:
        report(); return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.cells:
        run_cell(key, dev, args.case_slice, qset=args.set == "q")


if __name__ == "__main__":
    main()
