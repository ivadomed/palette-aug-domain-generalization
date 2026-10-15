#!/usr/bin/env python
"""Visual proof v16 (FINDINGS 43, pre-registered in outputs/plots/texig/PREREG.md): "what did the model use to predict this pixel" -- Integrated
Gradients from the texture-flattened twin, BraTS cells, arms noise (rung 4, lblvor) / flat (4.5) / real (5), folds 0-2 (fold-mean maps).

Baseline x_eq = the v15 counterfactual (visual_proof_texthink.make_counterfactuals with TT_ITER = 2 outer passes): the local fine-energy map
G2(fine^2) flattened to the body mean, every local pattern kept; mask-free, label-free. d = x - x_eq is zero outside the body.
Seeds = the section-36 seeds of the case (erf_outcome/<cell>/seeds.csv, <= MAX_SEEDS = 12 per case, rare outcome patterns first: same rule as
visual_proof_pathgrad.pick_seeds), each explained with each arm's OWN predicted class (`<arm>_pred`); f_v = log-odds of that class at v.
Integrated Gradients per arm, fold and seed batch (BATCH 2, fp32, autocast off): x_t = x_eq + t_m d, t_m = (m - 0.5) / M (midpoint rule,
M = env IG_STEPS, default 16); one backward per step gives the gradient of f_v w.r.t. the input AND w.r.t. the activations at s3 (encoder skip 3)
and d2 (decoder stage 2 output). A_v = d * mean_t df_v/dx (input level); C_v(L) = sum_c (h_c(x) - h_c(x_eq)) * mean_t df_v/dh_c at L (feature level,
Grad-CAM-style on the counterfactual path; no completeness), upsampled nearest to the patch. All maps fold-mean.
Completeness: sum_u A_v(u) vs f_v(x) - f_v(x_eq) (both fold-mean): absolute and relative error per seed in the csv.
Numbers per seed and arm (map M in A, C3, Cd2): <M>_sum (signed total of the map), <M>_in_abs / _ring_abs / _bg_abs / _out_abs = fractions of the
ABSOLUTE map in the GT target / ring (dilation 10 minus target, in body) / far background (body minus dilation 10) / outside the body, denominator =
sum |M| over the whole patch; <M>_in = SIGNED sum over the target / sum |M| over the patch (the pre-registered m_in); A_r10 = |A| fraction within
10 vox of the seed, A_fine = fine-detail share of A (energy above a sigma-1 Gaussian, body).
Added on request before the full run: texture-evidence density A_sm = 3D Gaussian (sigma 2 vox, signed) of A_v; the same region numbers on A_sm (suffix
_sm, plus A_r10_sm), its slice in the npz (<arm>_Asm), its panels in the example figure; P2 also read on A_sm and on C_v(s3).
Also added before the full run (coordinator): a second IG pass per seed explaining the TARGET class (edema) log-odds for ALL arms (columns suffix _tgt;
reused from the own pass when the arm predicts the target class); the report reads P1 and P2 on _tgt as PRIMARY, own class as secondary; the npz keeps
only the target-class maps.
Fold agreement (coordinator, before the second half of the main run and the rand pass): per-fold target-class A slices in the npz (<arm>_A_f<k>),
per seed cons_frac_tgt = sum |Acons| / sum |Abar|, Acons = sign(Abar) min_k max(A_k sign(Abar), 0), and Ac_* target-mass metrics on Acons (+ _sm);
report section "Fold agreement"; the main example figure uses Acons (fold-mean copy: _foldmean.png).
Edge-mask control (--edgemask, coordinator): baseline x_em = x_eq with every gain blended toward 1 near coarse edges (make_xem), three trained arms,
target-class IG with per-fold maps, first EM_N cases + the v7 example cases per cell; <case>_texig_em.csv (suffix _em, boundary edge-energy ratios),
<case>_texig_em_maps.npz; report section "Edge-mask control"; figure texig_examples_<cell>_edgemask.png.
Outputs per cell under OUT/<cell>/: <case>_texig_maps.npz (float16, per arm A and C3 slices at each seed's z stored as map / scale with a per-seed
scale; ip, Emap, gt slices; seeds; patient data: never commit), <case>_texig.csv (written last; one row per seed and arm). Finished cases skip.
--report: texig.md, texig_cells.csv, texig_summary.png/pdf. --figure: texig_examples_<cell>.png (main, Paul review v6: the fine texture each model drew on in its own colour, who-uses composite, model-specific excess
its evidence, added on request) and texig_examples_<cell>_signed.png (signed A / A_sm / C(s3) maps) (patient slices: never commit).
OUT = env VP_TEXIG_OUT or <THIS.parent>/outputs/plots/texig.
GPU: .venv/bin/python visual_proof_texig.py --cells <cell> [--case-slice i:j]      CPU: --report / --figure
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_erf_outcome as vo  # noqa: E402
import visual_proof_v2 as v2  # noqa: E402
import visual_proof_erf as e7  # noqa: E402
import visual_proof_texthink as tt  # noqa: E402
from visual_proof_switch import load_case, encoder_fwd, decoder_fwd, activations, models_for, wp, FOLDS, ARMS, PAL  # noqa: E402

tt.TT_ITER = 2   # x_eq = the two-outer-pass flattening (v15 primary construction); make_counterfactuals reads this module global at call time
OUT = Path(os.environ.get("VP_TEXIG_OUT") or (THIS.parent / "outputs/plots/texig")); OUT.mkdir(parents=True, exist_ok=True)
SEEDS_IN = vo.OUT
CELLS4 = ("glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w")
SUFX = "_rand"
IG_STEPS = int(os.environ.get("IG_STEPS", 16)); BATCH = 2; MAX_SEEDS = 12; RING = 10; R_NEAR = 10; FLOOR = 0.1; SIG_SM = 2.0
SIG_E = 2.0; FEAT = ("s3", "d2"); MAPS = ("A", "C3", "Cd2")
ARMS_ORDER = ("real", "flat", "noise")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pick_seeds(d):
    """<= MAX_SEEDS per case, rare patterns (everything but CCC / FFF) first, then CCC / FFF (visual_proof_pathgrad.pick_seeds, MAX_SEEDS = 12)."""
    rare = d[~d.pattern.isin(["CCC", "FFF"])]; common = d[d.pattern.isin(["CCC", "FFF"])]
    rng = np.random.default_rng(0)
    if len(rare) >= MAX_SEEDS:
        return rare.iloc[sorted(rng.choice(len(rare), MAX_SEEDS, replace=False))]
    k = MAX_SEEDS - len(rare); fill = common.iloc[sorted(rng.choice(len(common), min(k, len(common)), replace=False))] if len(common) else common
    return pd.concat([rare, fill]).sort_values("seed")


def read_seeds(key):
    f_in = SEEDS_IN / key / "seeds.csv"
    if not f_in.exists() or f_in.stat().st_size < 20:
        return None
    SD = pd.read_csv(f_in, dtype=str); SD = SD[SD.case != "case"]
    for c in ("seed", "z", "y", "x", "cls", "band", "depth") + tuple(f"{a}_pred" for a in ARMS):
        SD[c] = pd.to_numeric(SD[c])
    return SD


def fine_share(G, body, s=1.0):
    hp = G - ndi.gaussian_filter(G, s); return float((hp[body] ** 2).sum() / ((G[body] ** 2).sum() + 1e-30))


# ----------------------------------------------------------------------------------------------------------------------------------- IG
def ig_batch(net, x_eq, dx, V, cls, store, M):
    """Mean over the M midpoint steps of the gradient of each seed's log-odds w.r.t. the input, s3 and d2 (batch = len(V) seeds)."""
    import torch
    B = len(V); gx = g3 = gd = None
    for m in range(M):
        t = (m + 0.5) / M
        xb = (x_eq + t * dx)[None, None].repeat(B, 1, 1, 1, 1).clone().requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            skips = encoder_fwd(net, xb); o = decoder_fwd(net, skips)
            f = sum(e7.log_odds(o[b:b + 1], c)[0][tuple(v)] for b, (v, c) in enumerate(zip(V, cls)))
            a, b3, bd = torch.autograd.grad(f, [xb, skips[3], store["d2"]])
        a, b3, bd = a[:, 0].detach(), b3.detach(), bd.detach()
        gx = a if gx is None else gx + a; g3 = b3 if g3 is None else g3 + b3; gd = bd if gd is None else gd + bd
        del xb, skips, o, f
    return gx / M, g3 / M, gd / M


def region_stats(Mp, tg, ring, bg, body, pre):
    ab = np.abs(Mp); tot = float(ab.sum()) + 1e-30
    return {f"{pre}_sum": float(Mp.sum()), f"{pre}_in": float(Mp[tg].sum()) / tot, f"{pre}_in_abs": float(ab[tg].sum()) / tot,
            f"{pre}_ring_abs": float(ab[ring].sum()) / tot, f"{pre}_bg_abs": float(ab[bg].sum()) / tot, f"{pre}_out_abs": float(ab[~body].sum()) / tot}


EM_N = 12   # edge-mask control: first EM_N cases (sorted) of each cell + the v7 example cases


def make_xem(ip, bp, hp):
    """Edge-mask control twin: the x_eq construction of visual_proof_texthink.make_counterfactuals (N_PASS gain passes, then TT_ITER image-level passes),
    with every gain g blended toward 1 near coarse edges: g_em = 1 + (g - 1) e, e = exp(-(|grad G2(x)| / p90_body)^2), e from the ORIGINAL image."""
    f = hp.astype(np.float64); b = bp.astype(bool); m = float((f[b] ** 2).mean()); eps = tt.EPS_REL * m
    gr = ndi.gaussian_gradient_magnitude(ip.astype(np.float64), tt.SIG_E); e = np.exp(-(gr / (np.percentile(gr[b], 90) + 1e-12)) ** 2)

    def gain(s):
        for _ in range(tt.N_PASS):
            g = np.sqrt((m + eps) / (tt.G(s ** 2, tt.SIG_E) + eps)); s = s * (1 + (g - 1) * e)
        return s
    x = ip.astype(np.float64) - f + gain(f.copy()) * b
    for _ in range(tt.TT_ITER):
        r = (x - tt.G(x, tt.SIG_FINE)) * b; x = x - r + gain(r.copy()) * b
    return x.astype(np.float32)


def edge_ratio(ip, x, gp, T):
    """Mean |grad G2| on the GT boundary shell (dilate 2 minus erode 2), counterfactual / original (as visual_proof_texthink.qc_stats)."""
    tg = np.isin(gp, T); shell = ndi.binary_dilation(tg, iterations=2) & ~ndi.binary_erosion(tg, iterations=2)
    if not shell.any():
        return np.nan
    g0 = ndi.gaussian_gradient_magnitude(ip.astype(np.float64), tt.SIG_E); g1 = ndi.gaussian_gradient_magnitude(x.astype(np.float64), tt.SIG_E)
    return float(g1[shell].mean() / (g0[shell].mean() + 1e-12))


def fine_boundary(ip, x, gp, T):
    """Fine-scale boundary check on the GT shell (dilate 2 minus erode 2), counterfactual x vs original ip: Pearson correlation and mean ratio of the
    sigma-1 gradient magnitude, and a single-window SSIM of the sigma-1 high-pass (x - G1 x) within the shell (c1, c2 from the original's dynamic range)."""
    tg = np.isin(gp, T); sh = ndi.binary_dilation(tg, iterations=2) & ~ndi.binary_erosion(tg, iterations=2)
    if sh.sum() < 10:
        return dict(corr=np.nan, ratio=np.nan, ssim=np.nan)
    a0 = ndi.gaussian_gradient_magnitude(ip.astype(np.float64), 1.0)[sh]; a1 = ndi.gaussian_gradient_magnitude(x.astype(np.float64), 1.0)[sh]
    h0 = (ip.astype(np.float64) - ndi.gaussian_filter(ip.astype(np.float64), 1.0))[sh]; h1 = (x.astype(np.float64) - ndi.gaussian_filter(x.astype(np.float64), 1.0))[sh]
    Lr = float(h0.max() - h0.min()) or 1.0; c1, c2 = (0.01 * Lr) ** 2, (0.03 * Lr) ** 2
    ss = ((2 * h0.mean() * h1.mean() + c1) * (2 * np.mean((h0 - h0.mean()) * (h1 - h1.mean())) + c2)) / ((h0.mean() ** 2 + h1.mean() ** 2 + c1) * (h0.var() + h1.var() + c2))
    return dict(corr=float(np.corrcoef(a0, a1)[0, 1]), ratio=float(a1.mean() / (a0.mean() + 1e-12)), ssim=float(ss))


def example_cases(S, k):
    """The v7 example cases of a cell: among cases with per-fold maps, the per-case real - flat target-class delta (median over target seeds) closest to the
    cell median and to its 75th percentile. Returns (picks, drf)."""
    St = S[(S.cell == k) & (S.region == "target") & S.arm.isin(ARMS)]
    pc = St.groupby(["case", "arm"]).delta_tgt.median().unstack()
    pc = pc[[(OUT / k / f"{c}_texig_maps.npz").exists() for c in pc.index]]
    pc = pc[[("real_A_f0" in np.load(OUT / k / f"{c}_texig_maps.npz", allow_pickle=True).files) for c in pc.index]]
    if not len(pc) or "real" not in pc or "flat" not in pc:
        return [], pd.Series(dtype=float)
    drf = (pc["real"] - pc["flat"]).dropna(); picks = []
    for qv in (0.5, 0.75):
        tq = float(drf.quantile(qv))
        for c in (drf - tq).abs().sort_values().index:
            if c not in picks:
                picks.append(c); break
    return picks, drf


def run_cell(key, dev, sl=None, rand=False, em=False):
    """rand = architectural floor pass: one arm "rand" = the real arm's network re-initialised by reset_parameters (torch seed 1000 + k per fold, as
    visual_proof_pathgrad PG_RAND), target-class IG only; writes <case>_texig_rand.csv and <case>_texig_rand_maps.npz.
    em = edge-mask control: baseline x_em (make_xem: the x_eq gain blended toward 1 near coarse edges), three trained arms, target-class IG only, on the
    first EM_N cases (sorted) + the v7 example cases; columns suffix _em in <case>_texig_em.csv, maps in <case>_texig_em_maps.npz."""
    tgt_only = rand or em; sfx = SUFX if rand else ("_em" if em else "")
    import torch
    import torch.nn.functional as Fn
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    SD = read_seeds(key)
    if SD is None:
        log(f"[{key}] no section-36 seeds at {SEEDS_IN / key}"); return
    prs = {a: v2.load_predictor(d, dev) for a, d in models_for(key).items()}; pr0 = prs["flat"]
    if rand:
        prs = {"flat": prs["flat"], "rand": prs["real"]}
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    allc = sorted(cell["cases"]); cases = allc[slice(*[int(v) for v in sl.split(":")])] if sl else allc
    if em:
        _, S_ = load_all(); picks, _ = example_cases(S_, key); cases = list(dict.fromkeys(allc[:EM_N] + picks)); log(f"[{key}] edge-mask cases: {len(cases)} (examples {picks})")
    log(f"[{key}] IG_STEPS {IG_STEPS}, {len(cases)} cases, seeds from {SEEDS_IN / key / 'seeds.csv'}")
    for case in cases:
        f_out = od / f"{case}_texig{sfx}.csv"
        if f_out.exists():
            continue
        if case not in set(SD.case):
            log(f"[{key}] {case}: no seeds in the csv, skipped"); continue
        t0 = time.time(); d = pick_seeds(SD[SD.case == case]); n_csv = int((SD.case == case).sum())
        ip, bp, gp, hp = load_case(pr0, cell, case, T)
        b = bp.astype(bool); tg = np.isin(gp, T); dil = ndi.binary_dilation(tg, iterations=RING); ring = dil & ~tg & b; bg = b & ~dil
        x_eq, _, _, _, _ = tt.make_counterfactuals(ip, bp, hp, case)
        if em:
            x_em = make_xem(ip, bp, hp); er_eq, er_em = edge_ratio(ip, x_eq, gp, T), edge_ratio(ip, x_em, gp, T)
            fb = {nm: fine_boundary(ip, xx_, gp, T) for nm, xx_ in (("eq", x_eq), ("em", x_em))}; E2 = ndi.gaussian_filter(hp.astype(np.float64) ** 2, SIG_E)
            ctr0 = float(E2[np.isin(gp, T)].mean() / (E2[bp.astype(bool)].mean() + 1e-12)) if np.isin(gp, T).any() else np.nan
            x_eq = x_em   # IG baseline = x_em
        dx_np = (ip - x_eq) * b
        Emap = ndi.gaussian_filter(hp.astype(np.float64) ** 2, SIG_E)
        V = [(int(r.z), int(r.y), int(r.x)) for r in d.itertuples()]; ns = len(V)
        X = torch.from_numpy(ip)[None, None].to(dev); XE = torch.from_numpy(x_eq).to(dev); DX = torch.from_numpy(dx_np.astype(np.float32)).to(dev)
        zz, yy, xx = np.indices(ip.shape, sparse=True)
        rows = []; sv = dict(seeds=np.array(V, np.int32), seed_id=np.array(d.seed, np.int32), pattern=np.array(d.pattern.astype(str).tolist(), dtype="U8"), region=np.array(d.region.astype(str).tolist(), dtype="U8"), cls=np.array(d.cls, np.int32),
                             z=np.array([v[0] for v in V], np.int32), ip=np.stack([ip[v[0]] for v in V]).astype(np.float16),
                             emap=np.stack([Emap[v[0]] for v in V]).astype(np.float16), gt=np.stack([tg[v[0]] for v in V]))
        c0 = int(T[0])
        for arm, pr in prs.items():
            if rand and arm != "rand":
                continue
            net = pr.network; own = [int(getattr(r, f"{arm}_pred")) for r in d.itertuples()] if not tgt_only else [-1] * len(d)
            # IG jobs: own predicted class for every seed (secondary), target class c0 for every seed (PRIMARY; reused from the own pass when own == c0)
            jobs = ([("own", i, own[i]) for i in range(ns)] if not tgt_only else []) + [("tgt", i, c0) for i in range(ns) if own[i] != c0]
            store = {}; hook = net.decoder.stages[2].register_forward_hook(lambda mod, inp, out: store.__setitem__("d2", out))
            acc = {(kd, i): [0.0, 0.0, 0.0] for kd, i, _ in jobs}; fx = {(kd, i): 0.0 for kd, i, _ in jobs}; fe = dict(fx); accf = {(kd, i): [] for kd, i, _ in jobs}   # accf: per-fold A (fold agreement)
            try:
                for k in FOLDS:
                    if rand:   # architectural floor: same architecture, random initialisation per "fold"
                        torch.manual_seed(1000 + k)
                        for m_ in net.modules():
                            if hasattr(m_, "reset_parameters"):
                                m_.reset_parameters()
                        net.eval()
                    else:
                        net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                    with torch.no_grad(), torch.autocast("cuda", enabled=False):
                        Ax, lgx = activations(net, X); Ae, lge = activations(net, XE[None, None])
                        dh = {"s3": (Ax["s3"] - Ae["s3"])[0].float(), "d2": (Ax["d2"] - Ae["d2"])[0].float()}
                        for kd, i, c in jobs:
                            fx[(kd, i)] += float(e7.log_odds(lgx.float(), c)[0][V[i]]) / len(FOLDS); fe[(kd, i)] += float(e7.log_odds(lge.float(), c)[0][V[i]]) / len(FOLDS)
                    del Ax, Ae, lgx, lge
                    for j0 in range(0, len(jobs), BATCH):
                        jb = jobs[j0:j0 + BATCH]
                        gx, g3, gd = ig_batch(net, XE, DX, [V[i] for _, i, _ in jb], [c for _, _, c in jb], store, IG_STEPS)
                        with torch.no_grad():
                            A = (DX[None] * gx) / len(FOLDS)
                            c3 = Fn.interpolate((dh["s3"][None] * g3).sum(1, keepdim=True), size=ip.shape, mode="nearest")[:, 0] / len(FOLDS)
                            cd = Fn.interpolate((dh["d2"][None] * gd).sum(1, keepdim=True), size=ip.shape, mode="nearest")[:, 0] / len(FOLDS)
                        for j, (kd, i, _) in enumerate(jb):
                            a_ = acc[(kd, i)]; a_[0] = a_[0] + A[j]; a_[1] = a_[1] + c3[j]; a_[2] = a_[2] + cd[j]
                            accf[(kd, i)].append(A[j] * len(FOLDS))   # this fold's own A_k = d * mean_t grad_k (fp32: values ~1e-6 would underflow fp16)
                        del gx, g3, gd, A, c3, cd
                    del dh; torch.cuda.empty_cache()
            finally:
                hook.remove()
            for i in range(ns):
                if own[i] == c0:
                    acc[("tgt", i)] = acc[("own", i)]; fx[("tgt", i)] = fx[("own", i)]; fe[("tgt", i)] = fe[("own", i)]; accf[("tgt", i)] = accf[("own", i)]
            sA, sAs, s3, s3s, sM, sMs = [], [], [], [], [], []; sF, sFs = {}, {}
            for i, r in enumerate(d.itertuples()):
                v = V[i]; near = np.sqrt((zz - v[0]) ** 2 + (yy - v[1]) ** 2 + (xx - v[2]) ** 2) <= R_NEAR
                row = dict(cell=key, task=cell["task"], case=case, seed=int(r.seed), region=r.region, pattern=r.pattern, band=int(r.band), z=v[0], y=v[1], x=v[2],
                           cls=int(r.cls), arm=arm, pred=own[i], pred_in_T=int(own[i] in T), tgt_cls=c0, n_target=int(tg.sum()), n_seeds=ns, n_seeds_csv=n_csv, ig_steps=IG_STEPS)
                if em:
                    row.update(edge_ratio_eq=er_eq, edge_ratio_em=er_em, ctr_orig=ctr0, **{f"fine_{q_}_{nm}": fb[nm][q_] for nm in fb for q_ in ("corr", "ratio", "ssim")})
                for kd, suf in ((("own", ""), ("tgt", "_tgt")) if not tgt_only else (("tgt", "_em" if em else "_tgt"),)):
                    Am, C3, CD = (t_.cpu().numpy().astype(np.float64) for t_ in acc[(kd, i)])
                    delta = fx[(kd, i)] - fe[(kd, i)]; sumA = float(Am.sum()); As = ndi.gaussian_filter(Am, SIG_SM)
                    m = dict(f_x=float(fx[(kd, i)]), f_eq=float(fe[(kd, i)]), delta=float(delta), ig_abserr=abs(sumA - delta), ig_relerr=abs(sumA - delta) / (abs(delta) + 1e-6),
                             A_r10=float(np.abs(Am)[near].sum() / (np.abs(Am).sum() + 1e-30)), A_fine=fine_share(Am, b), A_r10_sm=float(np.abs(As)[near].sum() / (np.abs(As).sum() + 1e-30)))
                    m.update(region_stats(Am, tg, ring, bg, b, "A")); m.update(region_stats(C3, tg, ring, bg, b, "C3")); m.update(region_stats(CD, tg, ring, bg, b, "Cd2"))
                    m.update({f"{k_}_sm": v_ for k_, v_ in region_stats(As, tg, ring, bg, b, "A").items()})
                    if kd == "tgt":   # fold agreement: Acons = sign(Abar) min_k max(A_k sign(Abar), 0) (zero where any fold disagrees in sign with the mean)
                        Ak = np.stack([t_.cpu().numpy() for t_ in accf[(kd, i)]]).astype(np.float64); Ab = Ak.mean(0); sg = np.sign(Ab)
                        Ac = sg * np.min(np.maximum(Ak * sg[None], 0), axis=0); Acs = ndi.gaussian_filter(Ac, SIG_SM)
                        m["cons_frac"] = float(np.abs(Ac).sum() / (np.abs(Ab).sum() + 1e-30)); m.update(region_stats(Ac, tg, ring, bg, b, "Ac"))
                        m["Sc_in_mean"] = float(Acs[tg].sum() / max(int(tg.sum()), 1))   # signed smoothed fold-consistent evidence per target voxel
                        m.update({f"{k_}_sm": v_ for k_, v_ in region_stats(Acs, tg, ring, bg, b, "Ac").items()})
                        for kf in range(Ak.shape[0]):
                            a2f = Ak[kf][v[0]]; sf = float(np.abs(a2f).max()) or 1.0; sF.setdefault(kf, []).append((a2f / sf).astype(np.float16)); sFs.setdefault(kf, []).append(sf)
                    row.update({f"{k_}{suf}": v_ for k_, v_ in m.items()})
                    if kd == "tgt":   # the npz keeps the TARGET-class maps (the ones the figure shows)
                        a2, c2, m2 = Am[v[0]], C3[v[0]], As[v[0]]; sa = float(np.abs(a2).max()) or 1.0; sc = float(np.abs(c2).max()) or 1.0; sm = float(np.abs(m2).max()) or 1.0
                        sA.append((a2 / sa).astype(np.float16)); sAs.append(sa); s3.append((c2 / sc).astype(np.float16)); s3s.append(sc); sM.append((m2 / sm).astype(np.float16)); sMs.append(sm)
                rows.append(row)
            sv[f"{arm}_A"] = np.stack(sA); sv[f"{arm}_A_scale"] = np.array(sAs, np.float64); sv[f"{arm}_C3"] = np.stack(s3); sv[f"{arm}_C3_scale"] = np.array(s3s, np.float64)
            sv[f"{arm}_Asm"] = np.stack(sM); sv[f"{arm}_Asm_scale"] = np.array(sMs, np.float64)
            for kf in sF:   # per-fold target-class A slices (fold agreement)
                sv[f"{arm}_A_f{kf}"] = np.stack(sF[kf]); sv[f"{arm}_A_f{kf}_scale"] = np.array(sFs[kf], np.float64)
            del acc, accf; torch.cuda.empty_cache()
        sv["maps_class"] = np.array("target", dtype="U8")
        np.savez_compressed(od / f"{case}_texig{sfx}_maps.npz", **sv)
        R = pd.DataFrame(rows); R.to_csv(f_out, index=False)
        Rt = R[R.region == "target"]
        if em:
            log(f"[{key}] {case}: edge-mask, {ns} seeds, boundary edge-energy ratio x_em {er_em:.3f} (x_eq {er_eq:.3f}), fine shell corr / ratio / ssim em {fb['em']['corr']:.3f}/{fb['em']['ratio']:.3f}/{fb['em']['ssim']:.3f} eq {fb['eq']['corr']:.3f}/{fb['eq']['ratio']:.3f}/{fb['eq']['ssim']:.3f}, completeness rel. err median {R.ig_relerr_em.median():.3f}; "
                "target seeds delta_em " + " ".join(f"{a} {Rt[Rt.arm == a].delta_em.median():+.2f}" for a in ARMS) + f"; {time.time() - t0:.0f} s"); continue
        if rand:
            log(f"[{key}] {case}: rand floor, {ns} seeds, completeness rel. err median {R.ig_relerr_tgt.median():.3f}; target seeds delta_tgt {Rt.delta_tgt.median():+.3f}, A_in_abs_tgt {Rt.A_in_abs_tgt.median():.3f}; {time.time() - t0:.0f} s"); continue
        log(f"[{key}] {case}: {ns} seeds (csv {n_csv}), completeness rel. err median target-class {R.ig_relerr_tgt.median():.3f} own {R.ig_relerr.median():.3f} "
            f"(|delta| > {FLOOR}: {R[R.delta_tgt.abs() > FLOOR].ig_relerr_tgt.median():.3f} / {R[R.delta.abs() > FLOOR].ig_relerr.median():.3f}); "
            "target seeds delta_tgt / A_in_abs_tgt noise flat real: " + " ".join(f"{a} {Rt[Rt.arm == a].delta_tgt.median():+.2f}/{Rt[Rt.arm == a].A_in_abs_tgt.median():.2f}" for a in ARMS) + f"; {time.time() - t0:.0f} s")
    print("TEXIG_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
MET = [("delta", "signed total f(x) - f(x_eq) (nat)"), ("A_in", "A: signed target mass / |A| total (m_in, pre-registered)"), ("A_in_abs", "A: |A| target fraction"),
       ("A_ring_abs", "A: |A| ring fraction"), ("A_bg_abs", "A: |A| far-background fraction"), ("A_r10", "A: |A| within 10 vox of the seed"), ("A_fine", "A: fine-detail share"),
       ("A_in_sm", "A smoothed (sigma 2): signed target mass / |A_sm| total"), ("A_in_abs_sm", "A smoothed: |A_sm| target fraction"), ("A_ring_abs_sm", "A smoothed: |A_sm| ring fraction"),
       ("A_bg_abs_sm", "A smoothed: |A_sm| far-background fraction"), ("A_r10_sm", "A smoothed: |A_sm| within 10 vox of the seed"),
       ("C3_sum", "C(s3): signed total"), ("C3_in", "C(s3): signed target mass / |C| total"), ("C3_in_abs", "C(s3): |C| target fraction"), ("C3_ring_abs", "C(s3): |C| ring fraction"),
       ("C3_bg_abs", "C(s3): |C| far-background fraction"), ("C3_out_abs", "C(s3): |C| outside body"),
       ("Cd2_sum", "C(d2): signed total"), ("Cd2_in", "C(d2): signed target mass / |C| total"), ("Cd2_in_abs", "C(d2): |C| target fraction"), ("Cd2_ring_abs", "C(d2): |C| ring fraction"),
       ("Cd2_bg_abs", "C(d2): |C| far-background fraction")]


def fp(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.1g}"


def load_all():
    cells = [k for k in CELLS4 if (OUT / k).exists() and list((OUT / k).glob("*_texig.csv"))]
    fs = [f for k in cells for f in sorted((OUT / k).glob("*_texig.csv"))] + [f for k in cells for f in sorted((OUT / k).glob("*_texig_rand.csv"))]   # rand = floor pass
    return cells, (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame())


SUBS = (("tgt", "_tgt", "PRIMARY: TARGET-class (edema) log-odds for all three arms, all target seeds"),
        ("own", "", "secondary (pre-registered wording): each arm's OWN predicted class, all target seeds"),
        ("own_agree", "", "sensitivity: own class, target seeds where all three arms predict the target class"))
P2COLS = (("A_in", "pre-registered m_in"), ("A_in_abs", "absolute target mass"), ("A_in_sm", "A smoothed sigma 2, signed m_in"), ("A_in_abs_sm", "A smoothed sigma 2, absolute target mass"),
          ("C3_in_abs", "feature-level s3 map, absolute target mass"), ("C3_in", "feature-level s3 map, signed m_in"))


def report():
    cells, S = load_all()
    if not cells:
        print("no csv"); return
    nc = len(cells); L = []
    R0 = S[S.arm == "rand"].copy(); S = S[S.arm.isin(ARMS)].copy()   # R0 = architectural floor (random-init real network), target class only
    ncase = {k: S[S.cell == k].case.nunique() for k in cells}
    L += ["# What did the model use to predict this pixel: Integrated Gradients from the texture-flattened twin (visual proof v16, FINDINGS 43; pre-registered: PREREG.md)", "",
          "Cells: " + ", ".join(f"{k} (n = {ncase[k]} cases)" for k in cells) + f". Arms noise / flat / real, folds 0-2, fold-mean maps. IG steps: {sorted(S.ig_steps.unique().tolist())} (midpoint rule). "
          "Baseline x_eq = texture-energy map flattened to the body mean, local patterns kept (v15 construction, TT_ITER = 2). Seeds = section-36 seeds, <= 12 per case "
          "(rare patterns first). Two explained quantities per seed: the TARGET-class (edema) log-odds for all three arms (suffix _tgt; PRIMARY, added before the full run "
          "because the pre-registered own-class reading compares different classes across arms at seeds where an arm fails) and each arm's OWN predicted class (pre-registered wording, secondary). "
          "Units: per-case median over the case's TARGET seeds, then median over cases; Wilcoxon (two-sided) over cases on the paired per-case medians. "
          "Ring = dilation 10 minus target in body; background = body minus dilation 10. Fractions use the whole-patch |map| total as denominator; m_in = signed target mass / |map| total. "
          "_sm = the same on A smoothed by a 3D Gaussian (sigma 2 vox, signed).", ""]
    # ---- seed-count check and completeness
    L += ["## Checks", "", "| cell | cases | seeds per case (median) | cases with seeds != min(12, csv) | target seeds / case (median) | completeness rel. err median, target class (all seeds) | "
          f"(|delta| > {FLOOR}) | 90th pct | own class (|delta| > {FLOOR}) | abs err median target class (nat) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in cells:
        s = S[S.cell == k]; c = s.groupby("case").first(); bad = int((c.n_seeds != np.minimum(MAX_SEEDS, c.n_seeds_csv)).sum())
        nt = s[s.arm == "real"].groupby("case").apply(lambda g: (g.region == "target").sum()).median(); bt = s[s.delta_tgt.abs() > FLOOR]; bo = s[s.delta.abs() > FLOOR]
        L.append(f"| {k} | {ncase[k]} | {c.n_seeds.median():.0f} | {bad} | {nt:.0f} | {s.ig_relerr_tgt.median():.3f} | {bt.ig_relerr_tgt.median():.3f} | {bt.ig_relerr_tgt.quantile(0.9):.3f} | "
                 f"{bo.ig_relerr.median():.3f} | {s.ig_abserr_tgt.median():.4f} |")
    L += ["", "Completeness by arm, target class (median rel. err, |delta| > %g): " % FLOOR + "; ".join(f"{a} {S[(S.arm == a) & (S.delta_tgt.abs() > FLOOR)].ig_relerr_tgt.median():.3f}" for a in ARMS), ""]
    # ---- tables
    rows = []; con = {}
    for sub, suf, sub_nm in SUBS:
        Sx = S[S.region == "target"]
        if sub == "own_agree":
            ok = Sx.groupby(["cell", "case", "seed"]).pred_in_T.min().rename("agree").reset_index(); Sx = Sx.merge(ok, on=["cell", "case", "seed"]); Sx = Sx[Sx.agree == 1]
        L += [f"## Per cell and arm ({sub_nm}): median over cases [IQR]; real - flat and real - noise paired medians (p)", ""]
        mets = MET if sub != "own_agree" else [m_ for m_ in MET if m_[0] in ("delta", "A_in", "A_in_abs", "A_in_sm", "A_in_abs_sm", "C3_in_abs")]
        for col, title in mets:
            cc = col + suf
            L += [f"### {cc}: {title}", "", "| cell | n | noise | flat | real | real - flat | p | real > flat (%) | real - noise | p | flat - noise | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for k in cells:
                pc = {a: Sx[(Sx.cell == k) & (Sx.arm == a)].groupby("case")[cc].median() for a in ARMS}
                rf, rn, fn = (pc["real"] - pc["flat"]).dropna(), (pc["real"] - pc["noise"]).dropna(), (pc["flat"] - pc["noise"]).dropna()
                q = lambda s: f"{s.median():+.3f} [{s.quantile(0.25):+.3f}, {s.quantile(0.75):+.3f}]" if len(s) else "n/a"  # noqa: E731
                L.append(f"| {k} | {len(rf)} | {q(pc['noise'])} | {q(pc['flat'])} | {q(pc['real'])} | {rf.median():+.3f} | {fp(wp(rf))} | {100 * (rf > 0).mean() if len(rf) else np.nan:.0f} | "
                         f"{rn.median():+.3f} | {fp(wp(rn))} | {fn.median():+.3f} | {fp(wp(fn))} |")
                con[(sub, col, k)] = (rf, rn, pc)
                rows.append(dict(subset=sub, cell=k, metric=cc, n=len(rf), **{f"{a}": pc[a].median() for a in ARMS}, **{f"{a}_q25": pc[a].quantile(0.25) for a in ARMS},
                                 **{f"{a}_q75": pc[a].quantile(0.75) for a in ARMS}, real_minus_flat=rf.median(), p_rf=wp(rf), real_minus_noise=rn.median(), p_rn=wp(rn), flat_minus_noise=fn.median(), p_fn=wp(fn)))
            L.append("")
    pd.DataFrame(rows).to_csv(OUT / "texig_cells.csv", index=False)
    # ---- architectural floor (rand pass)
    if len(R0):
        R0t = R0[R0.region == "target"]; St0 = S[S.region == "target"]; frows = []
        L += ["## Architectural floor: the real arm's network at random initialisation (reset_parameters, torch seeds 1000-1002 as 'folds'), TARGET-class IG on the same seeds", "",
              "Rand completeness (median rel. err, |delta| > %g): %.3f. Rand cases per cell: " % (FLOOR, R0[R0.delta_tgt.abs() > FLOOR].ig_relerr_tgt.median())
              + ", ".join(f"{k} {R0[R0.cell == k].case.nunique()}" for k in cells) + ".", "",
              "Per cell: median over cases (per-case median over target seeds) of each arm and rand; excess = arm - rand (paired over cases, Wilcoxon p); "
              "ratio = median over cases of the per-case arm / rand ratio (|A| target-mass fractions).", ""]
        for col in ("delta_tgt", "A_in_abs_tgt", "A_in_sm_tgt", "A_in_abs_sm_tgt", "C3_in_abs_tgt"):
            L += [f"### {col}", "", "| cell | n | rand | noise | flat | real | noise - rand (p) | flat - rand (p) | real - rand (p) |" + (" ratio noise / flat / real vs rand |" if col != "delta_tgt" and "_in_" in col and "abs" in col else ""),
                  "|---|---|---|---|---|---|---|---|---|" + ("---|" if col != "delta_tgt" and "abs" in col else "")]
            for k in cells:
                rr = R0t[R0t.cell == k].groupby("case")[col].median(); pc = {a: St0[(St0.cell == k) & (St0.arm == a)].groupby("case")[col].median() for a in ARMS}
                ex = {a: (pc[a] - rr).dropna() for a in ARMS}; ratio = {a: (pc[a] / rr.where(rr.abs() > 1e-9)).dropna() for a in ARMS}
                line = f"| {k} | {len(ex['real'])} | {rr.median():+.3f} | " + " | ".join(f"{pc[a].median():+.3f}" for a in ARMS) + " | " + " | ".join(f"{ex[a].median():+.3f} ({fp(wp(ex[a]))})" for a in ARMS) + " |"
                if col != "delta_tgt" and "abs" in col:
                    line += " " + " / ".join(f"{ratio[a].median():.2f}" for a in ARMS) + " |"
                L.append(line)
                frows.append(dict(cell=k, metric=col, n=len(ex["real"]), rand=rr.median(), **{a: pc[a].median() for a in ARMS}, **{f"{a}_minus_rand": ex[a].median() for a in ARMS},
                                  **{f"p_{a}_minus_rand": wp(ex[a]) for a in ARMS}, **{f"{a}_ratio_rand": ratio[a].median() for a in ARMS}))
            L.append("")
        pd.DataFrame(frows).to_csv(OUT / "texig_floor_cells.csv", index=False)
    # ---- fold agreement (cases run after per-fold maps were added)
    if "cons_frac_tgt" in S.columns and S.cons_frac_tgt.notna().any():
        Sc = S[(S.region == "target") & S.cons_frac_tgt.notna()]; Rc = R0[(R0.region == "target") & R0.cons_frac_tgt.notna()] if len(R0) and "cons_frac_tgt" in R0.columns else R0.iloc[0:0]
        L += ["## Fold agreement: Acons = sign(Abar) min_k max(A_k sign(Abar), 0), target class (zero wherever a fold disagrees in sign with the fold mean)", "",
              "cons_frac = sum |Acons| / sum |Abar| over the patch (1 = all three folds agree everywhere, at the mean's magnitude). Ac_* = target-mass metrics on Acons. "
              "Only cases run after per-fold maps were added: " + ", ".join(f"{k} {Sc[Sc.cell == k].case.nunique()}" for k in cells) + " cases (rand: "
              + ", ".join(f"{k} {Rc[Rc.cell == k].case.nunique()}" for k in cells) + "). Median over cases of the per-case median over target seeds; paired Wilcoxon over cases.", ""]
        crow = []; summ = {}
        for col in [c_ for c_ in ("cons_frac_tgt", "Ac_in_abs_tgt", "Ac_in_sm_tgt", "Ac_in_abs_sm_tgt", "Sc_in_mean_tgt") if c_ in S.columns and S[c_].notna().any()]:   # Sc_in_mean: rand pass + later runs only
            L += [f"### {col}", "", "| cell | n | rand | noise | flat | real | real - flat (p) | real - noise (p) | flat - rand (p) | noise - rand (p) |", "|---|---|---|---|---|---|---|---|---|---|"]
            for k in cells:
                pc = {a: Sc[(Sc.cell == k) & (Sc.arm == a)].groupby("case")[col].median() for a in ARMS}
                rr = Rc[Rc.cell == k].groupby("case")[col].median() if len(Rc) else pd.Series(dtype=float)
                d = {"rf": (pc["real"] - pc["flat"]).dropna(), "rn": (pc["real"] - pc["noise"]).dropna(), "fr": (pc["flat"] - rr).dropna(), "nr": (pc["noise"] - rr).dropna()}
                L.append(f"| {k} | {len(d['rf'])} | {rr.median() if len(rr) else np.nan:.3g} | " + " | ".join(f"{pc[a].median():.3g}" for a in ARMS) + " | "
                         + " | ".join(f"{d[x].median():+.3g} ({fp(wp(d[x]))})" for x in ("rf", "rn", "fr", "nr")) + " |")
                summ[(col, k)] = d
                crow.append(dict(cell=k, metric=col, n=len(d["rf"]), rand=rr.median() if len(rr) else np.nan, **{a: pc[a].median() for a in ARMS},
                                 **{f"d_{x}": d[x].median() for x in d}, **{f"p_{x}": wp(d[x]) for x in d}))
            L.append("")
        pd.DataFrame(crow).to_csv(OUT / "texig_consistency_cells.csv", index=False)
        lf = [k for k in cells if len(summ[("cons_frac_tgt", k)]["rf"]) and summ[("cons_frac_tgt", k)]["rf"].median() > 0 and wp(summ[("cons_frac_tgt", k)]["rf"]) < 0.01]
        ln = [k for k in cells if len(summ[("cons_frac_tgt", k)]["rn"]) and summ[("cons_frac_tgt", k)]["rn"].median() > 0 and wp(summ[("cons_frac_tgt", k)]["rn"]) < 0.01]
        L += [f"Is the flat / noise arms' evidence less fold-consistent than the real arm's? cons_frac real > flat (p < 0.01) in {len(lf)}/{nc} cells ({', '.join(lf) or 'none'}); "
              f"real > noise (p < 0.01) in {len(ln)}/{nc} cells ({', '.join(ln) or 'none'}).", ""]
    # ---- edge-mask control (x_em baseline: the twin keeps the fine band near coarse edges)
    fe = [f for k in cells for f in sorted((OUT / k).glob("*_texig_em.csv"))]
    if fe:
        E = pd.concat([pd.read_csv(f) for f in fe], ignore_index=True); Et = E[E.region == "target"]; Sx = S[S.region == "target"]; erow = []; verdict = {}
        L += ["## Edge-mask control: is the evidence made of boundary edits?", "",
              "x_em = x_eq with every gain blended toward 1 near coarse edges (g_em = 1 + (g - 1) e, e = exp(-(|grad G2(x)| / p90_body)^2)); same seeds, same "
              "target-class IG, three trained arms. Cases: the first %d (sorted) of each cell + the v7 example cases. Boundary edge-energy ratio (GT shell, "
              "|grad G2|, counterfactual / original), median [min-max] per cell: " % EM_N
              + "; ".join(f"{k} x_em {Et[Et.cell == k].groupby('case').edge_ratio_em.first().median():.3f} [{Et[Et.cell == k].groupby('case').edge_ratio_em.first().min():.3f}-"
                          f"{Et[Et.cell == k].groupby('case').edge_ratio_em.first().max():.3f}], x_eq {Et[Et.cell == k].groupby('case').edge_ratio_eq.first().median():.3f}" for k in cells) + ".", "",
              "Per cell and arm, on the edge-mask cases: median over cases (per-case median over target seeds) under x_eq and x_em, paired em - eq (Wilcoxon), "
              "median relative change |em - eq| / |eq| (per case), and the median per-seed correlation of the smoothed (G2) fold-mean A slices eq vs em (map similarity).", ""]
        corr = {}
        for k in cells:
            for case in sorted(Et[Et.cell == k].case.unique()):
                f0, f1 = OUT / k / f"{case}_texig_maps.npz", OUT / k / f"{case}_texig_em_maps.npz"
                if not (f0.exists() and f1.exists()):
                    continue
                M0, M1 = np.load(f0, allow_pickle=True), np.load(f1, allow_pickle=True)
                if not np.array_equal(M0["seed_id"], M1["seed_id"]):
                    continue
                regs = M0["region"].astype(str); body = M0["ip"].astype(np.float64) != 0
                for a in ARMS:
                    cs = []
                    for i in range(len(regs)):
                        if regs[i] != "target":
                            continue
                        a0 = ndi.gaussian_filter(M0[f"{a}_A"][i].astype(np.float64) * M0[f"{a}_A_scale"][i], 2.0)
                        a1 = ndi.gaussian_filter(M1[f"{a}_A"][i].astype(np.float64) * M1[f"{a}_A_scale"][i], 2.0); bb = body[i]
                        if bb.sum() > 10 and a0[bb].std() > 0 and a1[bb].std() > 0:
                            cs.append(float(np.corrcoef(a0[bb], a1[bb])[0, 1]))
                    if cs:
                        corr.setdefault((k, a), []).append(float(np.median(cs)))
        # fine-scale boundary metrics (coordinator: the sigma-2 edge ratio is dominated by the coarse band) and the original texture contrast
        Ec = Et.groupby(["cell", "case"]).first().reset_index()
        def mr(s):
            s = s.dropna(); return f"{s.median():.3f} [{s.min():.3f}-{s.max():.3f}]" if len(s) else "n/a"
        if "fine_corr_em" in Ec.columns:
            L += ["### Boundary preservation at the fine scale (GT shell = dilate 2 minus erode 2; twin vs original image; median [min-max] over the edge-mask cases)", "",
                  "| cell | n | sigma-2 edge ratio eq | em | sigma-1 grad-magnitude corr eq | em | sigma-1 grad-magnitude ratio eq | em | SSIM of sigma-1 high-pass eq | em |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for k in cells:
                c_ = Ec[Ec.cell == k]
                L.append(f"| {k} | {len(c_)} | {mr(c_.edge_ratio_eq)} | {mr(c_.edge_ratio_em)} | {mr(c_.fine_corr_eq)} | {mr(c_.fine_corr_em)} | {mr(c_.fine_ratio_eq)} | {mr(c_.fine_ratio_em)} | "
                         f"{mr(c_.fine_ssim_eq)} | {mr(c_.fine_ssim_em)} |")
            L.append("")
        ctr = {}
        for k in cells:   # original target / body fine-energy ratio: all cases from the v15 texthink qc rows when available, else the edge-mask cases
            fq = sorted((tt.OUT / k).glob("*_texthink.csv")) if (tt.OUT / k).exists() else []
            vals = pd.concat([pd.read_csv(f_) for f_ in fq], ignore_index=True) if fq else pd.DataFrame()
            vals = vals[vals.kind == "qc"].ctr_orig if len(vals) and "kind" in vals else (Ec[Ec.cell == k].ctr_orig if "ctr_orig" in Ec.columns else pd.Series(dtype=float))
            ctr[k] = (vals.dropna(), "all cases (v15 qc)" if fq else "edge-mask cases")
        L += ["Original image, target / body fine-energy ratio (G2(fine^2), texture contrast), median [min-max]: "
              + "; ".join(f"{k} {mr(v)} (n = {len(v)}, {src})" for k, (v, src) in ctr.items()) + ".", ""]
        low = [k for k, (v, _) in ctr.items() if len(v) and v.median() < 1]
        if low:
            L += [f"In {', '.join(low)} this ratio is < 1: there, orange inside the lesion means the lesion's lower fine-scale energy relative to its surroundings "
                  "supported the edema call.", ""]
        for col_eq, col_em, nm in (("delta_tgt", "delta_em", "signed total f(x) - f(twin) (nat)"), ("A_in_sm_tgt", "A_in_sm_em", "signed smoothed in-target evidence (m_in on G2(A))"),
                                   ("A_in_abs_tgt", "A_in_abs_em", "|A| target fraction")):
            L += [f"### {nm}: {col_eq} vs {col_em}", "", "| cell | arm | n | eq | em | em - eq (p) | median rel. change | map corr eq vs em |", "|---|---|---|---|---|---|---|---|"]
            for k in cells:
                for a in ARMS:
                    pe = Et[(Et.cell == k) & (Et.arm == a)].groupby("case")[col_em].median(); pq = Sx[(Sx.cell == k) & (Sx.arm == a)].groupby("case")[col_eq].median().reindex(pe.index)
                    dd = (pe - pq).dropna(); rel = ((pe - pq).abs() / pq.abs().where(pq.abs() > 1e-9)).dropna(); cc = corr.get((k, a), [])
                    L.append(f"| {k} | {a} | {len(dd)} | {pq.median():+.3f} | {pe.median():+.3f} | {dd.median():+.3f} ({fp(wp(dd))}) | {rel.median():.2f} | {np.median(cc) if cc else np.nan:.2f} |")
                    erow.append(dict(cell=k, arm=a, metric=col_eq, n=len(dd), eq=pq.median(), em=pe.median(), d=dd.median(), p=wp(dd), rel=rel.median(), corr=np.median(cc) if cc else np.nan))
            L.append("")
        L += ["### real - flat and real - noise of the signed total under each twin (paired over the edge-mask cases)", "", "| cell | n | real - flat eq (p) | real - flat em (p) | real - noise eq (p) | real - noise em (p) |", "|---|---|---|---|---|---|"]
        for k in cells:
            pe = {a: Et[(Et.cell == k) & (Et.arm == a)].groupby("case").delta_em.median() for a in ARMS}; pq = {a: Sx[(Sx.cell == k) & (Sx.arm == a)].groupby("case").delta_tgt.median().reindex(pe["real"].index) for a in ARMS}
            r = {nm: (d_["real"] - d_[o]).dropna() for nm, d_ in (("eq", pq), ("em", pe)) for o in ("flat",)}; rn = {nm: (d_["real"] - d_["noise"]).dropna() for nm, d_ in (("eq", pq), ("em", pe))}
            L.append(f"| {k} | {len(r['em'])} | {r['eq'].median():+.3f} ({fp(wp(r['eq']))}) | {r['em'].median():+.3f} ({fp(wp(r['em']))}) | {rn['eq'].median():+.3f} ({fp(wp(rn['eq']))}) | {rn['em'].median():+.3f} ({fp(wp(rn['em']))}) |")
            verdict[k] = (np.sign(r["eq"].median()) == np.sign(r["em"].median())) and (np.sign(rn["eq"].median()) == np.sign(rn["em"].median()))
        pd.DataFrame(erow).to_csv(OUT / "texig_edgemask_cells.csv", index=False)
        cmed = np.nanmedian([np.median(v) for v in corr.values()]) if corr else np.nan
        same = (sum(verdict.values()) == len(verdict)) and np.isfinite(cmed) and cmed >= 0.8
        L += ["", f"Rule (fixed before reading): the maps 'do not change' if the median eq-vs-em map correlation over cells and arms is >= 0.8 AND the signs of the real - flat and "
              f"real - noise signed-total contrasts are the same under both twins in every cell. Median map correlation {cmed:.2f}; sign kept in "
              f"{sum(verdict.values())}/{len(verdict)} cells. **The maps {'do not change' if same else 'do change'} when edges are excluded.**", ""]
    # ---- verdicts

    def sig_pos(s):
        return len(s) > 0 and s.median() > 0 and wp(s) < 0.01
    V = []
    for sub, suf, sub_nm in SUBS:
        if (sub, "delta", cells[0]) not in con:
            continue
        tag = f" [{sub_nm}]"
        nrf = sum(sig_pos(con[(sub, "delta", k)][0]) for k in cells); nrn = sum(sig_pos(con[(sub, "delta", k)][1]) for k in cells)
        v = "SUPPORTED" if (nrf >= 3 and nrn == nc) else "NOT SUPPORTED"
        V.append(f"- P1{tag}: {v}: signed total real > flat (median paired difference > 0, p < 0.01) in {nrf}/{nc} cells (need >= 3), real > noise in {nrn}/{nc} (need {nc}/{nc}). "
                 + "; ".join(f"{k} r-f {con[(sub, 'delta', k)][0].median():+.3f} (p {fp(wp(con[(sub, 'delta', k)][0]))}), r-n {con[(sub, 'delta', k)][1].median():+.3f} (p {fp(wp(con[(sub, 'delta', k)][1]))})" for k in cells) + ".")
        for col, cdesc in P2COLS:
            if (sub, col, cells[0]) not in con:
                continue
            nrf = sum(sig_pos(con[(sub, col, k)][0]) for k in cells)
            low = sum(1 for k in cells if con[(sub, col, k)][2]["noise"].median() < min(con[(sub, col, k)][2]["flat"].median(), con[(sub, col, k)][2]["real"].median()))
            v = "SUPPORTED" if (nrf >= 3 and low >= 3) else ("PARTLY SUPPORTED (real > flat only)" if nrf >= 3 else "NOT SUPPORTED")
            V.append(f"  - P2 on {col}{suf} ({cdesc}{'; the pre-registered reading' if col == 'A_in' else ''}): {v}: real > flat (p < 0.01) in {nrf}/{nc} cells (need >= 3); noise lowest in {low}/{nc}. "
                     + "; ".join(f"{k} n/f/r {con[(sub, col, k)][2]['noise'].median():+.3f}/{con[(sub, col, k)][2]['flat'].median():+.3f}/{con[(sub, col, k)][2]['real'].median():+.3f} (r-f p {fp(wp(con[(sub, col, k)][0]))})" for k in cells) + ".")
    bt = S[S.delta_tgt.abs() > FLOOR]; ce = bt.ig_relerr_tgt.median(); bo = S[S.delta.abs() > FLOOR]
    V.append(f"- Completeness: median relative error, target class {ce:.3f} on seeds with |delta| > {FLOOR} ({S.ig_relerr_tgt.median():.3f} on all seeds); own class {bo.ig_relerr.median():.3f}; "
             f"pre-registered gate < 0.05: {'MET' if ce < 0.05 else 'NOT MET (raise M)'}.")
    V.append("- P3 (readability): judged by eye on texig_examples_<cell>.png (target-class maps), not tested here.")
    L += ["## Verdicts (operational rules: P1 per cell = median paired difference of per-case medians > 0 and two-sided Wilcoxon p < 0.01; P2 the same on the mass column, "
          "'noise lowest' = noise median below both flat and real medians, needed in >= 3 cells)", ""] + V + [""]
    (OUT / "texig.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
    # ---- summary figure (primary, target class)
    pan = (("delta", "signed total f(x) - f(x_eq), target class (nat)"), ("A_in_abs", "|A| target fraction"), ("A_in_sm", "A smoothed: signed m_in"), ("C3_in_abs", "C(s3): |C| target fraction"))
    fig, axs = plt.subplots(len(pan), nc, figsize=(3.4 * nc, 2.9 * len(pan)), squeeze=False)
    for c, k in enumerate(cells):
        for r, (col, yl) in enumerate(pan):
            ax = axs[r, c]; rf, rn, pc = con[("tgt", col, k)]
            for i, a in enumerate(ARMS):
                s = pc[a].dropna(); ax.bar(i, s.median(), 0.7, color=PAL[a], yerr=[[s.median() - s.quantile(0.25)], [s.quantile(0.75) - s.median()]], capsize=3, error_kw=dict(lw=0.8))
            ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, fontsize=8); ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{k} (n = {len(rf)})\nr-f {rf.median():+.3f} p {fp(wp(rf))}; r-n {rn.median():+.3f} p {fp(wp(rn))}", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(yl, fontsize=8)
    fig.suptitle("Integrated Gradients from the texture-flattened twin: target seeds, TARGET-class log-odds for all arms, per-case median over seeds, median and IQR over cases", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / "texig_summary.png", dpi=150); fig.savefig(OUT / "texig_summary.pdf"); plt.close(fig)
    print("TEXIG_REPORT_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- figure
def figure():
    cells, S = load_all(); W = 48
    for k in cells:
        q = S[(S.cell == k) & (S.arm == "real")].groupby("case").n_target.first().sort_values()
        q = q[[(OUT / k / f"{c}_texig_maps.npz").exists() for c in q.index]]
        if not len(q):
            continue
        picks = []
        for pc in (50, 75):
            tgt = np.percentile(q.values, pc)
            for c in (q - tgt).abs().sort_values().index:
                if c not in picks:
                    picks.append(c); break
        cols = ["image + GT + seed", "fine-energy map G2(fine^2)"] + [f"A_v {a}" for a in ("real", "flat", "noise")] + [f"A_v smoothed s2 {a}" for a in ("real", "flat", "noise")] + [f"C_v(s3) {a}" for a in ("real", "flat", "noise")]
        fig, axs = plt.subplots(2 * len(picks), len(cols), figsize=(1.95 * len(cols), 2.05 * 2 * len(picks) + 0.7), squeeze=False)
        for pi, case in enumerate(picks):
            M = np.load(OUT / k / f"{case}_texig_maps.npz", allow_pickle=True); pats = M["pattern"].astype(str); regs = M["region"].astype(str)
            cand = [i for i in range(len(pats)) if regs[i] == "target" and pats[i] == "CCC"] or [i for i in range(len(pats)) if regs[i] == "target" and pats[i] == "FFC"] \
                or [i for i in range(len(pats)) if regs[i] == "target"]
            if not cand:
                continue
            i = cand[0]; v = M["seeds"][i]; A = {a: M[f"{a}_A"][i].astype(np.float64) * M[f"{a}_A_scale"][i] for a in ARMS}; C = {a: M[f"{a}_C3"][i].astype(np.float64) * M[f"{a}_C3_scale"][i] for a in ARMS}
            Asm = {a: (M[f"{a}_Asm"][i].astype(np.float64) * M[f"{a}_Asm_scale"][i]) if f"{a}_Asm" in M.files else np.zeros_like(A[a]) for a in ARMS}
            vs = np.percentile(np.abs(np.stack(list(Asm.values()))), 99) or 1e-12
            va = np.percentile(np.abs(np.stack(list(A.values()))), 99) or 1e-12; vc = np.percentile(np.abs(np.stack(list(C.values()))), 99) or 1e-12
            im, em, gt = M["ip"][i].astype(np.float64), M["emap"][i].astype(np.float64), M["gt"][i].astype(float)
            sd = S[(S.cell == k) & (S.case == case) & (S.seed == int(M["seed_id"][i]))].set_index("arm")
            y0, x0 = max(0, int(v[1]) - W), max(0, int(v[2]) - W); y1, x1 = min(im.shape[0], y0 + 2 * W), min(im.shape[1], x0 + 2 * W); y0, x0 = max(0, y1 - 2 * W), max(0, x1 - 2 * W)
            for rr, crop in ((0, True), (1, False)):
                r = 2 * pi + rr; cy = slice(y0, y1) if crop else slice(None); cx = slice(x0, x1) if crop else slice(None); oy, ox = (y0, x0) if crop else (0, 0)
                for c, title in enumerate(cols):
                    ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
                    if c == 0:
                        a_ = im[cy, cx]; ax.imshow(a_.T, cmap="gray", origin="lower", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5), interpolation="nearest")
                        ax.set_ylabel(f"{case}{' (96x96 crop)' if crop else ' (full slice)'}\nz={int(v[0])}, n_target={int(q[case])}, {pats[i]}", fontsize=5.5)
                    elif c == 1:
                        ax.imshow(em[cy, cx].T, cmap="magma", origin="lower", vmin=0, vmax=np.percentile(em, 99.5), interpolation="nearest")
                    else:
                        a = cols[c].split()[-1]; mp, vl, mk = (A[a], va, "A_in") if c < 5 else ((Asm[a], vs, "A_in_sm") if c < 8 else (C[a], vc, "C3_in"))
                        ax.imshow(mp[cy, cx].T, cmap="RdBu_r", origin="lower", vmin=-vl, vmax=vl, interpolation="nearest")
                        if rr == 0 and a in sd.index:
                            s_ = sd.loc[a]; title = f"{title}\ndelta {s_.delta_tgt:+.2f}, m_in {s_.get(mk + '_tgt', np.nan):+.2f}"
                    ax.contour(gt[cy, cx].T, [0.5], colors="lime", linewidths=0.5)
                    ax.plot(int(v[1]) - oy, int(v[2]) - ox, "x", color="yellow", ms=5, mew=1.2)
                    if rr == 0:
                        ax.set_title(title, fontsize=5.5)
        fig.suptitle(f"{k}: Integrated Gradients from the texture-flattened twin, one target seed (yellow x) per case, TARGET-class (edema) log-odds for all arms; A_v = (x - x_eq) * mean gradient "
                     f"(RdBu_r, shared symmetric scale = 99th pct |A| over the three arms); A_v smoothed = 3D Gaussian sigma 2 of A_v (texture-evidence density, own shared scale); C_v(s3) feature-level (own shared scale); lime = GT edema", fontsize=6.5)
        fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / f"texig_examples_{k}_signed.png", dpi=150); plt.close(fig)
    print("TEXIG_FIGURE_DONE", flush=True)


def _picks(S, k):
    q = S[(S.cell == k) & (S.arm == "real")].groupby("case").n_target.first().sort_values()
    q = q[[(OUT / k / f"{c}_texig_maps.npz").exists() for c in q.index]]
    picks = []
    if len(q):
        for pc in (50, 75):
            tgt = np.percentile(q.values, pc)
            for c in (q - tgt).abs().sort_values().index:
                if c not in picks:
                    picks.append(c); break
    return q, picks


def _seed_index(M, dep=None):
    """Example pixel: a target seed, preferring depth >= 3 vox from the lesion boundary (seeds csv `depth`, when `dep` = {seed id: depth} is given),
    CCC first, then FFC; fallbacks: CCC / FFC at any depth, then the deepest target seed."""
    pats = M["pattern"].astype(str); regs = M["region"].astype(str); ids = M["seed_id"]
    dp = np.array([dep.get(int(s), np.nan) if dep else np.nan for s in ids], float)
    tgt = [i for i in range(len(pats)) if regs[i] == "target"]
    for deep in ((True, False) if dep else (False,)):
        for p in ("CCC", "FFC"):
            cand = [i for i in tgt if pats[i] == p and (not deep or dp[i] >= 3)]
            if cand:
                return max(cand, key=lambda i: dp[i] if np.isfinite(dp[i]) else -1) if deep else cand[0]
    return max(tgt, key=lambda i: dp[i] if np.isfinite(dp[i]) else -1) if tgt else None


def _amap(M, arm, i, cons):
    """Target-class A slice of one seed: fold mean (cons False) or fold-consistent Acons = sign(Abar) min_k max(A_k sign(Abar), 0) from the per-fold slices."""
    if not cons:
        return M[f"{arm}_A"][i].astype(np.float64) * M[f"{arm}_A_scale"][i]
    Ak = np.stack([M[f"{arm}_A_f{kf}"][i].astype(np.float64) * M[f"{arm}_A_f{kf}_scale"][i] for kf in range(len(FOLDS))]); sg = np.sign(Ak.mean(0))
    return sg * np.min(np.maximum(Ak * sg[None], 0), axis=0)


def figure_texture(floor=True, cons=False):
    """Main example figure (Paul's review, v6; from the saved npz only): 6 columns, rows = whole (patch) slice and lesion zoom (GT bounding box + 16 vox);
    2 cases per cell = the cases whose per-case real - flat target-class delta (median over target seeds) is closest to the cell median and to its 75th
    percentile; target-class maps. Outside the body black; body windowed at its 1-99th percentile.
    t = clip(|hp| / (G3(hp^2)^0.5 + eps) * exp(-(|grad G2(img)| / p90 in body)^2), 0, 1.5), hp = img - G1(img); w_arm = clip(G2(|A_arm|) / p99 over the three arms, 0, 1).
    col 1 image + GT (lime) + pixel (cyan x); cols 2-4 the arm's own colour (real red, flat blue, noise green) with o_arm = clip(1.5 t w_arm^0.5, 0, 1),
    composite grey (1 - 0.6 o) + colour o; col 5 'who uses this texture' = grey (1 - 0.7 max o) + 0.9 [o_real, o_noise, o_flat] (white = all three);
    col 6 'model-specific texture' = the same from o_spec = clip(1.5 t max(w_arm - min over arms, 0)^0.5, 0, 1).
    floor=True (main texig_examples_<cell>.png): w_arm replaced by clip(w_arm - w_rand, 0, 1), w_rand = G2(|A_rand|) / the same p99 (architectural floor, --rand
    pass); falls back to no floor (stated in the caption) when the rand maps are missing. floor=False: texig_examples_<cell>_nofloor.png.
    cons=True (fold agreement, the main figure): A = Acons from the per-fold slices (cases with per-fold maps only); cons=False with floor:
    texig_examples_<cell>_foldmean.png."""
    cells, S = load_all(); P = 2.7; MARG = 16
    COL = {"real": np.array([1.0, 0.25, 0.1]), "flat": np.array([0.2, 0.5, 1.0]), "noise": np.array([0.2, 0.9, 0.3])}
    LEG = [("real", (1.0, 0.25, 0.1)), ("flat", (0.2, 0.5, 1.0)), ("noise", (0.2, 0.9, 0.3)), ("all three", (1, 1, 1)), ("real+flat", (1, 0.2, 1)), ("real+noise", (1, 1, 0.2)), ("flat+noise", (0.2, 1, 1))]
    for k in cells:
        St = S[(S.cell == k) & (S.region == "target")]; SDk = read_seeds(k)
        pc = St.groupby(["case", "arm"]).delta_tgt.median().unstack()
        pc = pc[[(OUT / k / f"{c}_texig_maps.npz").exists() for c in pc.index]]
        if cons:   # only cases that carry per-fold maps
            pc = pc[[("real_A_f0" in np.load(OUT / k / f"{c}_texig_maps.npz", allow_pickle=True).files) for c in pc.index]]
        if not len(pc) or "real" not in pc or "flat" not in pc:
            continue
        drf = (pc["real"] - pc["flat"]).dropna(); med = float(drf.median()); picks = []
        for qv in (0.5, 0.75):
            tgt = float(drf.quantile(qv))
            for c in (drf - tgt).abs().sort_values().index:
                if c not in picks:
                    picks.append(c); break
        fig, axs = plt.subplots(2 * len(picks), 6, figsize=(P * 6, P * 2 * len(picks) + 0.8), squeeze=False)
        floor_used = []
        for pi, case in enumerate(picks):
            M = np.load(OUT / k / f"{case}_texig_maps.npz", allow_pickle=True)
            dep = {int(r_.seed): float(r_.depth) for r_ in SDk[SDk.case == case].itertuples()} if SDk is not None else None; i = _seed_index(M, dep)
            if i is None:
                continue
            v = M["seeds"][i]; im = M["ip"][i].astype(np.float64); gt = M["gt"][i].astype(bool); body = im != 0
            if not body.any():
                body = np.ones_like(im, bool)
            lo, hi = np.percentile(im[body], 1), np.percentile(im[body], 99)
            g0 = np.clip((im - lo) / (hi - lo + 1e-12), 0, 1); g0[~body] = 0; G3 = np.stack([g0] * 3, -1)
            hp = im - ndi.gaussian_filter(im, 1.0); lrms = np.sqrt(ndi.gaussian_filter(hp ** 2, 3.0)); eps = 0.05 * np.median(lrms[body]) + 1e-12
            gr = ndi.gaussian_gradient_magnitude(im, 2.0); edge = np.exp(-(gr / (np.percentile(gr[body], 90) + 1e-12)) ** 2)
            t = np.clip(np.abs(hp) / (lrms + eps) * edge, 0, 1.5); t[~body] = 0
            A = {a: _amap(M, a, i, cons) for a in ARMS}
            D = {a: ndi.gaussian_filter(np.abs(A[a]), 2.0) for a in ARMS}; dmax = np.percentile(np.stack(list(D.values())), 99) or 1e-30
            w = {a: np.clip(D[a] / dmax, 0, 1) for a in ARMS}; fr = OUT / k / f"{case}_texig_rand_maps.npz"; applied = False
            if floor and fr.exists():
                MR = np.load(fr, allow_pickle=True)
                if int(MR["seed_id"][i]) == int(M["seed_id"][i]):
                    if not cons or "rand_A_f0" in MR.files:
                        wrnd = np.clip(ndi.gaussian_filter(np.abs(_amap(MR, "rand", i, cons)), 2.0) / dmax, 0, 1)
                    else:
                        wrnd = None
                    if wrnd is not None:
                        w = {a: np.clip(w[a] - wrnd, 0, 1) for a in ARMS}; applied = True
            floor_used.append(applied)
            wmin = np.minimum(np.minimum(w["real"], w["flat"]), w["noise"])
            o = {a: np.clip(1.5 * t * w[a] ** 0.5, 0, 1) for a in ARMS}; osp = {a: np.clip(1.5 * t * np.maximum(w[a] - wmin, 0) ** 0.5, 0, 1) for a in ARMS}

            def one(oa, col):
                return np.clip(G3 * (1 - 0.6 * oa[..., None]) + col[None, None] * oa[..., None], 0, 1)

            def mix(od):
                rgb = np.stack([od["real"], od["noise"], od["flat"]], -1); mx = rgb.max(-1, keepdims=True)
                return np.clip(G3 * (1 - 0.7 * mx) + 0.9 * rgb, 0, 1)
            sd = St[(St.case == case) & (St.seed == int(M["seed_id"][i]))].set_index("arm")
            dl = {a: (f" ({sd.loc[a].delta_tgt:+.1f} nat)" if a in sd.index else "") for a in ARMS}
            panels = [("image", G3, False)] + [(f"texture used by {a}{dl[a]}", one(o[a], COL[a]), False) for a in ARMS_ORDER] + \
                     [("who uses this texture", mix(o), True), ("model-specific texture", mix(osp), True)]
            if gt.any():
                yy, xx = np.where(gt); y0, y1 = max(0, yy.min() - MARG), min(im.shape[0], yy.max() + MARG + 1); x0, x1 = max(0, xx.min() - MARG), min(im.shape[1], xx.max() + MARG + 1)
            else:
                y0, y1, x0, x1 = 0, im.shape[0], 0, im.shape[1]
            for rr, zoom in ((0, False), (1, True)):
                r = 2 * pi + rr; cy = slice(y0, y1) if zoom else slice(None); cx = slice(x0, x1) if zoom else slice(None); oy, ox = (y0, x0) if zoom else (0, 0)
                for c, (title, arr, leg) in enumerate(panels):
                    ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
                    ax.imshow(np.transpose(arr[cy, cx], (1, 0, 2)), origin="lower", interpolation="nearest")
                    ax.contour(gt[cy, cx].T.astype(float), [0.5], colors="lime", linewidths=0.6)
                    ax.plot(int(v[1]) - oy, int(v[2]) - ox, "x", color="cyan", ms=7, mew=1.5)
                    if leg and rr == 0:
                        for j, (nm, cc) in enumerate(LEG):
                            ax.text(0.02, 0.97 - 0.075 * j, nm, color=cc, fontsize=5.5, transform=ax.transAxes, va="top", ha="left",
                                    bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=0.6))
                    if rr == 0:
                        ax.set_title(title, fontsize=8.5)
                    if c == 0:
                        ax.set_ylabel(f"{'typical' if pi == 0 else '75th pct'} case ({case}),\nreal - flat {drf[case]:+.1f} nat, {'lesion zoom' if zoom else 'whole slice'}", fontsize=7.5)
        fig.suptitle(f"{k} - For the marked pixel: the image's fine texture each model drew on to call it edema. Red real fill, blue flat fill, green noise fill; "
                     f"white = used by all three. (Cell median real - flat = {med:+.2f} nat, n = {len(drf)})"
                     + ("\nFold-consistent evidence only (kept where all three folds agree in sign; magnitude = the smallest fold)." if cons else "\nFold-mean evidence.")
                     + ("\nArchitectural floor subtracted: evidence of the same network at random initialisation removed (w - w_rand)." if floor and floor_used and all(floor_used)
                        else ("\nFloor NOT subtracted (random-init pass not available for these cases)." if floor else "\nNo floor subtraction.")), fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(OUT / f"texig_examples_{k}{('_unsigned' if cons else '_foldmean') if floor else '_nofloor'}.png", dpi=130); plt.close(fig)
    print("TEXIG_FIGURE_TEXTURE_DONE", flush=True)


def figure_signed_texture(em=False):
    """Main example figure v7 (Paul: the separation is in the SIGN): 6 columns, rows = whole (patch) slice and lesion zoom, 2 cases per cell (closest to the
    median and the 75th percentile of the per-case real - flat target-class delta, among cases with per-fold maps). S_arm = G2(Acons_arm) (signed, fold-
    consistent, 2D on the seed slice); floor: S_arm zeroed where |S_arm| <= G2(|Acons_rand|) when the rand per-fold maps exist. s99 = 99th pct of |S| over the
    three arms. Grain t as in v4-v6. cols 2-4: orange where S > 0 (toward edema), cyan where S < 0 (away), intensity clip(1.5 t (|S| / s99)^0.5, 0, 1);
    col 5 / 6: the same on S_real - S_flat / S_real - S_noise, in MAGENTA (real more supported) and GREEN (the other model more); pixel = yellow x.
    em=True: the same figure from the edge-mask control maps (<case>_texig_em_maps.npz, x_em baseline), same example cases, no floor (the rand pass used x_eq);
    written to texig_examples_<cell>_edgemask.png."""
    cells, S = load_all(); P = 2.7; MARG = 16; OR = np.array([1.0, 0.55, 0.0]); CY = np.array([0.1, 0.8, 1.0]); MG = np.array([1.0, 0.2, 0.9]); GN = np.array([0.2, 1.0, 0.3])
    for k in cells:
        St = S[(S.cell == k) & (S.region == "target")]; SDk = read_seeds(k); EMS = "_em" if em else ""
        if em:   # per-seed deltas from the edge-mask csv (columns _em), renamed to the _tgt names used below
            fe = sorted((OUT / k).glob("*_texig_em.csv"))
            if not fe:
                continue
            Ek = pd.concat([pd.read_csv(f_) for f_ in fe], ignore_index=True); Ek = Ek[Ek.region == "target"].rename(columns={"delta_em": "delta_dl"})
        pc = St.groupby(["case", "arm"]).delta_tgt.median().unstack()
        pc = pc[[(OUT / k / f"{c}_texig_maps.npz").exists() for c in pc.index]]
        pc = pc[[("real_A_f0" in np.load(OUT / k / f"{c}_texig_maps.npz", allow_pickle=True).files) for c in pc.index]]
        if not len(pc) or "real" not in pc or "flat" not in pc:
            continue
        drf = (pc["real"] - pc["flat"]).dropna(); med = float(drf.median()); picks = []
        for qv in (0.5, 0.75):
            tq = float(drf.quantile(qv))
            for c in (drf - tq).abs().sort_values().index:
                if c not in picks:
                    picks.append(c); break
        fig, axs = plt.subplots(2 * len(picks), 6, figsize=(P * 6, P * 2 * len(picks) + 0.9), squeeze=False); floor_used = []
        for pi, case in enumerate(picks):
            if em and not (OUT / k / f"{case}_texig_em_maps.npz").exists():
                continue
            M = np.load(OUT / k / f"{case}_texig{EMS}_maps.npz", allow_pickle=True)
            dep = {int(r_.seed): float(r_.depth) for r_ in SDk[SDk.case == case].itertuples()} if SDk is not None else None; i = _seed_index(M, dep)
            if i is None:
                continue
            v = M["seeds"][i]; im = M["ip"][i].astype(np.float64); gt = M["gt"][i].astype(bool); body = im != 0
            if not body.any():
                body = np.ones_like(im, bool)
            lo, hi = np.percentile(im[body], 1), np.percentile(im[body], 99)
            g0 = np.clip((im - lo) / (hi - lo + 1e-12), 0, 1); g0[~body] = 0; G3 = np.stack([g0] * 3, -1)
            hp = im - ndi.gaussian_filter(im, 1.0); lrms = np.sqrt(ndi.gaussian_filter(hp ** 2, 3.0)); eps = 0.05 * np.median(lrms[body]) + 1e-12
            gr = ndi.gaussian_gradient_magnitude(im, 2.0); edge = np.exp(-(gr / (np.percentile(gr[body], 90) + 1e-12)) ** 2)
            t = np.clip(np.abs(hp) / (lrms + eps) * edge, 0, 1.5); t[~body] = 0
            Sm = {a: ndi.gaussian_filter(_amap(M, a, i, True), 2.0) for a in ARMS}
            fr = OUT / k / f"{case}_texig_rand_maps.npz"; applied = False
            if fr.exists() and not em:
                MR = np.load(fr, allow_pickle=True)
                if "rand_A_f0" in MR.files and int(MR["seed_id"][i]) == int(M["seed_id"][i]):
                    fl = ndi.gaussian_filter(np.abs(_amap(MR, "rand", i, True)), 2.0)
                    Sm = {a: np.where(np.abs(Sm[a]) > fl, Sm[a], 0.0) for a in ARMS}; applied = True
            floor_used.append(applied)
            s99 = np.percentile(np.abs(np.stack(list(Sm.values()))), 99) or 1e-30

            def paint(Mp, pos=OR, neg=CY):
                o = np.clip(1.5 * t * (np.abs(Mp) / s99) ** 0.5, 0, 1); o[~body] = 0; col = np.where((Mp > 0)[..., None], pos[None, None], neg[None, None])
                return np.clip(G3 * (1 - 0.6 * o[..., None]) + col * o[..., None], 0, 1)
            sd = St[(St.case == case) & (St.seed == int(M["seed_id"][i]))].set_index("arm")
            if em:
                sd = Ek[(Ek.case == case) & (Ek.seed == int(M["seed_id"][i]))].set_index("arm"); dl = {a: (f" ({sd.loc[a].delta_dl:+.1f} nat)" if a in sd.index else "") for a in ARMS}
            else:
                dl = {a: (f" ({sd.loc[a].delta_tgt:+.1f} nat)" if a in sd.index else "") for a in ARMS}
            panels = [("image", G3)] + [(f"for/against edema:\n{a}{dl[a]}", paint(Sm[a])) for a in ARMS_ORDER] + \
                     [("real \u2212 flat:\nmagenta = real more supported,\ngreen = flat more", paint(Sm["real"] - Sm["flat"], MG, GN)),
                      ("real \u2212 noise:\nmagenta = real more supported,\ngreen = noise more", paint(Sm["real"] - Sm["noise"], MG, GN))]
            if gt.any():
                yy, xx = np.where(gt); y0, y1 = max(0, yy.min() - MARG), min(im.shape[0], yy.max() + MARG + 1); x0, x1 = max(0, xx.min() - MARG), min(im.shape[1], xx.max() + MARG + 1)
            else:
                y0, y1, x0, x1 = 0, im.shape[0], 0, im.shape[1]
            for rr, zoom in ((0, False), (1, True)):
                r = 2 * pi + rr; cy = slice(y0, y1) if zoom else slice(None); cx = slice(x0, x1) if zoom else slice(None); oy, ox = (y0, x0) if zoom else (0, 0)
                for c, (title, arr) in enumerate(panels):
                    ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
                    ax.imshow(np.transpose(arr[cy, cx], (1, 0, 2)), origin="lower", interpolation="nearest")
                    ax.contour(gt[cy, cx].T.astype(float), [0.5], colors="lime", linewidths=0.6)
                    ax.plot(int(v[1]) - oy, int(v[2]) - ox, "x", color="yellow", ms=7, mew=1.5)
                    if rr == 0:
                        ax.set_title(title, fontsize=7.5 if "\u2212" in title else 8.5)
                    if c == 0:
                        ax.set_ylabel(f"{'typical' if pi == 0 else '75th pct'} case ({case}),\nreal - flat {drf[case]:+.1f} nat, {'lesion zoom' if zoom else 'whole slice'}", fontsize=7.5)
        fl_txt = ("EDGE-MASK CONTROL (twin keeps the fine band near coarse edges), no floor" if em else
                  ("floor-subtracted" if floor_used and all(floor_used) else "floor NOT subtracted (random-init maps not available)"))
        fig.suptitle(f"{k} - For the marked pixel: fine texture that pushes each model TOWARD calling it edema (orange) or AWAY from it (cyan); fold-consistent, {fl_txt};\n"
                     f"last two panels = where the real-fill model gets more support from texture than flat / noise (magenta) or less (green).\n(Cell median real - flat = {med:+.2f} nat, n = {len(drf)}; lime = GT edema, yellow x = the pixel)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / f"texig_examples_{k}{'_edgemask' if em else ''}.png", dpi=130); plt.close(fig)
    print("TEXIG_FIGURE_SIGNED_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--figure", action="store_true"); ap.add_argument("--rand", action="store_true", help="architectural floor pass")
    ap.add_argument("--edgemask", action="store_true", help="edge-mask control pass (x_em baseline)")
    args = ap.parse_args()
    if args.report or args.figure:
        if args.report:
            report()
        if args.figure:
            figure(); figure_texture(False, False); figure_texture(True, False); figure_texture(True, True); figure_signed_texture(); figure_signed_texture(em=True)
        return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.cells:
        run_cell(key, dev, args.case_slice, rand=args.rand, em=args.edgemask)


if __name__ == "__main__":
    main()
