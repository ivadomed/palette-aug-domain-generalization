#!/usr/bin/env python
"""Visual proof v18 (FINDINGS 45, pre-registered in outputs/plots/chanabl/PREREG.md): texture-energy-selective CHANNEL ABLATION (dissection then lesion)
inside the network, BraTS cells, arms noise (rung 4, lblvor) / flat (4.5) / real (5) + "init" (the real network re-initialised, torch seed 1000 + k), folds 0-2.

Descriptors (image-only, mask-free; FINDINGS 31, as v17): I = G2(x), E = |grad G2(x)|, Tf = sqrt(G2((x - G1 x)^2)); r = log Tf residualised per case on
[1, I, log E] over body voxels; body-masked area average on each stage's grid.
--fit   (GPU): first N_FIT (20) sorted cases. Per arm x fold x stage (s2, s3, s4 skips; d1, d2, d3 decoder stages): per fit case Spearman (per-case ranks) of
        every channel with r / I / E over the stage-body voxels; score = |mean over fit cases|; top-k = texture- / intensity- / edge-selective channels.
        Per-channel fit-set mean (the ablation value) and variance (pooled stage-body voxels). Texture units = channels whose score exceeds the p99 of a
        within-case spatial-shuffle null (20 shuffles) and of a case-swap null (activation of case i vs r of case i - L, L = 1..3). Variance-decile
        matched random channels (20 draws per k; nearest-decile fallback) and plain random channels (20 draws per k). -> OUT/<cell>/fit/chanabl_fit_<arm>_f<k>.npz,
        OUT/<cell>/fit/units.csv
run     (GPU): evaluation cases (sorted index >= N_FIT) in --case-slice. Per arm x fold x stage x k: MEAN-ablation (selected channels := fit mean at every
        voxel; skips: the skip path into the decoder only; decoder stages: the stage output), decoder re-entered from that point (batched over variants).
        J = edema (class 2) log-odds; E = J_nat - J_ablated; fold-mean; scalars on target / ring 1-10 vox / far background, Dice before / after. Variant
        "all" (every channel of the stage) = plumbing check. init: d3, k = 16 only. -> OUT/<cell>/<case>_chanabl.csv (written last),
        <case>_chanabl_maps.npz (one axial slice, float16, patient data: never commit)
--report (CPU): chanabl.md, chanabl_cells.csv, chanabl_depth_k.csv, chanabl_units.csv, chanabl_controls.csv, chanabl_summary.png/pdf.
--figure (CPU): chanabl_examples_<cell>.png (patient slices: never commit).
Env: VP_CHANABL_OUT, CA_FOLDS ("0 1 2"), CA_NFIT (20), CA_DRAWS (20), CA_SHUF (20), CA_KS ("4 8 16 32"), CA_BATCH (8).
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_v2 as v2  # noqa: E402
import visual_proof_erf as e7  # noqa: E402
from visual_proof_switch import load_case, activations, models_for, wp, ARMS, PAL  # noqa: E402

OUT = Path(os.environ.get("VP_CHANABL_OUT") or (THIS.parent / "outputs/plots/chanabl")); OUT.mkdir(parents=True, exist_ok=True)
CELLS4 = ("glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w")
FOLDS = tuple(int(v) for v in os.environ.get("CA_FOLDS", "0 1 2").split())
N_FIT = int(os.environ.get("CA_NFIT", 20)); N_DRAWS = int(os.environ.get("CA_DRAWS", 20)); N_SHUF = int(os.environ.get("CA_SHUF", 20))
KS = tuple(int(v) for v in os.environ.get("CA_KS", "4 8 16 32").split()); HEADK = 16; BATCH = int(os.environ.get("CA_BATCH", 8))
STAGES = ("s2", "s3", "s4", "d1", "d2", "d3"); HEAD = "d3"; LAGS = (1, 2, 3)
RING, ONE_NAT, EDEMA = 10, 1.0, 2
CONC = ("tex", "int", "edge"); DICEV = ("tex", "int", "edge", "mr0", "pr0", "all")
ARMS4 = tuple(ARMS) + ("init",); PAL4 = dict(PAL, init="#7f7f7f")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fp(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.1g}"


def hseed(*a):
    return int(hashlib.md5("|".join(map(str, a)).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------------------------------------------------------------- descriptors
def concepts(ip, bp):
    """FINDINGS 31 descriptors at full patch resolution: residualised log Tf (per-case OLS on [1, I, log E] over body voxels), I and E."""
    x = ip.astype(np.float64); b = bp.astype(bool)
    hp = x - ndi.gaussian_filter(x, 1.0); I = ndi.gaussian_filter(x, 2.0); E = ndi.gaussian_gradient_magnitude(x, 2.0); Tf = np.sqrt(ndi.gaussian_filter(hp ** 2, 2.0))
    X = np.c_[np.ones(int(b.sum())), I[b], np.log(E[b] + 1e-6)]; y = np.log(Tf[b] + 1e-6); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.zeros_like(x); r[b] = y - X @ beta
    return r, I, E


def stage_concepts(r, I, E, b, shape, dev):
    """-> stage-body mask (bool, flat), full-grid stage maps, and the values of tex / int / edge on the stage-body voxels."""
    import torch
    import torch.nn.functional as Fn
    t = torch.from_numpy(np.stack([r * b, I * b, E * b, b.astype(np.float64)]))[None].to(dev)
    a = Fn.interpolate(t, size=shape, mode="area")[0]; bm = (a[3] >= 0.5).reshape(-1); den = a[3].clamp_min(1e-6)
    full = {c: (a[i] / den).reshape(-1) for i, c in enumerate(CONC)}
    return dict(bm=bm, full=full, **{c: full[c][bm] for c in CONC})


def zrank(v):
    """Rank along the last axis (ties by order), z-scored: Pearson of two zrank vectors = mean of their product = Spearman."""
    import torch
    n = v.shape[-1]; idx = v.argsort(-1); r = torch.empty(v.shape, dtype=torch.float64, device=v.device)
    r.scatter_(-1, idx, torch.arange(n, dtype=torch.float64, device=v.device).expand(v.shape).contiguous())
    return (r - (n - 1) / 2.0) / np.sqrt((n * n - 1) / 12.0)


def matched_draw(var, sel, rng):
    """One variance-matched random set: for each selected channel a random NON-selected channel of the same variance decile (without replacement within
    the draw); exhausted decile -> nearest decile with a free channel (lower before higher). Returns (indices, number of fallbacks)."""
    C = len(var); dec = np.argsort(np.argsort(var, kind="stable"), kind="stable") * 10 // C
    free = np.ones(C, bool); free[sel] = False; out, fb = [], 0
    for c in sel:
        for dd in [0] + [s * j for j in range(1, 10) for s in (-1, 1)]:
            cand = np.flatnonzero(free & (dec == dec[c] + dd))
            if len(cand):
                break
        fb += int(dd != 0); p = int(rng.choice(cand)); free[p] = False; out.append(p)
    return np.array(out, dtype=np.int64), fb


# ----------------------------------------------------------------------------------------------------------------------------------- networks
def load_arms(key, dev):
    """noise / flat / real predictors + 'init' = the real network re-initialised by reset_parameters (torch seed 1000 + k per 'fold'), deep supervision off."""
    import torch
    md = models_for(key); prs = {a: v2.load_predictor(d, dev) for a, d in md.items()}
    pi = v2.load_predictor(md["real"], dev); net = pi.network; sds = []
    for k in range(3):
        torch.manual_seed(1000 + k)
        for m_ in net.modules():
            if hasattr(m_, "reset_parameters"):
                m_.reset_parameters()
        sds.append({n_: v_.detach().clone() for n_, v_ in net.state_dict().items()})
    pi.list_of_parameters = sds; prs["init"] = pi
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    return prs


def set_fold(pr, k):
    net = pr.network; net.load_state_dict(pr.list_of_parameters[k]); net.eval(); return net


def acts(net, x):
    import torch
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        return activations(net, x)


def decode_from(net, skips, ds, loc, Hb):
    """Re-enter net's decoder with the batch Hb (B, C, ...) in place of the activation at loc (v17 semantics). Skip s_j: only the tensor concatenated at
    decoder stage 4 - j is replaced; decoder stage d_s: the decoder continues from stage s + 1. Other inputs (batch 1) broadcast."""
    import torch
    dec = net.decoder; B = Hb.shape[0]; ns = len(dec.stages); over = {}
    if loc[0] == "s":
        j = int(loc[1:]); s0 = len(skips) - 2 - j; lres = skips[-1] if s0 == 0 else ds[s0 - 1]; over[s0] = Hb
    else:
        s0 = int(loc[1:]) + 1; lres = Hb
    for s in range(s0, ns):
        h = dec.transpconvs[s](lres); sk = over.get(s, skips[-(s + 2)])
        if h.shape[0] != B:
            h = h.expand(B, *h.shape[1:])
        if sk.shape[0] != B:
            sk = sk.expand(B, *sk.shape[1:])
        lres = dec.stages[s](torch.cat((h, sk), 1))
    return dec.seg_layers[-1](lres)


def ablate(h, mu, M):
    """h (1, C, ...), mu (C,), M (B, C) in {0, 1} -> (B, C, ...): channels with M = 1 replaced by their fit mean at every voxel."""
    tail = (1,) * (h.dim() - 2); m = M.view(M.shape[0], M.shape[1], *tail)
    return h * (1 - m) + mu.view(1, -1, *tail) * m


# ----------------------------------------------------------------------------------------------------------------------------------- fit
def fit_cell(key, dev):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key / "fit"; od.mkdir(parents=True, exist_ok=True)
    prs = load_arms(key, dev); pr0 = prs["flat"]; cases = sorted(cell["cases"])[:N_FIT]; acc = {}; hist = {}
    log(f"[{key}] fit on {len(cases)} cases, folds {FOLDS}, ks {KS}, draws {N_DRAWS}, shuffles {N_SHUF}")
    for ci, case in enumerate(cases):
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); r, I, E = concepts(ip, bp); b = bp.astype(bool)
        x = torch.from_numpy(ip)[None, None].to(dev); conc = {}
        for arm, pr in prs.items():
            for kf in FOLDS:
                net = set_fold(pr, kf); A, _ = acts(net, x)
                for L in STAGES:
                    h = A[L][0]; shp = tuple(h.shape[1:]); C = h.shape[0]
                    if shp not in conc:
                        c_ = stage_concepts(r, I, E, b, shp, dev); n_ = int(c_["bm"].sum()); g = torch.Generator().manual_seed(hseed(key, case, shp))
                        c_["z"] = {cc: zrank(c_[cc]) for cc in CONC}
                        c_["zperm"] = torch.stack([c_["z"]["tex"][torch.randperm(n_, generator=g).to(dev)] for _ in range(N_SHUF)])
                        conc[shp] = c_
                    c = conc[shp]; Hf = h.reshape(C, -1).double(); H = Hf[:, c["bm"]]; n = H.shape[1]
                    st = acc.setdefault((arm, kf, L), dict(n=0, s=0.0, ss=0.0, nc=0, shuf=0.0, rho={cc: 0.0 for cc in CONC}, cs={lag: [0.0, 0] for lag in LAGS}))
                    st["n"] += n; st["s"] = st["s"] + H.sum(1); st["ss"] = st["ss"] + (H ** 2).sum(1); st["nc"] += 1
                    Za = zrank(H)
                    for cc in CONC:
                        st["rho"][cc] = st["rho"][cc] + Za @ c["z"][cc] / n
                    st["shuf"] = st["shuf"] + Za @ c["zperm"].T / n
                    del Za
                    for lag in LAGS:
                        if ci - lag >= 0:
                            pb, pt = hist[shp][ci - lag]; m = c["bm"] & pb; nm = int(m.sum())
                            if nm > 100:
                                st["cs"][lag][0] = st["cs"][lag][0] + zrank(Hf[:, m]) @ zrank(pt[m]) / nm; st["cs"][lag][1] += 1
                del A; torch.cuda.empty_cache()
        for shp, c in conc.items():
            hist.setdefault(shp, {})[ci] = (c["bm"], c["full"]["tex"])
        log(f"[{key}] fit {case}: {time.time() - t0:.0f} s")
    urows = []
    for arm in prs:
        for kf in FOLDS:
            out = {}
            for L in STAGES:
                st = acc[(arm, kf, L)]; n = st["n"]; nc = st["nc"]; mu = (st["s"] / n); var = (st["ss"] / n - mu ** 2).cpu().numpy(); C = len(var)
                rho = {cc: (st["rho"][cc] / nc).cpu().numpy() for cc in CONC}; score = {cc: np.abs(rho[cc]) for cc in CONC}
                order = {cc: np.argsort(-score[cc], kind="stable") for cc in CONC}
                nul = np.abs((st["shuf"] / nc).cpu().numpy()); p99s = float(np.percentile(nul, 99)); us = int((score["tex"] > p99s).sum())
                csl = [(st["cs"][lag][0] / st["cs"][lag][1]).cpu().numpy() for lag in LAGS if st["cs"][lag][1] > 0]
                p99c = float(np.percentile(np.abs(np.stack(csl)), 99)) if csl else np.nan; ucs = int((score["tex"] > p99c).sum()) if csl else -1
                out.update({f"{L}_mu": mu.float().cpu().numpy(), f"{L}_var": var, f"{L}_n": int(n),
                            **{f"{L}_rho_{cc}": rho[cc] for cc in CONC}, **{f"{L}_order_{cc}": order[cc] for cc in CONC}})
                dec = np.argsort(np.argsort(var, kind="stable"), kind="stable") * 10 // C
                urow = dict(cell=key, arm=arm, fold=kf, loc=L, C=C, n_vox=int(n), units_shuffle=us, p99_shuffle=p99s, units_caseswap=ucs, p99_caseswap=p99c,
                            score_max=float(score["tex"].max()), score_k16=float(score["tex"][order["tex"][min(HEADK, C) - 1]]), score_median=float(np.median(score["tex"])),
                            frac_pos_top16=float((rho["tex"][order["tex"][:HEADK]] > 0).mean()), int_score_max=float(score["int"].max()), edge_score_max=float(score["edge"].max()),
                            jac_tex_int16=len(set(order["tex"][:HEADK]) & set(order["int"][:HEADK])) / len(set(order["tex"][:HEADK]) | set(order["int"][:HEADK])),
                            jac_tex_edge16=len(set(order["tex"][:HEADK]) & set(order["edge"][:HEADK])) / len(set(order["tex"][:HEADK]) | set(order["edge"][:HEADK])),
                            var_decile_top16=float(dec[order["tex"][:HEADK]].mean()))
                for kk in KS:
                    ke = min(kk, C // 2); sel = order["tex"][:ke]; rng = np.random.default_rng(hseed(key, arm, kf, L, kk)); mr, fb = [], 0
                    for d in range(N_DRAWS):
                        idx, f_ = matched_draw(var, sel, rng); mr.append(idx); fb += f_
                    free = np.setdiff1d(np.arange(C), sel); prr = np.stack([rng.choice(free, ke, replace=False) for _ in range(N_DRAWS)])
                    out[f"{L}_mr_k{kk}"] = np.stack(mr); out[f"{L}_pr_k{kk}"] = prr; urow[f"fallback_k{kk}"] = fb / (N_DRAWS * ke)
                urows.append(urow)
                log(f"[{key}] {arm} f{kf} {L}: C {C}, n {int(n)}, tex score max {score['tex'].max():.3f} k16 {urow['score_k16']:.3f}, units shuffle {us} (p99 {p99s:.4f}) "
                    f"caseswap {ucs} (p99 {p99c:.3f}), jac tex-int {urow['jac_tex_int16']:.2f}, fallback k16 {urow.get('fallback_k16', np.nan):.2f}")
            np.savez_compressed(od / f"chanabl_fit_{arm}_f{kf}.npz", **out)
    pd.DataFrame(urows).to_csv(od / "units.csv", index=False)
    print("CHANABL_FIT_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- run
def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; fd = od / "fit"; c0 = int(T[0])
    prs = load_arms(key, dev); pr0 = prs["flat"]
    allc = sorted(cell["cases"]); idx = range(len(allc))[slice(*[int(v) for v in sl.split(":")])] if sl else range(len(allc))
    cases = [allc[i] for i in idx if i >= N_FIT]
    F = {}
    for arm in ARMS4:
        for kf in FOLDS:
            Z = np.load(fd / f"chanabl_fit_{arm}_f{kf}.npz"); d = {}
            for L in STAGES:
                d[L] = dict(mu=torch.from_numpy(Z[f"{L}_mu"]).float().to(dev), C=len(Z[f"{L}_mu"]), order={cc: Z[f"{L}_order_{cc}"] for cc in CONC},
                            mr={kk: Z[f"{L}_mr_k{kk}"] for kk in KS}, pr={kk: Z[f"{L}_pr_k{kk}"] for kk in KS})
            F[(arm, kf)] = d
    log(f"[{key}] run: {len(cases)} evaluation cases, folds {FOLDS}, ks {KS}, draws {N_DRAWS}, batch {BATCH}")
    for case in cases:
        f_out = od / f"{case}_chanabl.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); b = bp.astype(bool)
        tg = gp == EDEMA; dil = ndi.binary_dilation(tg, iterations=RING); ring = dil & ~tg & b; bg = b & ~dil
        z = int(np.argmax(tg.sum((1, 2)))) if tg.any() else ip.shape[0] // 2
        x = torch.from_numpy(ip)[None, None].to(dev)
        R = {nm: torch.from_numpy(m_).to(dev) for nm, m_ in (("tg", tg), ("ring", ring), ("bg", bg))}
        base = dict(cell=key, task=cell["task"], case=case, n_target=int(tg.sum()), n_ring=int(ring.sum()), n_bg=int(bg.sum()), z=z)
        rows = []; maps = dict(z=np.int32(z), ip=ip[z], gt=tg[z]); nf = float(len(FOLDS)); sc = {}

        def rmean(Mp, nm):
            m_ = R[nm]; return float(Mp[m_].mean()) if bool(m_.any()) else np.nan

        def dice(P):
            pr_ = (P.argmax(0) == c0); g_ = R["tg"]; return float(2 * (pr_ & g_).sum() / (pr_.sum() + g_.sum() + 1e-9))
        for arm, pr in prs.items():
            stg = STAGES if arm != "init" else (HEAD,); kks = KS if arm != "init" else (HEADK,)
            accJ = 0; accP = 0; accR = {}; accPv = {}; accEm = {}
            for kf in FOLDS:
                net = set_fold(pr, kf); A, lg = acts(net, x); J = e7.log_odds(lg.float(), c0)[0]; P = torch.softmax(lg.float(), 1)[0]
                accJ = accJ + J; accP = accP + P; skips = [A[f"s{j}"] for j in range(6)]; ds = [A[f"d{s}"] for s in range(5)]
                for L in stg:
                    h = A[L]; f = F[(arm, kf)][L]; C = f["C"]; var_ = [("all", 0, np.arange(C))]
                    for kk in kks:
                        ke = min(kk, C // 2); var_ += [(cc, kk, f["order"][cc][:ke]) for cc in CONC]
                        var_ += [(f"mr{d_}", kk, f["mr"][kk][d_]) for d_ in range(N_DRAWS)] + [(f"pr{d_}", kk, f["pr"][kk][d_]) for d_ in range(N_DRAWS)]
                    with torch.no_grad(), torch.autocast("cuda", enabled=False):
                        if arm == "real" and kf == FOLDS[0]:
                            sc[L] = float((decode_from(net, skips, ds, L, h) - lg).abs().max())
                        for i0 in range(0, len(var_), BATCH):
                            ch = var_[i0:i0 + BATCH]; M = torch.zeros(len(ch), C, device=dev, dtype=h.dtype)
                            for jj, (_, _, ix) in enumerate(ch):
                                M[jj, torch.as_tensor(ix, device=dev)] = 1
                            lgb = decode_from(net, skips, ds, L, ablate(h, f["mu"], M)); Eb = J[None] - e7.log_odds(lgb.float(), c0)
                            for jj, (nm, kk, _) in enumerate(ch):
                                kv = (L, kk, nm); Ev = Eb[jj]
                                accR[kv] = accR.get(kv, 0) + np.array([rmean(Ev, "tg"), rmean(Ev, "ring"), rmean(Ev, "bg")]) / nf
                                if nm in DICEV:
                                    accPv[kv] = accPv.get(kv, 0) + torch.softmax(lgb[jj:jj + 1].float(), 1)[0]
                                if nm == "tex":
                                    accEm[kv] = accEm.get(kv, 0) + Ev / nf
                            del lgb, Eb, M
                del A, lg, skips, ds; torch.cuda.empty_cache()
            Jm = accJ / nf; Pm = accP / nf; dn = dice(Pm)
            rows.append(dict(base, kind="nat", arm=arm, var="nat", loc="", k=0, J_target=rmean(Jm, "tg"), J_ring=rmean(Jm, "ring"), J_bg=rmean(Jm, "bg"), dice=dn,
                             **({f"selfcheck_{L}": v_ for L, v_ in sc.items()} if arm == "real" else {})))
            maps[f"{arm}_J"] = Jm[z].cpu().numpy(); maps[f"{arm}_pred"] = (Pm.argmax(0) == c0)[z].cpu().numpy()
            for L in stg:
                e_ = accR[(L, 0, "all")]
                rows.append(dict(base, kind="E", arm=arm, var="all", loc=L, k=0, E_target=e_[0], E_ring=e_[1], E_bg=e_[2], dice_nat=dn, dice_after=dice(accPv[(L, 0, "all")] / nf)))
                for kk in kks:
                    for cc in CONC:
                        e_ = accR[(L, kk, cc)]; row = dict(base, kind="E", arm=arm, var=cc, loc=L, k=kk, E_target=e_[0], E_ring=e_[1], E_bg=e_[2], dice_nat=dn,
                                                           dice_after=dice(accPv[(L, kk, cc)] / nf))
                        if cc == "tex":
                            Em = accEm[(L, kk, "tex")]; row["frac_gt1"] = float((Em[R["tg"]] > ONE_NAT).float().mean()) if bool(R["tg"].any()) else np.nan
                        rows.append(row)
                    for pre, vn in (("mr", "mmean"), ("pr", "pmean")):
                        dr = np.array([accR[(L, kk, f"{pre}{d_}")] for d_ in range(N_DRAWS)]); dt = dr[:, 0]
                        rows.append(dict(base, kind="E", arm=arm, var=vn, loc=L, k=kk, E_target=float(dt.mean()), E_ring=float(np.nanmean(dr[:, 1])), E_bg=float(np.nanmean(dr[:, 2])),
                                         dice_nat=dn, dice_after=dice(accPv[(L, kk, f"{pre}0")] / nf), E_target_p95=float(np.percentile(dt, 95)), E_target_p025=float(np.percentile(dt, 2.5)),
                                         E_target_p975=float(np.percentile(dt, 97.5)), E_ring_p95=float(np.percentile(dr[:, 1], 95)), **{f"E_target_d{i}": float(dt[i]) for i in range(N_DRAWS)}))
            kv = (HEAD, HEADK)
            maps[f"{arm}_E_tex"] = accEm[kv + ("tex",)][z].cpu().numpy()
            for nm in ("tex", "mr0", "int"):
                maps[f"{arm}_pred_{nm}"] = (accPv[kv + (nm,)] / nf).argmax(0)[z].eq(c0).cpu().numpy()
            del accPv, accEm, accJ, accP; torch.cuda.empty_cache()
        np.savez_compressed(od / f"{case}_chanabl_maps.npz", **{k_: (v_.astype(np.float16) if isinstance(v_, np.ndarray) and v_.dtype.kind == "f" else v_) for k_, v_ in maps.items()})
        pd.DataFrame(rows).to_csv(f_out, index=False)
        S_ = pd.DataFrame(rows); s_ = S_[(S_.kind == "E") & (S_["loc"] == HEAD) & (S_.k == HEADK)]
        g_ = lambda a, v: s_[(s_.arm == a) & (s_["var"] == v)].E_target.iloc[0]   # noqa: E731
        log(f"[{key}] {case}: {HEAD} k{HEADK} target E tex / matched mean / int: " + "; ".join(f"{a} {g_(a, 'tex'):+.2f} / {g_(a, 'mmean'):+.2f} / {g_(a, 'int'):+.2f}" for a in ARMS4)
            + f"; selfcheck max {max(sc.values()):.1e}; {time.time() - t0:.0f} s")
    print("CHANABL_RUN_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
def load_all():
    cells = [k for k in CELLS4 if (OUT / k).exists() and list((OUT / k).glob("*_chanabl.csv"))]
    S = pd.concat([pd.read_csv(f) for k in cells for f in sorted((OUT / k).glob("*_chanabl.csv"))], ignore_index=True) if cells else pd.DataFrame()
    U = pd.concat([pd.read_csv(OUT / k / "fit" / "units.csv") for k in cells if (OUT / k / "fit" / "units.csv").exists()], ignore_index=True) if cells else pd.DataFrame()
    return cells, S, U


def piv(S, c, arm, var, loc, kk, col="E_target"):
    d = S[(S.kind == "E") & (S.cell == c) & (S.arm == arm) & (S["var"] == var) & (S["loc"] == loc) & (S.k == kk)]
    return d.set_index("case")[col]


def Dser(S, c, arm, loc, kk, var="tex"):
    """per-case selected-minus-matched-random effect."""
    return (piv(S, c, arm, var, loc, kk) - piv(S, c, arm, "mmean", loc, kk)).dropna()


def med_iqr(s):
    s = s.dropna()
    return f"{s.median():+.3f} [{s.quantile(0.25):+.3f}, {s.quantile(0.75):+.3f}]" if len(s) else "n/a"


def report():
    cells, S, U = load_all()
    if not cells:
        print("no csv"); return
    nc = len(cells); ncase = {c: S[(S.cell == c) & (S.kind == "nat")].case.nunique() for c in cells}; KH = HEADK
    L = ["# Texture-energy-selective channel ablation (FINDINGS 45, pre-registered: PREREG.md)", "",
         "Cells: " + ", ".join(f"{c} (n = {ncase[c]} evaluation cases)" for c in cells) + f". Selection on the first {N_FIT} sorted cases of each cell, effects on the others. "
         f"Arms noise / flat / real (folds {', '.join(map(str, FOLDS))}) + init (the real network re-initialised, its own selection; {HEAD}, k = {KH} only). Channels ranked by "
         "|mean over fit cases of the per-case Spearman| with the FINDINGS-31 residualised log fine-texture energy (texture), local intensity I (intensity, positive control) or "
         "gradient magnitude E (edge); top-k MEAN-ablated (fit-set mean at every voxel; skips s2-s4: skip path only; d1-d3: stage output) and the decoder re-entered. "
         "E = J(x) - J(ablated), J = edema log-odds, fold-mean, target = GT edema voxels; positive = the ablated channels pushed the edema belief UP. "
         f"Primary null: {N_DRAWS} variance-decile-matched random channel sets (D = E_selected - mean E_matched); secondary: {N_DRAWS} plain random sets. "
         "Wilcoxon two-sided over cases.", ""]
    # ---- plumbing gate
    nat = S[S.kind == "nat"]; scc = [c_ for c_ in nat.columns if c_.startswith("selfcheck_")]; scmax = float(np.nanmax(nat[scc].to_numpy())) if scc else np.nan
    L += ["## Plumbing gate", "", f"(a) decoder re-entry with no channel ablated vs natural logits (real arm, fold {FOLDS[0]}, every case and stage): max |diff| = {scmax:.2e}.", "",
          f"(b) ablating ALL channels of the stage at {HEAD} (median over cases):", "", "| cell | arm | E target | Dice natural | Dice after | gate |", "|---|---|---|---|---|---|"]
    gb = True
    for c in cells:
        for arm in ARMS:
            e_ = piv(S, c, arm, "all", HEAD, 0); dn = piv(S, c, arm, "all", HEAD, 0, "dice_nat"); da = piv(S, c, arm, "all", HEAD, 0, "dice_after")
            ok = abs(e_.median()) > 1 and (dn.median() - da.median()) > 0.2; gb &= ok
            L.append(f"| {c} | {arm} | {e_.median():+.2f} | {dn.median():.3f} | {da.median():.3f} | {'pass' if ok else 'FAIL'} |")
    L += ["", f"Gate (a) {'pass' if scmax < 1e-3 else 'FAIL'}; gate (b) {'pass' if gb else 'FAIL'}" + ("" if (gb and scmax < 1e-3) else " -> read the effects below with care"), ""]
    # ---- units
    urows = []
    if len(U):
        L += ["## Texture-energy-selective units per stage (fit cases; mean over folds)", "",
              "Counts: channels whose texture score exceeds the p99 of the within-case spatial-shuffle null (anti-conservative, ignores spatial autocorrelation) and of the "
              "case-swap null (activation of case i vs texture of case i - 1..3). Jaccard = overlap of the top-16 texture set with the top-16 intensity / edge sets.", "",
              "| cell | arm | stage | C | units shuffle | units case-swap | score max | score #16 | % top-16 positive | Jaccard tex-int | tex-edge | fallback k16 / k32 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        G = U.groupby(["cell", "arm", "loc"]).mean(numeric_only=True)
        for c in cells:
            for arm in ARMS4:
                for loc in STAGES:
                    if (c, arm, loc) not in G.index:
                        continue
                    q = G.loc[(c, arm, loc)]
                    L.append(f"| {c} | {arm} | {loc} | {q.C:.0f} | {q.units_shuffle:.1f} | {q.units_caseswap:.1f} | {q.score_max:.3f} | {q.score_k16:.3f} | {100 * q.frac_pos_top16:.0f} | "
                             f"{q.jac_tex_int16:.2f} | {q.jac_tex_edge16:.2f} | {q.get('fallback_k16', np.nan):.2f} / {q.get('fallback_k32', np.nan):.2f} |")
                    urows.append(dict(cell=c, arm=arm, loc=loc, **q.to_dict()))
        pd.DataFrame(urows).to_csv(OUT / "chanabl_units.csv", index=False); L.append("")
    # ---- headline
    rows = []
    L += [f"## Headline: {HEAD}, k = {KH}: target-mean effect per arm, median [IQR] over cases", "",
          "| cell | arm | E texture | E matched (mean of draws) | E matched p95 | D = tex - matched | % cases tex > p95 | D intensity | D edge | E plain random | Dice natural | after tex | after matched (draw 0) | after int | after edge |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for arm in ARMS4:
            Et, Em, Ep, Ei, Ee, Er = (piv(S, c, arm, v, HEAD, KH) for v in ("tex", "mmean", "mmean", "int", "edge", "pmean")); Ep = piv(S, c, arm, "mmean", HEAD, KH, "E_target_p95")
            D, Di, De = Dser(S, c, arm, HEAD, KH), Dser(S, c, arm, HEAD, KH, "int"), Dser(S, c, arm, HEAD, KH, "edge")
            dn, dt_, dm, di, de = (piv(S, c, arm, v, HEAD, KH, col) for v, col in (("tex", "dice_nat"), ("tex", "dice_after"), ("mmean", "dice_after"), ("int", "dice_after"), ("edge", "dice_after")))
            if not len(Et):
                continue
            L.append(f"| {c} | {arm} | {med_iqr(Et)} | {med_iqr(Em)} | {Ep.median():+.3f} | {med_iqr(D)} | {100 * ((Et - Ep).dropna() > 0).mean():.0f} | {Di.median():+.3f} | {De.median():+.3f} | "
                     f"{Er.median():+.3f} | {dn.median():.3f} | {dt_.median():.3f} | {dm.median():.3f} | {di.median():.3f} | {de.median():.3f} |")
            rows.append(dict(cell=c, arm=arm, loc=HEAD, k=KH, n=int(Et.notna().sum()), E_tex=Et.median(), E_tex_q25=Et.quantile(0.25), E_tex_q75=Et.quantile(0.75), E_matched_mean=Em.median(),
                             E_matched_p95=Ep.median(), D_tex=D.median(), D_tex_q25=D.quantile(0.25), D_tex_q75=D.quantile(0.75), p_D_tex=wp(D), D_int=Di.median(), D_edge=De.median(),
                             E_int=Ei.median(), E_edge=Ee.median(), E_plain=Er.median(), frac_tex_gt_p95=((Et - Ep).dropna() > 0).mean(), dice_nat=dn.median(), dice_tex=dt_.median(),
                             dice_matched0=dm.median(), dice_int=di.median(), dice_edge=de.median(), dice_drop_int=(dn - di).median(), dice_drop_tex=(dn - dt_).median()))
    pd.DataFrame(rows).to_csv(OUT / "chanabl_cells.csv", index=False); L.append("")
    # ---- arm contrasts on D
    crow = []; con = {}

    def contrast_line(c, loc, kk):
        Ds = {a: Dser(S, c, a, loc, kk) for a in ARMS}
        drf, drn, dfn = (Ds["real"] - Ds["flat"]).dropna(), (Ds["real"] - Ds["noise"]).dropna(), (Ds["flat"] - Ds["noise"]).dropna(); con[(c, loc, kk)] = (drf, drn)
        for nm, s_ in (("real-flat", drf), ("real-noise", drn), ("flat-noise", dfn)):
            crow.append(dict(cell=c, loc=loc, k=kk, arm="", contrast=f"D {nm}", median=s_.median(), p=wp(s_), frac_pos=(s_ > 0).mean()))
        return f"| {c} | {loc} | {kk} | {drf.median():+.3f} | {fp(wp(drf))} | {100 * (drf > 0).mean():.0f} | {drn.median():+.3f} | {fp(wp(drn))} | {dfn.median():+.3f} | {fp(wp(dfn))} |"
    L += [f"## Arm contrasts on D = E_tex - mean E_matched (paired over cases): every stage at k = {KH}, every k at {HEAD}", "",
          "| cell | stage | k | real - flat median | p | real > flat % | real - noise median | p | flat - noise median | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for loc in STAGES:
            L.append(contrast_line(c, loc, KH))
        for kk in KS:
            if kk != KH:
                L.append(contrast_line(c, HEAD, kk))
    L.append("")
    L += [f"Random-init floor ({HEAD}, k = {KH}): " + "; ".join(
        f"{c} init D {Dser(S, c, 'init', HEAD, KH).median():+.3f} (E tex {piv(S, c, 'init', 'tex', HEAD, KH).median():+.3f}), real - init {(Dser(S, c, 'real', HEAD, KH) - Dser(S, c, 'init', HEAD, KH)).dropna().median():+.3f} "
        f"p {fp(wp((Dser(S, c, 'real', HEAD, KH) - Dser(S, c, 'init', HEAD, KH)).dropna()))}" for c in cells), ""]
    for c in cells:
        s_ = (Dser(S, c, "real", HEAD, KH) - Dser(S, c, "init", HEAD, KH)).dropna(); crow.append(dict(cell=c, loc=HEAD, k=KH, arm="", contrast="D real-init", median=s_.median(), p=wp(s_), frac_pos=(s_ > 0).mean()))
    # ---- controls per arm
    ctl = {}
    L += ["## Controls per arm (paired over cases): texture vs matched p95 / matched mean / plain random mean; intensity and edge minus matched", "",
          "| cell | arm | stage | k | E tex - E matched p95 | p | % > 0 | D tex | p | E tex - E plain | p | D int | p | D edge | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for arm in ARMS:
            for loc in STAGES:
                for kk in KS:
                    Et = piv(S, c, arm, "tex", loc, kk); a1 = (Et - piv(S, c, arm, "mmean", loc, kk, "E_target_p95")).dropna(); a2 = Dser(S, c, arm, loc, kk)
                    a3 = (Et - piv(S, c, arm, "pmean", loc, kk)).dropna(); a4 = Dser(S, c, arm, loc, kk, "int"); a5 = Dser(S, c, arm, loc, kk, "edge"); ctl[(c, arm, loc, kk)] = (a1, a2, a3, a4, a5)
                    for nm, s_ in (("tex-matched_p95", a1), ("tex-matched_mean", a2), ("tex-plain_mean", a3), ("int-matched_mean", a4), ("edge-matched_mean", a5)):
                        crow.append(dict(cell=c, loc=loc, k=kk, arm=arm, contrast=nm, median=s_.median(), p=wp(s_), frac_pos=(s_ > 0).mean()))
                    if (loc == HEAD and kk == KH) or (arm == "real" and kk == KH) or (arm == "real" and loc == HEAD):
                        L.append(f"| {c} | {arm} | {loc} | {kk} | {a1.median():+.3f} | {fp(wp(a1))} | {100 * (a1 > 0).mean():.0f} | {a2.median():+.3f} | {fp(wp(a2))} | {a3.median():+.3f} | {fp(wp(a3))} | "
                                 f"{a4.median():+.3f} | {fp(wp(a4))} | {a5.median():+.3f} | {fp(wp(a5))} |")
    pd.DataFrame(crow).to_csv(OUT / "chanabl_controls.csv", index=False)
    L += ["(rows shown: every arm at the headline; the real arm at every stage for k = 16 and every k at d3; everything in chanabl_controls.csv)", ""]
    # ---- depth x k
    drows = []
    for c in cells:
        for arm in ARMS4:
            for loc in STAGES:
                for kk in KS:
                    Et = piv(S, c, arm, "tex", loc, kk)
                    if not len(Et):
                        continue
                    D = Dser(S, c, arm, loc, kk); g = lambda v, col="E_target": piv(S, c, arm, v, loc, kk, col).median()   # noqa: E731
                    drows.append(dict(cell=c, arm=arm, loc=loc, k=kk, n=len(Et), E_tex=Et.median(), E_matched_mean=g("mmean"), E_matched_p95=g("mmean", "E_target_p95"),
                                      E_matched_p025=g("mmean", "E_target_p025"), E_matched_p975=g("mmean", "E_target_p975"), E_plain_mean=g("pmean"), E_int=g("int"), E_edge=g("edge"),
                                      D_tex=D.median(), p_D_tex=wp(D), D_int=Dser(S, c, arm, loc, kk, "int").median(), D_edge=Dser(S, c, arm, loc, kk, "edge").median(),
                                      E_tex_ring=g("tex", "E_ring"), E_tex_bg=g("tex", "E_bg"), frac_gt1_tex=g("tex", "frac_gt1"), dice_nat=g("tex", "dice_nat"), dice_tex=g("tex", "dice_after"),
                                      dice_matched0=g("mmean", "dice_after"), dice_int=g("int", "dice_after"), dice_edge=g("edge", "dice_after")))
    Dd = pd.DataFrame(drows); Dd.to_csv(OUT / "chanabl_depth_k.csv", index=False)
    L += [f"## Depth profile at k = {KH}: median over cases of D tex (E tex / E matched mean)", "", "| cell | arm | " + " | ".join(STAGES) + " |", "|" + "---|" * (2 + len(STAGES))]
    for c in cells:
        for arm in ARMS:
            d = Dd[(Dd.cell == c) & (Dd.arm == arm) & (Dd.k == KH)].set_index("loc").reindex(list(STAGES))
            L.append(f"| {c} | {arm} | " + " | ".join(f"{q.D_tex:+.2f} ({q.E_tex:+.2f} / {q.E_matched_mean:+.2f})" for _, q in d.iterrows()) + " |")
    L += ["", f"## k profile at {HEAD}: median D tex (E tex / E matched mean)", "", "| cell | arm | " + " | ".join(f"k={kk}" for kk in KS) + " |", "|" + "---|" * (2 + len(KS))]
    for c in cells:
        for arm in ARMS:
            d = Dd[(Dd.cell == c) & (Dd.arm == arm) & (Dd["loc"] == HEAD)].set_index("k").reindex(list(KS))
            L.append(f"| {c} | {arm} | " + " | ".join(f"{q.D_tex:+.2f} ({q.E_tex:+.2f} / {q.E_matched_mean:+.2f})" for _, q in d.iterrows()) + " |")
    L.append("")
    # ---- verdicts
    V = []
    hk = (HEAD, KH); rf_ = [c for c in cells if con[(c,) + hk][0].median() > 0 and wp(con[(c,) + hk][0]) < 0.01]
    rn_ = [c for c in cells if con[(c,) + hk][1].median() > 0 and wp(con[(c,) + hk][1]) < 0.01]
    p1 = len(rf_) >= 3
    V.append(f"- P1 ({HEAD}, k = {KH}; D real > flat in >= 3/4 cells, p < 0.01): {'SUPPORTED' if p1 else 'NOT SUPPORTED (kill condition)'}: real > flat in {len(rf_)}/{nc} ("
             + "; ".join(f"{c} r-f {con[(c,) + hk][0].median():+.3f} p {fp(wp(con[(c,) + hk][0]))}" for c in cells) + f"). Reported, not a pass condition: real > noise in {len(rn_)}/{nc} ("
             + "; ".join(f"{c} r-n {con[(c,) + hk][1].median():+.3f} p {fp(wp(con[(c,) + hk][1]))}" for c in cells) + ").")
    for loc in STAGES:
        if loc != HEAD:
            a = sum(1 for c in cells if con[(c, loc, KH)][0].median() > 0 and wp(con[(c, loc, KH)][0]) < 0.01); V.append(f"  - same rule at {loc} (k = {KH}, not the headline): real > flat {a}/{nc}")
    for kk in KS:
        if kk != KH:
            a = sum(1 for c in cells if con[(c, HEAD, kk)][0].median() > 0 and wp(con[(c, HEAD, kk)][0]) < 0.01); V.append(f"  - same rule at {HEAD}, k = {kk}: real > flat {a}/{nc}")
    q1 = [c for c in cells if ctl[(c, "real", HEAD, KH)][0].median() > 0 and wp(ctl[(c, "real", HEAD, KH)][0]) < 0.01]
    V.append(f"- P2 (real arm, {HEAD}, k = {KH}; E tex > matched p95 in >= 3/4 cells): {'SUPPORTED' if len(q1) >= 3 else 'NOT SUPPORTED'} in {len(q1)}/{nc} ("
             + "; ".join(f"{c} {ctl[(c, 'real', HEAD, KH)][0].median():+.3f} p {fp(wp(ctl[(c, 'real', HEAD, KH)][0]))}" for c in cells) + ").")
    p3c = []
    for c in cells:
        drops = {a: (piv(S, c, a, "int", HEAD, KH, "dice_nat") - piv(S, c, a, "int", HEAD, KH, "dice_after")).median() for a in ARMS}; p3c.append((c, drops, all(v > 0.2 for v in drops.values())))
    n3 = sum(1 for _, _, ok in p3c if ok)
    V.append(f"- P3 (positive control, {HEAD}, k = {KH}; intensity-channel ablation drops median Dice by > 0.2 in every trained arm, 4/4 cells): {'SUPPORTED' if n3 == nc else 'NOT SUPPORTED'} in {n3}/{nc} ("
             + "; ".join(f"{c} " + ", ".join(f"{a} {v:+.2f}" for a, v in d_.items()) for c, d_, _ in p3c) + ").")
    V.append("- P4 (readability): judged by eye on chanabl_examples_<cell>.png, see FINDINGS 45.")
    L += ["## Verdicts (pre-registered rules)", ""] + V + [""]
    (OUT / "chanabl.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
    summary_figure(cells, S, Dd, pd.DataFrame(urows))
    print("CHANABL_REPORT_DONE", flush=True)


def k_panel(ax, Dd, c, title=True):
    for arm in ARMS4:
        d = Dd[(Dd.cell == c) & (Dd.arm == arm) & (Dd["loc"] == HEAD)].set_index("k").reindex(list(KS))
        if arm == "init":
            if d.E_tex.notna().any():
                ax.plot(KS, d.E_tex, "x", color=PAL4[arm], ms=6, label="init floor (k=16)")
            continue
        ax.plot(KS, d.E_tex, "-o", color=PAL4[arm], ms=3, label=f"{arm}: texture channels"); ax.fill_between(KS, d.E_matched_p025, d.E_matched_p975, color=PAL4[arm], alpha=0.15, lw=0)
        ax.plot(KS, d.E_int, ":", color=PAL4[arm], lw=1)
    ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.set_xlabel("k ablated channels")
    if title:
        ax.set_title(f"{HEAD}: target E vs k (line = texture channels, dotted = intensity channels,\nshaded = matched-random 2.5-97.5 pct of draws), median over cases", fontsize=7)


def units_panel(ax, Ud, c):
    xo = np.arange(len(STAGES)); w = 0.2
    for i, arm in enumerate(ARMS4):
        d = Ud[(Ud.cell == c) & (Ud.arm == arm)].set_index("loc").reindex(list(STAGES)) if len(Ud) else None
        if d is None or d.C.isna().all():
            continue
        ax.bar(xo + (i - 1.5) * w, d.units_caseswap / d.C, w, color=PAL4[arm], label=arm)
        ax.scatter(xo + (i - 1.5) * w, d.units_shuffle / d.C, marker="_", color="k", s=30, zorder=3, label="shuffle-null count" if i == 0 else None)
    ax.set_xticks(xo); ax.set_xticklabels(STAGES); ax.set_ylim(0, 1.05); ax.set_ylabel("fraction of channels")
    ax.set_title("texture-energy-selective units per stage (bars: case-swap null p99;\ndash: spatial-shuffle null p99), mean over folds", fontsize=7)


def summary_figure(cells, S, Dd, Ud):
    nc = len(cells); fig, axs = plt.subplots(3, nc, figsize=(4.8 * nc, 11), squeeze=False)
    for ci, c in enumerate(cells):
        ax = axs[0, ci]; w = 0.36
        for i, arm in enumerate(ARMS):
            D = Dser(S, c, arm, HEAD, HEADK); Di = Dser(S, c, arm, HEAD, HEADK, "int")
            ax.bar(i - w / 2, D.median(), w, color=PAL[arm], label="texture channels" if i == 0 else None)
            ax.errorbar(i - w / 2, D.median(), yerr=[[D.median() - D.quantile(0.25)], [D.quantile(0.75) - D.median()]], color="k", lw=0.8, capsize=2)
            ax.bar(i + w / 2, Di.median(), w, color="white", edgecolor=PAL[arm], hatch="....", label="intensity channels (control)" if i == 0 else None)
        fl = Dser(S, c, "init", HEAD, HEADK).median(); ax.axhline(fl, color="0.4", ls=":", lw=1, label="random-init floor (texture)"); ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"{c} (n = {S[(S.cell == c) & (S.kind == 'nat')].case.nunique()})\nD = E(selected) - E(variance-matched random), {HEAD}, k = {HEADK}", fontsize=8)
        if ci == 0:
            ax.set_ylabel("edema log-odds drop, mean over GT edema (nat)\nmedian [IQR] over cases"); ax.legend(fontsize=6, frameon=False)
        k_panel(axs[1, ci], Dd, c)
        if ci == 0:
            axs[1, ci].set_ylabel("target E (nat)"); axs[1, ci].legend(fontsize=6, frameon=False)
        units_panel(axs[2, ci], Ud, c)
        if ci == 0:
            axs[2, ci].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "chanabl_summary.png", dpi=150); fig.savefig(OUT / "chanabl_summary.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------------------- figure
def figure():
    cells, S, U = load_all(); Dd = pd.read_csv(OUT / "chanabl_depth_k.csv") if (OUT / "chanabl_depth_k.csv").exists() else None
    Ud = pd.read_csv(OUT / "chanabl_units.csv") if (OUT / "chanabl_units.csv").exists() else pd.DataFrame()
    OR = np.array([1.0, 0.55, 0.0]); CY = np.array([0.1, 0.8, 1.0]); Pz = 2.3
    for c in cells:
        drf = (Dser(S, c, "real", HEAD, HEADK) - Dser(S, c, "flat", HEAD, HEADK)).dropna(); drf = drf[[(OUT / c / f"{q}_chanabl_maps.npz").exists() for q in drf.index]]
        if not len(drf):
            continue
        picks = []
        for qv in (0.5, 0.75):
            tq = float(drf.quantile(qv))
            for q in (drf - tq).abs().sort_values().index:
                if q not in picks:
                    picks.append(q); break
        rows_ = [(case, arm) for case in picks for arm in ("real", "flat", "noise")]
        fig = plt.figure(figsize=(Pz * 5 + 4.4, Pz * len(rows_) + 0.9)); gs = fig.add_gridspec(len(rows_), 7, width_ratios=[1, 1, 1, 1, 1, 0.15, 1.9])
        for pi, case in enumerate(picks):
            M = np.load(OUT / c / f"{case}_chanabl_maps.npz"); im = M["ip"].astype(np.float64); gt = M["gt"].astype(bool); body = im != 0
            if not body.any():
                body = np.ones_like(im, bool)
            lo, hi = np.percentile(im[body], 1), np.percentile(im[body], 99); g0 = np.clip((im - lo) / (hi - lo + 1e-12), 0, 1); g0[~body] = 0; G3 = np.stack([g0] * 3, -1)
            Es = {a: M[f"{a}_E_tex"].astype(np.float64) for a in ARMS}; s99 = np.percentile(np.abs(np.stack(list(Es.values()))), 99) or 1e-12

            def paint(Mp):
                o = np.clip((np.abs(Mp) / s99) ** 0.6, 0, 1) * 0.9; o[~body] = 0; col = np.where((Mp > 0)[..., None], OR[None, None], CY[None, None])
                return np.clip(G3 * (1 - 0.6 * o[..., None]) + col * o[..., None], 0, 1)

            def predimg(mask, arm):
                cc = np.array(matplotlib.colors.to_rgb(PAL[arm])); o = mask.astype(float)[..., None] * 0.45
                return np.clip(G3 * (1 - o) + cc[None, None] * o, 0, 1)
            sr = S[(S.cell == c) & (S.case == case) & (S.kind == "E") & (S["loc"] == HEAD) & (S.k == HEADK)]
            for ai, arm in enumerate(("real", "flat", "noise")):
                rr = 3 * pi + ai; e_ = sr[(sr.arm == arm) & (sr["var"] == "tex")].iloc[0]; e_m = sr[(sr.arm == arm) & (sr["var"] == "mmean")].iloc[0]
                panels = [("image", G3, None), (f"prediction (Dice {e_.dice_nat:.2f})", predimg(M[f"{arm}_pred"], arm), M[f"{arm}_pred"]),
                          (f"after TEXTURE-channel ablation\n{HEAD}, k={HEADK} (Dice {e_.dice_after:.2f})", predimg(M[f"{arm}_pred_tex"], arm), M[f"{arm}_pred_tex"]),
                          (f"after MATCHED-random ablation\n(draw 0; Dice {e_m.dice_after:.2f})", predimg(M[f"{arm}_pred_mr0"], arm), M[f"{arm}_pred_mr0"]),
                          (f"effect E (target mean {e_.E_target:+.2f} nat;\nmatched-random mean {e_m.E_target:+.2f})", paint(Es[arm]), None)]
                for cc_, (title, arr, pm) in enumerate(panels):
                    ax = fig.add_subplot(gs[rr, cc_]); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
                    ax.imshow(np.transpose(arr, (1, 0, 2)), origin="lower", interpolation="nearest"); ax.contour(gt.T.astype(float), [0.5], colors="lime", linewidths=0.7)
                    if pm is not None and pm.any():
                        ax.contour(pm.T.astype(float), [0.5], colors=[PAL[arm]], linewidths=0.8)
                    ax.set_title(title, fontsize=6.5)
                    if cc_ == 0:
                        ax.set_ylabel(f"{arm}\n{'typical' if pi == 0 else '75th pct'} case {case}\nreal - flat D {drf.get(case, np.nan):+.2f} nat", fontsize=6.5)
        if Dd is not None:
            nr = len(rows_); ax = fig.add_subplot(gs[: nr // 2, 6]); k_panel(ax, Dd, c); ax.legend(fontsize=6, frameon=False); ax.set_ylabel("target E (nat)")
            ax = fig.add_subplot(gs[nr // 2:, 6]); units_panel(ax, Ud, c); ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"{c}: mean-ablating the {HEADK} TEXTURE-ENERGY-SELECTIVE channels at {HEAD} vs {HEADK} variance-matched random channels.\n"
                     "Filled = predicted edema (arm colour), lime = GT edema; effect: orange = the ablated channels supported the edema call, cyan = opposed (shared scale, p99 over arms)", fontsize=8.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"chanabl_examples_{c}.png", dpi=130); plt.close(fig)
    print("CHANABL_FIGURE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--fit", action="store_true"); ap.add_argument("--report", action="store_true"); ap.add_argument("--figure", action="store_true")
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
        fit_cell(key, dev) if args.fit else run_cell(key, dev, args.case_slice)


if __name__ == "__main__":
    main()
