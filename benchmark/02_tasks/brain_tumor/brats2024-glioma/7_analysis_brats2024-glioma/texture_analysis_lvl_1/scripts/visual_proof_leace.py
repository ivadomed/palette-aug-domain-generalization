#!/usr/bin/env python
"""Visual proof v17 (FINDINGS 44, pre-registered in outputs/plots/leace/PREREG.md): LEACE erasure of the local texture-energy subspace inside the
network, BraTS cells, arms noise (rung 4, lblvor) / flat (4.5) / real (5) + "init" (the real network re-initialised, torch seed 1000 + k), folds 0-2.

Concept (image-only, mask-free, label-free; FINDINGS 31 descriptors): I = G2(x), E = |grad G2(x)|, Tf = sqrt(G2((x - G1 x)^2)); r = log Tf residualised
per case on [1, I, log E] over body voxels at full resolution; then the body-masked area average on each stage's grid; then per-case quantile bins
(K = 8) -> one-hot Z. Control concept Zc = the same with I.
--fit   (GPU): first N_FIT (20) sorted cases of the cell; per arm x fold x stage (s2, s3, s4 skips; d1, d2, d3 decoder stages) float64 moments over the
        stage-body voxels; closed-form LEACE eraser A = W+ P W (W = (Sigma + lam I)^-1/2, P = projector on colspace(W Sigma_hz)); intensity eraser; 20
        random equal-rank whitened-space subspaces (stored as bases U, A_rand = W+ U U^T W); ridge probes (bin index and continuous value of each concept)
        before and refitted after erasure, in closed form from the same moments. -> OUT/<cell>/fit/leace_fit_<arm>_f<k>.npz
run     (GPU): evaluation cases (sorted index >= N_FIT) in --case-slice. Per arm x fold: natural forward; per stage, erasure h' = h - A (h - mu) at EVERY
        voxel of that stage's tensor (skips: the tensor concatenated into the decoder only; decoder stages: the stage output), decoder re-entered from that
        point (batched over erasers). J = edema (class 2) log-odds; E = J_nat - J_erased; fold-mean maps; scalars on target / ring 1-10 vox / far
        background, frac target > 1 nat, Dice (fold-mean softmax) before / after. Random draws: per-draw target / ring / background means, mean map,
        Dice for draw 0. Probe scoring (SSE, sums) on the case's h and h' -> held-out R^2 in the report.
        -> OUT/<cell>/<case>_leace.csv (written last), <case>_leace_probe.csv, <case>_leace_maps.npz (one axial slice, float16, patient data: never commit)
--report (CPU): leace.md, leace_cells.csv, leace_depth.csv, leace_controls.csv, leace_probe.csv, leace_summary.png/pdf.
--figure (CPU): leace_examples_<cell>.png (patient slices: never commit).
Env: VP_LEACE_OUT (output dir), LE_FOLDS ("0 1 2"), LE_NFIT (20), LE_DRAWS (20), LE_K (8), LE_BATCH (8), LE_SELFCHECK (1 = check decoder re-entry).
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

OUT = Path(os.environ.get("VP_LEACE_OUT") or (THIS.parent / "outputs/plots/leace")); OUT.mkdir(parents=True, exist_ok=True)
CELLS4 = ("glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w")
FOLDS = tuple(int(v) for v in os.environ.get("LE_FOLDS", "0 1 2").split())
N_FIT = int(os.environ.get("LE_NFIT", 20)); N_DRAWS = int(os.environ.get("LE_DRAWS", 20)); K = int(os.environ.get("LE_K", 8))
BATCH = int(os.environ.get("LE_BATCH", 8)); SELFCHECK = bool(int(os.environ.get("LE_SELFCHECK", 0)))
STAGES = ("s2", "s3", "s4", "d1", "d2", "d3"); HEAD = "d2"; DEPTH_ORDER = ("s2", "s3", "s4", "d1", "d2", "d3")
LAM, RTOL, RING, ONE_NAT, EDEMA = 1e-3, 1e-6, 10, 1.0, 2
ARMS4 = tuple(ARMS) + ("init",); PAL4 = dict(PAL, init="#7f7f7f")
BINV = np.arange(K, dtype=np.float64) / (K - 1)
PROBES = ("tb_pre", "tb_tex", "tb_r0", "tc_pre", "tc_tex", "tc_r0", "ib_pre", "ib_int", "ic_pre", "ic_int")   # target (t/i) x (bin/cont) x eraser


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fp(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.1g}"


# ----------------------------------------------------------------------------------------------------------------------------------- concept
def concepts(ip, bp):
    """FINDINGS 31 descriptors at full patch resolution: residualised log Tf (per-case OLS on [1, I, log E] over body voxels) and I."""
    x = ip.astype(np.float64); b = bp.astype(bool)
    hp = x - ndi.gaussian_filter(x, 1.0); I = ndi.gaussian_filter(x, 2.0); E = ndi.gaussian_gradient_magnitude(x, 2.0); Tf = np.sqrt(ndi.gaussian_filter(hp ** 2, 2.0))
    X = np.c_[np.ones(int(b.sum())), I[b], np.log(E[b] + 1e-6)]; y = np.log(Tf[b] + 1e-6); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.zeros_like(x); r[b] = y - X @ beta
    return r, I


def stage_concepts(r, I, b, shape, dev):
    """-> stage-body mask (bool, flat), and on the stage-body voxels: one-hot Z (n, K), one-hot Zc (n, K), r, I, bin index of r, bin index of I."""
    import torch
    import torch.nn.functional as Fn
    t = torch.from_numpy(np.stack([r * b, I * b, b.astype(np.float64)]))[None].to(dev)
    a = Fn.interpolate(t, size=shape, mode="area")[0]; bm = (a[2] >= 0.5).reshape(-1)
    den = a[2].clamp_min(1e-6); rs = (a[0] / den).reshape(-1)[bm]; Is = (a[1] / den).reshape(-1)[bm]

    def onehot(v):
        q = torch.quantile(v, torch.linspace(0, 1, K + 1, dtype=v.dtype, device=v.device)[1:-1]); idx = torch.bucketize(v, q).clamp(0, K - 1)
        return torch.nn.functional.one_hot(idx, K).double(), idx.double() / (K - 1)
    Zt, bt = onehot(rs); Zi, bi = onehot(Is)
    return dict(bm=bm, Zt=Zt, Zi=Zi, tc=rs, ic=Is, tb=bt, ib=bi)


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
    """Re-enter net's decoder with the batch Hb (B, C, ...) in place of the activation at loc. Skip s_j: only the tensor concatenated at decoder stage
    4 - j is replaced (the encoder downstream is untouched); decoder stage d_s: the decoder continues from stage s + 1. Other inputs (batch 1) broadcast."""
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


def erase(h, mu, As):
    """h (1, C, ...) -> (B, C, ...): h - A_b (h - mu) for each eraser A_b in As (B, C, C)."""
    C = h.shape[1]; sp = h.shape[2:]; hf = h[0].reshape(C, -1); d = hf - mu[:, None]
    return (hf[None] - As @ d[None]).reshape(As.shape[0], C, *sp)


# ----------------------------------------------------------------------------------------------------------------------------------- fit
def fit_cell(key, dev):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key / "fit"; od.mkdir(parents=True, exist_ok=True)
    prs = load_arms(key, dev); pr0 = prs["flat"]; cases = sorted(cell["cases"])[:N_FIT]; acc = {}
    log(f"[{key}] fit on {len(cases)} cases, folds {FOLDS}, K {K}, draws {N_DRAWS}")
    for case in cases:
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); r, I = concepts(ip, bp); b = bp.astype(bool)
        x = torch.from_numpy(ip)[None, None].to(dev); conc = {}
        for arm, pr in prs.items():
            for k in FOLDS:
                net = set_fold(pr, k); A, _ = acts(net, x)
                for L in STAGES:
                    h = A[L][0]; shp = tuple(h.shape[1:]); C = h.shape[0]
                    if shp not in conc:
                        conc[shp] = stage_concepts(r, I, b, shp, dev)
                    c = conc[shp]; Zf = torch.cat([c["Zt"], c["Zi"], c["tc"][:, None], c["ic"][:, None]], 1)
                    H = h.reshape(C, -1)[:, c["bm"]].double(); st = acc.setdefault((arm, k, L), [0, 0.0, 0.0, 0.0, 0.0])
                    st[0] += H.shape[1]; st[1] = st[1] + H.sum(1); st[2] = st[2] + H @ H.T; st[3] = st[3] + H @ Zf; st[4] = st[4] + Zf.sum(0)
                del A; torch.cuda.empty_cache()
        log(f"[{key}] fit {case}: {time.time() - t0:.0f} s")
    for arm in prs:
        for k in FOLDS:
            out = {}
            for L in STAGES:
                n, sh, shh, shz, sz = acc[(arm, k, L)]; mu = sh / n; Sig = shh / n - torch.outer(mu, mu); mz = sz / n; Shz = shz / n - torch.outer(mu, mz)
                C = Sig.shape[0]; Id = torch.eye(C, dtype=Sig.dtype, device=Sig.device); lam = LAM * torch.trace(Sig) / C
                ev, V = torch.linalg.eigh(Sig + lam * Id); ev = ev.clamp_min(1e-12); W = V @ torch.diag(ev ** -0.5) @ V.T; Wp = V @ torch.diag(ev ** 0.5) @ V.T

                def eraser(Sz):
                    U, S, _ = torch.linalg.svd(W @ Sz, full_matrices=False); rr = int((S > RTOL * S[0]).sum()); Ur = U[:, :rr]
                    return Wp @ Ur @ Ur.T @ W, rr, S
                Szt, Szi = Shz[:, :K], Shz[:, K:2 * K]; At, rt, St = eraser(Szt); Ai, ri, Si = eraser(Szi)
                ins_t = float(torch.linalg.norm((Id - At) @ Szt) / (torch.linalg.norm(Szt) + 1e-30)); ins_i = float(torch.linalg.norm((Id - Ai) @ Szi) / (torch.linalg.norm(Szi) + 1e-30))
                seed = int(hashlib.md5(f"{key}|{arm}|{k}|{L}".encode()).hexdigest()[:8], 16); rng = np.random.default_rng(seed); Ur = []
                for d in range(N_DRAWS):
                    G = torch.from_numpy(rng.standard_normal((C, rt))).to(Sig); Q, _ = torch.linalg.qr(W @ G); Ur.append(Q)
                Ur = torch.stack(Ur); A0 = Wp @ Ur[0] @ Ur[0].T @ W
                bv = torch.from_numpy(BINV).to(Sig)
                ys = {"tc": (Shz[:, 2 * K], mz[2 * K]), "tb": (Szt @ bv, mz[:K] @ bv), "ic": (Shz[:, 2 * K + 1], mz[2 * K + 1]), "ib": (Szi @ bv, mz[K:2 * K] @ bv)}
                ers = {"pre": None, "tex": At, "r0": A0, "int": Ai}
                for nm in PROBES:
                    yk, ek = nm.split("_"); cy, my = ys[yk]; Ae = ers[ek]
                    if Ae is None:
                        S_, c_ = Sig, cy
                    else:
                        B_ = Id - Ae; S_, c_ = B_ @ Sig @ B_.T, B_ @ cy
                    w = torch.linalg.solve(S_ + LAM * torch.trace(S_) / C * Id, c_); out[f"{L}_p_{nm}_w"] = w.float().cpu().numpy(); out[f"{L}_p_{nm}_b"] = float(my - w @ mu)
                out.update({f"{L}_mu": mu.float().cpu().numpy(), f"{L}_At": At.float().cpu().numpy(), f"{L}_Ai": Ai.float().cpu().numpy(), f"{L}_W": W.float().cpu().numpy(),
                            f"{L}_Wp": Wp.float().cpu().numpy(), f"{L}_Urand": Ur.float().cpu().numpy(), f"{L}_rt": rt, f"{L}_ri": ri, f"{L}_St": St.float().cpu().numpy(),
                            f"{L}_Si": Si.float().cpu().numpy(), f"{L}_n": int(n), f"{L}_ins_t": ins_t, f"{L}_ins_i": ins_i})
                log(f"[{key}] {arm} f{k} {L}: C {C}, n {int(n)}, rank tex {rt} int {ri}, in-sample |Cov(h',Z)|/|Cov(h,Z)| tex {ins_t:.2e} int {ins_i:.2e}, "
                    f"sv tex {' '.join(f'{v:.3g}' for v in St[:K].tolist())}")
            np.savez_compressed(od / f"leace_fit_{arm}_f{k}.npz", **out)
    print("LEACE_FIT_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- run
def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; fd = od / "fit"; c0 = int(T[0])
    prs = load_arms(key, dev); pr0 = prs["flat"]
    allc = sorted(cell["cases"]); idx = range(len(allc))[slice(*[int(v) for v in sl.split(":")])] if sl else range(len(allc))
    cases = [allc[i] for i in idx if i >= N_FIT]
    F = {}
    for arm in ARMS4:
        for k in FOLDS:
            Z = np.load(fd / f"leace_fit_{arm}_f{k}.npz"); d = {}
            for L in STAGES:
                g = lambda nm: torch.from_numpy(Z[f"{L}_{nm}"]).to(dev)   # noqa: E731
                W, Wp, Ur = g("W"), g("Wp"), g("Urand"); Ar = torch.stack([Wp @ Ur[i] @ Ur[i].T @ W for i in range(Ur.shape[0])])
                d[L] = dict(mu=g("mu"), As=torch.cat([g("At")[None], g("Ai")[None], Ar], 0), rt=int(Z[f"{L}_rt"]), ri=int(Z[f"{L}_ri"]),
                            p={nm: (g(f"p_{nm}_w"), float(Z[f"{L}_p_{nm}_b"])) for nm in PROBES})
            F[(arm, k)] = d
    names = ["tex", "int"] + [f"r{i}" for i in range(N_DRAWS)]; nv = len(names)
    log(f"[{key}] run: {len(cases)} evaluation cases, folds {FOLDS}, draws {N_DRAWS}, batch {BATCH}")
    for case in cases:
        f_out = od / f"{case}_leace.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); r, I = concepts(ip, bp); b = bp.astype(bool)
        tg = gp == EDEMA; dil = ndi.binary_dilation(tg, iterations=RING); ring = dil & ~tg & b; bg = b & ~dil
        z = int(np.argmax(tg.sum((1, 2)))) if tg.any() else ip.shape[0] // 2
        x = torch.from_numpy(ip)[None, None].to(dev); conc = {}
        R = {nm: torch.from_numpy(m_).to(dev) for nm, m_ in (("tg", tg), ("ring", ring), ("bg", bg))}
        base = dict(cell=key, task=cell["task"], case=case, n_target=int(tg.sum()), n_ring=int(ring.sum()), n_bg=int(bg.sum()), z=z)
        rows, prows = [], []; maps = dict(z=np.int32(z), ip=ip[z], gt=tg[z])

        def rmean(Mp, nm):
            m_ = R[nm]; return float(Mp[m_].mean()) if bool(m_.any()) else np.nan

        def dice(P):
            pr_ = (P.argmax(0) == c0); g_ = R["tg"]; return float(2 * (pr_ & g_).sum() / (pr_.sum() + g_.sum() + 1e-9))
        for arm, pr in prs.items():
            nf = float(len(FOLDS)); accJ = 0; accP = 0; accE = {}; accPv = {}; dr = {nm: torch.zeros(len(STAGES), N_DRAWS, dtype=torch.float64) for nm in ("tg", "ring", "bg")}
            for k in FOLDS:
                net = set_fold(pr, k); A, lg = acts(net, x); J = e7.log_odds(lg.float(), c0)[0]; P = torch.softmax(lg.float(), 1)[0]
                accJ = accJ + J; accP = accP + P; skips = [A[f"s{j}"] for j in range(6)]; ds = [A[f"d{s}"] for s in range(5)]
                for li, L in enumerate(STAGES):
                    h = A[L]; f = F[(arm, k)][L]; shp = tuple(h.shape[2:]); C = h.shape[1]
                    if shp not in conc:
                        conc[shp] = stage_concepts(r, I, b, shp, dev)
                    c = conc[shp]; ycase = {"tc": c["tc"], "tb": c["tb"], "ic": c["ic"], "ib": c["ib"]}
                    with torch.no_grad(), torch.autocast("cuda", enabled=False):
                        if SELFCHECK and arm == "real" and k == FOLDS[0]:
                            lgc = decode_from(net, skips, ds, L, h); log(f"[{key}] selfcheck {L}: max |logits re-entry - natural| = {float((lgc - lg).abs().max()):.2e}")
                        Hb_pre = h[0].reshape(C, -1)[:, c["bm"]].double()

                        def score(Hv, nm):
                            w, b_ = f["p"][nm]; yk = nm.split("_")[0]; y = ycase[yk]; pred = w.double() @ Hv + b_
                            prows.append(dict(base, arm=arm, fold=k, loc=L, probe=nm, n=int(y.numel()), sse=float(((y - pred) ** 2).sum()), sy=float(y.sum()), syy=float((y ** 2).sum())))
                        for nm in ("tb_pre", "tc_pre", "ib_pre", "ic_pre"):
                            score(Hb_pre, nm)
                        for i0 in range(0, nv, BATCH):
                            sel = list(range(i0, min(nv, i0 + BATCH))); Hb = erase(h, f["mu"], f["As"][sel]); lgb = decode_from(net, skips, ds, L, Hb)
                            Jb = e7.log_odds(lgb.float(), c0); Eb = J[None] - Jb
                            for jj, vi in enumerate(sel):
                                nm = names[vi]; Ev = Eb[jj]
                                if nm in ("tex", "int", "r0"):
                                    Pv = torch.softmax(lgb[jj:jj + 1].float(), 1)[0]; accPv[(nm, L)] = accPv.get((nm, L), 0) + Pv
                                    Hv = Hb[jj].reshape(C, -1)[:, c["bm"]].double()
                                    for pn in {"tex": ("tb_tex", "tc_tex"), "int": ("ib_int", "ic_int"), "r0": ("tb_r0", "tc_r0")}[nm]:
                                        score(Hv, pn)
                                if nm in ("tex", "int"):
                                    accE[(nm, L)] = accE.get((nm, L), 0) + Ev
                                else:
                                    d_ = int(nm[1:]); accE[("rmean", L)] = accE.get(("rmean", L), 0) + Ev / N_DRAWS
                                    for rn in ("tg", "ring", "bg"):
                                        dr[rn][li, d_] += rmean(Ev, rn)
                            del Hb, lgb, Jb, Eb
                del A, lg, skips, ds; torch.cuda.empty_cache()
            Jm = accJ / nf; Pm = accP / nf; dn = dice(Pm)
            rows.append(dict(base, kind="nat", arm=arm, var="nat", loc="", J_target=rmean(Jm, "tg"), J_ring=rmean(Jm, "ring"), J_bg=rmean(Jm, "bg"), dice=dn))
            maps[f"{arm}_J"] = Jm[z].cpu().numpy(); maps[f"{arm}_pred"] = (Pm.argmax(0) == c0)[z].cpu().numpy()
            for li, L in enumerate(STAGES):
                f0 = F[(arm, FOLDS[0])][L]; rt = float(np.mean([F[(arm, k)][L]["rt"] for k in FOLDS])); ri = float(np.mean([F[(arm, k)][L]["ri"] for k in FOLDS]))
                for var in ("tex", "int", "rmean"):
                    Em = accE[(var, L)] / nf; pv = "r0" if var == "rmean" else var; Pv = accPv[(pv, L)] / nf
                    row = dict(base, kind="E", arm=arm, var=var, loc=L, E_target=rmean(Em, "tg"), E_ring=rmean(Em, "ring"), E_bg=rmean(Em, "bg"),
                               frac_gt1=float((Em[R["tg"]] > ONE_NAT).float().mean()) if bool(R["tg"].any()) else np.nan, dice_nat=dn, dice_after=dice(Pv), rank_tex=rt, rank_int=ri)
                    if var == "rmean":
                        dt = (dr["tg"][li] / nf).numpy()
                        row.update(E_target_mean_draws=float(dt.mean()), E_target_p95=float(np.percentile(dt, 95)), E_target_p025=float(np.percentile(dt, 2.5)),
                                   E_target_p975=float(np.percentile(dt, 97.5)), E_ring_p95=float(np.percentile((dr["ring"][li] / nf).numpy(), 95)),
                                   **{f"E_target_d{i}": float(dt[i]) for i in range(N_DRAWS)})
                    rows.append(row)
                    if L == HEAD or var == "tex":
                        maps[f"{arm}_E_{var}_{L}"] = Em[z].cpu().numpy()
                    if L == HEAD:
                        maps[f"{arm}_pred_{pv}_{L}"] = (Pv.argmax(0) == c0)[z].cpu().numpy()
            del accE, accPv, accJ, accP; torch.cuda.empty_cache()
        np.savez_compressed(od / f"{case}_leace_maps.npz", **{k_: (v_.astype(np.float16) if isinstance(v_, np.ndarray) and v_.dtype.kind == "f" else v_) for k_, v_ in maps.items()})
        pd.DataFrame(prows).to_csv(od / f"{case}_leace_probe.csv", index=False); pd.DataFrame(rows).to_csv(f_out, index=False)
        S_ = pd.DataFrame(rows); s_ = S_[(S_.kind == "E") & (S_["loc"] == HEAD)]
        log(f"[{key}] {case}: target E_{HEAD} tex / rand-mean / int: " + "; ".join(
            f"{a} {s_[(s_.arm == a) & (s_['var'] == 'tex')].E_target.iloc[0]:+.2f} / {s_[(s_.arm == a) & (s_['var'] == 'rmean')].E_target.iloc[0]:+.2f} / {s_[(s_.arm == a) & (s_['var'] == 'int')].E_target.iloc[0]:+.2f}"
            for a in ARMS4) + f"; {time.time() - t0:.0f} s")
    print("LEACE_RUN_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- held-out probe
def probe_heldout_cell(key, dev):
    """Added after the main run (the pre-registered gate refits the probe on the FIT-case moments, where Cov(h', bin index) = 0 exactly, so the bin-index
    gate holds by construction): probes REFITTED ON THE EVALUATION CASES. Erasers fixed from the fit; evaluation cases split by sorted-index parity;
    ridge (lam = 1e-3 trace / C) fitted on one half's moments of h' (closed form from the moments of h, h' = (I - A) h + A mu), R^2 on the other half,
    both directions, mean over directions and folds. -> OUT/<cell>/probe_heldout.csv"""
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; fd = OUT / key / "fit"; prs = load_arms(key, dev); pr0 = prs["flat"]
    allc = sorted(cell["cases"]); ev = [(i, c) for i, c in enumerate(allc) if i >= N_FIT]; acc = {}; YK = ("tb", "tc", "ib", "ic")
    for i, case in ev:
        t0 = time.time(); g = i % 2; ip, bp, gp, _ = load_case(pr0, cell, case, T); r, I = concepts(ip, bp); b = bp.astype(bool); x = torch.from_numpy(ip)[None, None].to(dev); conc = {}
        for arm, pr in prs.items():
            for k in FOLDS:
                net = set_fold(pr, k); A, _ = acts(net, x)
                for L in STAGES:
                    h = A[L][0]; shp = tuple(h.shape[1:]); C = h.shape[0]
                    if shp not in conc:
                        conc[shp] = stage_concepts(r, I, b, shp, dev)
                    c = conc[shp]; Y = torch.stack([c[y] for y in YK], 1); H = h.reshape(C, -1)[:, c["bm"]].double()
                    st = acc.setdefault((arm, k, L, g), [0, 0.0, 0.0, 0.0, 0.0, 0.0]); st[0] += H.shape[1]; st[1] = st[1] + H.sum(1); st[2] = st[2] + H @ H.T
                    st[3] = st[3] + H @ Y; st[4] = st[4] + Y.sum(0); st[5] = st[5] + (Y ** 2).sum(0)
                del A; torch.cuda.empty_cache()
        log(f"[{key}] probe-heldout {case}: {time.time() - t0:.0f} s")
    rows = []
    for arm in prs:
        for k in FOLDS:
            Z = np.load(fd / f"leace_fit_{arm}_f{k}.npz")
            for L in STAGES:
                gt = lambda nm: torch.from_numpy(Z[f"{L}_{nm}"]).double().to(dev)   # noqa: E731
                mu = gt("mu"); W, Wp, U0 = gt("W"), gt("Wp"), gt("Urand")[0]; C = mu.shape[0]; Id = torch.eye(C, dtype=mu.dtype, device=dev)
                ers = {"pre": torch.zeros_like(Id), "tex": gt("At"), "int": gt("Ai"), "r0": Wp @ U0 @ U0.T @ W}
                for nm in PROBES:
                    yk, ek = nm.split("_"); yi = YK.index(yk); Ae = ers[ek]; B_ = Id - Ae; cvec = Ae @ mu; r2s = []
                    for gtr in (0, 1):
                        n, sh, shh, shy, sy, syy = acc[(arm, k, L, gtr)]; m = sh / n; Sig = shh / n - torch.outer(m, m); cy = shy[:, yi] / n - m * (sy[yi] / n)
                        Sp = B_ @ Sig @ B_.T; cp = B_ @ cy; mp = B_ @ m + cvec; w = torch.linalg.solve(Sp + LAM * torch.trace(Sp) / C * Id, cp); b0 = sy[yi] / n - w @ mp
                        n2, sh2, shh2, shy2, sy2, syy2 = acc[(arm, k, L, 1 - gtr)]
                        S1 = B_ @ sh2 + n2 * cvec; S2 = B_ @ shh2 @ B_.T + torch.outer(B_ @ sh2, cvec) + torch.outer(cvec, B_ @ sh2) + n2 * torch.outer(cvec, cvec); Sy_ = B_ @ shy2[:, yi] + cvec * sy2[yi]
                        sse = syy2[yi] - 2 * w @ Sy_ - 2 * b0 * sy2[yi] + w @ S2 @ w + 2 * b0 * (w @ S1) + n2 * b0 ** 2; sst = syy2[yi] - sy2[yi] ** 2 / n2
                        r2s.append(float(1 - sse / sst))
                    rows.append(dict(cell=key, arm=arm, fold=k, loc=L, probe=nm, r2=float(np.mean(r2s)), r2_dir0=r2s[0], r2_dir1=r2s[1]))
    pd.DataFrame(rows).to_csv(OUT / key / "probe_heldout.csv", index=False); print("LEACE_PROBE_HELDOUT_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
def load_all():
    cells = [k for k in CELLS4 if (OUT / k).exists() and list((OUT / k).glob("*_leace.csv"))]
    S = pd.concat([pd.read_csv(f) for k in cells for f in sorted((OUT / k).glob("*_leace.csv"))], ignore_index=True) if cells else pd.DataFrame()
    P = pd.concat([pd.read_csv(f) for k in cells for f in sorted((OUT / k).glob("*_leace_probe.csv"))], ignore_index=True) if cells else pd.DataFrame()
    return cells, S, P


def piv(S, k, arm, var, loc, col="E_target"):
    d = S[(S.kind == "E") & (S.cell == k) & (S.arm == arm) & (S["var"] == var) & (S["loc"] == loc)]
    return d.set_index("case")[col]


def med_iqr(s):
    s = s.dropna()
    return f"{s.median():+.3f} [{s.quantile(0.25):+.3f}, {s.quantile(0.75):+.3f}]" if len(s) else "n/a"


def probe_r2(P):
    """Held-out R^2 per cell x arm x loc x probe: per fold 1 - sum SSE / (sum y^2 - (sum y)^2 / n) over the evaluation cases, then mean over folds."""
    g = P.groupby(["cell", "arm", "loc", "probe", "fold"])[["n", "sse", "sy", "syy"]].sum().reset_index()
    g["r2"] = 1 - g.sse / (g.syy - g.sy ** 2 / g.n); return g.groupby(["cell", "arm", "loc", "probe"]).r2.mean().unstack("probe")


def report():
    cells, S, P = load_all()
    if not cells:
        print("no csv"); return
    ncase = {k: S[(S.cell == k) & (S.kind == "nat")].case.nunique() for k in cells}
    L = ["# LEACE erasure of the local texture-energy subspace (FINDINGS 44, pre-registered: PREREG.md)", "",
         "Cells: " + ", ".join(f"{k} (n = {ncase[k]} evaluation cases)" for k in cells) + f". Fit on the first {N_FIT} sorted cases of each cell, read on the others. "
         f"Arms noise / flat / real (folds {', '.join(map(str, FOLDS))}) + init (the real network re-initialised, its own fit). Concept Z = per-case quantile bins (K = {K}) of the FINDINGS-31 residualised "
         "log fine-texture energy, body-masked average on each stage's grid; control Zc = the same with local intensity I. LEACE eraser per cell x arm x fold x stage, applied at every voxel "
         "(skips s2-s4: the skip path into the decoder only; d1-d3: the decoder stage output). E = J(x) - J(x with the erased activation), J = edema log-odds per voxel, fold-mean; "
         f"positive = the subspace pushed the edema belief UP. Random control: {N_DRAWS} equal-rank whitened-space subspaces (mean and 95th percentile over draws, i.e. about the 2nd-largest draw). "
         "Target = GT edema voxels of the patch; ring = dilation 10 minus target in body; background = body minus dilation 10. Wilcoxon two-sided over cases.", ""]
    # ---- probe gate
    R2 = probe_r2(P); prow = []
    L += ["## Probe-at-chance gate (held-out R^2 on the evaluation cases; probes fitted on the fit cases, refitted after erasure; mean over folds)", "",
          "Gate (pre-registered): texture BIN-INDEX R^2 after texture erasure <= 0.05 x pre or <= 0.02 absolute, every trained arm x stage. Continuous residual Tf, intensity and the random-erasure "
          "(draw 0) probes reported, not gated.", "",
          "| cell | arm | stage | tex bin pre | tex bin after tex | ratio | gate | tex cont pre | after tex | tex bin after random | int bin pre | after int | int cont pre | after int |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ngate, nfail = 0, 0
    for k in cells:
        for arm in ARMS4:
            for loc in STAGES:
                if (k, arm, loc) not in R2.index:
                    continue
                q = R2.loc[(k, arm, loc)]; ratio = q.tb_tex / q.tb_pre if q.tb_pre > 0 else np.nan; ok = (q.tb_tex <= 0.05 * q.tb_pre) or (q.tb_tex <= 0.02)
                if arm != "init":
                    ngate += 1; nfail += int(not ok)
                L.append(f"| {k} | {arm} | {loc} | {q.tb_pre:.3f} | {q.tb_tex:.3f} | {ratio:.3f} | {'pass' if ok else 'FAIL'} | {q.tc_pre:.3f} | {q.tc_tex:.3f} | {q.tb_r0:.3f} | "
                         f"{q.ib_pre:.3f} | {q.ib_int:.3f} | {q.ic_pre:.3f} | {q.ic_int:.3f} |")
                prow.append(dict(cell=k, arm=arm, loc=loc, gate_pass=bool(ok), **{f"r2_{c}": q[c] for c in PROBES}))
    pd.DataFrame(prow).to_csv(OUT / "leace_probe.csv", index=False)
    gate_ok = nfail == 0
    L += ["", f"Gate: {ngate - nfail}/{ngate} trained-arm cells x stages pass" + (" -> LEACE does its job; effects below may be read." if gate_ok else " -> GATE FAILED: effects below are NOT to be read until the fit is fixed."), ""]
    fh = [OUT / k / "probe_heldout.csv" for k in cells if (OUT / k / "probe_heldout.csv").exists()]
    if fh:
        H = pd.concat([pd.read_csv(f_) for f_ in fh], ignore_index=True).groupby(["cell", "arm", "loc", "probe"]).r2.mean().unstack("probe"); hrow = []
        L += ["### Probes refitted on the EVALUATION cases (added after the main run, not pre-registered)", "",
              "The pre-registered gate refits the probe on the fit-case moments, where the bin index is exactly uncorrelated with h' by construction, so it cannot fail. "
              "Here the erasers stay fixed from the fit cases and the ridge probes are refitted on one half of the evaluation cases (sorted-index parity) and scored on the other half "
              "(both directions, mean over directions and folds): does the erased subspace still remove the concept on new cases?", "",
              "| cell | arm | stage | tex bin pre | after tex | after random | tex cont pre | after tex | int bin pre | after int |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k in cells:
            for arm in ARMS4:
                for loc in STAGES:
                    if (k, arm, loc) not in H.index:
                        continue
                    q = H.loc[(k, arm, loc)]
                    L.append(f"| {k} | {arm} | {loc} | {q.tb_pre:.3f} | {q.tb_tex:.3f} | {q.tb_r0:.3f} | {q.tc_pre:.3f} | {q.tc_tex:.3f} | {q.ib_pre:.3f} | {q.ib_int:.3f} |")
                    hrow.append(dict(cell=k, arm=arm, loc=loc, **{f"r2_{c}": q[c] for c in PROBES}))
        Hd = pd.DataFrame(hrow); Hd.to_csv(OUT / "leace_probe_heldout.csv", index=False); tr = Hd[Hd.arm != "init"]
        L += ["", f"Trained arms: texture bin-index R^2 before {tr.r2_tb_pre.min():.3f}-{tr.r2_tb_pre.max():.3f}, after texture erasure {tr.r2_tb_tex.min():.3f}-{tr.r2_tb_tex.max():.3f} "
              f"(max ratio after / before {(tr.r2_tb_tex / tr.r2_tb_pre).max():.3f}), after random erasure {tr.r2_tb_r0.min():.3f}-{tr.r2_tb_r0.max():.3f}; continuous residual Tf "
              f"{tr.r2_tc_pre.min():.3f}-{tr.r2_tc_pre.max():.3f} -> {tr.r2_tc_tex.min():.3f}-{tr.r2_tc_tex.max():.3f}; intensity bin {tr.r2_ib_pre.min():.3f}-{tr.r2_ib_pre.max():.3f} -> "
              f"{tr.r2_ib_int.min():.3f}-{tr.r2_ib_int.max():.3f}.", ""]
    # ---- headline table at d2
    rows = []
    L += [f"## Headline stage {HEAD}: target-mean effect per arm, median [IQR] over cases", "",
          "| cell | arm | E texture | E random (mean of draws) | E random p95 | E intensity | frac target > 1 nat (tex) | ring E tex | background E tex | Dice natural | after tex | after random (draw 0) | after int | rank tex / int |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in cells:
        for arm in ARMS4:
            Et, Er, Ep, Ei = piv(S, k, arm, "tex", HEAD), piv(S, k, arm, "rmean", HEAD), piv(S, k, arm, "rmean", HEAD, "E_target_p95"), piv(S, k, arm, "int", HEAD)
            fr, rg, bgr = piv(S, k, arm, "tex", HEAD, "frac_gt1"), piv(S, k, arm, "tex", HEAD, "E_ring"), piv(S, k, arm, "tex", HEAD, "E_bg")
            dn, dt_, dr_, di_ = piv(S, k, arm, "tex", HEAD, "dice_nat"), piv(S, k, arm, "tex", HEAD, "dice_after"), piv(S, k, arm, "rmean", HEAD, "dice_after"), piv(S, k, arm, "int", HEAD, "dice_after")
            rk = piv(S, k, arm, "tex", HEAD, "rank_tex").median(), piv(S, k, arm, "tex", HEAD, "rank_int").median()
            L.append(f"| {k} | {arm} | {med_iqr(Et)} | {med_iqr(Er)} | {Ep.median():+.3f} | {med_iqr(Ei)} | {fr.median():.3f} | {rg.median():+.3f} | {bgr.median():+.3f} | "
                     f"{dn.median():.3f} | {dt_.median():.3f} | {dr_.median():.3f} | {di_.median():.3f} | {rk[0]:.0f} / {rk[1]:.0f} |")
            rows.append(dict(cell=k, arm=arm, loc=HEAD, n=int(Et.notna().sum()), E_tex=Et.median(), E_tex_q25=Et.quantile(0.25), E_tex_q75=Et.quantile(0.75), E_rand_mean=Er.median(),
                             E_rand_p95=Ep.median(), E_int=Ei.median(), frac_gt1_tex=fr.median(), E_ring_tex=rg.median(), E_bg_tex=bgr.median(), dice_nat=dn.median(), dice_tex=dt_.median(),
                             dice_rand0=dr_.median(), dice_int=di_.median(), rank_tex=rk[0], rank_int=rk[1], mean_dice_nat=dn.mean(), mean_dice_tex=dt_.mean(), mean_dice_rand0=dr_.mean()))
    L.append("")
    # ---- arm contrasts, every stage
    con = {}; crow = []
    L += ["## Arm contrasts on target-mean E texture (paired over cases)", "", "| cell | stage | real - flat median | p | real > flat % | real - noise median | p | flat - noise median | p | real - init median | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in cells:
        for loc in STAGES:
            r_, f_, n_, i_ = (piv(S, k, a, "tex", loc) for a in ("real", "flat", "noise", "init"))   # by name (ARMS order is noise, flat, real)
            drf, drn, dfn, dri = (r_ - f_).dropna(), (r_ - n_).dropna(), (f_ - n_).dropna(), (r_ - i_).dropna(); con[(k, loc)] = (drf, drn)
            L.append(f"| {k} | {loc} | {drf.median():+.3f} | {fp(wp(drf))} | {100 * (drf > 0).mean():.0f} | {drn.median():+.3f} | {fp(wp(drn))} | {dfn.median():+.3f} | {fp(wp(dfn))} | {dri.median():+.3f} | {fp(wp(dri))} |")
            crow.append(dict(cell=k, loc=loc, contrast="real-flat", median=drf.median(), p=wp(drf))); crow.append(dict(cell=k, loc=loc, contrast="real-noise", median=drn.median(), p=wp(drn)))
            crow.append(dict(cell=k, loc=loc, contrast="flat-noise", median=dfn.median(), p=wp(dfn))); crow.append(dict(cell=k, loc=loc, contrast="real-init", median=dri.median(), p=wp(dri)))
    L.append("")
    # ---- controls: texture vs random / intensity per arm
    ctl = {}
    L += ["## Controls: texture erasure vs random-subspace and intensity erasure, per arm (paired over cases)", "",
          "| cell | arm | stage | E tex - E rand p95 median | p | % cases E tex > p95 | E tex - E rand mean | p | E tex - E int | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in cells:
        for arm in ARMS4:
            for loc in STAGES:
                Et, Ep, Er, Ei = piv(S, k, arm, "tex", loc), piv(S, k, arm, "rmean", loc, "E_target_p95"), piv(S, k, arm, "rmean", loc), piv(S, k, arm, "int", loc)
                a1, a2, a3 = (Et - Ep).dropna(), (Et - Er).dropna(), (Et - Ei).dropna(); ctl[(k, arm, loc)] = (a1, a2, a3)
                if loc == HEAD or arm == "real":
                    L.append(f"| {k} | {arm} | {loc} | {a1.median():+.3f} | {fp(wp(a1))} | {100 * (a1 > 0).mean():.0f} | {a2.median():+.3f} | {fp(wp(a2))} | {a3.median():+.3f} | {fp(wp(a3))} |")
                crow.append(dict(cell=k, loc=loc, arm=arm, contrast="tex-rand_p95", median=a1.median(), p=wp(a1), frac_pos=(a1 > 0).mean()))
                crow.append(dict(cell=k, loc=loc, arm=arm, contrast="tex-rand_mean", median=a2.median(), p=wp(a2), frac_pos=(a2 > 0).mean()))
                crow.append(dict(cell=k, loc=loc, arm=arm, contrast="tex-int", median=a3.median(), p=wp(a3), frac_pos=(a3 > 0).mean()))
    pd.DataFrame(crow).to_csv(OUT / "leace_controls.csv", index=False)
    L.append("(rows shown: every arm at the headline stage, the real arm at every stage; all in leace_controls.csv)")
    L.append("")
    # ---- depth
    drows = []
    L += ["## Depth profile: median over cases of target-mean E per stage (tex / random mean / random p95 / intensity)", "", "| cell | arm | " + " | ".join(DEPTH_ORDER) + " |", "|" + "---|" * (2 + len(DEPTH_ORDER))]
    for k in cells:
        for arm in ARMS4:
            cellv = []
            for loc in DEPTH_ORDER:
                Et, Er, Ep, Ei = piv(S, k, arm, "tex", loc), piv(S, k, arm, "rmean", loc), piv(S, k, arm, "rmean", loc, "E_target_p95"), piv(S, k, arm, "int", loc)
                lo, hi = piv(S, k, arm, "rmean", loc, "E_target_p025"), piv(S, k, arm, "rmean", loc, "E_target_p975"); ex = (Et - Er).dropna()
                cellv.append(f"{Et.median():+.2f} / {Er.median():+.2f} / {Ep.median():+.2f} / {Ei.median():+.2f}")
                drows.append(dict(cell=k, arm=arm, loc=loc, E_tex=Et.median(), E_rand_mean=Er.median(), E_rand_p95=Ep.median(), E_rand_p025=lo.median(), E_rand_p975=hi.median(),
                                  E_int=Ei.median(), tex_minus_rand=ex.median(), dice_nat=piv(S, k, arm, "tex", loc, "dice_nat").median(), dice_tex=piv(S, k, arm, "tex", loc, "dice_after").median(),
                                  dice_rand0=piv(S, k, arm, "rmean", loc, "dice_after").median(), dice_int=piv(S, k, arm, "int", loc, "dice_after").median()))
            L.append(f"| {k} | {arm} | " + " | ".join(cellv) + " |")
    D = pd.DataFrame(drows); D.to_csv(OUT / "leace_depth.csv", index=False); L.append("")
    # ---- verdicts
    nc = len(cells); V = []
    rf_ = sum(1 for k in cells if con[(k, HEAD)][0].median() > 0 and wp(con[(k, HEAD)][0]) < 0.01)
    rn_ = sum(1 for k in cells if con[(k, HEAD)][1].median() > 0 and wp(con[(k, HEAD)][1]) < 0.01)
    p1 = rf_ >= 3 and rn_ == nc
    V.append(f"- P1 ({HEAD}; real > flat in >= 3/4, real > noise in 4/4, p < 0.01): {'SUPPORTED' if p1 else 'NOT SUPPORTED (kill condition)'}: real > flat in {rf_}/{nc}, real > noise in {rn_}/{nc} ("
             + "; ".join(f"{k} r-f {con[(k, HEAD)][0].median():+.3f} p {fp(wp(con[(k, HEAD)][0]))}, r-n {con[(k, HEAD)][1].median():+.3f} p {fp(wp(con[(k, HEAD)][1]))}" for k in cells) + ").")
    for loc in STAGES:
        if loc == HEAD:
            continue
        a = sum(1 for k in cells if con[(k, loc)][0].median() > 0 and wp(con[(k, loc)][0]) < 0.01); b_ = sum(1 for k in cells if con[(k, loc)][1].median() > 0 and wp(con[(k, loc)][1]) < 0.01)
        V.append(f"  - same rule at {loc} (not the headline): real > flat {a}/{nc}, real > noise {b_}/{nc}")
    q1 = sum(1 for k in cells if ctl[(k, "real", HEAD)][0].median() > 0 and wp(ctl[(k, "real", HEAD)][0]) < 0.01)
    q2 = sum(1 for k in cells if ctl[(k, "real", HEAD)][2].median() > 0 and wp(ctl[(k, "real", HEAD)][2]) < 0.01)
    V.append(f"- P2 (real arm, {HEAD}): texture > random p95 in {q1}/{nc} (need 4/4), texture > intensity in {q2}/{nc} (need >= 3/4): {'SUPPORTED' if (q1 == nc and q2 >= 3) else 'NOT SUPPORTED'} ("
             + "; ".join(f"{k} tex-p95 {ctl[(k, 'real', HEAD)][0].median():+.3f} p {fp(wp(ctl[(k, 'real', HEAD)][0]))}, tex-int {ctl[(k, 'real', HEAD)][2].median():+.3f} p {fp(wp(ctl[(k, 'real', HEAD)][2]))}" for k in cells) + ").")
    pk = {}
    for k in cells:
        d = D[(D.cell == k) & (D.arm == "real")].set_index("loc"); pk[k] = d.tex_minus_rand.idxmax()
    n3 = sum(1 for k in cells if pk[k] in ("s3", "d1", "d2"))
    V.append(f"- P3 (real-arm texture-specific profile median(E_tex - E_rand_mean) peaks at s3 / d1 / d2 in >= 3/4 cells): {'SUPPORTED' if n3 >= 3 else 'NOT SUPPORTED'} in {n3}/{nc} (peaks: "
             + ", ".join(f"{k} {pk[k]}" for k in cells) + ").")
    V.append("- P4 (readability): judged by eye on leace_examples_<cell>.png, see FINDINGS 44.")
    L += ["## Verdicts (pre-registered rules)", ""] + ([] if gate_ok else ["- WARNING: probe gate failed; verdicts not interpretable."]) + V + [""]
    pd.DataFrame(rows).to_csv(OUT / "leace_cells.csv", index=False)
    (OUT / "leace.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
    summary_figure(cells, S, D)
    print("LEACE_REPORT_DONE", flush=True)


def summary_figure(cells, S, D):
    nc = len(cells); fig, axs = plt.subplots(2, nc, figsize=(4.6 * nc, 7.6), squeeze=False)
    for c, k in enumerate(cells):
        ax = axs[0, c]; xs = np.arange(len(ARMS)); w = 0.26
        for i, arm in enumerate(ARMS):
            Et = piv(S, k, arm, "tex", HEAD).median(); Er = piv(S, k, arm, "rmean", HEAD).median(); Ep = piv(S, k, arm, "rmean", HEAD, "E_target_p95").median(); Ei = piv(S, k, arm, "int", HEAD).median()
            q1, q3 = piv(S, k, arm, "tex", HEAD).quantile(0.25), piv(S, k, arm, "tex", HEAD).quantile(0.75)
            ax.bar(i - w, Et, w, color=PAL[arm], label="texture erasure" if i == 0 else None); ax.errorbar(i - w, Et, yerr=[[Et - q1], [q3 - Et]], color="k", lw=0.8, capsize=2)
            ax.bar(i, Er, w, color="white", edgecolor=PAL[arm], hatch="////", label="random subspace (mean; dash = p95)" if i == 0 else None); ax.plot([i - w / 2, i + w / 2], [Ep, Ep], color="k", lw=1.2)
            ax.bar(i + w, Ei, w, color="white", edgecolor=PAL[arm], hatch="....", label="intensity erasure" if i == 0 else None)
        fl = piv(S, k, "init", "tex", HEAD).median(); ax.axhline(fl, color="0.4", ls=":", lw=1, label="random-init floor (texture erasure)"); ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(ARMS); ax.set_title(f"{k} (n = {S[(S.cell == k) & (S.kind == 'nat')].case.nunique()})\nedema log-odds drop at {HEAD}, median over cases", fontsize=8)
        if c == 0:
            ax.set_ylabel("mean over GT edema voxels (nat)"); ax.legend(fontsize=6, frameon=False)
        ax = axs[1, c]; xo = np.arange(len(DEPTH_ORDER))
        for arm in ARMS4:
            d = D[(D.cell == k) & (D.arm == arm)].set_index("loc").reindex(DEPTH_ORDER)
            if arm == "init":
                ax.plot(xo, d.E_tex, ":", color=PAL4[arm], label="init floor"); continue
            ax.plot(xo, d.E_tex, "-o", color=PAL4[arm], ms=3, label=arm); ax.fill_between(xo, d.E_rand_p025, d.E_rand_p975, color=PAL4[arm], alpha=0.15, lw=0)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(xo); ax.set_xticklabels(DEPTH_ORDER); ax.set_title("depth: texture erasure (line) vs random-subspace band (2.5-97.5 pct of draws)", fontsize=7)
        if c == 0:
            ax.set_ylabel("target-mean E (nat), median over cases"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "leace_summary.png", dpi=150); fig.savefig(OUT / "leace_summary.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------------------- figure
def figure():
    cells, S, P = load_all(); D = pd.read_csv(OUT / "leace_depth.csv") if (OUT / "leace_depth.csv").exists() else None
    OR = np.array([1.0, 0.55, 0.0]); CY = np.array([0.1, 0.8, 1.0]); Pz = 2.3
    for k in cells:
        r_, f_ = piv(S, k, "real", "tex", HEAD), piv(S, k, "flat", "tex", HEAD); drf = (r_ - f_).dropna()
        drf = drf[[(OUT / k / f"{c}_leace_maps.npz").exists() for c in drf.index]]
        if not len(drf):
            continue
        picks = []
        for qv in (0.5, 0.75):
            tq = float(drf.quantile(qv))
            for c in (drf - tq).abs().sort_values().index:
                if c not in picks:
                    picks.append(c); break
        rows_ = [(case, arm) for case in picks for arm in ("real", "flat", "noise")]
        fig = plt.figure(figsize=(Pz * 5 + 4.2, Pz * len(rows_) + 0.9)); gs = fig.add_gridspec(len(rows_), 7, width_ratios=[1, 1, 1, 1, 1, 0.15, 1.8])
        for pi, case in enumerate(picks):
            M = np.load(OUT / k / f"{case}_leace_maps.npz"); im = M["ip"].astype(np.float64); gt = M["gt"].astype(bool); body = im != 0
            if not body.any():
                body = np.ones_like(im, bool)
            lo, hi = np.percentile(im[body], 1), np.percentile(im[body], 99); g0 = np.clip((im - lo) / (hi - lo + 1e-12), 0, 1); g0[~body] = 0; G3 = np.stack([g0] * 3, -1)
            fl = np.abs(M[f"init_E_tex_{HEAD}"].astype(np.float64)) if f"init_E_tex_{HEAD}" in M.files else np.zeros_like(im)
            Es = {a: M[f"{a}_E_tex_{HEAD}"].astype(np.float64) for a in ARMS}; Es = {a: np.where(np.abs(v) > fl, v, 0.0) for a, v in Es.items()}
            s99 = np.percentile(np.abs(np.stack(list(Es.values()))), 99) or 1e-12

            def paint(Mp):
                o = np.clip((np.abs(Mp) / s99) ** 0.6, 0, 1) * 0.9; o[~body] = 0; col = np.where((Mp > 0)[..., None], OR[None, None], CY[None, None])
                return np.clip(G3 * (1 - 0.6 * o[..., None]) + col * o[..., None], 0, 1)

            def predimg(mask, arm):
                c = np.array(matplotlib.colors.to_rgb(PAL[arm])); o = mask.astype(float)[..., None] * 0.45
                return np.clip(G3 * (1 - o) + c[None, None] * o, 0, 1)
            sr = S[(S.cell == k) & (S.case == case)]
            for ai, arm in enumerate(("real", "flat", "noise")):
                rr = 3 * pi + ai; e_ = sr[(sr.arm == arm) & (sr["var"] == "tex") & (sr["loc"] == HEAD)].iloc[0]; e_r = sr[(sr.arm == arm) & (sr["var"] == "rmean") & (sr["loc"] == HEAD)].iloc[0]
                panels = [("image", G3, None), (f"prediction (Dice {e_.dice_nat:.2f})", predimg(M[f"{arm}_pred"], arm), M[f"{arm}_pred"]),
                          (f"after TEXTURE erasure at {HEAD} (Dice {e_.dice_after:.2f})", predimg(M[f"{arm}_pred_tex_{HEAD}"], arm), M[f"{arm}_pred_tex_{HEAD}"]),
                          (f"after RANDOM erasure at {HEAD}, draw 0 (Dice {e_r.dice_after:.2f})", predimg(M[f"{arm}_pred_r0_{HEAD}"], arm), M[f"{arm}_pred_r0_{HEAD}"]),
                          (f"effect E_{HEAD} (target mean {e_.E_target:+.2f} nat;\nrandom {e_r.E_target:+.2f})", paint(Es[arm]), None)]
                for c, (title, arr, pm) in enumerate(panels):
                    ax = fig.add_subplot(gs[rr, c]); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
                    ax.imshow(np.transpose(arr, (1, 0, 2)), origin="lower", interpolation="nearest"); ax.contour(gt.T.astype(float), [0.5], colors="lime", linewidths=0.7)
                    if pm is not None and pm.any():
                        ax.contour(pm.T.astype(float), [0.5], colors=[PAL[arm]], linewidths=0.8)
                    ax.set_title(title, fontsize=6.5)
                    if c == 0:
                        ax.set_ylabel(f"{arm}\n{'typical' if pi == 0 else '75th pct'} case {case}\nreal - flat {drf.get(case, np.nan):+.2f} nat", fontsize=6.5)
        if D is not None:
            ax = fig.add_subplot(gs[:, 6]); xo = np.arange(len(DEPTH_ORDER))
            for arm in ARMS4:
                d = D[(D.cell == k) & (D.arm == arm)].set_index("loc").reindex(DEPTH_ORDER)
                if arm == "init":
                    ax.plot(xo, d.E_tex, ":", color=PAL4[arm], lw=1.5, label="random-init floor"); continue
                ax.plot(xo, d.E_tex, "-o", color=PAL4[arm], ms=4, label=f"{arm}: texture erasure"); ax.fill_between(xo, d.E_rand_p025, d.E_rand_p975, color=PAL4[arm], alpha=0.18, lw=0)
            ax.axhline(0, color="k", lw=0.5); ax.set_xticks(xo); ax.set_xticklabels(DEPTH_ORDER); ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"depth profile, all {S[(S.cell == k) & (S.kind == 'nat')].case.nunique()} evaluation cases:\nmedian target-mean edema log-odds drop;\nshaded = random-subspace erasure band (2.5-97.5 pct of draws)", fontsize=7.5)
            ax.set_ylabel("nat")
        fig.suptitle(f"{k}: erasing the linearly decodable LOCAL TEXTURE-ENERGY subspace inside the network (LEACE at {HEAD}) vs an equal-rank random subspace.\n"
                     "Filled = predicted edema (arm colour), lime = GT edema; effect: orange = the subspace supported the edema call, cyan = opposed (shared scale, p99 over arms; "
                     "zero where |E| <= the random-init network's)", fontsize=8.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"leace_examples_{k}.png", dpi=130); plt.close(fig)
    print("LEACE_FIGURE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--fit", action="store_true"); ap.add_argument("--report", action="store_true"); ap.add_argument("--figure", action="store_true")
    ap.add_argument("--probe-heldout", action="store_true", help="probes refitted on the evaluation cases (added after the main run)")
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
        (fit_cell(key, dev) if args.fit else (probe_heldout_cell(key, dev) if args.probe_heldout else run_cell(key, dev, args.case_slice)))


if __name__ == "__main__":
    main()
