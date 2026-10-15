#!/usr/bin/env python
"""Visual proof v19 (FINDINGS 46, pre-registered in outputs/plots/kspec/PREREG.md): WEIGHT-LEVEL evidence of fine-scale kernel content.

Part A (CPU, --spectra): permutation-invariant 3D DCT-II order spectra of EVERY 3x3x3 / 1x3x3 conv layer (encoder + decoder; seg heads, transposed convs
  and aliases excluded) of the noise (lblvor, rung 4) / flat (rung 4.5) / real (rung 5, val000) arms, folds 0-2, of all 15 training settings behind the
  23-cell roster, plus 3 random-init networks per setting (its own plans, nnU-Net He init, seeds 0-2). Unit-normalised kernels, orthonormal DCT, energy by
  total order o = i + j + l; HOF = energy fraction at o >= 3. -> OUT/kspec_A_layers.csv, OUT/_A/firstconv_kernels.npz
Part B (GPU): channel-selective smoothing of first-stage kernels (conv 1 / conv 2 / both; top-k output channels by HOF; project onto DCT orders 0-1, norm
  kept) vs random-channel smoothing (20 draws) and equal-angle random perturbation of the same kernels (20 draws, k = 8); BraTS 4 cells x 70 cases, folds
  0-2 (fold-mean). --smoke / run --cells --case-slice -> OUT/<cell>/<case>_kspec.csv (+ <case>_kspec_maps.npz: one axial slice, patient data, never commit),
  OUT/<cell>/<case>_blur.csv (input-blur equivalence check, first 5 cases).
--report (CPU): kspec.md, kspec_A_cells.csv, kspec_B_*.csv, kspec_summary_A.png, kspec_summary_B.png, kspec_firstconv_gallery.png.
--figure (CPU): kspec_examples_<cell>.png (patient slices: never commit).
Env: VP_KSPEC_OUT, KS_FOLDS ("0 1 2"), KS_DRAWS (20), KS_KS ("4 8 16"), KS_BATCH (8), KS_NBLUR (5).
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
OUT = Path(os.environ.get("VP_KSPEC_OUT") or (THIS.parent / "outputs/plots/kspec")); OUT.mkdir(parents=True, exist_ok=True)
FOLDS = tuple(int(v) for v in os.environ.get("KS_FOLDS", "0 1 2").split())
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728", "rand": "#9e9e9e"}
SEEDS_RAND = (0, 1, 2); EPS_DEAD = 1e-6
ENC_RE = re.compile(r"^encoder\.stages\.(\d+)\..*convs\.(\d+)\.conv\.weight$"); DEC_RE = re.compile(r"^decoder\.stages\.(\d+)\.convs\.(\d+)\.conv\.weight$")
BRATS = ("glioma_t2w", "glioma_t1n")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hseed(*a):
    return int(hashlib.md5("|".join(map(str, a)).encode()).hexdigest()[:8], 16)


def fp(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.1g}"


# ----------------------------------------------------------------------------------------------------------------------------------- DCT spectra
def order_grid(kshape):
    """total DCT order of every atom of a kernel of spatial shape kshape (size-1 axes contribute index 0)."""
    ix = np.meshgrid(*[np.arange(n) for n in kshape], indexing="ij"); return sum(ix).ravel()


def dct_energy(W):
    """W (N, kz, ky, kx) -> (energy fraction per atom (N, n_atoms), norms (N,)): orthonormal DCT-II per axis of the unit-normalised kernels."""
    from scipy.fft import dctn
    W = np.asarray(W, np.float64); nrm = np.sqrt((W ** 2).reshape(len(W), -1).sum(1))
    C = dctn(W / np.maximum(nrm, 1e-300)[:, None, None, None], type=2, norm="ortho", axes=(1, 2, 3)).reshape(len(W), -1)
    E = C ** 2; return E / np.maximum(E.sum(1, keepdims=True), 1e-300), nrm


def kernel_stats(W):
    """W (N, kz, ky, kx) -> dict per kernel: hof, dc, order profile (N, 7), alive mask."""
    E, nrm = dct_energy(W); og = order_grid(W.shape[1:]); prof = np.zeros((len(W), 7))
    for o in range(7):
        prof[:, o] = E[:, og == o].sum(1)
    alive = nrm >= EPS_DEAD * np.median(nrm)
    return dict(hof=prof[:, 3:].sum(1), dc=prof[:, 0], prof=prof, alive=alive, atoms=E, og=og)


def smooth_kernels(W):
    """Project kernels (N, kz, ky, kx) onto DCT orders 0-1, rescale to the original norm; kernels whose projection is ~0 are zeroed. -> (W', n_zeroed)."""
    from scipy.fft import dctn, idctn
    W = np.asarray(W, np.float64); og = order_grid(W.shape[1:]).reshape(W.shape[1:])
    C = dctn(W, type=2, norm="ortho", axes=(1, 2, 3)); C[:, og >= 2] = 0; Ws = idctn(C, type=2, norm="ortho", axes=(1, 2, 3))
    n0 = np.sqrt((W ** 2).reshape(len(W), -1).sum(1)); n1 = np.sqrt((Ws ** 2).reshape(len(W), -1).sum(1)); z = n1 < 1e-6 * np.maximum(n0, 1e-300)
    Ws = Ws * (n0 / np.maximum(n1, 1e-300))[:, None, None, None]; Ws[z] = 0; return Ws, int(z.sum())


def perturb_kernels(W, Ws, rng):
    """Equal-angle random perturbation: per kernel, theta = angle(k, k_smooth); k'' = |k| (cos theta k^ + sin theta v^), v^ random unit vector orthogonal
    to k^. Same norm and same |k'' - k| as the smoothing, random direction."""
    W = np.asarray(W, np.float64); sh = W.shape; A = W.reshape(sh[0], -1); B = np.asarray(Ws, np.float64).reshape(sh[0], -1)
    na = np.linalg.norm(A, axis=1); u = A / np.maximum(na, 1e-300)[:, None]
    nb = np.linalg.norm(B, axis=1); cos = np.clip((A * B).sum(1) / np.maximum(na * nb, 1e-300), -1, 1); th = np.arccos(cos); th[nb == 0] = np.pi / 2
    v = rng.standard_normal(A.shape); v -= (v * u).sum(1, keepdims=True) * u; v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)
    return (na[:, None] * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)).reshape(sh)


# ----------------------------------------------------------------------------------------------------------------------------------- roster
def settings():
    """setting -> dict(arms: {noise, flat, real} -> model dir, cells: [...]) from the 23-cell roster of FINDINGS 37."""
    import visual_proof_erf_tasks as vt
    import visual_proof_erf_outcome as vo
    S = {}
    for key in vt.CELLS:
        s = "_".join(key.split("_")[:2]); S.setdefault(s, dict(cells=[]))["cells"].append(key)
        if "arms" not in S[s]:
            S[s]["arms"] = vo.models(key)
    return S


def load_sd(path):
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False); sd = ck["network_weights"]
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}, int(ck.get("current_epoch", -1))


def build_net(md, seed):
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    plans = json.load(open(md / "plans.json")); dj = json.load(open(md / "dataset.json")); pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres"); lm = pm.get_label_manager(dj); torch.manual_seed(seed)
    return get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
                                  determine_num_input_channels(pm, cm, dj), lm.num_segmentation_heads, allow_init=True, deep_supervision=False)


def conv_layers(sd):
    """[(name, part, stage, conv, group)] for every encoder / decoder conv with a {1,3}^3 kernel (aliases excluded)."""
    rows = []; encs = [int(m.group(1)) for n in sd if (m := ENC_RE.match(n)) and not n.startswith("decoder.")]; n_enc = max(encs) + 1
    for n, v in sd.items():
        if n.startswith("decoder.encoder.") or ".all_modules." in n or v.ndim != 5 or not all(d in (1, 3) for d in v.shape[2:]) or np.prod(v.shape[2:]) == 1:
            continue
        m = ENC_RE.match(n)
        if m:
            s, j = int(m.group(1)), int(m.group(2)); rows.append((n, "enc", s, j, ("c1" if j == 0 else "c2") if s == 0 else f"E{s}")); continue
        m = DEC_RE.match(n)
        if m:
            s, j = int(m.group(1)), int(m.group(2)); rows.append((n, "dec", s, j, f"D{n_enc - 2 - s}"))
    return sorted(rows, key=lambda r: (r[1] != "enc", r[2], r[3]))


def first_conv(sd):
    L = conv_layers(sd); return [r for r in L if r[4] == "c1"][0][0], [r for r in L if r[4] == "c2"][0][0]


# ----------------------------------------------------------------------------------------------------------------------------------- Part A
def spectra():
    S = settings(); rows, atoms_rand, kern = [], {}, {}
    log(f"Part A: {len(S)} settings: " + ", ".join(f"{s} ({len(v['cells'])} cells)" for s, v in S.items()))
    for s, v in S.items():
        nets = [(arm, k, None) for arm in ARMS for k in FOLDS] + [("rand", sd_, sd_) for sd_ in SEEDS_RAND]; shapes = None
        for arm, k, seed in nets:
            if arm == "rand":
                net = build_net(v["arms"]["real"], seed); sd = {n: t.detach().float() for n, t in net.state_dict().items()}; ep = 0
            else:
                sd, ep = load_sd(v["arms"][arm] / f"fold_{k}" / "checkpoint_best.pth")
            L = conv_layers(sd); shp = [(n, tuple(sd[n].shape)) for n, *_ in L]
            if shapes is None:
                shapes = shp
            assert shp == shapes, (s, arm, k, "architecture differs between arms")
            for li, (n, part, st, j, g) in enumerate(L):
                w = sd[n].float().numpy(); O, I = w.shape[:2]; ks = kernel_stats(w.reshape(O * I, *w.shape[2:])); a = ks["alive"]
                prof = ks["prof"][a].mean(0)
                rows.append(dict(setting=s, arm=arm, fold=k, epoch=ep, layer=li, name=n, part=part, stage=st, conv=j, group=g, kshape="x".join(map(str, w.shape[2:])),
                                 n_atoms=int(np.prod(w.shape[2:])), c_out=O, c_in=I, n_kernels=O * I, n_dead=int((~a).sum()), hof_mean=float(ks["hof"][a].mean()),
                                 hof_median=float(np.median(ks["hof"][a])), dc_mean=float(ks["dc"][a].mean()), one_minus_dc=float(1 - ks["dc"][a].mean()),
                                 **{f"o{o}": float(prof[o]) for o in range(7)}))
                if arm == "rand":
                    key = (rows[-1]["kshape"], g == "c1"); atoms_rand.setdefault(key, []).append(ks["atoms"][a])
                if g == "c1":
                    kern[f"{s}|{arm}|{k}"] = w[:, 0]; kern[f"{s}|{arm}|{k}|hof"] = ks["hof"]
            log(f"{s} {arm} {'seed' if arm == 'rand' else 'fold'} {k}: epoch {ep}, {len(L)} conv layers, c1 HOF {[r['hof_mean'] for r in rows if r['setting'] == s and r['arm'] == arm and r['fold'] == k and r['group'] == 'c1'][0]:.4f}")
    D = pd.DataFrame(rows); D.to_csv(OUT / "kspec_A_layers.csv", index=False)
    # A0 sanity: random-init energy per atom must be flat (1 / n_atoms)
    L0 = ["# A0 random-init flatness (mean energy per atom / (1 / n_atoms); pooled over the 45 random networks)", ""]
    for (ksh, isc1), lst in sorted(atoms_rand.items()):
        E = np.concatenate(lst); m = E.mean(0) * E.shape[1]
        L0.append(f"- kernel {ksh}{' (first conv only)' if isc1 else ''}: {len(E)} kernels, atom energy ratio min {m.min():.3f} max {m.max():.3f} (expected 1.000)")
    (OUT / "_A").mkdir(exist_ok=True); (OUT / "_A" / "A0_flatness.md").write_text("\n".join(L0)); print("\n".join(L0), flush=True)
    np.savez_compressed(OUT / "_A" / "firstconv_kernels.npz", **kern)
    print("KSPEC_SPECTRA_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- Part B
CELLS4 = ("glioma_t2w_t1n", "glioma_t2w_t2f", "glioma_t1n_t2f", "glioma_t1n_t2w")
N_DRAWS = int(os.environ.get("KS_DRAWS", 20)); KS = tuple(int(v) for v in os.environ.get("KS_KS", "4 8 16").split()); HK = 8; HSET = "c12"
SETS = ("c1", "c2", "c12"); BATCH = int(os.environ.get("KS_BATCH", 8)); NBLUR = int(os.environ.get("KS_NBLUR", 5)); EDEMA, RING = 2, 10
SIGMAS = (0.5, 0.75, 1.0, 1.5)


def gauss_kernel(sigma, trunc=None):
    """normalised separable 3D Gaussian, radius ceil(4 sigma) (or trunc)."""
    r = int(np.ceil(4 * sigma)) if trunc is None else trunc; t = np.arange(-r, r + 1); g = np.exp(-t ** 2 / (2 * sigma ** 2)); g /= g.sum()
    return g[:, None, None] * g[None, :, None] * g[None, None, :]


def load_arms(key, dev):
    import visual_proof_v2 as v2
    import visual_proof_erf_outcome as vo
    prs = {a: v2.load_predictor(d, dev) for a, d in vo.models(key).items()}
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    return prs


def stage0(net):
    import torch.nn as nn
    st = net.encoder.stages[0]; scb = st[-1] if isinstance(st, nn.Sequential) else st; blocks = list(scb.convs)
    assert len(blocks) == 2 and all(tuple(b.conv.stride) == (1, 1, 1) for b in blocks), "stage 0 must be 2 stride-1 convs"
    assert getattr(net.encoder, "stem", None) is None
    return blocks


def fwd_variants(net, x, W1, W2):
    """Batched forward with per-variant stage-0 weights. x (1, 1, ...) tensor; W1 (B, 32, 1, *k1), W2 (B, 32, 32, *k2) tensors -> logits (B, C, ...)."""
    import torch
    import torch.nn.functional as Fn
    from visual_proof_switch import decoder_fwd
    b1, b2 = stage0(net); B = W1.shape[0]; C1 = W1.shape[1]; sp = x.shape[2:]
    with torch.no_grad():
        h = Fn.conv3d(x.expand(1, B, *sp), W1.reshape(B * C1, 1, *W1.shape[3:]), None if b1.conv.bias is None else b1.conv.bias.repeat(B),
                      padding=tuple(s // 2 for s in W1.shape[3:]), groups=B).view(B, C1, *sp)
        h = b1.all_modules[1:](h); C2 = W2.shape[1]
        h = Fn.conv3d(h.reshape(1, B * C1, *sp), W2.reshape(B * C2, C1, *W2.shape[3:]), None if b2.conv.bias is None else b2.conv.bias.repeat(B),
                      padding=tuple(s // 2 for s in W2.shape[3:]), groups=B).view(B, C2, *sp)
        h = b2.all_modules[1:](h); skips = [h]
        for st in net.encoder.stages[1:]:
            skips.append(st(skips[-1]))
        return decoder_fwd(net, skips)


def selection(W1, W2):
    """scores and top-k order per output channel (conv 1: kernel HOF; conv 2: mean HOF of its alive input kernels) + smoothed copies."""
    k1 = kernel_stats(W1[:, 0]); s1 = k1["hof"]
    O, I = W2.shape[:2]; k2 = kernel_stats(W2.reshape(O * I, *W2.shape[2:])); h2 = k2["hof"].reshape(O, I); a2 = k2["alive"].reshape(O, I)
    s2 = np.array([h2[o][a2[o]].mean() if a2[o].any() else 0.0 for o in range(O)])
    S1, z1 = smooth_kernels(W1[:, 0]); S2, z2 = smooth_kernels(W2.reshape(O * I, *W2.shape[2:]))
    return dict(s1=s1, s2=s2, o1=np.argsort(-s1, kind="stable"), o2=np.argsort(-s2, kind="stable"), S1=S1[:, None], S2=S2.reshape(W2.shape), z1=z1, z2=z2)


def variant_list(key, arm, kf, sel, W1, W2):
    """[(var, set, k, draw, W1v, W2v)] numpy float64; var in sel / rch / pert / all32."""
    C1, C2 = W1.shape[0], W2.shape[0]; V = []
    for S in SETS:
        for kk in KS:
            c1 = sel["o1"][:kk] if S in ("c1", "c12") else []; c2 = sel["o2"][:kk] if S in ("c2", "c12") else []
            a, b = W1.copy(), W2.copy(); a[c1] = sel["S1"][c1]; b[c2] = sel["S2"][c2]; V.append(("sel", S, kk, -1, a, b))
            rng = np.random.default_rng(hseed(key, arm, kf, S, kk, "rch"))
            for d in range(N_DRAWS):
                r1 = rng.choice(C1, kk, replace=False) if S in ("c1", "c12") else []; r2 = rng.choice(C2, kk, replace=False) if S in ("c2", "c12") else []
                a, b = W1.copy(), W2.copy(); a[r1] = sel["S1"][r1]; b[r2] = sel["S2"][r2]; V.append(("rch", S, kk, d, a, b))
            if kk == HK:
                rng = np.random.default_rng(hseed(key, arm, kf, S, kk, "pert"))
                for d in range(N_DRAWS):
                    a, b = W1.copy(), W2.copy()
                    if len(c1):
                        a[c1] = perturb_kernels(W1[c1], sel["S1"][c1], rng)
                    if len(c2):
                        sh = W2[c2].shape; b[c2] = perturb_kernels(W2[c2].reshape(-1, *sh[2:]), sel["S2"][c2].reshape(-1, *sh[2:]), rng).reshape(sh)
                    V.append(("pert", S, kk, d, a, b))
    V.append(("all32", "c1", 32, -1, sel["S1"].copy(), W2.copy()))
    return V


def run_cell(key, dev, sl=None, smoke=False):
    import torch
    import torch.nn.functional as Fn
    from scipy import ndimage as ndi
    import visual_proof_erf_tasks as vt
    import visual_proof_erf as e7
    from visual_proof_switch import load_case
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.deterministic = True
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    prs = load_arms(key, dev); pr0 = prs["flat"]; allc = sorted(cell["cases"])
    idx = range(len(allc))[slice(*[int(v) for v in sl.split(":")])] if sl else range(len(allc)); cases = [(i, allc[i]) for i in idx]
    # per (arm, fold): base weights, selection, variant weights (CPU float32 tensors), selection log
    BASE, VAR, srows = {}, {}, []
    for arm, pr in prs.items():
        for kf in FOLDS:
            p = pr.list_of_parameters[kf]; n1, n2 = first_conv(p); W1 = p[n1].double().cpu().numpy(); W2 = p[n2].double().cpu().numpy()
            sel = selection(W1, W2); V = variant_list(key, arm, kf, sel, W1, W2); BASE[(arm, kf)] = (n1, n2); VAR[(arm, kf)] = V
            for kk in KS:
                srows.append(dict(cell=key, arm=arm, fold=kf, k=kk, hof_c1_all=float(sel["s1"].mean()), hof_c1_sel=float(sel["s1"][sel["o1"][:kk]].mean()),
                                  hof_c2_all=float(sel["s2"].mean()), hof_c2_sel=float(sel["s2"][sel["o2"][:kk]].mean()), zeroed_c1=sel["z1"], zeroed_c2=sel["z2"],
                                  sel_c1=" ".join(map(str, sel["o1"][:kk])), sel_c2=" ".join(map(str, sel["o2"][:kk]))))
    fs = od / "selection.csv"
    if not fs.exists():
        tmp = od / f".selection_{os.getpid()}.csv"; pd.DataFrame(srows).to_csv(tmp, index=False); tmp.replace(fs)
    nv = len(next(iter(VAR.values()))); log(f"[{key}] {len(cases)} cases, folds {FOLDS}, {nv} variants per arm x fold, batch {BATCH}, draws {N_DRAWS}, ks {KS}")
    nf = float(len(FOLDS))
    for ci, case in cases:
        f_out = od / f"{case}_kspec.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); b = bp.astype(bool)
        tg = gp == EDEMA; dil = ndi.binary_dilation(tg, iterations=RING); ring = dil & ~tg & b; bg = b & ~dil
        z = int(np.argmax(tg.sum((1, 2)))) if tg.any() else ip.shape[0] // 2
        x = torch.from_numpy(ip)[None, None].to(dev); R = {nm: torch.from_numpy(m_).to(dev) for nm, m_ in (("tg", tg), ("ring", ring), ("bg", bg))}
        Rb = torch.from_numpy(b).to(dev)
        base = dict(cell=key, case=case, n_target=int(tg.sum()), n_ring=int(ring.sum()), n_bg=int(bg.sum()), z=z)
        rows, maps = [], dict(z=np.int32(z), ip=ip[z], gt=tg[z]); brows = []

        def rmean(Mp, nm):
            m_ = R[nm]; return float(Mp[m_].mean()) if bool(m_.any()) else np.nan

        def dice_of(P):
            pr_ = P.argmax(0) == EDEMA; g_ = R["tg"]; return float(2 * (pr_ & g_).sum() / (pr_.sum() + g_.sum() + 1e-9))
        for arm, pr in prs.items():
            net = pr.network; accJ = 0; accP = 0; accR = {}; accPv = {}; accE = {}; sc = []; rs = []
            for kf in FOLDS:
                net.load_state_dict(pr.list_of_parameters[kf]); net.eval(); n1, n2 = BASE[(arm, kf)]; p = pr.list_of_parameters[kf]
                W1n = p[n1].float().to(dev)[None]; W2n = p[n2].float().to(dev)[None]
                with torch.no_grad():
                    ref = net(x); ref = ref[0] if isinstance(ref, (list, tuple)) else ref
                lg0 = fwd_variants(net, x, W1n, W2n); sc.append(float((lg0 - ref).abs().max())); J0 = e7.log_odds(lg0.float(), EDEMA)[0]
                accJ = accJ + J0 / nf; P0 = torch.softmax(lg0.float(), 1)[0]; accP = accP + P0 / nf; V = VAR[(arm, kf)]
                for i0 in range(0, len(V), BATCH):
                    ch = V[i0:i0 + BATCH]
                    W1b = torch.from_numpy(np.stack([v[4] for v in ch])).float().to(dev); W2b = torch.from_numpy(np.stack([v[5] for v in ch])).float().to(dev)
                    lgb = fwd_variants(net, x, W1b, W2b); Eb = J0[None] - e7.log_odds(lgb.float(), EDEMA); Pb = torch.softmax(lgb.float(), 1)
                    for jj, (var, S, kk, d, _, _) in enumerate(ch):
                        kv = (var, S, kk, d); Ev = Eb[jj]
                        accR[kv] = accR.get(kv, 0) + np.array([rmean(Ev, "tg"), rmean(Ev, "ring"), rmean(Ev, "bg")]) / nf
                        accPv[kv] = accPv.get(kv, 0) + (Pb[jj] / nf).half()
                        if (S, kk) == (HSET, HK) and var == "sel" or (var, S, kk, d) == ("rch", HSET, HK, 0):
                            accE[kv] = accE.get(kv, 0) + Ev[z].cpu().numpy() / nf
                    del lgb, Eb, Pb, W1b, W2b
                lg1 = fwd_variants(net, x, W1n, W2n); rs.append(float((lg1 - lg0).abs().max())); del lg0, lg1, ref; torch.cuda.empty_cache()
                if ci < NBLUR:
                    brows += blur_check(net, x, p[n1].double().cpu().numpy(), p[n2].float().to(dev), J0, Rb, dev, arm, kf, VAR[(arm, kf)][-1][4])
            dn = dice_of(accP)
            rows.append(dict(base, kind="nat", arm=arm, var="nat", set="", k=0, draw=-1, J_target=rmean(accJ, "tg"), J_ring=rmean(accJ, "ring"), J_bg=rmean(accJ, "bg"),
                             dice=dn, selfcheck=max(sc), restore=max(rs)))
            for kv, e_ in accR.items():
                rows.append(dict(base, kind="E", arm=arm, var=kv[0], set=kv[1], k=kv[2], draw=kv[3], E_target=e_[0], E_ring=e_[1], E_bg=e_[2], dice_nat=dn,
                                 dice_after=dice_of(accPv[kv].float())))
            maps[f"{arm}_pred"] = (accP.argmax(0) == EDEMA)[z].cpu().numpy(); maps[f"{arm}_J"] = accJ[z].cpu().numpy()
            for nm, kv in (("sel", ("sel", HSET, HK, -1)), ("rch0", ("rch", HSET, HK, 0))):
                maps[f"{arm}_pred_{nm}"] = (accPv[kv].float().argmax(0) == EDEMA)[z].cpu().numpy(); maps[f"{arm}_E_{nm}"] = accE[kv]
            del accPv, accP, accJ; torch.cuda.empty_cache()
        np.savez_compressed(od / f"{case}_kspec_maps.npz", **{k_: (v_.astype(np.float16) if isinstance(v_, np.ndarray) and v_.dtype.kind == "f" else v_) for k_, v_ in maps.items()})
        if brows:
            pd.DataFrame([dict(base, **r) for r in brows]).to_csv(od / f"{case}_blur.csv", index=False)
        pd.DataFrame(rows).to_csv(f_out, index=False)
        S_ = pd.DataFrame(rows); s_ = S_[(S_.kind == "E") & (S_["set"] == HSET) & (S_.k == HK)]
        g_ = lambda a, v: s_[(s_.arm == a) & (s_["var"] == v)].E_target.mean()   # noqa: E731
        log(f"[{key}] {case}: {HSET} k{HK} target E sel / rch mean / pert mean: " + "; ".join(f"{a} {g_(a, 'sel'):+.3f} / {g_(a, 'rch'):+.3f} / {g_(a, 'pert'):+.3f}" for a in prs)
            + f"; selfcheck {S_[S_.kind == 'nat'].selfcheck.max():.1e}, restore {S_[S_.kind == 'nat'].restore.max():.1e}; {time.time() - t0:.0f} s")
        if smoke and ci >= (idx.start if sl else 0) + 1:
            break
    print("KSPEC_RUN_DONE", flush=True)


def blur_check(net, x, W1, W2n, J0, Rb, dev, arm, kf, W1dct):
    """Input-blur equivalence (one fold): returns rows with corr of the edema log-odds map over body voxels (and max |dJ| in the interior)."""
    import torch
    import torch.nn.functional as Fn
    import visual_proof_erf as e7
    from scipy.signal import convolve
    out = []; W2b = W2n[None]

    def blur(sig, trunc=None):
        g = torch.from_numpy(gauss_kernel(sig, trunc)).float().to(dev)[None, None]; return Fn.conv3d(x, g, padding=g.shape[-1] // 2)

    def J_of(xin, W1v):
        lg = fwd_variants(net, xin, torch.from_numpy(W1v).float().to(dev)[None], W2b); return e7.log_odds(lg.float(), EDEMA)[0]

    def cmp(Ja, Jb, name, sig):
        a, b_ = Ja[Rb].double(), Jb[Rb].double(); a, b_ = a - a.mean(), b_ - b_.mean(); r = float((a * b_).sum() / ((a.norm() * b_.norm()) + 1e-30))
        inner = (slice(8, -8),) * 3; md = float((Ja - Jb)[inner].abs().max())
        out.append(dict(arm=arm, fold=kf, comparison=name, sigma=sig, rJ=r, maxdiff_interior=md, dJ_body_mean_vs_natural=float((J0 - Ja)[Rb].mean())))
    s = gauss_kernel(0.75, 1); full = np.stack([convolve(W1[c, 0], s, mode="full") for c in range(W1.shape[0])])[:, None]
    xs = blur(0.75, 1); Jb = J_of(xs, W1)
    cmp(J_of(x, full), Jb, "full_support_kxs_vs_blurred_input", 0.75)
    cmp(J_of(x, full[:, :, 1:4, 1:4, 1:4].copy()), Jb, "truncated_kxs_3x3x3_vs_blurred_input", 0.75)
    Jd = J_of(x, W1dct)
    for sig in SIGMAS:
        cmp(Jd, J_of(blur(sig), W1), "dct_order01_all32_vs_blurred_input", sig)
    cmp(Jd, J0, "dct_order01_all32_vs_natural", 0.0)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- report: Part A
GROUP_ORDER = ["c1", "c2", "E1", "E2", "E3", "E4", "E5", "D4", "D3", "D2", "D1", "D0"]


def wil(x, alt="two-sided"):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    try:
        return wilcoxon(x, alternative=alt).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
    except ValueError:
        return np.nan


def a_tables():
    """-> per (setting, arm, fold, group) raw HOF / ratio / 1-DC / median-HOF table G, setting->cells map."""
    D = pd.read_csv(OUT / "kspec_A_layers.csv"); D["fold"] = D["fold"].astype(int)
    rnd = D[D.arm == "rand"].groupby(["setting", "name"])[["hof_mean", "hof_median", "one_minus_dc"]].mean().rename(columns=lambda c: c + "_rand")
    T = D[D.arm != "rand"].join(rnd, on=["setting", "name"])
    T["ratio"] = T.hof_mean / T.hof_mean_rand; T["ratio_med"] = T.hof_median / T.hof_median_rand; T["ratio_1mdc"] = T.one_minus_dc / T.one_minus_dc_rand
    G = T.groupby(["setting", "arm", "fold", "group"])[["hof_mean", "hof_median", "one_minus_dc", "ratio", "ratio_med", "ratio_1mdc", "hof_mean_rand"]].mean().reset_index()
    return D, G


def sep(a, b):
    """all 3 folds of a > all 3 folds of b."""
    return bool(len(a) == 3 and len(b) == 3 and np.min(a) > np.max(b))


def report_A(L):
    from scipy.stats import binomtest
    if not (OUT / "kspec_A_layers.csv").exists():
        L += ["## Part A", "", "kspec_A_layers.csv missing: Part A not run.", ""]; return {}
    import visual_proof_erf_tasks as vt
    D, G = a_tables(); sets = list(dict.fromkeys(D.setting)); ncell = {s: sum(1 for c in vt.CELLS if "_".join(c.split("_")[:2]) == s) for s in sets}
    groups = [g for g in GROUP_ORDER if g in set(G.group)]
    a0 = (OUT / "_A" / "A0_flatness.md").read_text() if (OUT / "_A" / "A0_flatness.md").exists() else "(missing)"
    L += ["## Part A: DCT order spectra of every conv layer, 15 training settings", "",
          f"{len(sets)} settings x arms noise / flat / real x folds {', '.join(map(str, FOLDS))} + 3 random-init networks per setting; "
          f"{D.name.nunique()} distinct layer names; kernel shapes: " + ", ".join(f"{k} ({n} layers)" for k, n in D[D.arm == 'real'].drop_duplicates(['setting', 'name']).kshape.value_counts().items())
          + f". Near-dead kernels skipped: {int(D.n_dead.sum())} of {int(D.n_kernels.sum())}.", "", "### A0 sanity: random-init energy per atom", "", a0.split("\n", 2)[-1], ""]
    cr, rows = {}, []
    for g in groups:
        for s in sets:
            q = G[(G.setting == s) & (G.group == g)]
            if not len(q):
                continue
            v = {a: q[q.arm == a].sort_values("fold") for a in ARMS}
            raw = {a: v[a].hof_mean.to_numpy() for a in ARMS}; rat = {a: v[a].ratio.to_numpy() for a in ARMS}
            row = dict(setting=s, group=g, n_cells=ncell[s], cells=" ".join(c for c in vt.CELLS if "_".join(c.split("_")[:2]) == s), hof_random=float(q.hof_mean_rand.iloc[0]),
                       **{f"hof_{a}_f{k}": float(raw[a][i]) for a in ARMS for i, k in enumerate(FOLDS) if i < len(raw[a])},
                       **{f"hof_{a}": float(raw[a].mean()) for a in ARMS}, **{f"ratio_{a}": float(rat[a].mean()) for a in ARMS},
                       **{f"hofmed_ratio_{a}": float(v[a].ratio_med.mean()) for a in ARMS}, **{f"onemdc_ratio_{a}": float(v[a].ratio_1mdc.mean()) for a in ARMS},
                       sep_real_gt_flat=sep(raw["real"], raw["flat"]), sep_flat_gt_real=sep(raw["flat"], raw["real"]), sep_real_gt_noise=sep(raw["real"], raw["noise"]),
                       sep_noise_gt_real=sep(raw["noise"], raw["real"]), sep_flat_gt_noise=sep(raw["flat"], raw["noise"]), sep_noise_gt_flat=sep(raw["noise"], raw["flat"]))
            rows.append(row)
    C = pd.DataFrame(rows); C.to_csv(OUT / "kspec_A_cells.csv", index=False)
    L += ["### Per stage, across settings (unit = training setting; pooled quantity = HOF / HOF_random of the same layer)", "",
          "delta = fold-mean ratio real - flat per setting. Wilcoxon one-sided (real > flat, pre-registered direction); sign test one-sided. "
          "Separated = 3 real folds > 3 flat folds on raw HOF within the setting (chance 0.75 per stage of 15).", "",
          "| stage | ratio noise / flat / real (median over settings) | real > flat (fold mean) | delta median | Wilcoxon p | sign p | separated real>flat / flat>real | "
          "non-BraTS: real > flat, p | 23-cell weighted real > flat | real > noise | flat > noise | 1-DC real > flat | median-HOF real > flat |", "|" + "---|" * 13]
    for g in groups:
        c = C[C.group == g]
        if not len(c):
            continue
        d = (c.ratio_real - c.ratio_flat).to_numpy(); n = len(d); npos = int((d > 0).sum()); p = wil(d, "greater"); ps = binomtest(npos, n, 0.5, alternative="greater").pvalue
        nb = c[~c.setting.isin(BRATS)]; dnb = (nb.ratio_real - nb.ratio_flat).to_numpy(); w23 = int(((c.ratio_real - c.ratio_flat) > 0).astype(int).mul(c.n_cells).sum())
        drn = (c.ratio_real - c.ratio_noise).to_numpy(); dfn = (c.ratio_flat - c.ratio_noise).to_numpy()
        d1 = (c.onemdc_ratio_real - c.onemdc_ratio_flat).to_numpy(); dm = (c.hofmed_ratio_real - c.hofmed_ratio_flat).to_numpy()
        cr[g] = dict(n=n, npos=npos, p=p, ps=ps, nsep=int(c.sep_real_gt_flat.sum()), nsep_rev=int(c.sep_flat_gt_real.sum()), d=d, pnb=wil(dnb, "greater"), npos_nb=int((dnb > 0).sum()),
                     n_nb=len(dnb), nrn=int((drn > 0).sum()), prn=wil(drn), nfn=int((dfn > 0).sum()), pfn=wil(dfn))
        L.append(f"| {g} | {c.ratio_noise.median():.3f} / {c.ratio_flat.median():.3f} / {c.ratio_real.median():.3f} | {npos}/{n} | {np.median(d):+.4f} | {fp(p)} | {fp(ps)} | "
                 f"{cr[g]['nsep']} / {cr[g]['nsep_rev']} | {cr[g]['npos_nb']}/{cr[g]['n_nb']}, {fp(cr[g]['pnb'])} | {w23}/{int(c.n_cells.sum())} | {cr[g]['nrn']}/{n} (p {fp(cr[g]['prn'])}) | "
                 f"{cr[g]['nfn']}/{n} (p {fp(cr[g]['pfn'])}) | {int((d1 > 0).sum())}/{n} (p {fp(wil(d1, 'greater'))}) | {int((dm > 0).sum())}/{n} (p {fp(wil(dm, 'greater'))}) |")
    L += ["", "### First conv (c1) per setting: raw HOF per fold (random-init mean in brackets)", "",
          "| setting | kernel | random | noise f0/f1/f2 | flat f0/f1/f2 | real f0/f1/f2 | real - flat (fold mean) | separated |", "|---|---|---|---|---|---|---|---|"]
    ksh = D[D.group == "c1"].drop_duplicates("setting").set_index("setting").kshape
    for s in sets:
        r = C[(C.setting == s) & (C.group == "c1")]
        if not len(r):
            continue
        r = r.iloc[0]; f3 = lambda a: "/".join(f"{r[f'hof_{a}_f{k}']:.3f}" for k in FOLDS)   # noqa: E731
        L.append(f"| {s} | {ksh[s]} | {r.hof_random:.3f} | {f3('noise')} | {f3('flat')} | {f3('real')} | {r.hof_real - r.hof_flat:+.4f} | "
                 f"{'real>flat' if r.sep_real_gt_flat else ('flat>real' if r.sep_flat_gt_real else 'overlap')} |")
    # c1 order profile (3x3x3 settings only)
    P = D[(D.group == "c1") & (D.kshape == "3x3x3")]; ocols = [f"o{o}" for o in range(7)]
    L += ["", "### First-conv order profile (energy fraction per DCT order, mean over kernels, folds and the 3x3x3-first-conv settings)", "",
          "| arm | " + " | ".join(f"order {o}" for o in range(7)) + " | HOF |", "|" + "---|" * 9]
    for a in ARMS + ("rand",):
        q = P[P.arm == a][ocols].mean(); L.append(f"| {a} | " + " | ".join(f"{v:.3f}" for v in q) + f" | {q[3:].sum():.3f} |")
    L.append("| iid expectation | " + " | ".join(f"{c / 27:.3f}" for c in (1, 3, 6, 7, 6, 3, 1)) + f" | {17 / 27:.3f} |")
    L.append("")
    return dict(C=C, cr=cr, groups=groups, D=D, G=G)


# ----------------------------------------------------------------------------------------------------------------------------------- report: Part B
def load_B():
    cells = [k for k in CELLS4 if (OUT / k).exists() and list((OUT / k).glob("*_kspec.csv"))]
    S = pd.concat([pd.read_csv(f) for k in cells for f in sorted((OUT / k).glob("*_kspec.csv"))], ignore_index=True) if cells else pd.DataFrame()
    Bl = pd.concat([pd.read_csv(f) for k in cells for f in sorted((OUT / k).glob("*_blur.csv"))], ignore_index=True) if cells else pd.DataFrame()
    Se = pd.concat([pd.read_csv(OUT / k / "selection.csv") for k in cells if (OUT / k / "selection.csv").exists()], ignore_index=True) if cells else pd.DataFrame()
    return cells, S, Bl, Se


def bser(S, c, arm, var, st, kk, col="E_target"):
    """per-case series; random draws averaged over draws."""
    d = S[(S.kind == "E") & (S.cell == c) & (S.arm == arm) & (S["var"] == var) & (S["set"] == st) & (S.k == kk)]
    return d.groupby("case")[col].mean()


def bq(S, c, arm, var, st, kk, q, col="E_target"):
    d = S[(S.kind == "E") & (S.cell == c) & (S.arm == arm) & (S["var"] == var) & (S["set"] == st) & (S.k == kk)]
    return d.groupby("case")[col].quantile(q)


def Dch(S, c, arm, st, kk):
    return (bser(S, c, arm, "sel", st, kk) - bser(S, c, arm, "rch", st, kk)).dropna()


def Dpert(S, c, arm, st, kk):
    return (bser(S, c, arm, "sel", st, kk) - bser(S, c, arm, "pert", st, kk)).dropna()


def ddrop(S, c, arm, var, st, kk):
    """per-case Dice drop (natural - after), random draws averaged."""
    return (bser(S, c, arm, var, st, kk, "dice_nat") - bser(S, c, arm, var, st, kk, "dice_after")).dropna()


def mi(s):
    s = s.dropna(); return f"{s.median():+.3f} [{s.quantile(0.25):+.3f}, {s.quantile(0.75):+.3f}]" if len(s) else "n/a"


def report_B(L):
    cells, S, Bl, Se = load_B()
    if not cells:
        L += ["## Part B", "", "no Part B csv yet.", ""]; return {}
    nc = len(cells); nat = S[S.kind == "nat"]; ncase = {c: nat[nat.cell == c].case.nunique() for c in cells}
    L += ["## Part B: channel-selective first-stage kernel smoothing (BraTS)", "",
          "Cells: " + ", ".join(f"{c} (n = {ncase[c]})" for c in cells) + f". Arms noise / flat / real, folds {', '.join(map(str, FOLDS))} (fold-mean). Selection from each model's own "
          f"weights (conv 1: kernel HOF; conv 2: mean HOF of the output channel's input kernels); SMOOTH = projection onto DCT orders 0-1, norm kept. E = J(natural) - J(modified), "
          f"J = edema log-odds, mean over GT edema; D_ch = E_sel - mean of {N_DRAWS} random-channel draws; D_pert = E_sel - mean of {N_DRAWS} equal-angle random perturbations of the "
          f"same kernels. Headline {HSET}, k = {HK}. Wilcoxon two-sided over cases.", ""]
    L += ["### Plumbing gates", "", f"(a) manual batched stage-0 forward with unmodified weights vs the network's own forward: max |logit diff| = {nat.selfcheck.max():.2e} "
          f"(gate < 1e-3: {'pass' if nat.selfcheck.max() < 1e-3 else 'FAIL'}); restore check after all variants of a fold: max |diff| = {nat.restore.max():.2e}.", "",
          "(b) all 32 conv-1 kernels smoothed (orders 0-1), median over cases:", "", "| cell | arm | E target | Dice natural | Dice after |", "|---|---|---|---|---|"]
    for c in cells:
        for a in ARMS:
            e_ = bser(S, c, a, "all32", "c1", 32); dn = bser(S, c, a, "all32", "c1", 32, "dice_nat"); da = bser(S, c, a, "all32", "c1", 32, "dice_after")
            L.append(f"| {c} | {a} | {e_.median():+.3f} | {dn.median():.3f} | {da.median():.3f} |")
    blur = {}
    if len(Bl):
        L += ["", "### Input-blur equivalence check (first 5 sorted cases per cell; corr of the edema log-odds map over body voxels, per fold; median [min, max] over cases x folds)", "",
              "| comparison | sigma | " + " | ".join(f"{a}" for a in ARMS) + " | max interior |dJ| (median) |", "|---|---|---|---|---|---|"]
        for (cmpn, sig), q in Bl.groupby(["comparison", "sigma"], sort=False):
            blur[(cmpn, sig)] = q.rJ.median()
            L.append(f"| {cmpn} | {sig} | " + " | ".join(f"{q[q.arm == a].rJ.median():.4f} [{q[q.arm == a].rJ.min():.4f}, {q[q.arm == a].rJ.max():.4f}]" for a in ARMS)
                     + f" | {q.maxdiff_interior.median():.2e} |")
    if len(Se):
        L += ["", "### Selection (mean over folds): HOF of selected vs all output channels; kernels zeroed by the smoothing guard", "",
              "| cell | arm | k | conv1 HOF sel / all | conv2 mean HOF sel / all | zeroed c1 / c2 |", "|---|---|---|---|---|---|"]
        for (c, a, kk), q in Se.groupby(["cell", "arm", "k"], sort=False):
            L.append(f"| {c} | {a} | {kk} | {q.hof_c1_sel.mean():.3f} / {q.hof_c1_all.mean():.3f} | {q.hof_c2_sel.mean():.3f} / {q.hof_c2_all.mean():.3f} | {q.zeroed_c1.sum()} / {q.zeroed_c2.sum()} |")
    # headline
    rows = []
    L += ["", f"### Headline ({HSET}, k = {HK}): median [IQR] over cases", "",
          "| cell | arm | E sel | E random-ch (mean) | D_ch | E pert (mean) | D_pert | Dice nat | after sel | after random-ch | after pert | E ring sel | E bg sel |", "|" + "---|" * 13]
    for c in cells:
        for a in ARMS:
            Es, Er, Ep = (bser(S, c, a, v, HSET, HK) for v in ("sel", "rch", "pert")); Dc, Dp = Dch(S, c, a, HSET, HK), Dpert(S, c, a, HSET, HK)
            dn = bser(S, c, a, "sel", HSET, HK, "dice_nat"); ds, dr, dpp = (bser(S, c, a, v, HSET, HK, "dice_after") for v in ("sel", "rch", "pert"))
            ers, ebs = bser(S, c, a, "sel", HSET, HK, "E_ring"), bser(S, c, a, "sel", HSET, HK, "E_bg")
            L.append(f"| {c} | {a} | {mi(Es)} | {mi(Er)} | {mi(Dc)} | {mi(Ep)} | {mi(Dp)} | {dn.median():.3f} | {ds.median():.3f} | {dr.median():.3f} | {dpp.median():.3f} | "
                     f"{ers.median():+.3f} | {ebs.median():+.3f} |")
            rows.append(dict(cell=c, arm=a, set=HSET, k=HK, n=int(Es.notna().sum()), E_sel=Es.median(), E_rch=Er.median(), E_pert=Ep.median(), D_ch=Dc.median(), D_ch_q25=Dc.quantile(0.25),
                             D_ch_q75=Dc.quantile(0.75), p_D_ch=wil(Dc), D_pert=Dp.median(), D_pert_q25=Dp.quantile(0.25), D_pert_q75=Dp.quantile(0.75), p_D_pert=wil(Dp),
                             dice_nat=dn.median(), dice_sel=ds.median(), dice_rch=dr.median(), dice_pert=dpp.median(), E_ring_sel=ers.median(), E_bg_sel=ebs.median()))
    pd.DataFrame(rows).to_csv(OUT / "kspec_B_cells.csv", index=False)
    # k / set profile
    krows = []
    for c in cells:
        for a in ARMS:
            for st in SETS:
                for kk in KS:
                    Es = bser(S, c, a, "sel", st, kk)
                    if not len(Es):
                        continue
                    Dc = Dch(S, c, a, st, kk); Dp = Dpert(S, c, a, st, kk) if kk == HK else pd.Series(dtype=float)
                    krows.append(dict(cell=c, arm=a, set=st, k=kk, n=len(Es), E_sel=Es.median(), E_rch=bser(S, c, a, "rch", st, kk).median(),
                                      E_rch_p025=bq(S, c, a, "rch", st, kk, 0.025).median(), E_rch_p975=bq(S, c, a, "rch", st, kk, 0.975).median(),
                                      D_ch=Dc.median(), p_D_ch=wil(Dc), D_pert=Dp.median() if len(Dp) else np.nan, p_D_pert=wil(Dp) if len(Dp) else np.nan,
                                      dice_nat=bser(S, c, a, "sel", st, kk, "dice_nat").median(), dice_sel=bser(S, c, a, "sel", st, kk, "dice_after").median(),
                                      dice_rch=bser(S, c, a, "rch", st, kk, "dice_after").median()))
    K = pd.DataFrame(krows); K.to_csv(OUT / "kspec_B_k.csv", index=False)
    L += ["", "### D_ch (median over cases) per conv set and k; Dice after selective smoothing in brackets", "",
          "| cell | arm | " + " | ".join(f"{st} k={kk}" for st in SETS for kk in KS) + " |", "|" + "---|" * (2 + len(SETS) * len(KS))]
    for c in cells:
        for a in ARMS:
            q = K[(K.cell == c) & (K.arm == a)].set_index(["set", "k"])
            L.append(f"| {c} | {a} | " + " | ".join(f"{q.loc[(st, kk)].D_ch:+.3f} ({q.loc[(st, kk)].dice_sel:.2f})" if (st, kk) in q.index else "n/a" for st in SETS for kk in KS) + " |")
    # arm contrasts and controls
    crow, con = [], {}
    L += ["", "### Arm contrasts on D_ch (paired over cases) and per-arm controls", "",
          "| cell | set | k | real - flat median | p | real > flat % | real - noise | p | flat - noise | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for st in SETS:
            for kk in KS:
                Ds = {a: Dch(S, c, a, st, kk) for a in ARMS}; rf, rn, fn = (Ds["real"] - Ds["flat"]).dropna(), (Ds["real"] - Ds["noise"]).dropna(), (Ds["flat"] - Ds["noise"]).dropna()
                con[(c, st, kk)] = (rf, rn)
                for nm, s_ in (("D_ch real-flat", rf), ("D_ch real-noise", rn), ("D_ch flat-noise", fn)):
                    crow.append(dict(cell=c, set=st, k=kk, arm="", contrast=nm, median=s_.median(), p=wil(s_), frac_pos=(s_ > 0).mean(), n=len(s_)))
                if (st == HSET) or kk == HK:
                    L.append(f"| {c} | {st} | {kk} | {rf.median():+.3f} | {fp(wil(rf))} | {100 * (rf > 0).mean():.0f} | {rn.median():+.3f} | {fp(wil(rn))} | {fn.median():+.3f} | {fp(wil(fn))} |")
        for a in ARMS:
            for st in SETS:
                for kk in KS:
                    s1 = Dch(S, c, a, st, kk); crow.append(dict(cell=c, set=st, k=kk, arm=a, contrast="D_ch (sel - random-ch)", median=s1.median(), p=wil(s1), frac_pos=(s1 > 0).mean(), n=len(s1)))
                    s3 = (ddrop(S, c, a, "sel", st, kk) - ddrop(S, c, a, "rch", st, kk)).dropna()
                    crow.append(dict(cell=c, set=st, k=kk, arm=a, contrast="Dice drop sel - random-ch", median=s3.median(), p=wil(s3), frac_pos=(s3 > 0).mean(), n=len(s3)))
                    if kk == HK:
                        s2 = Dpert(S, c, a, st, kk); crow.append(dict(cell=c, set=st, k=kk, arm=a, contrast="D_pert (sel - pert)", median=s2.median(), p=wil(s2), frac_pos=(s2 > 0).mean(), n=len(s2)))
    CT = pd.DataFrame(crow); CT.to_csv(OUT / "kspec_B_controls.csv", index=False)
    L += ["", f"(rows: every set at k = {HK} and every k for {HSET}; everything in kspec_B_controls.csv)", ""]
    # verdicts B
    V = []; hk = (HSET, HK)
    rf_ = [c for c in cells if con[(c,) + hk][0].median() > 0 and wil(con[(c,) + hk][0]) < 0.01]; rn_ = [c for c in cells if con[(c,) + hk][1].median() > 0 and wil(con[(c,) + hk][1]) < 0.01]
    V.append(f"- PB1 ({HSET}, k = {HK}; D_ch real > flat, p < 0.01, >= 3/4 cells): {'SUPPORTED' if len(rf_) >= 3 else 'NOT SUPPORTED (kill condition)'}: {len(rf_)}/{nc} ("
             + "; ".join(f"{c} r-f {con[(c,) + hk][0].median():+.3f} p {fp(wil(con[(c,) + hk][0]))}" for c in cells) + f"). Real > noise (reported): {len(rn_)}/{nc} ("
             + "; ".join(f"{c} r-n {con[(c,) + hk][1].median():+.3f} p {fp(wil(con[(c,) + hk][1]))}" for c in cells) + ").")
    for st in SETS:
        for kk in KS:
            if (st, kk) != hk:
                a = sum(1 for c in cells if con[(c, st, kk)][0].median() > 0 and wil(con[(c, st, kk)][0]) < 0.01); V.append(f"  - same rule at {st}, k = {kk}: real > flat {a}/{nc}")
    p2 = {c: Dpert(S, c, "real", HSET, HK) for c in cells}; q2 = [c for c in cells if p2[c].median() > 0 and wil(p2[c]) < 0.01]
    V.append(f"- PB2 (real arm, E_sel > equal-angle perturbation, >= 3/4 cells): {'SUPPORTED' if len(q2) >= 3 else 'NOT SUPPORTED'}: {len(q2)}/{nc} ("
             + "; ".join(f"{c} {p2[c].median():+.3f} p {fp(wil(p2[c]))}" for c in cells) + ").")
    p3 = {c: (ddrop(S, c, "real", "sel", HSET, HK) - ddrop(S, c, "real", "rch", HSET, HK)).dropna() for c in cells}; q3 = [c for c in cells if p3[c].median() > 0 and wil(p3[c]) < 0.01]
    V.append(f"- PB3 (real arm, Dice drop selective > random-channel, >= 3/4 cells): {'SUPPORTED' if len(q3) >= 3 else 'NOT SUPPORTED'}: {len(q3)}/{nc} ("
             + "; ".join(f"{c} {p3[c].median():+.3f} (mean {p3[c].mean():+.4f}) p {fp(wil(p3[c]))}" for c in cells) + ").")
    V.append("- PB4: by eye on kspec_examples_<cell>.png (FINDINGS 46).")
    return dict(cells=cells, S=S, K=K, V=V, blur=blur)


def report():
    L = ["# Visual proof v19: weight-level evidence of fine-scale kernel content (FINDINGS 46; pre-registered: PREREG.md)", ""]
    A = report_A(L); B = report_B(L)
    V = []
    if A:
        cr = A["cr"]

        def rule(g):
            return g in cr and cr[g]["npos"] >= int(np.ceil(2 * cr[g]["n"] / 3)) and np.isfinite(cr[g]["p"]) and cr[g]["p"] < 0.01
        V.append(f"- PA1 (c1: real > flat by fold mean in >= 2/3 of settings AND Wilcoxon p < 0.01): {'SUPPORTED' if rule('c1') else 'NOT SUPPORTED (kill condition)'}: "
                 + (f"{cr['c1']['npos']}/{cr['c1']['n']}, p {fp(cr['c1']['p'])}, separated {cr['c1']['nsep']} (flat > real {cr['c1']['nsep_rev']}); non-BraTS {cr['c1']['npos_nb']}/{cr['c1']['n_nb']} p {fp(cr['c1']['pnb'])}." if 'c1' in cr else "n/a"))
        V.append(f"- PA2 (c2 AND E1 by the same rule): {'SUPPORTED' if rule('c2') and rule('E1') else 'NOT SUPPORTED'}: " + "; ".join(
            f"{g} {cr[g]['npos']}/{cr[g]['n']} p {fp(cr[g]['p'])} separated {cr[g]['nsep']}" for g in ("c2", "E1") if g in cr) + ".")
        V.append("- PA3 (exploratory, decoder): " + "; ".join(f"{g} real > flat {cr[g]['npos']}/{cr[g]['n']} p {fp(cr[g]['p'])}" for g in A["groups"] if g.startswith("D")) + ".")
        V.append("- PA4 (noise arm, reported): " + "; ".join(f"{g} real > noise {cr[g]['nrn']}/{cr[g]['n']} (p {fp(cr[g]['prn'])}), flat > noise {cr[g]['nfn']}/{cr[g]['n']} (p {fp(cr[g]['pfn'])})"
                                                        for g in ("c1", "c2", "E1") if g in cr) + ".")
    if B:
        V += B["V"]
    L += ["## Verdicts (pre-registered rules)", ""] + V + [""]
    (OUT / "kspec.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
    if A:
        summary_A(A); gallery()
    if B:
        summary_B(B)
    print("KSPEC_REPORT_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- figures
def summary_A(A):
    C, groups, D = A["C"], A["groups"], A["D"]; fig, axs = plt.subplots(1, 3, figsize=(19, 5), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.9])); x = np.arange(len(groups))
    ax = axs[0]
    for i, a in enumerate(ARMS):
        med = [C[C.group == g][f"ratio_{a}"].median() for g in groups]; q1 = [C[C.group == g][f"ratio_{a}"].quantile(0.25) for g in groups]
        q3 = [C[C.group == g][f"ratio_{a}"].quantile(0.75) for g in groups]
        ax.errorbar(x + (i - 1) * 0.2, med, yerr=[np.subtract(med, q1), np.subtract(q3, med)], fmt="o", color=PAL[a], ms=4, capsize=2, label=f"{a} (median, IQR over settings)")
    ax.axhline(1, color=PAL["rand"], ls="--", lw=1, label="random init (= 1)"); ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("HOF / HOF_random (same layer)")
    ax.set_title("High-order (DCT order >= 3) kernel energy per stage, trained / random init\n15 training settings, fold-mean per setting", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axs[1]; rng = np.random.default_rng(0)
    for j, g in enumerate(groups):
        c = C[C.group == g]; d = (c.ratio_real - c.ratio_flat).to_numpy(); s_ = c.sep_real_gt_flat.to_numpy(); r_ = c.sep_flat_gt_real.to_numpy(); jx = j + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(jx[~(s_ | r_)], d[~(s_ | r_)], s=12, color="0.55", zorder=2); ax.scatter(jx[s_], d[s_], s=18, color=PAL["real"], zorder=3); ax.scatter(jx[r_], d[r_], s=18, color=PAL["flat"], zorder=3)
        ax.plot([j - 0.3, j + 0.3], [np.median(d)] * 2, color="k", lw=1.5)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("real - flat (fold-mean ratio)")
    ax.set_title("Per-setting real - flat; red = folds separated real > flat, blue = flat > real,\ngrey = overlap; bar = median over settings", fontsize=9)
    ax = axs[2]; P = D[(D.group == "c1") & (D.kshape == "3x3x3")]; oc = [f"o{o}" for o in range(7)]
    for a in ARMS + ("rand",):
        ax.plot(range(7), P[P.arm == a][oc].mean().to_numpy(), "-o", color=PAL[a], ms=4, label=a)
    ax.plot(range(7), np.array([1, 3, 6, 7, 6, 3, 1]) / 27, "k:", lw=1, label="iid expectation"); ax.set_xlabel("DCT total order"); ax.set_ylabel("energy fraction")
    ax.set_title("First conv (32 kernels x 1 input), mean order profile\n(settings with 3x3x3 first conv, folds pooled)", fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "kspec_summary_A.png", dpi=150); fig.savefig(OUT / "kspec_summary_A.pdf"); plt.close(fig)


def gallery():
    """First-conv kernels (fold 0 / random seed 0) of the two BraTS settings, one 4 x 8 mosaic per arm, kernels sorted by HOF (high -> low, row-major);
    each kernel = its 3 axial slices side by side, scaled to its own max |w|. Weights only (no patient data)."""
    f = OUT / "_A" / "firstconv_kernels.npz"
    if not f.exists():
        return
    Z = np.load(f); arms = ARMS + ("rand",); fig, axs = plt.subplots(len(BRATS), len(arms), figsize=(4.6 * len(arms), 3.0 * len(BRATS) + 0.8), squeeze=False)
    for r, s in enumerate(BRATS):
        for c, a in enumerate(arms):
            W = Z[f"{s}|{a}|0"]; h = Z[f"{s}|{a}|0|hof"]; o = np.argsort(-h); kz, ky, kx = W.shape[1:]; th, tw = ky + 1, kz * (kx + 1) + 1
            M = np.full((4 * th + 1, 8 * tw + 1), np.nan)
            for i, ki in enumerate(o):
                w = W[ki] / (np.abs(W[ki]).max() + 1e-12); y0, x0 = 1 + (i // 8) * th, 1 + (i % 8) * tw
                for z in range(kz):
                    M[y0:y0 + ky, x0 + z * (kx + 1):x0 + z * (kx + 1) + kx] = w[z]
            ax = axs[r, c]; ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"); ax.set_facecolor("0.85"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{s} {a}: mean HOF {h.mean():.3f}\n(top-left {h[o[0]]:.2f} -> bottom-right {h[o[-1]]:.2f})", fontsize=8, color=PAL[a])
    fig.suptitle("First-conv 3x3x3 kernels (fold 0; random = seed 0), each shown as its 3 axial slices side by side, scaled to its own max |w| (red +, blue -);\n"
                 "32 kernels per panel sorted by high-order DCT energy fraction (order >= 3), row-major", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT / "kspec_firstconv_gallery.png", dpi=150); plt.close(fig)


def summary_B(B):
    cells, S, K = B["cells"], B["S"], B["K"]; nc = len(cells); fig, axs = plt.subplots(2, nc, figsize=(4.8 * nc, 8.5), squeeze=False)
    for ci, c in enumerate(cells):
        ax = axs[0, ci]; w = 0.36
        for i, a in enumerate(ARMS):
            Dc, Dp = Dch(S, c, a, HSET, HK), Dpert(S, c, a, HSET, HK)
            ax.bar(i - w / 2, Dc.median(), w, color=PAL[a], label="selective - random channels" if i == 0 else None)
            ax.errorbar(i - w / 2, Dc.median(), yerr=[[Dc.median() - Dc.quantile(0.25)], [Dc.quantile(0.75) - Dc.median()]], color="k", lw=0.8, capsize=2)
            ax.bar(i + w / 2, Dp.median(), w, color="white", edgecolor=PAL[a], hatch="....", label="selective - equal-angle perturbation" if i == 0 else None)
            ax.errorbar(i + w / 2, Dp.median(), yerr=[[Dp.median() - Dp.quantile(0.25)], [Dp.quantile(0.75) - Dp.median()]], color="k", lw=0.8, capsize=2)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"{c} (n = {S[(S.cell == c) & (S.kind == 'nat')].case.nunique()})\n{HSET}, k = {HK}: edema log-odds drop, selective minus control", fontsize=8)
        if ci == 0:
            ax.set_ylabel("nat, mean over GT edema; median [IQR] over cases"); ax.legend(fontsize=6, frameon=False)
        ax = axs[1, ci]
        for a in ARMS:
            d = K[(K.cell == c) & (K.arm == a) & (K["set"] == HSET)].set_index("k").reindex(list(KS))
            ax.plot(KS, d.E_sel, "-o", color=PAL[a], ms=3, label=f"{a}: selective"); ax.fill_between(KS, d.E_rch_p025, d.E_rch_p975, color=PAL[a], alpha=0.15, lw=0)
            ax.plot(KS, d.E_rch, ":", color=PAL[a], lw=1)
        ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.set_xlabel("k smoothed output channels per conv")
        ax.set_title(f"{HSET}: target E vs k (line = selective, dotted = random-channel mean,\nshaded = random-channel 2.5-97.5 pct of draws), median over cases", fontsize=7)
        if ci == 0:
            ax.set_ylabel("target E (nat)"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "kspec_summary_B.png", dpi=150); fig.savefig(OUT / "kspec_summary_B.pdf"); plt.close(fig)


def figure():
    cells, S, _, _ = load_B()
    K = pd.read_csv(OUT / "kspec_B_k.csv") if (OUT / "kspec_B_k.csv").exists() else None
    D = pd.read_csv(OUT / "kspec_A_layers.csv") if (OUT / "kspec_A_layers.csv").exists() else None
    OR = np.array([1.0, 0.55, 0.0]); CY = np.array([0.1, 0.8, 1.0]); Pz = 2.3
    for c in cells:
        drf = (Dch(S, c, "real", HSET, HK) - Dch(S, c, "flat", HSET, HK)).dropna(); drf = drf[[(OUT / c / f"{q}_kspec_maps.npz").exists() for q in drf.index]]
        if not len(drf):
            continue
        picks = []
        for qv in (0.5, 0.75):
            tq = float(drf.quantile(qv))
            for q in (drf - tq).abs().sort_values().index:
                if q not in picks:
                    picks.append(q); break
        rws = [(case, a) for case in picks for a in ("real", "flat", "noise")]
        fig = plt.figure(figsize=(Pz * 5 + 4.6, Pz * len(rws) + 0.9)); gs = fig.add_gridspec(len(rws), 7, width_ratios=[1, 1, 1, 1, 1, 0.15, 1.9])
        for pi, case in enumerate(picks):
            M = np.load(OUT / c / f"{case}_kspec_maps.npz"); im = M["ip"].astype(np.float64); gt = M["gt"].astype(bool); body = im != 0
            if not body.any():
                body = np.ones_like(im, bool)
            lo, hi = np.percentile(im[body], 1), np.percentile(im[body], 99); g0 = np.clip((im - lo) / (hi - lo + 1e-12), 0, 1); g0[~body] = 0; G3 = np.stack([g0] * 3, -1)
            Es = {a: M[f"{a}_E_sel"].astype(np.float64) for a in ARMS}; s99 = np.percentile(np.abs(np.stack(list(Es.values()))), 99) or 1e-12

            def paint(Mp):
                o = np.clip((np.abs(Mp) / s99) ** 0.6, 0, 1) * 0.9; o[~body] = 0; col = np.where((Mp > 0)[..., None], OR[None, None], CY[None, None])
                return np.clip(G3 * (1 - 0.6 * o[..., None]) + col * o[..., None], 0, 1)

            def predimg(mask, a):
                cc = np.array(matplotlib.colors.to_rgb(PAL[a])); o = mask.astype(float)[..., None] * 0.45
                return np.clip(G3 * (1 - o) + cc[None, None] * o, 0, 1)
            sr = S[(S.cell == c) & (S.case == case) & (S.kind == "E") & (S["set"] == HSET) & (S.k == HK)]
            for ai, a in enumerate(("real", "flat", "noise")):
                rr = 3 * pi + ai; es = sr[(sr.arm == a) & (sr["var"] == "sel")].iloc[0]; er = sr[(sr.arm == a) & (sr["var"] == "rch")]; er0 = er[er.draw == 0].iloc[0]
                panels = [("image", G3, None), (f"prediction (Dice {es.dice_nat:.2f})", predimg(M[f"{a}_pred"], a), M[f"{a}_pred"]),
                          (f"after SELECTIVE smoothing\n{HSET}, k={HK} (Dice {es.dice_after:.2f})", predimg(M[f"{a}_pred_sel"], a), M[f"{a}_pred_sel"]),
                          (f"after RANDOM-CHANNEL smoothing\n(draw 0; Dice {er0.dice_after:.2f})", predimg(M[f"{a}_pred_rch0"], a), M[f"{a}_pred_rch0"]),
                          (f"effect E (target mean {es.E_target:+.2f} nat;\nrandom-channel mean {er.E_target.mean():+.2f})", paint(Es[a]), None)]
                for cc_, (title, arr, pm) in enumerate(panels):
                    ax = fig.add_subplot(gs[rr, cc_]); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
                    ax.imshow(np.transpose(arr, (1, 0, 2)), origin="lower", interpolation="nearest"); ax.contour(gt.T.astype(float), [0.5], colors="lime", linewidths=0.7)
                    if pm is not None and pm.any():
                        ax.contour(pm.T.astype(float), [0.5], colors=[PAL[a]], linewidths=0.8)
                    ax.set_title(title, fontsize=6.5)
                    if cc_ == 0:
                        ax.set_ylabel(f"{a}\n{'typical' if pi == 0 else '75th pct'} case {case}\nreal - flat D_ch {drf.get(case, np.nan):+.2f} nat", fontsize=6.5)
        nr = len(rws)
        if K is not None:
            ax = fig.add_subplot(gs[: nr // 2, 6])
            for a in ARMS:
                d = K[(K.cell == c) & (K.arm == a) & (K["set"] == HSET)].set_index("k").reindex(list(KS))
                ax.plot(KS, d.E_sel, "-o", color=PAL[a], ms=3, label=f"{a} selective"); ax.fill_between(KS, d.E_rch_p025, d.E_rch_p975, color=PAL[a], alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.legend(fontsize=6, frameon=False)
            ax.set_title("target E vs k (shaded: random-channel 2.5-97.5 pct)", fontsize=7); ax.set_xlabel("k")
        if D is not None:
            ax = fig.add_subplot(gs[nr // 2:, 6]); s = "_".join(c.split("_")[:2]); oc = [f"o{o}" for o in range(7)]; P = D[(D.setting == s) & (D.group == "c1")]
            for a in ARMS + ("rand",):
                q = P[P.arm == a]
                for k_, (_, r_) in enumerate(q.iterrows()):
                    ax.plot(range(7), r_[oc].to_numpy(float), "-", color=PAL[a], lw=0.8, alpha=0.8, label=a if k_ == 0 else None)
            ax.set_xlabel("DCT order"); ax.set_ylabel("energy fraction"); ax.legend(fontsize=6, frameon=False); ax.set_title(f"{s}: first-conv order profile per fold / random seed", fontsize=7)
        fig.suptitle(f"{c}: smoothing the first-stage kernels with the most high-order DCT energy ({HSET}, k = {HK}) vs k random output channels.\n"
                     "Filled = predicted edema (arm colour), lime = GT edema; effect: orange = the removed fine-scale kernel content supported edema, cyan = opposed (shared scale, p99 over arms)", fontsize=8.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"kspec_examples_{c}.png", dpi=130); plt.close(fig)
    print("KSPEC_FIGURE_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spectra", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    if args.spectra:
        spectra(); return
    if args.report or args.figure:
        if args.report:
            report()
        if args.figure:
            figure()
        return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.cells:
        run_cell(key, dev, args.case_slice, smoke=args.smoke)


if __name__ == "__main__":
    main()
