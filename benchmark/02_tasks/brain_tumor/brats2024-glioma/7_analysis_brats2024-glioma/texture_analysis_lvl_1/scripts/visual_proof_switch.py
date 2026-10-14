#!/usr/bin/env python
"""Visual proof v12 (FINDINGS 39): "find the texture switch and flip it". Interventions INSIDE the network, mask-free counterfactuals, BraTS first.

Three steps on the same patch per case (plan patch around the target, as in v10), arms noise (rung 4, lblvor) / flat (4.5) / real (5), 3 folds:
  1. Dose-response (no mask, no label): x_a = x + (a - 1) * highpass(x, sigma 1) inside the body, a in ALPHAS. Not a pure scaling, so instance norm does
     not cancel it; a > 1 ADDS fine detail (kills the "real fill just dislikes smooth input" reading). Dice of the target and J = mean target log-odds over
     the GT target voxels, per arm and a.
  2. Mediation (activation patching): decision on x with ONE location's activations taken from the x_a run (skips s0..s5 = encoder stage outputs,
     d0..d4 = decoder stage outputs); share = (J(mixed) - J(x)) / (J(x_a) - J(x)). Arms flat and real, a in MED_ALPHAS.
  3. The switch (concept erasure on NATURAL images): --calib accumulates, over the first N_CAL cases, the per-voxel activation displacement
     h(x_2) - h(x_0.5) at every location (mean direction + PCA of the displacement covariance, per arm and fold; label-free). Evaluation projects that
     subspace (rank 1 mean / PCA 1, 4, 16) out of the activations at one location and re-predicts the untouched image; controls: random orthonormal
     subspaces of rank 4 (RAND_DRAWS draws), and the flat model with its own directions.
Outputs: <OUT>/<cell>/{dirs_<arm>_f<k>.npz, <case>_switch.csv}; --report: switch.md, switch_*.csv, switch_*.png.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_erf_outcome as vo  # noqa: E402
import visual_proof_v2 as v2  # noqa: E402
import visual_proof_erf as e7  # noqa: E402

OUT = Path(os.environ.get("VP_SWITCH_OUT") or (THIS.parent / "outputs/plots/switch")); OUT.mkdir(parents=True, exist_ok=True)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
ALPHAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0); MED_ALPHAS = (0.5, 2.0); SIG = 1.0
N_CAL = int(os.environ.get("SW_N_CAL", 20)); RANKS = (1, 4, 16); RAND_K = 4; RAND_DRAWS = 3; FOLDS = v2.FOLDS
LOCS = [f"s{j}" for j in range(6)] + [f"d{j}" for j in range(5)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- network plumbing
def encoder_fwd(net, x):
    return list(net.encoder(x))


def decoder_fwd(net, skips, swap=None, proj=None):
    """nnU-Net UNetDecoder.forward re-implemented (deep supervision off) with two hooks: swap = {loc: tensor} replaces the activation at loc
    (sK = skip K before the decoder uses it, dS = output of decoder stage S); proj = {loc: Q (C, k)} projects the channel subspace Q out at loc."""
    dec = net.decoder; skips = list(skips); swap = swap or {}; proj = proj or {}

    def hook(loc, h):
        if loc in swap:
            h = swap[loc]
        if loc in proj:
            Q = proj[loc]; hc = h.movedim(1, -1); h = (hc - (hc @ Q) @ Q.T).movedim(-1, 1)
        return h
    for j in range(len(skips)):
        skips[j] = hook(f"s{j}", skips[j])
    lres = skips[-1]; out = None
    for s in range(len(dec.stages)):
        h = dec.transpconvs[s](lres); h = torch_cat(h, skips[-(s + 2)]); h = dec.stages[s](h); h = hook(f"d{s}", h)
        if s == len(dec.stages) - 1:
            out = dec.seg_layers[-1](h)
        lres = h
    return out


def torch_cat(a, b):
    import torch
    return torch.cat((a, b), 1)


def activations(net, x):
    """All hookable activations of one forward: dict loc -> tensor (detached), plus the logits."""
    import torch
    with torch.no_grad():
        skips = encoder_fwd(net, x); acts = {f"s{j}": s for j, s in enumerate(skips)}
        dec = net.decoder; lres = skips[-1]
        for s in range(len(dec.stages)):
            h = dec.transpconvs[s](lres); h = torch_cat(h, skips[-(s + 2)]); h = dec.stages[s](h); acts[f"d{s}"] = h; lres = h
        logits = dec.seg_layers[-1](lres)
    return acts, logits


def dice_and_j(logits, gp, T, folds_prob=None):
    """logits (1, C, ...): Dice of the target (argmax in T vs GT in T) and J = mean log-odds of the first target class over GT target voxels."""
    import torch
    with torch.no_grad():
        p = torch.softmax(logits.float(), 1)[0]; pred = p.argmax(0).cpu().numpy(); tg = np.isin(gp, T); pr = np.isin(pred, T)
        d = 2 * (tg & pr).sum() / (tg.sum() + pr.sum() + 1e-9); lo = e7.log_odds(logits.float(), int(T[0]))[0].cpu().numpy(); j = float(lo[tg].mean()) if tg.any() else np.nan
    return float(d), j, p


def dice_prob(P, gp, T):
    pred = P.argmax(0); tg = np.isin(gp, T); pr = np.isin(pred, T); return float(2 * (tg & pr).sum() / (tg.sum() + pr.sum() + 1e-9))


# ----------------------------------------------------------------------------------------------------------------------------------- data
def load_case(pr0, cell, case, T):
    data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T); ps = pr0.configuration_manager.patch_size
    body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
    box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
    padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
    if any(w for _, w in padw):
        ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
    hp = (ip - ndi.gaussian_filter(ip, SIG)) * bp
    return ip, bp, gp, hp


def counterfactual(ip, hp, a):
    return (ip + (a - 1.0) * hp).astype(np.float32)


def qc(ip, hp, gp, T, a):
    """What a changes besides texture: edge energy at the GT boundary (|grad G(x, 2)|) and fine-texture energy inside the target, ratio x_a / x."""
    xa = counterfactual(ip, hp, a); tg = np.isin(gp, T); bd = ndi.binary_dilation(tg, iterations=2) & ~ndi.binary_erosion(tg, iterations=2)
    def edge(x):
        g = ndi.gaussian_gradient_magnitude(x, 2.0); return float(g[bd].mean()) if bd.any() else np.nan
    def tex(x):
        h = x - ndi.gaussian_filter(x, SIG); return float(np.sqrt((h[tg] ** 2).mean())) if tg.any() else np.nan
    return edge(xa) / (edge(ip) + 1e-9), tex(xa) / (tex(ip) + 1e-9)


# ----------------------------------------------------------------------------------------------------------------------------------- phases
def models_for(key):
    return vo.models(key)


def calib(key, dev, sl=None):
    """Accumulate the displacement statistics h(x_2) - h(x_0.5) per arm / fold / location over the calibration cases -> dirs_<arm>_f<k>.npz."""
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in models_for(key).items() if a in ("flat", "real")}; pr0 = prs["flat"]
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = sorted(cell["cases"])[:N_CAL]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    acc = {}   # (arm, fold, loc) -> [n, sum (C), sum outer (C, C)]
    for case in cases:
        t0 = time.time(); ip, bp, gp, hp = load_case(pr0, cell, case, T)
        xs = {a: torch.from_numpy(counterfactual(ip, hp, a))[None, None].to(dev) for a in (0.5, 2.0)}
        for arm, pr in prs.items():
            net = pr.network
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                A2, _ = activations(net, xs[2.0]); A5, _ = activations(net, xs[0.5])
                for loc in LOCS:
                    D = (A2[loc] - A5[loc])[0].float(); C = D.shape[0]; Dm = D.reshape(C, -1)
                    # body mask at this resolution (nearest): keep only voxels inside the body so background does not dominate
                    m = torch.from_numpy(bp).to(dev)[None, None].float(); m = torch.nn.functional.interpolate(m, size=D.shape[1:], mode="nearest")[0, 0].reshape(-1) > 0.5
                    Dm = Dm[:, m]; n = int(Dm.shape[1]); s1 = Dm.sum(1); s2 = Dm @ Dm.T
                    st = acc.setdefault((arm, k, loc), [0, torch.zeros(C, device=dev), torch.zeros(C, C, device=dev)]); st[0] += n; st[1] += s1; st[2] += s2
                del A2, A5; torch.cuda.empty_cache()
        log(f"[{key}] calib {case}: {time.time() - t0:.0f} s")
    for arm in prs:
        for k in FOLDS:
            out = {}
            for loc in LOCS:
                n, s1, s2 = acc[(arm, k, loc)]; mu = s1 / n; cov = s2 / n - torch.outer(mu, mu)
                ev, evec = torch.linalg.eigh(cov); order = torch.argsort(ev, descending=True); ev, evec = ev[order], evec[:, order]
                out[f"{loc}_mean"] = (mu / (mu.norm() + 1e-12)).cpu().numpy(); out[f"{loc}_pcs"] = evec[:, :max(RANKS)].cpu().numpy(); out[f"{loc}_ev"] = ev.cpu().numpy(); out[f"{loc}_n"] = n
                out[f"{loc}_mean_share"] = float(((mu @ mu) / (ev.sum() + mu @ mu + 1e-12)).cpu())   # share of displacement energy carried by the mean direction
            np.savez_compressed(od / f"dirs_{arm}_f{k}_{sl.replace(':', '-') if sl else 'all'}.npz", **out)
    print("CALIB_DONE", flush=True)


def merge_dirs(od, arm, k):
    """If calibration ran in slices, merge the per-slice statistics is not possible from the saved eigen-decompositions; we require a single 'all' file."""
    f = od / f"dirs_{arm}_f{k}_all.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    return np.load(f)


def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in models_for(key).items()}; pr0 = prs["flat"]
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    allc = sorted(cell["cases"]); calset = set(allc[:N_CAL]); cases = allc[slice(*[int(v) for v in sl.split(":")])] if sl else allc
    dirs = {(arm, k): merge_dirs(od, arm, k) for arm in ("flat", "real") for k in FOLDS}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for case in cases:
        f_out = od / f"{case}_switch.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, hp = load_case(pr0, cell, case, T); rows = []
        base = dict(cell=key, task=cell["task"], case=case, calib=case in calset, n_target=int(np.isin(gp, T).sum()))
        # ---- step 1: dose-response, all arms
        for a in ALPHAS:
            xa = torch.from_numpy(counterfactual(ip, hp, a))[None, None].to(dev); eq, tq = qc(ip, hp, gp, T, a)
            for arm, pr in prs.items():
                net = pr.network; P = 0; js = []
                for k in FOLDS:
                    net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                    with torch.no_grad():
                        lg = decoder_fwd(net, encoder_fwd(net, xa))
                    d_, j_, p_ = dice_and_j(lg, gp, T); P = P + p_.cpu().numpy() / len(FOLDS); js.append(j_)
                rows.append(dict(base, step="dose", arm=arm, alpha=a, loc="", rank=0, draw=-1, dice=dice_prob(P, gp, T), J=float(np.mean(js)), edge_ratio=eq, tex_ratio=tq))
        # ---- steps 2 and 3: flat and real
        x1 = torch.from_numpy(ip)[None, None].to(dev)
        for arm in ("flat", "real"):
            pr = prs[arm]; net = pr.network
            P_nat = 0; med = {}; abl = {}
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                A1, lg1 = activations(net, x1); d1, j1, p1 = dice_and_j(lg1, gp, T); P_nat = P_nat + p1.cpu().numpy() / len(FOLDS)
                # step 2: mediation
                for a in MED_ALPHAS:
                    xa = torch.from_numpy(counterfactual(ip, hp, a))[None, None].to(dev); Aa, lga = activations(net, xa); da, ja, _ = dice_and_j(lga, gp, T)
                    med.setdefault(("full", a), []).append(ja - j1)
                    for loc in LOCS:
                        with torch.no_grad():
                            lgm = decoder_fwd(net, [A1[f"s{j}"] for j in range(6)], swap={loc: Aa[loc]})
                        dm, jm, _ = dice_and_j(lgm, gp, T); med.setdefault((loc, a), []).append(jm - j1)
                    del Aa
                # step 3: erasure on the natural image
                D = dirs[(arm, k)]
                for loc in LOCS:
                    subs = [("mean", 1, -1, torch.from_numpy(D[f"{loc}_mean"][:, None]).to(dev))] + [("pca", r, -1, torch.from_numpy(D[f"{loc}_pcs"][:, :r]).to(dev)) for r in RANKS]
                    C = D[f"{loc}_mean"].shape[0]
                    for dr in range(RAND_DRAWS):
                        Q, _ = torch.linalg.qr(torch.randn(C, RAND_K, generator=gen)); subs.append(("rand", RAND_K, dr, Q.to(dev)))
                    for kind, r, dr, Q in subs:
                        with torch.no_grad():
                            lgp = decoder_fwd(net, [A1[f"s{j}"] for j in range(6)], proj={loc: Q.float()})
                        dp, jp, pp = dice_and_j(lgp, gp, T); abl.setdefault((loc, kind, r, dr), [0, []]); abl[(loc, kind, r, dr)][0] += pp.cpu().numpy() / len(FOLDS); abl[(loc, kind, r, dr)][1].append(jp - j1)
                del A1; torch.cuda.empty_cache()
            rows.append(dict(base, step="natural", arm=arm, alpha=1.0, loc="", rank=0, draw=-1, dice=dice_prob(P_nat, gp, T), J=np.nan))
            for (loc, a), v in med.items():
                full = np.mean(med[("full", a)]); rows.append(dict(base, step="med", arm=arm, alpha=a, loc=loc, rank=0, draw=-1, dJ=float(np.mean(v)), dJ_full=float(full), share=float(np.mean(v) / full) if abs(full) > 1e-6 else np.nan))
            for (loc, kind, r, dr), (P, v) in abl.items():
                rows.append(dict(base, step="abl", arm=arm, alpha=1.0, loc=loc, kind=kind, rank=r, draw=dr, dice=dice_prob(P, gp, T), dJ=float(np.mean(v))))
        pd.DataFrame(rows).to_csv(f_out, index=False); log(f"[{key}] {case}: {len(rows)} rows, {time.time() - t0:.0f} s")
    print("SWITCH_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
def wp(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    try:
        return wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
    except ValueError:
        return np.nan


def report():
    cells = [k for k in vo.ORDER if (OUT / k).exists() and list((OUT / k).glob("*_switch.csv"))]
    S = pd.concat([pd.read_csv(f) for k in cells for f in sorted((OUT / k).glob("*_switch.csv"))], ignore_index=True)
    L = [f"# The texture switch (FINDINGS 39): mask-free fine-detail modulation, mediation across the U-Net, and erasure of the learned direction", "",
         f"Cells: {', '.join(cells)}; cases per cell: " + ", ".join(f"{k} {S[S.cell == k].case.nunique()}" for k in cells) + f". Counterfactual x_a = x + (a - 1) highpass_1(x) in the body. "
         f"Dice = target Dice on the plan patch (fold-mean softmax); J = mean target log-odds over GT target voxels (fold mean). Calibration cases (first {N_CAL} per cell) define the erasure directions and are excluded from step 3 tables.", ""]
    # step 1
    D = S[S.step == "dose"]; L += ["## 1. Dose-response: Dice (median over cases) per arm and a; last columns: change from a = 1, real - flat (Wilcoxon over cases)", "",
                                   "| cell | arm | " + " | ".join(f"a={a:g}" for a in ALPHAS) + " | edge ratio a=2 | tex ratio a=2 |", "|" + "---|" * (3 + len(ALPHAS))]
    rows1 = []
    for k in cells:
        for arm in ARMS:
            d = D[(D.cell == k) & (D.arm == arm)].pivot(index="case", columns="alpha", values="dice"); q = D[(D.cell == k) & (D.arm == arm) & (D.alpha == 2.0)]
            L.append(f"| {k} | {arm} | " + " | ".join(f"{d[a].median():.3f}" for a in ALPHAS) + f" | {q.edge_ratio.median():.2f} | {q.tex_ratio.median():.2f} |")
            for a in ALPHAS:
                rows1.append(dict(cell=k, arm=arm, alpha=a, dice=d[a].median(), d_from_1=(d[a] - d[1.0]).median(), p_from_1=wp(d[a] - d[1.0]), n=len(d)))
    L += [""]
    for k in cells:
        dr = D[(D.cell == k) & (D.arm == "real")].pivot(index="case", columns="alpha", values="dice"); df = D[(D.cell == k) & (D.arm == "flat")].pivot(index="case", columns="alpha", values="dice"); dn = D[(D.cell == k) & (D.arm == "noise")].pivot(index="case", columns="alpha", values="dice")
        L.append(f"- {k}: drop from a=1, real / flat / noise: a=0.5 {(dr[0.5] - dr[1.0]).median():+.3f} / {(df[0.5] - df[1.0]).median():+.3f} / {(dn[0.5] - dn[1.0]).median():+.3f}; a=2 {(dr[2.0] - dr[1.0]).median():+.3f} / {(df[2.0] - df[1.0]).median():+.3f} / {(dn[2.0] - dn[1.0]).median():+.3f}; "
                 f"real - flat drop: a=0.5 {((dr[0.5] - dr[1.0]) - (df[0.5] - df[1.0])).median():+.3f} (p {wp((dr[0.5] - dr[1.0]) - (df[0.5] - df[1.0])):.1g}), a=2 {((dr[2.0] - dr[1.0]) - (df[2.0] - df[1.0])).median():+.3f} (p {wp((dr[2.0] - dr[1.0]) - (df[2.0] - df[1.0])):.1g})")
    pd.DataFrame(rows1).to_csv(OUT / "switch_dose.csv", index=False)
    # step 2
    M = S[S.step == "med"]; L += ["", "## 2. Mediation: share of the counterfactual decision change dJ carried by each location alone (median over cases); dJ_full = J(x_a) - J(x)", ""]
    rows2 = []
    for a in MED_ALPHAS:
        L += [f"### a = {a:g}", "", "| cell | arm | dJ_full | " + " | ".join(LOCS) + " |", "|" + "---|" * (3 + len(LOCS))]
        for k in cells:
            for arm in ("flat", "real"):
                m = M[(M.cell == k) & (M.arm == arm) & (M.alpha == a)]; pv = m.pivot(index="case", columns="loc", values="share"); full = m.groupby("case").dJ_full.first()
                L.append(f"| {k} | {arm} | {full.median():+.3f} | " + " | ".join(f"{pv[l].median():.2f}" if l in pv else "" for l in LOCS) + " |")
                for l in LOCS:
                    if l in pv:
                        rows2.append(dict(cell=k, arm=arm, alpha=a, loc=l, share=pv[l].median(), dJ=m[m["loc"] == l].dJ.median(), dJ_full=full.median()))
        L.append("")
    pd.DataFrame(rows2).to_csv(OUT / "switch_mediation.csv", index=False)
    # step 3
    Bn = S[(S.step == "natural") & (~S.calib)]; B = S[(S.step == "abl") & (~S.calib)]; rows3 = []
    L += ["## 3. Erasure on natural images (non-calibration cases): Dice after projecting the learned subspace out at one location; natural Dice, then mean / pca1 / pca4 / pca16 / random-4 (mean of draws); flat Dice for reference", ""]
    for k in cells:
        nat = {arm: Bn[(Bn.cell == k) & (Bn.arm == arm)].set_index("case").dice for arm in ("flat", "real")}
        L += [f"### {k} (natural Dice flat {nat['flat'].median():.3f}, real {nat['real'].median():.3f}, n = {len(nat['real'])})", "", "| arm | loc | natural | mean | pca1 | pca4 | pca16 | rand4 | real(abl) - flat(nat) [pca4] | p(abl < nat) pca4 | p(abl < rand4) pca4 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ("flat", "real"):
            for l in LOCS:
                b = B[(B.cell == k) & (B.arm == arm) & (B["loc"] == l)]
                def col(kind, r):
                    s = b[(b.kind == kind) & (b["rank"] == r)].groupby("case").dice.mean(); return s.reindex(nat[arm].index)
                cm, c1, c4, c16, cr = col("mean", 1), col("pca", 1), col("pca", 4), col("pca", 16), col("rand", RAND_K)
                L.append(f"| {arm} | {l} | {nat[arm].median():.3f} | {cm.median():.3f} | {c1.median():.3f} | {c4.median():.3f} | {c16.median():.3f} | {cr.median():.3f} | {(c4 - nat['flat']).median():+.3f} | {wp(c4 - nat[arm]):.1g} | {wp(c4 - cr):.1g} |")
                rows3.append(dict(cell=k, arm=arm, loc=l, natural=nat[arm].median(), mean=cm.median(), pca1=c1.median(), pca4=c4.median(), pca16=c16.median(), rand4=cr.median(),
                                  d_pca4=(c4 - nat[arm]).median(), p_pca4=wp(c4 - nat[arm]), d_pca16=(c16 - nat[arm]).median(), p_pca16=wp(c16 - nat[arm]), d_rand4=(cr - nat[arm]).median(), p_pca4_vs_rand=wp(c4 - cr)))
        L.append("")
    R3 = pd.DataFrame(rows3); R3.to_csv(OUT / "switch_erasure.csv", index=False)
    (OUT / "switch.md").write_text("\n".join(L)); print("\n".join(L[:12]))
    # figures
    fig, axs = plt.subplots(3, len(cells), figsize=(3.6 * len(cells), 9.5), squeeze=False)
    for c, k in enumerate(cells):
        ax = axs[0, c]
        for arm in ARMS:
            d = D[(D.cell == k) & (D.arm == arm)].pivot(index="case", columns="alpha", values="dice"); ax.plot(ALPHAS, [d[a].median() for a in ALPHAS], "-o", color=PAL[arm], ms=3, label=arm)
            ax.fill_between(ALPHAS, [d[a].quantile(0.25) for a in ALPHAS], [d[a].quantile(0.75) for a in ALPHAS], color=PAL[arm], alpha=0.12, lw=0)
        ax.set_title(f"{k}\nDice vs fine-detail gain a", fontsize=8); ax.set_xlabel("a (1 = original)"); ax.axvline(1, color="k", lw=0.5, ls=":")
        if c == 0:
            ax.set_ylabel("target Dice (median, IQR)"); ax.legend(fontsize=7, frameon=False)
        ax = axs[1, c]; w = 0.38
        for i, arm in enumerate(("flat", "real")):
            m = M[(M.cell == k) & (M.arm == arm) & (M.alpha == 2.0)].pivot(index="case", columns="loc", values="share"); ax.bar(np.arange(len(LOCS)) + (i - 0.5) * w, [m[l].median() if l in m else 0 for l in LOCS], w, color=PAL[arm], label=arm)
        ax.set_xticks(range(len(LOCS))); ax.set_xticklabels(LOCS, fontsize=7); ax.axhline(0, color="k", lw=0.5); ax.set_title("mediation share of dJ (a = 2), one location patched", fontsize=8)
        if c == 0:
            ax.set_ylabel("share of the counterfactual change"); ax.legend(fontsize=7, frameon=False)
        ax = axs[2, c]; nat = {arm: Bn[(Bn.cell == k) & (Bn.arm == arm)].dice.median() for arm in ("flat", "real")}
        for i, arm in enumerate(("flat", "real")):
            r = R3[(R3.cell == k) & (R3.arm == arm)].set_index("loc").reindex(LOCS); ax.bar(np.arange(len(LOCS)) + (i - 0.5) * w, r.pca4, w, color=PAL[arm], label=f"{arm}: erase pca-4")
            ax.plot(np.arange(len(LOCS)) + (i - 0.5) * w, r.rand4, "_", color="k", ms=8, mew=1.2, label="random-4 control" if i == 0 else None); ax.axhline(nat[arm], color=PAL[arm], lw=0.8, ls="--")
        ax.set_xticks(range(len(LOCS))); ax.set_xticklabels(LOCS, fontsize=7); ax.set_title("natural-image Dice after erasing the texture subspace (dashed = untouched)", fontsize=8); ax.set_ylim(0, 1)
        if c == 0:
            ax.set_ylabel("target Dice"); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "switch_summary.png", dpi=150); fig.savefig(OUT / "switch_summary.pdf"); plt.close(fig); print("SWITCH_REPORT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--calib", action="store_true"); ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(); return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.cells:
        (calib if args.calib else run_cell)(key, dev, args.case_slice)


if __name__ == "__main__":
    main()
