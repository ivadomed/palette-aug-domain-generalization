#!/usr/bin/env python
"""
HiResCAM + first-stage activation energy, noise fill (rung 4) vs flat fill (rung 4.5) vs real fill (rung 5), EVERY held-out-contrast cell of
every paper task (FINDINGS section 33, pre-registered). Natural test images, no intervention. Re-runs the two earlier flat-vs-real leads with the
noise-fill robustness control added and all tasks included:
  * v2 (VISUAL_PROOF_PROPOSAL 3.3): interior share of the 1/4-res decoder HiResCAM of a structure's decision on structures both models segment alike;
  * v3b (3.5): first-encoder-stage activation energy follows image structure (Spearman with local high-pass texture / gradient magnitude).
Per case: one plan-sized patch centred on the target classes; per present target class c (>= MIN_VOX voxels, the MAX_STRUCT largest), per model
and fold: J_c = mean log-odds of c over the GT structure in the patch; HiResCAM_c = ReLU(sum_ch dJ/dA * A) at decoder stages[-3] (1/4 res) and
[-2] (1/2 res), Grad-CAM at [-3]; input gradient G_c = dJ_c/dx; encoder stages[0] channel-RMS energy E (one per model and fold).
Metrics (3D, body only): share of each map in the structure's interior (d >= 2 inside) and border (|d| <= 2), interior/border density ratio,
in-patch Dice / hit, fine-detail share of G (energy of G - Gaussian(G, 1)), |G| mass in the structure, Spearman(E, texture) / (E, edge),
E interior/border density ratio. Maps saved (fold mean) on the axial slice with the most target for the largest structure.
  GPU: --cells <cell> [--case-slice i:j]       CPU: --report
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import rankdata

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402

e7, v2 = vt.e7, vt.v2; FOLDS = (0, 1, 2)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
LAB = {"noise": "noise fill (rung 4)", "flat": "flat fill (rung 4.5)", "real": "real fill (rung 5)"}
OUT = Path(os.environ.get("VP_HIRES3_OUT") or (vt.OUT.parent / "hires3")); OUT.mkdir(parents=True, exist_ok=True)
MAX_CASES = int(os.environ.get("HIRES3_MAX_CASES", 40)); MAX_STRUCT = 12; MIN_VOX = 100; MATCH = 0.05; MAPS = ("hires3", "gcam3", "hires2")


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


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def fine_share(G, body):
    hp = G - ndi.gaussian_filter(G, 1.0); return float((hp[body] ** 2).sum() / ((G[body] ** 2).sum() + 1e-30))


def zones(gp, c):
    m = gp == c; din = ndi.distance_transform_edt(m); dout = ndi.distance_transform_edt(~m)
    return m, din >= 2, ((din > 0) & (din <= 2)) | ((dout > 0) & (dout <= 2))


def map_stats(mp, body, m, interior, border):
    tot = mp[body].sum() + 1e-30; di = mp[interior & body].mean() if (interior & body).any() else np.nan; db = mp[border & body].mean() if (border & body).any() else np.nan
    return dict(in_=float(mp[m & body].sum() / tot), interior=float(mp[interior & body].sum() / tot), border=float(mp[border & body].sum() / tot), ratio=float(di / db) if db else np.nan)


def run_cell(key, dev, sl=None):
    import torch, torch.nn.functional as F
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
        Z = {c: zones(gp, c) for c in structs}
        L = ndi.gaussian_filter(ip, 2.0); tex = ndi.gaussian_filter((ip - L) ** 2, 2.0); grad = np.sqrt(sum(ndi.sobel(ip, axis=i) ** 2 for i in range(3)))
        sub = np.flatnonzero(body); sub = np.random.default_rng(0).choice(sub, min(100000, sub.size), replace=False); t_, g_ = tex.ravel()[sub], grad.ravel()[sub]
        c0 = structs[0]; z = int(np.argmax((gp == c0).sum((1, 2)))); rows = []; maps = {"img": ip[z].astype(np.float16), "gt": gp[z].astype(np.int16), "z": z, "c0": c0}
        for a, pr in prs.items():
            net = pr.network; dec, enc = net.decoder.stages, net.encoder.stages; acc = {k: 0 for k in ("hires3", "gcam3", "G", "E", "lo")}
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval(); acts = {}
                def keep(name):
                    def h(m_, i_, o):
                        acts[name] = o   # returns None: the module output is not replaced
                    return h
                hs = [enc[0].register_forward_hook(keep("e0")), dec[len(dec) - 3].register_forward_hook(keep("d3")), dec[len(dec) - 2].register_forward_hook(keep("d2"))]
                x = torch.from_numpy(ip)[None, None].to(dev).requires_grad_(True)
                with torch.enable_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, x).float(); pred = o.argmax(1)[0].cpu().numpy()
                    E = acts["e0"].detach().float().pow(2).mean(1, keepdim=True).sqrt()
                    E = F.interpolate(E, size=tuple(ip.shape), mode="trilinear", align_corners=False)[0, 0].cpu().numpy() if E.shape[2:] != x.shape[2:] else E[0, 0].cpu().numpy()
                    rho_t, rho_e = spearman(E.ravel()[sub], t_), spearman(E.ravel()[sub], g_)
                    for c in structs:
                        m, interior, border = Z[c]; Mc = torch.from_numpy(m).to(dev); lo = e7.log_odds(o, c)[0]; J = lo[Mc].mean()
                        gA3, gA2, gx = torch.autograd.grad(J, [acts["d3"], acts["d2"], x], retain_graph=True)
                        A3, A2 = acts["d3"].detach().float(), acts["d2"].detach().float()
                        up = lambda t: F.interpolate(t, size=tuple(ip.shape), mode="trilinear", align_corners=False)[0, 0].detach().cpu().numpy()   # noqa: E731
                        cams = {"hires3": up(F.relu((gA3.float() * A3).sum(1, keepdim=True))), "gcam3": up(F.relu((gA3.float().mean((2, 3, 4), keepdim=True) * A3).sum(1, keepdim=True))),
                                "hires2": up(F.relu((gA2.float() * A2).sum(1, keepdim=True)))}
                        G = gx[0, 0].detach().cpu().numpy(); aG = np.abs(G); lov = lo.detach().cpu().numpy()
                        row = dict(cell=key, task=cell["task"], setting=setting(key), case=case, struct=c, vox=sizes[c], arm=a, fold=k,
                                   dice=vt.dice(pred == c, m), hit=float((lov[m] > 0).mean()), J=float(J), g_fine=fine_share(G, body), g_in=float(aG[m & body].sum() / (aG[body].sum() + 1e-30)),
                                   rho_tex=rho_t, rho_edge=rho_e, e_ratio=map_stats(E, body, m, interior, border)["ratio"], e_in=map_stats(E, body, m, interior, border)["in_"])
                        for nm, mp in cams.items():
                            row.update({f"{nm}_{kk}": v for kk, v in map_stats(mp, body, m, interior, border).items()})
                        rows.append(row)
                        if c == c0:
                            for kk, v in (("hires3", cams["hires3"]), ("gcam3", cams["gcam3"]), ("G", G), ("lo", lov)):
                                acc[kk] = acc[kk] + v[z] / len(FOLDS)
                    acc["E"] = acc["E"] + E[z] / len(FOLDS)
                for h in hs:
                    h.remove()
                del o, acts, x; torch.cuda.empty_cache()
            for kk, v in acc.items():
                maps[f"{a}_{kk}"] = np.asarray(v, np.float32)
        np.savez_compressed(od / f"{case}.npz", **maps)
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not f_csv.exists()); done.add(case)
        r = pd.DataFrame(rows).groupby("arm")[["dice", "hires3_interior", "rho_tex"]].mean()
        log(f"[{key}] {case} ({len(structs)} structures): dice " + " / ".join(f"{r.loc[a, 'dice']:.2f}" for a in ARMS) + " | HiResCAM interior " + " / ".join(f"{r.loc[a, 'hires3_interior']:.3f}" for a in ARMS)
            + " | rho(E,tex) " + " / ".join(f"{r.loc[a, 'rho_tex']:.2f}" for a in ARMS) + f"; {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------------------- report
METS = [("hires3_interior", "HiResCAM 1/4-res: share in the structure interior"), ("hires3_ratio", "HiResCAM 1/4-res: interior / border density"),
        ("gcam3_interior", "Grad-CAM 1/4-res: share in the interior"), ("hires2_interior", "HiResCAM 1/2-res: share in the interior"),
        ("g_in", "|input gradient| mass in the structure"), ("g_fine", "fine-detail share of the input gradient"),
        ("rho_tex", "Spearman(E, local texture)"), ("rho_edge", "Spearman(E, gradient magnitude)"), ("e_ratio", "E interior / border density"),
        ("dice", "in-patch Dice"), ("hit", "structure called (share)")]
ORDER = ["glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t2f", "glioma_t2w_t1n", "ms_flair_t2w", "ms_flair_t1w", "ms_t1w_flair", "ms_t1w_t2w", "breast_t1wce_t2w", "breast_t2w_t1wce",
         "brain_t1w_t2w", "brain_t2w_t1w", "abd_t1in_t2spir", "abd_t1in_ct", "abd_t2spir_t1in", "abd_t2spir_t1out", "abd_t2spir_ct", "mand_cbct_hanseg_ct", "mand_cbct_pddca_ct",
         "pelvis_mri_ct", "pelvis_ct_mri", "pancreas_t1wce_t2w", "pancreas_t2w_t1wce"]


def load_all():
    D = []
    for key in ORDER:
        f = OUT / key / "metrics.csv"
        if f.exists() and f.stat().st_size > 0:
            d = pd.read_csv(f); d["case"] = d.case.astype(str); D.append(d)
    return pd.concat(D, ignore_index=True) if D else pd.DataFrame()


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
    L = ["# HiResCAM / activation energy, noise vs flat vs real fill, all tasks (FINDINGS section 33, pre-registered)", "",
         "Per (case, structure): fold-mean; per case: median over structures; differences: median of per-case differences, Wilcoxon over cases, share of cases > 0. "
         f"MATCHED = structures with |Dice_a - Dice_real| <= {MATCH} for the pair (a, real) (pairwise matching, pre-registered primary). Texture step 4.5 -> 5 = paper tab:flatfill (Dice points).", ""]
    summ = []
    for m, nm in METS:
        L += [f"## {nm}", "", "| cell | tex. step | n (matched n-r / f-r) | noise | flat | real | real - noise MATCHED (p; >0) | real - flat MATCHED (p; >0) | real - noise ALL (p) | real - flat ALL (p) | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
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
            L.append(f"| {key} | {step:+.2f} | {rn['n']} / {rf['n']} | {rn['oth']:.3f} | {rf['oth']:.3f} | {rf['ref']:.3f} | {rn['med']:+.3f} ({rn['p']:.2g}; {rn['pos']:.0%}) | {rf['med']:+.3f} ({rf['p']:.2g}; {rf['pos']:.0%}) | "
                     f"{out[('noise', False)]['med']:+.3f} ({out[('noise', False)]['p']:.2g}) | {out[('flat', False)]['med']:+.3f} ({out[('flat', False)]['p']:.2g}) | {verdict} |")
            summ.append(dict(metric=m, cell=key, task=d.task.iloc[0], step=step, d_rn=rn["med"], p_rn=rn["p"], n_rn=rn["n"], d_rf=rf["med"], p_rf=rf["p"], n_rf=rf["n"], noise=rn["oth"], flat=rf["oth"], real=rf["ref"], verdict=verdict))
        S = pd.DataFrame([s for s in summ if s["metric"] == m]).dropna(subset=["d_rn", "step"])
        if len(S) >= 5:
            r1, p1 = spearmanr(S.d_rn, S.step); Sf = S.dropna(subset=["d_rf"]); r2, p2 = spearmanr(Sf.d_rf, Sf.step)
            L += ["", f"Across cells, Spearman(real - noise MATCHED, texture step) = {r1:+.2f} (p {p1:.2g}; n = {len(S)} cells); Spearman(real - flat MATCHED, texture step) = {r2:+.2f} (p {p2:.2g}; n = {len(Sf)} cells)."]
        L.append("")
    SM = pd.DataFrame(summ); SM.to_csv(OUT / "hires3_summary.csv", index=False)
    # skipped / coverage
    L += ["## Coverage", "", "| cell | cases with data | structures (case-level mean) | skipped |", "|---|---|---|---|"]
    for key in cells:
        d = D[D.cell == key]; sk = OUT / key / "skipped.csv"; ns = len(pd.read_csv(sk)) if sk.exists() and sk.stat().st_size > 0 else 0
        L.append(f"| {key} | {d.case.nunique()} | {d.groupby('case').struct.nunique().mean():.1f} | {ns} |")
    # gradmaps3 check requested by review: does cam4_in just restate hit?
    g3 = vt.OUT.parent / "gradmaps3"; rows = []
    for key in ("t1n_t2f", "t1n_t1n", "t2w_t2f"):
        f = g3 / key / "metrics.csv"
        if f.exists():
            G = pd.read_csv(f).groupby(["case", "arm"]).mean(numeric_only=True).reset_index()
            for a in ARMS:
                g = G[G.arm == a]; rows.append(f"| {key} | {a} | {spearmanr(g.cam4_in, g.hit)[0]:+.2f} | {spearmanr(g.cam3_in, g.hit)[0]:+.2f} | {spearmanr(g.in_e, g.hit)[0]:+.2f} | {spearmanr(g.fine_e, g.hit)[0]:+.2f} |")
    if rows:
        L += ["", "## gradmaps3 check: per-case Spearman of each metric with the edema hit rate (does the metric restate behaviour?)", "", "| cell | arm | cam4_in | cam3_in | in_e | fine_e |", "|---|---|---|---|---|---|"] + rows
    (OUT / "hires3.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # ---- figure 1: per-cell three-arm medians (matched structures for the pair with real) for the key metrics
    KEYM = [("hires3_interior", "HiResCAM (1/4 res)\nshare in structure interior"), ("hires3_ratio", "HiResCAM interior/border\ndensity"), ("rho_tex", "Spearman(E, texture)"), ("rho_edge", "Spearman(E, edges)"), ("g_fine", "fine-detail share\nof input gradient"), ("dice", "in-patch Dice")]
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
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "hires3_numbers.png", dpi=170); fig.savefig(OUT / "hires3_numbers.pdf"); plt.close(fig)
    # ---- figure 2: shift (real - noise, real - flat) vs texture step across cells, for HiResCAM interior share
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (col, nm) in zip(axs, (("d_rn", "real - noise"), ("d_rf", "real - flat"))):
        S = SM[SM.metric == "hires3_interior"].dropna(subset=[col, "step"])
        for _, s in S.iterrows():
            ax.scatter(s.step, s[col], color=vt.TCOL.get(s.task, "k"), s=30 if s[f"p{col[1:]}"] < 0.05 else 12, edgecolor="k" if s[f"p{col[1:]}"] < 0.05 else "none", lw=0.5)
            ax.annotate(s.cell, (s.step, s[col]), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6); ax.set_xlabel("texture step 4.5 -> 5 (Dice points, training setting)", fontsize=7); ax.set_ylabel(f"HiResCAM interior share, {nm} (matched)", fontsize=7); ax.tick_params(labelsize=7)
        if len(S) >= 5:
            r, p = spearmanr(S[col], S.step); ax.set_title(f"Spearman {r:+.2f}, p {p:.2g}, {len(S)} cells (filled = p < 0.05)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "hires3_shift_vs_step.png", dpi=170); plt.close(fig)
    # ---- figure 3: maps, one median-size case per cell (largest structure)
    for key in cells:
        fs = {f.stem: f for f in (OUT / key).glob("*.npz")}
        if not fs:
            continue
        vol = {c: int((np.load(f)["gt"] == np.load(f)["c0"]).sum()) for c, f in fs.items()}; order = sorted(vol, key=vol.get); pick = [order[int(round(q * (len(order) - 1)))] for q in (0.25, 0.5, 0.75)]
        cols = ["image"] + [f"{a}: HiResCAM 1/4" for a in ARMS] + [f"{a}: |grad|" for a in ARMS] + [f"{a}: E (enc. stage 0)" for a in ARMS]
        fig, axs = plt.subplots(len(pick), len(cols), figsize=(1.55 * len(cols), 1.9 * len(pick)), squeeze=False)
        for i, c in enumerate(pick):
            z = np.load(fs[c]); img = z["img"].astype(np.float32); g = z["gt"] == int(z["c0"]); body = img != 0 if (img == 0).mean() > 0.05 else np.ones(img.shape, bool)
            lo_, hi_ = np.percentile(img[body], [1, 99])
            G = {a: ndi.gaussian_filter(np.abs(z[f"{a}_G"]) / (np.abs(z[f"{a}_G"])[body].sum() + 1e-30), 1.0) for a in ARMS}; vg = np.percentile(np.concatenate([G[a][body] for a in ARMS]), 99.5)
            E = {a: z[f"{a}_E"] for a in ARMS}; ve = np.percentile(np.concatenate([E[a][body] for a in ARMS]), 99.5)
            for j, nm in enumerate(cols):
                ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.imshow(img, cmap="gray", vmin=lo_, vmax=hi_); ax.contour(g, [0.5], colors="#00b4ff", linewidths=0.6); ax.set_ylabel(f"{c}\nstructure {int(z['c0'])}, vol. q{[25, 50, 75][i]}", fontsize=5.5)
                else:
                    a = nm.split(":")[0]
                    if "HiResCAM" in nm:
                        ax.imshow(img, cmap="gray", vmin=lo_, vmax=hi_); cm = z[f"{a}_hires3"]; ax.imshow(np.ma.masked_where(cm <= 0.05 * cm.max(), cm), cmap="jet", alpha=0.45)
                    elif "grad" in nm:
                        ax.imshow(np.where(body, G[a], np.nan), cmap="magma", vmin=0, vmax=vg)
                    else:
                        ax.imshow(np.where(body, E[a], np.nan), cmap="viridis", vmin=0, vmax=ve)
                    ax.contour(g, [0.5], colors="white", linewidths=0.5)
                if i == 0:
                    ax.set_title(nm, fontsize=6, color=PAL.get(nm.split(":")[0], "k"))
        fig.suptitle(f"{key}: HiResCAM (decoder 1/4 res) of the structure's decision, |input gradient| (unit mass, shared scale per row), first-encoder-stage activation energy E. "
                     "Contour = ground-truth structure.", fontsize=6.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"hires3_maps_{key}.png", dpi=150); plt.close(fig)


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
