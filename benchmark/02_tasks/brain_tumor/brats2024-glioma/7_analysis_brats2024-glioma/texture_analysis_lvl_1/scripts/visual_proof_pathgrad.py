#!/usr/bin/env python
"""Visual proof v10 (2026-10-13, FINDINGS section 37): PATH-DECOMPOSED input gradients, non-interventional, on the section-36 seeds.

nnU-Net's forward is `skips = encoder(x); decoder(skips)`. The input gradient of a voxel's log-odds is the sum, over the skip connections and the
bottleneck, of the gradient that flows back through that path: detaching every skip but one during the backward pass (forward unchanged, so the
prediction is untouched) gives that path's part, and the parts add up to the full gradient exactly. Path 0 = the full-resolution skip (the only
path that never goes through pooling). The share of a voxel's decision carried by path 0 is a direct measure of fine-detail reliance, with no
spectral proxy (sections 32-36 used the fine-detail share of the full gradient).
Per seed (same voxels and outcome patterns as section 36, <= MAX_SEEDS per case, rare patterns kept first) and arm (noise / flat / real), gradient
of the arm's OWN predicted class, fold-mean: G (full), G0 (path 0), G1 (path 1), Gr = G - G0 - G1 (all pooled paths). Metrics:
share0 / share1 / share_r = <Gk, G> / |G|^2 (sum to 1), e0 = |G0|^2 / |G|^2, local0 / local_r = |Gk| mass within r_loc, fine0 / fine_r = fine-detail
share of Gk, fine = fine-detail share of G (section 36 check). Additivity is verified on the first seed of every case (all paths vs full).
Outputs: outputs/plots/pathgrad/<cell>/{seeds.csv, <case>_pg.npz (seed slices, NOT committed)}; --report: pathgrad.md, pathgrad_cells.csv,
pathgrad_summary.png, pathgrad_patterns.png, pathgrad_examples_<cell>.png (patient slices, NOT committed).
GPU: .venv/bin/python visual_proof_pathgrad.py --cells <cell> [--case-slice i:j]      CPU: --report
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
e7, v2 = vt.e7, vt.v2; FOLDS = (0, 1, 2)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
ALL_PATHS = os.environ.get("PG_ALL", "0") == "1"   # 1: one gradient per skip (p0 ... p{n-1}), Paul's request 2026-10-13 (texture = label attribution = deeper skips); 0: p0, p1, rest
SEEDS_IN = vo.OUT; OUT = Path(os.environ.get("VP_PATHGRAD_OUT") or (vt.OUT.parent / ("pathgrad_all" if ALL_PATHS else "pathgrad"))); OUT.mkdir(parents=True, exist_ok=True)
MAX_SEEDS = int(os.environ.get("PG_MAX_SEEDS", 24)); MAX_CASES = vo.MAX_CASES; BATCH = 2
ORDER = vo.ORDER; PATS = ["CCC", "FFC", "FCF", "CFF", "CCF", "CFC", "FCC", "FFF"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def path_grads(net, x, seeds, cls, dev, keep=None):
    """Gradients of the seeds' log-odds w.r.t. the input, flowing only through the skips in `keep` (None = all = the full gradient)."""
    import torch
    out = []; xt = torch.from_numpy(x).to(dev)
    for i in range(0, len(seeds), BATCH):
        sb, cb = seeds[i:i + BATCH], cls[i:i + BATCH]
        xb = xt[None, None].repeat(len(sb), 1, 1, 1, 1).clone().requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            skips = net.encoder(xb)
            if keep is not None:
                skips = [s if j in keep else s.detach() for j, s in enumerate(skips)]
            o = net.decoder(skips); o = o[0] if isinstance(o, (list, tuple)) else o
            f = sum(e7.log_odds(o[b:b + 1], c)[0][tuple(s)] for b, (s, c) in enumerate(zip(sb, cb)))
            g, = torch.autograd.grad(f, xb)
        out += [g[b, 0].detach() for b in range(len(sb))]
    return out


def fine_share(G, body):
    hp = G - ndi.gaussian_filter(G, 1.0); return float((hp[body] ** 2).sum() / ((G[body] ** 2).sum() + 1e-30))


def pick_seeds(d):
    """<= MAX_SEEDS per case, keeping the rare patterns (everything but CCC / FFF) first, then filling with CCC / FFF."""
    rare = d[~d.pattern.isin(["CCC", "FFF"])]; common = d[d.pattern.isin(["CCC", "FFF"])]
    rng = np.random.default_rng(0)
    if len(rare) >= MAX_SEEDS:
        return rare.iloc[sorted(rng.choice(len(rare), MAX_SEEDS, replace=False))]
    k = MAX_SEEDS - len(rare); fill = common.iloc[sorted(rng.choice(len(common), min(k, len(common)), replace=False))] if len(common) else common
    return pd.concat([rare, fill]).sort_values("seed")


def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; r_loc = cell["r_loc"]; od = OUT / key; od.mkdir(exist_ok=True)
    f_in = SEEDS_IN / key / "seeds.csv"
    if not f_in.exists() or f_in.stat().st_size < 20:
        log(f"[{key}] no section-36 seeds"); return
    SD = pd.read_csv(f_in, dtype=str); SD = SD[SD.case != "case"]
    for c in ("seed", "z", "y", "x", "cls", "band", "depth") + tuple(f"{a}_pred" for a in ARMS) + tuple(f"{a}_margin" for a in ARMS):
        SD[c] = pd.to_numeric(SD[c])
    prs = {a: v2.load_predictor(d, dev) for a, d in vo.models(key).items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]) if key.startswith("glioma") else list(cell["cases"])[:MAX_CASES]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "seeds.csv"
    try:
        done = set(pd.read_csv(f_csv).case.astype(str)) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    except pd.errors.EmptyDataError:
        done = set()
    n_paths = len(pr0.network.encoder.stages)
    for case in cases:
        if case in done or case not in set(SD.case):
            continue
        t0 = time.time(); d = pick_seeds(SD[SD.case == case]); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        V = [(int(r.z), int(r.y), int(r.x)) for r in d.itertuples()]
        G = {}; add_err = []
        for a, pr in prs.items():
            net = pr.network; cls = [int(getattr(r, f"{a}_pred")) for r in d.itertuples()]
            targets = [("full", None)] + ([(f"p{j}", {j}) for j in range(n_paths)] if ALL_PATHS else [("p0", {0}), ("p1", {1})]); acc = {k: None for k, _ in targets}
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                for nm, keep in targets:
                    g = path_grads(net, ip, V, cls, dev, keep); acc[nm] = g if acc[nm] is None else [x_ + y_ for x_, y_ in zip(acc[nm], g)]
                if k == FOLDS[0]:   # additivity check on the first seed: sum of all single-path gradients vs the full gradient
                    parts = [path_grads(net, ip, V[:1], cls[:1], dev, {j})[0] for j in range(n_paths)]; full = path_grads(net, ip, V[:1], cls[:1], dev, None)[0]
                    add_err.append(float((sum(parts) - full).abs().max() / (full.abs().max() + 1e-30)))
            for nm in acc:
                G[a, nm] = [(x_ / len(FOLDS)).cpu().numpy() for x_ in acc[nm]]
            G[a, "pr"] = [f_ - p0 - p1 for f_, p0, p1 in zip(G[a, "full"], G[a, "p0"], G[a, "p1"])]
            torch.cuda.empty_cache()
        zz, yy, xx = np.indices(ip.shape, sparse=True); rows = []; sv = {"img": [], "gt": [], "z": [], "pattern": []}
        PN = [f"p{j}" for j in range(n_paths)] if ALL_PATHS else ["p0", "p1", "pr"]; SV = (["full"] + PN[:4]) if ALL_PATHS else ["full", "p0", "pr"]
        for a in ARMS:
            for nm in SV:
                sv[f"{a}_{nm}"] = []
        for i, r in enumerate(d.itertuples()):
            v = V[i]; ball = (np.sqrt((zz - v[0]) ** 2 + (yy - v[1]) ** 2 + (xx - v[2]) ** 2) <= r_loc) & bp
            row = dict(cell=key, task=cell["task"], case=case, seed=int(r.seed), region=r.region, pattern=r.pattern, band=int(r.band), depth=float(r.depth), cls=int(r.cls))
            for a in ARMS:
                Gf = G[a, "full"][i]; den = float((Gf[bp] ** 2).sum()) + 1e-30; row[f"{a}_margin"] = float(getattr(r, f"{a}_margin")); row[f"{a}_pred"] = int(getattr(r, f"{a}_pred"))
                for nm in PN:
                    Gk = G[a, nm][i]; row[f"{a}_share_{nm}"] = float((Gk[bp] * Gf[bp]).sum() / den); row[f"{a}_e_{nm}"] = float((Gk[bp] ** 2).sum() / den)
                    row[f"{a}_local_{nm}"] = float(np.abs(Gk)[ball].sum() / (np.abs(Gk)[bp].sum() + 1e-30)); row[f"{a}_fine_{nm}"] = fine_share(Gk, bp)
                row[f"{a}_fine"] = fine_share(Gf, bp); row[f"{a}_local"] = float(np.abs(Gf)[ball].sum() / (np.abs(Gf)[bp].sum() + 1e-30))
                for nm in SV:
                    sv[f"{a}_{nm}"].append(G[a, nm][i][v[0]])
            sv["img"].append(ip[v[0]]); sv["gt"].append(gp[v[0]] == int(r.cls) if int(r.cls) else (gp[v[0]] > 0)); sv["z"].append(v[0]); sv["pattern"].append(r.pattern)
            rows.append(row)
        np.savez_compressed(od / f"{case}_pg.npz", seeds=np.array(V), pattern=np.array(sv.pop("pattern")), z=np.array(sv.pop("z")),
                            **{k: np.array(vv).astype(np.float16 if k != "gt" else bool) for k, vv in sv.items()})
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)
        R = pd.DataFrame(rows).groupby("pattern")[[f"{a}_share_p0" for a in ARMS]].median()
        log(f"[{key}] {case}: {len(d)} seeds ({n_paths} paths, additivity max rel. err {max(add_err):.1e}); share of path 0, noise/flat/real: "
            + "; ".join(f"{p} " + "/".join(f"{R.loc[p, f'{a}_share_p0']:.2f}" for a in ARMS) for p in R.index) + f"; {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------------------- report
def wp(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def report():
    cells = [k for k in ORDER if (OUT / k / "seeds.csv").exists() and (OUT / k / "seeds.csv").stat().st_size > 20]
    S = pd.concat([pd.read_csv(OUT / k / "seeds.csv", dtype=str) for k in cells], ignore_index=True); S = S[S.case != "case"]
    num = [c for c in S.columns if c not in ("cell", "task", "case", "region", "pattern")]; S[num] = S[num].apply(pd.to_numeric, errors="coerce")
    St = S[S.region == "target"]
    L = ["# Path-decomposed input gradients (FINDINGS section 37): share of each voxel's decision carried by the full-resolution skip (path 0)", "",
         "Seeds = section-36 voxels (outcome pattern over noise, flat, real; C correct / F failed by log-odds margin 2), <= 24 per case, rare patterns first. Gradient of each arm's OWN predicted class, "
         "fold-mean. share_p0 = <G0, G> / |G|^2 (path shares sum to 1); e_p0 = |G0|^2 / |G|^2. Units: per-case median over seeds, median over cases, Wilcoxon over cases.", ""]
    # per cell, all target seeds
    rows = []
    for k in cells:
        d = St[St.cell == k]; pc = d.groupby("case").median(numeric_only=True)
        r = dict(cell=k, task=d.task.iloc[0], n_cases=len(pc), n_seeds=len(d))
        for a in ARMS:
            for m in [m_ for m_ in ("share_p0", "share_p1", "share_pr", "e_p0", "local_p0", "local_pr", "fine_p0", "fine_pr", "fine") if f"{a}_{m_}" in pc.columns]:
                r[f"{a}_{m}"] = pc[f"{a}_{m}"].median()
        for m in ("share_p0", "e_p0"):
            for b in ("flat", "noise"):
                dd = pc[f"real_{m}"] - pc[f"{b}_{m}"]; r[f"d_{m}_r{b[0]}"] = dd.median(); r[f"frac_{m}_r{b[0]}"] = (dd > 0).mean(); r[f"p_{m}_r{b[0]}"] = wp(dd)
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(OUT / "pathgrad_cells.csv", index=False)
    L += ["## 1. Share of the decision carried by the full-resolution path, per cell (all target seeds)", "",
          "| cell | task | cases | share0 noise | share0 flat | share0 real | real - flat | frac > 0 | p | real - noise | frac > 0 | p | e0 noise/flat/real | share1 noise/flat/real |", "|" + "---|" * 14]
    for r in T.itertuples():
        L.append(f"| {r.cell} | {r.task} | {r.n_cases} | {r.noise_share_p0:.3f} | {r.flat_share_p0:.3f} | {r.real_share_p0:.3f} | {r.d_share_p0_rf:+.3f} | {r.frac_share_p0_rf:.2f} | {r.p_share_p0_rf:.2g} | "
                 f"{r.d_share_p0_rn:+.3f} | {r.frac_share_p0_rn:.2f} | {r.p_share_p0_rn:.2g} | {r.noise_e_p0:.3f}/{r.flat_e_p0:.3f}/{r.real_e_p0:.3f} | {r.noise_share_p1:.3f}/{r.flat_share_p1:.3f}/{r.real_share_p1:.3f} |")
    L += ["", f"real > flat (share0) in {(T.d_share_p0_rf > 0).sum()} / {len(T)} cells, p < 0.05 same sign {((T.p_share_p0_rf < 0.05) & (T.d_share_p0_rf > 0)).sum()} +, {((T.p_share_p0_rf < 0.05) & (T.d_share_p0_rf < 0)).sum()} -; "
          f"real > noise in {(T.d_share_p0_rn > 0).sum()} / {len(T)}, p < 0.05 {((T.p_share_p0_rn < 0.05) & (T.d_share_p0_rn > 0)).sum()} +, {((T.p_share_p0_rn < 0.05) & (T.d_share_p0_rn < 0)).sum()} -.", ""]
    # per pattern
    L += ["## 2. Per outcome pattern (noise, flat, real), share0 per arm and real - flat, cells with >= 3 cases", ""]
    rows = []
    for p in PATS:
        for k in cells:
            d = St[(St.cell == k) & (St.pattern == p)]
            if d.case.nunique() < 3:
                continue
            pc = d.groupby("case").median(numeric_only=True); dd = pc.real_share_p0 - pc.flat_share_p0; dn = pc.real_share_p0 - pc.noise_share_p0
            rows.append(dict(pattern=p, cell=k, task=d.task.iloc[0], n=len(pc), noise=pc.noise_share_p0.median(), flat=pc.flat_share_p0.median(), real=pc.real_share_p0.median(), d_rf=dd.median(), frac_rf=(dd > 0).mean(), p_rf=wp(dd), d_rn=dn.median(), p_rn=wp(dn)))
    P = pd.DataFrame(rows); P.to_csv(OUT / "pathgrad_patterns.csv", index=False)
    for p in PATS:
        q = P[P.pattern == p]
        if not len(q):
            continue
        L += [f"### {p}", "", "| cell | n | noise | flat | real | real - flat | frac > 0 | p | real - noise | p |", "|" + "---|" * 10]
        L += [f"| {r.cell} | {r.n} | {r.noise:.3f} | {r.flat:.3f} | {r.real:.3f} | {r.d_rf:+.3f} | {r.frac_rf:.2f} | {r.p_rf:.2g} | {r.d_rn:+.3f} | {r.p_rn:.2g} |" for r in q.itertuples()]
        L += [f"pooled: real > flat in {(q.d_rf > 0).sum()} / {len(q)} cells (p < 0.05: {((q.p_rf < 0.05) & (q.d_rf > 0)).sum()} +, {((q.p_rf < 0.05) & (q.d_rf < 0)).sum()} -)", ""]
    # within-arm: FFC vs CCC for real
    L += ["## 3. Within the real arm: share0 at only-real voxels minus share0 at all-correct voxels (same cases)", "", "| cell | n | d | frac > 0 | p |", "|---|---|---|---|---|"]
    for k in cells:
        d = St[St.cell == k]; a1 = d[d.pattern == "FFC"].groupby("case").real_share_p0.median(); a2 = d[d.pattern == "CCC"].groupby("case").real_share_p0.median(); cs = a1.index.intersection(a2.index)
        if len(cs) >= 3:
            dd = a1.loc[cs] - a2.loc[cs]; L.append(f"| {k} | {len(cs)} | {dd.median():+.3f} | {(dd > 0).mean():.2f} | {wp(dd):.2g} |")
    # figures
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axs[0]
    for i, r in T.iterrows():
        col = vt.TCOL[r.task]; ax.plot([0, 1, 2], [r.flat_share_p0, r.real_share_p0, r.noise_share_p0], "-o", color=col, lw=0.9, ms=3.5, alpha=0.85); ax.annotate(r.cell, (2.05, r.noise_share_p0), fontsize=4.5, va="center")
    ax.set_xlim(-0.3, 3.0); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["flat fill", "real fill", "noise fill"], fontsize=8); ax.set_ylabel("share of the voxel's decision through the full-resolution path", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title(f"per cell (median over cases): real > flat in {(T.d_share_p0_rf > 0).sum()} / {len(T)} cells", fontsize=8.5)
    ax = axs[1]; W = St.groupby(["cell", "task", "case"])[[f"{a}_share_p0" for a in ARMS]].median().reset_index()
    for t in vt.TASKS:
        w = W[W.task == t]
        if len(w):
            ax.scatter(w.flat_share_p0, w.real_share_p0, s=9, color=vt.TCOL[t], alpha=0.75, lw=0, label=f"{t} (n={len(w)})")
    lim = [W[[f"{a}_share_p0" for a in ("flat", "real")]].min().min() - 0.02, W[[f"{a}_share_p0" for a in ("flat", "real")]].max().max() + 0.02]; ax.plot(lim, lim, "k-", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("flat-fill model: full-resolution-path share", fontsize=8); ax.set_ylabel("real-fill model: same", fontsize=8); ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=7)
    ax.set_title(f"per case: real > flat in {(W.real_share_p0 > W.flat_share_p0).mean() * 100:.0f} % of {len(W)} cases", fontsize=8.5)
    fig.suptitle("Path-decomposed input gradient: how much of each voxel's decision flows through the full-resolution skip (no pooling), own-decision gradients, section-36 seeds, all cells", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "pathgrad_summary.png", dpi=180); fig.savefig(OUT / "pathgrad_summary.pdf"); plt.close(fig)
    fig, axs = plt.subplots(1, len(vt.TASKS), figsize=(2.1 * len(vt.TASKS), 2.8), squeeze=False)
    for j, t in enumerate(vt.TASKS):
        ax = axs[0, j]; d = St[St.task == t]
        for pi, p in enumerate(PATS):
            dp = d[d.pattern == p]
            if not len(dp):
                continue
            pc = dp.groupby(["cell", "case"])[[f"{a}_share_p0" for a in ARMS]].median()
            for ai, a in enumerate(ARMS):
                y = pc[f"{a}_share_p0"].dropna()
                if len(y):
                    ax.errorbar(pi + (ai - 1) * 0.25, y.median(), yerr=[[y.median() - y.quantile(0.25)], [y.quantile(0.75) - y.median()]], fmt="o", ms=3, color=PAL[a], lw=0.7)
            ax.text(pi, ax.get_ylim()[0], str(len(pc)), fontsize=4.5, ha="center", va="bottom")
        ax.set_xticks(range(len(PATS))); ax.set_xticklabels(PATS, fontsize=5, rotation=90); ax.tick_params(labelsize=6); ax.set_title(t, fontsize=8)
        if j == 0:
            ax.set_ylabel("full-resolution-path share", fontsize=7)
    fig.suptitle("Full-resolution-path share by outcome pattern (noise, flat, real) per task: median and IQR over cases; green noise, blue flat, red real", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT / "pathgrad_patterns.png", dpi=160); plt.close(fig)
    # depth profile (all-paths mode): share of the decision per skip depth, per arm, per task and per cell
    PNS = sorted({c.split("_share_")[1] for c in St.columns if "_share_p" in c and c.split("_share_")[1] != "pr"}, key=lambda s: int(s[1:]))
    if len(PNS) > 2:
        rows = []
        for k in cells:
            d = St[St.cell == k]; pc = d.groupby("case").median(numeric_only=True); r = dict(cell=k, task=d.task.iloc[0], n=len(pc))
            for a in ARMS:   # cancellation index: sum of single-path energies over the full energy minus 1 (0 = orthogonal paths; > 0 = paths cancel each other; literature review: report next to the shares)
                r[f"{a}_cancel"] = (pc[[f"{a}_e_{p}" for p in PNS]].sum(1) - 1).median()
            for p in PNS:
                for a in ARMS:
                    r[f"{a}_{p}"] = pc[f"{a}_share_{p}"].median()
                dd = pc[f"real_share_{p}"] - pc[f"flat_share_{p}"]; r[f"d_rf_{p}"] = dd.median(); r[f"p_rf_{p}"] = wp(dd)
                dn = pc[f"real_share_{p}"] - pc[f"noise_share_{p}"]; r[f"d_rn_{p}"] = dn.median(); r[f"p_rn_{p}"] = wp(dn)
            rows.append(r)
        D = pd.DataFrame(rows); D.to_csv(OUT / "pathgrad_depth_cells.csv", index=False)
        L += ["", "## 4. Share of the decision per skip depth (p0 = full resolution ... deepest = bottleneck), per cell: noise / flat / real, then real - flat (p)", "",
              "| cell | n | " + " | ".join(PNS) + " | " + " | ".join(f"d {p}" for p in PNS) + " | cancel n/f/r |", "|" + "---|" * (3 + 2 * len(PNS))]
        for r in D.itertuples():
            L.append(f"| {r.cell} | {r.n} | " + " | ".join(f"{getattr(r, f'noise_{p}'):.2f}/{getattr(r, f'flat_{p}'):.2f}/{getattr(r, f'real_{p}'):.2f}" for p in PNS) + " | "
                     + " | ".join(f"{getattr(r, f'd_rf_{p}'):+.3f} ({getattr(r, f'p_rf_{p}'):.1g})" for p in PNS) + f" | {r.noise_cancel:.2f}/{r.flat_cancel:.2f}/{r.real_cancel:.2f} |")
        L += ["", "cells with real > flat per depth: " + ", ".join(f"{p}: {(D[f'd_rf_{p}'] > 0).sum()}/{len(D)} (p<0.05 {((D[f'p_rf_{p}'] < 0.05) & (D[f'd_rf_{p}'] > 0)).sum()}+ {((D[f'p_rf_{p}'] < 0.05) & (D[f'd_rf_{p}'] < 0)).sum()}-)" for p in PNS),
              "cells with real > noise per depth: " + ", ".join(f"{p}: {(D[f'd_rn_{p}'] > 0).sum()}/{len(D)} (p<0.05 {((D[f'p_rn_{p}'] < 0.05) & (D[f'd_rn_{p}'] > 0)).sum()}+ {((D[f'p_rn_{p}'] < 0.05) & (D[f'd_rn_{p}'] < 0)).sum()}-)" for p in PNS)]
        fig, axs = plt.subplots(2, len(vt.TASKS), figsize=(2.1 * len(vt.TASKS), 5.2), squeeze=False)
        for j, t in enumerate(vt.TASKS):
            d = St[St.task == t]; pc = d.groupby(["cell", "case"]).median(numeric_only=True)
            ax = axs[0, j]
            for a in ARMS:
                y = [pc[f"{a}_share_{p}"].median() for p in PNS]; lo = [pc[f"{a}_share_{p}"].quantile(0.25) for p in PNS]; hi = [pc[f"{a}_share_{p}"].quantile(0.75) for p in PNS]
                ax.plot(range(len(PNS)), y, "-o", color=PAL[a], ms=3, lw=1); ax.fill_between(range(len(PNS)), lo, hi, color=PAL[a], alpha=0.12, lw=0)
            ax.set_title(f"{t} (n={len(pc)})", fontsize=8); ax.set_xticks(range(len(PNS))); ax.set_xticklabels(PNS, fontsize=6); ax.tick_params(labelsize=6); ax.axhline(0, color="0.7", lw=0.5)
            if j == 0:
                ax.set_ylabel("share of the decision", fontsize=7)
            ax = axs[1, j]; dd = [(pc[f"real_share_{p}"] - pc[f"flat_share_{p}"]) for p in PNS]; dn = [(pc[f"real_share_{p}"] - pc[f"noise_share_{p}"]) for p in PNS]
            ax.plot(range(len(PNS)), [x.median() for x in dd], "-o", color=PAL["flat"], ms=3, lw=1, label="real - flat"); ax.fill_between(range(len(PNS)), [x.quantile(0.25) for x in dd], [x.quantile(0.75) for x in dd], color=PAL["flat"], alpha=0.12, lw=0)
            ax.plot(range(len(PNS)), [x.median() for x in dn], "-o", color=PAL["noise"], ms=3, lw=1, label="real - noise"); ax.fill_between(range(len(PNS)), [x.quantile(0.25) for x in dn], [x.quantile(0.75) for x in dn], color=PAL["noise"], alpha=0.12, lw=0)
            ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(PNS))); ax.set_xticklabels(PNS, fontsize=6); ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel("real minus other arm", fontsize=7); ax.legend(fontsize=6, frameon=False)
        fig.suptitle("Where in the U-Net does the decision travel? Share of each voxel's input gradient per skip depth (p0 full resolution, last = bottleneck), own-decision gradients, median and IQR over cases; green noise, blue flat, red real", fontsize=7.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / "pathgrad_depth.png", dpi=170); fig.savefig(OUT / "pathgrad_depth.pdf"); plt.close(fig)
        # section 5: spread (local share of |Gk| within r_loc of the voxel) and grain (fine-detail share of Gk) per depth: share answers HOW MUCH, these answer HOW COMPACT and HOW FINE
        rows = []
        for k in cells:
            d = St[St.cell == k]; pc = d.groupby("case").median(numeric_only=True); r = dict(cell=k, task=d.task.iloc[0], n=len(pc))
            for m in ("local", "fine"):
                for p in PNS:
                    for a in ARMS:
                        r[f"{a}_{m}_{p}"] = pc[f"{a}_{m}_{p}"].median()
                    dd = pc[f"real_{m}_{p}"] - pc[f"flat_{m}_{p}"]; r[f"d_rf_{m}_{p}"] = dd.median(); r[f"p_rf_{m}_{p}"] = wp(dd)
                    dn = pc[f"real_{m}_{p}"] - pc[f"noise_{m}_{p}"]; r[f"d_rn_{m}_{p}"] = dn.median(); r[f"p_rn_{m}_{p}"] = wp(dn)
            rows.append(r)
        E = pd.DataFrame(rows); E.to_csv(OUT / "pathgrad_depth_localfine_cells.csv", index=False)
        for m, title in (("local", "spread: fraction of |Gk| within r_loc of the voxel (1 = compact)"), ("fine", "grain: fine-detail share of Gk (energy above a sigma-1 Gaussian)")):
            L += ["", f"## 5{'a' if m == 'local' else 'b'}. Per depth, {title}; noise / flat / real, then real - flat (p)", "",
                  "| cell | n | " + " | ".join(PNS) + " | " + " | ".join(f"d {p}" for p in PNS) + " |", "|" + "---|" * (2 + 2 * len(PNS))]
            for r in E.itertuples():
                L.append(f"| {r.cell} | {r.n} | " + " | ".join(f"{getattr(r, f'noise_{m}_{p}'):.2f}/{getattr(r, f'flat_{m}_{p}'):.2f}/{getattr(r, f'real_{m}_{p}'):.2f}" for p in PNS) + " | "
                         + " | ".join(f"{getattr(r, f'd_rf_{m}_{p}'):+.3f} ({getattr(r, f'p_rf_{m}_{p}'):.1g})" for p in PNS) + " |")
            L += ["", f"cells with real > flat ({m}) per depth: " + ", ".join(f"{p}: {(E[f'd_rf_{m}_{p}'] > 0).sum()}/{len(E)} (p<0.05 {((E[f'p_rf_{m}_{p}'] < 0.05) & (E[f'd_rf_{m}_{p}'] > 0)).sum()}+ {((E[f'p_rf_{m}_{p}'] < 0.05) & (E[f'd_rf_{m}_{p}'] < 0)).sum()}-)" for p in PNS),
                  f"cells with real > noise ({m}) per depth: " + ", ".join(f"{p}: {(E[f'd_rn_{m}_{p}'] > 0).sum()}/{len(E)} (p<0.05 {((E[f'p_rn_{m}_{p}'] < 0.05) & (E[f'd_rn_{m}_{p}'] > 0)).sum()}+ {((E[f'p_rn_{m}_{p}'] < 0.05) & (E[f'd_rn_{m}_{p}'] < 0)).sum()}-)" for p in PNS)]
        fig, axs = plt.subplots(2, len(vt.TASKS), figsize=(2.1 * len(vt.TASKS), 5.2), squeeze=False)
        for j, t in enumerate(vt.TASKS):
            d = St[St.task == t]; pc = d.groupby(["cell", "case"]).median(numeric_only=True)
            for i, (m, yl) in enumerate((("local", "spread: local share of |Gk|"), ("fine", "grain: fine-detail share of Gk"))):
                ax = axs[i, j]
                for a in ARMS:
                    y = [pc[f"{a}_{m}_{p}"].median() for p in PNS]; lo = [pc[f"{a}_{m}_{p}"].quantile(0.25) for p in PNS]; hi = [pc[f"{a}_{m}_{p}"].quantile(0.75) for p in PNS]
                    ax.plot(range(len(PNS)), y, "-o", color=PAL[a], ms=3, lw=1); ax.fill_between(range(len(PNS)), lo, hi, color=PAL[a], alpha=0.12, lw=0)
                ax.set_xticks(range(len(PNS))); ax.set_xticklabels(PNS, fontsize=6); ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(f"{t} (n={len(pc)})", fontsize=8)
                if j == 0:
                    ax.set_ylabel(yl, fontsize=7)
        fig.suptitle("Per skip depth: how compact (top) and how fine-grained (bottom) each path's gradient map is; median and IQR over cases; green noise, blue flat, red real", fontsize=7.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / "pathgrad_depth_localfine.png", dpi=170); fig.savefig(OUT / "pathgrad_depth_localfine.pdf"); plt.close(fig)
    for k in [c for c in ("glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t2f", "breast_t2w_t1wce", "pancreas_t1wce_t2w", "pelvis_ct_mri", "mand_cbct_hanseg_ct", "abd_t1in_ct") if c in cells]:
        examples(k, St[St.cell == k])
    (OUT / "pathgrad.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:8 + len(cells)])); print("PATHGRAD_DONE")


def examples(key, d, W=30):
    picks = []
    for p in ("CCC", "FFC", "FCF", "CFF", "FFF"):
        dp = d[d.pattern == p]
        if len(dp):
            dd = dp.real_share_p0 - dp.flat_share_p0; picks.append((p, dp.loc[(dd - dd.median()).abs().idxmin()]))   # the typical seed of the pattern
    if not picks:
        return
    z0 = np.load(OUT / key / f"{picks[0][1].case}_pg.npz"); parts = ["full", "p0", "p1", "p2", "p3"] if "flat_p3" in z0.files else ["full", "p0", "pr"]
    NAMES = {"full": "full", "p0": "path 0", "p1": "path 1", "p2": "path 2", "p3": "path 3", "pr": "pooled paths"}
    cols = ["image"] + [f"{a}: {NAMES[nm]}" for a in ("flat", "real") for nm in parts]
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.0 * len(cols), 2.1 * len(picks)), squeeze=False)
    for i, (p, s) in enumerate(picks):
        z = np.load(OUT / key / f"{s.case}_pg.npz")
        sd = pd.read_csv(OUT / key / "seeds.csv"); sd = sd[sd.case == s.case].reset_index(drop=True); j = int(sd.index[sd.seed == s.seed][0])
        v = (int(z["seeds"][j][1]), int(z["seeds"][j][2])); sl = (slice(max(0, v[0] - W), v[0] + W), slice(max(0, v[1] - W), v[1] + W))
        img = z["img"][j].astype(np.float32); g = z["gt"][j]; lo_, hi_ = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99])
        for c, nm in enumerate(cols):
            ax = axs[i, c]; ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(img[sl], cmap="gray", vmin=lo_, vmax=hi_); ax.set_ylabel(f"{p}\n{s.case}", fontsize=6)
            else:
                a, part = nm.split(": "); key_ = {v_: k_ for k_, v_ in NAMES.items()}[part]; M = np.abs(z[f"{a}_{key_}"][j].astype(np.float32))
                vmax = np.percentile(np.abs(z[f"{a}_full"][j].astype(np.float32))[sl], 99.5); ax.imshow(M[sl], cmap="magma", vmin=0, vmax=vmax)
                ax.set_title(f"{nm}\n{'correct' if p[1 if a == 'flat' else 2] == 'C' else 'FAILED'}; share0 {getattr(s, f'{a}_share_p0'):.2f}" if part == "full" else nm, fontsize=5.5, color=PAL[a])
            ax.contour(g[sl], [0.5], colors="#00b4ff" if c == 0 else "white", linewidths=0.5); ax.plot(v[1] - sl[1].start, v[0] - sl[0].start, "+", color="cyan", ms=6, mew=0.8)
    fig.suptitle(f"{key}: |input gradient| of each arm's own decision at the voxel (+), split into the full-resolution-skip path and the pooled paths (shared scale per arm and row). Contour = GT.", fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"pathgrad_examples_{key}.png", dpi=170); plt.close(fig)


PICKS = OUT / "pathgrad_task_examples_picks.csv"


def pick_examples(n_ex=3, ref="p2"):
    """Per task up to n_ex example voxels (different cells first, then different cases) where the real arm is correct, the one-arm-only patterns first; the
    typical seed of its pool by real - flat share of path `ref`. Deterministic; written to PICKS so the GPU pass and the figure agree."""
    cells = [k for k in ORDER if (OUT / k / "seeds.csv").exists() and (OUT / k / "seeds.csv").stat().st_size > 20]
    S = pd.concat([pd.read_csv(OUT / k / "seeds.csv", dtype=str) for k in cells], ignore_index=True); S = S[S.case != "case"]
    num = [c for c in S.columns if c not in ("cell", "task", "case", "region", "pattern")]; S[num] = S[num].apply(pd.to_numeric, errors="coerce")
    St = S[(S.region == "target") & S.pattern.str.endswith("C")]
    picks = []
    for t in vt.TASKS:
        d = St[St.task == t]; got = []
        for prio in (("FFC", "FCC", "CFC"), ("CCC",), tuple(sorted(d.pattern.unique()))):
            for k in [c for c in ORDER if c in set(d.cell)]:
                dp = d[(d.cell == k) & d.pattern.isin(prio) & ~d.case.isin([g.case for g in got])]
                if len(dp) and len(got) < n_ex:
                    dd = dp[f"real_share_{ref}"] - dp[f"flat_share_{ref}"]; got.append(dp.loc[(dd - dd.median()).abs().idxmin()])
            if len(got) >= n_ex:
                break
        picks += got[:n_ex]
    P = pd.DataFrame(picks); P.to_csv(PICKS, index=False); print(f"{len(P)} example seeds -> {PICKS}"); print(P[["task", "cell", "case", "seed", "pattern"]].to_string()); return P


def example_grads(key, dev):
    """GPU: for this cell's picked seeds, the fold-mean gradient map of the flat and real decisions through EVERY skip (p0..p5) at the seed slice -> <case>_pgex.npz."""
    import torch
    P = pd.read_csv(PICKS, dtype=str); P = P[P.cell == key]
    if not len(P):
        return
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key
    SD = pd.read_csv(SEEDS_IN / key / "seeds.csv", dtype=str); SD = SD[SD.case != "case"]
    prs = {a: v2.load_predictor(d, dev) for a, d in vo.models(key).items() if a in ("flat", "real")}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    n_paths = len(pr0.network.encoder.stages)
    for case in P.case.unique():
        t0 = time.time(); rows = SD[(SD.case == case) & SD.seed.isin(P[P.case == case].seed)]
        data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        box = e7.patch_at(tg, ps, img.shape); ip, gp = img[box].astype(np.float32), gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, gp = np.pad(ip, padw), np.pad(gp, padw)
        V = [(int(r.z), int(r.y), int(r.x)) for r in rows.itertuples()]; out = dict(seeds=np.array(V), seed=rows.seed.astype(int).values, pattern=rows.pattern.values,
                                                                                img=np.array([ip[v[0]] for v in V]), gt=np.array([(gp[v[0]] == int(r.cls)) if int(r.cls) else (gp[v[0]] > 0) for v, r in zip(V, rows.itertuples())]))
        for a, pr in prs.items():
            net = pr.network; cls = [int(getattr(r, f"{a}_pred")) for r in rows.itertuples()]
            for nm, keep in [("full", None)] + [(f"p{j}", {j}) for j in range(n_paths)]:
                acc = None
                for k in FOLDS:
                    net.load_state_dict(pr.list_of_parameters[k]); net.eval(); g = path_grads(net, ip, V, cls, dev, keep); acc = g if acc is None else [x_ + y_ for x_, y_ in zip(acc, g)]
                out[f"{a}_{nm}"] = np.array([(x_ / len(FOLDS))[v[0]].cpu().numpy() for x_, v in zip(acc, V)]).astype(np.float16)
            torch.cuda.empty_cache()
        np.savez_compressed(od / f"{case}_pgex.npz", **out); log(f"[{key}] {case}: {len(V)} example seeds, {n_paths} paths, flat + real; {time.time() - t0:.0f} s")


def task_figure(W=40, paths=("p2", "p3", "p4", "p5")):
    """One figure for all tasks from PICKS and the <case>_pgex.npz maps: columns image + GT contour, then for each path the flat and the real map (|Gk| of each
    arm's own decision); one colour scale shared by all maps of a row (99.5th pct of their max in the window)."""
    P = pd.read_csv(PICKS, dtype=str); allp = [r for r in P.itertuples() if (OUT / r.cell / f"{r.case}_pgex.npz").exists()]
    if not allp:
        print("no example maps yet"); return
    for t in vt.TASKS:   # one figure per task (24 rows in one figure is unreadable) + one with the first example of every task
        _task_fig([r for r in allp if r.task == t], W, paths, OUT / f"pathgrad_task_examples_{t}.png")
    _task_fig([next(r for r in allp if r.task == t) for t in vt.TASKS if any(r.task == t for r in allp)], W, paths, OUT / "pathgrad_task_examples_all.png"); print("TASKFIG_DONE")


def _task_fig(picks, W, paths, f_out):
    if not picks:
        return
    cols = ["image + GT"] + [f"{a}: path {p[1]}" for p in paths for a in ("flat", "real")]
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(1.9 * len(cols), 2.0 * len(picks)), squeeze=False)
    for i, s in enumerate(picks):
        z = np.load(OUT / s.cell / f"{s.case}_pgex.npz", allow_pickle=True); j = int(np.where(z["seed"] == int(s.seed))[0][0])
        v = (int(z["seeds"][j][1]), int(z["seeds"][j][2])); sl = (slice(max(0, v[0] - W), v[0] + W), slice(max(0, v[1] - W), v[1] + W))
        img = z["img"][j].astype(np.float32); g = z["gt"][j]; lo_, hi_ = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99])
        maps = {(a, p): np.abs(z[f"{a}_{p}"][j].astype(np.float32)) for p in paths for a in ("flat", "real")}
        vmax = max(np.percentile(M[sl], 99.5) for M in maps.values()) + 1e-12
        for c, nm in enumerate(cols):
            ax = axs[i, c]; ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(img[sl], cmap="gray", vmin=lo_, vmax=hi_); ax.set_ylabel(f"{s.task}\n{s.cell}\n{s.case} ({s.pattern})", fontsize=5.5)
                ax.contour(g[sl], [0.5], colors="#00b4ff", linewidths=0.6)
            else:
                a, part = nm.split(": "); p = "p" + part[-1]; ax.imshow(maps[(a, p)][sl], cmap="magma", vmin=0, vmax=vmax)
                ax.set_title(f"{nm}  share {float(getattr(s, f'{a}_share_{p}')):.2f}", fontsize=5.5, color=PAL[a])
                ax.contour(g[sl], [0.5], colors="white", linewidths=0.4)
            ax.plot(v[1] - sl[1].start, v[0] - sl[0].start, "+", color="cyan", ms=6, mew=0.8)
    fig.suptitle("Per task: |input gradient| of the decision at the voxel (+) carried by the stride-4 (p2), stride-8 (p3), stride-16 (p4) and stride-32 (p5) skips, flat vs real fill; one colour scale per row. Contour = GT.", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(f_out, dpi=170); plt.close(fig)


def zoom_figure(W=16, n_per_task=2):
    """Image-based view of the grain result: for a few picked seeds per task, a +/- W voxel crop of the image with the GT contour, the image's own fine detail
    (x - gaussian(x, 1)), and the FULL input gradient of the flat and the real decision at the same voxel (|G|, one colour scale per row), plus the stride-4 path.
    Title of each gradient panel = its grain (high-pass energy share within the crop)."""
    P = pd.read_csv(PICKS, dtype=str); allp = [r for r in P.itertuples() if (OUT / r.cell / f"{r.case}_pgex.npz").exists()]
    picks = []
    for t in dict.fromkeys(r.task for r in allp):
        picks += [r for r in allp if r.task == t][:n_per_task]
    cols = ["image + GT", "image fine detail", "flat: |G| (all paths)", "real: |G| (all paths)", "flat: path 2", "real: path 2"]
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.2 * len(cols), 2.25 * len(picks)), squeeze=False)
    for i, s in enumerate(picks):
        z = np.load(OUT / s.cell / f"{s.case}_pgex.npz", allow_pickle=True); j = int(np.where(z["seed"] == int(s.seed))[0][0])
        v = (int(z["seeds"][j][1]), int(z["seeds"][j][2])); sl = (slice(max(0, v[0] - W), v[0] + W), slice(max(0, v[1] - W), v[1] + W))
        img = z["img"][j].astype(np.float32); g = z["gt"][j]; lo_, hi_ = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99])
        hp = img - ndi.gaussian_filter(img, 1.0); hp[img == 0] = 0
        maps = {nm: np.abs(z[f"{nm.split(':')[0]}_{'full' if 'all' in nm else 'p2'}"][j].astype(np.float32)) for nm in cols[2:]}
        vmax = max(np.percentile(M[sl], 99.5) for M in maps.values()) + 1e-12
        for c, nm in enumerate(cols):
            ax = axs[i, c]; ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(img[sl], cmap="gray", vmin=lo_, vmax=hi_, interpolation="nearest"); ax.contour(g[sl], [0.5], colors="#00b4ff", linewidths=0.7)
                ax.set_ylabel(f"{s.task}\n{s.cell}\n{s.case} ({s.pattern})", fontsize=5.5)
            elif c == 1:
                a_ = np.percentile(np.abs(hp[sl]), 99) + 1e-9; ax.imshow(hp[sl], cmap="gray", vmin=-a_, vmax=a_, interpolation="nearest"); ax.set_title("x - gaussian(x, 1)", fontsize=6)
            else:
                M = maps[nm]; crop = M[sl]; grain = float(((crop - ndi.gaussian_filter(crop, 1.0)) ** 2).sum() / ((crop ** 2).sum() + 1e-30))
                ax.imshow(crop, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"); ax.contour(g[sl], [0.5], colors="white", linewidths=0.4)
                ax.set_title(f"{nm}\ngrain {grain:.2f}", fontsize=6, color=PAL[nm.split(":")[0]])
            ax.plot(v[1] - sl[1].start, v[0] - sl[0].start, "+", color="cyan", ms=7, mew=0.9)
    fig.suptitle(f"Zoom (+/- {W} voxels) on the decision voxel (+): the image, its own fine detail, and the input gradient of the flat vs the real decision (full gradient and the stride-4 path); "
                 "one colour scale per row; grain = high-pass energy share of the crop. Contour = GT.", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(OUT / "pathgrad_zoom_examples.png", dpi=170); plt.close(fig); print("ZOOMFIG_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    ap.add_argument("--pick-examples", action="store_true"); ap.add_argument("--example-grads", action="store_true"); ap.add_argument("--task-figure", action="store_true")
    ap.add_argument("--zoom-figure", action="store_true")
    a = ap.parse_args()
    if a.pick_examples:
        pick_examples()
    if a.task_figure:
        task_figure()
    if a.zoom_figure:
        zoom_figure()
    if a.report:
        report()
    if a.report or a.task_figure or a.pick_examples or a.zoom_figure:
        return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in a.cells:
        example_grads(k, dev) if a.example_grads else run_cell(k, dev, a.case_slice)


if __name__ == "__main__":
    main()
