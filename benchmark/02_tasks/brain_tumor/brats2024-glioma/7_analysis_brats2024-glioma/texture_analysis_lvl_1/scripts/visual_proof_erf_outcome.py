#!/usr/bin/env python
"""Visual proof v9 (2026-10-13, FINDINGS section 36): per-voxel evidence maps STRATIFIED BY OUTCOME across the three fill arms (noise / flat / real).

Sections 27/28 only explained voxels that BOTH models segmented correctly and confidently, i.e. voxels where the flat-fill model succeeded too, so
texture could not have mattered there. Here every target voxel gets an outcome per arm from the fold-mean log-odds of its ground-truth class:
C (correct, log-odds > +CONF), F (failed, < -CONF), A (ambiguous, in between; counted, never explained). The pattern = the (noise, flat, real)
outcome string, e.g. FFC = only the real-fill model gets the voxel. Seeds are sampled per (pattern, depth band) so patterns are compared at
matched depth, and every arm is explained through the gradient of ITS OWN decision at the voxel (the predicted class; `_pred`), plus the
gradient of the ground-truth class (`_gt`, equal to `_pred` for a correct arm). Background voxels in a ring around the target (false-positive
patterns: outcome per arm = background log-odds > +CONF correct, < -CONF failed) are seeded the same way.
Per seed and arm: local_share (|G| mass within r_loc of the voxel), inside_share (|G| mass in the GT region of the explained class; for background
seeds the predicted target class' region), fine_share (energy of G - gaussian(G, 1) over energy of G, whole patch), fine_local (same within r = 8),
align (Spearman over body voxels in a 48-vox box of |hp G| with |hp image|), margin (log-odds of the GT class). Fold-mean gradients.
Outputs: outputs/plots/erf_outcome/<cell>/{seeds.csv, counts.csv, skipped.csv, <case>_erf.npz (seed slices, NOT committed)};
--report: erf_outcome.md, erf_outcome_counts.png, erf_outcome_metrics.png, erf_outcome_cells.csv, erf_outcome_examples_<cell>.png (patient slices, NOT committed).
GPU: .venv/bin/python visual_proof_erf_outcome.py --cells <cell> [--case-slice i:j]      CPU: --report
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
e7, v2 = vt.e7, vt.v2; FOLDS = (0, 1, 2)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
OUT = Path(os.environ.get("VP_ERFO_OUT") or (vt.OUT.parent / "erf_outcome")); OUT.mkdir(parents=True, exist_ok=True)
CONF = 2.0; K_PER = int(os.environ.get("ERFO_K", 2)); MAX_SEEDS = int(os.environ.get("ERFO_MAX_SEEDS", 40)); RING = 6; BOX = 24; R_FINE = 8
MAX_CASES = int(os.environ.get("ERFO_MAX_CASES", 40)); MIN_VOX = 50
PATS = [a + b + c for a in "CF" for b in "CF" for c in "CF"]   # order noise, flat, real
ORDER = ["glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t2f", "glioma_t2w_t1n", "ms_flair_t2w", "ms_flair_t1w", "ms_t1w_flair", "ms_t1w_t2w", "breast_t1wce_t2w", "breast_t2w_t1wce",
         "brain_t1w_t2w", "brain_t2w_t1w", "abd_t1in_t2spir", "abd_t1in_ct", "abd_t2spir_t1in", "abd_t2spir_t1out", "abd_t2spir_ct", "mand_cbct_hanseg_ct", "mand_cbct_pddca_ct",
         "pelvis_mri_ct", "pelvis_ct_mri", "pancreas_t1wce_t2w", "pancreas_t2w_t1wce"]
METS = ["local_share", "inside_share", "fine_share", "fine_local", "align"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def models(key):
    cell = vt.CELLS[key]; flat, real = cell["models"]["flat"](), cell["models"]["real"]()
    runs = sorted(p for p in flat.parents[2].glob("*lblvor*") if p.is_dir()); assert len(runs) == 1, (key, runs)
    noise = sorted(runs[0].glob("Dataset*/*__3d_fullres")); assert len(noise) == 1, (key, noise)
    return {"noise": noise[0], "flat": flat, "real": real}


def hp(x):
    return x - ndi.gaussian_filter(x, 1.0)


def seed_metrics(Gs, v, reg, body, r_loc, hpi):
    """Gs: signed fold-mean gradient; reg: region of the explained class; hpi: |hp(image)|."""
    G = np.abs(Gs); zz, yy, xx = np.indices(G.shape, sparse=True); r = np.sqrt((zz - v[0]) ** 2 + (yy - v[1]) ** 2 + (xx - v[2]) ** 2)
    tot = G[body].sum() + 1e-30; H = hp(Gs); fb = r <= R_FINE
    sl = tuple(slice(max(0, int(c) - BOX), int(c) + BOX) for c in v); bb = body[sl]
    al = spearmanr(np.abs(H[sl])[bb], hpi[sl][bb]).correlation if bb.sum() > 100 else np.nan
    return dict(local_share=float(G[(r <= r_loc) & body].sum() / tot), inside_share=float(G[reg & body].sum() / tot),
                fine_share=float((H[body] ** 2).sum() / ((Gs[body] ** 2).sum() + 1e-30)), fine_local=float((H[fb & body] ** 2).sum() / ((Gs[fb & body] ** 2).sum() + 1e-30)), align=float(al))


def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; r_loc, d_min = cell["r_loc"], cell["d_min"]; od = OUT / key; od.mkdir(exist_ok=True)
    prs = {a: v2.load_predictor(d, dev) for a, d in models(key).items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]) if key.startswith("glioma") else list(cell["cases"])[:MAX_CASES]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv, f_cnt, f_skip = od / "seeds.csv", od / "counts.csv", od / "skipped.csv"
    try:   # parallel chunks: the file may be mid-write at start-up
        done = set(pd.read_csv(f_cnt).case.astype(str)) if f_cnt.exists() and f_cnt.stat().st_size > 0 else set()
    except pd.errors.EmptyDataError:
        done = set()
    rng = np.random.default_rng(0)
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        if tg.sum() < MIN_VOX:
            pd.DataFrame([dict(case=case, why="target < MIN_VOX")]).to_csv(f_skip, mode="a", index=False, header=not f_skip.exists()); continue
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        hpi = np.abs(hp(ip))
        # fold-mean logits per arm -> predicted class, log-odds of every class
        LO, PRED = {}, {}
        for a, pr in prs.items():
            net = pr.network; acc = 0
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()
                acc = acc + o
            o = acc / len(FOLDS); PRED[a] = o.argmax(1)[0].cpu().numpy()
            LO[a] = np.stack([e7.log_odds(o, c)[0].cpu().numpy() for c in range(o.shape[1])])   # (C, Z, Y, X)
            del o, acc; torch.cuda.empty_cache()
        inner = np.zeros(gp.shape, bool); inner[tuple(slice(min(e7.EDGE, s // 4), s - min(e7.EDGE, s // 4)) for s in gp.shape)] = True   # margin capped at a quarter of the axis: T2spir / ISPY2-T2w / PanSeg-T2w patches are 32-40 deep
        tgp = np.isin(gp, T); present = [c for c in T if (gp == c).sum() >= MIN_VOX]
        if not present:
            pd.DataFrame([dict(case=case, why="no structure >= MIN_VOX in the patch")]).to_csv(f_skip, mode="a", index=False, header=not f_skip.exists()); continue
        regs = {c: e7.region_geometry(gp, c) for c in present}   # reg, din, rim
        # outcome per arm: target voxels judged on their own class, ring background voxels on class 0
        gtc = gp.copy(); gtc[~np.isin(gp, present)] = 0
        lo_gt = {a: np.take_along_axis(LO[a], gtc[None], 0)[0] for a in ARMS}
        st = {a: np.where(lo_gt[a] > CONF, 0, np.where(lo_gt[a] < -CONF, 2, 1)) for a in ARMS}   # 0 C, 1 A, 2 F
        code = st["noise"] * 9 + st["flat"] * 3 + st["real"]
        depth = np.zeros(gp.shape, np.float32)
        for c in present:
            depth[regs[c][0]] = regs[c][1][regs[c][0]]
        ring = bp & ~tgp & (ndi.distance_transform_edt(~tgp) <= RING)
        tv = bp & np.isin(gp, present) & inner & (depth >= 1)
        cnt = []
        for nm, m in (("target", tv), ("ring", ring & inner)):
            for cd in np.unique(code[m]):
                s = "".join("CAF"[d] for d in (cd // 9, cd // 3 % 3, cd % 3)); cnt.append(dict(case=case, region=nm, pattern=s, n=int((code[m] == cd).sum())))
        pd.DataFrame(cnt).to_csv(f_cnt, mode="a", index=False, header=not (f_cnt.exists() and f_cnt.stat().st_size > 0)); done.add(case)
        # seeds: per (pattern, depth band) for target voxels, per pattern for ring voxels; definitive outcomes only
        defin = ~np.isin(st["noise"], [1]) & ~np.isin(st["flat"], [1]) & ~np.isin(st["real"], [1])
        dq = np.quantile(depth[tv], [0, 1 / 3, 2 / 3, 1]) if tv.sum() else np.array([0, 0, 0, 0])
        seeds = []
        for p in PATS:
            cd = sum(("CF".index(ch) * 2) * w for ch, w in zip(p, (9, 3, 1)))
            m = tv & defin & (code == cd)
            for b in range(3):
                idx = np.argwhere(m & (depth >= dq[b]) & (depth <= dq[b + 1]))
                for i in rng.choice(len(idx), min(len(idx), K_PER), replace=False) if len(idx) else []:
                    seeds.append(dict(v=tuple(idx[i]), region="target", pattern=p, band=b))
            idx = np.argwhere(ring & inner & defin & (code == cd))
            if p != "CCC":   # ring voxels every arm calls background correctly are not interesting
                for i in rng.choice(len(idx), min(len(idx), K_PER), replace=False) if len(idx) else []:
                    seeds.append(dict(v=tuple(idx[i]), region="ring", pattern=p, band=-1))
        if len(seeds) > MAX_SEEDS:
            seeds = [seeds[i] for i in sorted(rng.choice(len(seeds), MAX_SEEDS, replace=False))]
        if not seeds:
            log(f"[{key}] {case}: no definitive seed"); continue
        V = [s["v"] for s in seeds]; cgt = [int(gtc[s["v"]]) for s in seeds]; cpred = {a: [int(PRED[a][s["v"]]) for s in seeds] for a in ARMS}
        # gradients: per arm, _gt (class = GT class) and _pred (class = the arm's own prediction, computed only where it differs)
        G = {}
        for a, pr in prs.items():
            net = pr.network; need = [i for i in range(len(V)) if cpred[a][i] != cgt[i]]; acc_gt = None; acc_pr = None
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                g = e7.seed_grads(net, ip, V, cgt, dev); acc_gt = g if acc_gt is None else [x + y for x, y in zip(acc_gt, g)]
                if need:
                    g = e7.seed_grads(net, ip, [V[i] for i in need], [cpred[a][i] for i in need], dev); acc_pr = g if acc_pr is None else [x + y for x, y in zip(acc_pr, g)]
            G[a, "gt"] = [(x / len(FOLDS)).cpu().numpy() for x in acc_gt]
            G[a, "pred"] = list(G[a, "gt"])
            for j, i in enumerate(need):
                G[a, "pred"][i] = (acc_pr[j] / len(FOLDS)).cpu().numpy()
            torch.cuda.empty_cache()
        rows, sv = [], {"img": [], "gt": [], "z": [], "pattern": []}
        for a in ARMS:
            sv[f"G_{a}"] = []
        for i, s in enumerate(seeds):
            v = s["v"]; c = cgt[i]
            row = dict(cell=key, task=cell["task"], case=case, seed=i, region=s["region"], pattern=s["pattern"], band=s["band"], z=v[0], y=v[1], x=v[2], cls=c, depth=float(depth[v]))
            for a in ARMS:
                row[f"{a}_margin"] = float(lo_gt[a][v]); row[f"{a}_pred"] = cpred[a][i]
                reg_gt = regs[c][0] if c in regs else tgp
                for kind in ("gt", "pred"):
                    cexp = c if kind == "gt" else cpred[a][i]; reg = regs[cexp][0] if cexp in regs else (tgp if cexp in T else reg_gt)
                    for mk, mv in seed_metrics(G[a, kind][i], v, reg, bp, r_loc, hpi).items():
                        row[f"{a}_{mk}_{kind}"] = mv
                sv[f"G_{a}"].append(np.abs(G[a, "pred"][i][v[0]]))
            sv["img"].append(ip[v[0]]); sv["gt"].append(gp[v[0]] == c if c else tgp[v[0]]); sv["z"].append(v[0]); sv["pattern"].append(s["pattern"] + ("" if s["region"] == "target" else "_ring"))
            rows.append(row)
        np.savez_compressed(od / f"{case}_erf.npz", seeds=np.array(V), cls=np.array(cgt), pattern=np.array(sv.pop("pattern")), z=np.array(sv.pop("z")),
                            **{k: np.array(vv).astype(np.float16 if k != "gt" else bool) for k, vv in sv.items()})
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0))
        pc = pd.Series([s["pattern"] for s in seeds if s["region"] == "target"]).value_counts().to_dict()
        log(f"[{key}] {case}: {len(seeds)} seeds, target patterns {pc}; {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------------------- report
def wp(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else np.nan


def report():
    cells = [k for k in ORDER if (OUT / k / "counts.csv").exists() and (OUT / k / "counts.csv").stat().st_size > 20]   # a cell whose file is empty (no case seeded) is left out
    C = pd.concat([pd.read_csv(OUT / k / "counts.csv").assign(cell=k) for k in cells], ignore_index=True)
    S = pd.concat([pd.read_csv(OUT / k / "seeds.csv", dtype=str) for k in cells if (OUT / k / "seeds.csv").exists() and (OUT / k / "seeds.csv").stat().st_size > 20], ignore_index=True); S = S[S.case != "case"]
    num = [c for c in S.columns if c not in ("cell", "task", "case", "region", "pattern")]; S[num] = S[num].apply(pd.to_numeric, errors="coerce")   # parallel chunks may repeat the header
    C = C[C.case != "case"]; C["n"] = pd.to_numeric(C.n, errors="coerce")
    S["cell"] = pd.Categorical(S.cell, cells); S = S.sort_values(["cell", "case", "seed"]); S["cell"] = S.cell.astype(str)
    L = ["# Per-voxel evidence by outcome pattern across the three fill arms (FINDINGS section 36)", "",
         f"Outcome per arm from the fold-mean log-odds of the voxel's ground-truth class: C > +{CONF}, F < -{CONF}, A ambiguous (counted, not explained). Pattern order: noise, flat, real. "
         f"Seeds: up to {K_PER} per (pattern, depth tertile) target voxels + {K_PER} per pattern ring-background voxels (false positives), max {MAX_SEEDS} per case. "
         "Each arm explained through the gradient of its OWN predicted class at the voxel (`pred`); `gt` = gradient of the ground-truth class. Units: per-case median over seeds, then median over cases; Wilcoxon over cases.", ""]
    # 1. pattern counts (voxel share per cell; definitive target voxels)
    Ct = C[C.region == "target"]; Ct = Ct[~Ct.pattern.str.contains("A")]
    tot = Ct.groupby("cell").n.sum(); P = Ct.pivot_table(index="cell", columns="pattern", values="n", aggfunc="sum", fill_value=0).reindex(cells).fillna(0)
    P = P.reindex(columns=[p for p in PATS if p in P.columns], fill_value=0); Pf = P.div(tot, axis=0)
    ncase = Ct[Ct.n > 0].groupby(["cell", "pattern"]).case.nunique().unstack().reindex(cells).reindex(columns=P.columns).fillna(0).astype(int)
    amb = C[(C.region == "target")].assign(a=lambda d: d.pattern.str.contains("A")).groupby("cell").apply(lambda d: d[d.a].n.sum() / d.n.sum()).reindex(cells)
    L += ["## 1. Outcome patterns of the target voxels (share of definitive voxels per cell; in brackets: cases contributing)", "",
          "| cell | definitive voxels | ambiguous share | " + " | ".join(P.columns) + " |", "|" + "---|" * (3 + len(P.columns))]
    for k in cells:
        L.append(f"| {k} | {int(tot.get(k, 0))} | {amb.get(k, np.nan):.2f} | " + " | ".join(f"{Pf.loc[k, p]:.3f} ({ncase.loc[k, p]})" for p in P.columns) + " |")
    Pf.to_csv(OUT / "erf_outcome_pattern_share.csv")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(cells) + 1.5))
    im = ax.imshow(Pf.values, cmap="Blues", vmin=0, vmax=min(1, np.nanmax(Pf.values[:, 1:]) if Pf.shape[1] > 1 else 1), aspect="auto")
    ax.set_xticks(range(len(P.columns))); ax.set_xticklabels([f"{p}\n" + {"CCC": "all", "FFC": "only real", "FCF": "only flat", "CFF": "only noise", "FFF": "none", "CCF": "real fails", "CFC": "flat fails", "FCC": "noise fails"}.get(p, "") for p in P.columns], fontsize=7)
    ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells, fontsize=7)
    for i in range(len(cells)):
        for j in range(len(P.columns)):
            ax.text(j, i, f"{Pf.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if Pf.values[i, j] > 0.5 * im.get_clim()[1] else "k")
    ax.set_title("share of definitive target voxels per outcome pattern (noise, flat, real); colour scale excludes CCC", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "erf_outcome_counts.png", dpi=170); plt.close(fig)
    # 2. metrics by pattern: per cell, per arm, _pred gradients; per-case medians
    St = S[S.region == "target"]
    rows = []
    for (k, p), d in St.groupby(["cell", "pattern"]):
        pc = d.groupby("case")[[f"{a}_{m}_{kind}" for a in ARMS for m in METS for kind in ("gt", "pred")] + [f"{a}_margin" for a in ARMS] + ["depth"]].median()
        r = dict(cell=k, task=d.task.iloc[0], pattern=p, n_seeds=len(d), n_cases=len(pc), depth=pc.depth.median())
        for a in ARMS:
            for m in METS:
                r[f"{a}_{m}"] = pc[f"{a}_{m}_pred"].median(); r[f"{a}_{m}_gt"] = pc[f"{a}_{m}_gt"].median()
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(OUT / "erf_outcome_cells.csv", index=False)
    # 3. the comparisons Paul asked for, per cell: (a) within the real arm: FFC vs CCC (does real use something else where only it succeeds?);
    #    (b) at FFC voxels: real vs flat vs noise own-decision gradients; mirror for FCF (only flat) and CFF (only noise); (c) two-agree patterns.
    def cmp_within(arm, p1, p2, m):
        out = []
        for k in cells:
            d = St[St.cell == k]; a1 = d[d.pattern == p1].groupby("case")[f"{arm}_{m}_pred"].median(); a2 = d[d.pattern == p2].groupby("case")[f"{arm}_{m}_pred"].median()
            cs = a1.index.intersection(a2.index)
            if len(cs) >= 3:
                dd = a1.loc[cs] - a2.loc[cs]; out.append(dict(cell=k, n=len(cs), d=dd.median(), frac=(dd > 0).mean(), p=wp(dd)))
        return pd.DataFrame(out)

    def cmp_arms(p, a1, a2, m):
        out = []
        for k in cells:
            d = St[(St.cell == k) & (St.pattern == p)]
            if d.case.nunique() >= 3:
                dd = d.groupby("case").apply(lambda g: (g[f"{a1}_{m}_pred"] - g[f"{a2}_{m}_pred"]).median()); out.append(dict(cell=k, n=len(dd), d=dd.median(), frac=(dd > 0).mean(), p=wp(dd)))
        return pd.DataFrame(out)

    def table(df, title):
        if not len(df):
            return [f"### {title}", "", "(no cell with >= 3 cases)", ""]
        return [f"### {title}", "", "| cell | n cases | median d | share > 0 | p |", "|---|---|---|---|---|"] + [f"| {r.cell} | {r.n} | {r.d:+.4f} | {r.frac:.2f} | {r.p:.2g} |" for r in df.itertuples()] + [
            f"pooled over cells: median of cell medians {df.d.median():+.4f}, cells with d > 0: {(df.d > 0).sum()} / {len(df)}, p < 0.05 same sign: {((df.p < 0.05) & (df.d > 0)).sum()} +, {((df.p < 0.05) & (df.d < 0)).sum()} -", ""]
    L += ["", "## 2. Does a model use something different where ONLY it succeeds? Within-arm comparison of its own-decision gradient, pattern 'only me' vs 'all correct' (same cases, matched by sampling per depth tertile)", ""]
    for arm, p in (("real", "FFC"), ("flat", "FCF"), ("noise", "CFF")):
        for m in METS:
            L += table(cmp_within(arm, p, "CCC", m), f"{arm}: {p} - CCC, {m}")
    L += ["", "## 3. At the SAME voxels, what does each arm rely on? Own-decision gradients, real - flat and real - noise, per pattern", ""]
    for p, lab in (("FFC", "only real correct"), ("FCF", "only flat correct"), ("CFF", "only noise correct"), ("CCC", "all correct"), ("FFF", "all fail"), ("CCF", "real fails, others correct"), ("CFC", "flat fails"), ("FCC", "noise fails")):
        for m in METS:
            L += table(cmp_arms(p, "real", "flat", m), f"{p} ({lab}): real - flat, {m}") + table(cmp_arms(p, "real", "noise", m), f"{p} ({lab}): real - noise, {m}")
    # 4. ring (false-positive) seeds: same within-arm and cross-arm tables, compact: fine_share and local_share only
    Sr = S[S.region == "ring"]
    L += ["", "## 4. Ring background voxels (false-positive patterns; C = correctly background), own-decision gradients, fine_share and local_share", ""]
    for p in ("FFC", "FCF", "CFF", "CCF", "CFC", "FCC", "FFF"):
        d = Sr[Sr.pattern == p]
        if d.case.nunique() < 3:
            continue
        for m in ("fine_share", "local_share"):
            out = []
            for k in cells:
                dk = d[d.cell == k]
                if dk.case.nunique() >= 3:
                    dd = dk.groupby("case").apply(lambda g: (g[f"real_{m}_pred"] - g[f"flat_{m}_pred"]).median()); out.append(dict(cell=k, n=len(dd), d=dd.median(), frac=(dd > 0).mean(), p=wp(dd)))
            L += table(pd.DataFrame(out), f"ring {p}: real - flat, {m}")
    # 5. figure: per task pooled, metric by pattern and arm (own-decision), target seeds
    pats = ["CCC", "FFC", "FCF", "CFF", "CCF", "CFC", "FCC", "FFF"]
    fig, axs = plt.subplots(len(METS), len(vt.TASKS), figsize=(2.1 * len(vt.TASKS), 2.2 * len(METS)), squeeze=False)
    for j, t in enumerate(vt.TASKS):
        d = St[St.task == t]
        for i, m in enumerate(METS):
            ax = axs[i, j]
            for pi, p in enumerate(pats):
                dp = d[d.pattern == p]
                if not len(dp):
                    continue
                pc = dp.groupby(["cell", "case"])[[f"{a}_{m}_pred" for a in ARMS]].median()
                for ai, a in enumerate(ARMS):
                    y = pc[f"{a}_{m}_pred"].dropna()
                    if len(y):
                        ax.errorbar(pi + (ai - 1) * 0.25, y.median(), yerr=[[y.median() - y.quantile(0.25)], [y.quantile(0.75) - y.median()]], fmt="o", ms=3, color=PAL[a], lw=0.7)
                ax.text(pi, ax.get_ylim()[0], str(len(pc)), fontsize=4.5, ha="center", va="bottom")
            ax.set_xticks(range(len(pats))); ax.set_xticklabels(pats, fontsize=5, rotation=90); ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(t, fontsize=8)
            if j == 0:
                ax.set_ylabel(m, fontsize=7)
    fig.suptitle("Own-decision gradient metrics by outcome pattern (noise, flat, real), per task: median and IQR over cases; green noise, blue flat, red real; number = cases", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / "erf_outcome_metrics.png", dpi=160); plt.close(fig)
    # 6. examples: one seed per pattern, glioma + a few others (patient slices; never committed)
    for k in [c for c in ("glioma_t1n_t2w", "glioma_t2w_t2f", "glioma_t1n_t2f", "breast_t2w_t1wce", "pancreas_t1wce_t2w", "pelvis_ct_mri", "mand_cbct_hanseg_ct") if c in cells]:
        examples(k, St[St.cell == k])
    (OUT / "erf_outcome.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:4 + len(cells)])); print("ERFO_DONE")


def examples(key, d, W=30):
    pats = ["CCC", "FFC", "FCF", "CFF", "CCF", "CFC", "FCC", "FFF"]; picks = []
    for p in pats:
        dp = d[d.pattern == p]
        if len(dp):
            # the seed whose real-vs-flat difference in fine_share is the median of the pattern (a typical one, not a cherry-pick)
            dd = (dp.real_fine_share_pred - dp.flat_fine_share_pred); picks.append((p, dp.loc[(dd - dd.median()).abs().idxmin()]))
    if not picks:
        return
    fig, axs = plt.subplots(len(picks), 4, figsize=(8.4, 2.1 * len(picks)), squeeze=False)
    for i, (p, s) in enumerate(picks):
        z = np.load(OUT / key / f"{s.case}_erf.npz"); j = int(s.seed); v = (int(s.y), int(s.x))
        sl = (slice(max(0, v[0] - W), v[0] + W), slice(max(0, v[1] - W), v[1] + W)); img = z["img"][j].astype(np.float32); g = z["gt"][j]
        lo_, hi_ = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99])
        Gs = {a: z[f"G_{a}"][j].astype(np.float32) for a in ARMS}; vmax = np.percentile(np.concatenate([Gs[a][sl].ravel() for a in ARMS]), 99.5)
        for c, nm in enumerate(["image"] + list(ARMS)):
            ax = axs[i, c]; ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(img[sl], cmap="gray", vmin=lo_, vmax=hi_); ax.set_ylabel(f"{p}\n{s.case}\ndepth {s.depth:.0f}", fontsize=6)
            else:
                ax.imshow(Gs[nm][sl], cmap="magma", vmin=0, vmax=vmax)
                ax.set_title(f"{nm}: {'correct' if p[c - 1] == 'C' else 'FAILED'} (margin {getattr(s, f'{nm}_margin'):+.1f}), pred {int(getattr(s, f'{nm}_pred'))}\nfine {getattr(s, f'{nm}_fine_share_pred'):.2f} local {getattr(s, f'{nm}_local_share_pred'):.3f}", fontsize=5.5, color=PAL[nm])
            ax.contour(g[sl], [0.5], colors="#00b4ff" if c == 0 else "white", linewidths=0.5); ax.plot(v[1] - sl[1].start, v[0] - sl[0].start, "+", color="cyan", ms=6, mew=0.8)
    fig.suptitle(f"{key}: one typical voxel per outcome pattern (noise, flat, real); |gradient| of each arm's OWN decision at the voxel (+), shared scale per row, in-plane slice. Contour = GT class.", fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / f"erf_outcome_examples_{key}.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in a.cells:
        run_cell(k, dev, a.case_slice)


if __name__ == "__main__":
    main()
