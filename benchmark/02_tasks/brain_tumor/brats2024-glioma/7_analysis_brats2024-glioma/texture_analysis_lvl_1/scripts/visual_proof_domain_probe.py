#!/usr/bin/env python
"""
Domain-flag representation probe (2026-10-12): does the rung-4.5 FLAT-fill glioma model encode "textured = real training-contrast
image" and route held-out test images to the real-image side, and is its FLAIR edema failure a READOUT (not feature) failure?
Non-interventional: features of unmodified images only. Pre-registration: outputs/plots/domain_probe/PREREGISTRATION.md.

Extraction (GPU, one Slurm job per part; resumable per case):
  --extract --group t1n|t2w --fold 0 --part train            (a) real training patches + (b_flat, b_real) synthesised from the same patch
  --extract --group t1n --fold 0 --part test_t2f             (c) held-out test patches of contrast t2f (also t1n, t2w)
  networks: flat (rung 4.5), real (rung 5), rand (same architecture, random init, seed 0); fold-k weights only (no fold averaging).
  per patch: GAP of encoder stages 0-5 for each (network, image); for real images (a, c): 4x4x4 cells (edema vs peritumoural
  background, >= 90 % purity) with enc-stage-2 features, last-decoder-stage cell mean, the model's own cell-mean edema log-odds.
Report (CPU): --report  -> domain_probe.md, *.png, *.csv
"""
from __future__ import annotations
import argparse, copy, json, os, random, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
REPO = THIS.parents[6]
W = Path(os.environ.get("DP_WORK", "/scratch/paulh/domain_probe"))
GLR = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w"
STAGED = Path(os.environ.get("DP_STAGED", "/scratch/paulh/train_synth_vis"))
N_TRAIN_PATCH, N_TEST_PATCH, MAX_CELLS = 6, 2, 150
NETS = ("flat", "real", "rand")
TEST_CONTRASTS = {"t1n": ("t1n", "t2f", "t2w"), "t2w": ("t2w", "t2f", "t1n")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ───────────────────────── extraction ─────────────────────────
def build_nets(group, fold, dev):
    import torch
    import visual_proof_v2 as v2, visual_proof_erf_tasks as et
    M = et.G_T1N if group == "t1n" else et.G_T2W
    prs = {a: v2.load_predictor(M[a](), dev) for a in ("flat", "real")}
    nets = {}
    for a, pr in prs.items():
        net = pr.network; net.load_state_dict(pr.list_of_parameters[fold]); net.eval()
        if hasattr(net.decoder, "deep_supervision"):
            net.decoder.deep_supervision = False
        nets[a] = net
    rnet = copy.deepcopy(nets["flat"]); torch.manual_seed(0)
    for m in rnet.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    rnet.eval(); nets["rand"] = rnet
    return prs, nets


def forward_feats(net, x, dev, cells=None):
    """x: (D,H,W) float32 numpy patch. Returns GAP dict (e0..e5) and, if cells=(cz,cy,cx) index arrays, cell features."""
    import torch, torch.nn.functional as F
    acts = {}
    hs = [net.encoder.stages[i].register_forward_hook(lambda m, i_, o, i=i: acts.__setitem__(f"e{i}", o)) for i in range(len(net.encoder.stages))]
    hs.append(net.decoder.stages[-1].register_forward_hook(lambda m, i_, o: acts.__setitem__("dlast", o)))
    xt = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(dev)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        out = net(xt); out = out[0] if isinstance(out, (list, tuple)) else out
        gap = {k: v.float().mean((2, 3, 4))[0].cpu().numpy() for k, v in acts.items() if k.startswith("e")}
        res = {"gap": gap}
        if cells is not None:
            cz, cy, cx = [torch.from_numpy(c).long().to(dev) for c in cells]
            e2 = acts["e2"].float()[0]                                                   # (128, D/4, H/4, W/4)
            dl = F.avg_pool3d(acts["dlast"].float(), 4)[0]                               # (32, D/4, ...)
            oth = torch.cat([out[:, :2], out[:, 3:]], 1)
            lo = (out[:, 2] - torch.logsumexp(oth, 1))[:, None]                          # edema log-odds vs all other classes
            pe = (out.argmax(1) == 2).float()[:, None]
            res["enc2"] = e2[:, cz, cy, cx].T.cpu().numpy()
            res["dlast"] = dl[:, cz, cy, cx].T.cpu().numpy()
            res["lo"] = F.avg_pool3d(lo, 4)[0, 0][cz, cy, cx].cpu().numpy()
            res["pfrac"] = F.avg_pool3d(pe, 4)[0, 0][cz, cy, cx].cpu().numpy()
    for h in hs:
        h.remove()
    return res


def pick_cells(img_p, seg_p, dwt_p, rng):
    """4x4x4 cells: edema (label 2) purity >= 0.9 vs band background (label 0, brain, 2 < d(WT) <= 15) purity >= 0.9."""
    sh = tuple(s // 4 for s in seg_p.shape)
    def frac(m):
        return m[:sh[0] * 4, :sh[1] * 4, :sh[2] * 4].reshape(sh[0], 4, sh[1], 4, sh[2], 4).mean((1, 3, 5))
    fe = frac(seg_p == 2); fb = frac((seg_p == 0) & (img_p != 0) & (dwt_p > 2) & (dwt_p <= 15))
    dc = frac(np.minimum(dwt_p, 50).astype(np.float32))
    idx_e = np.argwhere(fe >= 0.9); idx_b = np.argwhere(fb >= 0.9)
    idx_e = idx_e[rng.choice(len(idx_e), min(len(idx_e), MAX_CELLS), replace=False)] if len(idx_e) else idx_e
    idx_b = idx_b[rng.choice(len(idx_b), min(len(idx_b), MAX_CELLS), replace=False)] if len(idx_b) else idx_b
    idx = np.concatenate([idx_e, idx_b]).astype(int).reshape(-1, 3); y = np.r_[np.ones(len(idx_e)), np.zeros(len(idx_b))].astype(np.int8)
    return (idx[:, 0], idx[:, 1], idx[:, 2]), y, dc[tuple(idx.T)] if len(idx) else np.zeros(0)


def extract(args):
    import torch
    from scipy import ndimage as ndi
    import train_synth_edema_visibility as ts
    dev = torch.device("cuda")
    od = W / "feat" / f"{args.group}_f{args.fold}" / args.part; od.mkdir(parents=True, exist_ok=True)
    prs, nets = build_nets(args.group, args.fold, dev)
    log("nets ready", args.group, args.fold, args.part)
    if args.part == "train":
        members = json.load(open(W / "fold_train_cases.json"))[str(args.fold)]
        cases = [c.strip() for c in open(STAGED / "cases.txt") if c.strip() and c.strip() in members]
        import auglab.transforms.gpu.fromSeg as _fs
        auglab_root = Path(_fs.__file__).resolve().parents[2]; Ts = ts.get_transforms(auglab_root)
        (od / "provenance.txt").write_text(f"auglab: {_fs.__file__}\nfold-{args.fold} training cases used: {len(cases)}\n")
    else:
        con = args.part.split("_", 1)[1]; I = GLR / f"imagesTs_{con}"
        cases = sorted(p.name[:-12] for p in I.glob("*_0000.nii.gz") if (GLR / "labelsTr" / f"{p.name[:-12]}.nii.gz").exists())
    if args.case_slice:
        a, b = (int(v) for v in args.case_slice.split(":")); cases = cases[a:b]
    for case in cases:
        of = od / f"{case}.npz"
        if of.exists():
            continue
        t0 = time.time(); cid = int(case[-8:]); rng = np.random.default_rng(cid + 7)
        if args.part == "train":
            img, seg = ts.load_case(STAGED / "data", args.group, case)
        else:
            data, seg = __import__("visual_proof_v2").preprocess(prs["flat"], I / f"{case}_0000.nii.gz", GLR / "labelsTr" / f"{case}.nii.gz")
            img = data[0]
        if (seg > 0).sum() < 1000:
            np.savez_compressed(of, skipped=1); continue
        dwt = ndi.distance_transform_edt(~(seg > 0)).astype(np.float32)
        fg = np.argwhere(seg > 0); npatch = N_TRAIN_PATCH if args.part == "train" else N_TEST_PATCH
        store = {}; cells_all = {"y": [], "d": [], "patch": []}
        def put(k, v):
            store.setdefault(k, []).append(v)
        for p in range(npatch):
            starts = ts.patch_box(seg.shape, fg[rng.integers(len(fg))])
            xp = ts.crop(img, starts, np.float32(0)); sp = ts.crop(seg, starts, np.int16(-1)); dp = ts.crop(dwt, starts, np.float32(99))
            cells, y, dc = pick_cells(xp, sp, dp, rng)
            cells_all["y"].append(y); cells_all["d"].append(dc); cells_all["patch"].append(np.full(len(y), p, np.int16))
            imgs = {"a" if args.part == "train" else "c": xp}
            if args.part == "train":
                xt = torch.from_numpy(xp)[None, None]; sg = torch.from_numpy(sp.astype(np.float32))[None, None]
                seed = (1000 * cid + 500 + p) % 2**31
                for r, T in Ts.items():
                    torch.manual_seed(seed); random.seed(seed)
                    imgs[f"b{r}"] = T.apply_transform(xt, {"seg": sg}, {})[0, 0].numpy().astype(np.float32)
            for n, net in nets.items():
                for im, x in imgs.items():
                    res = forward_feats(net, x, dev, cells if im in ("a", "c") and len(y) else None)
                    for k, v in res["gap"].items():
                        put(f"gap/{n}/{im}/{k}", v.astype(np.float32))
                    for k in ("enc2", "dlast", "lo", "pfrac"):
                        if k in res:
                            put(f"cell/{n}/{k}", res[k].astype(np.float16 if k in ("enc2", "dlast") else np.float32))
        arr = {k: (np.stack(v) if k.startswith("gap/") else np.concatenate(v)) for k, v in store.items()}
        arr.update({f"cell_{k}": np.concatenate(v) for k, v in cells_all.items()})
        np.savez_compressed(of, **arr)
        log(f"{args.group} f{args.fold} {args.part} {case}: {npatch} patches, {int(arr['cell_y'].sum())}/{len(arr['cell_y'])} edema cells, {time.time()-t0:.0f}s")


# ───────────────────────── analysis ─────────────────────────
def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bal_acc(y, pred):
    y = np.asarray(y).astype(bool); pred = np.asarray(pred).astype(bool)
    return float(0.5 * (pred[y].mean() + (~pred[~y]).mean())) if y.any() and (~y).any() else np.nan


def boot(groups, fn, n=1000, seed=0):
    """case bootstrap: fn(mask_index_array) -> stat; returns (lo, hi)."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); pos = {g: np.where(groups == g)[0] for g in ug}; out = []
    for _ in range(n):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))]); v = fn(idx)
        if np.isfinite(v):
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def make_lr(k=None):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    steps = [StandardScaler()] + ([PCA(k, random_state=0)] if k else []) + [LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)]
    return make_pipeline(*steps)


def load_part(group, fold, part):
    d = W / "feat" / f"{group}_f{fold}" / part; out = {}
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        if "skipped" in z:
            continue
        out[f.stem] = {k: z[k] for k in z.files}
    return out


def image_level(group, fold, rows, tpos):
    """P1 (a vs b separability) and P2 (placement of other image sets on the a-vs-b probe axis)."""
    from sklearn.model_selection import GroupKFold
    tr = load_part(group, fold, "train"); tests = {c: load_part(group, fold, f"test_{c}") for c in TEST_CONTRASTS[group]}
    cases = sorted(tr); log(group, fold, "train cases", len(cases), {c: len(v) for c, v in tests.items()})
    for net in NETS:
        for lvl in ("e5", "e2", "e4", "e3", "e1", "e0"):
            A = np.concatenate([tr[c][f"gap/{net}/a/{lvl}"] for c in cases]); ga = np.concatenate([[c] * len(tr[c][f"gap/{net}/a/{lvl}"]) for c in cases])
            B = {f: np.concatenate([tr[c][f"gap/{net}/b{f}/{lvl}"] for c in cases]) for f in ("flat", "real")}
            C = {con: (np.concatenate([v[f"gap/{net}/c/{lvl}"] for v in d.values()]), np.concatenate([[k] * len(v[f"gap/{net}/c/{lvl}"]) for k, v in d.items()]))
                 for con, d in tests.items() if d}
            for fill in ("flat", "real"):
                other = "real" if fill == "flat" else "flat"
                X = np.r_[A, B[fill]]; y = np.r_[np.ones(len(A)), np.zeros(len(B[fill]))]; g = np.r_[ga, ga]
                for k in ([None, 1, 3] if lvl in ("e5", "e2") else [None]):
                    oof = np.full(len(y), np.nan); place = {s: [] for s in ["b_" + other] + list(C)}
                    for tri, tei in GroupKFold(5).split(X, y, g):
                        m = make_lr(k).fit(X[tri], y[tri]); s = m.decision_function(X[tei]); oof[tei] = s
                        if k is None:
                            ma, mb = s[y[tei] == 1].mean(), s[y[tei] == 0].mean()
                            hc = np.isin(ga, np.unique(g[tei]))
                            sets = {"b_" + other: (B[other][hc], ga[hc])}; sets.update(C)
                            for nm, (Xs, gs) in sets.items():
                                ss = m.decision_function(Xs); place[nm].append(pd_frame(ss, (ss - mb) / (ma - mb), gs))
                            place.setdefault("a_oof", []).append(pd_frame(s[y[tei] == 1], (s[y[tei] == 1] - mb) / (ma - mb), g[tei][y[tei] == 1]))
                    a_ = auc(y, oof); ci = boot(g, lambda i: auc(y[i], oof[i]))
                    rows.append(dict(group=group, fold=fold, net=net, level=lvl, fill=fill, pcs=k or "all", auc=a_, lo=ci[0], hi=ci[1],
                                     bacc=bal_acc(y, oof > 0), n_a=len(A), n_cases=len(cases)))
                    if k is None and lvl in ("e5", "e2"):
                        import pandas as pd
                        a_oof = pd.concat(place.pop("a_oof")); p5, p95 = np.percentile(a_oof.t, [5, 95])
                        for nm, lst in place.items():
                            if not lst:
                                continue
                            df = pd.concat(lst); per = df.groupby("case")[["pa", "t"]].mean()
                            # held-out test patches are scored by all 5 fold models -> per-patch mean over models first
                            gs = per.index.values
                            tpos.append(dict(group=group, fold=fold, net=net, level=lvl, probe=f"a vs b_{fill}", set=nm, n_cases=len(per),
                                             p_real=float(per.pa.mean()), p_real_ci=boot(gs, lambda i: per.pa.values[i].mean()),
                                             t_median=float(np.median(df.t)), t_mean=float(per.t.mean()), t_ci=boot(gs, lambda i: per.t.values[i].mean()),
                                             frac_in_a=float(((df.t >= p5) & (df.t <= p95)).mean()), frac_beyond_a=float((df.t > p95).mean())))


def pd_frame(s, t, g):
    import pandas as pd
    return pd.DataFrame(dict(pa=(s > 0).astype(float), t=t, case=g))


def voxel_level(group, fold, rows):
    """P3: edema vs peritumoural background decoding from enc2 / dlast cells; D_train (fit on real training patches), D_within, own readout."""
    from sklearn.model_selection import GroupKFold
    tr = load_part(group, fold, "train"); tests = {c: load_part(group, fold, f"test_{c}") for c in TEST_CONTRASTS[group]}
    rng = np.random.default_rng(0)
    def cat(d, key):
        ks = [k for k in sorted(d) if len(d[k]["cell_y"])]
        return np.concatenate([d[k][key] for k in ks]).astype(np.float32), np.concatenate([[k] * len(d[k]["cell_y"]) for k in ks])
    for net in NETS:
        for feat in ("enc2", "dlast"):
            Xa, ga = cat(tr, f"cell/{net}/{feat}"); ya, _ = cat(tr, "cell_y")
            sub = rng.choice(len(ya), min(len(ya), 20000), replace=False)
            Dtr = make_lr().fit(Xa[sub], ya[sub])
            for con, d in tests.items():
                if not d:
                    continue
                Xc, gc = cat(d, f"cell/{net}/{feat}"); yc, _ = cat(d, "cell_y")
                s = Dtr.decision_function(Xc)
                oof = np.full(len(yc), np.nan)
                for tri, tei in GroupKFold(5).split(Xc, yc, gc):
                    oof[tei] = make_lr().fit(Xc[tri], yc[tri]).decision_function(Xc[tei])
                base = dict(group=group, fold=fold, net=net, feat=feat, test=con, n_cases=len(np.unique(gc)), n_cells=len(yc), frac_edema=float(yc.mean()))
                rows.append(dict(base, decoder="D_train", auc=auc(yc, s), ci=boot(gc, lambda i: auc(yc[i], s[i])), bacc=bal_acc(yc, s > 0)))
                rows.append(dict(base, decoder="D_within", auc=auc(yc, oof), ci=boot(gc, lambda i: auc(yc[i], oof[i])), bacc=bal_acc(yc, oof > 0)))
                if feat == "enc2" and net != "rand":
                    lo, _ = cat(d, f"cell/{net}/lo"); pf, _ = cat(d, f"cell/{net}/pfrac")
                    rows.append(dict(base, feat="own", decoder="own_readout", auc=auc(yc, lo), ci=boot(gc, lambda i: auc(yc[i], lo[i])), bacc=bal_acc(yc, pf >= 0.5)))
            if feat == "enc2" and net != "rand":   # own readout on its own training images (sanity)
                lo, _ = cat(tr, f"cell/{net}/lo")
                rows.append(dict(group=group, fold=fold, net=net, feat="own", test=f"{group} TRAIN (a)", decoder="own_readout", auc=auc(ya, lo),
                                 ci=boot(ga, lambda i: auc(ya[i], lo[i])), bacc=np.nan, n_cases=len(np.unique(ga)), n_cells=len(ya), frac_edema=float(ya.mean())))


def report(args):
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, tpos, vrows = [], [], []
    for group in ("t1n", "t2w"):
        for fold in args.folds:
            if not (W / "feat" / f"{group}_f{fold}" / "train").exists():
                continue
            image_level(group, fold, rows, tpos); voxel_level(group, fold, vrows)
    S = pd.DataFrame(rows); P = pd.DataFrame(tpos); V = pd.DataFrame(vrows)
    S.to_csv(out / "p1_separability.csv", index=False); P.to_csv(out / "p2_placement.csv", index=False); V.to_csv(out / "p3_voxel_decoding.csv", index=False)
    f = lambda v: f"{v:.3f}"  # noqa: E731
    fci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"  # noqa: E731
    L = [(out / "PREREGISTRATION.md").read_text().strip() if (out / "PREREGISTRATION.md").exists() else "(pre-registration file missing)", "",
         "---", "", "# Results (generated by `scripts/visual_proof_domain_probe.py --report`)", ""]
    L += ["## P1: a (real training patches) vs b (synthesised from the same patches), case-CV AUC [95 % case-bootstrap CI]", "",
          "| group | fold | level | PCs | network | a vs b_flat | a vs b_real |", "|---|---|---|---|---|---|---|"]
    for (g, fo, lvl, k), d in S.groupby(["group", "fold", "level", "pcs"], sort=False):
        for net in NETS:
            r = {fl: d[(d.net == net) & (d.fill == fl)].iloc[0] for fl in ("flat", "real")}
            L.append(f"| {g} | {fo} | {lvl} | {k} | {net} | " + " | ".join(f"{f(r[fl].auc)} [{r[fl].lo:.2f}, {r[fl].hi:.2f}]" for fl in ("flat", "real")) + " |")
    L += ["", "## P2: placement on the a-vs-b probe axis (out-of-fold; P(real) = share of probe calls on the a side; t = (s-mean_b)/(mean_a-mean_b), "
          "1 = centre of a, 0 = centre of b; in-a = share of patches within a's OOF 5-95 % t range; beyond = above a's 95 %)", "",
          "| group | fold | level | network | probe | set | n cases | P(real) [CI] | mean t [CI] | median t | in-a | beyond-a |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in P.iterrows():
        L.append(f"| {r.group} | {r.fold} | {r.level} | {r.net} | {r.probe} | {r.set} | {r.n_cases} | {f(r.p_real)} {fci(r.p_real_ci)} | {r.t_mean:.2f} {fci(r.t_ci)} | "
                 f"{r.t_median:.2f} | {r.frac_in_a:.2f} | {r.frac_beyond_a:.2f} |")
    L += ["", "## P3: voxel-level (4x4x4 cells) edema vs peritumoural background, AUC [95 % CI] (balanced accuracy at the decoder's own threshold)", "",
          "D_train = logistic decoder fit on the real TRAINING-contrast patches (a) and applied unchanged; D_within = fit on the test contrast itself "
          "(5-fold case CV); own = the model's own cell-mean edema log-odds (balanced accuracy = cell majority predicted edema).", "",
          "| group | fold | test | feature | decoder | flat | real | rand |", "|---|---|---|---|---|---|---|---|"]
    for (g, fo, te, ft, dec), d in V.groupby(["group", "fold", "test", "feat", "decoder"], sort=False):
        cell = []
        for net in NETS:
            x = d[d.net == net]
            cell.append("-" if x.empty else f"{f(x.iloc[0].auc)} {fci(x.iloc[0].ci)} ({x.iloc[0].bacc:.2f})")
        L.append(f"| {g} | {fo} | {te} | {ft} | {dec} | " + " | ".join(cell) + " |")
    (out / "domain_probe.md").write_text("\n".join(L) + "\n")
    # figure: P1 bars (fold 0, e5/e2 all PCs), P2 t for held-out sets, P3 decoders on held-out contrasts
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    col = {"flat": "#d95f02", "real": "#1b9e77", "rand": "#7570b3"}
    for gi, group in enumerate(("t1n", "t2w")):
        ax = axes[gi, 0]; d = S[(S.group == group) & (S.fold == args.folds[0]) & (S.pcs.astype(str).isin(["all", "1"])) & (S.level.isin(["e5", "e2"]))]
        xs, lab = 0, []
        for (lvl, k, fill), dd in d.groupby(["level", "pcs", "fill"], sort=False):
            for j, net in enumerate(NETS):
                r = dd[dd.net == net]
                if len(r):
                    r = r.iloc[0]; ax.errorbar(xs + (j - 1) * 0.25, r.auc, yerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o", color=col[net], label=net if xs == 0 else None)
            lab.append(f"{lvl} PCs={k}\na vs b_{fill}"); xs += 1
        ax.set_xticks(range(xs)); ax.set_xticklabels(lab, fontsize=7); ax.set_ylim(0.4, 1.02); ax.axhline(0.5, color="k", lw=0.5)
        ax.set_title(f"P1 {group.upper()}-trained: a vs b separability (AUC)", fontsize=9); ax.legend(fontsize=7)
        ax = axes[gi, 1]; d = P[(P.group == group) & (P.fold == args.folds[0]) & (P.level == "e5")]
        xs, lab = 0, []
        for (probe, st), dd in d.groupby(["probe", "set"], sort=False):
            for j, net in enumerate(NETS):
                r = dd[dd.net == net]
                if len(r):
                    r = r.iloc[0]; ax.errorbar(xs + (j - 1) * 0.25, r.t_mean, yerr=[[r.t_mean - r.t_ci[0]], [r.t_ci[1] - r.t_mean]], fmt="o", color=col[net])
            lab.append(f"{probe}\n{st}"); xs += 1
        ax.axhline(1, color="k", lw=0.5, ls="--"); ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xticks(range(xs)); ax.set_xticklabels(lab, fontsize=6, rotation=30, ha="right"); ax.set_ylabel("t (1 = real-image centre, 0 = synthetic centre)")
        ax.set_title(f"P2 {group.upper()}-trained, bottleneck: placement", fontsize=9); ax.set_yscale("symlog")
        ax = axes[gi, 2]; d = V[(V.group == group) & (V.fold == args.folds[0]) & (V.feat.isin(["enc2", "own"])) & (~V.test.str.contains("TRAIN"))]
        xs, lab = 0, []
        for (te, dec), dd in d.groupby(["test", "decoder"], sort=False):
            for j, net in enumerate(NETS):
                r = dd[dd.net == net]
                if len(r):
                    r = r.iloc[0]; ax.errorbar(xs + (j - 1) * 0.25, r.auc, yerr=[[r.auc - r.ci[0]], [r.ci[1] - r.auc]], fmt="o", color=col[net])
            lab.append(f"{te}\n{dec}"); xs += 1
        ax.set_xticks(range(xs)); ax.set_xticklabels(lab, fontsize=7, rotation=30, ha="right"); ax.set_ylim(0, 1.02); ax.axhline(0.5, color="k", lw=0.5)
        ax.set_title(f"P3 {group.upper()}-trained: edema vs peritumoural bg (AUC), enc stage 2", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "domain_probe.png", dpi=140); plt.close(fig)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--group", choices=["t1n", "t2w"]); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="train"); ap.add_argument("--case-slice", default="")
    ap.add_argument("--folds", type=int, nargs="+", default=[0]); ap.add_argument("--out", default=str(W / "report"))
    args = ap.parse_args()
    if args.extract:
        extract(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
