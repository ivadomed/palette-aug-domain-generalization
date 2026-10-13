#!/usr/bin/env python
"""
Texture content of every encoder/decoder stage and of the segmentation readout, flat-fill (rung 4.5) vs real-fill (rung 5) glioma models
(FINDINGS section 31, pre-registered before extraction). Natural, unmodified test patches only (no intervention).
  --extract --group {t1n,t2w} --test {t2f,own}   (GPU)  per case: sampled voxels, image descriptors, every stage's activations for flat/real x folds 0-2
  --report                                         (CPU)  M1 ridge R^2 of residualised texture per stage; M2 readout alignment; figures + md
Outputs: VP8_OUT/texture_content/{feat/<group>_<test>/<case>.npz, texture_content.md, *.png}
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402

e7, v2 = vt.e7, vt.v2; ARMS = ("flat", "real"); FOLDS = (0, 1, 2)
OUT = vt.OUT / "texture_content"; OUT.mkdir(parents=True, exist_ok=True)
FEAT = Path(os.environ.get("TC_FEAT") or (OUT / "feat")); FEAT.mkdir(parents=True, exist_ok=True)   # bulky per-case activations (scratch)
N_CASES = int(os.environ.get("TC_NCASES", 30)); N_PER = 300; EDEMA = 2; TAG = os.environ.get("TC_TAG", "")
SRC = {("t1n", "t2f"): "glioma_t1n_t2f", ("t1n", "own"): "glioma_t2w_t1n", ("t2w", "t2f"): "glioma_t2w_t2f", ("t2w", "own"): "glioma_t1n_t2w"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def descriptors(ip):
    g1 = ndi.gaussian_filter(ip, 1.0); hp = ip - g1
    return dict(I=ndi.gaussian_filter(ip, 2.0), E=ndi.gaussian_gradient_magnitude(ip, 2.0), Tf=np.sqrt(ndi.gaussian_filter(hp ** 2, 2.0)),
                H=np.sqrt(np.maximum(ndi.gaussian_filter(g1 ** 2, 3.0) - ndi.gaussian_filter(g1, 3.0) ** 2, 0)))


def extract(group, test, sl=None):
    import torch
    dev = torch.device("cuda"); M = vt.G_T1N if group == "t1n" else vt.G_T2W; cell = vt.CELLS[SRC[(group, test)]]
    prs = {a: v2.load_predictor(M[a](), dev) for a in ARMS}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    od = FEAT / f"{group}_{test}"; od.mkdir(parents=True, exist_ok=True)
    cases = list(cell["cases"])[:N_CASES]; cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    rng = np.random.default_rng(0)
    for case in cases:
        if (od / f"{case}.npz").exists():
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = gt == EDEMA
        if tg.sum() < 200:
            continue
        box = e7.patch_at(tg, ps, img.shape); ip, gp = img[box].astype(np.float32), gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, gp = np.pad(ip, padw), np.pad(gp, padw)
        body = ip != 0; inner = np.zeros(gp.shape, bool); inner[tuple(slice(8, n - 8) for n in gp.shape)] = True
        dt = ndi.distance_transform_edt(gp == 0)
        strata = {0: (gp == EDEMA) & body & inner, 1: (gp == 0) & (dt <= 10) & body & inner, 2: (gp == 0) & (dt > 10) & body & inner}
        idx, kind = [], []
        for k, m in strata.items():
            w = np.flatnonzero(m)
            if len(w):
                w = rng.choice(w, min(N_PER, len(w)), replace=False); idx.append(w); kind.append(np.full(len(w), k, np.int8))
        idx = np.concatenate(idx); kind = np.concatenate(kind); zyx = np.stack(np.unravel_index(idx, gp.shape), 1)
        de = descriptors(ip); out = dict(kind=kind, zyx=zyx.astype(np.int16), lab=gp.ravel()[idx].astype(np.int8), **{k: v.ravel()[idx].astype(np.float32) for k, v in de.items()})
        xt = torch.from_numpy(np.ascontiguousarray(ip))[None, None].to(dev)
        for a, pr in prs.items():
            net = pr.network; enc, dec = net.encoder.stages, net.decoder.stages
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval(); acts = {}
                hs = [enc[i].register_forward_hook(lambda m_, i_, o, i=i: acts.__setitem__(f"e{i}", o)) for i in range(len(enc))]
                hs += [dec[i].register_forward_hook(lambda m_, i_, o, i=i: acts.__setitem__(f"d{i}", o)) for i in range(len(dec))]
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    o = net(xt); o = (o[0] if isinstance(o, (list, tuple)) else o).float()
                for h in hs:
                    h.remove()
                oth = torch.cat([o[:, :EDEMA], o[:, EDEMA + 1:]], 1); lo = (o[:, EDEMA] - torch.logsumexp(oth, 1))[0].cpu().numpy()
                out[f"{a}_f{k}_lo"] = lo.ravel()[idx].astype(np.float32)
                for s, A in acts.items():
                    A = A.float()[0]; st = [ps_ // n for ps_, n in zip(ip.shape, A.shape[1:])]
                    q = [np.minimum(zyx[:, d] // st[d], A.shape[1 + d] - 1) for d in range(3)]
                    out[f"{a}_f{k}_{s}"] = A[:, torch.from_numpy(q[0]).long(), torch.from_numpy(q[1]).long(), torch.from_numpy(q[2]).long()].T.cpu().numpy().astype(np.float16)
                out[f"{a}_f{k}_seghead"] = net.decoder.seg_layers[-1].weight.detach().float().cpu().numpy()[:, :, 0, 0, 0]
        np.savez_compressed(od / f"{case}.npz", **out); log(f"[{group} {test}] {case}: {len(idx)} voxels, {time.time() - t0:.0f} s")


def ridge_cv(X, y, groups, alphas=(1e-1, 1, 10, 100, 1000), k=5):
    """Case-grouped k-fold CV; standardised X; returns out-of-fold R^2 (best alpha by inner mean) and the full-data weight vector (raw X space)."""
    ug = np.unique(groups); fold_of = {g: i % k for i, g in enumerate(np.random.default_rng(0).permutation(ug))}; fid = np.array([fold_of[g] for g in groups])
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; yc = y - y.mean(); best = (-np.inf, None, None)
    for al in alphas:
        pred = np.zeros_like(y)
        for f in range(k):
            tr, te = fid != f, fid == f
            if te.sum() == 0 or tr.sum() < 10:
                continue
            Zt = Z[tr]; ym = y[tr].mean(); w = np.linalg.solve(Zt.T @ Zt + al * np.eye(Z.shape[1]), Zt.T @ (y[tr] - ym)); pred[te] = Z[te] @ w + ym
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if r2 > best[0]:
            best = (r2, al, pred)
    al = best[1]; w = np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ yc)
    return best[0], w / sd


def residualise(df, col):
    """log(col) residual after per-case OLS on I and log E."""
    out = np.full(len(df), np.nan)
    for _, ix in df.groupby("case").indices.items():
        d = df.iloc[ix]; X = np.c_[np.ones(len(d)), d.I, np.log(d.E + 1e-6)]; y = np.log(d[col].values + 1e-6)
        b = np.linalg.lstsq(X, y, rcond=None)[0]; out[ix] = y - X @ b
    return out


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, align = [], []
    for (group, test) in SRC:
        fs = sorted((FEAT / f"{group}_{test}").glob("*.npz"))
        if not fs:
            continue
        Z = [np.load(f) for f in fs]; stages = sorted({k.split("_", 2)[2] for k in Z[0].files if k.startswith("flat_f0_") and k[8] in "ed"}, key=lambda s: (s[0] != "e", int(s[1:])))
        meta = pd.concat([pd.DataFrame(dict(case=f.stem, kind=z["kind"], I=z["I"], E=z["E"], Tf=z["Tf"], H=z["H"])) for f, z in zip(fs, Z)], ignore_index=True)
        meta["rTf"] = residualise(meta, "Tf"); meta["rH"] = residualise(meta, "H"); sel = (meta.kind <= 1).values; g = meta.case.values[sel]
        for a in ARMS:
            for k in FOLDS:
                for s in stages:
                    X = np.concatenate([z[f"{a}_f{k}_{s}"].astype(np.float32) for z in Z])[sel]
                    for tgt in ("rTf", "rH", "I"):
                        r2, w = ridge_cv(X, meta[tgt].values[sel], g); rows.append(dict(group=group, test=test, arm=a, fold=k, stage=s, target=tgt, r2=r2))
                        if s == stages[-1] and tgt in ("rTf", "rH"):
                            sh = next((z[f"{a}_f{k}_seghead"] for z in Z if f"{a}_f{k}_seghead" in z.files), None)
                            if sh is not None:
                                de = sh[EDEMA] - np.delete(sh, EDEMA, 0).mean(0); c = float(w @ de / (np.linalg.norm(w) * np.linalg.norm(de) + 1e-12))
                                rnd = np.random.default_rng(0).standard_normal((2000, len(de))); cr = np.abs(rnd @ de / (np.linalg.norm(rnd, axis=1) * np.linalg.norm(de)))
                                align.append(dict(group=group, test=test, arm=a, fold=k, target=tgt, cos=c, null95=float(np.percentile(cr, 95))))
                log(f"{group} {test} {a} f{k} done")
    R = pd.DataFrame(rows); A = pd.DataFrame(align); R.to_csv(OUT / f"m1_r2{TAG}.csv", index=False); A.to_csv(OUT / f"m2_alignment{TAG}.csv", index=False)
    L = ["# Texture content per stage and readout alignment (FINDINGS section 31; natural test patches; edema + peritumoural ring voxels)", "",
         "R^2 = out-of-fold (case-grouped) ridge R^2 of the target from the stage's features; rTf / rH = fine texture / patch heterogeneity residualised on "
         "intensity and edges; I = local intensity (reference). 'sep' = all 3 real folds above all 3 flat folds (+) or below (-).", ""]
    for (group, test), d in R.groupby(["group", "test"], sort=False):
        L += [f"## {group}-trained, test = {'FLAIR' if test == 't2f' else 'training contrast'}", "", "| stage | target | flat R^2 (folds) | real R^2 (folds) | sep |", "|---|---|---|---|---|"]
        for (s, t), e in d.groupby(["stage", "target"], sort=False):
            f_, r_ = e[e.arm == "flat"].sort_values("fold").r2.values, e[e.arm == "real"].sort_values("fold").r2.values
            sep = "+" if r_.min() > f_.max() else ("-" if r_.max() < f_.min() else "")
            L.append(f"| {s} | {t} | {f_.mean():.3f} ({'/'.join(f'{x:.3f}' for x in f_)}) | {r_.mean():.3f} ({'/'.join(f'{x:.3f}' for x in r_)}) | {sep} |")
        L.append("")
    L += ["## M2: last decoder stage, cosine(texture ridge direction, seg-head edema direction); |cos| above null95 = aligned beyond random", "",
          "| group | test | target | flat cos (folds) | real cos (folds) | null95 |", "|---|---|---|---|---|---|"]
    for (gr, te, t), d in A.groupby(["group", "test", "target"], sort=False):
        f_, r_ = d[d.arm == "flat"].sort_values("fold").cos.values, d[d.arm == "real"].sort_values("fold").cos.values
        L.append(f"| {gr} | {te} | {t} | {'/'.join(f'{x:+.2f}' for x in f_)} | {'/'.join(f'{x:+.2f}' for x in r_)} | {d.null95.iloc[0]:.2f} |")
    (OUT / f"texture_content{TAG}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    combos = list(R.groupby(["group", "test"], sort=False).groups)
    fig, axs = plt.subplots(3, len(combos), figsize=(3.6 * len(combos), 7.5), squeeze=False)
    for j, (gr, te) in enumerate(combos):
        d = R[(R.group == gr) & (R.test == te)]; st = list(dict.fromkeys(d.stage)); x = np.arange(len(st))
        for i, (t, nm) in enumerate((("rTf", "fine texture (resid.)"), ("rH", "patch heterogeneity (resid.)"), ("I", "local intensity"))):
            ax = axs[i, j]
            for a, col in (("flat", "#1f77b4"), ("real", "#d62728")):
                for k in FOLDS:
                    e = d[(d.arm == a) & (d.fold == k) & (d.target == t)].set_index("stage").reindex(st)
                    ax.plot(x, e.r2.values, "-o", color=col, ms=2.5, lw=0.9, alpha=0.8, label=f"{a}-fill" if k == 0 else None)
            ax.set_xticks(x); ax.set_xticklabels(st, fontsize=6, rotation=90); ax.tick_params(labelsize=7); ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
            ax.set_ylabel(f"R^2: {nm}", fontsize=7)
            if i == 0:
                ax.set_title(f"{gr}-trained, test {'FLAIR' if te == 't2f' else 'own contrast'}", fontsize=8); ax.legend(fontsize=6)
    fig.suptitle("How much of the input's local texture each stage's features carry (natural test patches, edema + peritumoural ring; one line per fold)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / f"texture_content{TAG}.png", dpi=170); fig.savefig(OUT / f"texture_content{TAG}.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--group", default="t1n"); ap.add_argument("--test", default="t2f"); ap.add_argument("--case-slice", default=None)
    a = ap.parse_args()
    if a.report:
        report()
    else:
        extract(a.group, a.test, a.case_slice)


if __name__ == "__main__":
    main()
