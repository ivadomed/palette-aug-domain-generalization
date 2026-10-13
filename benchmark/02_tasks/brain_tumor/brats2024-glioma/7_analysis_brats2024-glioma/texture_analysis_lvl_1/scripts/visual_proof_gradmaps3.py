#!/usr/bin/env python
"""
Gradient and activation maps, noise-fill (rung 4) vs flat-fill (rung 4.5) vs real-fill (rung 5) glioma models (FINDINGS section 32, pre-registered).
Natural held-out-contrast test images, no intervention. Per case and model (folds 0-2):
  G_e = d J_edema / d input,  J_edema = mean edema log-odds (edema vs all other classes) over the ground-truth edema of the patch
  G_b = d J_bg / d input,     J_bg = mean background log-odds over the peritumoural ring (1-10 voxels from any tumour label)
  CAM = Seg-Grad-CAM of J_edema at the last two decoder stages (ReLU(sum_c mean(dJ/dA_c) A_c)), upsampled to the patch.
Per-case numbers (3D, body only): fine-detail share of G (energy of G - Gaussian(G, 1) / energy of G); share of |G| inside the edema / ring /
elsewhere; CAM mass inside the edema; edema hit rate. Saved per fold; maps (fold mean) on the axial slice with the most edema.
  GPU: --cell {t1n_t2f,t1n_t1n,t2w_t2f} [--case-slice i:j]       CPU: --report
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402

e7, v2 = vt.e7, vt.v2; FOLDS = (0, 1, 2); EDEMA = 2; TUMOUR = (1, 2, 3, 4)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
LAB = {"noise": "noise fill (rung 4)", "flat": "flat fill (rung 4.5)", "real": "real fill (rung 5)"}
PRED = vt.T2 / vt.GLP
RUNS = {"t1n": {"noise": "t1n/auglab/brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_lblvor_20261007_170548",
                "flat": "t1n/auglab/brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190618",
                "real": "t1n/auglab/brats2024-glioma_t1n_v26_6_2_train050_val000_20261005_222227"},
        "t2w": {"noise": "t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_lblvor_20261007_170621",
                "flat": "t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261008_190652",
                "real": "t2w/auglab/brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}}
CELLS = {"t1n_t2f": ("t1n", "glioma_t1n_t2f"), "t1n_t1n": ("t1n", "glioma_t2w_t1n"), "t2w_t2f": ("t2w", "glioma_t2w_t2f")}
TITLE = {"t1n_t2f": "T1n-trained -> FLAIR", "t1n_t1n": "T1n-trained -> T1n (in-domain)", "t2w_t2f": "T2w-trained -> FLAIR"}
OUT = vt.OUT.parent / "gradmaps3"; OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def model_dir(tc, a):
    hits = sorted((PRED / RUNS[tc][a]).glob("Dataset*/*__3d_fullres")); assert len(hits) == 1, (tc, a, hits); return hits[0]


def fine_share(G, body):
    hp = G - ndi.gaussian_filter(G, 1.0); return float((hp[body] ** 2).sum() / ((G[body] ** 2).sum() + 1e-30))


def run_cell(key, dev, sl=None):
    import torch, torch.nn.functional as F
    tc, src = CELLS[key]; cell = vt.CELLS[src]; od = OUT / key; od.mkdir(exist_ok=True)
    prs = {a: v2.load_predictor(model_dir(tc, a), dev) for a in ARMS}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]); cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "metrics.csv"; done = set(pd.read_csv(f_csv).case.astype(str)) if f_csv.exists() else set()
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; ed = gt == EDEMA
        if ed.sum() < 200:
            continue
        box = e7.patch_at(ed, ps, img.shape); ip, gp = img[box].astype(np.float32), gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, gp = np.pad(ip, padw), np.pad(gp, padw)
        body = ip != 0; E = (gp == EDEMA); tum = np.isin(gp, TUMOUR); dt = ndi.distance_transform_edt(~tum); ring = (gp == 0) & (dt >= 1) & (dt <= 10) & body
        z = int(np.argmax(E.sum((1, 2)))); Et, Rt = torch.from_numpy(E).to(dev), torch.from_numpy(ring).to(dev)
        rows, maps = [], {"img": ip[z].astype(np.float16), "gt": gp[z].astype(np.int8), "z": z}
        for a, pr in prs.items():
            net = pr.network; dec = net.decoder.stages; acc = {k: 0 for k in ("Ge", "Gb", "cam3", "cam4", "lo")}
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval(); acts = {}
                def keep(i):
                    def h(m_, i_, o):
                        acts[i] = o   # returns None: the module output is not replaced
                    return h
                hs = [dec[i].register_forward_hook(keep(i)) for i in (len(dec) - 2, len(dec) - 1)]
                x = torch.from_numpy(ip)[None, None].to(dev).requires_grad_(True)
                with torch.enable_grad(), torch.autocast("cuda", enabled=False):
                    o = e7.fwd(net, x).float(); oth = torch.cat([o[:, :EDEMA], o[:, EDEMA + 1:]], 1); le = (o[:, EDEMA] - torch.logsumexp(oth, 1))[0]
                    lb = (o[:, 0] - torch.logsumexp(o[:, 1:], 1))[0]
                    Je = le[Et].mean(); Jb = lb[Rt].mean()
                    ge, = torch.autograd.grad(Je, x, retain_graph=True)
                    cams = {}
                    for j, (i, A) in enumerate(sorted(acts.items())):
                        gA, = torch.autograd.grad(Je, A, retain_graph=True); w = gA.mean((2, 3, 4), keepdim=True)
                        cam = F.relu((w * A).sum(1, keepdim=True)); cams[j] = F.interpolate(cam, size=tuple(ip.shape), mode="trilinear", align_corners=False)[0, 0].detach()
                    gb, = torch.autograd.grad(Jb, x)
                for h in hs:
                    h.remove()
                Ge, Gb = ge[0, 0].detach().cpu().numpy(), gb[0, 0].detach().cpu().numpy(); c3, c4 = cams[0].cpu().numpy(), cams[1].cpu().numpy(); lev = le.detach().cpu().numpy()
                aG = np.abs(Ge); tot = aG[body].sum() + 1e-30
                rows.append(dict(case=case, arm=a, fold=k, hit=float((lev[E] > 0).mean()), J_e=float(Je), J_b=float(Jb),
                                 fine_e=fine_share(Ge, body), fine_b=fine_share(Gb, body),
                                 in_e=float(aG[E & body].sum() / tot), ring_e=float(aG[ring].sum() / tot), out_e=float(aG[body & ~E & ~ring].sum() / tot),
                                 cam3_in=float(c3[E].sum() / (c3[body].sum() + 1e-30)), cam4_in=float(c4[E].sum() / (c4[body].sum() + 1e-30))))
                for kk, v in (("Ge", Ge), ("Gb", Gb), ("cam3", c3), ("cam4", c4), ("lo", lev)):
                    acc[kk] = acc[kk] + v[z] / len(FOLDS)
            for kk, v in acc.items():
                maps[f"{a}_{kk}"] = np.asarray(v, np.float32)
        np.savez_compressed(od / f"{case}.npz", **maps)
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not f_csv.exists()); done.add(case)
        r = pd.DataFrame(rows).groupby("arm")[["hit", "fine_e"]].mean()
        log(f"[{key}] {case}: hit " + " / ".join(f"{a} {r.loc[a, 'hit']:.2f}" for a in ARMS) + " | fine share " + " / ".join(f"{r.loc[a, 'fine_e']:.3f}" for a in ARMS) + f"; {time.time() - t0:.0f} s")


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    pw = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.asarray(x) != 0) else np.nan   # noqa: E731
    METS = [("hit", "edema called (share)"), ("fine_e", "fine-detail share of the edema gradient"), ("fine_b", "fine-detail share of the background gradient"),
            ("in_e", "edema-gradient mass inside the edema"), ("ring_e", "edema-gradient mass in the peritumoural ring"), ("cam4_in", "Grad-CAM (last stage) inside the edema"),
            ("cam3_in", "Grad-CAM (stage before) inside the edema")]
    L = ["# Gradient / activation maps, noise vs flat vs real fill (FINDINGS section 32, pre-registered decision rule)", "",
         "Per case: fold-mean metric; median over cases; differences: median of per-case differences, Wilcoxon over cases; folds: sign of the median per-case "
         "difference in each fold (+/-).", ""]
    data = {}
    for key in CELLS:
        f = OUT / key / "metrics.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f); D["case"] = D.case.astype(str); data[key] = D
        C = D.groupby(["case", "arm"]).mean(numeric_only=True).unstack("arm")
        L += [f"## {TITLE[key]} (n {C.shape[0]} cases)", "", "| metric | noise | flat | real | real - noise (p; folds) | noise - flat (p; folds) | real - flat (p) | verdict |", "|---|---|---|---|---|---|---|---|"]
        for m, nm in METS:
            c = C[m]; d_rn, d_nf, d_rf = c.real - c.noise, c.noise - c.flat, c.real - c.flat
            def folds(x, y):
                s = []
                for k in FOLDS:
                    t = D[D.fold == k].pivot(index="case", columns="arm", values=m); s.append("+" if (t[x] - t[y]).median() > 0 else "-")
                return "".join(s)
            fr, fn = folds("real", "noise"), folds("noise", "flat"); prn, pnf = pw(d_rn), pw(d_nf)
            sig_rn = prn < 0.01 and len(set(fr)) == 1; sig_nf = pnf < 0.01 and len(set(fn)) == 1
            verdict = "texture-specific" if (sig_rn and abs(d_rn.median()) > abs(d_nf.median())) else ("robustness" if (sig_nf and not prn < 0.01) else "mixed")
            L.append(f"| {nm} | {c.noise.median():.3f} | {c.flat.median():.3f} | {c.real.median():.3f} | {d_rn.median():+.3f} ({prn:.1g}; {fr}) | {d_nf.median():+.3f} ({pnf:.1g}; {fn}) | "
                     f"{d_rf.median():+.3f} ({pw(d_rf):.1g}) | {verdict} |")
        L.append("")
    (OUT / "gradmaps3.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # numbers figure
    keys = list(data); fig, axs = plt.subplots(len(METS), len(keys), figsize=(3.3 * len(keys), 1.9 * len(METS)), squeeze=False)
    for j, key in enumerate(keys):
        C = data[key].groupby(["case", "arm"]).mean(numeric_only=True).unstack("arm")
        for i, (m, nm) in enumerate(METS):
            ax = axs[i, j]
            for xi, a in enumerate(ARMS):
                v = C[m][a].dropna().values; ax.scatter(np.full(len(v), xi) + np.random.default_rng(xi).uniform(-0.15, 0.15, len(v)), v, s=4, color=PAL[a], alpha=0.35, lw=0)
                b = [np.median(np.random.default_rng(0).choice(v, len(v))) for _ in range(2000)]; lo, hi = np.percentile(b, [2.5, 97.5])
                ax.errorbar(xi, np.median(v), yerr=[[max(0, np.median(v) - lo)], [max(0, hi - np.median(v))]], fmt="o", color="k", mfc=PAL[a], ms=5, capsize=2, lw=1)
            for ci in range(len(C)):
                ax.plot(range(3), [C[m][a].iloc[ci] for a in ARMS], color="0.8", lw=0.3, zorder=0)
            ax.set_xticks(range(3)); ax.set_xticklabels(["noise", "flat", "real"], fontsize=7); ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(nm, fontsize=6.5)
            if i == 0:
                ax.set_title(TITLE[key], fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "gradmaps3_numbers.png", dpi=170); fig.savefig(OUT / "gradmaps3_numbers.pdf"); plt.close(fig)
    # maps figure: fixed-rule cases (quantiles of edema volume) for the main cell
    for key in keys:
        fs = {f.stem: f for f in (OUT / key).glob("*.npz")}; vol = {}
        for c, f in fs.items():
            z = np.load(f); vol[c] = int((z["gt"] == EDEMA).sum())
        order = sorted(vol, key=vol.get); pick = [order[int(round(q * (len(order) - 1)))] for q in (0.25, 0.5, 0.75)]
        cols = ["image"] + [f"{a}: |grad|" for a in ARMS] + [f"{a}: fine detail" for a in ARMS] + [f"{a}: Grad-CAM" for a in ARMS]
        fig, axs = plt.subplots(len(pick), len(cols), figsize=(1.55 * len(cols), 1.9 * len(pick)), squeeze=False)
        for i, c in enumerate(pick):
            z = np.load(fs[c]); img = z["img"].astype(np.float32); g = z["gt"] == EDEMA; body = img != 0
            lo_, hi_ = np.percentile(img[body], [1, 99]) if body.any() else (0, 1)
            G = {a: np.abs(z[f"{a}_Ge"]) for a in ARMS}; Fd = {a: np.abs(z[f"{a}_Ge"] - ndi.gaussian_filter(z[f"{a}_Ge"], 1.0)) for a in ARMS}
            for a in ARMS:
                G[a] = ndi.gaussian_filter(G[a] / (G[a][body].sum() + 1e-30), 1.0); Fd[a] = ndi.gaussian_filter(Fd[a] / (np.abs(z[f"{a}_Ge"])[body].sum() + 1e-30), 1.0)
            vg = np.percentile(np.concatenate([G[a][body] for a in ARMS]), 99.5); vf = np.percentile(np.concatenate([Fd[a][body] for a in ARMS]), 99.5)
            for j, nm in enumerate(cols):
                ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.imshow(img, cmap="gray", vmin=lo_, vmax=hi_); ax.contour(g, [0.5], colors="#00b4ff", linewidths=0.6)
                    ax.set_ylabel(f"{c}\nedema vol. q{[25, 50, 75][i]}", fontsize=5.5)
                else:
                    a = nm.split(":")[0]
                    if "grad" in nm:
                        ax.imshow(np.where(body, G[a], np.nan), cmap="magma", vmin=0, vmax=vg)
                    elif "fine" in nm:
                        ax.imshow(np.where(body, Fd[a], np.nan), cmap="magma", vmin=0, vmax=vf)
                    else:
                        ax.imshow(img, cmap="gray", vmin=lo_, vmax=hi_); cm = z[f"{a}_cam4"]; ax.imshow(np.ma.masked_where(cm <= 0.05 * cm.max(), cm), cmap="jet", alpha=0.45)
                    ax.contour(g, [0.5], colors="#00b4ff" if "CAM" in nm else "white", linewidths=0.5)
                if i == 0:
                    ax.set_title(nm, fontsize=6, color=PAL.get(nm.split(":")[0], "k"))
        fig.suptitle(f"{TITLE[key]}: where each model's edema decision is sensitive (|gradient| of the mean edema log-odds, each map normalised to unit mass, "
                     "shared colour scale per row), its fine-detail part (|G - Gaussian(G,1)|), and Seg-Grad-CAM at the last decoder stage. Blue/white contour = ground-truth edema.", fontsize=6.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"gradmaps3_maps_{key}.png", dpi=170); fig.savefig(OUT / f"gradmaps3_maps_{key}.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", default="t1n_t2f"); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        import torch
        run_cell(a.cell, torch.device("cuda"), a.case_slice)


if __name__ == "__main__":
    main()
