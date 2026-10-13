#!/usr/bin/env python
"""
Fine-texture sensitivity of each model's decision, gradient-based only (2026-10-12): radially averaged power spectrum of the SIGNED input gradient
(Hann-windowed 32^3 cube around the voxel) of the edema / background log-odds, flat-fill (rung 4.5) vs real-fill (rung 5) glioma models, on
unperturbed test images. Summary per voxel: HF share = share of the gradient power at |k| > 0.25 cycles/voxel.
Pre-registration, definitions and results: outputs/plots/ig_texture/ig_texture.md (written before the run). No intervention on the input.
Modes:
  GPU  --cells <cell> [--case-slice i:j]     sampled voxels (both-hit / rescued / both-miss edema, background) -> <cell>/spec.csv, <cell>/counts.csv
  GPU  --maps --cases A B C                   dense HF-share map on one slice per case (main cell)       -> t1n_t2f/maps/<case>.npz
  CPU  --report                               ig_texture_results.md, ig_texture_fig.{pdf,png}, example_cases.txt, per-case CSVs
"""
from __future__ import annotations
import argparse, copy, os, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
OUT = Path(os.environ.get("IG_OUT") or (THIS.parent / "outputs/plots/ig_texture")); OUT.mkdir(parents=True, exist_ok=True)

BATCH = int(os.environ.get("IG_BATCH", 4)); K = int(os.environ.get("IG_K", 8)); NEAR = 15; MIN_STRAT = 20; CUBE = 32; EDEMA = 2; STRIDE = 4
STRATA = ("both", "resc", "miss", "bg"); NB = 16; HF_BIN = 8; WHITE_HF = np.nan          # bins 8..15 = |k| in (8/32, 16/32] = (0.25, 0.5] cycles/voxel
TITLE = {"t1n_t2f": "T1n-trained → FLAIR", "t1n_t1n": "T1n-trained → T1n", "t2w_t2f": "T2w-trained → FLAIR"}
CELL_ORDER = ("t1n_t2f", "t1n_t1n", "t2w_t2f")


def _vt():
    import visual_proof_erf_tasks as vt  # noqa: E402  (torch / nnU-Net; GPU side only)
    return vt


def cells():
    vt = _vt(); R = vt.GLR
    return {"t1n_t2f": dict(models=vt.G_T1N, **vt.nn(f"{R}/imagesTs_t2f", f"{R}/labelsTr")),
            "t1n_t1n": dict(models=vt.G_T1N, **vt.nn(f"{R}/imagesTs_t1n", f"{R}/labelsTr")),
            "t2w_t2f": dict(models=vt.G_T2W, **vt.nn(f"{R}/imagesTs_t2f", f"{R}/labelsTr"))}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------------------- spectra
_WIN = _BIN = _OK = None


def spectrum(g, v):
    """signed gradient (numpy, patch shape) -> normalised radial power spectrum (16 bins of 1/32 cycles/voxel) of the Hann-windowed 32^3 cube at v."""
    global _WIN, _BIN, _OK
    if _WIN is None:
        h = np.hanning(CUBE); _WIN = h[:, None, None] * h[None, :, None] * h[None, None, :]
        f = np.fft.fftfreq(CUBE); rho = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
        _OK = (rho > 0) & (rho <= 0.5); _BIN = np.minimum(np.ceil(rho[_OK] * CUBE).astype(int) - 1, NB - 1)
    h = CUBE // 2; c = g[v[0] - h:v[0] + h, v[1] - h:v[1] + h, v[2] - h:v[2] + h].astype(np.float64)
    p = np.abs(np.fft.fftn((c - c.mean()) * _WIN)) ** 2
    prof = np.bincount(_BIN, p[_OK], minlength=NB); return prof / (prof.sum() + 1e-300)


def hf(prof):
    return float(np.asarray(prof)[..., HF_BIN:].sum(-1)) if np.ndim(prof) == 1 else np.asarray(prof)[..., HF_BIN:].sum(-1)


# ----------------------------------------------------------------------------------------------------------------------------- GPU side
def lo_cls(o, c):
    import torch
    return o[:, c] - torch.logsumexp(torch.cat([o[:, :c], o[:, c + 1:]], 1), 1)


def grads(net, xt, vox, cls, dev):
    """signed input gradients of the log-odds of class cls[i] at voxel vox[i] (batched copies of the patch) -> list of numpy arrays."""
    import torch
    out = []
    for i in range(0, len(vox), BATCH):
        vb, cb = vox[i:i + BATCH], cls[i:i + BATCH]
        xin = xt[None, None].repeat(len(vb), 1, 1, 1, 1).clone().requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            o = _vt().e7.fwd(net, xin)
            f = sum(lo_cls(o[b:b + 1, :, v[0], v[1], v[2]][..., None], c)[0, 0] for b, (v, c) in enumerate(zip(vb, cb)))
            g, = torch.autograd.grad(f, xin)
        out += [g[b, 0].detach().cpu().numpy() for b in range(len(vb))]
    return out


def setup(key, dev):
    import torch
    torch.backends.cudnn.benchmark = True
    vt = _vt(); cell = cells()[key]
    prs = {a: vt.v2.load_predictor(f(), dev) for a, f in cell["models"].items()}
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    rnet = copy.deepcopy(prs["flat"].network); torch.manual_seed(0)
    for m in rnet.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    rnet.eval(); return cell, prs, rnet


def prep_case(cell, case, prs, dev):
    """patch (visual_proof_bg rule) + fold-mean-logit edema log-odds of each model + strata."""
    import torch
    from scipy import ndimage as ndi
    vt = _vt(); e7, v2 = vt.e7, vt.v2; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = gt == EDEMA
    if tg.sum() < 50:
        return None
    body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
    box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
    padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
    if any(w for _, w in padw):
        ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
    inner = np.zeros(gp.shape, bool); inner[tuple(slice(e7.EDGE, n - e7.EDGE) for n in gp.shape)] = True
    L = {}
    for a, pr in prs.items():
        net = pr.network; acc = 0
        for k in v2.FOLDS:
            net.load_state_dict(pr.list_of_parameters[k]); net.eval()
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                acc = acc + e7.fwd(net, torch.from_numpy(ip)[None, None].to(dev)).float()
        L[a] = lo_cls(acc / len(v2.FOLDS), EDEMA)[0].cpu().numpy()
    tum = gp > 0; sd = ndi.distance_transform_edt(tum) - ndi.distance_transform_edt(~tum)
    E = (gp == EDEMA) & bp & inner; hf_, hr = L["flat"] > 0, L["real"] > 0
    S = {"all": E, "both": E & hf_ & hr, "resc": E & ~hf_ & hr, "miss": E & ~hf_ & ~hr, "lost": E & hf_ & ~hr, "bg": (gp == 0) & bp & inner & (sd <= -3) & (sd >= -NEAR)}
    return dict(ip=ip, bp=bp, gp=gp, inner=inner, L=L, S=S)


def run_cell(key, dev, sl=None):
    import torch
    v2 = _vt().v2; cell, prs, rnet = setup(key, dev); od = OUT / key; od.mkdir(parents=True, exist_ok=True)
    cases = sorted(cell["cases"])[0::2]; cases = cases[slice(*[int(x) for x in sl.split(":")])] if sl else cases
    f_sp, f_ct = od / "spec.csv", od / "counts.csv"
    done = set(pd.read_csv(f_sp).case.astype(str)) if f_sp.exists() and f_sp.stat().st_size else set()
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); P = prep_case(cell, case, prs, dev)
        if P is None:
            log(f"[{key}] {case}: edema < 50 voxels, skipped"); continue
        S = P["S"]; rng = np.random.default_rng(zlib.crc32(case.encode())); vox, cls, st = [], [], []
        for s in STRATA:
            idx = np.argwhere(S[s])
            if len(idx) >= MIN_STRAT:
                for i in sorted(rng.choice(len(idx), min(K, len(idx)), replace=False)):
                    vox.append(tuple(int(c) for c in idx[i])); cls.append(0 if s == "bg" else EDEMA); st.append(s)
        xt = torch.from_numpy(P["ip"]).to(dev); rows = []
        base = [dict(cell=key, case=case, j=j, stratum=s, z=v[0], y=v[1], x=v[2]) for j, (v, s) in enumerate(zip(vox, st))]
        for a, pr in prs.items():
            net = pr.network; profs = []
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                pk = [spectrum(g, v) for g, v in zip(grads(net, xt, vox, cls, dev), vox)]; profs.append(pk)
                rows += [dict(b, model=a, fold=str(k), hf=hf(p), **{f"p{i}": float(p[i]) for i in range(NB)}) for b, p in zip(base, pk)]
            pm = np.mean(np.array(profs), 0)
            rows += [dict(b, model=a, fold="ens", hf=hf(p), **{f"p{i}": float(p[i]) for i in range(NB)}) for b, p in zip(base, pm)]
        pr_ = [spectrum(g, v) for g, v in zip(grads(rnet, xt, vox, cls, dev), vox)]
        rows += [dict(b, model="random", fold="ens", hf=hf(p), **{f"p{i}": float(p[i]) for i in range(NB)}) for b, p in zip(base, pr_)]
        ct = dict(cell=key, case=case, **{f"n_{s}": int(m.sum()) for s, m in S.items()},
                  **{f"hit_{a}": float((P["L"][a][S["all"]] > 0).mean()) for a in ("flat", "real")})
        pd.DataFrame([ct]).to_csv(f_ct, mode="a", index=False, header=not (f_ct.exists() and f_ct.stat().st_size))
        pd.DataFrame(rows).to_csv(f_sp, mode="a", index=False, header=not (f_sp.exists() and f_sp.stat().st_size)); done.add(case)
        e = pd.DataFrame(rows); e = e[e.fold == "ens"].groupby("model").hf.median()
        log(f"[{key}] {case}: {len(vox)} voxels ({', '.join(f'{s} {st.count(s)}' for s in STRATA)}), median HF flat {e.get('flat', np.nan):.3f} real "
            f"{e.get('real', np.nan):.3f} random {e.get('random', np.nan):.3f}; {time.time() - t0:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def run_maps(cases, dev, key="t1n_t2f"):
    import torch
    v2 = _vt().v2; cell, prs, _ = setup(key, dev); od = OUT / key / "maps"; od.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if (od / f"{case}.npz").exists():
            continue
        t0 = time.time(); P = prep_case(cell, case, prs, dev); S = P["S"]
        z = int(np.argmax(S["resc"].sum((1, 2)))) if S["resc"].any() else int(np.argmax(S["all"].sum((1, 2))))
        m2 = P["bp"][z] & P["inner"][z]; yy, xx = np.meshgrid(np.arange(0, m2.shape[0], STRIDE), np.arange(0, m2.shape[1], STRIDE), indexing="ij")
        ok = m2[yy, xx]; pts = [(z, int(y), int(x)) for y, x in zip(yy[ok], xx[ok])]
        xt = torch.from_numpy(P["ip"]).to(dev); H = {}
        for a, pr in prs.items():
            net = pr.network; acc = np.zeros((len(pts), NB))
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                for i0 in range(0, len(pts), 32):
                    pc_ = pts[i0:i0 + 32]; acc[i0:i0 + 32] += np.array([spectrum(g, v) for g, v in zip(grads(net, xt, pc_, [EDEMA] * len(pc_), dev), pc_)])
            H[a] = hf(acc / len(v2.FOLDS))
        np.savez_compressed(od / f"{case}.npz", z=z, pts=np.array(pts), hf_flat=H["flat"], hf_real=H["real"], img=P["ip"][z].astype(np.float16), gt=P["gp"][z].astype(np.int8),
                            hit_flat=(P["L"]["flat"][z] > 0), hit_real=(P["L"]["real"][z] > 0), body=P["bp"][z])
        log(f"[maps] {case}: slice {z}, {len(pts)} grid voxels, {time.time() - t0:.0f} s")


def spoke_metrics(g, v):
    """POST HOC (added after the pre-registered results): is the near-Nyquist gradient power isotropic or on the axis-aligned spokes that a stride-2
    transposed-convolution checkerboard produces? Returns the fold's normalised 3-D power + masks are applied by the caller."""
    h = CUBE // 2; c = g[v[0] - h:v[0] + h, v[1] - h:v[1] + h, v[2] - h:v[2] + h].astype(np.float64)
    p = np.abs(np.fft.fftn((c - c.mean()) * _win3())) ** 2; return p / p[_mask3()["ok"]].sum()


_M3 = {}


def _win3():
    if "w" not in _M3:
        hh = np.hanning(CUBE); _M3["w"] = hh[:, None, None] * hh[None, :, None] * hh[None, None, :]
    return _M3["w"]


def _mask3():
    if "ok" not in _M3:
        f = np.fft.fftfreq(CUBE); F = np.stack(np.meshgrid(f, f, f, indexing="ij")); rho = np.sqrt((F ** 2).sum(0)); a = np.abs(F)
        ok = (rho > 0) & (rho <= 0.5)
        spoke = np.zeros(rho.shape, bool)
        for ax in range(3):
            o = [i for i in range(3) if i != ax]; spoke |= (a[ax] >= 14 / 32) & (a[o[0]] <= 2 / 32) & (a[o[1]] <= 2 / 32)
        _M3.update(ok=ok, top=ok & (rho > 13 / 32), hfm=ok & (rho > 0.25), spoke=spoke & ok)
    return _M3


def spoke_row(P3):
    M = _mask3(); ns = ~M["spoke"]
    return dict(spoke_share_top=float(P3[M["top"] & M["spoke"]].sum() / P3[M["top"]].sum()),
                spoke_count_top=float((M["top"] & M["spoke"]).sum() / M["top"].sum()),
                hf_nospoke=float(P3[M["hfm"] & ns].sum() / P3[M["ok"] & ns].sum()),
                top_nospoke=float(P3[M["top"] & ns].sum() / P3[M["ok"] & ns].sum()))


def run_spoke(key, dev, sl=None):
    v2 = _vt().v2; import torch
    cell, prs, rnet = setup(key, dev); od = OUT / key; f_o = od / "spoke.csv"
    Sp = pd.read_csv(od / "spec.csv", dtype={"fold": str}); Sp = Sp[(Sp.model == "flat") & (Sp.fold == "ens")]
    cases = sorted(Sp.case.unique()); cases = cases[slice(*[int(x) for x in sl.split(":")])] if sl else cases
    done = set(pd.read_csv(f_o).case.astype(str)) if f_o.exists() and f_o.stat().st_size else set()
    for case in cases:
        if case in done:
            continue
        t0 = time.time(); P = prep_case(cell, case, prs, dev); s = Sp[Sp.case == case].sort_values("j")
        vox = [tuple(int(a) for a in r) for r in s[["z", "y", "x"]].values]; cls = [0 if t == "bg" else EDEMA for t in s.stratum]
        xt = torch.from_numpy(P["ip"]).to(dev); rows = []; acc = {}
        for a, pr in prs.items():
            net = pr.network; acc[a] = [0] * len(vox)
            for k in v2.FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval()
                for i, (g, v) in enumerate(zip(grads(net, xt, vox, cls, dev), vox)):
                    acc[a][i] = acc[a][i] + spoke_metrics(g, v) / len(v2.FOLDS)
        acc["random"] = [spoke_metrics(g, v) for g, v in zip(grads(rnet, xt, vox, cls, dev), vox)]
        for a, L_ in acc.items():
            rows += [dict(cell=key, case=case, j=int(j), stratum=t, model=a, **spoke_row(p3)) for j, t, p3 in zip(s.j, s.stratum, L_)]
        pd.DataFrame(rows).to_csv(f_o, mode="a", index=False, header=not (f_o.exists() and f_o.stat().st_size)); done.add(case)
        d = pd.DataFrame(rows).groupby("model")[["spoke_share_top", "hf_nospoke"]].median()
        log(f"[spoke {key}] {case}: {len(vox)} voxels; spoke share of |k|>0.41 power (count fraction {rows[0]['spoke_count_top']:.3f}) "
            + ", ".join(f"{m} {d.loc[m, 'spoke_share_top']:.3f}" for m in d.index) + "; HF share w/o spokes " + ", ".join(f"{m} {d.loc[m, 'hf_nospoke']:.3f}" for m in d.index) + f"; {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------------- CPU report
PAL = {"flat": "#1f77b4", "real": "#d62728", "random": "#7f7f7f"}
SLAB = {"edema": "edema (prevalence-weighted)", "both": "both hit", "resc": "rescued", "miss": "both miss", "bg": "background (3-15 vox)"}


def pw(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def boot(x, B=10000, seed=0, stat=np.median):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed); m = stat(x[r.integers(0, len(x), (B, len(x)))], axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def f3(x, ci=None):
    s = f"{x:.3f}" if np.isfinite(x) else "n/a"
    return s + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None and np.isfinite(ci[0]) else "")


def fd(x, ci):
    return (f"{x:+.3f}" if np.isfinite(x) else "n/a") + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if np.isfinite(ci[0]) else "")


def fp(p):
    return f"{p:.1g}" if np.isfinite(p) else "n/a"


def case_table(Sp, Ct, fold="ens", col="hf"):
    """per (cell, case, model): per-stratum mean of col over sampled voxels + prevalence-weighted edema value."""
    g = Sp[Sp.fold == fold].groupby(["cell", "case", "model", "stratum"])[col].mean().unstack("stratum")
    ct = Ct.set_index(["cell", "case"])
    w = np.stack([ct.reindex(g.index.droplevel("model"))[f"n_{s}"].values for s in ("both", "resc", "miss")], 1).astype(float)
    v = g.reindex(columns=["both", "resc", "miss"]).values; w = np.where(np.isfinite(v), w, 0)
    g["edema"] = np.where(w.sum(1) > 0, np.nansum(v * w, 1) / np.maximum(w.sum(1), 1e-9), np.nan)
    return g


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    Sp = pd.concat([pd.read_csv(OUT / c / "spec.csv", dtype={"fold": str}) for c in CELL_ORDER if (OUT / c / "spec.csv").exists()], ignore_index=True)
    Ct = pd.concat([pd.read_csv(OUT / c / "counts.csv") for c in CELL_ORDER if (OUT / c / "counts.csv").exists()], ignore_index=True).drop_duplicates(["cell", "case"])
    nd = int(Sp.duplicated(["cell", "case", "j", "model", "fold"]).sum()); Sp = Sp.drop_duplicates(["cell", "case", "j", "model", "fold"])
    rpv = Sp.groupby(["cell", "case", "j"]).size(); integ = f"integrity: {nd} duplicate rows dropped; rows per voxel {sorted(set(rpv.tolist()))} (expected [9])"
    G = {f: case_table(Sp, Ct, f) for f in ("ens", "0", "1", "2")}
    global WHITE_HF
    rw = np.random.default_rng(1); WHITE_HF = float(np.mean([hf(spectrum(rw.standard_normal((CUBE,) * 3), (CUBE // 2,) * 3)) for _ in range(300)]))
    L = ["## 1. Results", "", integ, "", f"Cases per cell: " + ", ".join(f"{TITLE[c]} {Ct[Ct.cell == c].case.nunique()}" for c in CELL_ORDER if c in set(Ct.cell)) +
         f". Sampled voxels: {len(Sp[(Sp.fold == 'ens') & (Sp.model == 'flat')])}. Edema hit rate on the patch (mean over cases, flat / real): " +
         "; ".join(f"{TITLE[c]} {Ct[Ct.cell == c].hit_flat.mean():.2f} / {Ct[Ct.cell == c].hit_real.mean():.2f}" for c in CELL_ORDER if c in set(Ct.cell)) +
         f". Reference: HF share of white noise through the same window = {WHITE_HF:.3f} (power counted per frequency sample, so a flat spectrum puts ~7/8 above 0.25)" +
         ". Median rescued share of GT edema: " + "; ".join(f"{TITLE[c]} {(Ct[Ct.cell == c].n_resc / Ct[Ct.cell == c].n_all).median():.2f}" for c in CELL_ORDER if c in set(Ct.cell)) + ".", "",
         "### 1.1 HF share (gradient power at |k| > 0.25 cycles/voxel), per case; median over cases [95 % case-bootstrap CI]; p = Wilcoxon real - flat", "",
         "| cell | voxels | n cases | flat | real | random init | real - flat [CI] | p | fold 0 real - flat (p) | fold 1 | fold 2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    D = {}
    for c in CELL_ORDER:
        for s in ("edema", "both", "resc", "miss", "bg"):
            g = G["ens"]
            try:
                t = g.loc[c][s].unstack("model").dropna()
            except KeyError:
                continue
            if t.empty:
                continue
            d = t.real - t.flat; D[(c, s)] = t; pf = []
            for k in ("0", "1", "2"):
                tk = G[k].loc[c][s].unstack("model").dropna(); dk = tk.real - tk.flat; pf.append(f"{dk.median():+.3f} ({fp(pw(dk))})")
            L.append(f"| {TITLE[c]} | {SLAB[s]} | {len(t)} | {f3(t.flat.median(), boot(t.flat))} | {f3(t.real.median(), boot(t.real))} | {f3(t.random.median())} | "
                     f"{fd(d.median(), boot(d))} | {fp(pw(d))} | " + " | ".join(pf) + " |")
    L += ["", "### 1.2 Paired cross-cell tests (same patients): per-case d = HF real - HF flat (edema, prevalence-weighted)", "",
          "| comparison | n | median d main | median d other | median (d main - d other) [CI] | p |", "|---|---|---|---|---|---|"]
    if ("t1n_t2f", "edema") in D:
        dm = D[("t1n_t2f", "edema")].real - D[("t1n_t2f", "edema")].flat
        for cc in ("t1n_t1n", "t2w_t2f"):
            if (cc, "edema") in D:
                do = D[(cc, "edema")].real - D[(cc, "edema")].flat; j = dm.index.intersection(do.index); x = dm[j] - do[j]
                L.append(f"| {TITLE['t1n_t2f']} vs {TITLE[cc]} | {len(j)} | {dm[j].median():+.3f} | {do[j].median():+.3f} | {fd(x.median(), boot(x))} | {fp(pw(x))} |")
        for cc in CELL_ORDER:
            if (cc, "edema") in D and (cc, "bg") in D:
                de = D[(cc, "edema")].real - D[(cc, "edema")].flat; db = D[(cc, "bg")].real - D[(cc, "bg")].flat; j = de.index.intersection(db.index); x = de[j] - db[j]
                L.append(f"| {TITLE[cc]}: edema vs background | {len(j)} | {de[j].median():+.3f} | {db[j].median():+.3f} | {fd(x.median(), boot(x))} | {fp(pw(x))} |")
    # ---- POST HOC (added after the pre-registered results were seen): band split of the HF share, and the checkerboard-spoke test
    Sp["mid"] = Sp[[f"p{i}" for i in range(8, 13)]].sum(1); Sp["top"] = Sp[[f"p{i}" for i in range(13, NB)]].sum(1)
    L += ["", "### 1.4 POST HOC: where above 0.25 cycles/voxel is the difference? (share of gradient power; mid = 0.25-0.41, top = 0.41-0.5 cycles/voxel)", "",
          "| cell | voxels | band | flat | real | random | real - flat [CI] | p | per fold real - flat (p) |", "|---|---|---|---|---|---|---|---|---|"]
    for col in ("mid", "top"):
        Gb = {f: case_table(Sp, Ct, f, col) for f in ("ens", "0", "1", "2")}
        for c in CELL_ORDER:
            for s in ("edema", "resc", "bg"):
                try:
                    t = Gb["ens"].loc[c][s].unstack("model").dropna()
                except KeyError:
                    continue
                d = t.real - t.flat; pf = []
                for k in ("0", "1", "2"):
                    tk = Gb[k].loc[c][s].unstack("model").dropna(); dk = tk.real - tk.flat; pf.append(f"{dk.median():+.3f} ({fp(pw(dk))})")
                L.append(f"| {TITLE[c]} | {SLAB[s]} | {col} | {t.flat.median():.3f} | {t.real.median():.3f} | {t.random.median():.3f} | {fd(d.median(), boot(d))} | {fp(pw(d))} | {'; '.join(pf)} |")
    sk = [OUT / c / "spoke.csv" for c in CELL_ORDER if (OUT / c / "spoke.csv").exists()]
    if sk:
        K_ = pd.concat([pd.read_csv(f) for f in sk], ignore_index=True).drop_duplicates(["cell", "case", "j", "model"])
        L += ["", "### 1.5 POST HOC: checkerboard-spoke test (full 3-D spectra of the same voxels, fold-averaged normalised power)", "",
              f"spoke = frequency samples with one coordinate |f| >= 14/32 and the other two |f| <= 2/32 (axis-aligned Nyquist spike of a stride-2 transposed "
              f"convolution, widened by the Hann main lobe); they are {K_.spoke_count_top.iloc[0]:.3f} of the samples at |k| > 0.41. Per case: mean over the sampled voxels "
              "(edema = equal weight over sampled edema voxels). Median over cases [CI]; p = Wilcoxon real - flat.", "",
              "| cell | voxels | metric | flat | real | random | real - flat [CI] | p |", "|---|---|---|---|---|---|---|---|"]
        for c in CELL_ORDER:
            kc = K_[K_.cell == c]
            if kc.empty:
                continue
            for nm, ss in (("edema", ["both", "resc", "miss"]), ("rescued", ["resc"]), ("background", ["bg"])):
                kk = kc[kc.stratum.isin(ss)]
                for met in ("spoke_share_top", "hf_nospoke", "top_nospoke"):
                    t = kk.groupby(["case", "model"])[met].mean().unstack().dropna(); d = t.real - t.flat
                    L.append(f"| {TITLE[c]} | {nm} | {met} | {t.flat.median():.3f} | {t.real.median():.3f} | {t.random.median():.3f} | {fd(d.median(), boot(d))} | {fp(pw(d))} |")
    # mean spectra per cell / voxel group with case-bootstrap CI of the mean
    pc = [f"p{i}" for i in range(NB)]; SPEC = {}
    L += ["", "### 1.3 Mean normalised spectra (case means; share of power per 1/32-cycle bin, bins 1-16)", "", "| cell | voxels | model | " + " | ".join(str(i + 1) for i in range(NB)) + " |", "|---|---|---|" + "---|" * NB]
    for c in CELL_ORDER:
        for grp, ss in (("edema", ["both", "resc", "miss"]), ("resc", ["resc"]), ("bg", ["bg"])):
            e = Sp[(Sp.cell == c) & (Sp.fold == "ens") & Sp.stratum.isin(ss)]
            if e.empty:
                continue
            for a in ("flat", "real", "random"):
                M = e[e.model == a].groupby("case")[pc].mean().values; SPEC[(c, grp, a)] = M
                L.append(f"| {TITLE[c]} | {grp} | {a} | " + " | ".join(f"{x:.3f}" for x in M.mean(0)) + " |")
    (OUT / "ig_texture_results.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    out = G["ens"].reset_index(); out.to_csv(OUT / "hf_share_per_case.csv", index=False)
    # fixed-rule example cases (main cell)
    cm = Ct[Ct.cell == "t1n_t2f"].set_index("case"); share = (cm.n_resc / cm.n_all).dropna().sort_values(); ex = []
    for q in (0.25, 0.5, 0.75):
        tgt = share.quantile(q, interpolation="nearest"); ex.append((q, share.index[int(np.argmin(np.abs(share.values - tgt)))], float(tgt)))
    (OUT / "example_cases.txt").write_text("\n".join(f"{c}\t{q}\t{s:.3f}" for q, c, s in ex) + "\n")
    figure(D, SPEC, ex, plt, GridSpec, GridSpecFromSubplotSpec)


def shell_counts():
    spectrum(np.zeros((CUBE + 2,) * 3), (CUBE // 2 + 1,) * 3)   # initialises the bin map
    return np.bincount(_BIN, minlength=NB).astype(float)


def figure(D, SPEC, ex, plt, GridSpec, GridSpecFromSubplotSpec):
    from matplotlib.ticker import NullFormatter
    from scipy.interpolate import griddata
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.75, 1.0], hspace=0.45, wspace=0.22, left=0.08, right=0.985, top=0.95, bottom=0.075)
    md = OUT / "t1n_t2f" / "maps"; have = [(q, c, s) for q, c, s in ex if (md / f"{c}.npz").exists()]
    ga = GridSpecFromSubplotSpec(max(1, len(have)), 4, gs[0, 0], wspace=0.03, hspace=0.06)
    if have:
        maps = [np.load(md / f"{c}.npz") for _, c, _ in have]; dense = []
        for z in maps:
            pts = z["pts"][:, 1:]; gy, gx = np.mgrid[0:z["img"].shape[0], 0:z["img"].shape[1]]; m = z["body"]
            dd = {}
            for a in ("flat", "real"):
                arr = np.full(m.shape, np.nan)
                for (py, px), val in zip(pts, z[f"hf_{a}"]):
                    arr[py:py + STRIDE, px:px + STRIDE] = val   # each grid voxel shown as its STRIDE x STRIDE block (no interpolation)
                dd[a] = np.where(m, arr, np.nan)
            dense.append(dd)
        vals = np.concatenate([np.concatenate([d["flat"][np.isfinite(d["flat"])], d["real"][np.isfinite(d["real"])]]) for d in dense])
        lo_, hi_ = np.percentile(vals, [1, 99]); dl = np.nanpercentile(np.abs(np.concatenate([(d["real"] - d["flat"]).ravel() for d in dense])), 99)
        for i, ((q, c, s), z, d) in enumerate(zip(have, maps, dense)):
            img = z["img"].astype(float); body = z["body"]; ok = np.isfinite(d["flat"]); ys, xs = np.where(ok)
            yb = slice(max(ys.min() - 2, 0), ys.max() + 3); xb = slice(max(xs.min() - 2, 0), xs.max() + 3)
            gt = (z["gt"] == EDEMA)[yb, xb].astype(float)
            for col, (nm, arr) in enumerate((("FLAIR + calls", None), ("flat-fill", d["flat"]), ("real-fill", d["real"]), ("real − flat", d["real"] - d["flat"]))):
                ax = fig.add_subplot(ga[i, col]); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if arr is None:
                    ax.imshow(img[yb, xb], cmap="gray", interpolation="nearest")
                    ax.contour(gt, [0.5], colors="#2ca02c", linewidths=0.7)
                    ax.contour(z["hit_flat"][yb, xb].astype(float), [0.5], colors=PAL["flat"], linewidths=0.6)
                    ax.contour(z["hit_real"][yb, xb].astype(float), [0.5], colors=PAL["real"], linewidths=0.6, linestyles="--")
                    ax.set_ylabel(f"P{int(q * 100)}\nrescued {s:.2f}", fontsize=6.5)
                else:
                    if col < 3:
                        im = ax.imshow(arr[yb, xb], cmap="magma", vmin=lo_, vmax=hi_, interpolation="nearest")
                    else:
                        imd = ax.imshow(arr[yb, xb], cmap="RdBu_r", vmin=-dl, vmax=dl, interpolation="nearest")
                    ax.contour(gt, [0.5], colors="#2ca02c" if col == 3 else "w", linewidths=0.6)
                if i == 0:
                    ax.set_title(nm, fontsize=7)
        p0 = fig.axes[0].get_position(); p2 = fig.axes[2].get_position(); p3 = fig.axes[3].get_position(); yb_ = fig.axes[-1].get_position().y0 - 0.03
        cax = fig.add_axes([fig.axes[1].get_position().x0, yb_, p2.x1 - fig.axes[1].get_position().x0, 0.009])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("HF share of the edema-decision gradient (|k| > 0.25 cyc/vox)", fontsize=6, labelpad=1); cb.ax.tick_params(labelsize=5.5, pad=1)
        cax = fig.add_axes([p3.x0, yb_, p3.x1 - p3.x0, 0.009]); cb = fig.colorbar(imd, cax=cax, orientation="horizontal", ticks=[-0.05, 0.05]); cb.ax.set_xticklabels(["\u22120.05", "+0.05"]); cb.ax.tick_params(labelsize=5.5, pad=1)
        fig.text(p0.x0 - 0.01, yb_ - 0.062, "column 1: green = GT edema, blue / red dashed = flat / real call edema; maps: one value per 4x4 grid block", fontsize=5.5)
    fig.text(0.005, 0.965, "a", fontweight="bold", fontsize=9)
    ax = fig.add_subplot(gs[0, 1]); xt, xl, pos, heads = [], [], 0, []
    for c in CELL_ORDER:
        p0_ = pos
        for s in ("edema", "resc", "bg"):
            if (c, s) not in D:
                continue
            t = D[(c, s)]
            for k, a in enumerate(("flat", "real")):
                m = t[a].median(); lo, hi = boot(t[a]); ax.errorbar(pos + (k - 0.5) * 0.36, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=2.6, color=PAL[a], elinewidth=0.8,
                                                                    label=("flat-fill (rung 4.5)" if a == "flat" else "real-fill (rung 5)") if pos == 0 else None)
            xt.append(pos); xl.append({"edema": "edema", "resc": "rescued", "bg": "backgr."}[s]); pos += 1
        heads.append(((p0_ + pos - 1) / 2, c)); pos += 0.8
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=5.8, rotation=90); ax.set_ylabel("HF share of gradient power (case median, 95% CI)")
    ax.axhline(WHITE_HF, color="0.5", lw=0.5, ls=":"); ax.text(pos - 0.9, WHITE_HF, "white noise", fontsize=5.2, color="0.4", ha="right", va="bottom")
    for x_, c in heads:
        ax.text(x_, 1.005, TITLE[c].replace(" → ", "→\n"), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
    ax.legend(loc="lower left", frameon=False, fontsize=6, handletextpad=0.1, borderaxespad=0.0)
    fig.text(0.645, 0.965, "b", fontweight="bold", fontsize=9); fig.text(0.005, 0.375, "c", fontweight="bold", fontsize=9)
    gc = GridSpecFromSubplotSpec(1, 3, gs[1, :], wspace=0.28); fr = (np.arange(NB) + 0.5) / 32; sc = shell_counts()
    for i, c in enumerate(CELL_ORDER):
        ax = fig.add_subplot(gc[i])
        for grp, ls in (("edema", "-"), ("bg", ":")):
            for a in ("flat", "real", "random"):
                if (c, grp, a) not in SPEC:
                    continue
                M = SPEC[(c, grp, a)] / sc; M = M / M.sum(1, keepdims=True); m = M.mean(0); r = np.random.default_rng(0)
                bs = np.array([M[r.integers(0, len(M), len(M))].mean(0) for _ in range(2000)]); lo, hi = np.percentile(bs, [2.5, 97.5], 0)
                ax.plot(fr, m, ls, color=PAL[a], lw=0.9, label=(f"{a}-fill" if a != "random" else "random init") if grp == "edema" and i == 0 else None)
                ax.fill_between(fr, lo, hi, color=PAL[a], alpha=0.25, lw=0)
        ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks([0.02, 0.03, 0.05, 0.1]); ax.set_yticklabels(["0.02", "0.03", "0.05", "0.1"]); ax.axvline(0.25, color="k", lw=0.5, ls="--"); ax.set_xlim(0, 0.5); ax.set_title(TITLE[c], fontsize=7)
        ax.set_xlabel("spatial frequency |k| (cycles / voxel)")
        if i == 0:
            from matplotlib.lines import Line2D
            h_, l_ = ax.get_legend_handles_labels(); h_ += [Line2D([], [], color="k", lw=0.9), Line2D([], [], color="k", lw=0.9, ls=":")]; l_ += ["edema voxels", "background voxels"]
            ax.set_ylabel("radially averaged power\n(normalised, case mean, 95% CI)"); ax.legend(h_, l_, loc="lower right", frameon=False, fontsize=5.5, handletextpad=0.2, labelspacing=0.2)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"ig_texture_fig.{e}", dpi=300)
    plt.close(fig); print("figure written")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice"); ap.add_argument("--maps", action="store_true"); ap.add_argument("--spoke", action="store_true")
    ap.add_argument("--cases", nargs="*", default=[]); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.report:
        report(); return
    import torch
    dev = torch.device("cuda")
    if a.maps:
        run_maps(a.cases, dev); return
    if a.spoke:
        for c in a.cells:
            run_spoke(c, dev, a.case_slice)
        return
    for c in a.cells:
        run_cell(c, dev, a.case_slice)


if __name__ == "__main__":
    main()
