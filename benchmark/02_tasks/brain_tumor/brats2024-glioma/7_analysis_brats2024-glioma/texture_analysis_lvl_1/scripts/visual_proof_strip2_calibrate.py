#!/usr/bin/env python
"""
Calibrate the edge-preserving texture-removal path for visual_proof_strip2.py BEFORE any model sees it (CPU/GPU, images only, no network).
For a few cases per dataset: Perona-Malik diffusion over a grid of kappa (percentile of |grad| in the body) x iterations, plus guided-filter and
Gaussian references; reports per setting the median over cases of
  texture left  = high-pass energy (x - G_1 x)^2 inside the target / original
  contrast left = (mean of the target's inner shell 1..3 vox) - (mean of the outer ring 1..3 vox), relative to the original (signed step kept?)
  edge left     = mean |grad G_1 x| on the boundary band (|d| <= 1) / original
The GT only defines where these QC numbers are read. Output: outputs/plots/visual_proof_strip2/calibration.md
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_strip2 as s2  # noqa: E402
import visual_proof_v2 as v2  # noqa: E402

CASES = {"bratsfull_t1n_t2f": 3, "bratsfull_t2w_t1n": 3, "openms_t2w": 3, "pelvis_ct_full": 3}
KAPPA_Q = [50]; ITS = [10, 25]; WS_STEP = [3, 5, 7, 10]; WS_COMP = [0.0, 0.001, 0.01]


def qc(x, x0, tgt):
    din = ndi.distance_transform_edt(tgt); dout = ndi.distance_transform_edt(~tgt)
    inner = tgt & (din >= 1) & (din <= 3); outer = (~tgt) & (dout >= 1) & (dout <= 3); band = ((din <= 1) & tgt) | ((dout <= 1) & ~tgt)
    def m(v):
        hp = v - ndi.gaussian_filter(v, 1.0); g = np.sqrt(sum(ndi.sobel(ndi.gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
        return (hp[tgt] ** 2).mean(), v[inner].mean() - v[outer].mean(), g[band].mean()
    t, c, e = m(x); t0, c0, e0 = m(x0)
    return t / t0, c / c0 if c0 else np.nan, e / e0


def ws_flat(ip, body, step, comp):
    """Piecewise-constant approximation with COMPACT WATERSHED supervoxels (Neubert & Protzel 2014): seeds on a regular grid of spacing `step`
    inside the body, grown on the gradient magnitude of the lightly smoothed image (regions follow image edges), each region -> its mean."""
    from skimage.segmentation import watershed
    g = np.sqrt(sum(ndi.sobel(ndi.gaussian_filter(ip, 0.7), axis=a) ** 2 for a in range(3)))
    mk = np.zeros(ip.shape, np.int32); sl = tuple(slice(step // 2, None, step) for _ in range(3)); grid = np.zeros(ip.shape, bool); grid[sl] = True; grid &= body
    mk[grid] = np.arange(1, grid.sum() + 1)
    lab = watershed(g, markers=mk, compactness=comp, mask=body)
    m = ndi.mean(ip, labels=lab, index=np.arange(lab.max() + 1)); out = ip.copy(); out[lab > 0] = np.asarray(m)[lab[lab > 0]]
    return out.astype(np.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for key, n in CASES.items():
        cell = s2.CELLS[key]; pr0 = v2.load_predictor(cell["models"]["flat"], dev); ps = pr0.configuration_manager.patch_size
        for case in list(cell["cases"])[:n]:
            data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, cell["target"])
            body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
            box = s2.patch_at(tg, ps, img.shape); ip, bp, tp = img[box].astype(np.float32), body[box], tg[box]
            g = np.sqrt(sum(ndi.sobel(ip, axis=a) ** 2 for a in range(3))) / 8.0
            for s in (1.0, 2.0):
                x = np.where(bp, ndi.gaussian_filter(ip, s), ip); rows.append(dict(cell=key, case=case, method=f"gauss s{s:g}", **dict(zip(("tex", "contrast", "edge"), qc(x, ip, tp)))))
            for q in KAPPA_Q:
                kap = float(np.percentile(g[bp], q)); pm = s2.perona_malik(torch.from_numpy(ip).to(dev), torch.from_numpy(bp).to(dev), ITS, kap)
                for it in ITS:
                    x = pm[it].cpu().numpy(); rows.append(dict(cell=key, case=case, method=f"PM q{q} it{it}", **dict(zip(("tex", "contrast", "edge"), qc(x, ip, tp)))))
            for st in WS_STEP:
                for comp in WS_COMP:
                    x = ws_flat(ip, bp, st, comp); rows.append(dict(cell=key, case=case, method=f"WS s{st} c{comp:g}", **dict(zip(("tex", "contrast", "edge"), qc(x, ip, tp)))))
            print(key, case, flush=True)
    df = pd.DataFrame(rows); out = s2.OUT / "calibration.csv"; df.to_csv(out, index=False)
    t = df.groupby(["cell", "method"])[["tex", "contrast", "edge"]].median().reset_index()
    piv = t.pivot(index="method", columns="cell", values=["tex", "contrast", "edge"])
    lines = ["# Calibration of the texture-removal paths (median over cases; 1 = original)", "", piv.round(2).to_string()]
    (s2.OUT / "calibration.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
