"""QC figure: one anatomy per sweep point (rows): training-domain image, label, flat fill, texture fill
(same torch seed -> same partition and region means), and the same anatomy in 3 held-out domains.
Also prints simple generator checks: target-vs-host mean and std difference per lam (train domain).
Usage: python qc_figure.py <out.png>
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generator as G  # noqa: E402
import pa_arms  # noqa: E402

LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main(out):
    seed = 0
    dtr = G.train_domain(seed)
    tds = G.test_domains()[:3]
    rng = np.random.default_rng(G.TRAIN_ANAT_SEED(seed))
    anat = G.sample_anatomy(rng)
    tf_t, tf_f = pa_arms.make_transform("texture"), pa_arms.make_transform("flat")
    cols = ["train domain", "label", "flat fill", "texture fill", "held-out 1", "held-out 2", "held-out 3"]
    fig, ax = plt.subplots(len(LAMS), len(cols), figsize=(2.1 * len(cols), 2.1 * len(LAMS)))
    for r, lam in enumerate(LAMS):
        img, lab = G.render(anat, dtr, lam)
        x = torch.from_numpy(img)[None, None, None]
        y = torch.from_numpy(lab.astype(np.int64))[None, None, None]
        torch.manual_seed(123 + r)
        flat = tf_f.apply_transform(x, {"seg": y}, {})[0, 0, 0].numpy()
        torch.manual_seed(123 + r)
        tex = tf_t.apply_transform(x, {"seg": y}, {})[0, 0, 0].numpy()
        panels = [img, lab, flat, tex] + [G.render(anat, d, lam)[0] for d in tds]
        for c, p in enumerate(panels):
            a = ax[r, c]
            a.imshow(p, cmap="gray", interpolation="nearest")
            if c != 1:
                a.contour(lab, levels=[0.5], colors="r", linewidths=0.4)
            a.set_xticks([]), a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=8)
        ax[r, 0].set_ylabel(f"lam={lam:.1f}", fontsize=9)
        host = (anat["tissue"] == G.HOST) & ~anat["target"]
        t = anat["target"]
        print(f"lam={lam:.1f} train-domain target-host mean diff {img[t].mean() - img[host].mean():+.3f}, "
              f"std target {img[t].std():.3f} host {img[host].std():.3f}")
    fig.suptitle("Synthetic texture sweep (red = target contour; lam = texture fraction)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
