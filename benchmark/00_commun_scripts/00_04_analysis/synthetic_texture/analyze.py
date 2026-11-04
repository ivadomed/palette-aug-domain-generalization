"""Analysis of the synthetic texture experiment, exactly as preregistered in README.md.

Usage: python analyze.py <runs_dir> <out_dir>
Writes <out_dir>/results.md, results.json, gain_vs_lam.png.
Units: Dice in points (x100). Seed = replicate. OOD Dice of a run = equal-weight mean over the 8
held-out domains of the mean per-image Dice. gain = texture fill - flat fill (paired within seed).
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import t as student_t, wilcoxon

ARMS = ["plain", "flat", "texture"]


def tq(n):
    return float(student_t.ppf(0.975, n - 1))


def signflip(d, alternative):
    """Exact paired sign-flip test of mean(d) (d = per-seed differences)."""
    d = np.asarray(d, float)
    obs = d.mean()
    null = np.array([np.mean(d * np.array(s)) for s in itertools.product([1, -1], repeat=len(d))])
    if alternative == "greater":
        return float(np.mean(null >= obs - 1e-12))
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def main(runs, out):
    os.makedirs(out, exist_ok=True)
    R = {}
    for fn in glob.glob(os.path.join(runs, "*.json")):
        r = json.load(open(fn))
        if "test" not in r:
            continue
        R[(round(r["lam"], 1), r["seed"], r["arm"])] = r
    lams = sorted({k[0] for k in R})
    seeds = sorted({k[1] for k in R})
    complete = [s for s in seeds if all((lam, s, a) in R for lam in lams for a in ARMS)]
    missing = sorted(set(seeds) - set(complete))
    seeds = complete

    def ood(lam, s, a):
        return 100 * float(np.mean([np.mean(t) for t in R[(lam, s, a)]["test"]]))

    def ind(lam, s, a):
        return 100 * float(np.mean(R[(lam, s, a)]["val"]))

    rows, gains = [], {}
    for lam in lams:
        g = np.array([ood(lam, s, "texture") - ood(lam, s, "flat") for s in seeds])
        gains[lam] = g
        # secondary: per test image x domain, Dice averaged over seeds, texture - flat
        per_img = np.mean([np.concatenate([np.array(t) for t in R[(lam, s, "texture")]["test"]])
                           - np.concatenate([np.array(t) for t in R[(lam, s, "flat")]["test"]]) for s in seeds], 0)
        nz = per_img[np.abs(per_img) > 0]
        row = dict(lam=lam, n_seeds=len(seeds),
                   **{f"ind_{a}": float(np.mean([ind(lam, s, a) for s in seeds])) for a in ARMS},
                   **{f"ood_{a}": float(np.mean([ood(lam, s, a) for s in seeds])) for a in ARMS},
                   **{f"ood_{a}_sd": float(np.std([ood(lam, s, a) for s in seeds], ddof=1)) for a in ARMS},
                   gain=float(g.mean()), gain_sd=float(g.std(ddof=1)),
                   gain_ci95=[float(g.mean() - tq(len(g)) * g.std(ddof=1) / np.sqrt(len(g))),
                              float(g.mean() + tq(len(g)) * g.std(ddof=1) / np.sqrt(len(g)))],
                   gain_pos_seeds=int((g > 0).sum()),
                   p_gain_greater=signflip(g, "greater"), p_gain_two=signflip(g, "two"),
                   tex_minus_plain=float(np.mean([ood(lam, s, "texture") - ood(lam, s, "plain") for s in seeds])),
                   flat_minus_plain=float(np.mean([ood(lam, s, "flat") - ood(lam, s, "plain") for s in seeds])),
                   p_img_wilcoxon=float(wilcoxon(nz).pvalue) if len(nz) > 10 else None,
                   plain_ood_over_ind=float(np.mean([ood(lam, s, "plain") for s in seeds])
                                            / np.mean([ind(lam, s, "plain") for s in seeds])))
        rows.append(row)

    lam_arr = np.array(lams)
    slopes = np.array([np.polyfit(lam_arr, [gains[lam][j] for lam in lams], 1)[0] for j in range(len(seeds))])
    p_slope = signflip(slopes, "greater")
    lo, hi = min(lams), max(lams)
    T1 = bool(slopes.mean() > 0 and p_slope < 0.05)
    T2 = bool(gains[hi].mean() > 0 and signflip(gains[hi], "greater") < 0.05)
    T3 = bool(abs(gains[lo].mean()) <= 2.0)
    if T1 and T2 and T3:
        verdict = "REPRODUCED"
    elif sum([T1, T2, T3]) == 2:
        verdict = "PARTIAL"
    else:
        verdict = "NOT REPRODUCED"
    V1 = all(r["ind_plain"] >= 80 for r in rows)
    V2 = {lam: r["plain_ood_over_ind"] <= 0.8 for lam, r in zip(lams, rows) if lam in (lo, hi)}
    summary = dict(seeds=seeds, missing_seeds=missing, lams=lams, rows=rows,
                   slopes=slopes.tolist(), slope_mean=float(slopes.mean()), p_slope=p_slope,
                   T1_trend=T1, T2_texture_end=T2, T3_boundary_end=T3, verdict=verdict,
                   V1_plain_ind_ge80=V1, V2_plain_ood_drop={str(k): v for k, v in V2.items()},
                   commits=sorted({r["commit"] for r in R.values()}),
                   auglab=sorted({r["auglab_pin"] for r in R.values()}),
                   gpu_hours_train=sum(r["t_train"] for r in R.values()) / 3600)
    json.dump(summary, open(os.path.join(out, "results.json"), "w"), indent=1)

    L = ["# Synthetic texture experiment: results", "",
         f"Seeds used: {seeds} (incomplete, dropped: {missing}); commits {summary['commits']}; AugLab {summary['auglab']}", "",
         "Dice points. IND = in-domain validation, OOD = mean over 8 held-out domains (mean over seeds).",
         "gain = texture fill - flat fill (OOD), paired by seed; p from exact sign-flip over seeds.", "",
         "| lam | IND plain | IND flat | IND texture | OOD plain | OOD flat | OOD texture | gain (sd) | 95% CI | seeds >0 | p one-sided | p two-sided | p per-image (Wilcoxon) | texture-plain | flat-plain |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = r["gain_ci95"]
        cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        L.append(f"| {r['lam']:.1f} | {r['ind_plain']:.1f} | {r['ind_flat']:.1f} | {r['ind_texture']:.1f} | "
                 f"{r['ood_plain']:.1f} | {r['ood_flat']:.1f} | {r['ood_texture']:.1f} | {r['gain']:+.2f} ({r['gain_sd']:.2f}) | {cis} | "
                 f"{r['gain_pos_seeds']}/{r['n_seeds']} | {r['p_gain_greater']:.4f} | {r['p_gain_two']:.4f} | "
                 f"{r['p_img_wilcoxon'] if r['p_img_wilcoxon'] is None else format(r['p_img_wilcoxon'], '.2g')} | "
                 f"{r['tex_minus_plain']:+.2f} | {r['flat_minus_plain']:+.2f} |")
    L += ["", f"Trend: per-seed OLS slope of gain on lam, mean {slopes.mean():+.2f} Dice per unit lam "
          f"(per-seed {np.round(slopes, 2).tolist()}), one-sided sign-flip p = {p_slope:.4f}", "",
          f"T1 trend (slope > 0, p < 0.05): {T1}", f"T2 texture end lam={hi} (gain > 0, p < 0.05): {T2}",
          f"T3 boundary end lam={lo} (|gain| <= 2 Dice): {T3}", f"**Verdict (preregistered rule): {verdict}**", "",
          f"Validity V1 (plain IND >= 80 at every lam): {V1}; V2 (plain OOD <= 0.8 x IND at the ends): {V2}",
          f"Training GPU-time (sum of process wall time): {summary['gpu_hours_train']:.1f} h"]
    open(os.path.join(out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for j in range(len(seeds)):
        ax[0].plot(lams, [gains[lam][j] for lam in lams], color="0.75", lw=0.8)
    ax[0].plot(lams, [gains[lam].mean() for lam in lams], "o-", color="C3", lw=2, label="mean over seeds")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("lam (0 = mean-step / boundary target, 1 = texture-only target)")
    ax[0].set_ylabel("OOD Dice gain, texture - flat fill")
    ax[0].legend(fontsize=8)
    for a, c in zip(ARMS, ["C7", "C0", "C3"]):
        ax[1].plot(lams, [r[f"ood_{a}"] for r in rows], "o-", color=c, label=f"{a} OOD")
        ax[1].plot(lams, [r[f"ind_{a}"] for r in rows], "--", color=c, alpha=0.6, label=f"{a} IND")
    ax[1].set_xlabel("lam")
    ax[1].set_ylabel("Dice")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gain_vs_lam.png"), dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
