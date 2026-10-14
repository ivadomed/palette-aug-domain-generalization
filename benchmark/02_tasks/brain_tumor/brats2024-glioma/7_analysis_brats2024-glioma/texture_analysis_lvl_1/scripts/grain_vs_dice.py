#!/usr/bin/env python
"""Does the per-cell GRAIN advantage of the real-fill model (FINDINGS 37 part 2 section 5) track its per-cell DICE gain?

Grain = fine-detail share of the input gradient of the arm's own decision (energy above a sigma-1 Gaussian), per path depth, from
outputs/plots/pathgrad_all/pathgrad_depth_localfine_cells.csv (d_rf_fine_p2/p3 = median over cases of real - flat, stride-4 / stride-8 path;
d_rn_* = real - noise). The full-gradient grain is not in that table: it is recomputed here from pathgrad_all/<cell>/seeds.csv (columns
<arm>_fine) with the same recipe (target seeds, per-case median, median over cases, Wilcoxon over cases); the p2/p3 values are recomputed
the same way and asserted equal to the table, as a check of the recipe.
Dice = the cell's held-out test contrast, scored by the project's evaluate drivers (02_metrics/<model>/<train contrast>/[ablations/]
<category>_<RUN_ID>/fold{0,1,2}/eval_all.csv). Arms: noise = rung 4 (*lblvor*), flat = rung 4.5 (flatfill), real = rung 5 (v26_6_2 val000);
run ids from visual_proof_erf_outcome.models() / visual_proof_erf_tasks.CELLS. Per case: mean of every finite (label, fold) Dice, i.e. the
ladder engine's load_case_means unit (asserted identical); per cell: mean over cases x 100. Primary = restricted to the cases the grain
was measured on (seeds.csv); secondary = every test case of that contrast. Glioma: all four labels (ladder convention) and SNFH only
(the grain target class, label 2).
CPU only. Run: .venv/bin/python grain_vs_dice.py   -> outputs/plots/pathgrad_all/grain_vs_dice.{md,csv,png}
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon, fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_erf_outcome as vo  # noqa: E402
REPO = THIS.parents[6]; EVAL = REPO / "benchmark/00_commun_scripts/00_03_evaluate"
sys.path.insert(0, str(EVAL)); sys.path.insert(0, str(EVAL.parent / "00_00_utils"))
from eval_folds import EVAL_FOLD_INDICES, filter_fold_dirs  # noqa: E402
import ladder_ood_common as loc  # noqa: E402

T2 = vt.T2; PG = THIS.parent / "outputs/plots/pathgrad_all"
ARMS = ("noise", "flat", "real"); ORDER = vo.ORDER; TARGET_LABEL = {"Glioma": "SNFH"}
KCOL = {"appearance": "#d62728", "interface": "#1f77b4"}
assert tuple(EVAL_FOLD_INDICES) == (0, 1, 2), EVAL_FOLD_INDICES


def wp(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def metrics_dir_of(cell, model_dir):
    """model_dir = .../01_predictions/<model>/<contrast>/<cat>/<run>/Dataset*/<trainer>__3d_fullres; metrics live under the dataset that
    owns the cell's GT (the companion dataset for the mandible cells)."""
    case0 = cell["cases"][0]; rel = Path(cell["gt"](case0)).relative_to(T2); ds = Path(*rel.parts[:2])
    run, cat, contrast, model = model_dir.parents[1].name, model_dir.parents[2].name, model_dir.parents[3].name, model_dir.parents[4].name
    root = T2 / ds / f"8_results_{ds.name}" / "02_metrics" / model / contrast
    hits = [p for p in (root / "ablations" / f"{cat}_{run}", root / f"{cat}_{run}") if p.is_dir()]
    return (hits[0] if hits else None), run


def test_group(cell):
    p = Path(cell["img"](cell["cases"][0]))
    return p.parent.parent.name if p.parent.name == "images_ras" else p.parent.name.replace("imagesTs_", "")


def load_cases(run_dir, group, label=None):
    """{case: mean of finite Dice over (label, fold)} for one test group; same unit as ladder_ood_common.load_case_means."""
    per, nfold = defaultdict(list), defaultdict(set)
    for fd in filter_fold_dirs(sorted(run_dir.glob("fold*"))):
        with (fd / "eval_all.csv").open() as f:
            for row in csv.DictReader(f):
                if row["group"] != group or (label and row["label"] != label):
                    continue
                nfold[row["case"]].add(fd.name)
                try:
                    v = float(row["dice"])
                except ValueError:
                    continue
                if np.isfinite(v):
                    per[row["case"]].append(v)
    return {c: float(np.mean(v)) for c, v in per.items() if v}, {c: len(s) for c, s in nfold.items()}


def grain_from_seeds(key):
    S = pd.read_csv(PG / key / "seeds.csv", dtype=str); S = S[(S.case != "case") & (S.region == "target")]
    cols = [f"{a}_fine{s}" for a in ARMS for s in ("", "_p2", "_p3")]; S[cols] = S[cols].apply(pd.to_numeric, errors="coerce")
    pc = S.groupby("case")[cols].median(); r = {}
    for s, nm in (("", "full"), ("_p2", "p2"), ("_p3", "p3")):
        for b, tag in (("flat", "rf"), ("noise", "rn")):
            d = pc[f"real_fine{s}"] - pc[f"{b}_fine{s}"]; r[f"d_{tag}_fine_{nm}"] = float(d.median()); r[f"p_{tag}_fine_{nm}"] = wp(d)
    return r, sorted(pc.index.astype(str))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); n = int(m.sum())
    if n < 3:
        return dict(n=n, spearman=np.nan, p_s=np.nan, pearson=np.nan, p_p=np.nan)
    s, p = spearmanr(x[m], y[m]); r, q = pearsonr(x[m], y[m])
    return dict(n=n, spearman=float(s), p_s=float(p), pearson=float(r), p_p=float(q))


def main():
    E = pd.read_csv(PG / "pathgrad_depth_localfine_cells.csv").set_index("cell"); rows, srcs, checks = [], [], []
    for key in ORDER:
        cell = vt.CELLS[key]; task = cell["task"]; mods = vo.models(key); grp = test_group(cell); g, gcases = grain_from_seeds(key)
        for nm in ("p2", "p3"):   # recipe check against the published table
            for tag in ("rf", "rn"):
                a, b = g[f"d_{tag}_fine_{nm}"], float(E.loc[key, f"d_{tag}_fine_{nm}"])
                assert abs(a - b) < 1e-6, (key, nm, tag, a, b)
        assert len(gcases) == int(E.loc[key, "n"]), (key, len(gcases), E.loc[key, "n"])
        row = dict(cell=key, task=task, kind=vt.KIND[task], test_group=grp, n_grain=len(gcases), **g)
        D, Dall, Dlab = {}, {}, {}
        for a in ARMS:
            md, run = metrics_dir_of(cell, mods[a])
            row[f"run_{a}"] = run; srcs.append(dict(cell=key, arm=a, metrics_dir=str(md.relative_to(REPO)) if md else "MISSING", group=grp))
            if md is None:
                D[a] = Dall[a] = Dlab[a] = None; continue
            cm, nf = load_cases(md, grp)
            ref = loc.load_case_means(md, "dice").get(grp, {})   # the ladder engine's own loader: must agree exactly
            assert set(ref) == set(cm) and all(abs(ref[c] - cm[c]) < 1e-12 for c in cm), (key, a)
            bad = sorted(c for c, k in nf.items() if k != len(EVAL_FOLD_INDICES))
            if bad:
                checks.append(f"{key}/{a}: {len(bad)} case(s) not in all 3 folds: {bad[:5]}")
            Dall[a] = cm; D[a] = {c: cm[c] for c in gcases if c in cm}
            miss = [c for c in gcases if c not in cm]
            if miss:
                checks.append(f"{key}/{a}: {len(miss)} grain case(s) without a Dice: {miss[:5]}")
            if task in TARGET_LABEL:
                Dlab[a] = {c: v for c, v in load_cases(md, grp, TARGET_LABEL[task])[0].items() if c in set(gcases)}
        for a in ARMS:
            row[f"dice_{a}"] = 100 * np.mean(list(D[a].values())) if D[a] else np.nan
            row[f"dice_{a}_allcases"] = 100 * np.mean(list(Dall[a].values())) if Dall[a] else np.nan
            row[f"n_dice_{a}"] = len(D[a]) if D[a] else 0; row[f"n_dice_{a}_allcases"] = len(Dall[a]) if Dall[a] else 0
            row[f"dice_{a}_target"] = 100 * np.mean(list(Dlab[a].values())) if Dlab.get(a) else np.nan
        for b, tag in (("flat", "rf"), ("noise", "rn")):
            if D["real"] and D[b]:
                cs = sorted(set(D["real"]) & set(D[b])); dd = np.array([D["real"][c] - D[b][c] for c in cs]) * 100
                row[f"ddice_{tag}"] = float(dd.mean()); row[f"p_ddice_{tag}"] = wp(dd); row[f"n_pair_{tag}"] = len(cs)
                ca = sorted(set(Dall["real"]) & set(Dall[b])); row[f"ddice_{tag}_allcases"] = float(np.mean([Dall["real"][c] - Dall[b][c] for c in ca]) * 100)
                if Dlab.get("real") and Dlab.get(b):
                    cl = sorted(set(Dlab["real"]) & set(Dlab[b])); row[f"ddice_{tag}_target"] = float(np.mean([Dlab["real"][c] - Dlab[b][c] for c in cl]) * 100)
            else:
                row[f"ddice_{tag}"] = np.nan
        rows.append(row)
    R = pd.DataFrame(rows); R.to_csv(PG / "grain_vs_dice.csv", index=False); pd.DataFrame(srcs).to_csv(PG / "grain_vs_dice_sources.csv", index=False)

    # ---------------------------------------------------------------- statistics
    L = ["# Grain advantage vs Dice gain of real fill, per cell", "",
         "Grain delta = median over cases of grain(real) - grain(flat or noise) (fine-detail share of the input gradient of the arm's own decision; "
         "p2 = stride-4 path, p3 = stride-8 path, full = whole gradient), from `pathgrad_depth_localfine_cells.csv` (p2/p3, reproduced exactly from "
         "`<cell>/seeds.csv`) and `<cell>/seeds.csv` (full). Dice delta = mean over the grain cases of Dice(real) - Dice(flat or noise), "
         "Dice per case = mean over its labels and folds 0-2 (ladder unit), x100. Sources per arm: `grain_vs_dice_sources.csv`.", ""]
    L += ["## Per cell", "", "| cell | kind | n | Dice noise | Dice flat | Dice real | dDice r-f (p) | dDice r-n (p) | grain r-f p2 | p3 | full | grain r-n p2 | p3 | full |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=2: "" if not np.isfinite(v) else f"{v:+.{d}f}"   # noqa: E731
    fp = lambda v: "" if not np.isfinite(v) else f"{v:.1g}"   # noqa: E731
    for r in R.itertuples():
        L.append(f"| {r.cell} | {r.kind[:4]} | {r.n_grain} | {r.dice_noise:.1f} | {r.dice_flat:.1f} | {r.dice_real:.1f} | {f(r.ddice_rf)} ({fp(r.p_ddice_rf)}) | "
                 f"{f(r.ddice_rn)} ({fp(r.p_ddice_rn)}) | {f(r.d_rf_fine_p2, 3)} | {f(r.d_rf_fine_p3, 3)} | {f(r.d_rf_fine_full, 3)} | {f(r.d_rn_fine_p2, 3)} | "
                 f"{f(r.d_rn_fine_p3, 3)} | {f(r.d_rn_fine_full, 3)} |")
    L += ["", "Glioma, SNFH only (the grain target class): " + "; ".join(f"{r.cell} dDice r-f {r.ddice_rf_target:+.2f}, r-n {r.ddice_rn_target:+.2f}"
                                                                           for r in R.itertuples() if r.task == "Glioma"), ""]
    C = []
    for tag, nm in (("rf", "real - flat"), ("rn", "real - noise")):
        for pth in ("p2", "p3", "full"):
            for unit, df in (("cell", R), ("task mean", R.groupby("task")[[f"d_{tag}_fine_{pth}", f"ddice_{tag}"]].mean().reset_index())):
                c = corr(df[f"d_{tag}_fine_{pth}"], df[f"ddice_{tag}"]); C.append(dict(comparison=nm, path=pth, unit=unit, **c))
            for dv in ("allcases",):
                c = corr(R[f"d_{tag}_fine_{pth}"], R[f"ddice_{tag}_{dv}"]); C.append(dict(comparison=nm, path=pth, unit="cell, Dice on all test cases", **c))
            if pth in ("p2", "p3"):   # leave-one-task-out range of the cell-level Spearman
                loo = [corr(R.loc[R.task != t, f"d_{tag}_fine_{pth}"], R.loc[R.task != t, f"ddice_{tag}"])["spearman"] for t in vt.TASKS]
                C[-3]["loto_spearman_min"], C[-3]["loto_spearman_max"] = float(np.nanmin(loo)), float(np.nanmax(loo))
    C = pd.DataFrame(C); C.to_csv(PG / "grain_vs_dice_correlations.csv", index=False)
    L += ["## Correlations (grain delta vs Dice delta)", "", "| comparison | path | unit | n | Spearman (p) | Pearson (p) | leave-one-task-out Spearman |", "|---|---|---|---|---|---|---|"]
    for r in C.itertuples():
        lo = f"{r.loto_spearman_min:+.2f} .. {r.loto_spearman_max:+.2f}" if hasattr(r, "loto_spearman_min") and np.isfinite(r.loto_spearman_min) else ""
        L.append(f"| {r.comparison} | {r.path} | {r.unit} | {r.n} | {r.spearman:+.2f} ({r.p_s:.2g}) | {r.pearson:+.2f} ({r.p_p:.2g}) | {lo} |")
    L += ["", "Cells are not independent: 23 cells come from 17 models; the 4 glioma cells share 2 models and one test cohort, the abdomen cells 2 models. "
          "The task-mean rows (n = 8) are the conservative unit.", "", "## Sign agreement (cell level, Dice on the grain cases)", ""]
    for tag, nm in (("rf", "real - flat"), ("rn", "real - noise")):
        for pth in ("p2", "p3", "full"):
            x, y = R[f"d_{tag}_fine_{pth}"].values, R[f"ddice_{tag}"].values; m = np.isfinite(x) & np.isfinite(y)
            t = np.array([[np.sum((x[m] > 0) & (y[m] > 0)), np.sum((x[m] > 0) & (y[m] <= 0))], [np.sum((x[m] <= 0) & (y[m] > 0)), np.sum((x[m] <= 0) & (y[m] <= 0))]])
            L.append(f"- {nm}, {pth}: grain+/Dice+ {t[0, 0]}, grain+/Dice- {t[0, 1]}, grain-/Dice+ {t[1, 0]}, grain-/Dice- {t[1, 1]} "
                     f"(agree {t[0, 0] + t[1, 1]}/{m.sum()}, Fisher p {fisher_exact(t)[1]:.2g})")
    if checks:
        L += ["", "## Data checks", ""] + [f"- {c}" for c in checks]
    (PG / "grain_vs_dice.md").write_text("\n".join(L) + "\n")

    # ---------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for i, (tag, nm) in enumerate((("rf", "flat"), ("rn", "noise"))):
        for j, pth in enumerate(("p2", "p3", "full")):
            ax = axs[i, j]; x, y = R[f"d_{tag}_fine_{pth}"], R[f"ddice_{tag}"]
            for r, xv, yv in zip(R.itertuples(), x, y):
                if not (np.isfinite(xv) and np.isfinite(yv)):
                    continue
                hollow = r.n_grain < 5
                ax.scatter(xv, yv, s=28, facecolor="none" if hollow else KCOL[r.kind], edgecolor=KCOL[r.kind], zorder=3)
                ax.annotate(r.cell, (xv, yv), fontsize=5.5, xytext=(3, 2), textcoords="offset points", color="#333333")
            ax.axhline(0, color="#999999", lw=0.6); ax.axvline(0, color="#999999", lw=0.6)
            c = corr(x, y); ax.set_title(f"real - {nm}, grain path {pth}: Spearman {c['spearman']:+.2f} (p {c['p_s']:.2g}), "
                                         f"Pearson {c['pearson']:+.2f} (p {c['p_p']:.2g}), n {c['n']}", fontsize=7.5)
            ax.set_xlabel(f"grain delta (real - {nm}), median over cases", fontsize=8); ax.set_ylabel(f"Dice delta (real - {nm}), points", fontsize=8); ax.tick_params(labelsize=7)
    for k, col in KCOL.items():
        axs[0, 0].scatter([], [], color=col, label=f"{k}-defined target")
    axs[0, 0].scatter([], [], facecolor="none", edgecolor="#555555", label="n < 5 cases (abdomen MRI)"); axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Does the real-fill model's finer gradient grain go with its Dice gain? One point per cell (train contrast -> held-out test contrast); "
                 "Dice on the cases the grain was measured on", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(PG / "grain_vs_dice.png", dpi=170); plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
