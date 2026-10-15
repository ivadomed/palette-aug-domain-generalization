#!/usr/bin/env python
"""
texture_rule_edgegap.py -- does the TRAIN/EVAL ratio of target-contour edge visibility separate the cells where texture helps from
those where it does not?  (Paul's hypothesis, 2026-10-15.)  Design written and committed BEFORE the gap or any score below was computed.

Outcome per cell (cell = training contrast x eval item of one task, the OOD items of every paper ladder, texture_rule_cells.csv):
  Delta = 100 x (real - flat) Dice, rung 4.5 flat fill -> rung 5 real fill (unchanged from FINDINGS 47/48).
  class: HELPS iff Delta > 1 Dice point, else DOES NOT HELP.
Cells: texture_rule_polarity.EXCLUDE_TASKS removes pancreas (pansegdata + its test sets), pelvis (totalseg-pelvic) and HCC: 56 cells,
  6 tasks. Pelvis and HCC are HELD OUT (kept for a later check of whatever survives here): never loaded, printed or scored here.
Feature (per contrast key, already in texture_rule_interface_cells.csv, texture_rule_interface.py, FINDINGS 48):
  EDGE = log mean |grad z| on the target's 1-voxel boundary shell (1 mm grid, foreground-z-scored image).
  GAP = EDGE(train) - EDGE(eval) = log(edge visibility train / edge visibility eval).
Rule (Paul's direction): GAP > 0 (contour better defined on the training contrast) -> DOES NOT HELP; GAP < 0 -> HELPS. Threshold 0
  (ratio 1), not fitted. Paul allows the reverse direction, so every test is TWO-SIDED (statistic = the better of the two directions).
Tests:
  T1 PRIMARY: sign-rule accuracy, two-sided (max(hits, n - hits)), null = per-key EDGE values permuted WITHIN each task (the task's
     keys exchange their values, gaps recomputed; cells sharing a key move together), 20k draws.  This asks whether the gap separates
     cells beyond what the task already explains.  Kill: p >= 0.05.
  T2: AUC of -GAP for HELPS (threshold-free), two-sided |AUC - 0.5|, same within-task null.
  T3: Spearman(-GAP, Delta), two-sided, same null.
  Reported, not tests: the same three with the null permuting key values ACROSS all tasks (confounded with the task label); hits on
  Holm-significant cells; baselines "never helps" and the paper's conceptual task label (appearance-defined -> helps); per-task hits.
Secondary metric, registered here before its gap was looked at: BGR = log(boundary |grad| / whole-image mean |grad|), the edge relative
  to the image's own gradient level, which removes the global normalisation / resolution scale that EDGE tracks (FINDINGS 48: EDGE
  follows slice thickness). Same rule and tests. Comparison row (already seen in texture_rule_search.md): d gap.
Blindness: the analyst HAD seen EDGE(train) and EDGE(eval) per cell, and the cell-level Spearman of EDGE(eval) with Delta (+0.20,
  texture_rule_interface.md) and of d gap (+0.28). The gap itself, the rule's hits, and every test here were not computed before this
  commit. What is frozen: direction, threshold 0, outcome cut at 1 Dice, nulls, primary = T1 on EDGE.
Output: outputs/tables/texture_rule_edgegap.md, outputs/data/texture_rule_edgegap_cells.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import texture_rule_polarity as T  # noqa: E402
DATA, TAB = T.DATA, T.TAB; NPERM = 20000; RNG = np.random.RandomState(49); CUT = 1.0
APPEARANCE = {"brats2024-glioma", "open-ms", "ispy2"}   # paper's conceptual labels (texture_rule_interface.CONCEPT_INTERFACE complement)
METRICS = {"EDGE": ("edge", "edge_eval"), "BGR": ("log_bgr", "log_bgr_eval")}


def stats(gap, y, delta):
    """two-sided sign-rule hits, |AUC-0.5|, |rho| plus their signed values; rule: gap < 0 -> helps."""
    hits = int(((gap < 0) == y).sum()); n = len(y)
    r = rankdata(-gap); n1 = y.sum(); auc = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    rho = spearmanr(-gap, delta).correlation
    return hits, auc, rho, max(hits, n - hits), abs(auc - 0.5), abs(rho)


def null(c, cols, stratified):
    """permute per-key metric values within each task (stratified) or across all keys; returns the two-sided statistics."""
    kv = pd.concat([pd.Series(c[cols[0]].values, index=c.task + "|" + c.train_key), pd.Series(c[cols[1]].values, index=c.task + "|" + c.eval_key)])
    kv = kv[~kv.index.duplicated()]; tasks = kv.index.str.split("|").str[0]
    tk = (c.task + "|" + c.train_key).values; ek = (c.task + "|" + c.eval_key).values; y = (c.delta > CUT).values; d = c.delta.values
    out = np.empty((NPERM, 3))
    for i in range(NPERM):
        v = kv.values.copy()
        if stratified:
            for t in np.unique(tasks):
                ix = np.where(tasks == t)[0]; v[ix] = v[RNG.permutation(ix)]
        else:
            v = RNG.permutation(v)
        m = pd.Series(v, index=kv.index)
        out[i] = stats(m[tk].values - m[ek].values, y, d)[3:]
    return out


def posthoc_within(c, cols, name, nperm=5000, rng=None):
    """POST HOC (added after T1-T3 were seen): the T1 within-task null keeps no task base rate, so a task where every cell has the
    same class (on-harmony, toothfairy2, open-ms) scores for the gap whenever its training contrasts happen to be the sharper ones.
    Here only WITHIN-task variation counts: (a) hits in tasks with both classes vs the within-task majority; (b) mean over tasks of
    the within-task Spearman(-GAP, Δ), null = key values permuted within task."""
    rng = rng or np.random.RandomState(4949)   # own stream: the pre-registered T1-T3 draws stay identical
    rho_t = lambda cc, g: np.nanmean([spearmanr(-g[ix], cc.delta.values[ix]).correlation for ix in cc.groupby("task").indices.values()])  # noqa: E731
    gap = (c[cols[0]] - c[cols[1]]).values; obs = rho_t(c, gap)
    kv = pd.concat([pd.Series(c[cols[0]].values, index=c.task + "|" + c.train_key), pd.Series(c[cols[1]].values, index=c.task + "|" + c.eval_key)])
    kv = kv[~kv.index.duplicated()]; tasks = kv.index.str.split("|").str[0]
    tk = (c.task + "|" + c.train_key).values; ek = (c.task + "|" + c.eval_key).values; nl = np.empty(nperm)
    for i in range(nperm):
        v = kv.values.copy()
        for t in np.unique(tasks):
            ix = np.where(tasks == t)[0]; v[ix] = v[rng.permutation(ix)]
        m = pd.Series(v, index=kv.index); nl[i] = rho_t(c, m[tk].values - m[ek].values)
    mix = c.groupby("task").filter(lambda g: 0 < (g.delta > CUT).sum() < len(g))
    hm = int(((mix[cols[0]] - mix[cols[1]] < 0) == (mix.delta > CUT)).sum())
    maj = int(mix.groupby("task").apply(lambda g: max((g.delta > CUT).sum(), (g.delta <= CUT).sum())).sum())
    return [f"- POST HOC, within-task only: tasks with both classes ({', '.join(sorted(mix.task.unique()))}) rule hits {hm}/{len(mix)} vs within-task majority {maj}/{len(mix)}; "
            f"mean within-task Spearman(-GAP, Δ) = {obs:+.3f}, two-sided p = {float((np.abs(nl) >= abs(obs) - 1e-12).mean()):.3g} ({nperm} within-task key permutations)", ""]


def main():
    c = T.load_cells("texture_rule_interface_cells.csv")
    assert not c.task.isin(T.EXCLUDE_TASKS).any()
    y = (c.delta > CUT).values; n = len(c); sig = (c.p_holm < 0.05).values
    lab = c.task.isin(APPEARANCE).values
    L = ["# Train/eval edge-visibility gap vs texture gain (rung 4.5 flat -> 5 real); design frozen in texture_rule_edgegap.py", "",
         f"Cells: {n} ({c.task.nunique()} tasks; pancreas excluded, pelvis + HCC held out). HELPS (Δ > {CUT:g} Dice): {y.sum()}/{n}. "
         f"Holm-significant cells: {sig.sum()} ({(y & sig).sum()} of them HELPS).", "",
         "Rule: GAP = log(edge train / edge eval) > 0 -> does not help; < 0 -> helps. Two-sided p (better direction), 20k permutations.", "",
         "## Baselines", "",
         f"- never helps: {(~y).sum()}/{n}; significant cells {(~y & sig).sum()}/{sig.sum()}",
         f"- task label (appearance-defined -> helps): {(lab == y).sum()}/{n}; significant cells {((lab == y) & sig).sum()}/{sig.sum()}", ""]
    for name, cols in METRICS.items():
        gap = (c[cols[0]] - c[cols[1]]).values; c[f"gap_{name}"] = gap
        hits, auc, rho, *obs = stats(gap, y, c.delta.values)
        ns, nu = null(c, cols, True), null(c, cols, False)
        p = lambda nl, j: float((nl[:, j] >= obs[j] - 1e-12).mean())  # noqa: E731
        tag = "PRIMARY" if name == "EDGE" else "secondary"
        L += [f"## {name} gap ({tag})", "",
              f"- T1 sign-rule hits: {hits}/{n} as stated (reverse {n - hits}/{n}); within-task p = {p(ns, 0):.4g}; across-task p = {p(nu, 0):.4g}",
              f"- T2 AUC(-GAP -> helps) = {auc:.3f}; within-task p = {p(ns, 1):.4g}; across-task p = {p(nu, 1):.4g}",
              f"- T3 Spearman(-GAP, Δ) = {rho:+.3f}; within-task p = {p(ns, 2):.4g}; across-task p = {p(nu, 2):.4g}",
              f"- Holm-significant cells, rule as stated: {(((gap < 0) == y) & sig).sum()}/{sig.sum()}",
              "- per task, rule as stated (hits/cells, HELPS cells): " + "; ".join(
                  f"{t} {int(((g[f'gap_{name}'] < 0) == (g.delta > CUT)).sum())}/{len(g)} ({int((g.delta > CUT).sum())})" for t, g in c.groupby("task")), ""]
        L += posthoc_within(c, cols, name)
    d = c.sort_values(["task", "train_key", "eval_key"])
    L += ["## Cells", "", "| task | train | eval | Δ real-flat | p Holm | HELPS | EDGE train | EDGE eval | EDGE gap | BGR gap | rule (EDGE) hit |",
          "|---|---|---|--:|--:|:-:|--:|--:|--:|--:|:-:|"]
    for r in d.itertuples():
        L.append(f"| {r.task} | {r.train_key} | {r.eval_key} | {r.delta:+.2f} | {r.p_holm:.2g} | {'y' if r.delta > CUT else ''} | {r.edge:+.3f} | {r.edge_eval:+.3f} | "
                 f"{r.gap_EDGE:+.3f} | {r.gap_BGR:+.3f} | {'y' if (r.gap_EDGE < 0) == (r.delta > CUT) else ''} |")
    c.to_csv(DATA / "texture_rule_edgegap_cells.csv", index=False)
    (TAB / "texture_rule_edgegap.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
