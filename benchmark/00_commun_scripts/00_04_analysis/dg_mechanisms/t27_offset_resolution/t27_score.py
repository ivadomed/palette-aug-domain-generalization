#!/usr/bin/env python3
"""T27 scoring (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...).

Reads items.tsv (t27_offsets.py). Primary rule (within setting, unit = item pair): every pair of OOD items of one setting
whose t differs by a factor >= 1.5; M16 predicts O(thicker) > O(thinner).
  M16 supported: >= 70% of pairs in the predicted direction, >= 8 pairs, pairs from >= 2 tasks.
  M16 contradicted: <= 50% in the predicted direction (>= 8 pairs).  Otherwise, or < 8 pairs: undetermined.
Secondary (descriptive): (a) O / t; (b) Spearman rho of O vs t and O vs target size over items and over settings (setting
median of O vs setting median of t); (c) CF09 split; (d) the same rule on the plain model (reference only); sensitivity: the
rule with t from the GT (scoring) grid.
Usage: python3 t27_score.py --items TSV --out DIR
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RATIO = 1.5
OWN = {"pansegdata", "totalsegmri-pancreas", "msd-pancreas", "amos-pancreas", "chaos", "amos", "sliver07"}


def cf09(item_key: str) -> str:
    """Registration (c): mask from another contrast = Brain non-T1w, MS non-FLAIR, Mandible HaN-Seg MR; own mask =
    Pancreas, CHAOS, Abdomen companions (CT); everything else unclassified."""
    ds, it = item_key.split("__", 1)
    if ds == "on-harmony":
        return "own" if it == "T1w" else "other"
    if ds == "open-ms":
        return "own" if it == "flair" else "other"
    if ds == "hanseg" and it.startswith("mr"):
        return "other"
    if ds in OWN:
        return "own"
    return "unclassified"


def pairs(I: pd.DataFrame, ocol: str, tcol: str) -> pd.DataFrame:
    rows = []
    for s, g in I.groupby("setting"):
        g = g[np.isfinite(g[ocol]) & np.isfinite(g[tcol])]
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            hi, lo = (a, b) if a[tcol] >= b[tcol] else (b, a)
            if lo[tcol] <= 0 or hi[tcol] / lo[tcol] < RATIO:
                continue
            rows.append(dict(setting=s, task=a.task, thick=hi.item_key, thin=lo.item_key, t_thick=hi[tcol], t_thin=lo[tcol],
                             ratio=hi[tcol] / lo[tcol], O_thick=hi[ocol], O_thin=lo[ocol],
                             predicted=bool(hi[ocol] > lo[ocol])))
    return pd.DataFrame(rows, columns=["setting", "task", "thick", "thin", "t_thick", "t_thin", "ratio", "O_thick", "O_thin",
                                       "predicted"])


def outcome(P: pd.DataFrame) -> dict:
    n = len(P)
    k = int(P.predicted.sum()) if n else 0
    tasks = sorted(P.task.unique()) if n else []
    share = k / n if n else float("nan")
    if n >= 8 and share >= 0.70 and len(tasks) >= 2:
        o = "M16 supported"
    elif n >= 8 and share <= 0.50:
        o = "M16 contradicted"
    else:
        o = "undetermined"
    per_task = {t: f"{int(g.predicted.sum())}/{len(g)}" for t, g in P.groupby("task")} if n else {}
    return dict(n_pairs=n, n_predicted=k, share=share, tasks=tasks, per_task=per_task, outcome=o)


def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return dict(rho=float("nan"), n=int(m.sum()))
    r = spearmanr(np.asarray(x)[m], np.asarray(y)[m])
    return dict(rho=float(r.statistic if hasattr(r, "statistic") else r[0]), n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    I = pd.read_csv(a.items, sep="\t")
    I["cf09"] = I.item_key.map(cf09)
    I["O_over_t"] = I.O / I.t
    res = {}
    P = pairs(I, "O", "t")
    P.to_csv(out / "pairs.tsv", sep="\t", index=False)
    res["primary"] = outcome(P)
    res["sens_t_gt"] = outcome(pairs(I, "O", "t_gt"))
    Pp = pairs(I, "O_plain", "t")
    Pp.to_csv(out / "pairs_plain.tsv", sep="\t", index=False)
    res["plain_reference"] = outcome(Pp)
    # (a)
    res["a_O_over_t"] = dict(n_items=int(np.isfinite(I.O_over_t).sum()), n_gt_1=int((I.O_over_t > 1).sum()),
                             median=float(I.O_over_t.median()), n_O_positive=int((I.O > 0).sum()))
    # (b)
    Sg = I.groupby("setting").agg(O=("O", "median"), t=("t", "median"), target_ml=("target_ml", "median"), task=("task", "first"))
    res["b_spearman"] = dict(items_O_t=rho(I.O, I.t), items_O_size=rho(I.O, I.target_ml),
                             settings_O_t=rho(Sg.O, Sg.t), settings_O_size=rho(Sg.O, Sg.target_ml),
                             items_O_t_gt=rho(I.O, I.t_gt))
    # (c)
    res["c_cf09"] = {k: dict(n_items=len(g), median_O=float(g.O.median()), median_t=float(g.t.median()))
                     for k, g in I.groupby("cf09")}
    # (d)
    res["d_plain"] = dict(median_O_plain=float(I.O_plain.median()), items_O_plain_t=rho(I.O_plain, I.t))
    (out / "rules.json").write_text(json.dumps(res, indent=1, default=str))
    I.to_csv(out / "items_scored.tsv", sep="\t", index=False)
    cols = ["setting", "item_key", "cf09", "n_cases", "O", "O_plain", "t", "p", "t_gt", "p_gt", "target_ml", "t_train", "p_train",
            "O_over_t", "n_rand_arms", "all_folds"]
    pd.set_option("display.width", 250)
    print(I[cols].round(3).to_string())
    print(Sg.round(3).to_string())
    print(P.round(3).to_string())
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
