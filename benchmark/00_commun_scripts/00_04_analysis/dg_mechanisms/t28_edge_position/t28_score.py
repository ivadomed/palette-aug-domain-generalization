#!/usr/bin/env python3
"""T28 scoring (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T10_T28.md, section T28, sha256 e761e43e...).

Inputs: T27/items_scored.tsv (O per (setting, item), read unchanged) and items_edge_full.tsv (t28_edge.py: E, W per
(setting, item)). Unit: item pair inside one setting; across settings only descriptive Spearman over the (setting, item) rows.
Rule 1 (M17's predictor): pairs with |E_a - E_b| >= 0.25 mm; ordered = O larger on the item with the larger E (O ties count as
  not ordered). Supported: >= 70% ordered, >= 8 pairs, pairs from >= 2 tasks, AND Spearman rho(O, E) over items >= 0.5.
  Not supported: <= 50% ordered, or rho <= 0.2. Otherwise undetermined.
Rule 2 (level): median over items of (O - E); within +-0.5 mm -> level match; else the residual is reported.
Rule 3 (M16 contrast, descriptive): rho(O, W) and the same pair count with W ([impl] same 0.25 mm threshold).
Sensitivity (descriptive): rules 1-2 with E measured from the surface-voxel centre instead of the drawn boundary.
Usage: python3 t28_score.py --out DIR   (reads DIR/items_edge_full.tsv)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T27 = Path("/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T27")
DIFF = 0.25


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return dict(rho=float("nan"), n=int(m.sum()))
    r = spearmanr(x[m], y[m])
    return dict(rho=float(r.statistic if hasattr(r, "statistic") else r[0]), n=int(m.sum()))


def pairs(I: pd.DataFrame, pcol: str) -> pd.DataFrame:
    rows = []
    for s, g in I.groupby("setting"):
        g = g[np.isfinite(g[pcol]) & np.isfinite(g.O)]
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            if abs(a[pcol] - b[pcol]) < DIFF:
                continue
            hi, lo = (a, b) if a[pcol] > b[pcol] else (b, a)
            rows.append(dict(setting=s, task=a.task, hi=hi.item_key, lo=lo.item_key, x_hi=hi[pcol], x_lo=lo[pcol],
                             O_hi=hi.O, O_lo=lo.O, ordered=bool(hi.O > lo.O)))
    return pd.DataFrame(rows, columns=["setting", "task", "hi", "lo", "x_hi", "x_lo", "O_hi", "O_lo", "ordered"])


def count(P: pd.DataFrame) -> dict:
    n = len(P)
    k = int(P.ordered.sum()) if n else 0
    return dict(n_pairs=n, n_ordered=k, share=k / n if n else float("nan"),
                tasks=sorted(P.task.unique()) if n else [],
                per_task={t: f"{int(g.ordered.sum())}/{len(g)}" for t, g in P.groupby("task")} if n else {},
                per_setting={t: f"{int(g.ordered.sum())}/{len(g)}" for t, g in P.groupby("setting")} if n else {})


def rule1(P: pd.DataFrame, r: float) -> dict:
    c = count(P)
    sh = c["share"]
    if c["n_pairs"] >= 8 and sh >= 0.70 and len(c["tasks"]) >= 2 and r >= 0.5:
        o = "M17 supported"
    elif (np.isfinite(sh) and sh <= 0.50) or r <= 0.2:
        o = "M17 not supported"
    else:
        o = "undetermined"
    return dict(c, rho_O_E=r, outcome=o)


def level(I: pd.DataFrame, ecol: str) -> dict:
    d = (I.O - I[ecol]).dropna()
    med = float(d.median())
    return dict(n_items=int(len(d)), median_O_minus_E=med, q25=float(d.quantile(0.25)), q75=float(d.quantile(0.75)),
                outcome="level match (within +-0.5 mm)" if abs(med) <= 0.5 else f"residual {med:+.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    O = pd.read_csv(T27 / "items_scored.tsv", sep="\t")
    G27 = pd.read_csv(T27 / "gate.tsv", sep="\t")
    Ed = pd.read_csv(out / "items_edge_full.tsv", sep="\t")
    Ed = Ed[Ed.variant == "orig"].drop(columns=["variant", "task"])
    I = O[["setting", "task", "item_key", "group", "n_cases", "O", "O_plain", "t", "cf09"]].merge(
        Ed, on=["setting", "item_key"], how="left", suffixes=("", "_edge"))
    res = dict(inputs=dict(t27_items_md5=hashlib.md5((T27 / "items_scored.tsv").read_bytes()).hexdigest(),
                           t27_gate_ok=f"{int(G27.ok.sum())}/{len(G27)}",
                           n_rows=len(I), n_rows_with_E=int(I.E_mm.notna().sum()),
                           n_cases_match=bool((I.n_cases == I.n_cases_edge).all())))
    rE = rho(I.O, I.E_mm)
    PE = pairs(I, "E_mm")
    PE.to_csv(out / "pairs_E.tsv", sep="\t", index=False)
    res["rule1"] = rule1(PE, rE["rho"])
    res["rule1"]["rho_n"] = rE["n"]
    res["rule2"] = level(I, "E_mm")
    PW = pairs(I, "W_mm")
    PW.to_csv(out / "pairs_W.tsv", sep="\t", index=False)
    res["rule3"] = dict(rho_O_W=rho(I.O, I.W_mm), pairs_W=count(PW), rho_E_W=rho(I.E_mm, I.W_mm))
    # sensitivity (descriptive): voxel-centre origin
    PC = pairs(I, "E_center_mm")
    rC = rho(I.O, I.E_center_mm)
    res["sens_center_origin"] = dict(rule1=rule1(PC, rC["rho"]), rule2=level(I, "E_center_mm"))
    # descriptive: per-task medians, and setting-level rho
    res["per_task"] = {t: dict(n_rows=len(g), median_O=float(g.O.median()), median_E=float(g.E_mm.median()),
                               median_W=float(g.W_mm.median()), median_O_minus_E=float((g.O - g.E_mm).median()))
                       for t, g in I.groupby("task")}
    Sg = I.groupby("setting").agg(O=("O", "median"), E=("E_mm", "median"), W=("W_mm", "median"))
    res["settings_rho"] = dict(O_E=rho(Sg.O, Sg.E), O_W=rho(Sg.O, Sg.W))
    res["drop_shares"] = dict(weak_median=float(I.share_dropped_weak.median()), out_of_array_max=float(I.share_out_of_array.max()),
                              no_zero_crossing_max=float(I.share_no_zero_crossing.max()),
                              W_undefined_max=float(I.share_W_undefined.max()), resampled_cases=int(I.n_resampled.sum()))
    (out / "rules.json").write_text(json.dumps(res, indent=1, default=str))
    I.to_csv(out / "items_scored.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(I[["setting", "item_key", "O", "E_mm", "E_center_mm", "W_mm", "L_median", "t", "n_cases", "n_errors",
             "share_dropped_weak"]].round(3).to_string())
    print(PE.round(3).to_string())
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
