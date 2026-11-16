#!/usr/bin/env python3
"""T28b scoring (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T28b.md, sha256 0d47f4af...,
tag reg-T28b-20261116). Reuses t28_score's pair / rho helpers (T28 rule 1 unchanged).

Inputs: DIR/gate_all.tsv and DIR/items_edge_gateall.tsv (t28_edge.py --mode gateall: T28's response gate on every T27
(setting, item) row, own cases), T27/items_scored.tsv (O, read unchanged), T28/gate.tsv (the 18 gate rows already seen).
Edge-measurable = rows passing the gate in both directions. E (mm) = per point E_vox x the mm length of the index step along
the sampled normal, per-case median, per-item median (variant 'orig').
Subsets: 'all' = every edge-measurable row; 'clean' = edge-measurable rows whose item key is not among T28's 18 gate items
([impl] by item key: the same image files appear in several settings, so their E has been seen).
A (level, primary): not supported if median E <= 0.25 x median O; level-consistent if median |O - E| <= 0.25 mm and median
  E > 0; otherwise undetermined.
B (within setting, T28 rule 1 unchanged, restricted): pairs |dE| >= 0.25 mm, O ordered as E; supported if >= 70%, >= 8 pairs,
  >= 2 tasks and rho(O, E) >= 0.5; not supported if <= 50% or rho <= 0.2; otherwise undetermined; undetermined if < 8 pairs.
C (descriptive): rho(O, W) and pairs on W (0.25 mm).
Stated outcome = the clean subset's when it has >= 6 items from >= 3 tasks, else "post hoc only" (both reported).
Usage: python3 t28b_score.py --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import t28_score as S28

ROOT = Path("/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms")


def rule_a(I: pd.DataFrame) -> dict:
    mE, mO = float(I.E_mm.median()), float(I.O.median())
    mabs = float((I.O - I.E_mm).abs().median())
    if mE <= 0.25 * mO:
        o = "M17 not supported"
    elif mabs <= 0.25 and mE > 0:
        o = "M17 level-consistent"
    else:
        o = "undetermined"
    return dict(n_rows=len(I), median_E=mE, median_O=mO, median_abs_O_minus_E=mabs, median_O_minus_E=float((I.O - I.E_mm).median()),
                outcome=o)


def rule_b(I: pd.DataFrame) -> dict:
    P = S28.pairs(I, "E_mm")
    r = S28.rho(I.O, I.E_mm)
    out = S28.rule1(P, r["rho"])
    out["rho_n"] = r["n"]
    if out["n_pairs"] < 8:
        out["outcome"] = "undetermined (< 8 pairs)"
    return out, P


def block(I: pd.DataFrame, tag: str, out: Path) -> dict:
    b, P = rule_b(I)
    P.to_csv(out / f"pairs_E_{tag}.tsv", sep="\t", index=False)
    PW = S28.pairs(I, "W_mm")
    PW.to_csv(out / f"pairs_W_{tag}.tsv", sep="\t", index=False)
    return dict(n_rows=len(I), n_item_keys=int(I.item_key.nunique()), tasks=sorted(I.task.unique()),
                rows_per_task=I.groupby("task").size().to_dict(),
                A=rule_a(I) if len(I) else None, B=b,
                C=dict(rho_O_W=S28.rho(I.O, I.W_mm), pairs_W=S28.count(PW), median_W=float(I.W_mm.median()) if len(I) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    O = pd.read_csv(ROOT / "T27/items_scored.tsv", sep="\t")
    G = pd.read_csv(out / "gate_all.tsv", sep="\t")
    Ed = pd.read_csv(out / "items_edge_gateall.tsv", sep="\t")
    Ed = Ed[Ed.variant == "orig"].drop(columns=["variant", "task"])
    G28 = pd.read_csv(ROOT / "T28/gate.tsv", sep="\t")
    seen = set(G28.item_key)
    I = O[["setting", "task", "item_key", "group", "n_cases", "O", "O_plain", "t", "cf09"]].merge(
        Ed, on=["setting", "item_key"], how="left", suffixes=("", "_edge")).merge(
        G[["setting", "item_key", "dE_dilate_vox", "dE_erode_vox", "ok_dilate", "ok_erode", "ok", "failing"]],
        on=["setting", "item_key"], how="left")
    I["seen_in_T28"] = I.item_key.isin(seen)
    # consistency: T28's 18 gate rows recomputed here must equal T28/gate.tsv (deterministic operator)
    chk = G28.merge(G, on=["setting", "item_key"], suffixes=("_t28", "_t28b"))
    maxdiff = float(np.nanmax(np.abs(np.r_[chk.dE_dilate_vox_t28 - chk.dE_dilate_vox_t28b,
                                           chk.dE_erode_vox_t28 - chk.dE_erode_vox_t28b, chk.orig_t28 - chk.orig_t28b])))
    res = dict(inputs=dict(t27_items_md5=hashlib.md5((ROOT / "T27/items_scored.tsv").read_bytes()).hexdigest(),
                           n_rows=len(I), n_gated=int(I.ok.notna().sum()), n_cases_match=bool((I.n_cases == I.n_cases_edge).all()),
                           t28_gate_rows_rechecked=len(chk), t28_gate_max_abs_diff_vox=maxdiff,
                           n_errors=int(I.n_errors.fillna(0).sum())))
    res["gate"] = dict(n_pass=int(I.ok.sum()), n=len(I),
                       per_task={t: f"{int(g.ok.sum())}/{len(g)}" for t, g in I.groupby("task")},
                       passing=[f"{r.setting}:{r.item_key}" for r in I[I.ok].itertuples()],
                       failing={f"{r.setting}:{r.item_key}": r.failing for r in I[~I.ok].itertuples()})
    M = I[I.ok]
    C = M[~M.seen_in_T28]
    res["all"] = block(M, "all", out)
    res["clean"] = block(C, "clean", out)
    big = len(C) >= 6 and C.task.nunique() >= 3
    res["stated_outcome"] = (dict(source="clean subset", A=res["clean"]["A"]["outcome"], B=res["clean"]["B"]["outcome"])
                             if big else dict(source="post hoc only (clean subset < 6 items or < 3 tasks)",
                                              A_all=res["all"]["A"]["outcome"] if res["all"]["A"] else None,
                                              A_clean=res["clean"]["A"]["outcome"] if res["clean"]["A"] else None,
                                              B_all=res["all"]["B"]["outcome"], B_clean=res["clean"]["B"]["outcome"]))
    res["clean_threshold"] = dict(rows=len(C), item_keys=int(C.item_key.nunique()), tasks=int(C.task.nunique()), met=big)
    (out / "rules.json").write_text(json.dumps(res, indent=1, default=str))
    I.to_csv(out / "items_scored.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(I[["setting", "item_key", "ok", "failing", "dE_dilate_vox", "dE_erode_vox", "seen_in_T28", "O", "E_mm", "E_vox",
             "E_center_mm", "W_mm", "L_median"]].round(3).to_string())
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
