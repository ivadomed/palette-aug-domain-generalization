#!/usr/bin/env python3
"""T11 scoring (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...).

Reads t11_cases.py outputs. Per setting (own in-domain item, folds 0-2 pooled, unit = case x fold):
  E = share of the total Dice deficit sum(1 - dice_fg) carried by case-folds with an empty prediction (non-empty GT);
  F = share of the false-positive voxel mass, components >= 10 voxels, that sits in components whose minimum distance to the
      GT is > 20 mm (F_all: the same with the < 10-voxel components included, reported apart).
Classification of each P14 setting: ME05 if E >= 0.5; M08 if E < 0.5 and F >= 0.5; mixed otherwise (F undefined -> mixed).
Outcome: M08 favoured on P14 if >= 9/13 are M08; ME05 favoured if >= 9/13 are ME05; else mixed. Thresholds as registered
(out of 13), not rescaled to the computable settings.
M08 prediction 2: CF14 (median over in-domain test cases) vs R28 = SynthSeg-noEM IND / SynthSeg-EM IND (paper table), Spearman
rho over settings, predicted >= 0.5 with the sign kept when any one task is left out. No p-value.
Gate (registered): recomputed SynthSeg-noEM in-domain Dice (label Dice, paper estimand: per fold the mean over all (case,
label) values, aggregate_from_config.cross_fold_stats) within 0.5 Dice of the paper's IND row; where that row pools other
cohorts (Breast T1-CE, HCC, Pancreas T2w) the own column of the stored eval_all.csv is the reference instead. Label-level QC
vs the eval_all.csv found by the T04 lookup (|diff| > 1e-4) is reported, not gated.
Usage: python3 t11_score.py --dir DIR --tex TEX
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t01_step_noise"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
sys.path.insert(0, str(REPO / "paper/scripts"))
import t01_score as TS  # noqa: E402
import aggregate_from_config as AGG  # noqa: E402
from make_suppl_tables import TASKS  # noqa: E402

SNAP_TASK = {"Glioma": "Glioma", "MS": "MS", "Breast": "Breast", "HCC": "HCC", "Abdomen": "Abdomen", "Brain": "Brain",
             "Mandible": "Mandible", "Pelvis": "Pelvis", "Pancreas": "Pancreas", "Spine": "Spine"}
# P14 members, mechanisms/v2/patterns.tsv (R28 <= 0.5): Glioma x4, MS x2, Breast x2, HCC, Pelvis x2, Pancreas x2
P14 = ["Glioma T1w", "Glioma T1-CE", "Glioma T2w", "Glioma FLAIR", "MS FLAIR", "MS T1w", "Breast T1-CE", "Breast T2w",
       "HCC T1-CE", "Pelvis CT", "Pelvis MRI", "Pancreas T1-CE", "Pancreas T2w"]
MULTI_COHORT_IND = {"Breast T1-CE", "HCC T1-CE", "Pancreas T2w"}
TOL = 0.5


def label_of(s: str) -> str:
    task, tc = s.split("_", 1)
    for lab, _ds, _t, mods in TASKS:
        if lab == task:
            return f"{lab} {mods[tc]}"
    raise KeyError(s)


def fold_value(df: pd.DataFrame, col: str) -> float:
    per = defaultdict(list)
    for f, v in zip(df.fold, df[col]):
        per[f].append(v)
    return AGG.cross_fold_stats(per)[0] * 100.0


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    r = spearmanr(x[m], y[m])
    return float(r.statistic if hasattr(r, "statistic") else r[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tex", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    R = pd.read_csv(D / "roster.tsv", sep="\t")
    C = pd.read_parquet(D / "case_rows.parquet")
    L = pd.read_parquet(D / "label_rows.parquet")
    K = pd.read_parquet(D / "fp_components.parquet")
    CF = pd.read_parquet(D / "cf14_cases.parquet")
    ST = pd.read_parquet(D / "stored_label_dice.parquet")
    tex = TS.parse_indood_tex(Path(a.tex))
    settings = list(dict.fromkeys(R.setting))
    rows = []
    for s in settings:
        lab = label_of(s)
        r = dict(setting=s, label=lab, task=lab.split()[0], p14=lab in P14)
        ro = R[R.setting == s]
        r["n_cases_eval_fold0"] = int(ro[ro.fold == 0].n_cases_eval.iloc[0])
        r["pred_folds"] = int((ro.n_eval_cases_with_pred == ro.n_cases_eval).sum())
        r["pred_complete"] = bool(r["pred_folds"] == 3 and (ro.n_cases_eval > 0).all())
        r["missing_case_folds"] = int((ro.n_cases_eval - ro.n_eval_cases_with_pred).sum())
        pub = ((tex.get(lab) or {}).get("IND") or {})
        noem, em = pub.get("synthseg_noEM"), pub.get("synthseg_EM")
        r["paper_ind_noEM"] = noem[0] if noem else np.nan
        r["paper_ind_EM"] = em[0] if em else np.nan
        r["R28"] = r["paper_ind_noEM"] / r["paper_ind_EM"] if noem and em else np.nan
        cf = CF[CF.setting == s]
        r["cf14"] = float(cf.cf14.median()) if len(cf) else np.nan
        r["n_cf14_cases"] = int(cf.cf14.notna().sum()) if len(cf) else 0
        st = ST[ST.setting == s]
        r["stored_own_item_ind"] = fold_value(st, "stored_dice") if len(st) else np.nan
        lr = L[L.setting == s].merge(st, on=["setting", "case", "fold", "label"], how="inner")
        if len(lr):
            r["recomputed_own_item_ind"] = fold_value(lr, "dice")
            diff = (lr.dice - lr.stored_dice).abs()
            r["qc_label_rows"] = len(lr)
            r["qc_label_mismatch_1e-4"] = int(((diff > 1e-4) | (lr.dice.isna() != lr.stored_dice.isna())).sum())
            r["qc_label_max_abs_diff"] = float(diff.max())
        else:
            r["recomputed_own_item_ind"] = np.nan
        own = lab not in MULTI_COHORT_IND
        # registered gate: recomputed in-domain Dice vs the paper's IND value (0.5). Where the paper's IND row pools other
        # cohorts (MULTI_COHORT_IND) the own column is compared instead, read from the stored eval_all.csv of the run.
        if len(lr):
            ref = r["paper_ind_noEM"] if own else r["stored_own_item_ind"]
            r["gate_reference"] = "paper IND row" if own else "stored own column (paper row pools other cohorts)"
            r["gate_diff"] = r["recomputed_own_item_ind"] - ref
            g = bool(abs(r["gate_diff"]) <= TOL)
        else:
            r["gate_reference"], r["gate_diff"], g = "no predictions", np.nan, False
        r["paper_minus_stored"] = r["paper_ind_noEM"] - r["stored_own_item_ind"]
        r["gate"] = bool(r["pred_complete"] and g)
        c = C[C.setting == s]
        if len(c) and r["pred_complete"]:
            deficit = 1.0 - c.dice_fg
            emp = c.empty & ~c.gt_empty
            r.update(n_case_folds=len(c), n_empty=int(emp.sum()), n_gt_empty=int(c.gt_empty.sum()),
                     mean_dice_fg=float(c.dice_fg.mean()) * 100, deficit_total=float(deficit.sum()),
                     E=float(deficit[emp].sum() / deficit.sum()) if deficit.sum() > 0 else np.nan)
            k = K[K.setting == s]
            big = k[~k.small]
            r.update(n_fp_comp=len(k), n_fp_comp_small=int(k.small.sum()), fp_ml_big=float(big.ml.sum()),
                     fp_ml_small=float(k[k.small].ml.sum()), gt_ml_total=float(c.gt_ml.sum()),
                     median_fp_dist_big=float(big.min_dist_mm.median()) if len(big) else np.nan,
                     F=float(big[big.min_dist_mm > 20].n_vox.sum() / big.n_vox.sum()) if big.n_vox.sum() > 0 else np.nan,
                     F_all=float(k[k.min_dist_mm > 20].n_vox.sum() / k.n_vox.sum()) if k.n_vox.sum() > 0 else np.nan,
                     case_folds_with_fp=int((c.n_fp_comp > 0).sum()))
            if r["E"] >= 0.5:
                cls = "ME05"
            elif np.isfinite(r["F"]) and r["F"] >= 0.5:
                cls = "M08"
            else:
                cls = "mixed"
            r["class"] = cls if r["gate"] else f"(gate failed) {cls}"
        else:
            r["class"] = "not computable (no predictions)" if not r["pred_complete"] else "no rows"
        rows.append(r)
    T = pd.DataFrame(rows)
    # Spine: in the registered scope (21 settings) but absent from the snapshot and from Vulcan
    for lab in ("Spine CT", "Spine Dixon in-phase"):
        pub = ((tex.get(lab) or {}).get("IND") or {})
        noem, em = pub.get("synthseg_noEM"), pub.get("synthseg_EM")
        T = pd.concat([T, pd.DataFrame([dict(setting=lab.replace(" ", "_"), label=lab, task="Spine", p14=False,
                                             paper_ind_noEM=noem[0] if noem else np.nan, paper_ind_EM=em[0] if em else np.nan,
                                             R28=(noem[0] / em[0]) if noem and em else np.nan, cf14=np.nan,
                                             cls="", **{"class": "not computable (no data on Vulcan)"})])], ignore_index=True)
    T = T.drop(columns=[c for c in ("cls",) if c in T])
    T.to_csv(D / "per_setting.tsv", sep="\t", index=False)
    # outcome on P14
    P = T[T.p14]
    cnt = P["class"].value_counts().to_dict()
    n_m08, n_me05 = int((P["class"] == "M08").sum()), int((P["class"] == "ME05").sum())
    n_unc = int((~P["class"].isin(["M08", "ME05", "mixed"])).sum())
    if n_m08 >= 9:
        outc = "M08 favoured"
    elif n_me05 >= 9:
        outc = "ME05 favoured"
    elif n_m08 + n_unc < 9 and n_me05 + n_unc < 9:
        outc = "mixed (neither can reach 9/13 even if every uncomputed member went its way)"
    else:
        outc = "undetermined (uncomputed members could still decide it)"
    # prediction 2
    Q = T[np.isfinite(T.cf14) & np.isfinite(T.R28)]
    r_all = rho(Q.R28, Q.cf14)
    loto = {t: rho(Q[Q.task != t].R28, Q[Q.task != t].cf14) for t in sorted(Q.task.unique())}
    sign_kept = all(np.sign(v) == np.sign(r_all) for v in loto.values() if np.isfinite(v))
    pred2 = dict(n_settings=len(Q), n_tasks=Q.task.nunique(), rho=r_all, loto=loto,
                 loto_range=[min(loto.values()), max(loto.values())] if loto else None, sign_kept=sign_kept,
                 passes=bool(r_all >= 0.5 and sign_kept),
                 note="CF14 vs R28; registered 21 settings, computed on those with CF14")
    res = dict(p14_counts=cnt, n_m08=n_m08, n_me05=n_me05, n_uncomputable=n_unc, outcome=outc, prediction2=pred2,
               gate_all=bool(T[T.pred_complete == True].gate.all()))  # noqa: E712
    (D / "rules.json").write_text(json.dumps(res, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    cols = ["label", "p14", "pred_complete", "n_cases_eval_fold0", "paper_ind_noEM", "stored_own_item_ind", "recomputed_own_item_ind",
            "qc_label_mismatch_1e-4", "gate_diff", "gate", "R28", "cf14", "n_case_folds", "n_empty", "mean_dice_fg", "E", "F", "F_all",
            "n_fp_comp", "n_fp_comp_small", "median_fp_dist_big", "class"]
    print(T[[c for c in cols if c in T]].round(3).to_string())
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
