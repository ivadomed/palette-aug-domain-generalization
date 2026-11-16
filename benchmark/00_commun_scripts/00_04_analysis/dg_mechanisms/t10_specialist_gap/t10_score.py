#!/usr/bin/env python3
"""T10 step 2 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T10_T28.md, section T10, sha256 e761e43e...):
gates, then the registered rules, on t10_cases.py's outputs. No case ids in the outputs written here.

Gates (registered unless marked):
  G1 specialist inputs: re-predicted (in-domain stage) -> the acceptance row of repredict_indomain/qc_indomain.tsv must pass;
     first-pass files (repo tree) -> every recomputed (case, label, fold) Dice within 1e-4 of the stored eval_all.csv value.
  G2 per cell, the four voxel classes sum back to FN + FP of each model (exact, every row).
  G3 Dice recomputed from the voxels vs the stored value, 1e-3. [declared reading] first-pass (repo-tree) files: every row;
     staged re-predictions (T04 dev12 / in-domain stage, already accepted under their own registered rule): median and
     |mean signed| within 1e-3, rows above 1e-3 counted and reported. NaN == NaN is a match; stored finite / new NaN = 1.
  G4 [brief, not in the registration] the cell Dice of arm and specialist, computed from the recomputed rows with
     aggregate_from_config.cross_fold_class_mean (the table's estimand), equals T02 cells.tsv dice_<arm> / dice_spec within 0.1.
  G5 [impl] GT of every model's item equal to the reference GT; the cells' test images identical (exact array equality).
Rules (registered): excess per class = arm - specialist voxel count summed over the cell's common cases, labels and folds; shares
over positive excess. M19: band share >= 0.5 AND excess band error per weak-edge point >= 1.5 x that per strong-edge point.
M03: missed share >= 0.5 AND median CF03 of the arm's missed objects < median CF03 of its detected objects. Else mixed.
[declared] ratio: weak > 0 and strong <= 0 -> satisfied (infinite); weak <= 0 -> not satisfied. No positive excess -> mixed
(undefined shares). Setting = majority (> half) of its cells; P15: >= 3/5 settings M19 -> M19 favoured; >= 3/5 M03 -> M03; else mixed.
ME04 (descriptive): GT-component size tertiles pooled over the cell's components (<= q1/3, > q2/3, else middle); missed rate over
(component, fold) instances, arm vs specialist; "not where the gap is" if every tertile within 5 points and arm Dice < specialist.
Region view (Glioma, descriptive): band excess per label next to the label's defining contrast (ET: T1-CE; SNFH: FLAIR / T2w).
Usage: .venv/bin/python t10_score.py --dir OUT --scratch DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import aggregate_from_config as AGG  # noqa: E402

QC_IND = REPO / "benchmark/01_commun_results/dg_mechanisms/repredict_indomain/qc_indomain.tsv"
CLASSES = ("missed", "band", "far_fp", "interior")
DEFINING = {"ET": ("t1c",), "SNFH": ("t2f", "t2w")}
FIRST_PASS_HOW = ("glob", "ref", "origts", "standard", "ref_flat_runid_subst")


def dice_diff(r):
    s, n = r["stored_dice"], r["dice"]
    if not np.isfinite(s) and not np.isfinite(n):
        return 0.0
    if np.isfinite(s) and not np.isfinite(n):
        return 1.0
    if not np.isfinite(s) and np.isfinite(n):
        return np.nan          # stored NaN / new finite: counted, not scored (clarification 1 of the acceptance rule)
    return n - s


def cell_dice(rows: pd.DataFrame, col="dice") -> float:
    data = {"dice": {"c": defaultdict(lambda: defaultdict(list))}}
    for r in rows.itertuples():
        data["dice"]["c"][r.label][f"fold{r.fold}"].append(float(getattr(r, col)))
    return AGG.cross_fold_class_mean(data, "dice", "c") * 100


def md_table(df: pd.DataFrame, fmt=".4g") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(format(v, fmt) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def settings_outcome(CELL):
    S = []
    for st, g in CELL.groupby("setting", sort=False):
        n = len(g)
        cnt = g.outcome.value_counts().to_dict()
        out = "M19" if cnt.get("M19", 0) > n / 2 else "M03" if cnt.get("M03", 0) > n / 2 else "mixed"
        S.append(dict(setting=st, n_cells=n, n_M19=cnt.get("M19", 0), n_M03=cnt.get("M03", 0), n_mixed=cnt.get("mixed", 0), outcome=out))
    SET = pd.DataFrame(S)
    n19, n03 = int((SET.outcome == "M19").sum()), int((SET.outcome == "M03").sum())
    return SET, ("M19 favoured" if n19 >= 3 else "M03 favoured" if n03 >= 3 else "mixed"), n19, n03


def cell_metrics(c, cell, ncases, A, B, P, Ca):
    """The registered per-cell rule on the given rows (primary run and the declared G3-clean sensitivity)."""
    sa = {k: int(A[k].sum()) for k in CLASSES}
    sb = {k: int(B[k].sum()) for k in CLASSES}
    exc = {k: sa[k] - sb[k] for k in CLASSES}
    pos = sum(max(v, 0) for v in exc.values())
    share = {k: (max(exc[k], 0) / pos if pos > 0 else np.nan) for k in CLASSES}
    n_weak, n_strong = float(P.n_weak.sum()), float(P.n_strong.sum())
    ew = (A.band_weak.sum() - B.band_weak.sum()) / n_weak if n_weak else np.nan
    es = (A.band_strong.sum() - B.band_strong.sum()) / n_strong if n_strong else np.nan
    if not np.isfinite(ew) or ew <= 0:
        ratio, r_ok = (ew / es if np.isfinite(es) and es > 0 else np.nan), False
    elif not np.isfinite(es) or es <= 0:
        ratio, r_ok = np.inf, True
    else:
        ratio = ew / es
        r_ok = bool(ratio >= 1.5)
    mis = Ca.missed.astype(bool)
    cm, cd = Ca.loc[mis, "cnr"].dropna(), Ca.loc[~mis, "cnr"].dropna()
    med_m, med_d = (float(cm.median()) if len(cm) else np.nan), (float(cd.median()) if len(cd) else np.nan)
    cnr_ok = bool(np.isfinite(med_m) and np.isfinite(med_d) and med_m < med_d)
    m19 = bool(pos > 0 and share["band"] >= 0.5 and r_ok)
    m03 = bool(pos > 0 and share["missed"] >= 0.5 and cnr_ok)
    outcome = "M19" if m19 and not m03 else "M03" if m03 and not m19 else "mixed"
    da, db = cell_dice(A), cell_dice(B)
    return dict(setting=c.setting, cell=cell, arm=c.arm, spec=f"{c.spec_setting}/{c.spec_arm_key}", n_cases=ncases,
                dice_arm=da, dice_spec=db, gap=db - da, **{f"arm_{k}": sa[k] for k in CLASSES}, **{f"spec_{k}": sb[k] for k in CLASSES},
                **{f"exc_{k}": exc[k] for k in CLASSES}, **{f"share_{k}": share[k] for k in CLASSES},
                exc_per_weak_pt=ew, exc_per_strong_pt=es, weak_strong_ratio=ratio, ratio_ok=r_ok,
                cnr_missed_med=med_m, cnr_detected_med=med_d, n_missed_inst=int(mis.sum()), n_detected_inst=int((~mis).sum()),
                cnr_ok=cnr_ok, outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--spec-qc", default="", help="qc_indomain.py output (qc_indomain.tsv) for the first-pass specialists")
    a = ap.parse_args()
    D = Path(a.dir)
    V = pd.read_parquet(D / "vox.parquet")
    Pt = pd.read_parquet(D / "points.parquet")
    Cp = pd.read_parquet(D / "comps.parquet")
    K = pd.read_parquet(D / "checks.parquet")
    R = pd.read_csv(D / "roster.tsv", sep="\t")
    C = pd.read_csv(D / "cells_scope.tsv", sep="\t", dtype=str)
    CS = json.load(open(Path(a.scratch) / "case_sets.json"))
    QC = pd.read_csv(QC_IND, sep="\t")
    V["band"] = V.band_fn + V.band_fp
    V["interior"] = V.interior_fn + V.interior_fp
    V["diff"] = V.apply(dice_diff, axis=1)
    how = R.groupby("model").how.agg(lambda x: ",".join(sorted(set(x))))
    lines = ["# T10 scores (t10_score.py; numbers [computed])", ""]

    # ── G5 / inputs ──────────────────────────────────────────────────────────
    g5 = dict(gt_equal_all=bool(K.gt_equal_all.all()), test_images_equal=bool(K.test_images_equal.all()),
              nib_transpose_ok=bool(K.nib_transpose_ok.all()), n_resampled=int(K.resampled.sum()),
              n_cnr_error=int(K.get("cnr_error", pd.Series(dtype=str)).notna().sum()), n_absent_dirs=int((R.pred_dir == "ABSENT").sum()))
    lines += ["## G5 inputs", "", json.dumps(g5), ""]

    # ── G1 specialist inputs ────────────────────────────────────────────────
    g1 = []
    for mid in sorted(V[V.model.str.startswith("spec:")].model.unique()):
        _, s, arm = mid.split(":")
        h = how[mid]
        Vm = V[(V.model == mid) & V.stored_present]
        d = Vm["diff"].abs()
        if "staged_indomain" in h:
            q = QC[(QC.setting == s) & (QC.arm == arm)]
            ok = bool(len(q) == 1 and bool(q.passed.iloc[0]))
            rule = "repredict-indomain acceptance (qc_indomain.tsv)"
        else:
            exact = bool(d.max() <= 1e-4) if len(d) else False
            acc = None
            if a.spec_qc and Path(a.spec_qc).exists():
                q2 = pd.read_csv(a.spec_qc, sep="\t")
                q2 = q2[(q2.setting == s) & (q2.arm == arm)]
                acc = bool(len(q2) == 1 and bool(q2.passed.iloc[0]))
            ok = exact or bool(acc)
            rule = f"first-pass: all rows within 1e-4 ({exact}) OR acceptance rule on these files ({acc})"
        g1.append(dict(model=mid, how=h, rule=rule, rows=len(Vm), max_abs=float(d.max()), n_over_1e4=int((d > 1e-4).sum()),
                       n_stored_nan_new_finite=int(Vm["diff"].isna().sum()), passed=ok))
    G1 = pd.DataFrame(g1)
    lines += ["## G1 specialist inputs", "", md_table(G1), ""]

    # ── per cell ────────────────────────────────────────────────────────────
    cell_rows, sens_rows, g2g3g4, me04, region, pts_rows = [], [], [], [], [], []
    for c in C.itertuples():
        tag = f"{c.dataset}:{c.item}"
        am, sm = f"arm:{c.s_key}:{c.arm}", f"spec:{c.spec_key}:{c.spec_arm_key}"
        common = sorted(set(CS[am + "|" + tag]) & set(CS[sm + "|" + tag]))
        cell = f"{c.setting} -> {c.item}"
        A = V[(V.item == tag) & (V.model == am) & V.case.isin(common)]
        B = V[(V.item == tag) & (V.model == sm) & V.case.isin(common)]
        # gates
        for role, X, mid in (("arm", A, am), ("spec", B, sm)):
            d = X.loc[X.stored_present, "diff"]
            fp = all(h in FIRST_PASS_HOW for h in how[mid].split(","))
            ad = d.abs()
            g3 = bool(ad.max() <= 1e-3) if fp else bool(ad.median() <= 1e-3 and abs(d.mean()) <= 1e-3)
            g2g3g4.append(dict(cell=cell, role=role, model=mid, how=how[mid], n_rows=len(X), n_folds=X.fold.nunique(),
                               n_cases=X.case.nunique(), G2_sum_exact=bool(X.sum_ok.all()),
                               G3_median=float(ad.median()), G3_mean_signed=float(d.mean()), G3_max=float(ad.max()),
                               G3_n_over_1e3=int((ad > 1e-3).sum()), G3_reading="first-pass: every row" if fp else "staged: median+mean",
                               G3_pass=g3, dice_recomputed=cell_dice(X), dice_stored=cell_dice(X[X.stored_present], "stored_dice"),
                               dice_cells_tsv=float(getattr(c, f"dice_{c.best_randomised_arm}") if role == "arm" else c.dice_spec)))
        P = Pt[(Pt.item == tag) & Pt.case.isin(common)]
        T = Cp[(Cp.item == tag) & (Cp.model == "") & Cp.case.isin(common)][["case", "label", "comp", "cnr", "size_vox"]]
        Ca = Cp[(Cp.item == tag) & (Cp.model == am) & Cp.case.isin(common)].drop(columns=["cnr"]).merge(
            T[["case", "label", "comp", "cnr"]], on=["case", "label", "comp"])
        Cb = Cp[(Cp.item == tag) & (Cp.model == sm) & Cp.case.isin(common)]
        cell_rows.append(cell_metrics(c, cell, len(common), A, B, P, Ca))
        n_weak, n_strong = float(P.n_weak.sum()), float(P.n_strong.sum())
        da, db = cell_rows[-1]["dice_arm"], cell_rows[-1]["dice_spec"]
        # [declared] sensitivity: drop every (case, label) with a row whose recomputed Dice differs from the stored one by > 1e-3
        X = pd.concat([A, B])
        bset = set(map(tuple, X[(X["diff"].abs() > 1e-3) & X.stored_present][["case", "label"]].drop_duplicates().values.tolist()))
        kp = lambda Z: Z[[(cc, ll) not in bset for cc, ll in zip(Z.case, Z.label)]]  # noqa: E731
        r2 = cell_metrics(c, cell, len(common), kp(A), kp(B), kp(P), kp(Ca))
        r2["n_case_labels_dropped"] = len(bset)
        sens_rows.append(r2)
        pts_rows.append(dict(cell=cell, n_case_labels=len(P), n_with_gt=int((P.n_gt > 0).sum()), n_sampled=int(P.n_sampled.sum()),
                             n_valid=int(P.n_valid.sum()), n_locatable_T28_p10=int(P.n_locatable.sum()),
                             share_valid=float(P.n_valid.sum() / max(P.n_sampled.sum(), 1)),
                             n_edge_at_window_end=int(P.n_edge_at_window_end.sum()), n_weak=int(n_weak), n_strong=int(n_strong)))
        # ME04
        U = T.drop_duplicates(["case", "label", "comp"])
        if len(U):
            q1, q2 = np.quantile(U.size_vox, [1 / 3, 2 / 3])
            tert = lambda s: np.where(s <= q1, "T1_small", np.where(s > q2, "T3_large", "T2_mid"))  # noqa: E731
            row = dict(cell=cell, n_components=len(U), q1_vox=float(q1), q2_vox=float(q2))
            okall = True
            for nm, X in (("arm", Ca), ("spec", Cb)):
                X = X.assign(t=tert(X.size_vox.values))
                for t in ("T1_small", "T2_mid", "T3_large"):
                    row[f"{nm}_{t}"] = 100 * float(X[X.t == t].missed.astype(bool).mean()) if (X.t == t).any() else np.nan
            for t in ("T1_small", "T2_mid", "T3_large"):
                dd = row[f"arm_{t}"] - row[f"spec_{t}"]
                row[f"diff_{t}"] = dd
                okall &= bool(np.isfinite(dd) and abs(dd) <= 5)
            row["rates_match_5pt"] = okall
            row["gap_persists"] = bool(da < db)
            row["reading"] = "misses not where the gap is" if okall and da < db else "misses differ" if not okall else "no gap"
            me04.append(row)
        # region view
        if c.dataset == "brats2024-glioma":
            for lab in ("NCR", "SNFH", "ET", "RC"):
                xa, xb = A[A.label == lab], B[B.label == lab]
                be = int(xa.band.sum() - xb.band.sum())
                ngt = float(xa[xa.fold == xa.fold.min()].n_gt.sum()) if len(xa) else 0.0
                region.append(dict(cell=cell, test=c.item, label=lab, defined_on=",".join(DEFINING.get(lab, ("-",))),
                                   test_is_defining=c.item in DEFINING.get(lab, ()), band_excess=be,
                                   band_excess_per_1000_gt_vox=1000 * be / ngt / 3 if ngt else np.nan,
                                   exc_missed=int(xa.missed.sum() - xb.missed.sum()),
                                   dice_arm=cell_dice(xa), dice_spec=cell_dice(xb)))
    CELL = pd.DataFrame(cell_rows)
    GT = pd.DataFrame(g2g3g4)
    GT["G4_diff_vs_cells_tsv"] = GT.dice_recomputed - GT.dice_cells_tsv
    GT["G4_pass"] = GT.G4_diff_vs_cells_tsv.abs() <= 0.1
    ME = pd.DataFrame(me04)
    RG = pd.DataFrame(region)
    PT = pd.DataFrame(pts_rows)
    gates_pass = bool(G1.passed.all() and GT.G2_sum_exact.all() and GT.G3_pass.all() and GT.G4_pass.all()
                      and g5["gt_equal_all"] and g5["test_images_equal"] and g5["n_absent_dirs"] == 0)
    lines += ["## G2 / G3 / G4 per cell and model", "", md_table(GT), "", f"ALL GATES PASS: {gates_pass}", ""]

    SET, p15, n19, n03 = settings_outcome(CELL)
    SENS = pd.DataFrame(sens_rows)
    SET2, p15_2, n19_2, n03_2 = settings_outcome(SENS)
    lines += ["## Cells", "", md_table(CELL), "", "## Edge points per cell (T28 operator; never gating)", "", md_table(PT), "",
              "## Settings", "", md_table(SET), "", f"P15 outcome: {p15} (M19 settings {n19}/5, M03 settings {n03}/5)", "",
              "## ME04 (descriptive): missed-object rate per GT-component size tertile, %", "", md_table(ME), "",
              "## Region view (Glioma, descriptive): band excess per label", "", md_table(RG), "",
              "## Sensitivity [declared]: (case, label) pairs with any |Dice - stored| > 1e-3 dropped", "",
              md_table(SENS[["cell", "n_case_labels_dropped", "share_missed", "share_band", "share_far_fp", "share_interior",
                             "weak_strong_ratio", "cnr_missed_med", "cnr_detected_med", "outcome"]]), "", md_table(SET2), "",
              f"P15 outcome (sensitivity): {p15_2} (M19 settings {n19_2}/5, M03 settings {n03_2}/5)", ""]
    (D / "scores.md").write_text("\n".join(lines) + "\n")
    for nm, df in (("cells", CELL), ("gates", GT), ("gate_specialists", G1), ("settings", SET), ("cells_g3clean", SENS), ("settings_g3clean", SET2), ("me04", ME), ("region", RG),
                   ("edge_points", PT)):
        df.to_csv(D / f"{nm}.tsv", sep="\t", index=False)
    json.dump(dict(gates_pass=gates_pass, g5=g5, p15=p15, n_M19_settings=n19, n_M03_settings=n03, p15_g3clean=p15_2), open(D / "rules.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
