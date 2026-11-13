#!/usr/bin/env python3
"""T04 analysis (registration 2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710..., test T04).

Reads DIR/rows/<setting>.parquet and DIR/qc/<setting>.parquet (t04_signed.py). Aggregation = E1's (e1_analyze.agg,
GROUP_REMAP, POOL_RESTRICT imported, not copied): per (case, label, fold) value -> mean over labels per case -> mean over
cases per out-of-domain group -> mean over groups = one value per fold. Per setting and arm: mean over the three folds,
SD over folds (ddof = 1), z = mean / (SD / sqrt 3), read against t(4) = 2.776 (sensitivity t(2) = 4.303).

Gate: the e1 component set reproduces e1_arm_values.tsv (flat, noise, texture ASD within 0.01 mm).
Rules (per setting, primary = per-arm matching; secondary = intersection set where its coverage >= 20%):
  M16/M17/M19: every scored arm (7) has the same ASD sign AND >= 4 of the 5 core arms have |z| >= 2.776.
  M14: E1's texture - flat ASD > 0 (P07 sign) with z >= 2.776 (e1_per_setting.tsv, E1's fold null) AND the M16 rule fails.
  M28 (interface settings): d = mean ASD of {SynthSeg-EM, AugLab, Palette+AugLab} - mean of {Plain, Palette}; registered:
      same sign in >= 7/10 settings and median |d| >= 0.1 mm. With fewer than 10 settings available the outcome is reported as
      PASS / FAIL only when the missing settings cannot change it, else UNDETERMINED.

  t04_analyze.py --dir DIR [--smoke]
Writes DIR/arm_asd.tsv, per_setting.tsv, rules.tsv, gate.tsv, qc_dice.tsv, coverage.tsv, tables.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t04_arms as TA  # noqa: E402  (puts E1 on sys.path)
import e1_analyze as EA  # noqa: E402

T4, T2 = 2.776, 4.303
TOL_GATE = 0.01
TOL_DICE = 1e-4
N_INTERFACE_REG = 10


def md(df, index=True):
    """pipe table without the tabulate dependency."""
    d = df.reset_index() if index else df
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def prep(R: pd.DataFrame, s: str) -> pd.DataFrame:
    rm = EA.GROUP_REMAP.get(s, {})
    R = R.copy()
    R["group"] = R["group"].map(lambda g: rm.get(g, g))
    if s in EA.POOL_RESTRICT:
        R = R[R.item_key.isin(EA.POOL_RESTRICT[s])]
    return R[R.group != "in-domain"]


def fold_values(sub: pd.DataFrame, arm: str, q: str = "asd") -> list:
    A = sub[sub.arm == arm]
    return [EA.agg(A[A.fold == k].rename(columns={q: "v"}), "v") for k in (0, 1, 2)]


def coverage(sub: pd.DataFrame, arm: str, fold=None):
    """GT-volume share of matched components (fold-specific for the per_fold set; the other sets are fold-independent)."""
    A = sub[sub.arm == arm]
    if fold is not None:
        A = A[A.fold == fold]
    lab = A.drop_duplicates(["group", "case", "label"])
    tot = lab.gt_ml_total.sum()
    return (float(lab.gt_ml_set.sum() / tot) if tot > 0 else np.nan), int(lab.n_comp_set.sum()), int(lab.n_comp_total.sum())


def stats(v):
    v = np.asarray(v, float)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    return m, sd, (m / (sd / np.sqrt(3)) if sd > 0 else np.nan)


def m28_outcome(d: dict):
    """d: {setting: diff} over the available interface settings. Registered over 10."""
    vals = np.array(list(d.values()), float)
    n = len(vals); miss = N_INTERFACE_REG - n
    pos, neg = int((vals > 0).sum()), int((vals < 0).sum())
    best = max(pos, neg)
    sign_ok = "PASS" if best >= 7 else ("FAIL" if best + miss < 7 else "UNDETERMINED")
    med = float(np.median(np.abs(vals))) if n else np.nan
    lo = float(np.median(np.r_[np.abs(vals), np.zeros(miss)]))       # missing settings at |d| = 0
    hi = float(np.median(np.r_[np.abs(vals), np.full(miss, np.inf)]))  # missing settings at |d| = inf
    med_ok = "PASS" if lo >= 0.1 else ("FAIL" if hi < 0.1 else "UNDETERMINED")
    overall = "PASS" if sign_ok == med_ok == "PASS" else ("FAIL" if "FAIL" in (sign_ok, med_ok) else "UNDETERMINED")
    return dict(n_available=n, n_registered=N_INTERFACE_REG, n_pos=pos, n_neg=neg, sign_rule=sign_ok,
                median_abs_d_available=med, median_abs_d_bounds=f"[{lo:.3f}, {hi:.3f}]", median_rule=med_ok, outcome=overall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    D = Path(a.dir)
    S = TA.settings()
    e1v = pd.read_csv(TA.E1 / "e1_arm_values.tsv", sep="\t").set_index(["setting", "arm"])
    e1s = pd.read_csv(TA.E1 / "e1_per_setting.tsv", sep="\t")
    e1tf = e1s[e1s.comparison == "texture_minus_flat"].set_index("setting")
    sfx = ".smoke.parquet" if a.smoke else ".parquet"
    files = sorted(f for f in (D / "rows").glob(f"*{sfx}") if a.smoke or not f.name.endswith(".smoke.parquet"))
    arm_rows, per, gate, qcs, cov_rows, rules = [], [], [], [], [], []
    for f in files:
        s = f.name[: -len(sfx)]
        st = S[s]
        R = prep(pd.read_parquet(f), s)
        Q = pd.read_parquet(D / "qc" / f.name)
        Q = prep(Q, s)
        # Dice QC vs the stored evaluation
        for col in ("stored_dice", "stored_csv"):
            if col not in Q:
                Q[col] = np.nan
        for arm, g in Q.groupby("arm"):
            g = g[g.stored_csv.notna()]
            diff = (g.dice - g.stored_dice).abs()
            bad = (diff > TOL_DICE) | (g.dice.isna() != g.stored_dice.isna())
            qcs.append(dict(setting=s, arm=arm, n_pred_rows=int((Q.arm == arm).sum()), n_with_stored=len(g),
                            n_mismatch=int(bad.sum()), max_abs_diff=float(diff.max()) if diff.notna().any() else np.nan,
                            stored_csvs=g.stored_csv.nunique()))
        for set_name in ("e1", "per_arm", "inter", "per_fold"):
            sub = R[R.set == set_name]
            if not len(sub):
                continue
            for arm in sorted(sub.arm.unique()):
                cf = [coverage(sub, arm, k) for k in (0, 1, 2)]
                c, ncs, nct = float(np.mean([x[0] for x in cf])), int(round(np.mean([x[1] for x in cf]))), cf[0][2]
                fv = fold_values(sub, arm)
                rv = fold_values(sub, arm, "rvd")
                mv = fold_values(sub, arm, "mean_abs")
                for k in (0, 1, 2):
                    arm_rows.append(dict(setting=s, set=set_name, arm=arm, fold=k, asd=fv[k], rvd=rv[k], mean_abs=mv[k],
                                         coverage=cf[k][0], n_comp_matched=cf[k][1], n_comp_total=nct))
                m, sd, z = stats(fv)
                per.append(dict(setting=s, task=st["task"], group_type=st["group"], set=set_name, arm=arm,
                                display=TA.DISPLAY.get(arm, arm), role="scored" if arm in TA.SCORED else "reported",
                                mean_asd=m, sd_folds=sd, z=z, sig_t4=bool(abs(z) >= T4) if np.isfinite(z) else False,
                                sig_t2=bool(abs(z) >= T2) if np.isfinite(z) else False, rvd=float(np.mean(rv)),
                                mean_abs=float(np.mean(mv)), coverage=c, n_comp_matched=ncs, n_comp_total=nct,
                                folds=";".join(f"{v:+.3f}" for v in fv)))
                if set_name == "e1" and arm in TA.FILL and (s, arm) in e1v.index:
                    ref = e1v.loc[(s, arm)]
                    gate.append(dict(setting=s, arm=arm, t04_asd=m, e1_asd=float(ref.asd), diff=m - float(ref.asd),
                                     t04_rvd=float(np.mean(rv)), e1_rvd=float(ref.rvd),
                                     t04_mean_abs=float(np.mean(mv)), e1_mean_abs=float(ref.mean_abs),
                                     passed=bool(abs(m - float(ref.asd)) <= TOL_GATE)))
            if set_name == "e1":
                c, _, _ = coverage(sub, "texture")
                cov_rows.append(dict(setting=s, set="e1", coverage=c))
            if set_name == "inter":
                c, _, _ = coverage(sub, "texture")
                cov_rows.append(dict(setting=s, set="inter", coverage=c))
    P = pd.DataFrame(per)
    G = pd.DataFrame(gate)
    QCd = pd.DataFrame(qcs)
    qc_bad = {}
    if len(QCd):
        for s_, g_ in QCd[QCd.arm.isin(TA.SCORED)].groupby("setting"):
            qc_bad[s_] = int(g_.n_mismatch.sum()) + int((g_.n_pred_rows - g_.n_with_stored).sum())
    gate_ok = {s: bool(g.passed.all()) and len(g) == 3 for s, g in G.groupby("setting")} if len(G) else {}
    # rules
    m28 = {"per_arm": {}, "inter": {}, "per_fold": {}}
    for (s, set_name), g in P[P.set.isin(["per_arm", "inter", "per_fold"])].groupby(["setting", "set"]):
        g = g.set_index("arm")
        st = S[s]
        row = dict(setting=s, task=st["task"], group_type=st["group"], set=set_name, gate=gate_ok.get(s, False))
        present = [x for x in TA.SCORED if x in g.index]
        defined = [x for x in present if np.isfinite(g.loc[x, "mean_asd"])]
        reasons = []
        if not row["gate"]:
            reasons.append("gate failed")
        if len(present) < len(TA.SCORED):
            reasons.append("predictions absent: " + ",".join(x for x in TA.SCORED if x not in present))
        if qc_bad.get(s, 0):
            reasons.append(f"Dice QC: {qc_bad[s]} prediction rows of scored arms not matching / not found in the stored eval_all.csv")
        if len(defined) < len(present):
            reasons.append("no matched GT component in any case (ASD undefined): " + ",".join(x for x in present if x not in defined))
        if set_name == "inter":
            cov = float(g.loc["texture", "coverage"]) if "texture" in g.index else np.nan
            row["inter_coverage"] = cov
            if not cov >= 0.20:
                reasons.append("intersection coverage < 20%")
        row["scorable"] = not reasons
        row["note"] = "; ".join(reasons)
        tf_asd = float(e1tf.loc[s, "asd"]) if s in e1tf.index else np.nan
        tf_z = float(e1tf.loc[s, "asd_z"]) if s in e1tf.index else np.nan
        row.update(e1_tex_minus_flat=tf_asd, e1_tex_minus_flat_z=tf_z)
        if defined:
            means = g.loc[defined, "mean_asd"]
            core = [x for x in TA.CORE5 if x in defined]
            zs = g.loc[core, "z"]
            same = bool((means > 0).all() or (means < 0).all())
            row.update(n_arms_defined=len(defined), sign_all=("+" if (means > 0).all() else "-" if (means < 0).all() else "mixed"),
                       n_pos=int((means > 0).sum()), n_neg=int((means < 0).sum()),
                       n_core_sig_t4=int((zs.abs() >= T4).sum()), n_core_sig_t2=int((zs.abs() >= T2).sum()))
            m16, m16b = bool(same and row["n_core_sig_t4"] >= 4), bool(same and row["n_core_sig_t2"] >= 4)
            # registered outcome only when scorable; the same rule on the defined arms is kept as an unscored sensitivity
            row["M16_on_defined_arms"] = m16
            row["M16"] = m16 if row["scorable"] else None
            row["M16_t2"] = m16b if row["scorable"] else None
            row["M14"] = bool(tf_asd > 0 and tf_z >= T4 and not m16) if row["scorable"] else None
            row["M14_t2"] = bool(tf_asd > 0 and tf_z >= T2 and not m16b) if row["scorable"] else None
            if all(x in defined for x in TA.BLUR + TA.UNBLUR):
                d = float(g.loc[list(TA.BLUR), "mean_asd"].mean() - g.loc[list(TA.UNBLUR), "mean_asd"].mean())
                row["M28_d_blur_minus_unblur"] = d
                if st["group"] == "interface" and row["scorable"]:
                    m28[set_name][s] = d
        rules.append(row)
    Rl = pd.DataFrame(rules)
    M28 = pd.DataFrame([dict(set=k, **m28_outcome(v)) for k, v in m28.items() if v])
    # outputs
    pre = "smoke_" if a.smoke else ""
    pd.DataFrame(arm_rows).to_csv(D / f"{pre}arm_asd.tsv", sep="\t", index=False)
    rcols = [c for c in ("M16", "M16_t2", "M16_on_defined_arms", "M14", "M14_t2", "M28_d_blur_minus_unblur", "sign_all", "n_core_sig_t4", "scorable", "note")
             if c in Rl.columns]
    PS = P.merge(Rl[["setting", "set"] + rcols], on=["setting", "set"], how="left") if len(Rl) else P
    PS.to_csv(D / f"{pre}per_setting.tsv", sep="\t", index=False)
    Rl.to_csv(D / f"{pre}rules.tsv", sep="\t", index=False)
    G.to_csv(D / f"{pre}gate.tsv", sep="\t", index=False)
    pd.DataFrame(qcs).to_csv(D / f"{pre}qc_dice.tsv", sep="\t", index=False)
    pd.DataFrame(cov_rows).to_csv(D / f"{pre}coverage.tsv", sep="\t", index=False)
    M28.to_csv(D / f"{pre}m28.tsv", sep="\t", index=False)
    L = ["# T04 tables (generated by t04_analyze.py)", "", "## Gate (e1 set vs e1_arm_values.tsv, tolerance 0.01 mm)", "",
         md(G.round(4), index=False) if len(G) else "(none)", "",
         "## Dice QC (prediction file vs stored eval_all.csv, |diff| > 1e-4 counted)", "",
         md(pd.DataFrame(qcs).round(6), index=False), "",
         "## Per-arm ASD (mm, + = prediction inside GT), primary per-arm matching", ""]
    if len(P):
        W = P[P.set == "per_arm"].pivot(index="setting", columns="arm", values="mean_asd")
        Z = P[P.set == "per_arm"].pivot(index="setting", columns="arm", values="z")
        C = P[P.set == "per_arm"].pivot(index="setting", columns="arm", values="coverage")
        cols = [c for c in TA.SCORED + TA.REPORTED if c in W.columns]
        L += ["Mean over folds:", "", md(W[cols].round(3)), "", "z (t(4) threshold 2.776):", "",
              md(Z[cols].round(2)), "", "Coverage (share of out-of-domain GT volume in matched components):", "",
              md(C[cols].round(3)), ""]
        Wi = P[P.set == "inter"].pivot(index="setting", columns="arm", values="mean_asd")
        if len(Wi):
            L += ["## Secondary: intersection over all scored arms (mean ASD)", "",
                  md(Wi[[c for c in cols if c in Wi.columns]].round(3)), ""]
    L += ["## Rules", "", md(Rl.round(3), index=False) if len(Rl) else "(none)", "", "## M28", "",
          md(M28.round(3), index=False) if len(M28) else "(none)", ""]
    (D / f"{pre}tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not a.smoke:
        write_report(D, S, G, pd.DataFrame(qcs), P, Rl, M28, files)


def write_report(D, S, G, QC, P, Rl, M28, files):
    """REPORT.md: facts only (rule outcomes, counts, gate, QC); the narrative is added by hand below the marker."""
    done = sorted(f.name[: -len(".parquet")] for f in files)
    missing = [s for s in TA.SETTINGS16 if s not in done]
    L = ["# T04 signed boundary shift of every arm vs GT (R18): report", "",
         "Registration: research/dg_mechanisms/plans/2026-11-12-PREDICTION_T01-T05.md (sha256 0b9f8710...), test T04.",
         "Generated by t04_analyze.py; numbers [computed] unless tagged. ASD in mm, + = prediction inside GT (under-segmentation).", "",
         f"Settings computed: {len(done)}/16 ({', '.join(done)}). Not computed: {', '.join(missing) or 'none'}.", "",
         "## Gate (E1 component set, flat/noise/texture vs e1_arm_values.tsv, tolerance 0.01 mm)", ""]
    if len(G):
        L += [f"{int(G.passed.sum())}/{len(G)} arm values within 0.01 mm; max |diff| {G['diff'].abs().max():.2e} mm.", "",
              md(G[["setting", "arm", "t04_asd", "e1_asd", "diff", "passed"]].round(5), index=False), ""]
    if len(QC):
        L += ["## Prediction-file check (whole-label Dice vs stored eval_all.csv, tolerance 1e-4)", "",
              f"{int(QC.n_mismatch.sum())} mismatches over {int(QC.n_with_stored.sum())} (case, label, arm, fold) rows with a stored value;"
              f" rows without a stored value: {int((QC.n_pred_rows - QC.n_with_stored).sum())}.", "",
              md(QC.groupby("setting")[["n_pred_rows", "n_with_stored", "n_mismatch"]].sum()), ""]
    for set_name, title in (("per_arm", "Primary: per-arm matching (components detected by all three folds of the arm)"),
                            ("inter", "Secondary: intersection over all scored arms"),
                            ("per_fold", "Sensitivity: per-prediction matching (each fold on the components it detects itself)")):
        R = Rl[Rl.set == set_name] if len(Rl) else Rl
        if not len(R):
            continue
        sc = R[R.scorable == True]  # noqa: E712
        L += [f"## {title}", ""]
        if set_name == "inter":
            L += [f"Scored where intersection coverage >= 20%: {len(sc)}/{len(R)} settings.", ""]
        cols = [c for c in ("setting", "group_type", "scorable", "n_arms_defined", "sign_all", "n_pos", "n_neg", "n_core_sig_t4", "n_core_sig_t2",
                            "M16", "M16_t2", "M16_on_defined_arms", "e1_tex_minus_flat", "e1_tex_minus_flat_z", "M14", "M14_t2",
                            "M28_d_blur_minus_unblur", "inter_coverage", "note") if c in R.columns]
        L += [md(R[cols].round(3), index=False), ""]
        if len(sc):
            for gt in ("interface", "appearance"):
                g = sc[sc.group_type == gt].copy()
                for c in ("M16", "M14", "M16_t2", "M14_t2"):
                    g[c] = g[c].fillna(False).astype(bool)
                L += [f"- {gt} settings scored: {len(g)}; M16/M17/M19 favoured (rule passes): {int(g.M16.sum())}/{len(g)}"
                      f" ({', '.join(g[g.M16].setting) or '-'}); M14 favoured: {int(g.M14.sum())}/{len(g)} ({', '.join(g[g.M14].setting) or '-'})."
                      f" Sensitivity t(2): M16 {int(g.M16_t2.sum())}, M14 {int(g.M14_t2.sum())}."]
            L += [""]
        m = M28[M28.set == set_name] if len(M28) else M28
        if len(m):
            r = m.iloc[0]
            L += [f"M28 ({title.split(':')[0].lower()}): {r.n_available}/{r.n_registered} interface settings available; d > 0 in {r.n_pos},"
                  f" d < 0 in {r.n_neg}; sign rule {r.sign_rule}; median |d| over available {r.median_abs_d_available:.3f} mm"
                  f" (bounds with the missing settings {r.median_abs_d_bounds}); median rule {r.median_rule}; outcome {r.outcome}.", ""]
    W = P[P.set == "per_arm"].pivot(index="setting", columns="arm", values="mean_asd") if len(P) else None
    if W is not None and len(W):
        Z = P[P.set == "per_arm"].pivot(index="setting", columns="arm", values="z")
        C = P[P.set == "per_arm"].pivot(index="setting", columns="arm", values="coverage")
        cols = [c for c in TA.SCORED + TA.REPORTED if c in W.columns]
        L += ["## Per-arm values (primary matching)", "", "Mean ASD over folds (mm):", "", md(W[cols].round(3)), "",
              "z = mean / (fold SD / sqrt 3):", "", md(Z[cols].round(2)), "", "Coverage:", "", md(C[cols].round(3)), ""]
    L += ["## Files", "", f"All in {D}: arm_asd.tsv, per_setting.tsv, rules.tsv, m28.tsv, gate.tsv, qc_dice.tsv, coverage.tsv,"
          " tables.md, rows/, qc/, prov/, roster.tsv, PROVENANCE.txt.", "", "<!-- NARRATIVE BELOW (hand-written, kept in NARRATIVE.md) -->", ""]
    nar = D / "NARRATIVE.md"
    if nar.exists():
        L += [nar.read_text()]
    (D / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
