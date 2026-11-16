#!/usr/bin/env python3
"""T07 step 2 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T07.md, sha256 771c20db...): gates and
the registered rules. Reads T02/cells.tsv (cells, R05_c, Dorder, Dq; no Dice recomputed), T07/<Task>/cells_axes.tsv and
merge_settings.tsv (t07_axes.py), and the paper's _suppl_ind_ood.tex (setting-level OOD rows, gate for R05 / R08).

Fixed before any axis value was computed:
  * cells = T02 cells with R05_c defined AND T02 image features (Spine has no images on Vulcan: excluded).
  * axes (5): A_res, A_noise, A_int_order (= T02 Dorder, class-macro), A_int_q (= T02 Dq = class-macro |q_t - q_s|),
    A_merge (setting value shared by its cells). Secondary (reported only): A_res_inplane, A_res_train, A_noise_train.
  * Rule 1: pooled Spearman rho(axis, R05_c) >= 0.4; every leave-one-task-out rho > 0; within-task rho > 0 in >= 60% of
    the tasks with >= 3 cells. A within-task rho is undefined when the axis (or R05_c) is constant inside the task (e.g.
    A_merge in a one-setting task); such tasks leave that axis's denominator. The literal count (undefined = not > 0, over
    all tasks with >= 3 cells) is printed next to it.
  * Rule 3: leader = predicting axis with the highest pooled rho; separated from the runner-up predicting axis iff
    partial rho(leader | other) >= 0.3 and partial rho(other | leader) < 0.2 (partial Spearman: ranks, OLS residuals on
    the control's ranks, Pearson of residuals).
  * Rule 4: settings with A_merge (19: Spine has no images); R08 per setting = equal-weight mean of its OOD cells
    (Palette - SynthSeg-EM), gated against the paper rows; rho <= -0.4 and every LOTO rho < 0.
  * Gate tolerance vs the paper: 0.1 Dice (T02's; the paper prints one decimal).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

R = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
T02 = R / "benchmark/01_commun_results/dg_mechanisms/T02"
AXES = ["A_res", "A_noise", "A_int_order", "A_int_q", "A_merge"]
SECONDARY = ["A_res_inplane", "A_res_train", "A_noise_train"]
P03 = {"Glioma FLAIR", "Pelvis CT", "Breast T1-CE", "Glioma T2w"}
ARMS = ["ssnoem", "ssem", "plain", "srcsm", "auglab", "palette", "pa_auglab"]
TOL = 0.1


def read(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fl(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), int(len(x))
    return float(spearmanr(x, y).statistic), int(len(x))


def partial_rho(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(pearsonr(ex, ey).statistic)


def parse_paper(tex: Path) -> dict:
    """{setting: {arm: OOD value}} from the first (per training domain) table of _suppl_ind_ood.tex."""
    out, cur = {}, None
    for line in tex.read_text().splitlines():
        if "tab:suppl-ind-ood-dice-task" in line or line.startswith("Task &"):
            break
        cols = [c.strip() for c in line.replace("\\rowcolor{gray!12}", "").split("&")]
        if len(cols) < 10 or cols[1] not in ("IND", "OOD"):
            continue
        if cols[1] == "IND":
            cur = cols[0]
            continue
        vals = []
        for c in cols[3:10]:
            m = re.search(r"(-?\d+\.\d)", c)
            vals.append(float(m.group(1)) if m else float("nan"))
        out[cur] = dict(zip(ARMS, vals))
    return out


def rule1(cells, axis, P):
    x = [fl(c[axis]) for c in cells]
    y = [fl(c["R05_c"]) for c in cells]
    r, n = rho(x, y)
    tasks = sorted({c["task"] for c in cells})
    loto = {}
    for t in tasks:
        sub = [c for c in cells if c["task"] != t]
        loto[t] = rho([fl(c[axis]) for c in sub], [fl(c["R05_c"]) for c in sub])[0]
    within = {}
    for t in tasks:
        sub = [c for c in cells if c["task"] == t]
        if len(sub) >= 3:
            within[t] = rho([fl(c[axis]) for c in sub], [fl(c["R05_c"]) for c in sub])[0]
    defined = {t: v for t, v in within.items() if np.isfinite(v)}
    npos = sum(v > 0 for v in defined.values())
    frac = npos / len(defined) if defined else float("nan")
    lit = sum((np.isfinite(v) and v > 0) for v in within.values()) / len(within) if within else float("nan")
    c1, c2, c3 = r >= 0.4, all(np.isfinite(v) and v > 0 for v in loto.values()), bool(defined) and frac >= 0.6
    pred = bool(c1 and c2 and c3)
    P(f"\n### {axis}: pooled rho = {r:+.3f} (n = {n}) -> {'>= 0.4' if c1 else '< 0.4'}")
    P("LOTO: " + ", ".join(f"{t} {v:+.2f}" for t, v in loto.items()) + f" -> all > 0: {c2}")
    P("within task (>= 3 cells): " + ", ".join(f"{t} {v:+.2f}" if np.isfinite(v) else f"{t} undefined" for t, v in within.items())
      + f" -> {npos}/{len(defined)} defined > 0 ({frac:.0%}; >= 60%: {c3}); literal {lit:.0%} of {len(within)}")
    P(f"PREDICTS: {pred}")
    return dict(axis=axis, rho=r, n=n, loto_min=min(loto.values()), loto_all_pos=c2, within_pos=npos,
                within_defined=len(defined), within_frac=frac, within_literal=lit, predicts=pred,
                loto=";".join(f"{t}:{v:.3f}" for t, v in loto.items()),
                within=";".join(f"{t}:{v:.3f}" for t, v in within.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="T07 output dir")
    ap.add_argument("--paper", type=Path, required=True, help="copy of paper/cvpr_format_latex/sec/_suppl_ind_ood.tex")
    a = ap.parse_args()
    D = a.dir
    lines = []

    def P(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        lines.append(s)

    t02 = read(T02 / "cells.tsv")
    P(f"# T07 scores\nT02/cells.tsv sha256 {hashlib.sha256((T02 / 'cells.tsv').read_bytes()).hexdigest()}")
    P(f"paper table sha256 {hashlib.sha256(a.paper.read_bytes()).hexdigest()}")

    # ── axes per cell ────────────────────────────────────────────────────────
    ax, merge = {}, {}
    for d in sorted(D.glob("axes_*")):
        for r in read(d / "cells_axes.tsv"):
            ax[(r["setting"], r["group"])] = r
        for r in read(d / "merge_settings.tsv"):
            merge[r["setting"]] = fl(r["A_merge"])

    # A_int recomputed from cells_per_class.tsv must equal cells.tsv (class-macro means)
    pcl = defaultdict(list)
    for r in read(T02 / "cells_per_class.tsv"):
        pcl[(r["setting"], r["group"])].append(r)
    gate_rows, int_bad = [], 0
    cells = []
    for c in t02:
        key = (c["setting"], c["group"])
        if not np.isfinite(fl(c["R05_c"])) or not np.isfinite(fl(c["Dorder"])):
            continue
        pc = pcl.get(key, [])
        do = float(np.mean([fl(r["Dorder"]) for r in pc])) if pc else float("nan")
        dq = float(np.mean([abs(fl(r["Dq"])) for r in pc])) if pc else float("nan")
        if not (abs(do - fl(c["Dorder"])) < 1e-4 and abs(dq - fl(c["Dq"])) < 1e-4):
            int_bad += 1
        r = ax.get(key)
        row = dict(setting=c["setting"], task=c["task"], group=c["group"], n_cases=c["n_cases"], R05_c=fl(c["R05_c"]),
                   R08_c=fl(c["R08_c"]), A_int_order=do, A_int_q=dq, A_merge=merge.get(c["setting"], float("nan")),
                   P03=c["setting"] in P03)
        for k in ("A_res", "A_noise", "A_res_inplane", "A_res_train", "A_noise_train", "t_test", "t_source", "n_test", "n_source"):
            row[k] = fl(r[k]) if r else float("nan")
        if r is None:
            P(f"MISSING axes for cell {key}")
        cells.append(row)
    n_spine = sum(c["task"] == "Spine" for c in t02)
    P(f"\n## Gates\nCells: {len(t02)} T02 cells, {n_spine} Spine (no images, excluded); {len(cells)} scored cells "
      f"(R05_c and T02 features defined), {len({c['setting'] for c in cells})} settings, {len({c['task'] for c in cells})} tasks.")
    P(f"A_int from cells_per_class.tsv equals cells.tsv Dorder/Dq: {len(cells) - int_bad}/{len(cells)}")

    # count gates from the axes jobs
    cg = []
    for d in sorted(D.glob("axes_*")):
        cg += read(d / "gate_counts.tsv")
    nbad = sum(g["ok"] != "True" for g in cg)
    P(f"A_res count gate: {len(cg)} rows (per item, per cell vs cells.tsv n_cases, per IND pool), failing {nbad}")
    for g in cg:
        if g["ok"] != "True":
            P("  FAIL", g)
    ng = []
    for d in sorted(D.glob("axes_*")):
        ng += read(d / "gate_noise.tsv")
    P(f"A_noise response gate (sigma = 0.05 x IQR): {sum(g['ok'] == 'True' for g in ng)}/{len(ng)} images rise")
    for g in ng:
        P(f"  {g['task']} {g['setting']} -> {g['group']}: n {fl(g['n_clean']):.4g} -> {fl(g['n_noisy']):.4g}, ratio {fl(g['ratio']):.3f}")
    noise_gate_ok = bool(ng) and all(g["ok"] == "True" for g in ng)
    P("A_merge: Palette's _kmeans_1d imported from AugLab and its assignment reproduced line by line; no re-implementation "
      "of the clustering, so the >= 99% agreement gate does not apply.")

    # paper gate (setting-level OOD rows) for the arms behind R05 and R08
    paper = parse_paper(a.paper)
    by_set = defaultdict(list)
    for c in t02:
        by_set[c["setting"]].append(c)
    setv = {}
    pg_bad = 0
    for s, cs in by_set.items():
        m = {arm: float(np.mean([fl(c[f"dice_{arm}"]) for c in cs])) for arm in ("palette", "ssem", "pa_auglab")}
        pp = paper.get(s)
        for arm in m:
            ok = pp is not None and abs(m[arm] - pp[arm]) <= TOL
            pg_bad += not ok
            gate_rows.append(dict(setting=s, arm=arm, cells_mean=round(m[arm], 4), paper=pp[arm] if pp else "", ok=ok))
        setv[s] = dict(R05=m["pa_auglab"] - m["palette"], R08=m["palette"] - m["ssem"], task=cs[0]["task"])
    P(f"Paper gate (_suppl_ind_ood.tex OOD rows; Palette, SynthSeg-EM, Palette+AugLab; tol {TOL}): "
      f"{len(gate_rows) - pg_bad}/{len(gate_rows)} pass, max |diff| "
      f"{max(abs(g['cells_mean'] - g['paper']) for g in gate_rows if g['paper'] != ''):.3f}")
    gate_ok = pg_bad == 0 and nbad == 0 and int_bad == 0
    with open(D / "gate.tsv", "w") as f:
        f.write("setting\tarm\tcells_mean\tpaper\tok\n")
        for g in gate_rows:
            f.write(f"{g['setting']}\t{g['arm']}\t{g['cells_mean']}\t{g['paper']}\t{g['ok']}\n")
    active_axes = [x for x in AXES if not (x == "A_noise" and not noise_gate_ok)]
    if not noise_gate_ok:
        P("A_noise gate FAILED: axis stopped (registration).")
    if not gate_ok:
        P("A gate failed: see above.")

    # ── per-cell table ───────────────────────────────────────────────────────
    keys = ["setting", "task", "group", "P03", "n_cases", "R05_c", "R08_c"] + AXES + SECONDARY + ["t_test", "t_source", "n_test", "n_source"]
    with open(D / "cells_t07.tsv", "w") as f:
        f.write("\t".join(keys) + "\n")
        for c in sorted(cells, key=lambda c: (c["task"], c["setting"], c["group"])):
            f.write("\t".join(f"{c[k]:.6g}" if isinstance(c[k], float) else str(c[k]) for k in keys) + "\n")

    # ── rule 1-3 ─────────────────────────────────────────────────────────────
    P("\n## Rule 1 (axis predicts R05_c, cells)")
    res = [rule1(cells, x, P) for x in active_axes]
    with open(D / "rule1.tsv", "w") as f:
        ks = list(res[0])
        f.write("\t".join(ks) + "\n")
        for r in res:
            f.write("\t".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")
    predicting = [r for r in res if r["predicts"]]
    P(f"\n## Rule 2 (M04 criterion 3): {'MET' if predicting else 'NOT MET'}; predicting axes: {[r['axis'] for r in predicting]}")
    P("\n## Rule 3 (naming)")
    name = {"A_res": "M05", "A_noise": "M06", "A_merge": "M07", "A_int_order": "intensity-relation coverage (M04)",
            "A_int_q": "intensity-relation coverage (M04)"}
    allr = sorted(res, key=lambda r: -r["rho"] if np.isfinite(r["rho"]) else 9)
    P(f"Highest pooled rho over all axes: {allr[0]['axis']} ({allr[0]['rho']:+.3f})")
    if predicting:
        pr = sorted(predicting, key=lambda r: -r["rho"])
        lead = pr[0]
        P(f"Leader among predicting axes: {lead['axis']} -> {name[lead['axis']]}")
        if len(pr) >= 2:
            other = pr[1]
            X = lambda k: [c[k] for c in cells]  # noqa: E731
            y = X("R05_c")
            pl = partial_rho(X(lead["axis"]), y, X(other["axis"]))
            po = partial_rho(X(other["axis"]), y, X(lead["axis"]))
            sep = pl >= 0.3 and po < 0.2
            P(f"Partial rho({lead['axis']} | {other['axis']}) = {pl:+.3f}; partial rho({other['axis']} | {lead['axis']}) = {po:+.3f}"
              f" -> {'separated' if sep else 'not separated'}")
            for o in pr[2:]:
                P(f"  (also) partial {lead['axis']}|{o['axis']} {partial_rho(X(lead['axis']), y, X(o['axis'])):+.3f}; "
                  f"{o['axis']}|{lead['axis']} {partial_rho(X(o['axis']), y, X(lead['axis'])):+.3f}")
        else:
            P("Only one predicting axis: nothing to separate.")
    else:
        P("No predicting axis: no candidate named.")

    # ── rule 4 ───────────────────────────────────────────────────────────────
    P("\n## Rule 4 (M07 prediction 1: A_merge vs R08, settings)")
    S = [(s, v) for s, v in setv.items() if s in merge and np.isfinite(merge[s])]
    xm = [merge[s] for s, _ in S]
    yr = [v["R08"] for _, v in S]
    r4, n4 = rho(xm, yr)
    tasks = sorted({v["task"] for _, v in S})
    loto4 = {t: rho([merge[s] for s, v in S if v["task"] != t], [v["R08"] for s, v in S if v["task"] != t])[0] for t in tasks}
    ok4 = r4 <= -0.4 and all(np.isfinite(v) and v < 0 for v in loto4.values())
    P(f"rho(A_merge, R08) = {r4:+.3f} (n = {n4} settings; Spine's 2 have no images)")
    P("LOTO: " + ", ".join(f"{t} {v:+.2f}" for t, v in loto4.items()))
    P(f"M07 prediction (1): {'SUPPORTED' if ok4 else 'NOT SUPPORTED'}")
    P("Settings (A_merge, R08, R05), P03 marked:")
    for s, v in sorted(S, key=lambda sv: merge[sv[0]]):
        P(f"  {'P03 ' if s in P03 else '    '}{s:22s} A_merge {merge[s]:.3f}  R08 {v['R08']:+.2f}  R05 {v['R05']:+.2f}")
    p03m = [merge[s] for s, _ in S if s in P03]
    rest = [merge[s] for s, _ in S if s not in P03]
    P(f"A_merge median: P03 {np.median(p03m):.3f} (n {len(p03m)}) vs rest {np.median(rest):.3f} (n {len(rest)})")
    r4_nop03 = rho([merge[s] for s, _ in S if s not in P03], [v["R08"] for s, v in S if s not in P03])
    P(f"rho(A_merge, R08) without P03: {r4_nop03[0]:+.3f} (n {r4_nop03[1]})")

    # ── reported, no outcome ─────────────────────────────────────────────────
    P("\n## Reported (no outcome)")
    P("Spearman matrix of the axes over cells:")
    allax = AXES + SECONDARY
    P("               " + " ".join(f"{x[:13]:>13s}" for x in allax))
    for x in allax:
        P(f"{x:15s}" + " ".join(f"{rho([c[x] for c in cells], [c[y] for c in cells])[0]:>13.2f}" for y in allax))
    P("\nSecondary axes vs R05_c (pooled rho, LOTO min):")
    for x in SECONDARY:
        r = rule1(cells, x, lambda *_: None)
        P(f"  {x}: rho {r['rho']:+.3f} (n {r['n']}), LOTO min {r['loto_min']:+.2f}, within {r['within_pos']}/{r['within_defined']}")
    P("\nSetting-level version of rule 1 (setting medians of each axis vs setting R05 = mean of its OOD cells):")
    sets = sorted({c["setting"] for c in cells})
    for x in AXES:
        xs = [float(np.nanmedian([c[x] for c in cells if c["setting"] == s])) for s in sets]
        ys = [setv[s]["R05"] for s in sets]
        r, n = rho(xs, ys)
        ts = [setv[s]["task"] for s in sets]
        lo = {t: rho([xx for xx, tt in zip(xs, ts) if tt != t], [yy for yy, tt in zip(ys, ts) if tt != t])[0] for t in sorted(set(ts))}
        P(f"  {x}: rho {r:+.3f} (n {n}); LOTO min {min(lo.values()):+.2f}, max {max(lo.values()):+.2f}")
    P("\nP03 cells vs the rest (median axis; rho(axis, R05_c) inside each subset):")
    p3 = [c for c in cells if c["P03"]]
    rs = [c for c in cells if not c["P03"]]
    P(f"  R05_c median: P03 {np.median([c['R05_c'] for c in p3]):+.2f} (n {len(p3)}) vs rest {np.median([c['R05_c'] for c in rs]):+.2f} (n {len(rs)})")
    for x in AXES:
        P(f"  {x}: median P03 {np.nanmedian([c[x] for c in p3]):.3f} vs rest {np.nanmedian([c[x] for c in rs]):.3f}; "
          f"rho P03 {rho([c[x] for c in p3], [c['R05_c'] for c in p3])[0]:+.2f}, rest {rho([c[x] for c in rs], [c['R05_c'] for c in rs])[0]:+.2f}")
    P("\nScatter table per axis: cells_t07.tsv (one row per cell, task labels, all axes, R05_c).")
    (D / "scores.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
