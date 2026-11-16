#!/usr/bin/env python3
"""T13 scoring (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...).

Reads t13_folds.json (t13_fold_values.py). Per setting and fold: D_IND = val000 - val100 IND Dice, D_OOD the same on OOD.
theta(s) = 2.776 x SD(D, ddof 1) / sqrt 3 (own setting); theta_pool = 2.776 x sqrt(mean over in-scope settings of SD^2)
/ sqrt 3 (T01's pooled SD), per row (IND, OOD). A setting 'differs' on a row when |mean D| >= theta(s) AND >= theta_pool.
Rule 1 (M25): supported if >= 5 settings differ in IND or in OOD (count of settings differing on at least one row; the two
per-row counts are also reported), not supported if <= 2, else undetermined. Thresholds as registered (out of 21), not
rescaled to the settings in scope.
Rule 2 (P13): R21 = Palette - Palette+AugLab IND (the P13 / T01 sign: positive = Palette+AugLab lower). For each of the 9
strong P13 members: accounted for by M25 if R21(val100) < 3.2 while R21(val000) >= 3.2; P13 'explained by selection' only
if >= 5/9.
Gate: val000 (and Palette) 3-fold mean within 0.1 Dice of tab:suppl-ind-ood-dice (t01_score.parse_indood_tex), and val000
folds equal T01's indood_folds.json (same path, 1e-6).
Blocks: 'mirror' (scored) and 'config' (the config's val100 run when it is a separate training run; unscored sensitivity).
Usage: python3 t13_score.py --folds J --tex TEX --t01 J --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t01_step_noise"))
import t01_score as TS  # noqa: E402

T4 = 2.776
SQ3 = math.sqrt(3.0)
REF, PA = TS.REF, TS.PA
P13 = TS.REG["P13"]
TOL = 0.1
N_REG = 21


def fmean(f):
    return float(np.mean(list(f.values())))


def block(S: dict, key: str):
    """per-setting D rows for one val100 block; pooled theta over the settings in that block."""
    rows = {}
    for s, v in S.items():
        F = v["folds"]
        if key not in F or REF not in F:
            continue
        r = {}
        for row in ("IND", "OOD"):
            a, b = F[REF].get(row), F[key].get(row)
            if not a or not b or sorted(a) != sorted(b) or len(a) != 3:
                continue
            d = {k: a[k] - b[k] for k in sorted(a)}
            r[row] = dict(d=d, mean=fmean(d), sd=float(np.std(list(d.values()), ddof=1)))
        rows[s] = r
    pool = {}
    for row in ("IND", "OOD"):
        sds = [r[row]["sd"] for r in rows.values() if row in r]
        pool[row] = (math.sqrt(float(np.mean(np.square(sds)))) if sds else float("nan"), len(sds))
    for r in rows.values():
        for row, x in r.items():
            x["theta"] = T4 * x["sd"] / SQ3
            x["theta_pool"] = T4 * pool[row][0] / SQ3
            x["beyond_own"] = abs(x["mean"]) >= x["theta"]
            x["beyond_pool"] = abs(x["mean"]) >= x["theta_pool"]
            x["differs"] = bool(x["beyond_own"] and x["beyond_pool"])
    return rows, pool


def _out(n, n_missing):
    """registered thresholds over 21 settings; settings not computable may still differ (as T04's M28 handling: an outcome
    is given only when the missing settings cannot change it)."""
    if n >= 5:
        return "supported"
    if n + n_missing <= 2:
        return "not supported"
    if n + n_missing < 5 and n > 2:
        return "undetermined"
    return "undetermined (missing settings could change it)" if n <= 2 else "undetermined"


def rule1(rows):
    ind = sorted(s for s, r in rows.items() if r.get("IND", {}).get("differs"))
    ood = sorted(s for s, r in rows.items() if r.get("OOD", {}).get("differs"))
    union = sorted(set(ind) | set(ood))
    n, miss = len(union), N_REG - len(rows)
    on_computed = "supported" if n >= 5 else ("not supported" if n <= 2 else "undetermined")
    return dict(n_in_scope=len(rows), n_registered=N_REG, n_missing=miss, ind=ind, ood=ood, union=union, n_union=n,
                outcome=_out(n, miss), outcome_on_computed_settings_only=on_computed,
                outcome_ind_only=_out(len(ind), miss), outcome_ood_only=_out(len(ood), miss))


def rule2(S, key):
    res = []
    for s in P13:
        v = S.get(s)
        F = (v or {}).get("folds", {})
        if not v or PA not in F or REF not in F or not F[PA].get("IND") or not F[REF].get("IND"):
            res.append(dict(setting=s, computable=False, why="no Palette / val000 IND folds"))
            continue
        r0 = fmean(F[PA]["IND"]) - fmean(F[REF]["IND"])
        if key not in F or not F[key].get("IND"):
            res.append(dict(setting=s, computable=False, r21_val000=r0, why=f"no {key} run with complete metrics"))
            continue
        r1 = fmean(F[PA]["IND"]) - fmean(F[key]["IND"])
        res.append(dict(setting=s, computable=True, r21_val000=r0, r21_val100=r1, accounted=bool(r1 < 3.2 and r0 >= 3.2)))
    n_acc = sum(1 for r in res if r.get("accounted"))
    n_comp = sum(1 for r in res if r["computable"])
    max_possible = n_acc + (len(P13) - n_comp)
    out = ("explained by selection" if n_acc >= 5 else
           ("not explained by selection" if max_possible < 5 else "undetermined (uncomputable members could reach 5/9)"))
    return res, dict(n_accounted=n_acc, n_computable=n_comp, n_members=len(P13), max_possible=max_possible, outcome=out)


def main():
    ap = argparse.ArgumentParser()
    for k in ("folds", "tex", "t01", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    J = json.loads(Path(a.folds).read_text())
    S = J["settings"]
    tex = TS.parse_indood_tex(Path(a.tex))
    T01 = json.loads(Path(a.t01).read_text())["settings"]
    # ── gate ──
    gate, ok_all = [], True
    for s, v in S.items():
        for arm in (REF, PA):
            for row in ("IND", "OOD"):
                f = (v["folds"].get(arm) or {}).get(row)
                if not f:
                    continue
                m, sd = fmean(f), float(np.std(list(f.values())))
                pub = ((tex.get(s) or {}).get(row) or {}).get(arm)
                t01f = ((T01.get(s, {}).get("folds", {}).get(arm) or {}).get(row)) or {}
                t01_same = bool(t01f) and sorted(t01f) == sorted(f) and all(abs(t01f[k] - f[k]) <= 1e-6 for k in f)
                ok = pub is not None and abs(m - pub[0]) <= TOL + 1e-9 and abs(sd - pub[1]) <= TOL + 1e-9
                ok_all &= ok
                gate.append(dict(setting=s, arm=TS.SHORT[arm], row=row, mean_of_folds=m, fold_std_pop=sd,
                                 published=pub[0] if pub else None, published_std=pub[1] if pub else None,
                                 t01_identical=t01_same, gate="ok" if ok else "FAIL"))
    with open(out / "gate.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gate[0]), delimiter="\t")
        w.writeheader()
        for g in gate:
            w.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in g.items()})
    # ── scope ──
    scope = []
    for s, v in S.items():
        st = v["status"]
        ck = st.get("ckpt", {})
        scope.append(dict(setting=s, val000_run=st["val000_run"], mirror_run=st["mirror_run"],
                          mirror_metrics_complete=st.get("val100_mirror", {}).get("included", False),
                          mirror_coverage_note="; ".join(st.get("val100_mirror", {}).get("coverage_notes", [])[:3]),
                          trainer_val000=",".join(ck.get("val000", {}).get("trainer", [])),
                          trainer_mirror=",".join(ck.get("mirror", {}).get("trainer", [])),
                          mirror_same_checkpoint_final=json.dumps(ck.get("mirror_same_final")),
                          config_val100_run=st["config_val100_run"],
                          config_val100_separate_run_included=st.get("val100_config", {}).get("included", False),
                          config_same_checkpoint_final=json.dumps(ck.get("config_same_final"))))
    with open(out / "scope.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scope[0]), delimiter="\t")
        w.writeheader()
        w.writerows(scope)
    # ── rules ──
    result = {"gate_all_ok": bool(ok_all), "blocks": {}}
    lines = []
    for key, name in (("val100_mirror", "mirror"), ("val100_config", "config (separate val100 training run, unscored)")):
        rows, pool = block(S, key)
        r1 = rule1(rows)
        r2, r2s = rule2(S, key)
        result["blocks"][key] = {"rows": rows, "pool": pool, "rule1": r1, "rule2": r2, "rule2_summary": r2s}
        for s, r in rows.items():
            for row, x in r.items():
                lines.append(dict(block=key, setting=s, row=row, d_fold0=x["d"].get("fold0", np.nan),
                                  d_fold1=x["d"].get("fold1", np.nan), d_fold2=x["d"].get("fold2", np.nan),
                                  mean_D=x["mean"], sd_pair=x["sd"], theta=x["theta"], theta_pool=x["theta_pool"],
                                  beyond_own=x["beyond_own"], beyond_pool=x["beyond_pool"], differs=x["differs"]))
        print(f"== {name}: {len(rows)} settings; pooled SD IND {pool['IND'][0]:.3f} (n {pool['IND'][1]}), "
              f"OOD {pool['OOD'][0]:.3f} (n {pool['OOD'][1]})")
        print("   rule 1:", {k: v for k, v in r1.items()})
        print("   rule 2:", r2s)
        for x in r2:
            print("     ", x)
    if lines:
        with open(out / "mirror_diffs.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lines[0]), delimiter="\t")
            w.writeheader()
            for g in lines:
                w.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in g.items()})
    (out / "rules.json").write_text(json.dumps(result, indent=1, default=str))
    print("gate:", "ALL OK" if ok_all else "FAILURES", f"({sum(g['gate'] == 'ok' for g in gate)}/{len(gate)} rows ok;"
          f" T01-identical {sum(g['t01_identical'] for g in gate)}/{len(gate)})")
    for g in gate:
        if g["gate"] != "ok":
            print("  GATE FAIL", g)


if __name__ == "__main__":
    main()
