#!/usr/bin/env python3
"""Analysis of the texture-removed test item experiment (after 05_evaluate.sh).

Per task (one training contrast), out-of-domain pool and contrast groups exactly as in its ladder_series.json, case
values from the ladder engine's own loader (ladder_ood_common._labelled_case_values: Dice per case averaged over labels
and folds 0-2), patient units from ladder_ood_common._patient_key:
  Dice(orig)  = originals re-predicted on the same machine (<cat>_<RUN_ID>_origts)
  Dice(texrm) = texture-removed item                      (<cat>_<RUN_ID>_texrm)
  drop_m      = Dice(orig) - Dice(texrm) for m in {flat (rung 4.5), texture (rung 5, PA alone)}
  DiD         = drop_texture - drop_flat (> 0: the texture-fill model loses more when texture is removed)
Every number is a macro mean over the ladder's contrast groups (equal weight per group, then cases). DiD test: paired
sign-flip over patients (stat_tests.macro_perm_design: one unit per patient across groups, exact for the heaviest 20
units). Cross-task: exact two-sided permutation of the 9 task labels (C(9,4) = 126), statistic mean DiD appearance-
defined minus interface-bounded (as the paper's group test). Sanity: re-predicted originals vs the cluster-made
originals the ladders use (mean |delta Dice| per case, share of cases within 1 Dice point).
Prints aggregates only; --out writes the same table as TSV (scratch, never git).

  python analyze.py [--tasks MS,Glioma] [--out <file.tsv>] [--tags origts,texrm]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import ladder_ood_common as loc  # noqa: E402
from stat_tests import macro_perm_design  # noqa: E402
from prepare import P, ladder_sources  # noqa: E402

MODELS = (("flat", 4), ("texture", 5))


def tagged(task: str, key: str, tag: str) -> str:
    if tag == "cluster":
        return key
    if task == "Mandible":                       # evaluators without CKPT_TAG: <metrics>/texture_removal_<tag>/<run>
        return f"texture_removal_{tag}/" + key.rsplit("/", 1)[-1]
    return f"{key}_{tag}"


def group_of(d: dict) -> dict:
    if d.get("mode") == "grouped_by_contrast":
        return {m: g for g, ms in d["ood_groups"].items() for m in ms}
    return None


def load(task, t, d, tag):
    own, ood_c, extras = ladder_sources(d, t)
    gmap = group_of(d)
    out = {}
    for m, i in MODELS:
        vals = loc._labelled_case_values(own, ood_c, extras, tagged(task, d["run_keys"][i], tag), "dice")
        for lab, cases in vals.items():
            g = gmap.get(lab) if gmap else lab
            if g is None:
                continue
            for ck, v in cases.items():
                out.setdefault(m, {}).setdefault(g, {})[(lab, ck)] = 100.0 * v
    return out


def task_row(task, t, tags=("origts", "texrm")):
    d = json.loads(P(t["ladder"]).read_text())
    A, B, C = load(task, t, d, tags[0]), load(task, t, d, tags[1]), load(task, t, d, "cluster")
    if not all(A.get(m) and B.get(m) for m, _ in MODELS):
        return None
    groups = sorted(set.intersection(*[set(X[m]) for X in (A, B) for m, _ in MODELS]))
    entries, K = [], len(groups)
    mean = {k: [] for k in ("of", "rf", "ot", "rt")}
    n_units = set()
    for gi, g in enumerate(groups):
        keys = sorted(set(A["flat"][g]) & set(B["flat"][g]) & set(A["texture"][g]) & set(B["texture"][g]))
        if not keys:
            continue
        of = np.array([A["flat"][g][k] for k in keys]); rf = np.array([B["flat"][g][k] for k in keys])
        ot = np.array([A["texture"][g][k] for k in keys]); rt = np.array([B["texture"][g][k] for k in keys])
        for name, arr in (("of", of), ("rf", rf), ("ot", ot), ("rt", rt)):
            mean[name].append(arr.mean())
        did = (ot - rt) - (of - rf)
        for (lab, ck), c in zip(keys, did):
            u = loc._patient_key(ck, lab)
            n_units.add(u)
            entries.append((gi, lab, u, c / len(keys)))
    est, p_two, _ = macro_perm_design(entries, K, higher_better=True)
    san = []
    for m, _ in MODELS:
        for g in groups:
            for k in set(A[m][g]) & set(C.get(m, {}).get(g, {})):
                san.append(A[m][g][k] - C[m][g][k])
    san = np.abs(np.array(san)) if san else np.array([np.nan])
    mo = {k: float(np.mean(v)) for k, v in mean.items()}
    return dict(task=task, group=t["group"], contrast=t["training_contrast"], n_groups=K, n_patients=len(n_units),
                flat_orig=mo["of"], flat_texrm=mo["rf"], flat_drop=mo["of"] - mo["rf"],
                tex_orig=mo["ot"], tex_texrm=mo["rt"], tex_drop=mo["ot"] - mo["rt"], did=est, p_did=p_two,
                sanity_mean_abs=float(np.nanmean(san)), sanity_within1=float(np.mean(san <= 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tags", default="origts,texrm", help="orig,texrm metric tags (self-test: other tags)")
    a = ap.parse_args()
    cfg = yaml.safe_load((HERE / "tasks.yaml").read_text())
    names = [x for x in a.tasks.split(",") if x] or list(cfg["tasks"])
    rows = []
    for n in names:
        r = task_row(n, cfg["tasks"][n], tuple(a.tags.split(",")))
        if r is None:
            print(f"{n:9s} metrics not complete yet (need _origts and _texrm for both models)")
            continue
        rows.append(r)
    print("task      group       n_pat | flat orig/texrm/drop   | texture orig/texrm/drop | DiD    p(sign-flip) | "
          "re-pred vs cluster: mean|dDice|, share<=1")
    for r in rows:
        print(f"{r['task']:9s} {r['group']:11s} {r['n_patients']:5d} | {r['flat_orig']:5.1f} {r['flat_texrm']:5.1f} "
              f"{r['flat_drop']:+5.1f}      | {r['tex_orig']:5.1f} {r['tex_texrm']:5.1f} {r['tex_drop']:+5.1f}       | "
              f"{r['did']:+5.2f}  {r['p_did']:.2g}        | {r['sanity_mean_abs']:.2f}, {r['sanity_within1']:.2f}")
    if len(rows) >= 3 and len({r["group"] for r in rows}) == 2:
        v = np.array([r["did"] for r in rows])
        app = [i for i, r in enumerate(rows) if r["group"] == "appearance"]
        stat = lambda g: v[list(g)].mean() - np.delete(v, list(g)).mean()
        obs = stat(app)
        perms = [stat(g) for g in combinations(range(len(v)), len(app))]
        k = sum(abs(x) >= abs(obs) - 1e-9 for x in perms)
        print(f"group permutation over {len(rows)} tasks: DiD appearance - interface = {obs:+.2f} Dice, "
              f"two-sided p = {k}/{len(perms)} = {k / len(perms):.3f}; "
              f"appearance > every interface task: {min(v[app]) > max(np.delete(v, app))}")
    if a.out:
        import csv
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
