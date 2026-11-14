#!/usr/bin/env python3
"""T02 step 3: assemble cells.tsv and score the registered T02 predictions (registration
plans/2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...). Light (TSV in, TSV/text out).

Rules, as registered (unit = cell = setting x OOD domain group; Spearman rho; supported if |rho| >= 0.4 in the predicted
direction and, where computable, task-blocked permutation p < 0.05):
  M01: reversed cells have R01_c < 0.3 and kept-order cells mostly >= 0.3; rho(R01_c, Dorder) <= -0.4; reversed cells
       with R01_c >= 0.6 are counter-examples.
  M02: among kept-order cells, rho(R01_c, Dq) <= -0.4.
  M03: rho(Dice_c, cnr_t) >= 0.4 for Palette, Palette+AugLab and SynthSeg-EM, and rho(spec_c, cnr_t) >= 0.4.
  Separation: cells where order and quantile disagree (kept order with large Dq, or reversed with small Dq); < 5 ->
       "not separable on our cells".
  Membership: P01 (R01 < 0.30 and R03 >= 3.2), P02 (R01 >= 0.60), P11 (texture-step loss significant at patient level
       on the cell while the setting's t >= 0), P15 (R02 best randomised >= 3.2 and |SynthSeg-EM - max(AugLab, Palette,
       Palette+AugLab)| < 3.2), per cell vs per setting.
Interpretations fixed before reading any value (listed in REPORT.md DEVIATIONS):
  * multi-class cell "reversed" call = share of reversed classes >= 0.5 (sensitivity: any class reversed);
  * "large"/"small" Dq for the separation count = above / below the median Dq of the scored cells;
  * task-blocked permutation: cells ordered by (setting, group) inside a task; all permutations enumerated; no p-value
    when fewer than 20 distinct permutations exist ("too few tasks match").
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TH = 3.2
MIN_PERMS = 20
F024_P11 = {("Breast T1-CE", "precontrast"), ("Breast T2w", "precontrast")}   # the two named F024 cells; two CT cells unnamed


def read(path: Path) -> list:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return np.nan


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k)))
                              for k in keys) + "\n")


def task_perm_test(rows, xk, yk, direction):
    """Spearman rho(x, y) over rows + task-blocked permutation p (one-sided in `direction`: -1 = predicted negative).
    Returns (rho, n, p or None, n_perms)."""
    rows = [r for r in rows if np.isfinite(r[xk]) and np.isfinite(r[yk])]
    if len(rows) < 4:
        return np.nan, len(rows), None, 0
    rho = spearmanr([r[xk] for r in rows], [r[yk] for r in rows]).correlation
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["setting"], r["group"])):
        by[r["task"]].append(r)
    size = defaultdict(list)
    for t, rs in by.items():
        size[len(rs)].append(t)
    nperm = math.prod(math.factorial(len(ts)) for ts in size.values())
    if nperm < MIN_PERMS:
        return rho, len(rows), None, nperm
    y = [r[yk] for t in sorted(by) for r in by[t]]
    classes = [sorted(ts) for ts in size.values()]
    stats_ = []
    for combo in itertools.product(*[itertools.permutations(ts) for ts in classes]):
        assign = {}
        for ts, perm in zip(classes, combo):
            assign.update(dict(zip(ts, perm)))
        x = [r[xk] for t in sorted(by) for r in by[assign[t]]]
        stats_.append(spearmanr(x, y).correlation)
    stats_ = np.array(stats_)
    p = float(np.mean(stats_ <= rho + 1e-12)) if direction < 0 else float(np.mean(stats_ >= rho - 1e-12))
    return rho, len(rows), p, nperm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="T02 output dir (step1_<task>/, features_<task>/)")
    a = ap.parse_args()
    D = a.dir
    cells, settings, gates, feats, pcls = [], [], [], {}, []
    for d in sorted(D.glob("step1_*")):
        if d.name == "step1_verify":
            continue
        cells += read(d / "cells_step1.tsv")
        settings += read(d / "settings_step1.tsv")
        gates += read(d / "gate_step1.tsv")
    for d in sorted(D.glob("features_*")):
        if (d / "cells_features.tsv").exists():
            for r in read(d / "cells_features.tsv"):
                feats[(r["setting"], r["group"])] = r
            pcls += read(d / "cells_per_class.tsv")
    num = lambda r, ks: {k: fnum(r.get(k)) for k in ks}
    S = {s["setting"]: s for s in settings}
    # setting-level OOD rows (run_one) and published rows from the gate file
    srow = defaultdict(dict)
    for g in gates:
        if g["source"].startswith("_suppl_ind_ood"):
            srow[g["setting"]][g["arm"]] = fnum(g["run_one_row"])
    out = []
    for r in cells:
        v = num(r, [k for k in r if k.startswith("dice_")] + ["plain_ind", "p_texture_step_cell", "texture_step_cell_macro_delta"])
        o = dict(setting=r["setting"], task=r["task"], group=r["group"], items=r["items"], n_cases=int(r["n_cases"]))
        o.update({k: v[k] for k in v if k.startswith("dice_")})
        o["plain_ind"] = v["plain_ind"]
        o["R01_c"] = v["dice_plain"] / v["plain_ind"]
        o["R03_c"] = v["dice_palette"] - v["dice_plain"]
        o["R05_c"] = v["dice_pa_auglab"] - v["dice_palette"]
        o["R08_c"] = v["dice_palette"] - v["dice_ssem"]
        o["R09_c"] = v["dice_palette"] - v.get("dice_flat", np.nan)
        o["R10_c"] = v.get("dice_noise", np.nan) - v.get("dice_flat", np.nan)
        rand = {k: v[f"dice_{k}"] for k in ("ssem", "auglab", "palette", "pa_auglab")}
        best = max(rand, key=lambda k: rand[k])
        same = r.get("spec_same_items") == "True"
        o["spec_setting"], o["spec_arm"] = r.get("spec_setting", ""), r.get("spec_arm", "")
        o["dice_spec"] = fnum(r.get("dice_spec")) if same else np.nan
        o["spec_same_items"] = r.get("spec_same_items", "")
        o["best_randomised_arm"] = best
        o["R02_c"] = o["dice_spec"] - rand[best]
        o["R02_c_plain"] = o["dice_spec"] - v["dice_plain"]
        o["texture_step_cell_macro_delta"] = v["texture_step_cell_macro_delta"]
        o["p_texture_step_cell"] = v["p_texture_step_cell"]
        f = feats.get((r["setting"], r["group"]))
        for k in ("o_s", "o_t", "reversed", "Dorder", "q_s", "q_t", "Dq", "cnr_t", "cnr_s", "n_classes", "n_cases_unmapped_features"):
            o[k] = fnum(f.get(k)) if f else np.nan
        o["reversed_call"] = (o["reversed"] >= 0.5) if np.isfinite(o["reversed"]) else None
        o["reversed_any"] = (o["reversed"] > 0) if np.isfinite(o["reversed"]) else None
        # per-cell memberships
        o["P01_c"] = bool(o["R01_c"] < 0.30 and o["R03_c"] >= TH)
        o["P02_c"] = bool(o["R01_c"] >= 0.60)
        t_set = fnum(S.get(r["setting"], {}).get("t_published"))
        o["P11_c"] = (bool(o["texture_step_cell_macro_delta"] < 0 and o["p_texture_step_cell"] < 0.05 and t_set >= 0)
                      if np.isfinite(o["p_texture_step_cell"]) else None)
        fam = max(rand["auglab"], rand["palette"], rand["pa_auglab"])
        o["P15_c"] = (bool(o["R02_c"] >= TH and abs(rand["ssem"] - fam) < TH) if np.isfinite(o["R02_c"]) else None)
        out.append(o)

    # setting-level calls (same rules, setting OOD rows)
    by_set = defaultdict(list)
    for o in out:
        by_set[o["setting"]].append(o)
    set_calls = {}
    for s, os_ in by_set.items():
        row = srow.get(s, {})
        pi = fnum(S[s]["plain_ind"])
        r01 = row.get("plain", np.nan) / pi
        r03 = row.get("palette", np.nan) - row.get("plain", np.nan)
        specs = [o["dice_spec"] for o in os_]
        if all(np.isfinite(specs)):
            rand = {k: row.get(k, np.nan) for k in ("ssem", "auglab", "palette", "pa_auglab")}
            r02 = float(np.mean(specs)) - max(rand.values())
            p15 = bool(r02 >= TH and abs(rand["ssem"] - max(rand["auglab"], rand["palette"], rand["pa_auglab"])) < TH)
        else:
            r02, p15 = np.nan, None
        set_calls[s] = dict(R01=r01, R03=r03, R02=r02, P01=bool(r01 < 0.30 and r03 >= TH), P02=bool(r01 >= 0.60), P15=p15)
    for o in out:
        sc = set_calls[o["setting"]]
        o["P01_set"], o["P02_set"], o["P15_set"] = sc["P01"], sc["P02"], sc["P15"]
        o["P11_F024"] = (o["setting"], o["group"]) in F024_P11
    write(D / "cells.tsv", out)
    write(D / "cells_per_class.tsv", pcls)
    write(D / "gate.tsv", gates)

    # ── scoring ──
    L = []
    P = L.append
    P(f"T02 scores. Cells (OOD setting x domain group): {len(out)}; with features: "
      f"{sum(np.isfinite(o['Dorder']) for o in out)}; settings: {len(by_set)}; tasks: {len({o['task'] for o in out})}.")
    nf = sum(g["ok"] != "True" for g in gates)
    P(f"Gate rows: {len(gates)}, failing: {nf}.")
    for g in gates:
        if g["ok"] != "True":
            P(f"  GATE FAIL {g['setting']} {g['arm']} [{g['source']}]: cells {g['cells_mean']} vs published {g['published']}")
    F = [o for o in out if np.isfinite(o["Dorder"]) and np.isfinite(o["R01_c"])]
    P("")
    P("== M01 (order) ==")
    rev = [o for o in F if o["reversed_call"]]
    kept = [o for o in F if o["reversed_call"] is False]
    P(f"reversed cells (class share >= 0.5): {len(rev)}; R01_c < 0.3 in {sum(o['R01_c'] < 0.3 for o in rev)}/{len(rev)}")
    for o in sorted(rev, key=lambda o: o["R01_c"]):
        P(f"  reversed: {o['setting']} -> {o['group']}: R01_c {o['R01_c']:.3f}, Dorder {o['Dorder']:.2f}, reversed share {o['reversed']:.2f}")
    P(f"kept-order cells: {len(kept)}; R01_c >= 0.3 in {sum(o['R01_c'] >= 0.3 for o in kept)}/{len(kept)}")
    ce = [o for o in rev if o["R01_c"] >= 0.6]
    P(f"counter-examples (reversed, R01_c >= 0.6): {len(ce)}" + "".join(f"\n  {o['setting']} -> {o['group']} R01_c {o['R01_c']:.3f}" for o in ce))
    rho, n, p, npm = task_perm_test(F, "Dorder", "R01_c", -1)
    P(f"rho(R01_c, Dorder) = {rho:.3f} (n = {n} cells); task-blocked permutations: {npm}; p = {p if p is not None else 'not computable'}")
    rev_any = [o for o in F if o["reversed_any"]]
    P(f"sensitivity (any class reversed): {len(rev_any)} cells, R01_c < 0.3 in {sum(o['R01_c'] < 0.3 for o in rev_any)}")
    m01 = rho <= -0.4 and (p is None or p < 0.05)
    P(f"M01 correlation rule: {'SUPPORTED' if m01 else 'NOT SUPPORTED'}")
    P("")
    P("== M02 (quantile, kept order) ==")
    rho2, n2, p2, npm2 = task_perm_test([o for o in kept if np.isfinite(o["Dq"])], "Dq", "R01_c", -1)
    P(f"rho(R01_c, Dq | kept order) = {rho2:.3f} (n = {n2}); permutations {npm2}; p = {p2 if p2 is not None else 'not computable'}")
    P(f"M02: {'SUPPORTED' if (rho2 <= -0.4 and (p2 is None or p2 < 0.05)) else 'NOT SUPPORTED'}")
    P("")
    P("== M03 (target CNR on the test contrast) ==")
    ok3 = True
    for arm in ("palette", "pa_auglab", "ssem"):
        r_, n_, p_, npm_ = task_perm_test([o for o in out if np.isfinite(o["cnr_t"])], "cnr_t", f"dice_{arm}", +1)
        ok = r_ >= 0.4 and (p_ is None or p_ < 0.05)
        ok3 &= ok
        P(f"rho(Dice_{arm}, cnr_t) = {r_:.3f} (n = {n_}); permutations {npm_}; p = {p_ if p_ is not None else 'not computable'} -> {'yes' if ok else 'no'}")
    sp = [o for o in out if np.isfinite(o["cnr_t"]) and np.isfinite(o["dice_spec"])]
    r_, n_, p_, npm_ = task_perm_test(sp, "cnr_t", "dice_spec", +1)
    oks = r_ >= 0.4 and (p_ is None or p_ < 0.05)
    P(f"rho(spec_c, cnr_t) = {r_:.3f} (n = {n_}); permutations {npm_}; p = {p_ if p_ is not None else 'not computable'} -> {'yes' if oks else 'no'}")
    P(f"M03: {'SUPPORTED' if (ok3 and oks) else 'NOT SUPPORTED'}")
    if sp:
        qc = np.quantile([o["cnr_t"] for o in sp], 0.25)
        ms = np.median([o["dice_spec"] for o in sp])
        P(f"cells with a specialist and cnr_t in the lowest quartile (<= {qc:.2f}); spec Dice vs median spec {ms:.1f}:")
        for o in sorted([o for o in sp if o["cnr_t"] <= qc], key=lambda o: o["cnr_t"]):
            P(f"  {o['setting']} -> {o['group']}: cnr_t {o['cnr_t']:.2f}, spec {o['dice_spec']:.1f} "
              f"({'HIGH: M03 does not explain R02 here' if o['dice_spec'] >= ms else 'low'}), R02_c {o['R02_c']:.1f}")
    P("")
    P("== Separation of M01 and M02 ==")
    dqs = [o["Dq"] for o in F if np.isfinite(o["Dq"])]
    med = float(np.median(dqs)) if dqs else np.nan
    dis = [o for o in F if np.isfinite(o["Dq"]) and ((o["reversed_call"] is False and o["Dq"] > med) or (o["reversed_call"] and o["Dq"] < med))]
    P(f"median Dq = {med:.3f}; disagreeing cells: {len(dis)}" + (" -> not separable on our cells" if len(dis) < 5 else ""))
    for o in dis:
        P(f"  {o['setting']} -> {o['group']}: {'reversed' if o['reversed_call'] else 'kept'}, Dq {o['Dq']:.3f}, R01_c {o['R01_c']:.3f}")
    P("")
    P("== Membership per cell vs per setting ==")
    for pat in ("P01", "P02", "P15"):
        diff = [o for o in out if o[f"{pat}_c"] is not None and o[f"{pat}_set"] is not None and o[f"{pat}_c"] != o[f"{pat}_set"]]
        P(f"{pat}: cells in {sum(bool(o[f'{pat}_c']) for o in out)}; settings in "
          f"{sum(bool(c[pat]) for c in set_calls.values())}; cells whose call differs from their setting's: {len(diff)}")
        for o in diff:
            P(f"  {o['setting']} -> {o['group']}: cell {o[f'{pat}_c']}, setting {o[f'{pat}_set']} "
              f"(R01_c {o['R01_c']:.3f}, R03_c {o['R03_c']:.1f}, R02_c {o['R02_c']:.1f})")
    p11 = [o for o in out if o["P11_c"]]
    P(f"P11: cells with a significant texture-step loss while the setting's t >= 0: {len(p11)}")
    for o in p11:
        P(f"  {o['setting']} -> {o['group']}: cell step {o['texture_step_cell_macro_delta']:.2f}, p {o['p_texture_step_cell']:.2g}"
          f"{' (F024 named)' if o['P11_F024'] else ''}")
    miss = [k for k in F024_P11 if not any((o["setting"], o["group"]) == k and o["P11_c"] for o in out)]
    P(f"F024-named P11 cells not reproduced: {miss}")
    P("")
    P("Setting-level calls (same rules on the setting OOD rows):")
    for s, c in set_calls.items():
        P(f"  {s}: R01 {c['R01']:.3f} R03 {c['R03']:.1f} R02 {c['R02']:.1f} P01 {c['P01']} P02 {c['P02']} P15 {c['P15']}")
    (D / "scores.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
