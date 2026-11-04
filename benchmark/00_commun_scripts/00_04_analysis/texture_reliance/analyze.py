#!/usr/bin/env python3
"""Texture-reliance experiment, step 6: statistics and report (preregistered in README.md; nothing here may change
after the first result is computed).

Per training domain, from <work>/score/<domain>.tsv:
  case sensitivity  S = 1 - mean of the finite agreement Dice over (evaluated labels, folds 0-2)
                    (same "mean over finite label x fold rows" rule as ladder_ood_common.load_case_means)
  case GT drop      G = mean finite Dice(orig, GT) - mean finite Dice(T, GT), same rule
  domain value      = equal weight per OOD stratum (the ladder's own ood_groups, or one stratum per item), cases pooled
                      inside a stratum (= the ladder's grouped-mode estimand)
  effect            = S_texture - S_flat (positive = scrambling fine texture changes the texture-fill model's
                      predictions more than the flat-fill model's); GT effect = G_texture - G_flat
  within-domain test: patient-level sign-flip on the effect, stat_tests.macro_perm_design (units = physical patients,
                      ladder_ood_common._patient_key: same patient across items merged, I-SPY2 _uni/_bil merged,
                      on-harmony = session, totalseg-pelvic CT/MRI cohorts kept apart), one-sided (texture > flat)
                      and two-sided; Holm over the 19 domains reported alongside.
Across domains: exact sign test over the 19 domains (stat_tests.sign_test, S_texture vs S_flat), never a mean over
tasks, never patients pooled across tasks. Appearance vs interface: descriptive counts only.
Verdict rule (README): SUPPORTED if the sign test is one-sided p < 0.05 (>= 14/19 positive) AND >= 10/19 domains
have effect > 0 with within-domain one-sided p < 0.05; KILLED if <= 9/19 domains have effect > 0 with p < 0.05 and
the sign test is not significant; otherwise INCONCLUSIVE.
Writes <work>/analysis/{REPORT.md, domains.tsv} (aggregates only; scratch, never git).

  .venv/bin/python analyze.py [--work ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import stat_tests as st  # noqa: E402
from ladder_ood_common import _patient_key  # noqa: E402

# preregistered per-image QC gates (README.md); an image passes when all hold
QC_GATES = dict(scr_corr_abs_max=0.05, le_corr_min=0.80, fine_corr_max=0.50, lp4_frac_max=0.25, lp4_rel_max=0.15)
QC_PASS_FRAC = 0.95       # QC of the operator passes when >= 95% of the T images pass every gate


def gate(r):
    return (abs(r["scr_corr"]) <= QC_GATES["scr_corr_abs_max"] and r["le_corr"] >= QC_GATES["le_corr_min"]
            and r["fine_corr"] <= QC_GATES["fine_corr_max"] and r["lp4_frac"] <= QC_GATES["lp4_frac_max"]
            and r["lp4_rel"] <= QC_GATES["lp4_rel_max"])


def f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def case_values(rows, key):
    """{(model, label, case): mean of finite values of `key` over label x fold rows}."""
    acc = defaultdict(list)
    for r in rows:
        v = r[key]
        if np.isfinite(v):
            acc[(r["model"], r["label"], r["case"])].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def domain_stats(D, rows, metric):
    """metric 'S' (1 - agreement) or 'G' (GT Dice drop). Returns dict of domain-level numbers."""
    if metric == "S":
        cv = {k: 1.0 - v for k, v in case_values(rows, "agree").items()}
    else:
        o, t = case_values(rows, "gt_o"), case_values(rows, "gt_T")
        cv = {k: o[k] - t[k] for k in o if k in t}
    strata = list(D["strata"].items())
    ds_of = {it["label"]: it["dataset"] for it in D["items"]}
    K, entries, sv = 0, [], {"flat": [], "texture": []}
    n_cases, dropped = 0, 0
    for g, labels in strata:
        pairs = []
        for lab in labels:
            it = next(x for x in D["items"] if x["label"] == lab)
            for c in it["cases"]:
                a, b = cv.get(("texture", lab, c)), cv.get(("flat", lab, c))
                if a is None or b is None:
                    dropped += 1
                    continue
                pairs.append((lab, c, a, b))
        if not pairs:
            continue
        idx = K
        K += 1
        n = len(pairs)
        n_cases += n
        sv["texture"].append(np.mean([p[2] for p in pairs]))
        sv["flat"].append(np.mean([p[3] for p in pairs]))
        for lab, c, a, b in pairs:
            unit = _patient_key(f"{ds_of[lab]}|{c}", lab)
            entries.append((idx, lab, unit, (a - b) / n))
    if not K:
        return None
    eff, p2, p1 = st.macro_perm_design(entries, K, higher_better=True)
    units = {e[2] for e in entries}
    return dict(tex=float(np.mean(sv["texture"])), flat=float(np.mean(sv["flat"])), effect=eff, p_one=p1, p_two=p2,
                n_patients=len(units), n_cases=n_cases, n_strata=K, n_dropped=dropped)


def read_score(p: Path):
    out = []
    with p.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out.append(dict(model=r["model"], label=r["label"], case=r["case"], agree=f(r["agree"]),
                            gt_o=f(r["gt_o"]), gt_T=f(r["gt_T"])))
    return out


def qc_summary(work: Path):
    """Gate outcome per generated input dir, from <work>/gen_stats/*.tsv (mode T) and the control rows."""
    res = {"T": [], "control_all": []}
    for p in sorted((work / "gen_stats").glob("*.tsv")):
        with p.open() as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                res.setdefault(r["mode"], []).append({k: f(v) for k, v in r.items() if k not in ("case", "mode", "shape")})
    out = {}
    for mode, rows in res.items():
        if not rows:
            continue
        ok = np.array([gate(r) for r in rows])
        med = {k: float(np.nanmedian([r[k] for r in rows])) for k in ("scr_corr", "le_corr", "fine_corr", "fine_ratio", "lp4_frac", "lp4_rel")}
        out[mode] = dict(n=len(rows), pass_frac=float(ok.mean()), **med,
                         lp4_frac_p95=float(np.nanpercentile([r["lp4_frac"] for r in rows], 95)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/scratch/paulh/texture_reliance"))
    ap.add_argument("--qc-only", action="store_true")
    a = ap.parse_args()
    if a.qc_only:
        q = qc_summary(a.work)
        for mode, v in q.items():
            print(mode, json.dumps(v))
        ok = "T" in q and q["T"]["pass_frac"] >= QC_PASS_FRAC and ("control_all" not in q or q["control_all"]["pass_frac"] <= 0.05)
        print(f"QC verdict: {'PASS' if ok else 'FAIL'} (T pass >= {QC_PASS_FRAC}, control_all pass <= 0.05)")
        return
    M = json.loads((a.work / "manifest.json").read_text())["domains"]
    res, missing = {}, []
    for dom, D in M.items():
        p = a.work / "score" / f"{dom}.tsv"
        if not p.exists():
            missing.append(dom)
            continue
        rows = read_score(p)
        res[dom] = dict(task=D["task"], group=D["group"], S=domain_stats(D, rows, "S"), G=domain_stats(D, rows, "G"))
    od = a.work / "analysis"
    od.mkdir(parents=True, exist_ok=True)
    doms = [d for d in M if d in res and res[d]["S"]]
    L = ["# Texture reliance: phase-scrambled fine texture, texture fill vs flat fill (scratch only, not for git)", "",
         f"Domains scored: {len(doms)}/19" + (f"; missing: {', '.join(missing)}" if missing else ""), ""]
    qc = qc_summary(a.work)
    L += ["## QC of perturbation T (gates per image: |scr_corr| <= 0.05, le_corr >= 0.80, fine_corr <= 0.50, "
          "lp4_frac <= 0.25, lp4_rel <= 0.15; operator passes if >= 95% of T images pass; control_all must fail)", "",
          "| mode | n images | pass | med scr_corr | med le_corr | med fine_corr | med fine_ratio | med lp4_frac | p95 lp4_frac | med lp4_rel |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, q in qc.items():
        L.append(f"| {mode} | {q['n']} | {q['pass_frac']:.3f} | {q['scr_corr']:.3f} | {q['le_corr']:.3f} | {q['fine_corr']:.3f} | "
                 f"{q['fine_ratio']:.3f} | {q['lp4_frac']:.3f} | {q['lp4_frac_p95']:.3f} | {q['lp4_rel']:.4f} |")
    if "T" in qc:
        okT = qc["T"]["pass_frac"] >= QC_PASS_FRAC
        okC = "control_all" not in qc or qc["control_all"]["pass_frac"] <= 0.05
        L.append(f"\nQC verdict: operator {'PASS' if okT else 'FAIL'}; positive control "
                 f"{'responds (fails the gates)' if okC else 'DOES NOT RESPOND'}.")
    L.append("")
    for metric, title in (("S", "Primary: sensitivity S = 1 - agreement Dice(pred orig, pred T)"),
                          ("G", "Secondary: GT Dice drop G = Dice(orig, GT) - Dice(T, GT)")):
        ps = [res[d][metric]["p_one"] if res[d][metric] else float("nan") for d in doms]
        fin = [i for i, p in enumerate(ps) if np.isfinite(p)]
        holm = dict(zip(fin, st.holm([ps[i] for i in fin])))
        L += [f"## {title}", "",
              "| domain | group | n patients | n cases | S/G texture | S/G flat | effect (tex - flat) | p one-sided | p two-sided | p Holm (19) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        tab = []
        for i, d in enumerate(doms):
            s = res[d][metric]
            if not s:
                L.append(f"| {d} | {res[d]['group']} | - | - | - | - | - | - | - | - |")
                continue
            L.append(f"| {d} | {res[d]['group']} | {s['n_patients']} | {s['n_cases']} | {s['tex']:.4f} | {s['flat']:.4f} | "
                     f"{s['effect']:+.4f} | {st.fmt_p(s['p_one'])} | {st.fmt_p(s['p_two'])} | {st.fmt_p(holm.get(i, float('nan')))} |")
            tab.append(dict(metric=metric, domain=d, task=res[d]["task"], group=res[d]["group"], **s,
                            p_holm=holm.get(i, float("nan"))))
        w, n, ties, p = st.sign_test([t["tex"] for t in tab], [t["flat"] for t in tab], higher_better=True)
        w2, n2, _, p2 = st.sign_test([t["tex"] for t in tab], [t["flat"] for t in tab], higher_better=True, two_sided=True)
        sigpos = sum(t["effect"] > 0 and t["p_one"] < 0.05 for t in tab)
        sigpos_h = sum(t["effect"] > 0 and t["p_holm"] < 0.05 for t in tab)
        signeg = sum(t["effect"] < 0 and t["p_two"] < 0.05 for t in tab)
        L += ["", f"Sign test over domains: texture > flat in {w}/{n} (ties {ties}), exact one-sided p = {st.fmt_p(p)}, "
              f"two-sided p = {st.fmt_p(p2)}. Effect > 0 with p_one < 0.05 in {sigpos}/{len(tab)} domains "
              f"({sigpos_h} after Holm); effect < 0 with p_two < 0.05 in {signeg}.", ""]
        if metric == "S":
            if p < 0.05 and sigpos >= 10:
                verdict = "SUPPORTED"
            elif sigpos <= 9 and not p < 0.05:
                verdict = "KILLED"
            else:
                verdict = "INCONCLUSIVE"
            L += [f"**Preregistered verdict (primary): {verdict}** (rule in README.md).", ""]
        for grp in ("appearance", "interface"):
            sub = [t for t in tab if t["group"] == grp]
            if sub:
                L.append(f"- {grp} (descriptive): effect > 0 in {sum(t['effect'] > 0 for t in sub)}/{len(sub)} domains, "
                         f"median effect {np.median([t['effect'] for t in sub]):+.4f}")
        L.append("")
        with (od / "domains.tsv").open("a" if metric == "G" else "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(tab[0].keys()), delimiter="\t")
            if metric == "S":
                wr.writeheader()
            wr.writerows(tab)
    (od / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
