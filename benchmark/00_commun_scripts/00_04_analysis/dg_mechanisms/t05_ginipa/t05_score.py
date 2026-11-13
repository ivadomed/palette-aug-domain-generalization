#!/usr/bin/env python3
"""T05 scoring: GIN-IPA vs Palette (R37), registration research/dg_mechanisms/plans/2026-11-12-PREDICTION_T01-T05.md
(sha256 0b9f8710...), predictions verbatim from round 3 REVIEW_T2.md section 5.

Input: indood_folds.json of t01_step_noise/t01_fold_values.py --part indood --extras ...,ginipa (GIN-IPA added as an extra
`runs:` key of the setting's combined config, only where its post status is DONE + AUDIT OK and its metrics are complete
in every source metrics dir), i.e. the make_suppl_ind_ood estimand (setting-level OOD Dice, same OOD exclusions).
Step per fold d_k = GIN-IPA_k - Palette_k (OOD); SD_pair ddof=1; theta = 2.776 SD_pair / sqrt(3) (T01 estimator).
Scoring (registered): K2 (= P03: Glioma FLAIR, Pelvis CT, Breast T1-CE, Glioma T2w) hit if diff >= 3.2; K3 (= P04 minus
MS FLAIR: MS T1w, Glioma T1-CE) hit if diff <= 0; MS FLAIR scored separately with the K3 rule; K1 (P01 members not in P03,
P04, P05: Glioma T1w, Abdomen T1in, Abdomen T2spir, Brain T1w, Spine CT, Pelvis MRI, Pancreas T2w) hit if |diff| < 3.2.
Usage: python3 t05_score.py --indood J --t01-gate GATE_TSV --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PA, REF = "palette_alone", "auglabAug_v26_6_2_train050_val000"
GROUPS = [("K2 (P03)", ["Glioma FLAIR", "Pelvis CT", "Breast T1-CE", "Glioma T2w"], "diff >= 3.2",
           lambda d: d >= 3.2),
          ("K3 (P04 minus MS FLAIR)", ["MS T1w", "Glioma T1-CE"], "diff <= 0", lambda d: d <= 0),
          ("MS FLAIR (K3 rule)", ["MS FLAIR"], "diff <= 0", lambda d: d <= 0),
          ("K1", ["Glioma T1w", "Abdomen T1in", "Abdomen T2spir", "Brain T1w", "Spine CT", "Pelvis MRI", "Pancreas T2w"],
           "|diff| < 3.2", lambda d: abs(d) < 3.2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indood", required=True)
    ap.add_argument("--t01-gate", required=True, help="T01 gate.tsv (Palette OOD level gate per setting)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    IO = json.loads(Path(a.indood).read_text())
    gate = {}
    with open(a.t01_gate) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["path"] == "indood" and r["arm"] == "Palette" and r["row"] == "OOD":
                gate[r["setting"]] = r["gate"]
    grp_of = {s: g for g, ss, _, _ in GROUPS for s in ss}
    rule_of = {g: (txt, fn) for g, _, txt, fn in GROUPS}
    rows, scored = [], {}
    for s, si in IO["settings"].items():
        st = IO["extras_status"].get(s, {}).get("ginipa", {})
        fg, fp = (si["folds"].get("ginipa") or {}).get("OOD"), (si["folds"].get(PA) or {}).get("OOD")
        fr, fe = (si["folds"].get(REF) or {}).get("OOD"), (si["folds"].get("synthseg_EM") or {}).get("OOD")
        grp = grp_of.get(s, "not in a T05 group")
        if not st.get("included") or not fg or not fp:
            post = st.get("post", "")
            rows.append([s, grp, "not scored (no GIN-IPA run)" if post.startswith("no GIN-IPA run") else
                         "pending (predict not finished)" if "predict" in post else "not scored", "", "", "", "", "", "", "", "", "", "", "",
                         (st.get("post") or "no GIN-IPA run") + ("; coverage: " + "; ".join(st.get("coverage_notes", []))
                                                                 if st.get("coverage_notes") else "")])
            continue
        ks = sorted(set(fg) & set(fp))
        d = {k: fg[k] - fp[k] for k in ks}
        m, sd = float(np.mean(list(d.values()))), float(np.std(list(d.values()), ddof=1))
        theta = 2.776 * sd / math.sqrt(3)
        hit = ""
        if grp in rule_of:
            hit = "hit" if rule_of[grp][1](m) else "miss"
            scored.setdefault(grp, []).append((s, m, hit))
        rows.append([s, grp, "scored" if grp in rule_of else "computed, not in a group",
                     f"{np.mean(list(fg.values())):.2f}", f"{np.mean(list(fp.values())):.2f}"] +
                    [f"{d.get(f'fold{k}', np.nan):+.2f}" for k in range(3)] +
                    [f"{m:+.2f}", f"{sd:.2f}", f"{theta:.2f}", hit,
                     f"{np.mean(list(fg.values())) - np.mean(list(fr.values())):+.2f}" if fr else "",
                     f"{np.mean(list(fg.values())) - np.mean(list(fe.values())):+.2f}" if fe else "",
                     f"{st.get('post')}; Palette OOD gate {gate.get(s, '?')}"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ginipa_vs_palette.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["setting", "group", "status", "ginipa_OOD", "palette_OOD", "d_fold0", "d_fold1", "d_fold2",
                    "diff_mean", "sd_pair", "theta", "hit", "ginipa_minus_palette_auglab", "ginipa_minus_synthseg_em",
                    "note"])
        w.writerows(rows)
    print("\t".join(["setting", "group", "status", "GIN", "PA", "d0", "d1", "d2", "diff", "sd", "theta", "hit"]))
    for r in rows:
        print("\t".join(map(str, r[:12])), "|", r[-1][:120])
    print()
    for g, ss, txt, _ in GROUPS:
        sc = scored.get(g, [])
        print(f"{g}: rule {txt}: hits {sum(1 for x in sc if x[2] == 'hit')}/{len(sc)} scored of {len(ss)} registered; "
              f"scored: {[(x[0], round(x[1], 2), x[2]) for x in sc]}; pending/not scored: "
              f"{[s for s in ss if s not in [x[0] for x in sc]]}")


if __name__ == "__main__":
    main()
