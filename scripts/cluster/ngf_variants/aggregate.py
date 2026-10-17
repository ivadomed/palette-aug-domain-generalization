#!/usr/bin/env python
"""Aggregate the PA-variant NGF CSVs like paper/scripts/make_ngf_methods.py (variant noblur, ROI fg, draws -> scan mean, mean over
a task's contrasts of the per-contrast scan means, then over the 9 tasks) and put the paper's reference rows next to them.
usage: python aggregate.py [OUT_DIR]   -> OUT_DIR/REPORT.md, OUT_DIR/summary.csv
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/paulh/review_20261017/ngf_variants")
REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "paper/generated_results/ngf_all_methods/ngf_per_draw.csv.gz"
TASKS = ["glioma", "ms", "breast", "liver", "abdomen", "brain_healthy", "mandible", "pelvis", "pancreas"]
NAMES = {"glioma": "Glioma", "ms": "MS", "breast": "Breast", "liver": "HCC", "abdomen": "Abdomen", "brain_healthy": "Brain",
         "mandible": "Mandible", "pelvis": "Pelvis", "pancreas": "Pancreas"}
DEV = ["brain_healthy", "abdomen", "ms"]
ROWS = [("ctrl", "ctrl", "ctrl_p1"), ("p075", "p075", "p075_p1"), ("edge2x", "edge2x", "edge2x_p1"), ("alphawide", "alphawide", "alphawide_p1"),
        ("paper PA", "ours_train050", None), ("paper PA (no Auglab)", "palette_train050", "palette"),
        ("paper Auglab", "auglab_default", None), ("paper flat fill", "flatfill_train050", "flatfill")]


def tm(df, method):
    d = df[(df.variant == "noblur") & (df.roi == "fg") & (df.method == method)]
    if d.empty:
        return None
    s = d.groupby(["task", "contrast", "scan"], as_index=False).ngf_all.mean()
    return s.groupby(["task", "contrast"]).ngf_all.mean().groupby("task").mean()


new = pd.concat([pd.read_csv(f) for f in sorted((OUT / "data").glob("ngfv_*.csv"))])
old = pd.read_csv(PAPER)
rows, recs = [], []
for label, used, forced in ROWS:
    src = new if label in ("ctrl", "p075", "edge2x", "alphawide") else old
    for kind, key in (("as used", used), ("synthesized only", forced)):
        if key is None:
            continue
        v = tm(src, key)
        if v is None:
            continue
        r = {"config": label, "kind": kind, **{NAMES[t]: v.get(t) for t in TASKS}}
        r["mean9"] = v.reindex(TASKS).mean(); r["dev3"] = v.reindex(DEV).mean()
        recs.append(r)
df = pd.DataFrame(recs)
df.to_csv(OUT / "summary.csv", index=False)
cols = [NAMES[t] for t in TASKS]
md = ["# NGF of the PA round-2 variants (noblur, ROI fg eroded x3; paper protocol)", "",
      "N = per-contrast task means over 20 scans x 5 draws; `as used` = training config; `synthesized only` = PA transform forced to p=1 "
      "(variants: whole chain with PA p=1; paper rows: the paper's forced keys palette / flatfill, no Auglab chain). dev3 = Brain, Abdomen, MS.", "",
      "| config | kind | " + " | ".join(cols) + " | mean9 | dev3 |", "|" + "---|" * (len(cols) + 4)]
for r in recs:
    md.append(f"| {r['config']} | {r['kind']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['mean9']:.3f} | {r['dev3']:.3f} |")
n = new[(new.variant == "noblur") & (new.roi == "fg")].groupby(["contrast", "method"]).scan.nunique().unstack().min().min()
md += ["", f"min scans per (contrast, method) in the new data: {n}"]
(OUT / "REPORT.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
