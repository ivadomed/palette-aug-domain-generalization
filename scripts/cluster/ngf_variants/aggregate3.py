#!/usr/bin/env python
"""Round-3 aggregation: per arm x (as_used, p1) task means (paper protocol), delta vs the same-run ctrl, paired noise estimate.
Noise = sd over scans-paired differences: for each task, the SE of (arm - ctrl) over the task's scans (paired, same seeds), propagated to mean9/dev3.
usage: python aggregate3.py [OUT_DIR]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/paulh/review_20261017/ngf_variants/round3")
ARMS = ["ctrl", "unsharp_off", "redist_off", "edgehalf", "texhalf", "structoff", "structoff_redistoff", "structoff_bluroff"]
VARIANT = __import__("os").environ.get("NGF_VARIANT", "noblur")
TASKS = ["glioma", "ms", "breast", "liver", "abdomen", "brain_healthy", "mandible", "pelvis", "pancreas"]
DEV = {"Brain": "brain_healthy", "Abdomen": "abdomen", "MS": "ms"}
frames = []
for a in ARMS:
    for f in sorted((OUT / "data").glob(f"ngf3_*__{a}.csv")):
        d = pd.read_csv(f); d = d[(d.variant == VARIANT) & (d.roi == "fg") & d.method.isin(["as_used", "p1"])]; d["arm"] = a; frames.append(d)
d = pd.concat(frames)
s = d.groupby(["arm", "method", "task", "contrast", "scan"], as_index=False).ngf_all.mean()
def tmeans(x): return x.groupby(["task", "contrast"]).ngf_all.mean().groupby("task").mean()
def summ(v): return dict(mean9=v.reindex(TASKS).mean(), Brain=v["brain_healthy"], Abdomen=v["abdomen"], MS=v["ms"])
def boot_se(arm, meth, B=500):
    # paired scan-level bootstrap of (arm - ctrl) for mean9, resampling scans within contrast
    a = s[(s.arm == arm) & (s.method == meth)].set_index(["task", "contrast", "scan"]).ngf_all
    c = s[(s.arm == "ctrl") & (s.method == meth)].set_index(["task", "contrast", "scan"]).ngf_all
    diff = (a - c).dropna().reset_index(); rng = np.random.default_rng(0); out = []
    groups = {k: g.ngf_all.values for k, g in diff.groupby(["task", "contrast"])}
    for _ in range(B):
        per = {k: rng.choice(v, len(v)).mean() for k, v in groups.items()}
        t = pd.Series(per).groupby(level=0).mean(); out.append(t.reindex(TASKS).mean())
    return float(np.std(out))
rows = []
ARMS = [a for a in ARMS if (s.arm == a).any()]
base = {m: summ(tmeans(s[(s.arm == "ctrl") & (s.method == m)])) for m in ["as_used", "p1"]}
for a in ARMS:
    r = {"config": a}
    for m, lab in (("as_used", "used"), ("p1", "synth")):
        v = summ(tmeans(s[(s.arm == a) & (s.method == m)]))
        se = boot_se(a, m) if a != "ctrl" else float("nan")
        for k, x in v.items():
            r[f"{k} {lab}"] = x; r[f"d{k} {lab}"] = x - base[m][k]
        r[f"se9 {lab}"] = se
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT / "summary3.csv", index=False)
md = ["| config | kind | mean9 | d9 (+-SE) | Brain | dBrain | Abdomen | dAbd | MS | dMS |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    for lab, kind in (("used", "as used"), ("synth", "synthesized only")):
        md.append(f"| {r['config']} | {kind} | {r[f'mean9 {lab}']:.3f} | {r[f'dmean9 {lab}']:+.3f} (+-{r[f'se9 {lab}']:.3f}) | " +
                  " | ".join(f"{r[f'{k} {lab}']:.3f} | {r[f'd{k} {lab}']:+.3f}" for k in ("Brain", "Abdomen", "MS")) + " |")
n = d[d.method == "as_used"].groupby(["task", "contrast"]).scan.nunique()
md.append(f"\nscans per contrast: {n.min()}-{n.max()}; SE = paired scan bootstrap (500) of arm-ctrl mean9, same seeds for all arms.")
(OUT / "REPORT3.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
