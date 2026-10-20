#!/usr/bin/env python3
"""
Does pooling BraTS-SSA 2024 into the Glioma task change the PA-alone verdict? (review 2026-10-20)

Builds SCRATCH-only copies of the Glioma combined config (the headline file is never touched; every
output_dir points under $SCRATCH/review_20261020/brats_ssa/analysis):
  without  = the headline brats_combined_01_results.yaml + run "palette_alone" (PA alone = the ladder's
             real-fill rung, val000), mirrored with symlinks exactly like
             paper/scripts/compare_palette_alone_per_task.py does;
  with     = the same, each training modality gets a second source (BraTS-SSA metrics, column_prefix
             "ssa_"), and contrast_groups {t1n: [t1n, ssa_t1n], t1c: [...], t2w: [...], t2f: [...]}: cases
             pooled by TRUE contrast into the existing 4 columns (pansegdata/ispy2 convention);
  ssa_only = BraTS-SSA alone (the 4 SSA contrasts, all 4 training families).
For each: the shared combined_modality_summary.py tables (per-contrast columns + `all`), and in-process,
with the SAME functions (combined_modality_summary.significance_column = build_design +
macro_perm_design + Holm, patient-level units):
  * task value per method (= meta_task_heatmap.task_method_value, the tab:meta estimand);
  * one-sided "ref better" and "competitor better" Holm p, ref in {palette_alone, PA+Auglab val000},
    family A = the 5 competitors (+ the other PA arm reported raw), family B = every other run in the
    config (the family behind the 0.0022 of the 2026-10-20 reverse test);
  * the OOD-only variant (meta_task_heatmap.apply_ood_exclusions: each training modality's model
    loses its own contrast group, SSA leaf included).
Validation: 'without' must reproduce the tab:meta Glioma values (43.7 / 44.8 / 45.9 ...) and the
reverse p 0.0022 (Auglab ahead of PA alone) before 'with' is trusted.

Usage (CPU job): .venv/bin/python scripts/cluster/brats_ssa_pa_alone/analysis.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "benchmark/00_commun_scripts/00_03_evaluate"
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(EV))
from stat_tests import fmt_p, holm, macro_perm_design  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EV / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CMS = _load("combined_modality_summary")
MT = _load("meta_task_heatmap")
AGG = CMS._agg

OUT = Path(os.environ["SCRATCH"]) / "review_20261020/brats_ssa/analysis"
GM = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model"
SM = REPO / "benchmark/03_archive/brats-ssa2024/8_results_brats-ssa2024/02_metrics/brats2024_glioma_model"
HEAD = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/brats_combined_01_results.yaml"
MODS = ("t1n", "t2w", "t2f", "t1c")
CONTRASTS = ("t1c", "t1n", "t2f", "t2w")
PA, PAA, PAA100 = "palette_alone", "auglabAug_v26_6_2_train050_val000", "auglabAug_v26_6_2_train050_val100"
COMP5 = ["baseline", "synthseg_noEM", "synthseg_EM", "srcsm", "auglab_default"]
NAMES = {"baseline": "Plain nnU-Net", "synthseg_noEM": "SynthSeg-noEM", "synthseg_EM": "SynthSeg-EM",
         "srcsm": "SRCSM (srcmatch)", "auglab_default": "Auglab", PA: "PA alone", PAA: "PA+Auglab",
         PAA100: "PA+Auglab val100"}
ORDER = ["baseline", "synthseg_noEM", "synthseg_EM", "srcsm", "auglab_default", PA, PAA, PAA100]


def pa_alone_rid(mod):
    d = json.loads((GM / mod / "ablations/ladder_series.json").read_text())
    i = next(i for i, l in enumerate(d["labels"]) if "real fill" in l)
    return d["run_keys"][i].split("/")[-1]


def mirror(src: Path, dst: Path, extra: Path | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    for e in (src.iterdir() if src.is_dir() else []):
        if e.is_dir() and not (dst / e.name).exists():
            (dst / e.name).symlink_to(e)
    if extra is not None and not (dst / extra.name).exists():
        (dst / extra.name).symlink_to(extra)
    return dst


def build_configs():
    head = yaml.safe_load(HEAD.read_text())
    mdir = OUT / "mirror"
    cfgs = {}
    base = copy.deepcopy(head)
    with_ = copy.deepcopy(head)
    ssa = copy.deepcopy(head)
    for c in (base, with_, ssa):
        c.pop("contrast_groups", None)
    for i, mod in enumerate(MODS):
        rid = pa_alone_rid(mod)
        rung = next(p for p in (GM / mod / "ablations" / rid, GM / mod / "ablations" / f"auglab_{rid}",
                                GM / mod / rid, GM / mod / f"auglab_{rid}") if p.is_dir())
        nat = mirror(GM / mod, mdir / f"native_{mod}", rung)
        ss = mirror(SM / mod, mdir / f"ssa_{mod}")
        for c in (base, with_, ssa):
            m = next(x for x in c["modalities"] if x["name"] == mod)
            m["runs"] = dict(m["runs"], **{PA: rid})
        mb = next(x for x in base["modalities"] if x["name"] == mod)
        mb["metrics_dir"] = str(nat)
        mw = next(x for x in with_["modalities"] if x["name"] == mod)
        mw.pop("metrics_dir", None)
        mw["sources"] = [{"metrics_dir": str(nat)}, {"metrics_dir": str(ss), "column_prefix": "ssa_"}]
        ms = next(x for x in ssa["modalities"] if x["name"] == mod)
        ms["metrics_dir"] = str(ss)
    with_["contrast_groups"] = {c: [c, f"ssa_{c}"] for c in CONTRASTS}
    base["title"] = "Glioma, headline config + PA alone (SCRATCH copy, no BraTS-SSA)"
    with_["title"] = "Glioma + BraTS-SSA pooled by true contrast (SCRATCH copy)"
    ssa["title"] = "BraTS-SSA 2024 alone, all 4 brats2024-glioma training families (SCRATCH copy)"
    for name, c in (("without", base), ("with", with_), ("ssa_only", ssa)):
        for ref in (PA, PAA):
            cc = copy.deepcopy(c)
            cc["ref"] = ref
            tag = f"{name}_ref_{'pa_alone' if ref == PA else 'pa_auglab'}"
            cc["output_dir"] = str(OUT / "tables" / tag)
            cc["output_prefix"] = tag
            p = OUT / "configs" / f"{tag}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(cc, sort_keys=False))
            cfgs[tag] = (p, cc)
    return cfgs


def modalities_of(cfg):
    mods = []
    for m in cfg["modalities"]:
        srcs = m.get("sources") or [{"metrics_dir": m["metrics_dir"]}]
        mods.append({"name": m["name"], "runs": m["runs"],
                     "sources": [{**s, "metrics_dir": Path(s["metrics_dir"])} for s in srcs]})
    return mods, cfg.get("contrast_groups")


def column_values(mods, groups, key, metric):
    data, _ = CMS.load_combined_run(mods, key)
    if groups:
        return {g: AGG.resolve_group_value(node, data, metric) for g, node in groups.items()}
    return {c: AGG.cross_fold_class_mean(data, metric, c) for c in sorted(data.get(metric, {}))}


def tests(mods, groups, ref, family, metric):
    cols = sorted({c for k in [ref] + family for c in CMS.load_combined_run(mods, k)[0].get(metric, {})})
    kw = dict(contrast_groups=groups)
    win = CMS.significance_column([ref] + family, ref, cols, metric, mods, **kw)
    lose = CMS.significance_column([ref] + family, ref, cols, metric, mods, reverse=True, **kw)
    return {k: (win[k], lose[k]) for k in family}


def analyse(tag, cfg, ood: bool):
    mods, groups = modalities_of(copy.deepcopy(cfg))
    ref = cfg["ref"]
    notes = []
    if ood:
        notes = MT.apply_ood_exclusions({"name": tag}, cfg, mods, groups, ref)
    keys = [k for k in ORDER if any(k in m["runs"] for m in mods)]
    res = {"tag": tag, "ood": ood, "ref": ref, "notes": notes, "value": {}, "cols": {}, "famA": {}, "famB": {}, "pair": {}}
    for metric in ("dice", "hd95"):
        res["value"][metric] = {k: MT.task_method_value(mods, k, metric, groups) for k in keys}
        res["cols"][metric] = {k: column_values(mods, groups, k, metric) for k in keys}
        other = PAA if ref == PA else PA
        res["famA"][metric] = tests(mods, groups, ref, COMP5, metric)
        res["pair"][metric] = tests(mods, groups, ref, [other], metric)
        res["famB"][metric] = tests(mods, groups, ref, [k for k in keys if k != ref], metric)
    return res


def fmt_v(v, metric):
    return "--" if v is None or not np.isfinite(v) else (f"{100 * v:.1f}" if metric == "dice" else f"{v:.1f}")


def render(r):
    L = [f"### {r['tag']} | {'OOD only (training contrast excluded)' if r['ood'] else 'all contrasts (tab:meta estimand)'} | ref = {NAMES[r['ref']]}", ""]
    if r["notes"]:
        L += ["OOD exclusions: " + "; ".join(r["notes"]), ""]
    for metric in ("dice", "hd95"):
        keys = list(r["value"][metric])
        cols = sorted({c for k in keys for c in r["cols"][metric][k]})
        L += [f"**{metric.upper()}**", "",
              "| method | " + " | ".join(cols) + " | task | p ref better (Holm, 5 comp.) | p comp. better (Holm, 5 comp.) | p ref better (Holm, all) | p comp. better (Holm, all) |",
              "|" + "---|" * (len(cols) + 6)]
        for k in keys:
            a = r["famA"][metric].get(k) or r["pair"][metric].get(k)
            a_note = "" if k in r["famA"][metric] else " (single test)"
            b = r["famB"][metric].get(k)
            cells = [fmt_v(r["cols"][metric][k].get(c), metric) for c in cols]
            pa = ("", "") if a is None else (fmt_p(a[0]) + a_note, fmt_p(a[1]) + a_note)
            pb = ("", "") if b is None else (fmt_p(b[0]), fmt_p(b[1]))
            L.append(f"| {NAMES[k]}{' (ref)' if k == r['ref'] else ''} | " + " | ".join(cells)
                     + f" | {fmt_v(r['value'][metric][k], metric)} | {pa[0]} | {pa[1]} | {pb[0]} | {pb[1]} |")
        L.append("")
    return "\n".join(L)


def main():
    cfgs = build_configs()
    only = sys.argv[1:]          # optional tag substrings, e.g. "without" for the validation pass
    if only:
        cfgs = {t: v for t, v in cfgs.items() if any(o in t for o in only)}
    env = dict(os.environ, PROJECT_ROOT=str(REPO))
    for tag, (p, _) in cfgs.items():
        r = subprocess.run([sys.executable, str(EV / "combined_modality_summary.py"), str(p)],
                           env=env, capture_output=True, text=True)
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        (OUT / "tables" / f"{tag}.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            print(f"!! combined_modality_summary failed for {tag}:\n{r.stderr[-1500:]}")
    results = []
    for tag, (_, cfg) in cfgs.items():
        for ood in (False, True):
            results.append(analyse(tag, cfg, ood))
            print("done", tag, ood, flush=True)
    suffix = ("_" + "_".join(only)) if only else ""
    (OUT / f"results{suffix}.json").write_text(json.dumps(results, indent=1, default=float))
    txt = ["# BraTS-SSA pooled into Glioma: PA alone vs competitors (generated by "
           "scripts/cluster/brats_ssa_pa_alone/analysis.py)", "",
           "Dice in %, HD95 in mm; p = one-sided sign-flip (patient-level units, macroΔ design of the "
           "tables' own sig. column), Holm within the family named in the header.", ""]
    txt += [render(r) for r in results]
    (OUT / f"RESULTS{suffix}.md").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
