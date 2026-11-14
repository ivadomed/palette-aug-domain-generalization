#!/usr/bin/env python3
"""T02 step 1 (registration plans/2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...): the OOD cells
(setting x OOD domain group) of the IND/OOD table pipeline, per-cell arm Dice and readouts.

No second implementation of the table:
  * columns, IND/OOD kind and the setting rows = paper/scripts/make_suppl_ind_ood.run_one (which runs the shared
    combined_modality_summary.py through make_suppl_tables.run_setting and mirrors Palette in with mirror_with_pa);
  * a column value = aggregate_from_config.resolve_group_value on combined_modality_summary.load_combined_run
    (the call run_one makes for its own column check);
  * flat and noise fill (ladder rungs 4.5 and 4) are mirrored in exactly as Palette is (make_suppl_tables.mirror_with_pa,
    run ids from the ladder_series.json files listed in compute_dissociation_pvalues.ROWS), so their cells are the
    table's cells;
  * per-cell patient-level test of the texture step (P11): aggregate_from_config.build_design on that one group +
    stat_tests.macro_perm_design (the test the paper's tables use), two-sided.
Gates: mean of a setting's OOD cells == run_one's OOD row (1e-6) and == the printed _suppl_ind_ood.tex OOD row (0.1);
flat / texture / noise step of the OOD cells == the printed _suppl_texture_steps.tex setting row (0.1).

Outputs: <out>/cells_step1.tsv, <out>/gate_step1.tsv, <out>/settings_step1.tsv (no case ids);
<scratch>/cell_cases.json (case ids: scratch only, never the tracked tree).
Usage (Slurm only): .venv/bin/python t02_cells.py --out DIR --scratch DIR --paper-ref DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
os.environ.setdefault("PROJECT_ROOT", str(REPO))
sys.path.insert(0, str(REPO / "paper/scripts"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
import make_suppl_ind_ood as M                       # noqa: E402
import make_suppl_tables as MS                       # noqa: E402
import compute_dissociation_pvalues as C             # noqa: E402
from compare_palette_alone_all_domains import MOD    # noqa: E402
from stat_tests import macro_perm_design             # noqa: E402

AGG, CMS, MT = M.AGG, M.CMS, M.MT
PA, REF = M.PA, M.REF
ARMS7 = [k for k, _ in M.METHODS]       # SynthSeg-noEM, SynthSeg-EM, Plain, SRCSM, AugLab, Palette, Palette+AugLab
SHORT = {"synthseg_noEM": "ssnoem", "synthseg_EM": "ssem", "baseline": "plain", "srcsm": "srcsm",
         "auglab_default": "auglab", PA: "palette", REF: "pa_auglab", "flat": "flat", "noise": "noise"}
RANDOMISED = ["synthseg_EM", "auglab_default", PA, REF]      # P15's "best randomised arm" pool


def _dataset_of(metrics_dir) -> str:
    """Dataset name of a metrics dir (.../02_tasks/<task>/<dataset>/...); the path itself if not under 02_tasks."""
    parts = Path(metrics_dir).parts
    return parts[parts.index("02_tasks") + 2] if "02_tasks" in parts else Path(metrics_dir).as_posix()


def ladder_rungs() -> dict:
    """(task, modality) -> {flat, noise, real} run keys from the ladder_series.json files of the paper's texture tables."""
    out = {}
    for lab, _, rel in C.ROWS:
        task, mod = lab.split(" ", 1)
        d = json.load(open(REPO / rel))

        def pick(s):
            return d["run_keys"][next(i for i, lab_ in enumerate(d["labels"]) if s in lab_)]
        out[(task, MOD[task][mod])] = {"flat": pick("flat fill"), "noise": pick("noise fill"), "real": pick("real fill"),
                                       "ladder": rel}
    return out


def mirror_tolerant(cfg: dict, mod_name: str, rid: str, ds: str, task: str, scratch: Path):
    """make_suppl_tables.mirror_with_pa with one change: a source that lacks the rung is still mirrored (without it)
    instead of aborting the whole modality, and its index is returned. Needed because the flat/noise rungs were not
    always evaluated on in-domain-only companions; an OOD cell is then kept only if the rung scores exactly the
    baseline's cases there (checked by the caller)."""
    import copy
    from compare_palette_alone_all_domains import find_rung
    mr = str(REPO / f"benchmark/02_tasks/{task}/{ds}/8_results_{ds}/02_metrics")
    bare = rid.split("/")[-1]
    for pre in ("nnUNet_", "auglab_"):
        bare = bare[len(pre):] if bare.startswith(pre) else bare
    out = copy.deepcopy(cfg)
    m = [x for x in out["modalities"] if x["name"] == mod_name][0]
    srcs = m.get("sources") or [{"metrics_dir": m["metrics_dir"]}]
    missing = []
    for i, s in enumerate(srcs):
        md = Path(os.path.expandvars(s["metrics_dir"].replace("${METRICS_ROOT}", mr)))
        r = find_rung(md, rid)
        mirror = scratch / "mirror_dirs" / f"{ds}_{mod_name}_src{i}"
        mirror.mkdir(parents=True, exist_ok=True)
        for e in md.iterdir():
            if e.is_dir() and not (mirror / e.name).exists():
                (mirror / e.name).symlink_to(e)
        if r is None:
            missing.append(i)
        elif not (mirror / r.name).exists():
            (mirror / r.name).symlink_to(r)
        s["metrics_dir"] = str(mirror)
    if "sources" in m:
        m["sources"] = srcs
    else:
        m["metrics_dir"] = srcs[0]["metrics_dir"]
    m["runs"] = dict(m["runs"], palette_alone=bare)
    return out, missing


def ladder_cell_values(rel: str, cell_items: dict) -> dict:
    """{(rung, col): ladder per_contrast Dice x100} for the ladder groups whose (dataset, item) set equals a cell's."""
    d = json.load(open(REPO / rel))
    own = Path(rel).parts[Path(rel).parts.index("02_tasks") + 2]
    out = {}
    pc = d.get("per_contrast", {}).get("dice", {})
    for g, members in (d.get("ood_groups") or {g: [g] for g in pc}).items():
        its = {((m.split("/")[0], m.split("/")[1]) if "/" in m else (own, m)) for m in members}
        cols = [c for c, ci in cell_items.items() if ci == its]
        if len(cols) != 1 or g not in pc:
            continue
        vals = pc[g]
        sc = 100.0 if max(abs(v) for v in vals if v is not None and np.isfinite(v)) <= 1.0 else 1.0
        for rung, sub in (("flat", "flat fill"), ("noise", "noise fill"), ("real", "real fill")):
            i = next((i for i, l in enumerate(d["labels"]) if sub in l), None)
            if i is not None and vals[i] is not None:
                out[(rung, cols[0])] = vals[i] * sc
    return out


def cut_yaml(cfg: dict, mod: str, path: Path) -> Path:
    one = dict(cfg)
    one["modalities"] = [m for m in cfg["modalities"] if m["name"] == mod]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(one, sort_keys=False))
    return path


def leaf_refs(node) -> list:
    """[(column, label or None)] under a contrast_groups node, dict nodes recursed (aggregate_from_config._leaf_refs
    handles leaves and lists only)."""
    if isinstance(node, dict) and "column" not in node:
        return [r for v in node.values() for r in leaf_refs(v)]
    return AGG._leaf_refs(node)


def ds_name(metrics_dir) -> str:
    p = Path(str(metrics_dir)).parts
    return p[p.index("02_tasks") + 2] if "02_tasks" in p else str(metrics_dir)


def column_values(mods, groups, cols, key) -> dict:
    """{col: (value x100, {fold tag: value x100})} for one run key (the run_one column path)."""
    data, _ = CMS.load_combined_run(mods, key)
    out = {}
    for c in cols:
        node = groups[c] if groups else c
        v = AGG.resolve_group_value(node, data, "dice") * 100
        f = {t: x * 100 for t, x in M.fold_values(data, node, "dice").items()}
        out[c] = (v, f)
    return out


def case_scan(sources, run_id) -> dict:
    """col -> {(source index, raw group, case): set(labels)} from eval_all.csv of folds 0-2 (same prefix / rename
    semantics as aggregate_from_config.load_run_cases)."""
    out = defaultdict(lambda: defaultdict(set))
    for si, src in enumerate(sources):
        rd = AGG.resolve_run_dir(src["metrics_dir"], run_id)
        if rd is None:
            continue
        pre, ren = src.get("column_prefix", ""), src.get("column_rename", {})
        for k in (0, 1, 2):
            p = rd / f"fold{k}" / "eval_all.csv"
            if not p.exists():
                continue
            for row in csv.DictReader(p.open()):
                col = ren.get(row["group"], f"{pre}{row['group']}")
                out[col][(si, row["group"], row["case"])].add(row["label"])
    return out


def parse_ind_ood_tex(path: Path) -> dict:
    """{setting label: {IND|OOD: [7 values in METHODS order]}} from the printed tab:suppl-ind-ood-dice."""
    txt = path.read_text().split(r"\label{tab:suppl-ind-ood-dice}")[0]
    rows, cur = {}, None
    for line in txt.splitlines():
        line = line.replace(r"\rowcolor{gray!12}", "")
        if " & " not in line or not line.rstrip().endswith(r"\\"):
            continue
        cells = [c.strip() for c in line.rstrip()[:-2].split(" & ")]
        if cells[1] not in ("IND", "OOD"):
            continue
        if cells[0]:
            cur = cells[0]
        vals = []
        for c in cells[3:]:
            m = re.search(r"(\d+\.\d)", c.split(r"\,")[0])
            vals.append(float(m.group(1)) if m else None)
        rows.setdefault(cur, {})[cells[1]] = vals
    return rows


def parse_texture_steps_tex(path: Path) -> dict:
    """{setting label: (flat, texture step, noise step)} from the printed tab:texture-steps setting rows."""
    out, task = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"\\textbf\{(\w+)\} &", line.strip())
        if m:
            task = m.group(1)
            continue
        m = re.match(r"\\quad ([\w-]+)(?:\$\^\\ddagger\$)? & ([\d.]+) & \$([+-][\d.]+)[^$]*\$ \(.*?\) & \$([+-][\d.]+)", line.strip())
        if m:
            out[f"{task} {m.group(1)}"] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--paper-ref", type=Path, required=True, help="dir with _suppl_ind_ood.tex, _suppl_texture_steps.tex")
    ap.add_argument("--tasks", default="", help="comma-separated task labels (verification runs); default all")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.scratch / "work"
    (work / "mirr").mkdir(parents=True, exist_ok=True)
    pub = parse_ind_ood_tex(a.paper_ref / "_suppl_ind_ood.tex")
    tsteps = parse_texture_steps_tex(a.paper_ref / "_suppl_texture_steps.tex")
    pa_rungs = M.ladder_rungs()
    lad = ladder_rungs()

    cells, settings, gates, case_out = [], [], [], {}
    ind_cols = {}       # (task, col) -> {"setting", "vals": {arm: v}, "cases": set}
    raw_cache = {}      # (setting label, arm) -> load_combined_cases_raw dict (for P11)
    for label, ds, tdir, mods in MS.TASKS:
        if a.tasks and label not in a.tasks.split(","):
            continue
        cfg_path = next((REPO / f"benchmark/02_tasks/{tdir}/{ds}/5_scripts_{ds}/06_evaluate/configs")
                        .glob("*combined_01_results.yaml"))
        cfg = yaml.safe_load(cfg_path.read_text())
        for mod, disp in mods.items():
            slabel = f"{label} {disp}"
            rid = pa_rungs.get((label, mod))
            nfail0 = len(M.FAIL)
            s = M.run_one(slabel, ds, tdir, cfg, mod, rid, work)
            yml = (work / "mirr" if rid else work) / f"{ds}_{mod}.yaml"
            _, mods_l, groups = M.load_modalities(yml, ds, tdir)
            cols, kind = s["cols"], s["kind"]
            vals = {k: column_values(mods_l, groups, cols, k) for k in ARMS7 if mods_l[0]["runs"].get(k)}
            raws = {k: CMS.load_combined_cases_raw(mods_l, k, "dice") for k in (PA, )}
            arm_scans = {PA: case_scan(mods_l[0]["sources"], mods_l[0]["runs"][PA])} if mods_l[0]["runs"].get(PA) else {}
            note = []
            if (label, mod) in lad:
                if lad[(label, mod)]["real"] != rid:
                    note.append(f"ladder real-fill {lad[(label, mod)]['real']} != Palette run {rid}")
                for arm in ("flat", "noise"):
                    cfg_x, miss = mirror_tolerant(cfg, mod, lad[(label, mod)][arm], ds, tdir, work / arm)
                    if miss:
                        note.append(f"{arm}: rung absent in source(s) {miss}")
                    y = cut_yaml(cfg_x, mod, work / arm / f"{ds}_{mod}.yaml")
                    _, mods_x, groups_x = M.load_modalities(y, ds, tdir)
                    vals[arm] = column_values(mods_x, groups_x, cols, PA)
                    raws[arm] = CMS.load_combined_cases_raw(mods_x, PA, "dice")
                    arm_scans[arm] = case_scan(mods_x[0]["sources"], mods_x[0]["runs"][PA])
            # case scan on the un-mirrored config: baseline and Palette+AugLab must score the same cases
            _, mods_o, _ = M.load_modalities(work / f"{ds}_{mod}.yaml", ds, tdir)
            src = mods_o[0]["sources"]
            scan = case_scan(src, mods_o[0]["runs"]["baseline"])
            scan_ref = case_scan(src, mods_o[0]["runs"][REF])
            for c in cols:
                lc = M.leaf_columns(groups, c)
                b = {k for col in lc for k in scan.get(col, {})}
                r = {k for col in lc for k in scan_ref.get(col, {})}
                if b != r:
                    note.append(f"{c}: baseline scores {len(b)} cases, Palette+AugLab {len(r)}")
            cell_items = {}
            for c in cols:
                lc = M.leaf_columns(groups, c)
                b = {k for col in lc for k in scan.get(col, {})}
                cell_items[c] = {(ds_name(src[si]["metrics_dir"]), g) for (si, g, cs) in b}
                for arm, sc_ in arm_scans.items():
                    got = {k for col in lc for k in sc_.get(col, {})}
                    if got != b and arm in vals and c in vals[arm]:
                        note.append(f"{c}: {SHORT[arm] if arm in SHORT else arm} scores {len(got)} of {len(b)} baseline cases -> cell value dropped")
                        vals[arm][c] = (np.nan, {})
            if (label, mod) in lad:
                for (rung, c), lv in ladder_cell_values(lad[(label, mod)]["ladder"], cell_items).items():
                    if kind.get(c) != "OOD":
                        continue
                    mine = vals.get({"flat": "flat", "noise": "noise", "real": PA}[rung], {}).get(c, (np.nan,))[0]
                    gates.append(dict(setting=slabel, arm=f"{rung}@{c}", source="ladder_series.json per_contrast",
                                      cells_mean=mine, run_one_row=None, published=lv,
                                      ok=bool(np.isfinite(mine) and abs(mine - lv) <= 0.1 + 1e-9)))
            plain_ind = s["mean"]["dice"]["baseline"]["IND"]
            ood = [c for c in cols if kind[c] == "OOD"]
            # gates
            for k in ARMS7:
                if k not in vals or "OOD" not in s["mean"]["dice"].get(k, {}):
                    continue
                m_cells = float(np.mean([vals[k][c][0] for c in ood]))
                row = s["mean"]["dice"][k]["OOD"]
                pv = pub.get(slabel, {}).get("OOD", [None] * 7)[ARMS7.index(k)]
                gates.append(dict(setting=slabel, arm=SHORT[k], source="_suppl_ind_ood.tex OOD", cells_mean=m_cells,
                                  run_one_row=row, published=pv,
                                  ok=abs(m_cells - row) < 1e-6 and pv is not None and abs(m_cells - pv) <= 0.1 + 1e-9))
            if slabel in tsteps and "flat" in vals and "noise" in vals:
                f = float(np.mean([vals["flat"][c][0] for c in ood]))     # NaN if any OOD cell was dropped
                rl = float(np.mean([vals[PA][c][0] for c in ood]))
                n = float(np.mean([vals["noise"][c][0] for c in ood]))
                pf, pt, pn = tsteps[slabel]
                for nm, mine, p in (("flat", f, pf), ("texture_step", rl - f, pt), ("noise_step", n - f, pn)):
                    gates.append(dict(setting=slabel, arm=nm, source="_suppl_texture_steps.tex", cells_mean=mine,
                                      run_one_row=None, published=p, ok=bool(np.isfinite(mine) and abs(mine - p) <= 0.1 + 1e-9)))
            # IND columns -> specialists
            for c in cols:
                if kind[c] == "IND":
                    lc = M.leaf_columns(groups, c)
                    ind_cols[(label, c)] = {"setting": slabel,
                                            "vals": {k: vals[k][c][0] for k in ARMS7 if k in vals},
                                            "cases": {(str(src[si]["metrics_dir"]), g, cs)
                                                      for col in lc for (si, g, cs) in scan.get(col, {})}}
            t_set = tsteps.get(slabel, (None, None, None))[1]
            settings.append(dict(setting=slabel, task=label, dataset=ds, modality=mod, n_cols=len(cols),
                                 ind_cols=",".join(c for c in cols if kind[c] == "IND"), ood_cols=",".join(ood),
                                 plain_ind=plain_ind, t_published=t_set,
                                 run_one_failures=len(M.FAIL) - nfail0, notes=" | ".join(note)))
            for c in cols:
                lc = M.leaf_columns(groups, c)
                refs = leaf_refs(groups[c]) if groups else [(c, None)]
                entries = sorted({(si, g, cs, ",".join(sorted(labs)), col)
                                  for col in lc for (si, g, cs), labs in scan.get(col, {}).items()})
                items = sorted({(Path(src[si]["metrics_dir"]).as_posix(), g) for (si, g, cs, _, _) in entries})
                case_out[f"{slabel}||{c}"] = {
                    "setting": slabel, "task": label, "dataset": ds, "modality": mod, "column": c, "kind": kind[c],
                    "leaf_refs": [list(r) for r in refs],
                    "sources": [{"metrics_dir": str(x["metrics_dir"]), "column_prefix": x.get("column_prefix", ""),
                                 "column_rename": x.get("column_rename", {})} for x in src],
                    "cases": [{"source": si, "group": g, "case": cs, "labels": lb, "column": col}
                              for (si, g, cs, lb, col) in entries]}
                if kind[c] != "OOD":
                    continue
                v = {k: vals[k][c][0] for k in vals}
                row = dict(setting=slabel, task=label, dataset=ds, modality=mod, group=c,
                           items="; ".join(f"{Path(md).parts[Path(md).parts.index('02_tasks') + 2] if '02_tasks' in Path(md).parts else md}:{g}"
                                           for md, g in items),
                           n_cases=len({(si, cs) for (si, g, cs, _, _) in entries}),
                           plain_ind=plain_ind)
                for k in list(ARMS7) + ["flat", "noise"]:
                    row[f"dice_{SHORT[k]}"] = v.get(k, np.nan)
                # P11 cell test (texture step real - flat, patients, two-sided)
                p11, d11 = np.nan, np.nan
                if "flat" in raws and np.isfinite(row["dice_flat"]):
                    node = groups[c] if groups else c
                    ents, K = AGG.build_design({c: node}, raws[PA], raws["flat"])
                    if K:
                        d11, p11, _ = macro_perm_design(ents, K, True, scale=100.0)
                row["texture_step_cell_macro_delta"] = d11
                row["p_texture_step_cell"] = p11
                cells.append(row)
            print(slabel, "cells:", [c for c in cols if kind[c] == "OOD"], "notes:", note, flush=True)

    # specialist per cell: best IND arm of the setting whose IND column is this group (same task, same column)
    for row in cells:
        spec = ind_cols.get((row["task"], row["group"]))
        mine = case_out[f"{row['setting']}||{row['group']}"]
        mine_cases = {(mine["sources"][e["source"]]["metrics_dir"], e["group"], e["case"]) for e in mine["cases"]}
        if spec is None:
            row.update(spec_setting="", spec_arm="", dice_spec=np.nan, spec_same_items="")
            continue
        best = max(spec["vals"], key=lambda k: spec["vals"][k])
        # same items = same dataset, eval group and case ids; the metrics dir itself differs by training contrast
        # (<model>/<training contrast>/...), so it is reduced to the dataset name before comparing.
        same = {(_dataset_of(m), g, cs) for m, g, cs in spec["cases"]} == {(_dataset_of(m), g, cs) for m, g, cs in mine_cases}
        row.update(spec_setting=spec["setting"], spec_arm=SHORT[best], dice_spec=spec["vals"][best],
                   spec_same_items=str(same), spec_n_cases=len(spec["cases"]))

    a.scratch.mkdir(parents=True, exist_ok=True)
    json.dump(case_out, open(a.scratch / "cell_cases.json", "w"), indent=0)

    def write(path, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k)))
                                  for k in keys) + "\n")
    write(a.out / "cells_step1.tsv", cells)
    write(a.out / "settings_step1.tsv", settings)
    write(a.out / "gate_step1.tsv", gates)
    print(f"{len(cells)} OOD cells; gate rows {len(gates)}, failing {sum(not g['ok'] for g in gates)}; "
          f"run_one check failures {len(M.FAIL)}", flush=True)
    for f in M.FAIL:
        print("  run_one:", f)


if __name__ == "__main__":
    main()
