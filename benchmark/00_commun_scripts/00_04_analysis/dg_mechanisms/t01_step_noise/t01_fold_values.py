#!/usr/bin/env python3
"""T01 / T05 step 1 (registration research/dg_mechanisms/plans/2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...):
per-FOLD Dice of every arm, setting level, by the paper's own estimands. Values only (no case ids).

Part A (--part indood), the IND/OOD table path (tab:suppl-ind-ood-dice):
  * every setting is run through paper/scripts/make_suppl_ind_ood.run_one UNCHANGED (it runs the shared
    combined_modality_summary.py via make_suppl_tables.run_setting, mirrors Palette in with mirror_with_pa, applies
    meta_task_heatmap.apply_ood_exclusions + setting_level_tests.OOD_EXCLUDE, and asserts every column);
  * extra arms are added to the setting's config as extra `runs:` keys (as make_suppl_sem50_ind_ood.py does):
    sem50 (SynthSeg-EM at 50%, runs *_synthsegEM_train050_val000_20261025_111254; Spine 20260805_130846) and ginipa
    (runs *_auglab_ginipa_20261107_161425), each only when its metrics are complete in EVERY source metrics dir of the
    setting (fold0-2 eval_all.csv, same row count as Palette+AugLab) and, for ginipa, the run root's post status is
    DONE with AUDIT OK;
  * per-fold row values: run_one keeps only the std of its per-fold row values, so they are rebuilt here from the same
    module functions (load_modalities on the yaml run_one wrote, combined_modality_summary.load_combined_run,
    make_suppl_ind_ood.fold_values per column, equal-weight mean of the row's columns per fold, as run_one line ~290)
    and ASSERTED equal to run_one's own row mean and population std (1e-6).
Part B (--part ladder), the ladder path (tab:texture-steps, table aggregation, paper ladder mirror):
  * per ladder (make_per_contrast_curves.PANELS, 19 settings), every rung run key + the flat+AugLab key of
    paper/generated_results/texture_step_in_auglab_19.json; OOD strata and trees = compute_flatfill_decomposition.
    _table_specs (the paper's own strata), stratum value = ladder_ood_common.table_value; fold k = the same call on the
    value ids of fold k only (compute_flatfill_decomposition._keep_rows); setting value = equal-weight mean over strata
    (as texture_step_in_auglab.strata_values / mean_finite). ASSERTED: full value == the mirror's stored series (1e-6)
    and mean of the three fold values == full value (1e-6).
Usage (Slurm only): .venv/bin/python t01_fold_values.py --part indood|ladder --out DIR --scratch DIR
       [--settings "Glioma T1w,MS T1w"] [--extras sem50,ginipa] [--ginipa-root DIR]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
os.environ.setdefault("PROJECT_ROOT", str(REPO))
os.environ.setdefault("SCRATCH", "/scratch/paulh")
# ladder part: the paper's texture tables read TABLE-aggregated ladders from this mirror (texture_step_in_auglab_19.json
# meta); both variables are read at import by make_per_contrast_curves, so they are set before any import below.
os.environ["LADDER_TABLE_AGG"] = "1"
os.environ["LADDER_SERIES_ROOT"] = "/project/aip-jcohen/paulh/paper_inputs_20261025/labelfirst/ladders_table"
sys.path.insert(0, str(REPO / "paper/scripts"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import make_suppl_ind_ood as M  # noqa: E402
from setting_level_tests import ladder_rungs  # noqa: E402

AGG, CMS = M.AGG, M.CMS
PA, REF = M.PA, M.REF
ARMS7 = [k for k, _ in M.METHODS]
GINIPA_TS = "20261107_161425"
GINIPA_ROOT = Path(f"/scratch/paulh/ginipa_{GINIPA_TS}")
SEM50_SUF = "synthsegEM_train050_val000_20261025_111254"
SEM50_SUF_BY_DS = {"healthy-spine-tum": "synthsegEM_train050_val000_20260805_130846"}
# post/<key> of the GIN-IPA run root -> (dataset, modality)
POST_KEY = {("brats2024-glioma", "t1n"): "brats_t1n", ("brats2024-glioma", "t1c"): "brats_t1c",
            ("brats2024-glioma", "t2w"): "brats_t2w", ("brats2024-glioma", "t2f"): "brats_t2f",
            ("open-ms", "flair"): "openms_flair", ("open-ms", "t1w"): "openms_t1w",
            ("ispy2", "t1wce"): "ispy2_t1wce", ("ispy2", "t2w"): "ispy2_t2w", ("atlas-liver-hcc", "t1w"): "atlas_t1w",
            ("chaos", "t1in"): "chaos_t1in", ("chaos", "t2spir"): "chaos_t2spir",
            ("on-harmony", "T1w"): "onh_T1w", ("on-harmony", "T2w"): "onh_T2w", ("on-harmony", "dwi_ap"): "onh_dwi_ap",
            ("toothfairy2", "cbct"): "tf2_cbct", ("totalseg-pelvic", "ct"): "pelvic_ct",
            ("totalseg-pelvic", "mri"): "pelvic_mri", ("pansegdata", "t1wce"): "panseg_t1wce",
            ("pansegdata", "t2w"): "panseg_t2w"}


def nrows(p: Path) -> int:
    with open(p) as f:
        return max(0, sum(1 for _ in f) - 1)


def run_dir(md: Path, bare: str):
    for pre in ("auglab_", "nnUNet_", ""):
        if (md / f"{pre}{bare}").is_dir():
            return md / f"{pre}{bare}"
    return None


def coverage(mods_all, mod: str, bare: str) -> tuple:
    """(ok, notes): the extra run has fold0-2 eval_all.csv in EVERY source metrics dir of modality `mod`, with the same
    row count as Palette+AugLab's run there (the check of make_suppl_sem50_ind_ood.coverage, re-stated as a file test)."""
    import ladder_ood_common as LOC
    m = next(x for x in mods_all if x["name"] == mod)
    notes, ok = [], True
    for s in m["sources"]:
        md = Path(s["metrics_dir"])
        d = run_dir(md, bare)
        ref = LOC.resolve_run_dir(md, m["runs"][REF])
        for k in range(3):
            e = None if d is None else d / f"fold{k}" / "eval_all.csv"
            r = ref / f"fold{k}" / "eval_all.csv"
            if e is None or not e.is_file():
                ok = False
                notes.append(f"{md.relative_to(REPO) if md.is_relative_to(REPO) else md} fold{k}: missing")
            elif not r.is_file() or nrows(e) != nrows(r):
                ok = False
                notes.append(f"{md.relative_to(REPO) if md.is_relative_to(REPO) else md} fold{k}: rows "
                             f"{nrows(e)} vs Palette+AugLab {nrows(r) if r.is_file() else 'missing'}")
    return ok, notes


def ginipa_status(ds: str, mod: str, root: Path) -> tuple:
    """(done, text): post/<key>/POST.status last line contains DONE and POST_AUDIT.txt ends with AUDIT OK, every row ok."""
    key = POST_KEY.get((ds, mod))
    if key is None:
        return False, "no GIN-IPA run for this setting"
    st, au = root / "post" / key / "POST.status", root / "post" / key / "POST_AUDIT.txt"
    last = st.read_text().strip().splitlines()[-1] if st.is_file() and st.read_text().strip() else "no POST.status"
    if "DONE" not in last:
        return False, last
    lines = [l for l in au.read_text().splitlines() if l.strip()] if au.is_file() else []
    if not lines or lines[-1].strip() != "AUDIT OK" or any(not l.startswith("ok") for l in lines[:-1]):
        return False, f"{last}; audit not OK ({lines[-1] if lines else 'no POST_AUDIT.txt'})"
    return True, f"{last}; AUDIT OK ({len(lines) - 1} rows)"


def fold_rows(out: dict, yml: Path, ds: str, tdir: str, keys: list) -> dict:
    """{key: {"IND"|"OOD": {fold idx: value}, "cols": {col: {fold idx: value}}}} rebuilt from the module's own functions,
    asserted against run_one's row mean / population std."""
    _, mods, groups = M.load_modalities(yml, ds, tdir)
    res = {}
    for k in keys:
        if k not in out["mean"]["dice"]:
            continue
        data, _ = CMS.load_combined_run(mods, k)
        colf = {}
        for c in out["cols"]:
            node = groups[c] if groups else c
            colf[c] = {M.fold_index(t): v * 100.0 for t, v in M.fold_values(data, node, "dice").items()}
        r_out = {"cols": colf}
        for r in ("IND", "OOD"):
            cs = [c for c in out["cols"] if out["kind"][c] == r and colf[c]]
            if not cs:
                continue
            tags = sorted(set.intersection(*[set(colf[c]) for c in cs]))
            pf = {t: float(np.mean([colf[c][t] for c in cs])) for t in tags}
            mean, std = out["mean"]["dice"][k].get(r), out["std"]["dice"][k].get(r)
            if len(pf) != 3 or mean is None or abs(np.mean(list(pf.values())) - mean) > 1e-6 \
                    or abs(float(np.std(list(pf.values()))) - std) > 1e-6:
                M.fail(f"{out['label']} {k} {r}: rebuilt per-fold {pf} vs run_one mean {mean} std {std}")
            r_out[r] = pf
        res[k] = r_out
    return res


def part_indood(a):
    scratch = Path(a.scratch)
    (scratch / "mirr").mkdir(parents=True, exist_ok=True)
    extras = [e for e in a.extras.split(",") if e]
    want = {s.strip() for s in a.settings.split(",") if s.strip()} if a.settings else None
    rungs = ladder_rungs()
    result = {"settings": {}, "extras_status": {}, "failures": None}
    for label, ds, tdir, mods in M.TASKS:
        cfg_path = next((REPO / f"benchmark/02_tasks/{tdir}/{ds}/5_scripts_{ds}/06_evaluate/configs")
                        .glob("*combined_01_results.yaml"))
        cfg = yaml.safe_load(cfg_path.read_text())
        _, mods_all, _ = M.load_modalities(cfg_path, ds, tdir)
        for mod, disp in mods.items():
            slab = f"{label} {disp}"
            if want and slab not in want:
                continue
            cfg2, st = copy.deepcopy(cfg), {}
            for ex in extras:
                if ex == "sem50":
                    bare = f"{ds}_{mod}_{SEM50_SUF_BY_DS.get(ds, SEM50_SUF)}"
                    gate_ok, gtxt = True, "n/a"
                elif ex == "ginipa":
                    bare = f"{ds}_{mod}_auglab_ginipa_{GINIPA_TS}"
                    gate_ok, gtxt = ginipa_status(ds, mod, Path(a.ginipa_root))
                else:
                    raise SystemExit(f"unknown extra {ex}")
                cov_ok, notes = coverage(mods_all, mod, bare) if gate_ok else (False, [])
                st[ex] = {"run": bare, "post": gtxt, "coverage_ok": cov_ok, "coverage_notes": notes[:20],
                          "included": bool(gate_ok and cov_ok)}
                if gate_ok and cov_ok:
                    m = next(x for x in cfg2["modalities"] if x["name"] == mod)
                    m["runs"] = dict(m["runs"], **{ex: bare})
            for ex in extras:
                if ex not in dict(M.METHODS):
                    M.METHODS.append((ex, ex))
                    M.PLAIN[ex] = ex
            rid = rungs.get((label, mod))
            out = M.run_one(slab, ds, tdir, cfg2, mod, rid, scratch)
            yml = scratch / "mirr" / f"{ds}_{mod}.yaml" if rid else scratch / f"{ds}_{mod}.yaml"
            keys = ARMS7 + [ex for ex in extras if st[ex]["included"]]
            for ex in extras:
                if st[ex]["included"] and ex not in out["mean"]["dice"]:
                    M.fail(f"{slab}: extra arm {ex} injected but absent from the shared summary")
            folds = fold_rows(out, yml, ds, tdir, keys)
            result["settings"][slab] = {"task": label, "ds": ds, "mod": mod, "palette_run": rid, "cols": out["cols"],
                                        "kind": out["kind"], "desc": out["desc"],
                                        "mean": out["mean"]["dice"], "std": out["std"]["dice"], "folds": folds}
            result["extras_status"][slab] = st
            print(slab, {k: {r: round(v, 2) for r, v in out["mean"]["dice"].get(k, {}).items()} for k in keys},
                  flush=True)
    result["failures"] = list(M.FAIL)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "indood_folds.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"{len(M.FAIL)} check failure(s)")
    for f in M.FAIL:
        print("  -", f)


# ── Part B ────────────────────────────────────────────────────────────────────────────────────────────────────────────
RUNG_NAMES = ["r0_plain", "r1_kmeans", "r2_label_remap", "r3_noise", "r45_flat", "r5_real", "r6_pa_auglab"]
LABEL_FIX = {"T1WCE": "T1-CE"}


def part_ladder(a):
    import make_per_contrast_curves as mpc
    import compute_flatfill_decomposition as cfd
    LOC = cfd._loc
    aug = json.loads((REPO / "paper/generated_results/texture_step_in_auglab_19.json").read_text())
    fa_key = {s["ladder_json"]: s["keys"]["flat_auglab"] for s in aug["settings"]}
    fa_pub = {s["ladder_json"]: s["ood_dice"] for s in aug["settings"]}
    fails, result = [], {"settings": {}, "meta": {"series_root": os.environ["LADDER_SERIES_ROOT"],
                                                  "aggregation": "table", "flat_auglab_json_meta": aug["meta"]}}
    for title, grp, entries in mpc.PANELS:
        for lab, rel in entries:
            slab = f"{title} {LABEL_FIX.get(lab, lab)}"
            if a.settings and slab not in {x.strip() for x in a.settings.split(",")}:
                continue
            d = mpc.load(rel)
            if d is None:
                fails.append(f"{slab}: ladder {rel} missing")
                continue
            labs = d["labels"]
            exp = ["baseline", "+kmeans", "+label_remap", "+voronoi", "flat fill", "real fill", "+AugLab"]
            if len(labs) < 7 or not all(e in l for e, l in zip(exp, labs)) or "val100" in labs[6]:
                fails.append(f"{slab}: unexpected rung labels {labs}")
                continue
            keys = list(d["run_keys"][:7])
            names = list(RUNG_NAMES)
            if rel in fa_key:
                keys.append(fa_key[rel])
                names.append("flat_auglab")
            specs = cfd._table_specs(d, "dice", keys)
            row = {"task": title, "group": grp, "ladder_json": rel, "run_keys": dict(zip(names, keys)),
                   "strata": sorted(specs), "full": {}, "folds": {}, "series": {}}
            for nm, rk in zip(names, keys):
                full, per = [], {}
                for sname, (tree, _u, rows) in specs.items():
                    rws = rows[rk]
                    full.append(float(LOC.table_value(tree, rws)) * 100.0)
                    ids = LOC.table_weights(tree, rws)
                    for f in sorted({rid[3] for rid in ids}):
                        keep = {rid for rid in ids if rid[3] == f}
                        per.setdefault(f, []).append(float(LOC.table_value(tree, cfd._keep_rows(rws, keep))) * 100.0)
                fullv = float(np.mean([v for v in full if np.isfinite(v)]))
                pf = {f: float(np.mean([v for v in vs if np.isfinite(v)])) for f, vs in per.items()
                      if len(vs) == len(full)}
                row["full"][nm], row["folds"][nm] = fullv, pf
                if nm in RUNG_NAMES:
                    ser = float(d["dice"][RUNG_NAMES.index(nm)])
                    row["series"][nm] = ser
                    if abs(ser - fullv) > 1e-6:
                        fails.append(f"{slab} {nm}: strata value {fullv:.6f} != mirror series {ser:.6f}")
                elif abs(fa_pub[rel]["flat"] - fullv) > 1e-6:
                    fails.append(f"{slab} flat_auglab: {fullv:.6f} != texture_step_in_auglab_19.json {fa_pub[rel]['flat']:.6f}")
                if len(pf) != 3 or abs(np.mean(list(pf.values())) - fullv) > 1e-6:
                    fails.append(f"{slab} {nm}: per-fold {pf} mean != full {fullv:.6f}")
            result["settings"][slab] = row
            print(slab, {nm: [round(row["full"][nm], 2)] + [round(v, 2) for v in row["folds"][nm].values()]
                         for nm in names}, flush=True)
    result["failures"] = fails
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ladder_folds.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"{len(fails)} check failure(s)")
    for f in fails:
        print("  -", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("indood", "ladder"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--settings", default="")
    ap.add_argument("--extras", default="sem50,ginipa")
    ap.add_argument("--ginipa-root", default=str(GINIPA_ROOT))
    a = ap.parse_args()
    part_indood(a) if a.part == "indood" else part_ladder(a)


if __name__ == "__main__":
    main()
