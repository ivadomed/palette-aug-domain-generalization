#!/usr/bin/env python3
"""Texture-reliance experiment, step 1: build the manifest from the 19 ablation ladders (see README.md).

For every <task>/<training contrast> ladder_series.json listed in DOMAINS:
  * flat model = run_keys[4] ("flat fill (rung 4.5)"), texture model = run_keys[5] ("... (real fill)"), checked;
  * out-of-domain (OOD) items and their cases = exactly the ladder engine's OOD pool
    (ladder_ood_common._labelled_case_values over the ladder's own + extra sources, cross-dataset ladders included),
    strata = the ladder's ood_groups (grouped ladders) or one stratum per item;
  * per item: the input dir the ladder's original predictions were made from (fold*/<item>/predict_from_raw_data_args
    .json, remapped to the current Vulcan layout), the GT dir next to it (labelsTs_*), the evaluated label names
    (the `label` column of the original eval_all.csv) with their ids in the model's dataset.json;
  * per model: trainer dir (Dataset*/<trainer>__nnUNetPlans__3d_fullres of the run), checkpoint_best for folds 0-2 with
    the trainer name stored in each checkpoint, plans normalization (must be equal for the two models of a domain).
Writes <work>/manifest.json, <work>/gen_jobs.tsv (one perturbation job per unique input dir and task) and
<work>/pred_jobs.tsv (one GPU job per domain x model x fold). Case ids stay on scratch (manifest is not tracked).
Any failed check stops the build.

  .venv/bin/python build_manifest.py [--work /scratch/paulh/texture_reliance] [--skip-ckpt]   (01_build.sh: CPU job)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
T = REPO / "benchmark/02_tasks"
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import ladder_ood_common as loc  # noqa: E402

FOLDS = (0, 1, 2)
# task -> (group, [(training contrast, ladder path relative to benchmark/02_tasks)])
_M = "8_results_{d}/02_metrics/{m}/{c}/ablations/ladder_series.json"
def _lad(task_dir, ds, model, c):
    return f"{task_dir}/{ds}/" + _M.format(d=ds, m=model, c=c)
DOMAINS = {
    "Glioma":   ("appearance", [(c, _lad("brain_tumor", "brats2024-glioma", "brats2024_glioma_model", c)) for c in ("t1n", "t2w", "t2f", "t1c")]),
    "MS":       ("appearance", [(c, _lad("brain_ms", "open-ms", "open_ms_model", c)) for c in ("flair", "t1w")]),
    "Breast":   ("appearance", [(c, _lad("breast_cancer", "ispy2", "ispy2_model", c)) for c in ("t1wce", "t2w")]),
    "HCC":      ("appearance", [(c, _lad("liver_hcc", "atlas-liver-hcc", "atlas_liver_hcc_model", c)) for c in ("t1w",)]),
    "Abdomen":  ("interface",  [(c, _lad("abdomen_healthy", "chaos", "chaos_model", c)) for c in ("t1in", "t2spir")]),
    "Brain":    ("interface",  [(c, _lad("brain_healthy", "on-harmony", "on_harmony_model", c)) for c in ("T1w", "T2w", "dwi_ap")]),
    "Mandible": ("interface",  [(c, _lad("mandible_healthy", "toothfairy2", "toothfairy2_model", c)) for c in ("cbct",)]),
    "Pelvis":   ("interface",  [(c, _lad("pelvis_healthy", "totalseg-pelvic", "totalseg_pelvic_model", c)) for c in ("ct", "mri")]),
    "Pancreas": ("interface",  [(c, _lad("pancreas_disease", "pansegdata", "pansegdata_model", c)) for c in ("t1wce", "t2w")]),
}
# cohorts whose license allows showing images (QC slices); everything else never appears in a figure
OPEN_IMAGE_COHORTS = {"open-ms", "chaos", "ispy2", "totalseg-pelvic", "on-harmony"}


def die(msg):
    sys.exit(f"ERROR: {msg}")


def ds_root_of(p: Path) -> Path:
    for i, part in enumerate(p.parts):
        if part.startswith("8_results_"):
            return Path(*p.parts[:i])
    die(f"no 8_results_ in {p}")


def remap(src: str) -> Path:
    """An input path recorded by the original predictions (other machine / pre-restructuring layout) -> current path."""
    p = Path(src)
    if p.is_dir():
        return p
    m = re.search(r"/([^/]+)/2_nnUNet(?:_[^/]+)?/(.+)$", src)
    if not m:
        die(f"cannot remap input path {src}")
    ds, rest = m.group(1), m.group(2)
    cands = sorted(T.glob(f"*/{ds}/2_nnUNet_{ds}/{rest}"))
    if len(cands) != 1:
        die(f"remap {src}: candidates {cands}")
    return cands[0]


def split_cat(name: str):
    for c in ("nnUNet", "auglab"):
        if name.startswith(c + "_"):
            return c, name[len(c) + 1:]
    return None, name


def pred_run_dirs(metrics_run_dir: Path):
    """Prediction run dir(s) matching a metrics run dir (same dataset, model type, training contrast, RUN_ID)."""
    root = ds_root_of(metrics_run_dir)
    rel = metrics_run_dir.relative_to(root)                  # 8_results_<ds>/02_metrics/<mt>/<tc>/.../<cat>_<rid>
    mt, tc = rel.parts[2], rel.parts[3]
    cat, rid = split_cat(metrics_run_dir.name)
    base = root / rel.parts[0] / "01_predictions" / mt / tc
    cats = [cat] if cat else ["nnUNet", "auglab"]
    return [base / c / rid for c in cats + [x for x in ("nnUNet", "auglab") if x not in cats] if (base / c / rid).is_dir()]


def item_pred_dirs(pred_run: Path, item: str):
    for sub in (item, f"{item}_fovcrop", f"sif/{item}"):
        if (pred_run / "fold0" / sub).is_dir():
            return sub
    return None


def case_of(fn: str) -> str:
    for s in ("_0000.nii.gz", ".nii.gz"):
        if fn.endswith(s):
            return fn[: -len(s)]
    return fn


def eval_label_names(metrics_run_dir: Path, item: str) -> list:
    names = set()
    for k in FOLDS:
        f = metrics_run_dir / f"fold{k}" / "eval_all.csv"
        if not f.exists():
            die(f"missing {f}")
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if (r.get("contrast") or r.get("group")) == item:
                    names.add(r["label"])
    return sorted(names)


def ladder_sources(d: dict, lad: Path):
    own_root = lad.parent.parent
    if d.get("ood_sources"):
        return own_root, [], [Path(s) for s in d["ood_sources"]]
    extras = []
    for s in d.get("extra_ood_sources", []):
        e = dict(s)
        e["metrics_root"] = Path(e["metrics_root"])
        extras.append(e)
    return own_root, list(d.get("ood_contrasts", [])), extras


def label_metrics_dir(label, own_root, extras, run_key):
    """(metrics run dir, item name inside its eval_all.csv) for one OOD label of one rung."""
    if "/" not in label:
        return loc.resolve_run_dir(own_root, run_key), label
    ds, item = label.split("/", 1)
    for e in extras:
        r = loc._src_root(e)
        if loc._dataset_name(r) != ds:
            continue
        sub = e.get("run_subdir") if isinstance(e, dict) else None
        if sub not in (None, item):
            continue
        rd = loc.resolve_run_dir(r, loc._src_key(e, run_key))
        if (rd / "fold0" / "eval_all.csv").exists() and item in loc.load_case_means(rd, "dice"):
            return rd, item
    die(f"label {label}: no source found for {run_key}")


def trainer_dir(run_dir: Path) -> Path:
    c = sorted(run_dir.glob("Dataset*/*__nnUNetPlans__3d_fullres"))
    if len(c) != 1:
        die(f"expected one trainer dir under {run_dir}, found {c}")
    return c[0]


def norm_of(plans: dict):
    c = plans["configurations"]["3d_fullres"]
    sch = c["normalization_schemes"][0]
    win = None
    if sch == "CTNormalization":
        fi = plans["foreground_intensity_properties_per_channel"]["0"]
        win = [float(fi["percentile_00_5"]), float(fi["percentile_99_5"])]
    return {"scheme": sch, "ct_window": win, "spacing": c["spacing"]}


def same_content(dirs, quick=False) -> bool:
    """True when every dir holds the same *_0000.nii.gz names with identical arrays and affines (the two models of
    Abdomen t2spir were predicted from two copies of the same crop-before-predict crops)."""
    import nibabel as nib
    import numpy as np
    names = [sorted(p.name for p in d.glob("*_0000.nii.gz")) for d in dirs]
    if any(n != names[0] for n in names[1:]):
        return False
    if quick:
        print(f"[warn] content of {dirs} not compared (--skip-ckpt)")
        return True
    for n in names[0]:
        ref = nib.load(dirs[0] / n)
        a0 = np.asarray(ref.dataobj)
        for d in dirs[1:]:
            o = nib.load(d / n)
            if o.shape != ref.shape or not np.allclose(o.affine, ref.affine, atol=1e-4) or not np.array_equal(np.asarray(o.dataobj), a0):
                return False
    print(f"[info] identical content: {[str(d) for d in dirs]} (using the first)")
    return True


ONH_TESTSET = T / "brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model/_test_set"
# GT label ids that differ from the model's ids (the dataset evaluators' own maps): amos 06_00_evaluate_amos.py ORGAN_MAP
GT_ID_MAP = {"amos": {"liver": 6, "right_kidney": 2, "left_kidney": 3, "spleen": 1}}


def gt_of(inp: Path, ds: str, item: str):
    """(GT dir on the grid the dataset's evaluator scores, native-resample reference dir or None), as the evaluators do:
    labelsTs_<x> next to imagesTs_<x>; BraTS: the raw dataset's labelsTr; HaN-Seg MR (S-I flipped): the CT GT
    labelsTs_ct_sif (06_03_eval_mandible_only.sh GT_OVERRIDE); on-harmony: _test_set/<item>/gt_native, predictions
    resampled to images_native first (05_02_resample_predictions_to_native.py, the predict driver's post-item hook)."""
    if ds == "on-harmony":
        return ONH_TESTSET / item / "gt_native", ONH_TESTSET / item / "images_native"
    if ds == "hanseg" and item == "mrt1":
        return inp.parent / "labelsTs_ct_sif", None
    if not inp.name.startswith("imagesTs"):
        die(f"input dir {inp} is not an imagesTs* dir")
    g = inp.parent / inp.name.replace("imagesTs", "labelsTs", 1)
    if not g.is_dir() and (inp.parent / "labelsTr").is_dir():
        g = inp.parent / "labelsTr"
    return g, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/scratch/paulh/texture_reliance"))
    ap.add_argument("--skip-ckpt", action="store_true", help="do not torch.load checkpoints (quick dry check)")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    manifest, problems, gen = {"domains": {}}, [], {}
    for task, (group, lads) in DOMAINS.items():
        for tc, lrel in lads:
            dom = f"{task}_{tc}"
            lad = T / lrel
            d = json.loads(lad.read_text())
            rk, labs = d["run_keys"], d["labels"]
            if not (labs[4].startswith("flat fill") and "real fill" in labs[5]):
                problems.append(f"{dom}: rungs 4/5 are {labs[4]!r}/{labs[5]!r}")
            own_root, ood_c, extras = ladder_sources(d, lad)
            vals = {m: loc._labelled_case_values(own_root, ood_c, extras, rk[i], "dice")
                    for m, i in (("flat", 4), ("texture", 5))}
            labels = sorted(set(vals["flat"]) | set(vals["texture"]))
            if d.get("mode") == "grouped_by_contrast":
                strata = {g: list(ms) for g, ms in d["ood_groups"].items()}
                want = {x for ms in strata.values() for x in ms}
                if want != set(labels):
                    problems.append(f"{dom}: ood_groups {sorted(want)} != loaded labels {labels}")
            else:
                strata = {lab: [lab] for lab in labels}
            # models
            train_root = ds_root_of(lad)
            models = {}
            for m, i in (("flat", 4), ("texture", 5)):
                rid = split_cat(rk[i].split("/")[-1])[1]
                rcand = sorted((train_root / f"8_results_{train_root.name}" / "01_predictions").glob(f"*/{tc}/*/{rid}"))
                rcand = [r for r in rcand if list(r.glob("Dataset*/*__nnUNetPlans__3d_fullres"))]
                if len(rcand) != 1:
                    problems.append(f"{dom}/{m}: run dir with trainer for {rid}: {rcand}")
                    continue
                td = trainer_dir(rcand[0])
                plans = json.loads((td / "plans.json").read_text())
                info = {"run_id": rid, "run_key": rk[i], "run_dir": str(rcand[0]), "trainer_dir": str(td),
                        "trainer": td.name.split("__")[0], "dataset": td.parent.name,
                        "labels": {k: v for k, v in json.loads((td / "dataset.json").read_text())["labels"].items()},
                        "norm": norm_of(plans)}
                for k in FOLDS:
                    ck = td / f"fold_{k}" / "checkpoint_best.pth"
                    if not ck.exists():
                        problems.append(f"{dom}/{m}: missing {ck}")
                    elif not a.skip_ckpt:
                        import torch
                        tn = torch.load(ck, map_location="cpu", weights_only=False)["trainer_name"]
                        if tn != info["trainer"]:
                            problems.append(f"{dom}/{m}: checkpoint trainer {tn} != dir trainer {info['trainer']}")
                models[m] = info
            if len(models) == 2 and models["flat"]["norm"] != models["texture"]["norm"]:
                problems.append(f"{dom}: plans normalization/spacing differ {models['flat']['norm']} vs {models['texture']['norm']}")
            if len(models) == 2 and models["flat"]["labels"] != models["texture"]["labels"]:
                problems.append(f"{dom}: label maps differ")
            # items
            items = []
            for lab in labels:
                ds = lab.split("/", 1)[0] if "/" in lab else train_root.name
                inputs, mdirs, item = set(), {}, None
                for m, i in (("flat", 4), ("texture", 5)):
                    md, item = label_metrics_dir(lab, own_root, extras, rk[i])
                    mdirs[m] = str(md)
                    for pr in pred_run_dirs(md):
                        sub = item_pred_dirs(pr, item)
                        if sub is None:
                            continue
                        for k in FOLDS:
                            aj = pr / f"fold{k}" / sub / "predict_from_raw_data_args.json"
                            if aj.exists():
                                src = json.loads(aj.read_text())["list_of_lists_or_source_folder"]
                                if isinstance(src, list):
                                    problems.append(f"{dom}/{lab}: list input in {aj}")
                                else:
                                    inputs.add(str(remap(src)))
                alts = sorted(inputs)
                if len(alts) > 1 and not same_content([Path(x) for x in alts], a.skip_ckpt):
                    problems.append(f"{dom}/{lab}: input dirs differ in content {alts}")
                    continue
                if not alts:
                    problems.append(f"{dom}/{lab}: no input dir found")
                    continue
                inp = Path(alts[0])
                gt, native_ref = gt_of(inp, ds, item)
                in_cases = {case_of(p.name) for p in inp.glob("*_0000.nii.gz")}
                gt_cases = {case_of(p.name) for p in gt.glob("*.nii.gz")} if gt.is_dir() else set()
                ns = f"{ds}|"
                lad_cases = {}
                for m in ("flat", "texture"):
                    lad_cases[m] = {k[len(ns):] if k.startswith(ns) else k for k in vals[m].get(lab, {})}
                both = lad_cases["flat"] & lad_cases["texture"]
                if lad_cases["flat"] != lad_cases["texture"]:
                    print(f"[warn] {dom}/{lab}: ladder case sets differ (flat {len(lad_cases['flat'])}, "
                          f"texture {len(lad_cases['texture'])}); using the {len(both)} common cases")
                miss = both - in_cases
                if miss:
                    problems.append(f"{dom}/{lab}: {len(miss)} ladder cases not in input dir {inp}")
                if len(both) != len(in_cases):
                    print(f"[info] {dom}/{lab}: ladder has {len(both)} finite cases, input dir {len(in_cases)} "
                          f"(all input cases are predicted; only ladder cases are scored)")
                gt_missing = len(both - gt_cases)
                names = eval_label_names(Path(mdirs["texture"]), item)
                lab_ids = models.get("texture", {}).get("labels", {})
                miss_names = [n for n in names if n not in lab_ids]
                if miss_names:
                    problems.append(f"{dom}/{lab}: evaluated labels {miss_names} not in model dataset.json {sorted(lab_ids)}")
                gkey = (task, str(inp.resolve()))
                if gkey not in gen:
                    h = hashlib.sha1(str(inp.resolve()).encode()).hexdigest()[:8]
                    gen[gkey] = dict(task=task, input=str(inp), dataset=ds, item=item,
                                     output=str(a.work / "images" / task / f"{ds}__{item}__{h}"),
                                     salt=f"{task}/{ds}/{item}", open_license=int(ds in OPEN_IMAGE_COHORTS))
                items.append(dict(label=lab, dataset=ds, item=item, input=str(inp), gt=str(gt),
                                  perturbed=gen[gkey]["output"], input_alternatives=alts[1:], metrics_dirs=mdirs, cases=sorted(both),
                                  n_input=len(in_cases), n_gt_missing=gt_missing,
                                  native_ref=str(native_ref) if native_ref else None,
                                  eval_labels={n: int(lab_ids[n]) for n in names if n in lab_ids},
                                  gt_ids={n: int(GT_ID_MAP.get(ds, {}).get(n, lab_ids[n])) for n in names if n in lab_ids}))
                print(f"[item] {dom:16s} {lab:34s} n={len(both):4d} in={len(in_cases):4d} gt_missing={gt_missing} "
                      f"labels={names} input={inp.relative_to(REPO) if inp.is_relative_to(REPO) else inp}")
            manifest["domains"][dom] = dict(task=task, group=group, training_contrast=tc, ladder=str(lad),
                                            strata=strata, models=models, items=items)
    if problems:
        print("\n".join("PROBLEM: " + p for p in problems))
        sys.exit(1)
    (a.work / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with (a.work / "gen_jobs.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(gen.values())).keys()), delimiter="\t")
        w.writeheader()
        w.writerows(gen.values())
    with (a.work / "pred_jobs.tsv").open("w") as f:
        f.write("domain\tmodel\tfold\tn_cases\n")
        for dom, D in manifest["domains"].items():
            n = sum(it["n_input"] for it in D["items"])
            for m in ("flat", "texture"):
                for k in FOLDS:
                    f.write(f"{dom}\t{m}\t{k}\t{n}\n")
    nd = len(manifest["domains"])
    print(f"[manifest] {nd} domains, {len(gen)} perturbation inputs, {sum(len(D['items']) for D in manifest['domains'].values())} "
          f"items -> {a.work}/manifest.json, gen_jobs.tsv, pred_jobs.tsv")
    if nd != 19:
        die(f"expected 19 training domains, got {nd}")


if __name__ == "__main__":
    main()
