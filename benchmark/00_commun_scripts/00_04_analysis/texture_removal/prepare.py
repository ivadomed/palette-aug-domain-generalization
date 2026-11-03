#!/usr/bin/env python3
"""Check and prepare the "texture removed at test time" experiment (see tasks.yaml), on Vulcan.

Checks (any failure stops the build):
  * each task's ladder_series.json: run_keys[4] / run_keys[5] are the yaml's flat / texture RUN_IDs and their labels
    are "flat fill ..." / "... real fill ..."; the yaml items cover exactly the ladder's out-of-domain pool;
  * the ORIGINAL metrics (Dice(original), condition a) exist for both models, every OOD label, folds 0-2, with as many
    cases as the input dir holds (read with the ladder engine's own loader, ladder_ood_common._labelled_case_values);
  * checkpoint_best.pth for folds 0-2, the trainer name stored INSIDE each checkpoint (what nnU-Net's predictor uses),
    plans normalization identical for the two models of a task (one texture-removed input serves both);
  * every trainer resolves in this venv (the shared predict driver will find it).
Prepares: the new test item of every OOD input, <input dir>_texrm, as a symlink to its (to be generated) dir
<work_root>/images/<task>/<item> (chaos amos/sliver07: <work_root>/fovcrop/<dataset>/t1in/imagesTs_<item> + a link to
the original cropped labels), <work_root>/gen_jobs.tsv (02_generate.sh) and <work_root>/runs.tsv (trainer per run).
Nothing existing is modified; an unexpected existing <input>_texrm stops the run.

  .venv/bin/python prepare.py [--tasks yaml] [--check-only]          (through 01_prepare.sh: a CPU job)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import ladder_ood_common as loc  # noqa: E402

MODELS = ("flat", "texture")
FOLDS = (0, 1, 2)


def P(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else REPO / p


def ladder_ood_labels(d: dict) -> set:
    if d.get("mode") == "grouped_by_contrast":
        return {m for ms in d["ood_groups"].values() for m in ms}
    return {k for k in d["per_contrast"]["dice"] if "(in-domain)" not in k}


def ladder_sources(d: dict, task: dict):
    """(own metrics root, own ood contrasts, extra sources) in the form _labelled_case_values takes."""
    lad = P(task["ladder"])
    own_root = lad.parent.parent                      # .../<model_type>/<contrast>
    if d.get("ood_sources"):                          # cross-dataset ladder (Mandible): every source is "extra"
        return own_root, [], [Path(s) for s in d["ood_sources"]]
    extras = []
    for s in d.get("extra_ood_sources", []):
        e = dict(s)
        e["metrics_root"] = Path(e["metrics_root"])
        extras.append(e)
    return own_root, list(d.get("ood_contrasts", [])), extras


def label_of(task: dict, it: dict) -> str:
    ds = it["key"].split("__")[0]
    return it["item"] if ds == task["dataset"] else f"{ds}/{it['item']}"


def trainer_dir(run_dir: Path) -> Path:
    c = sorted(run_dir.glob("Dataset*/*__nnUNetPlans__3d_fullres"))
    if len(c) != 1:
        sys.exit(f"ERROR: expected one trainer dir under {run_dir}, found {c}")
    return c[0]


def norm_of(plans: dict):
    c = plans["configurations"]["3d_fullres"]
    sch = c["normalization_schemes"][0]
    win = None
    if sch == "CTNormalization":
        fi = plans["foreground_intensity_properties_per_channel"]["0"]
        win = (float(fi["percentile_00_5"]), float(fi["percentile_99_5"]))
    return sch, win


def check_arch(trainers: set) -> dict:
    out = {}
    try:
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    except Exception as e:  # pragma: no cover
        return {t: f"not checked ({e})" for t in trainers}
    folder = str(Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer")
    for t in sorted(trainers):
        try:
            cls = recursive_find_python_class(folder, t, "nnunetv2.training.nnUNetTrainer")
        except Exception as e:
            out[t] = f"import error: {type(e).__name__}: {e}"
            continue
        if cls is None:
            out[t] = "not resolvable in this venv"
            continue
        owner = next(k for k in cls.__mro__ if "build_network_architecture" in k.__dict__)
        out[t] = "default architecture" if owner is nnUNetTrainer else f"OVERRIDES build_network_architecture in {owner}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=HERE / "tasks.yaml")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.tasks.read_text())
    rows, gen, problems, trainers = [], [], [], set()
    import torch

    for tname, t in cfg["tasks"].items():
        d = json.loads(P(t["ladder"]).read_text())
        rk, labs = d["run_keys"], d["labels"]
        for i, m in ((4, "flat"), (5, "texture")):
            if not rk[i].endswith(t["models"][m]["run_id"]):
                problems.append(f"{tname}: ladder run_keys[{i}]={rk[i]} != {m} run {t['models'][m]['run_id']}")
        if not (labs[4].startswith("flat fill") and "real fill" in labs[5]):
            problems.append(f"{tname}: ladder rungs 4/5 are {labs[4]!r}/{labs[5]!r}")
        want = ladder_ood_labels(d)
        have = {label_of(t, it) for it in t["items"]}
        if want != have:
            problems.append(f"{tname}: yaml OOD items {sorted(have)} != ladder OOD pool {sorted(want)}")
        own_root, ood_c, extras = ladder_sources(d, t)

        # original metrics, both models
        n_in = {}
        for it in t["items"]:
            n_in[label_of(t, it)] = len(list(P(it["input"]).glob("*_0000.nii.gz")))
        for m in MODELS:
            vals = loc._labelled_case_values(own_root, ood_c, extras, rk[4 if m == "flat" else 5], "dice")
            for lab, n in n_in.items():
                got = len(vals.get(lab, {}))
                if got == 0:
                    problems.append(f"{tname}/{m}: no original metrics for {lab}")
                elif got != n:   # a case without a finite Dice is dropped by the ladder loader too
                    print(f"[warn] {tname}/{m}: original metrics for {lab}: {got} finite cases, input has {n}")
            for lab in n_in:   # every fold present
                key = rk[4 if m == "flat" else 5]
                src_root = own_root
                for e in extras:
                    r = loc._src_root(e)
                    sub = e.get("run_subdir") if isinstance(e, dict) else None
                    if lab.startswith(loc._dataset_name(r) + "/") and sub in (None, lab.split("/", 1)[1]):
                        src_root, key = r, loc._src_key(e, key)
                        break
                rd = loc.resolve_run_dir(src_root, key)
                nf = sum((rd / f"fold{k}" / "eval_all.csv").exists() for k in FOLDS)
                if nf != 3:
                    problems.append(f"{tname}/{m}/{lab}: {nf}/3 fold eval_all.csv in {rd}")

        # checkpoints, trainers, plans
        norms = {}
        for m in MODELS:
            md = t["models"][m]
            td = trainer_dir(P(md["run_dir"]))
            plans = json.loads((td / "plans.json").read_text())
            norms[m] = norm_of(plans)
            t["models"][m]["_td"] = td
            tr_dir = td.name.split("__")[0]
            for k in FOLDS:
                ck = td / f"fold_{k}" / "checkpoint_best.pth"
                if not ck.exists():
                    problems.append(f"{tname}/{m}: missing {ck}")
                    continue
                tn = torch.load(ck, map_location="cpu", weights_only=False)["trainer_name"]
                trainers.add(tn)
                if tn != tr_dir:
                    problems.append(f"{tname}/{m}: checkpoint trainer {tn} != dir trainer {tr_dir}")
            t["models"][m]["_trainer"] = tr_dir
            t["models"][m]["_dsname"] = td.parent.name
        if norms["flat"] != norms["texture"]:
            problems.append(f"{tname}: normalization differs between models {norms}")
        t["_norm"] = norms["texture"]
        print(f"[check] {tname}: {len(t['items'])} OOD items ({sum(n_in.values())} cases), norm {t['_norm']}, "
              f"trainers {t['models']['flat']['_trainer']} / {t['models']['texture']['_trainer']}")

    arch = check_arch(trainers)
    for tn, s in arch.items():
        print(f"[arch] {tn}: {s}")
        if not s.startswith(("default", "OVERRIDES")):
            problems.append(f"trainer {tn}: {s}")
    if problems:
        print("\n".join("PROBLEM: " + p for p in problems))
        sys.exit(1)
    print("[check] all checks passed")
    if a.check_only:
        return

    # ---- stage: the new test item <input>_texrm (symlink -> generated dir on scratch), generator jobs, run table ----
    work = Path(cfg["work_root"])
    for tname, t in cfg["tasks"].items():
        win = t["_norm"][1]
        for it in t["items"]:
            src = P(it["input"])
            gen_dir = work / "images" / tname / it["key"]
            gen_dir.mkdir(parents=True, exist_ok=True)
            if "/_fovcrop/" in str(src):            # chaos amos/sliver07: crop-reuse dir for fov_crop_predict_evaluate.sh
                ds = it["key"].split("__")[0]
                cdir = work / "fovcrop" / ds / t["training_contrast"]
                cdir.mkdir(parents=True, exist_ok=True)
                links = {cdir / src.name: gen_dir,
                         cdir / src.name.replace("imagesTs_", "labelsTs_"): src.parent / src.name.replace("imagesTs_", "labelsTs_")}
            else:
                links = {src.parent / (src.name + "_texrm"): gen_dir}
            for lk, tgt in links.items():
                if lk.is_symlink() and Path(lk.resolve()) == tgt.resolve():
                    continue
                if lk.exists() or lk.is_symlink():
                    sys.exit(f"ERROR: {lk} exists and is not the expected link to {tgt}")
                lk.symlink_to(tgt)
            gen.append(dict(task=tname, key=it["key"], input=str(src), output=str(gen_dir),
                            ct_lo="-" if win is None else win[0], ct_hi="-" if win is None else win[1],
                            restricted=int(bool(t.get("restricted")))))
        for m in MODELS:
            md = t["models"][m]
            rows.append(dict(task=tname, group=t["group"], training_contrast=t["training_contrast"], model=m,
                             run_id=md["run_id"], category=md["category"], trainer=md["_trainer"],
                             dataset=md["_dsname"], run_dir=str(P(md["run_dir"])),
                             items=" ".join(it["key"] for it in t["items"]),
                             n_cases=sum(len(list(P(it["input"]).glob("*_0000.nii.gz"))) for it in t["items"])))
    for name, tab in (("gen_jobs.tsv", gen), ("runs.tsv", rows)):
        with (work / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tab[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(tab)
    print(f"[stage] {len(gen)} texture-removed items linked, {len(rows)} runs -> {work}/gen_jobs.tsv, runs.tsv")

if __name__ == "__main__":
    main()
