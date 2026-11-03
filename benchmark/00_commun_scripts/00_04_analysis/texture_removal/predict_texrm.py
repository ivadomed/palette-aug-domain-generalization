#!/usr/bin/env python3
"""Predict one task's OOD items through the SHARED predict drivers, either condition:
  --condition texrm : the texture-removed test item <item>_texrm -> <run dir>/fold{k}/texrm/<item>/
  --condition orig  : the ORIGINAL inputs, re-predicted on the same machine -> <run dir>/fold{k}/origts/<item>/
(both conditions on one machine and software stack; the cluster-made originals fold{k}/<item>/ stay untouched and
serve only as a sanity check of the re-predicted ones).

No prediction logic here: for each model (flat / texture) and each dataset holding OOD items, this calls that
dataset's own predict common shim (-> benchmark/00_commun_scripts/00_02_predict/predict_common.sh: nnUNetv2_predict
-c 3d_fullres -f <k> --disable_tta -chk checkpoint_best.pth, the dataset's own PREDICT_EXTRA_FLAGS, folds 0 1 2) with
the run's RUN_ID / trainer / category and the driver's input-perturbation hooks PREDICT_INPUT_SUFFIX=_texrm,
PREDICT_OUTPUT_SUBDIR=texrm -> <run dir>/fold{k}/texrm/<item>/ (Mandible: texrm/sif/<item>). The trainer is read from
the run dir (Dataset*/<trainer>__nnUNetPlans__3d_fullres), never from a method wrapper.

Runs on Vulcan (Slurm) or tassan (source scripts/cluster/tassan_env.sh first: set_slot backend, folds wrap onto the
two slot/GPU pairs). Abdomen's amos/sliver07 crop-before-predict crops are the test item <item>_fovcrop (+ _texrm),
predicted by amos/sliver07's own cross-mode shim; 05_evaluate.sh scores them with fov_crop_predict_evaluate.sh.

  python predict_texrm.py <task> [--condition texrm|orig] [--models flat,texture] [--folds all|0|1|2] [--dry-run]
                          [--status]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
T = "benchmark/02_tasks"

# per task: list of (dataset scripts dir, env file sourced first, shim, items, extra exports, input suffix, output subdir)
SUF, SUB = "_texrm", "texrm"
GROUPS = {
    "MS": [(f"{T}/brain_ms/open-ms/5_scripts_open-ms", "env.sh", "05_01_predict_common.sh", "t1w t2w", "", SUF, SUB)],
    "Glioma": [(f"{T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma", "env.sh", "05_01_predict_common.sh",
                "t1c t2w t2f", "TRAINING_CONTRAST=t1n", SUF, SUB)],
    "Pelvis": [(f"{T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic", "env.sh", "05_01_predict_common.sh",
                "mri", "DATASET_ID=130", SUF, SUB)],
    "Brain": [(f"{T}/brain_healthy/on-harmony/5_scripts_on-harmony", "env.sh", "05_01_predict_common.sh",
               "T2w bold dwi_ap epi_ap gre_echo1_mag", "", SUF, SUB)],
    "Mandible": [(f"{T}/mandible_healthy/hanseg/5_scripts_hanseg", "env.sh", "05_01_predict_common.sh", "ct mrt1",
                  "DATASET_ID=${TF2_DATASET_ID}", "_sif" + SUF, SUB + "/sif"),
                 (f"{T}/mandible_healthy/pddca/5_scripts_pddca", "env.sh", "05_01_predict_common.sh", "ct",
                  "DATASET_ID=${TF2_DATASET_ID}", "_sif" + SUF, SUB + "/sif")],
    "Pancreas": [(f"{T}/pancreas_disease/pansegdata/5_scripts_pansegdata", "env.sh", "05_01_predict_common.sh", "t2w",
                  "DATASET_ID=${DATASET_ID_T1WCE}", SUF, SUB),
                 (f"{T}/pancreas_disease/totalsegmri-pancreas/5_scripts_totalsegmri-pancreas", "env.sh",
                  "05_01_predict_pansegdata_common.sh", "t2like t1gre",
                  "PANSEG_TRAINING_CONTRAST=t1wce PANSEG_DATASET_ID=150", SUF, SUB),
                 (f"{T}/pancreas_disease/msd-pancreas/5_scripts_msd-pancreas", "env.sh",
                  "05_01_predict_pansegdata_common.sh", "ct", "PANSEG_TRAINING_CONTRAST=t1wce PANSEG_DATASET_ID=150",
                  SUF, SUB)],
    "Breast": [(f"{T}/breast_cancer/ispy2/5_scripts_ispy2", "env.sh", "05_20_predict_common.sh", "t2w",
                "DATASET_ID=${DATASET_ID_T1WCE}", SUF, SUB),
               (f"{T}/breast_cancer/duke-breast-mri/5_scripts_duke-breast-mri", "env.sh",
                "05_01_predict_ispy2_common.sh", "precontrast_uniap", "ISPY2_TRAINING_CONTRAST=t1wce ISPY2_DATASET_ID=100",
                SUF, SUB),
               (f"{T}/breast_cancer/ispy1/5_scripts_ispy1", "env.sh", "05_01_predict_ispy2_common.sh", "precontrast",
                "ISPY2_TRAINING_CONTRAST=t1wce ISPY2_DATASET_ID=100", SUF, SUB),
               (f"{T}/breast_cancer/acrin6698/5_scripts_acrin6698", "env.sh", "05_01_predict_ispy2_common.sh",
                "dwi_uniap", "ISPY2_TRAINING_CONTRAST=t1wce ISPY2_DATASET_ID=100", SUF, SUB)],
    "HCC": [(f"{T}/liver_hcc/lld-mmri-hcc/5_scripts_lld-mmri-hcc", "env.sh", "05_01_predict_common.sh",
             "t2wi dwi inphase outphase", "", SUF, SUB)],
    "Abdomen": [(f"{T}/abdomen_healthy/chaos/5_scripts_chaos", "env.sh", "05_01_predict_common.sh", "t1out t2spir ct",
                 "", SUF, SUB),
                # crop-before-predict crops (test item <item>_fovcrop, the inputs of the ladder's fov_crop predictions)
                (f"{T}/abdomen_healthy/amos/5_scripts_amos", "env.sh", "05_01_predict_common.sh", "ct mri", "",
                 "_fovcrop" + SUF, SUB),
                (f"{T}/abdomen_healthy/sliver07/5_scripts_sliver07", "env.sh", "05_01_predict_common.sh", "ct", "",
                 "_fovcrop" + SUF, SUB)],
}


def trainer_of(run_dir: Path) -> str:
    c = sorted(run_dir.glob("Dataset*/*__nnUNetPlans__3d_fullres"))
    if len(c) != 1:
        sys.exit(f"ERROR: expected one trainer dir under {run_dir}, found {c}")
    return c[0].name.split("__")[0]


def pred_root_of(scripts_dir: str, run_dir: Path, task: dict) -> Path:
    """Where the shim writes: <dataset>/8_results_<dataset>/01_predictions/<model_type>/<contrast>/<category>/<RUN_ID>."""
    ds_root = REPO / scripts_dir.rsplit("/", 1)[0]
    own = run_dir.parents[3]                                    # .../01_predictions of the training dataset
    rel = run_dir.relative_to(own)                              # <model_type>/<contrast>/<category>/<RUN_ID>
    return ds_root / f"8_results_{ds_root.name}" / "01_predictions" / rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--models", default="flat,texture")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--condition", choices=("texrm", "orig"), default="texrm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="count predictions vs inputs, run nothing")
    a = ap.parse_args()
    cfg = yaml.safe_load((HERE / "tasks.yaml").read_text())
    t = cfg["tasks"][a.task]
    rc = 0
    for m in a.models.split(","):
        md = t["models"][m]
        run_dir = REPO / md["run_dir"]
        tr = trainer_of(run_dir)
        for sdir, envf, shim, items, extra, suf, sub in GROUPS[a.task]:
            if a.condition == "orig":                    # same input minus "_texrm", output namespace origts
                suf, sub = suf[: -len(SUF)], sub.replace(SUB, "origts", 1)
            if a.status:
                pr = pred_root_of(sdir, run_dir, t)
                for it in items.split():
                    nin = None
                    for x in t["items"]:
                        if x["item"] == it and x["key"].split("__")[0] == Path(sdir).name.replace("5_scripts_", ""):
                            nin = len(list((REPO / (x["input"] + (SUF if a.condition == "texrm" else "")))
                                           .glob("*_0000.nii.gz")))
                    got = [len(list((pr / f"fold{k}" / sub / it).glob("*.nii.gz"))) for k in (0, 1, 2)]
                    ok = nin is not None and all(g == nin for g in got)
                    print(f"{a.task:9s} {m:8s} {Path(sdir).name[10:]:22s} {it:18s} in={nin} folds={got} "
                          f"{'OK' if ok else 'INCOMPLETE'}")
                    rc |= not ok
                continue
            cmd = (f"set -euo pipefail; cd {shlex.quote(str(REPO))}; source {sdir}/00_utils/{envf}; "
                   f"export METHOD={a.condition}_{m} TRAINER={tr} CATEGORY={md['category']} "
                   f"PREDICT_INPUT_SUFFIX={suf} PREDICT_OUTPUT_SUBDIR={sub} {extra}; "
                   f"source {sdir}/05_predict/{shim} {md['run_id']} {a.folds} {items}")
            print(f"[predict_texrm] {a.task}/{m} {Path(sdir).name}: {items}\n  bash -c {shlex.quote(cmd)}", flush=True)
            if not a.dry_run:
                r = subprocess.run(["bash", "-c", cmd], env=os.environ.copy())
                if r.returncode:
                    print(f"[predict_texrm] FAILED rc={r.returncode}", flush=True)
                    rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
