#!/usr/bin/env python3
"""Texture-reliance experiment, step 4: predict one (domain, model, fold) on every OOD item, original AND perturbed,
in the same job, same GPU, same software stack (README.md: both conditions always come from this script, so the
agreement Dice contains no cross-machine noise).

The prediction is the shared driver's call (benchmark/00_commun_scripts/00_02_predict/predict_common.sh:
nnUNetv2_predict -c 3d_fullres -f <k> --disable_tta -chk checkpoint_best.pth) made through nnU-Net's own Python
entry point on the run's model folder (Dataset*/<trainer>__nnUNetPlans__3d_fullres), with nnUNetv2_predict's
defaults: tile step 0.5, Gaussian weighting, no mirroring. The model is loaded once per job. Outputs go to scratch:
  <work>/pred/<domain>/<model>/fold<k>/{orig,T}/<dataset>__<item>/<case>.nii.gz
Brain (on-harmony): both conditions are resampled to the native grid with the dataset's own post-item hook
(05_02_resample_predictions_to_native.py), as the benchmark's predictions are. Reruns skip finished item dirs.

  .venv/bin/python predict.py --domain Glioma_t1n --model texture --fold 0 [--work ...] [--conditions orig,T]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESAMPLE = REPO / "benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony/05_predict/05_02_resample_predictions_to_native.py"


def cases_in(d: Path, suffix: str):
    return {p.name[: -len(suffix)] for p in d.glob(f"*{suffix}")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True, choices=("flat", "texture"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--work", type=Path, default=Path("/scratch/paulh/texture_reliance"))
    ap.add_argument("--conditions", default="orig,T")
    ap.add_argument("--npp", type=int, default=4)
    ap.add_argument("--nps", type=int, default=3)
    a = ap.parse_args()
    D = json.loads((a.work / "manifest.json").read_text())["domains"][a.domain]
    md = D["models"][a.model]
    # trainer registration shims import the dataset's trainer package via NNUNET_PROJECT_ROOT (as predict_common.sh
    # exports it in every predict job); the training dataset's scripts dir also goes on sys.path like env.sh does
    os.environ["NNUNET_PROJECT_ROOT"] = str(REPO)
    ds_root = Path(md["run_dir"]).parents[5]
    sys.path.insert(0, str(ds_root / f"5_scripts_{ds_root.name}"))
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    t0 = time.time()
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                           perform_everything_on_device=True, device=torch.device("cuda"), verbose=False,
                           verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(md["trainer_dir"], use_folds=(a.fold,), checkpoint_name="checkpoint_best.pth")
    print(f"[predict] {a.domain} {a.model} fold{a.fold}: {md['run_id']} ({md['trainer']}) loaded in {time.time() - t0:.0f} s", flush=True)
    rc = 0
    for it in D["items"]:
        for cond in a.conditions.split(","):
            src = Path(it["input"] if cond == "orig" else it["perturbed"])
            out = a.work / "pred" / a.domain / a.model / f"fold{a.fold}" / cond / f"{it['dataset']}__{it['item']}"
            want = cases_in(src, "_0000.nii.gz")
            if len(want) != it["n_input"]:
                print(f"[predict] ERROR {src}: {len(want)} inputs, manifest says {it['n_input']}", flush=True)
                rc = 1
                continue
            done_flag = out / ".done"
            if done_flag.exists() and cases_in(out, ".nii.gz") >= want:
                continue
            out.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            pred.predict_from_files(str(src), str(out), save_probabilities=False, overwrite=False,
                                    num_processes_preprocessing=a.npp, num_processes_segmentation_export=a.nps)
            if it.get("native_ref"):
                r = subprocess.run([sys.executable, str(RESAMPLE), str(out), it["native_ref"]])
                if r.returncode:
                    rc = 1
                    continue
            got = cases_in(out, ".nii.gz")
            if got >= want:
                done_flag.write_text(f"{md['run_id']} {md['trainer_dir']} fold{a.fold} checkpoint_best.pth\n")
            else:
                print(f"[predict] ERROR {out}: {len(want - got)} predictions missing", flush=True)
                rc = 1
            print(f"[predict] {a.domain} {a.model} fold{a.fold} {cond} {it['label']}: {len(got)} cases, "
                  f"{time.time() - t1:.0f} s", flush=True)
    print(f"[predict] done rc={rc} in {time.time() - t0:.0f} s", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
