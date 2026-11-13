#!/usr/bin/env python3
"""T04 arm roster and prediction lookup (registration 2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710..., test T04).

Arms:
  fill arms (E1's): flat, noise, texture (= Palette, the paper's 'Palette' row, ladder rung 5) -- run ids and prediction
      directories exactly as E1 (decompose_v2.run_id / decompose_v2.pred_dir on the frozen settings snapshot);
  headline arms: baseline (Plain nnU-Net), synthseg_EM, auglab_default (AugLab), pa_auglab (Palette+AugLab, val000 key
      auglabAug_v26_6_2_train050_val000) and, reported only, synthseg_noEM and srcsm (the published SRC + test-time source
      matching variant) -- run ids read from the task's *_combined_01_results.yaml, i.e. the paper's headline configs
      (make_suppl_tables.TASKS points the paper's tables at the same files).
Prediction directory of a headline arm: the same sub-path below fold<k>/ as E1's texture prediction of that item
(e.g. 'ct_fovcrop', 'sif/ct'), under <8_results_ds>/01_predictions/<model>/<tc>/<category>/<run id>/fold<k>/;
srcsm: <plain run id>/fold<k>/srcmatch/<sub> (or sif_srcmatch/<x> for 'sif/<x>').

  t04_arms.py --list OUT.tsv     (globs only; writes the roster: setting, arm, run id, item, fold, directory, n files)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

R = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
V2 = R / "benchmark/00_commun_scripts/00_04_analysis/gain_decomposition/v2"
E1 = R / "benchmark/01_commun_results/gain_decomposition/v2/e1_signed_boundary"
PAPER = R / "paper/scripts"
for p in (V2, E1):
    sys.path.insert(0, str(p))
import decompose_v2 as dv  # noqa: E402

FILL = ("flat", "noise", "texture")
HEAD = {"baseline": "baseline", "synthseg_EM": "synthseg_EM", "auglab_default": "auglab_default",
        "pa_auglab": "auglabAug_v26_6_2_train050_val000", "synthseg_noEM": "synthseg_noEM", "srcsm": "srcsm"}
CORE5 = ("baseline", "synthseg_EM", "auglab_default", "texture", "pa_auglab")   # registered core arms (texture = Palette)
SCORED = CORE5 + ("flat", "noise")
REPORTED = ("synthseg_noEM", "srcsm")
BLUR = ("synthseg_EM", "auglab_default", "pa_auglab")
UNBLUR = ("baseline", "texture")
DISPLAY = {"baseline": "Plain nnU-Net", "synthseg_EM": "SynthSeg-EM", "auglab_default": "AugLab", "texture": "Palette",
           "pa_auglab": "Palette+AugLab", "flat": "flat fill", "noise": "noise fill", "synthseg_noEM": "SynthSeg-noEM",
           "srcsm": "SRCSM"}
DATASET = {"Glioma": "brats2024-glioma", "MS": "open-ms", "Abdomen": "chaos", "Brain": "on-harmony",
           "Mandible": "toothfairy2", "Pelvis": "totalseg-pelvic", "Pancreas": "pansegdata", "Breast": "ispy2",
           "HCC": "atlas-liver-hcc"}
# E1's 16 settings (registered: 10 interface + 6 MS/Glioma; Breast and HCC are not part of T04)
SETTINGS16 = ["Abdomen_t1in", "Abdomen_t2spir", "Brain_T1w", "Brain_T2w", "Brain_dwi_ap", "Mandible_cbct", "Pelvis_ct",
              "Pelvis_mri", "Pancreas_t1wce", "Pancreas_t2w", "Glioma_t1n", "Glioma_t1c", "Glioma_t2w", "Glioma_t2f",
              "MS_flair", "MS_t1w"]


def settings():
    return yaml.safe_load(dv.SETTINGS_YAML.read_text())["settings"]


def combined_config(task: str) -> Path:
    ds = DATASET[task]
    hits = sorted((R / "benchmark/02_tasks").glob(f"*/{ds}/5_scripts_{ds}/06_evaluate/configs/*combined_01_results.yaml"))
    if len(hits) != 1:
        raise RuntimeError(f"{task}: {len(hits)} combined configs")
    return hits[0]


def run_ids(st: dict) -> dict:
    """{arm: run id} for every arm of one setting."""
    out = {a: dv.run_id(st, a) for a in FILL}
    cfg = yaml.safe_load(combined_config(st["task"]).read_text())
    mod = [m for m in cfg["modalities"] if m["name"] == st["training_contrast"]]
    if len(mod) != 1:
        raise RuntimeError(f"{st['task']} {st['training_contrast']}: {len(mod)} modalities in the combined config")
    for arm, key in HEAD.items():
        rid = mod[0]["runs"].get(key)
        if rid:
            out[arm] = rid
    return out


def _sub(it: dict) -> str | None:
    ref = it.get("ref_pred", {}).get("texture")
    return ref.split("/fold0/", 1)[1] if ref and "/fold0/" in ref else None


def pred_dir(st: dict, it: dict, arm: str, rid: str, fold: int):
    """(Path or None, how)."""
    if arm in FILL:
        p, how = dv.pred_dir(st, it, arm, fold)
        return p, how
    sub = _sub(it)
    if sub is None:
        return None, "no texture ref (item not in E1)"
    ds = it["dataset"]
    base = R / dv.sd.T / dv.sd.TASKDIR[ds] / ds / f"8_results_{ds}" / "01_predictions" / dv.sd.MODEL[st["task"]] / st["training_contrast"]
    if arm == "srcsm":
        plain = rid[: -len("_srcmatch")] if rid.endswith("_srcmatch") else rid
        subs = [f"sif_srcmatch/{sub[4:]}"] if sub.startswith("sif/") else [f"srcmatch/{sub}"]
        cands = [h for s in subs for h in glob.glob(str(base / "*" / plain / f"fold{fold}" / s))]
    else:
        cands = glob.glob(str(base / "*" / rid / f"fold{fold}" / sub))
    cands = [Path(h) for h in cands if Path(h).is_dir()]
    if len(cands) == 1:
        return cands[0], "glob"
    return None, f"absent ({len(cands)} hits)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    a = ap.parse_args()
    S = settings()
    rows = ["setting\tarm\trun_id\titem_key\tgroup\tfold\tdir\tn_files\thow"]
    for s in SETTINGS16:
        st = S[s]
        rid = run_ids(st)
        for it in st["items"]:
            if it["group"] == "in-domain":
                continue
            for arm in SCORED + REPORTED:
                for f in dv.FOLDS:
                    p, how = pred_dir(st, it, arm, rid.get(arm, ""), f) if arm in rid else (None, "no run id")
                    n = len(glob.glob(str(p / "*.nii.gz"))) if p else 0
                    rows.append(f"{s}\t{arm}\t{rid.get(arm, '')}\t{it['key']}\t{it['group']}\t{f}\t{p or 'ABSENT'}\t{n}\t{how}")
    Path(a.list).write_text("\n".join(rows) + "\n")
    print(f"[t04_arms] {len(rows) - 1} rows -> {a.list}")


if __name__ == "__main__":
    main()
