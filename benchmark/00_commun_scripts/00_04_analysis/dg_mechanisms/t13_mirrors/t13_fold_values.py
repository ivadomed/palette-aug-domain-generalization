#!/usr/bin/env python3
"""T13 step 1 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...):
per-fold IND / OOD Dice of the two checkpoint mirrors of every Palette+AugLab run, setting level.

Reuse (no second implementation): t01_fold_values (DG-mechanisms T01) is imported unchanged. Every setting runs through
paper/scripts/make_suppl_ind_ood.run_one (shared combined_modality_summary + meta_task_heatmap exclusions); the val100 run is
added to the setting's config as an extra `runs:` key, exactly as T01 adds sem50 / ginipa, and only when its metrics are
complete in EVERY source metrics dir (t01_fold_values.coverage: fold0-2 eval_all.csv, same row count as val000). Per-fold
row values: t01_fold_values.fold_rows (asserted equal to run_one's row mean and population std, 1e-6).

Which val100 run (fixed before any value was read):
  mirror      RUN_ID of the paper's val000 run with 'val000' -> 'val100' (the DualVal trainer writes both mirrors of one
              training under the same RUN_ID stem). Checked further on disk: trainer class name and md5 of
              checkpoint_final.pth of folds 0-2 in both run dirs (identical file = one training).
  config      the val100 run named in the task's combined config, when it is NOT the mirror (a separate training run):
              reported as a separate, unscored block.
Usage (Slurm only): .venv/bin/python t13_fold_values.py --out DIR --scratch DIR
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t01_step_noise"))
import t01_fold_values as T1  # noqa: E402  (sets PROJECT_ROOT / sys.path for make_suppl_ind_ood)
import yaml  # noqa: E402

M = T1.M
REF, PA = M.REF, M.PA
V100_CFG = "auglabAug_v26_6_2_train050_val100"
K_MIRR, K_CFG = "val100_mirror", "val100_config"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_info(ds: str, mod: str, run: str) -> dict:
    """{dirs, trainer, folds: {k: {final, size, inode}}} from the run's training dir under 01_predictions (where the repo
    keeps checkpoints); empty when the training dir is not on this cluster."""
    hits = sorted(glob.glob(str(REPO / f"benchmark/02_tasks/*/{ds}/8_results_{ds}/01_predictions/*/{mod}/*/{run}/Dataset*/*")))
    hits = [h for h in hits if Path(h).is_dir()]
    out = {"dirs": hits, "trainer": sorted({Path(h).name for h in hits}), "folds": {}}
    for h in hits[:1]:
        for k in (0, 1, 2):
            f = Path(h) / f"fold_{k}" / "checkpoint_final.pth"
            out["folds"][k] = ({"final": str(f), "size": f.stat().st_size, "inode": f.stat().st_ino} if f.is_file()
                               else {"final": None})
    return out


def same_final(a: dict, b: dict) -> dict:
    """{fold: True/False/None}: checkpoint_final.pth of both runs is one file (same inode), or byte-identical (md5, only
    computed when the sizes match), or absent (None)."""
    out = {}
    for k in (0, 1, 2):
        x, y = a.get("folds", {}).get(k, {}), b.get("folds", {}).get(k, {})
        if not x.get("final") or not y.get("final"):
            out[k] = None
        elif x["inode"] == y["inode"]:
            out[k] = True
        elif x["size"] != y["size"]:
            out[k] = False
        else:
            out[k] = md5(Path(x["final"])) == md5(Path(y["final"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--settings", default="")
    a = ap.parse_args()
    scratch = Path(a.scratch)
    (scratch / "mirr").mkdir(parents=True, exist_ok=True)
    want = {s.strip() for s in a.settings.split(",") if s.strip()} or None
    for k in (K_MIRR, K_CFG):
        if k not in dict(M.METHODS):
            M.METHODS.append((k, k))
            M.PLAIN[k] = k
    rungs = T1.ladder_rungs()
    res = {"settings": {}, "failures": None}
    for label, ds, tdir, mods in M.TASKS:
        cfg_path = next((REPO / f"benchmark/02_tasks/{tdir}/{ds}/5_scripts_{ds}/06_evaluate/configs")
                        .glob("*combined_01_results.yaml"))
        cfg = yaml.safe_load(cfg_path.read_text())
        _, mods_all, _ = M.load_modalities(cfg_path, ds, tdir)
        for mod, disp in mods.items():
            slab = f"{label} {disp}"
            if want and slab not in want:
                continue
            m_cfg = next(x for x in cfg["modalities"] if x["name"] == mod)
            r0 = m_cfg["runs"].get(REF)
            mirror = r0.replace("val000", "val100") if r0 and "val000" in r0 else None
            cfg_v = m_cfg["runs"].get(V100_CFG)
            st = {"val000_run": r0, "mirror_run": mirror, "config_val100_run": cfg_v}
            cfg2 = copy.deepcopy(cfg)
            m2 = next(x for x in cfg2["modalities"] if x["name"] == mod)
            keys = [REF, PA]
            for key, run in ((K_MIRR, mirror), (K_CFG, cfg_v if cfg_v and cfg_v != mirror else None)):
                if not run:
                    st[key] = {"included": False, "why": "no run id" if key == K_MIRR else "config val100 absent or = mirror"}
                    continue
                ok, notes = T1.coverage(mods_all, mod, run)
                st[key] = {"run": run, "included": ok, "coverage_notes": notes[:20]}
                if ok:
                    m2["runs"] = dict(m2["runs"], **{key: run})
                    keys.append(key)
            # checkpoint provenance (mirror check)
            st["ckpt"] = {"val000": ckpt_info(ds, mod, r0) if r0 else {}}
            if mirror:
                st["ckpt"]["mirror"] = ckpt_info(ds, mod, mirror)
                st["ckpt"]["mirror_same_final"] = same_final(st["ckpt"]["val000"], st["ckpt"]["mirror"])
            if st[K_CFG].get("included"):
                st["ckpt"]["config"] = ckpt_info(ds, mod, cfg_v)
                st["ckpt"]["config_same_final"] = same_final(st["ckpt"]["val000"], st["ckpt"]["config"])
            rid = rungs.get((label, mod))
            out = M.run_one(slab, ds, tdir, cfg2, mod, rid, scratch)
            yml = scratch / "mirr" / f"{ds}_{mod}.yaml" if rid else scratch / f"{ds}_{mod}.yaml"
            for k in keys:
                if k not in out["mean"]["dice"]:
                    M.fail(f"{slab}: key {k} injected but absent from the shared summary")
            folds = T1.fold_rows(out, yml, ds, tdir, keys)
            res["settings"][slab] = {"task": label, "ds": ds, "mod": mod, "palette_run": rid, "cols": out["cols"],
                                     "kind": out["kind"], "mean": {k: out["mean"]["dice"].get(k) for k in keys},
                                     "std": {k: out["std"]["dice"].get(k) for k in keys},
                                     "folds": {k: {r: v for r, v in f.items() if r in ("IND", "OOD")} for k, f in folds.items()},
                                     "status": st}
            print(slab, {k: {r: round(v, 2) for r, v in (out["mean"]["dice"].get(k) or {}).items()} for k in keys}, flush=True)
    res["failures"] = list(M.FAIL)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "t13_folds.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{len(M.FAIL)} check failure(s)")
    for f in M.FAIL:
        print("  -", f)


if __name__ == "__main__":
    main()
