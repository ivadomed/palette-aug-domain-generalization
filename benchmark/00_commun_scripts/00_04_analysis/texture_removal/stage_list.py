#!/usr/bin/env python3
"""List, repo-relative, the files another machine (tassan) needs to predict one task's texture-removed item with the
shared drivers, in the standard benchmark/ layout: for both models the run dir's Dataset*/<trainer>__nnUNetPlans__
3d_fullres/{dataset.json, plans.json, dataset_fingerprint.json, fold_{0,1,2}/checkpoint_best.pth}; every OOD item's
<input>_texrm dir (a symlink on Vulcan: pull with rsync -L), its original <input> dir (re-predicted on the same
machine) and the dataset.json of its raw dataset dir when it has one; Brain: the native reference images the on-harmony predict hook resamples to (_test_set/<item>/images_native).
Writes <work_root>/stage/<task>.files (entries) and <task>.sizes (one file per line with its size), used by
transfer.sh pull <task> on the other machine (or by hand:
  rsync -rlptL --files-from=:<task>.files vulcan.alliancecan.ca:/project/aip-jcohen/paulh/mri_synthesis_project/ <repo>/)

  python stage_list.py <task> [<task> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def main():
    cfg = yaml.safe_load((HERE / "tasks.yaml").read_text())
    out_dir = Path(cfg["work_root"]) / "stage"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in sys.argv[1:]:
        t = cfg["tasks"][task]
        paths, size = [], 0
        for m in ("flat", "texture"):
            run_dir = REPO / t["models"][m]["run_dir"]
            (td,) = sorted(run_dir.glob("Dataset*/*__nnUNetPlans__3d_fullres"))
            for f in ["dataset.json", "plans.json", "dataset_fingerprint.json"] + \
                     [f"fold_{k}/checkpoint_best.pth" for k in (0, 1, 2)]:
                if (td / f).exists():
                    paths.append(td / f)
                    size += (td / f).stat().st_size
        for it in t["items"]:
            src = REPO / it["input"] if not it["input"].startswith("/") else Path(it["input"])
            if "/_fovcrop/" in str(src):
                continue                                        # Abdomen crops: separate staging (scratch, not repo)
            tx = src.parent / (src.name + "_texrm")
            if not any(tx.glob("*_0000.nii.gz")):
                sys.exit(f"ERROR: {tx} is empty -- run 02_generate.sh {task} first")
            paths += [tx, src]                                 # texture-removed item + the original (re-predicted too)
            size += sum(p.stat().st_size for d in (tx, src) for p in d.glob("*.nii.gz"))
            if (src.parent / "dataset.json").exists():
                paths.append(src.parent / "dataset.json")
        if t.get("resample_to_native"):
            for it in t["items"]:
                d = REPO / t["resample_to_native"] / it["item"] / "images_native"
                paths.append(d)
                size += sum(p.stat().st_size for p in d.glob("*.nii.gz"))
        rel = sorted({str(p.relative_to(REPO)) for p in paths})
        f = out_dir / f"{task}.files"
        f.write_text("\n".join(rel) + "\n")
        # one line per FILE, "<bytes>\t<path>" (dirs expanded): transfer.sh balances its parallel chunks by size
        files = {}
        for r in rel:
            q = REPO / r
            for x in (sorted(q.iterdir()) if q.is_dir() else [q]):
                if x.is_file():
                    files[str(q.relative_to(REPO) / x.name) if q.is_dir() else r] = x.stat().st_size
        (out_dir / f"{task}.sizes").write_text("".join(f"{n}\t{k}\n" for k, n in sorted(files.items())))
        print(f"[stage] {task}: {len(rel)} entries, {len(files)} files, {size / 1e9:.2f} GB -> {f} (+ .sizes)")


if __name__ == "__main__":
    main()
