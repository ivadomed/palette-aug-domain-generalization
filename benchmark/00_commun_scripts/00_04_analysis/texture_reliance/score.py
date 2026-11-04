#!/usr/bin/env python3
"""Texture-reliance experiment, step 5: score one domain's predictions (README.md).

Per model, fold, OOD item, ladder case and evaluated label (the label column of the ladder's original eval_all.csv,
ids from the model's dataset.json; GT ids via build_manifest.GT_ID_MAP):
  agree = Dice(pred_T == l, pred_orig == l)          (primary, no GT)
  gt_o  = Dice(pred_orig == l, GT == g), gt_T = Dice(pred_T == l, GT == g)   (secondary)
Dice = benchmark/00_commun_scripts/00_00_utils/eval_metrics.dice_score (NaN when both masks are empty: dropped
later exactly like ladder_ood_common.load_case_means drops non-finite rows).
Validation (label map + GT grid + re-prediction fidelity): gt_o of these Vulcan re-predictions vs the ladder's own
eval_all.csv Dice for the same (model, fold, case, label) -> <work>/score/<domain>_validation.tsv.
Output: <work>/score/<domain>.tsv (long format, case ids: scratch only).

  .venv/bin/python score.py --domain Glioma_t1n [--work ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
from eval_metrics import dice_score  # noqa: E402

FOLDS = (0, 1, 2)


def read(p: Path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) if p.exists() else None


def score_case(args):
    case, po, pt, gtp, labels, gt_ids = args
    a, b = read(po), read(pt)
    if a is None or b is None:
        return case, None, "missing prediction"
    if a.shape != b.shape:
        return case, None, f"shape mismatch {a.shape} vs {b.shape}"
    g = read(gtp) if gtp else None
    if g is not None and g.shape != a.shape:
        g, note = None, f"GT shape {g.shape} != pred {a.shape}"
    else:
        note = "" if g is not None else "no GT"
    rows = []
    for name, lid in labels.items():
        ma, mb = a == lid, b == lid
        r = dict(label_name=name, agree=dice_score(mb, ma), gt_o=float("nan"), gt_T=float("nan"))
        if g is not None:
            mg = g == gt_ids[name]
            r["gt_o"], r["gt_T"] = dice_score(ma, mg), dice_score(mb, mg)
        rows.append(r)
    return case, rows, note


def orig_eval(mdir: Path, k: int, item: str):
    out = {}
    f = mdir / f"fold{k}" / "eval_all.csv"
    if f.exists():
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if (r.get("contrast") or r.get("group")) == item:
                    try:
                        out[(r["case"], r["label"])] = float(r["dice"])
                    except ValueError:
                        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--work", type=Path, default=Path("/scratch/paulh/texture_reliance"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    D = json.loads((a.work / "manifest.json").read_text())["domains"][a.domain]
    od = a.work / "score"
    od.mkdir(parents=True, exist_ok=True)
    out_rows, val_rows, problems = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for m in ("flat", "texture"):
            for k in FOLDS:
                for it in D["items"]:
                    tag = f"{it['dataset']}__{it['item']}"
                    base = a.work / "pred" / a.domain / m / f"fold{k}"
                    gt_dir = Path(it["gt"])
                    jobs = []
                    for c in it["cases"]:
                        gtp = gt_dir / f"{c}.nii.gz"
                        jobs.append((c, base / "orig" / tag / f"{c}.nii.gz", base / "T" / tag / f"{c}.nii.gz",
                                     gtp if gtp.exists() else None, it["eval_labels"], it["gt_ids"]))
                    ref = orig_eval(Path(it["metrics_dirs"][m]), k, it["item"])
                    for case, rows, note in ex.map(score_case, jobs, chunksize=2):
                        if rows is None:
                            problems.append(f"{m} fold{k} {it['label']} {case}: {note}")
                            continue
                        if note and note != "no GT":
                            problems.append(f"{m} fold{k} {it['label']} {case}: {note}")
                        for r in rows:
                            out_rows.append(dict(domain=a.domain, model=m, fold=k, label=it["label"], case=case, **r))
                            ov = ref.get((case, r["label_name"]))
                            if ov is not None and np.isfinite(ov) and np.isfinite(r["gt_o"]):
                                val_rows.append((it["label"], m, abs(ov - r["gt_o"])))
    if problems:
        print("\n".join("PROBLEM: " + p for p in problems[:50]))
        print(f"[score] {len(problems)} problems")
    with (od / f"{a.domain}.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    # validation summary per item and model
    with (od / f"{a.domain}_validation.tsv").open("w") as f:
        f.write("label\tmodel\tn\tmedian_absdiff\tfrac_gt_0.02\tfrac_gt_0.10\n")
        for lab in sorted({v[0] for v in val_rows}):
            for m in ("flat", "texture"):
                d = np.array([v[2] for v in val_rows if v[0] == lab and v[1] == m])
                if d.size:
                    f.write(f"{lab}\t{m}\t{d.size}\t{np.median(d):.4f}\t{(d > 0.02).mean():.3f}\t{(d > 0.10).mean():.3f}\n")
    print(f"[score] {a.domain}: {len(out_rows)} rows -> {od / (a.domain + '.tsv')}; validation -> {a.domain}_validation.tsv")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
