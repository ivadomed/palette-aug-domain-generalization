#!/usr/bin/env python3
"""T04: signed boundary shift (ASD vs GT) of every arm, one setting (registration 2026-11-12-PREDICTION_T01-T05.md, sha256
0b9f8710..., test T04).

Reuses E1 unchanged where it can: e1_lib.signed_surface (GT-anchored average signed distance, + = prediction inside GT, mm),
the frozen v2 settings snapshot, decompose_v2's prediction lookup / GT loading / binarisation, and gd2_lib's component
matching (gd2_lib.Arm + gd2_lib._match at f = 0.10, 'touch', 26-connectivity), so items, cases, folds and the matching rule
are E1's. Cases per item = E1's (GT dir intersected with the nine flat/noise/texture prediction dirs; no cap); in-domain
items are skipped (the analysis drops them, as e1_analyze does).

Adaptation of E1's per-case block (declared): E1 forms ONE component set (GT components detected by all nine fill
predictions) and scores the nine predictions on it. Here the same block runs on three component sets, per case and label:
  e1        GT components detected by all nine flat/noise/texture predictions (E1 verbatim; the reproduction gate);
  per_arm   for each arm, GT components detected by all three fold predictions of that arm (the registered per-arm
            matching, read as E1's rule applied to one arm, so the arm's three fold values sit on the same surfaces);
  inter     GT components detected by every fold prediction of every scored arm (registered secondary);
  per_fold  each prediction on the GT components it detects itself (sensitivity: the other reading of 'per arm').
Kept prediction voxels = predicted components touching a GT component of the set (E1's rule). Exactness of the crop: all
work runs on a box that contains every GT voxel and every predicted component touching the GT (+3 voxels, clipped to the
image), so component identities, matches, kept masks and the border-face rule are those of the full image.
Also writes, per prediction, whole-label Dice vs GT (QC against the stored eval_all.csv: right files, right space).

  t04_signed.py --setting MS_flair --out DIR [--workers 8] [--max-cases N]
Writes DIR/rows/<setting>.parquet, DIR/qc/<setting>.parquet, DIR/prov/<setting>.tsv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t04_arms as TA  # noqa: E402  (also puts v2 + E1 on sys.path)
import gd2_lib as gl  # noqa: E402
from e1_lib import signed_surface  # noqa: E402

dv = TA.dv
F = 0.10
NAN = dict(asd=np.nan, mean_abs=np.nan, under_share=np.nan, over_share=np.nan, n_faces=0)


def _crop(gb, preds_b):
    """Box containing gb and every predicted component touching gb, + 3 voxels, clipped."""
    lo = np.array([a.min() for a in np.nonzero(gb)])
    hi = np.array([a.max() for a in np.nonzero(gb)])
    for pb in preds_b:
        if not pb.any():
            continue
        lab, _ = gl._components(pb)
        ids = np.unique(lab[gb])
        ids = ids[ids > 0]
        if not ids.size:
            continue
        m = np.isin(lab, ids)
        idx = np.nonzero(m)
        lo = np.minimum(lo, [a.min() for a in idx])
        hi = np.maximum(hi, [a.max() for a in idx])
    return tuple(slice(max(int(l) - 3, 0), min(int(h) + 4, s)) for l, h, s in zip(lo, hi, gb.shape))


def _score(common, lab_g, n_g, arms, keys, sp, vox_ml, base, set_name):
    rows = []
    n_set = int(common.sum())
    lut = np.zeros(n_g + 1, bool); lut[1:] = common
    G = lut[lab_g]
    vg = int(G.sum())
    for k in keys:
        arm, fold = k
        row = dict(base, set=set_name, arm=arm, fold=fold, n_comp_set=n_set, gt_ml_set=float(vg * vox_ml))
        if n_set == 0:
            rows.append(dict(row, pred_ml_kept=np.nan, rvd=np.nan, **NAN))
            continue
        a = arms[k]
        ids = np.unique(a.pa[lut[a.pg]]) if a.pa.size else np.zeros(0, np.int64)
        lp = np.zeros(a.n + 1, bool); lp[ids] = True
        P = lp[a.lab]
        vp = int(P.sum())
        r = signed_surface(G, P, sp)
        rows.append(dict(row, pred_ml_kept=float(vp * vox_ml), rvd=(vp - vg) / vg, **(r or NAN)))
    return rows


def process_case(task):
    it, case, dirs = task["item"], task["case"], task["dirs"]
    rows, qc, errs = [], [], []
    try:
        gt_img, gt = dv.scoring.load_gt(dv.R, it, case)
        sp = gt_img.GetSpacing()[::-1]
        vox_ml = float(np.prod(sp)) / 1000.0
        preds = {}
        for arm, by_fold in dirs.items():
            for fold, d in by_fold.items():
                p = dv.scoring.read_stored_pred(Path(d) / f"{case}.nii.gz", it, dv.R, case)
                if p.shape != gt.shape:
                    raise RuntimeError(f"shape mismatch {arm} fold {fold}: {p.shape} vs {gt.shape}")
                if p.max() > 255 or p.min() < 0:
                    raise RuntimeError(f"label range {arm} fold {fold}: {p.min()}..{p.max()}")
                preds[(arm, int(fold))] = p.astype(np.uint8)
        arm_names = sorted({k[0] for k in preds})
        for name, pid, gid in it["triples"]:
            gb = dv.binarize(gt, gid)
            key = dict(item_key=it["key"], group=it["group"], case=case, label=name)
            n_gt = int(gb.sum())
            pbs = {}
            for k, p in preds.items():
                pb = dv.binarize(p, pid)
                n_p, n_tp = int(pb.sum()), int((pb & gb).sum())
                qc.append(dict(key, arm=k[0], fold=k[1], n_gt=n_gt, n_pred=n_p,
                               dice=float(gl.dice_from_counts(n_p, n_tp, n_gt))))
                pbs[k] = pb
            lab_full, n_g = gl._components(gb)
            if n_g == 0:
                continue
            g_size = np.bincount(lab_full.ravel(), minlength=n_g + 1).astype(np.int64)
            base = dict(key, n_comp_total=int(n_g), gt_ml_total=float(n_gt * vox_ml))
            slc = _crop(gb, pbs.values())
            lab_g, Gc = lab_full[slc], gb[slc]
            arms, mg = {}, {}
            for k, pb in pbs.items():
                a = gl.Arm(pb[slc], lab_g, n_g, Gc)
                mg[k] = gl._match(a, n_g, g_size, F, "touch")[0]
                arms[k] = a
            del pbs
            # e1: all nine fill predictions
            fill = [k for k in preds if k[0] in TA.FILL]
            if len(fill) == 9:
                c = np.logical_and.reduce([mg[k] for k in fill])
                rows += _score(c, lab_g, n_g, arms, fill, sp, vox_ml, base, "e1")
            # per arm: that arm's three folds
            for arm in arm_names:
                ks = [k for k in preds if k[0] == arm]
                c = np.logical_and.reduce([mg[k] for k in ks])
                rows += _score(c, lab_g, n_g, arms, ks, sp, vox_ml, base, "per_arm")
            # per_fold (sensitivity, the other reading of 'per arm'): each prediction on the GT components it detects itself
            for k in sorted(preds):
                rows += _score(mg[k], lab_g, n_g, arms, [k], sp, vox_ml, base, "per_fold")
            # inter: every fold of every scored arm (scored only when all seven arms are present)
            sc = [k for k in preds if k[0] in TA.SCORED]
            if len(sc) == 3 * len(TA.SCORED):
                c = np.logical_and.reduce([mg[k] for k in sc])
                rows += _score(c, lab_g, n_g, arms, sorted(preds), sp, vox_ml, base, "inter")
    except Exception as e:  # noqa: BLE001
        errs.append(f"{it['key']}\t{case}\t{e!r}")
    return rows, qc, errs


def stored_dice(st, it, rid, fold):
    """({(case, label): dice}, csv) from the stored evaluation of that run (sd.candidates; checkpoint sweeps excluded)."""
    best = ({}, None)
    for f in dv.sd.candidates(it["dataset"], dv.sd.MODEL[st["task"]], st["training_contrast"], rid, fold):
        if "checkpoint_comparison" in f:
            continue
        with open(f) as fh:
            rr = [r for r in csv.DictReader(fh) if r["group"] in (it["item"], it["item"] + "_fovcrop")]
        if len(rr) > len(best[0]):
            best = ({(r["case"], r["label"]): float(r["dice"]) if r["dice"] not in ("", "nan") else np.nan for r in rr}, f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    st = TA.settings()[a.setting]
    rid = TA.run_ids(st)
    out = Path(a.out)
    for d in ("rows", "qc", "prov"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tasks, prov, stored = [], [], []
    for it in st["items"]:
        if it["group"] == "in-domain":
            continue
        fill_dirs = {arm: {f: dv.pred_dir(st, it, arm, f)[0] for f in dv.FOLDS} for arm in TA.FILL}
        gt_dir = Path(it["gt_dir"]) if it["gt_dir"].startswith("/") else dv.R / it["gt_dir"]
        cases = {f.name[:-7] for f in gt_dir.glob("*.nii.gz")}
        ok = all(fill_dirs[arm][f] is not None for arm in TA.FILL for f in dv.FOLDS)
        if ok:
            for arm in TA.FILL:
                for f in dv.FOLDS:
                    cases &= {x.name[:-7] for x in Path(fill_dirs[arm][f]).glob("*.nii.gz")}
        cases = sorted(cases) if ok else []
        if a.max_cases:
            cases = cases[: a.max_cases]
        dirs = {}
        for arm in TA.SCORED + TA.REPORTED:
            if arm not in rid:
                prov.append(dict(setting=a.setting, item_key=it["key"], group=it["group"], arm=arm, run_id="", fold=-1,
                                 dir="ABSENT", how="no run id", n_cases=len(cases), missing_cases=len(cases)))
                continue
            by_fold, complete = {}, True
            for f in dv.FOLDS:
                p, how = TA.pred_dir(st, it, arm, rid[arm], f)
                have = {x.name[:-7] for x in p.glob("*.nii.gz")} if p else set()
                miss = len(set(cases) - have)
                prov.append(dict(setting=a.setting, item_key=it["key"], group=it["group"], arm=arm, run_id=rid[arm], fold=f,
                                 dir=str(p) if p else "ABSENT", how=how, n_cases=len(cases), missing_cases=miss))
                complete &= (p is not None and miss == 0)
                by_fold[f] = str(p) if p else None
                if p is not None:
                    d, src = stored_dice(st, it, rid[arm], f)
                    stored += [dict(item_key=it["key"], case=c, label=l, arm=arm, fold=f, stored_dice=v, stored_csv=src)
                               for (c, l), v in d.items() if c in cases]
            if complete:
                dirs[arm] = by_fold
            elif cases:
                print(f"[absent] {a.setting} {it['key']} {arm}: incomplete predictions (arm dropped for this item)", flush=True)
        if not cases:
            print(f"[absent] {a.setting} {it['key']}: no E1 cases (fill arms absent), item skipped", flush=True)
            continue
        tasks += [dict(item=it, case=c, dirs=dirs) for c in cases]
    print(f"[t04] {a.setting}: {len(tasks)} case tasks; run ids: {rid}", flush=True)
    rows, qc, errs = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(process_case, t) for t in tasks]), 1):
            r, q, e = fut.result(); rows += r; qc += q; errs += e
            if i % 50 == 0:
                print(f"[t04] {i}/{len(tasks)}", flush=True)
    sfx = ".smoke" if a.max_cases else ""
    df = pd.DataFrame(rows); df.insert(0, "setting", a.setting)
    df.to_parquet(out / "rows" / f"{a.setting}{sfx}.parquet")
    Q = pd.DataFrame(qc)
    if len(Q) and stored:
        Q = Q.merge(pd.DataFrame(stored), on=["item_key", "case", "label", "arm", "fold"], how="left")
    Q.insert(0, "setting", a.setting)
    Q.to_parquet(out / "qc" / f"{a.setting}{sfx}.parquet")
    pd.DataFrame(prov).to_csv(out / "prov" / f"{a.setting}{sfx}.tsv", sep="\t", index=False)
    print(f"[t04] {a.setting}: {len(df)} rows, {len(Q)} qc rows, errors {len(errs)}", flush=True)
    for e in errs:
        print("[error]", e, flush=True)
    sys.exit(3 if errs else 0)


if __name__ == "__main__":
    main()
