#!/usr/bin/env python3
"""T02 step 2 (registration plans/2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...): label-measured intensity
features of the TEST images, per case and foreground class, never from predictions.

  target = voxels of class c; ring = voxels within (0, 3 mm] of class c (spacing-aware Euclidean distance transform),
  not in c (any other label allowed). Intensities = the image given to the network (before nnU-Net normalisation);
  for MRI, target and ring keep nonzero voxels only.
  CF01: order = median(target) > median(ring)           (cell: o = share of cases)
  CF02: q = percentile rank of median(target) among the case's nonzero voxels   (cell: median over cases)
  CF03: cnr = |mean(target) - mean(ring)| / SD(ring)     (cell: median over cases)

Cell -> image mapping: the cases a cell scores (step 1's cell_cases.json: source metrics dir, raw eval group, case id,
labels) are looked up in the scoring recipes of benchmark/01_commun_results/gain_decomposition/v2/settings_snapshot.yaml
(built by skip_perturbation/build_settings.items_for from each dataset's own evaluator: input dir, GT dir, label
triples, crops), keyed by (setting, dataset, raw group). Items the snapshot lacks are taken from EXTRA below and checked
by recomputing the stored fold-0 Dice of one case (--map). Items with no recipe are reported, never guessed.

Modes (all through the job wrapper, never on a login node):
  --map         write <scratch>/manifest.json + <out>/mapping_report.tsv (no case ids in <out>)
  --features    per-case features -> <scratch>/case_features.tsv (case ids: scratch only)
  --aggregate   cell-level CF01/CF02/CF03 -> <out>/cells_features.tsv, <out>/cells_per_class.tsv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

R = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
SNAP = R / "benchmark/01_commun_results/gain_decomposition/v2/settings_snapshot.yaml"
RING_MM = 3.0
T = "benchmark/02_tasks"

# Recipes for items the snapshot lacks (in-domain pools of HCC). (dataset, raw group) -> (input dir, GT dir,
# {eval label name: (pred id, GT id)}). Same layout as the snapshot's lld-mmri-hcc items (build_settings.items_for).
_L = f"{T}/liver_hcc/lld-mmri-hcc/2_nnUNet_lld-mmri-hcc/raw"
_S = f"{T}/liver_hcc/liverhccseg/2_nnUNet_liverhccseg/raw"
EXTRA = {}
for _p in ("ce-art", "ce-del", "ce-pre", "ce-pv"):
    EXTRA[("lld-mmri-hcc", _p)] = (f"{_L}/imagesTs_{_p}", f"{_L}/labelsTs_{_p}", {"tumour": (2, 1)})
    EXTRA[("liverhccseg", f"{_p}_T1w")] = (f"{_S}/imagesTs_{_p}_T1w", f"{_S}/labelsTs_{_p}_T1w", {"tumour": (2, 1)})

CT_RE = re.compile(r"(^|[_/-])(ct|cbct)($|[_/-])", re.I)


def setting_key(cell: dict) -> str:
    return f"{cell['task']}_{cell['modality']}"


def ds_of(metrics_dir: str) -> str:
    p = Path(metrics_dir).parts
    return p[p.index("02_tasks") + 2] if "02_tasks" in p else metrics_dir


def abspath(p: str) -> Path:
    return Path(p) if p.startswith("/") else R / p


def is_ct(dataset: str, group: str) -> bool:
    return bool(CT_RE.search(group)) or dataset in ("sliver07", "msd-pancreas")


# ── mapping ──────────────────────────────────────────────────────────────────
def build_manifest(cells: dict, snap: dict):
    """-> (manifest {cell key: {...,'cases': [{img, lab, classes, ct, ...}]}}, report rows, problems)."""
    man, report, problems = {}, [], []
    seen_items = {}
    for key, cell in cells.items():
        st = snap["settings"].get(setting_key(cell))
        sitems = {(it["dataset"], it["item"]): it for it in (st["items"] if st else [])}
        col_labels = defaultdict(set)              # column -> labels it contributes (empty = all labels)
        for col, lab in cell["leaf_refs"]:
            col_labels[col].add(lab)
        out_cases = []
        for e in cell["cases"]:
            src = cell["sources"][e["source"]]
            ds = ds_of(src["metrics_dir"])
            g = e["group"]
            it = sitems.get((ds, g))
            how = "snapshot"
            if it is not None:
                inp, gtd = it["input_dir"], it["gt_dir"]
                tri = {t[0]: (t[1], t[2]) for t in it["triples"]}
                if it.get("pred_transform"):          # on-harmony: native image on the GT grid (same intensities)
                    inp = it["pred_transform"]["ref_dir"]
                    how = "snapshot_native_image"
            elif (ds, g) in EXTRA:
                inp, gtd, tri = EXTRA[(ds, g)]
                how = "extra"
            else:
                seen_items.setdefault((cell["setting"], ds, g), ("UNMAPPED", 0))
                continue
            allow = col_labels.get(e["column"], {None})
            labs = [l for l in e["labels"].split(",") if l and (None in allow or l in allow)]
            classes = [(l, tri[l][1]) for l in labs if l in tri]
            miss = [l for l in labs if l not in tri]
            if miss:
                problems.append(f"{cell['setting']} {ds}:{g}: eval labels {miss} not in the recipe's triples")
            img = abspath(inp) / f"{e['case']}_0000.nii.gz"
            lab = abspath(gtd) / f"{e['case']}.nii.gz"
            out_cases.append(dict(dataset=ds, group=g, case=e["case"], img=str(img), lab=str(lab), classes=classes,
                                  ct=is_ct(ds, g), how=how))
            k = (cell["setting"], ds, g)
            n = seen_items.get(k, (how, 0))[1]
            seen_items[k] = (how, n + 1)
        man[key] = dict({k: cell[k] for k in ("setting", "task", "modality", "column", "kind")}, cases=out_cases)
    for (s, ds, g), (how, n) in sorted(seen_items.items()):
        report.append(dict(setting=s, dataset=ds, group=g, recipe=how, n_cases=n))
    return man, report, problems


def check_files(man: dict) -> list:
    probs, seen = [], set()
    for cell in man.values():
        for c in cell["cases"]:
            for p in (c["img"], c["lab"]):
                if p not in seen:
                    seen.add(p)
                    if not Path(p).exists():
                        probs.append(f"missing file: {cell['setting']} {c['dataset']}:{c['group']} {Path(p).parent}")
    return probs


def verify_extra(man: dict, cells: dict) -> list:
    """Recompute the stored fold-0 Dice of the first case of every EXTRA item (baseline-run prediction) and compare
    with that run's eval_all.csv row."""
    import nibabel as nib
    out, done = [], set()
    for key, cell in man.items():
        for c in cell["cases"]:
            if c["how"] != "extra" or (cell["setting"], c["dataset"], c["group"]) in done:
                continue
            done.add((cell["setting"], c["dataset"], c["group"]))
            src = next(s for s in cells[key]["sources"] if ds_of(s["metrics_dir"]) == c["dataset"])
            md = Path(src["metrics_dir"])
            evs = sorted(md.glob("nnUNet_*baseline*/fold0/eval_all.csv"))
            pred_base = Path(str(md).replace("/02_metrics/", "/01_predictions/"))
            row = None
            for ev in evs:
                rid = ev.parent.parent.name[len("nnUNet_"):]
                for r in csv.DictReader(ev.open()):
                    if r["group"] == c["group"] and r["case"] == c["case"]:
                        hits = glob.glob(str(pred_base / "*" / rid / "fold0" / c["group"] / f"{c['case']}.nii.gz"))
                        if hits:
                            row = (r, hits[0])
                            break
                if row:
                    break
            if row is None:
                out.append(dict(setting=cell["setting"], dataset=c["dataset"], group=c["group"], check="no stored prediction found", ok=False))
                continue
            gt = np.asanyarray(nib.load(c["lab"]).dataobj)
            pr = np.asanyarray(nib.load(row[1]).dataobj)
            for name, gid in c["classes"]:
                pid = EXTRA[(c["dataset"], c["group"])][2][name][0]
                g, p = gt == gid, pr == pid
                d = 2 * (g & p).sum() / max(g.sum() + p.sum(), 1)
                stored = float(row[0]["dice"]) if row[0]["label"] == name else float("nan")
                out.append(dict(setting=cell["setting"], dataset=c["dataset"], group=c["group"], check=f"dice {name}",
                                recomputed=round(float(d), 4), stored=round(stored, 4),
                                ok=bool(abs(d - stored) < 1e-3) if np.isfinite(stored) else None))
    return out


# ── features ─────────────────────────────────────────────────────────────────
def case_features(img_path: str, lab_path: str, classes: list, ct: bool) -> list:
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt
    im, lb = nib.load(img_path), nib.load(lab_path)
    a = np.asanyarray(im.dataobj).astype(np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    lab = np.rint(np.asanyarray(lb.dataobj)).astype(np.int32)
    if a.shape != lab.shape:
        raise RuntimeError(f"shape {a.shape} vs label {lab.shape}")
    sp = np.array(im.header.get_zooms()[:3], dtype=float)
    aff_dev = float(np.abs(im.affine - lb.affine).max())
    nz = np.sort(a[a != 0].ravel())
    valid = None if ct else (a != 0)
    rows = []
    for name, gid in classes:
        gids = gid if isinstance(gid, (list, tuple)) else [gid]
        tgt = np.isin(lab, gids)
        r = dict(label=name, n_target=int(tgt.sum()), order=np.nan, q=np.nan, cnr=np.nan, med_t=np.nan, med_r=np.nan,
                 n_ring=0, aff_dev=aff_dev)
        if not tgt.any():
            rows.append(r)
            continue
        idx = np.argwhere(tgt)
        m = np.ceil(RING_MM / sp).astype(int) + 1
        lo = np.maximum(idx.min(0) - m, 0)
        hi = np.minimum(idx.max(0) + m + 1, np.array(a.shape))
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        st, sa = tgt[sl], a[sl]
        dt = distance_transform_edt(~st, sampling=sp)
        ring = (dt > 0) & (dt <= RING_MM)
        tsel = st if valid is None else st & valid[sl]
        if valid is not None:
            ring &= valid[sl]
        tv, rv = sa[tsel], sa[ring]
        r["n_ring"] = int(rv.size)
        if tv.size == 0 or rv.size < 2:
            rows.append(r)
            continue
        mt, mr = float(np.median(tv)), float(np.median(rv))
        r.update(med_t=mt, med_r=mr, order=float(mt > mr),
                 q=float((np.searchsorted(nz, mt, "left") + np.searchsorted(nz, mt, "right")) / 2 / nz.size) if nz.size else np.nan,
                 cnr=float(abs(tv.mean() - rv.mean()) / rv.std(ddof=1)) if rv.std(ddof=1) > 0 else np.nan)
        rows.append(r)
    return rows


def _work(job):
    img, lab, classes, ct = job
    try:
        return job, case_features(img, lab, classes, ct), None
    except Exception as e:  # noqa: BLE001
        return job, None, f"{type(e).__name__}: {e}"


# ── aggregation ──────────────────────────────────────────────────────────────
def summarise(rows: list) -> dict:
    """{label: (o, q, cnr, n)} over the case rows of one pool (cases where the class has a value)."""
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    out = {}
    for lab, rs in by.items():
        o = [r["order"] for r in rs if np.isfinite(r["order"])]
        q = [r["q"] for r in rs if np.isfinite(r["q"])]
        c = [r["cnr"] for r in rs if np.isfinite(r["cnr"])]
        out[lab] = (float(np.mean(o)) if o else np.nan, float(np.median(q)) if q else np.nan,
                    float(np.median(c)) if c else np.nan, len(o))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True, help="step 1 cell_cases.json (scratch)")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    cells = json.load(open(a.cells))
    if a.tasks:
        cells = {k: v for k, v in cells.items() if v["task"] in a.tasks.split(",")}

    def write(path, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k)))
                                  for k in keys) + "\n")

    if a.map:
        snap = yaml.safe_load(SNAP.read_text())
        man, report, probs = build_manifest(cells, snap)
        probs += check_files(man)
        ver = verify_extra(man, cells)
        json.dump(man, open(a.scratch / "manifest.json", "w"))
        write(a.out / "mapping_report.tsv", report)
        write(a.out / "mapping_verify_extra.tsv", ver or [dict(note="no EXTRA items used")])
        (a.out / "mapping_problems.txt").write_text("\n".join(probs) + "\n")
        print(f"cells {len(man)}; items {len(report)}; unmapped {sum(r['recipe'] == 'UNMAPPED' for r in report)}; "
              f"problems {len(probs)}; extra checks {sum(bool(v.get('ok')) for v in ver)}/{len(ver)} ok", flush=True)

    if a.features:
        man = json.load(open(a.scratch / "manifest.json"))
        jobs = {}
        for cell in man.values():
            for c in cell["cases"]:
                jobs.setdefault((c["img"], c["lab"], c["ct"]), set()).update(tuple((n, g if not isinstance(g, list) else tuple(g)) for n, g in c["classes"]))
        todo = [(img, lab, sorted(cls), ct) for (img, lab, ct), cls in jobs.items()]
        print(f"{len(todo)} distinct (image, label) pairs", flush=True)
        out_rows, errs = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(_work, j) for j in todo]
            for i, f in enumerate(as_completed(futs)):
                job, rows, err = f.result()
                if err:
                    errs.append(f"{job[0]}: {err}")
                    continue
                for r in rows:
                    out_rows.append(dict(img=job[0], lab=job[1], ct=job[3], **r))
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        write(a.scratch / "case_features.tsv", out_rows)
        (a.scratch / "case_feature_errors.txt").write_text("\n".join(errs) + "\n")
        print(f"features: {len(out_rows)} rows, {len(errs)} errors", flush=True)

    if a.aggregate:
        man = json.load(open(a.scratch / "manifest.json"))
        feat = defaultdict(dict)
        with open(a.scratch / "case_features.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                feat[(r["img"], r["lab"])][r["label"]] = {k: float(r[k]) if r[k] not in ("", "nan") else np.nan
                                                           for k in ("order", "q", "cnr")}
        def pool_rows(cell):
            rows, miss = [], 0
            for c in cell["cases"]:
                fr = feat.get((c["img"], c["lab"]))
                if fr is None:
                    miss += 1
                    continue
                for name, _ in c["classes"]:
                    if name in fr:
                        rows.append(dict(label=name, **fr[name]))
            return rows, miss
        by_setting = defaultdict(list)
        for cell in man.values():
            by_setting[cell["setting"]].append(cell)
        cells_out, per_class = [], []
        for s, cs in by_setting.items():
            src_rows, src_miss = [], 0
            for cell in cs:
                if cell["kind"] == "IND":
                    r_, m_ = pool_rows(cell)
                    src_rows += r_
                    src_miss += m_
            src = summarise(src_rows)
            for cell in cs:
                if cell["kind"] != "OOD":
                    continue
                rows, miss = pool_rows(cell)
                tst = summarise(rows)
                labs = [l for l in tst if l in src and np.isfinite(tst[l][0]) and np.isfinite(src[l][0])]
                pcs = []
                for l in labs:
                    o_t, q_t, cnr_t, n_t = tst[l]
                    o_s, q_s, cnr_s, n_s = src[l]
                    rev = float((o_s - .5) * (o_t - .5) < 0 and abs(o_s - .5) >= .25 and abs(o_t - .5) >= .25)
                    pc = dict(setting=s, group=cell["column"], label=l, n_cases_t=n_t, n_cases_s=n_s, o_s=o_s, o_t=o_t,
                              reversed=rev, Dorder=abs(o_t - o_s), q_s=q_s, q_t=q_t, Dq=abs(q_t - q_s),
                              cnr_t=cnr_t, cnr_s=cnr_s)
                    pcs.append(pc)
                per_class += pcs
                mean = (lambda k: float(np.nanmean([p[k] for p in pcs])) if pcs else np.nan)
                cells_out.append(dict(setting=s, group=cell["column"], n_cases_mapped=len({(c['img']) for c in cell["cases"]}),
                                      n_cases_unmapped_features=miss, n_classes=len(pcs), source_missing=src_miss,
                                      o_s=mean("o_s"), o_t=mean("o_t"), reversed=mean("reversed"), Dorder=mean("Dorder"),
                                      q_s=mean("q_s"), q_t=mean("q_t"), Dq=mean("Dq"), cnr_t=mean("cnr_t"), cnr_s=mean("cnr_s")))
        write(a.out / "cells_features.tsv", cells_out)
        write(a.out / "cells_per_class.tsv", per_class)
        print(f"aggregate: {len(cells_out)} OOD cells with features, {len(per_class)} class rows", flush=True)


if __name__ == "__main__":
    main()
