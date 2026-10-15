#!/usr/bin/env python
"""
texture_rule_polarity.py -- which (train contrast, eval contrast) cells gain from texture (rung 4.5 FLAT fill -> rung 5 REAL fill)?
Candidate rule: TARGET POLARITY FLIP. Written and committed 2026-10-15 BEFORE any polarity feature was computed (the commit is the
time stamp). The real-vs-flat Delta of every cell HAS been seen (FINDINGS 47); the rule is not fitted to it, it is derived from the
BraTS mechanism of FINDINGS 30 (crossed inversion / flattening counterfactual): the flat-fill model binds the TRAINING contrast's
target polarity (edema darker than its surroundings on T1n, brighter on T2w) to textured inputs, the real-fill model is nearly
polarity-invariant (its synthetic images carry texture with random per-region sign). Prediction:
    texture helps (Delta_real-flat > 0) where the target's polarity vs its surroundings FLIPS between the training and the eval
    contrast; where the polarity is preserved flat ~ real, i.e. texture does not help (Delta <= 0).
Because BraTS edema motivated the rule, BraTS is the DERIVATION task and every confirmatory test excludes it.

Definitions (frozen):
  polarity of label l on contrast key k: d = (mean target - mean ring) / sqrt((var target + var ring) / 2) on the z-scored
    foreground image, ring = 1-5 voxels around the label inside the foreground (cross_dataset_fillswap_model.prep/label_feats
    geometry), mean over <= 40 test cases of that dataset/contrast. Also stored: signed AUC P(target > ring). BraTS: d from the
    existing bigfeat shards (R_mean, ring_mean, R_std, ring_std; same z-scoring and ring).
  per-label flip: sign(d_train) != sign(d_eval).  pooled cell: flip fraction f = eval-voxel-weighted share of the cell's labels
    that flip; cell is a FLIP cell iff f >= 0.5.  continuous: agreement A = sum_l w_l d_train,l d_eval,l / sum_l w_l |d_train,l d_eval,l|.
  cells = exactly the OOD items of every paper ladder (own ood_contrasts + extra_ood_sources/items, mandible = hanseg ct/mrt1 +
    pddca ct), one cell per (training setting, source dataset, eval item). Delta = 100 x (real - flat) Dice, folds 0-2 averaged per
    case, labels averaged per case for pooled cells; p = paired Wilcoxon over patients, Holm within task.
Tests (frozen):
  P1 PRIMARY (confirmatory = all tasks but BraTS): S = mean over tasks having both cell types of [mean Delta(flip cells) -
     mean Delta(preserved cells)]; null = flip labels permuted WITHIN task (exact if <= 200k assignments, else 200k random);
     one-sided (S > 0).  Kill: p >= 0.05.
  P2 patient-level macro test (project standard): within each task with both cell types, per patient
     d_p = mean Delta over its flip cells - mean Delta over its preserved cells (patients present in both); task statistic = mean d_p;
     S2 = mean over tasks; null = independent random sign flips of every d_p (100k); one-sided.
  P3 description: sign accuracy (flip -> +, preserved -> -) on Holm-significant cells and on all cells vs "always helps";
     Spearman(-A, Delta); unstratified version of P1 (labels permuted across all cells).
  Reported for: confirmatory set (primary), derivation set (BraTS), all tasks; pooled cells (primary), per-label cells (secondary).
Stages: extract -> outputs/data/pol_label_<key>.csv;  cells -> outputs/data/texture_rule_cells{,_perlabel,_patient}.csv;
        score -> outputs/tables/texture_rule_polarity.md
"""
from __future__ import annotations
import argparse, glob, itertools, json, math, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata, spearmanr, wilcoxon
THIS = Path(__file__).resolve().parent; REPO = THIS.parents[6]; sys.path.insert(0, str(THIS))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils")); sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import cross_dataset_fillswap_model as xd  # noqa: E402
import visibility_sign_rule as vs  # noqa: E402
from stat_tests import holm  # noqa: E402
from ladder_ood_common import _patient_key  # noqa: E402
T2 = REPO / "benchmark/02_tasks"; DATA = THIS.parent / "outputs/data"; TAB = THIS.parent / "outputs/tables"
MAXC = 40; RS = np.random.RandomState(0)
ALIAS = {"brats2024-glioma": "brats", "on-harmony": "onharmony", "duke-breast-mri": "duke", "acrin6698": "acrin", "atlas-liver-hcc": "atlas"}
AMOS_MAP = {6: 1, 2: 2, 3: 3, 1: 4}   # amos ids -> chaos ids (liver, right kidney, left kidney, spleen); 06_00_evaluate_amos.py
EXCLUDE_TASKS = {"pansegdata", "totalseg-pelvic", "atlas-liver-hcc"}   # 2026-10-15, Paul: pancreas, then pelvis + HCC (the latter after the analyst had run all 3 analyses with them; Paul blind to results) removed from the texture-rule analyses


def load_cells(name):
    """texture_rule_cells{,_perlabel,_patient}.csv without the excluded tasks."""
    d = pd.read_csv(DATA / name)
    return d[~d.task.isin(EXCLUDE_TASKS)].reset_index(drop=True)


# ---------------- features ----------------
def manifest():
    m = vs.manifest(); nn = vs._nn
    m["amos:ct"] = ("amos", lambda: nn(T2 / "abdomen_healthy/amos/2_nnUNet_amos/raw", "imagesTs_ct", "labelsTs_ct"))
    m["amos:mri"] = ("amos", lambda: nn(T2 / "abdomen_healthy/amos/2_nnUNet_amos/raw", "imagesTs_mri", "labelsTs_mri"))
    m["sliver07:ct"] = (False, lambda: nn(T2 / "abdomen_healthy/sliver07/2_nnUNet_sliver07/raw", "imagesTs_ct", "labelsTs_ct"))
    m["atlas:t1w"] = (False, lambda: nn(T2 / "liver_hcc/atlas-liver-hcc/2_nnUNet_atlas-liver-hcc/raw/Dataset080_AtlasLiverHCC", "imagesTs_t1w", "labelsTs_t1w"))
    for it in ("t2wi", "dwi", "inphase", "outphase"):
        m[f"lld-mmri-hcc:{it}"] = (False, lambda it=it: nn(T2 / "liver_hcc/lld-mmri-hcc/2_nnUNet_lld-mmri-hcc/raw", f"imagesTs_{it}", f"labelsTs_{it}"))
    return m


def pol_feats(z, fg, mask, pad=12):
    if mask.sum() < xd.MINVOX:
        return None
    idx = np.where(mask); sl = tuple(slice(max(0, i.min() - pad), i.max() + pad + 1) for i in idx)
    m = mask[sl]; zz, f = z[sl], fg[sl]
    ring = (distance_transform_edt(~m) <= 5) & ~m & f
    if ring.sum() < 50:
        return None
    t, r = zz[m], zz[ring]
    sd = np.sqrt((t.var() + r.var()) / 2); d = (t.mean() - r.mean()) / sd if sd > 0 else np.nan
    ts, rs_ = xd._sub(t), xd._sub(r); rk = rankdata(np.concatenate([ts, rs_])); n1, n2 = ts.size, rs_.size
    auc = (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return d, auc, int(m.sum())


def case_task(task):
    key, cid, ip, lp, mode = task; rows = []
    try:
        im = nib.load(str(ip)); z, _, fg, _ = xd.prep(np.asarray(im.dataobj), im.header.get_zooms())
        lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
        if mode == "amos":
            lbl = sum((lbl == a) * b for a, b in AMOS_MAP.items())
        labs = [1] if mode is True else [int(v) for v in np.unique(lbl) if v > 0]
        for lab in labs:
            o = pol_feats(z, fg, (lbl > 0) if mode is True else (lbl == lab))
            if o:
                rows.append((key, cid, lab, o[2], o[0], o[1]))
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {key} {cid}: {e}", flush=True)
    return rows


def extract(nproc):
    tasks = []
    for k, (mode, fn) in manifest().items():
        if k.startswith("brats:") or (DATA / f"pol_label_{k.replace(':', '_')}.csv").exists():
            continue
        cs = fn()[:MAXC]
        print(f"{k}: {len(cs)} cases", flush=True); tasks += [(k, c, i, l, mode) for c, i, l in cs]
    if tasks:
        with Pool(nproc) as pool:
            res = [r for rs in pool.imap_unordered(case_task, tasks, chunksize=1) for r in rs]
        df = pd.DataFrame(res, columns=["key", "case", "label", "n_vox", "d", "auc"])
        for k, g in df.groupby("key"):
            g.to_csv(DATA / f"pol_label_{k.replace(':', '_')}.csv", index=False)
    b = pd.concat([pd.read_csv(f, usecols=["patient", "contrast", "region", "n_R", "R_mean", "R_std", "ring_mean", "ring_std"]) for f in glob.glob(str(DATA / "bigfeat_shard*.csv"))]).drop_duplicates(["patient", "contrast", "region"])
    b["d"] = (b.R_mean - b.ring_mean) / np.sqrt((b.R_std ** 2 + b.ring_std ** 2) / 2); b["auc"] = np.nan
    b = b.rename(columns={"patient": "case", "n_R": "n_vox"}); b["key"] = "brats:" + b.contrast; b["label"] = b.region.map(xd.BREG)
    for k, g in b.groupby("key"):
        g[["key", "case", "label", "n_vox", "d", "auc"]].to_csv(DATA / f"pol_label_{k.replace(':', '_')}.csv", index=False)


# ---------------- cells ----------------
def _runs(root, base, sub=None):
    pats = [f"*_{base}", f"ablations/*_{base}"] + ([f"{sub}/*_{base}", f"ablations/{sub}/*_{base}"] if sub else [])
    c = [p for pat in pats for p in Path(root).glob(pat) if p.is_dir() and p.name.split("_", 1)[1] == base and len(list(p.glob("fold[0-2]/eval_all.csv"))) == 3]
    if sub:
        c = [p for p in c if f"/{sub}/" in str(p) + "/" or not (Path(root) / sub).is_dir()]
    return c[0] if c else None


def _per_case(run):
    df = pd.concat([pd.read_csv(x) for x in sorted(run.glob("fold[0-2]/eval_all.csv"))]); df = df[df.dice.notna()]
    return df.groupby(["group", "case", "label"]).dice.mean()


def ladder_specs():
    specs = []
    for f in sorted(glob.glob(str(T2 / "*/*/8_results_*/02_metrics/*/*/ablations/ladder_series.json"))):
        d = json.load(open(f)); mroot = Path(f).parents[1]; task = Path(f).parts[-7]; train = mroot.name
        lab = d["labels"]; keys = [Path(k).name.removeprefix("nnUNet_").removeprefix("auglab_") for k in d["run_keys"]]   # some ladders store the category prefix
        real = keys[next(i for i, l in enumerate(lab) if "real fill" in l)]
        flat = next((keys[i] for i, l in enumerate(lab) if "flat fill" in l), None)
        if flat is None:
            g = sorted(mroot.glob("ablations/*_baseline_kmeans_label_remap_voronoi_flatfill_*"))
            flat = g[-1].name.split("_", 1)[1] if g else None
        srcs = [(mroot, task, None, d.get("ood_contrasts") or [])]
        for e in d.get("extra_ood_sources") or []:
            r = Path(e["metrics_root"]); ds = r.parts[r.parts.index("02_tasks") + 2]
            srcs.append((r, ds, e.get("run_subdir"), e.get("items") or ([e["run_subdir"]] if e.get("run_subdir") else None)))
        if task == "toothfairy2":
            for ds in ("hanseg", "pddca"):
                srcs.append((T2 / f"mandible_healthy/{ds}/8_results_{ds}/02_metrics/toothfairy2_model/cbct", ds, None, None))
        specs.append(dict(task=task, train=train, real=real, flat=flat, srcs=srcs))
    return specs


def build_cells():
    pooled, perlab, pat = [], [], []
    names = {}
    for _, _, _, js in vs.LADDERS:
        names.update({k: str(i) for i, k in vs.label_names(js).items()})
    for s in ladder_specs():
        tkey = f"{ALIAS.get(s['task'], s['task'])}:{s['train']}"
        for root, ds, sub, items in s["srcs"]:
            rr, rf = _runs(root, s["real"], sub), _runs(root, s["flat"], sub)
            if rr is None or rf is None:
                print("MISSING", s["task"], s["train"], ds, sub, "real" if rr is None else "flat"); continue
            m = pd.concat({"flat": _per_case(rf), "real": _per_case(rr)}, axis=1).dropna().reset_index()
            for grp in sorted(m.group.unique()):
                if (items is not None and grp not in items) or (ds == s["task"] and grp == s["train"]) or grp.endswith("(in-domain)"):
                    continue
                g = m[m.group == grp].copy(); g["lab"] = g.label.astype(str).map(lambda x: str(xd.BREG.get(x, names.get(x, x))))
                g["lab"] = g.lab.replace({"mandible": "1", "tumour": "1", "lesion": "1", "pancreas": "1"})
                ekey = f"{ALIAS.get(ds, ds)}:{grp}"; cell = f"{tkey}->{ekey}"
                g["pid"] = [_patient_key(c, grp) for c in g.case]
                for lab, gl in g.groupby("lab"):
                    gp = gl.groupby("pid")[["flat", "real"]].mean()
                    if len(gp) >= 5:
                        perlab.append(dict(task=s["task"], cell=cell, train_key=tkey, eval_key=ekey, label=lab, n=len(gp), delta=100 * (gp.real - gp.flat).mean(),
                                           p=wilcoxon(gp.real, gp.flat).pvalue if (gp.real != gp.flat).any() else 1.0))
                gp = g.groupby("pid")[["flat", "real"]].mean()   # labels and same-patient variants averaged per patient
                pooled.append(dict(task=s["task"], cell=cell, train_key=tkey, eval_key=ekey, labels=",".join(sorted(g.lab.unique())), n=len(gp),
                                   delta=100 * (gp.real - gp.flat).mean(), p=wilcoxon(gp.real, gp.flat).pvalue if (gp.real != gp.flat).any() else 1.0))
                pat += [dict(task=s["task"], cell=cell, pid=f"{ds}:{i}", delta=100 * (r.real - r.flat)) for i, r in gp.iterrows()]
    out = []
    for df in (pd.DataFrame(pooled), pd.DataFrame(perlab)):
        df["p_holm"] = np.nan
        for _, idx in df.groupby("task").groups.items():
            df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"].tolist())
        out.append(df)
    out[0].to_csv(DATA / "texture_rule_cells.csv", index=False); out[1].to_csv(DATA / "texture_rule_cells_perlabel.csv", index=False)
    pd.DataFrame(pat).to_csv(DATA / "texture_rule_cells_patient.csv", index=False)
    print(out[0].to_string()); return out


# ---------------- features per cell + tests ----------------
def pol_table():
    v = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "pol_label_*.csv"))]); v["label"] = v.label.astype(str)
    return v.groupby(["key", "label"]).agg(d=("d", "mean"), auc=("auc", "mean"), n_vox=("n_vox", "mean")).reset_index()


def add_features(pooled, perlab, pol):
    P = {(r.key, r.label): r for r in pol.itertuples()}
    perlab = perlab.copy()
    perlab["d_train"] = [getattr(P.get((r.train_key, r.label)), "d", np.nan) for r in perlab.itertuples()]
    perlab["d_eval"] = [getattr(P.get((r.eval_key, r.label)), "d", np.nan) for r in perlab.itertuples()]
    perlab["w"] = [getattr(P.get((r.eval_key, r.label)), "n_vox", np.nan) for r in perlab.itertuples()]
    perlab["flip"] = (np.sign(perlab.d_train) != np.sign(perlab.d_eval)).astype(object)
    perlab.loc[perlab.d_train.isna() | perlab.d_eval.isna(), "flip"] = np.nan
    rows = []
    for r in pooled.itertuples():
        labs = [(l, P.get((r.train_key, l)), P.get((r.eval_key, l))) for l in r.labels.split(",")]
        labs = [(l, a, b) for l, a, b in labs if a is not None and b is not None and np.isfinite(a.d) and np.isfinite(b.d)]
        if not labs:
            rows.append((np.nan, np.nan, np.nan)); continue
        w = np.array([b.n_vox for _, _, b in labs]); pr = np.array([a.d * b.d for _, a, b in labs])
        f = float((w * (pr < 0)).sum() / w.sum()); A = float((w * pr).sum() / (w * np.abs(pr)).sum()) if (w * np.abs(pr)).sum() > 0 else 0.0
        rows.append((f, A, f >= 0.5))
    pooled = pooled.copy(); pooled[["flip_frac", "agree", "flip"]] = pd.DataFrame(rows, index=pooled.index)
    return pooled, perlab


def strat_perm(df, nmax=200000, rng=np.random.RandomState(1)):
    """P1: mean over tasks of within-task [mean Delta(flip) - mean Delta(preserved)], flip labels permuted within task."""
    df = df[df.flip.notna()]; groups = [(g.delta.values, g.flip.values.astype(bool)) for _, g in df.groupby("task") if 0 < g.flip.astype(bool).sum() < len(g)]
    if not groups:
        return np.nan, np.nan, 0

    def stat(assign):
        return np.mean([x[a].mean() - x[~a].mean() for (x, _), a in zip(groups, assign)])
    obs = stat([f for _, f in groups])
    total = np.prod([float(math.comb(len(x), int(f.sum()))) for x, f in groups])   # count before enumerating
    if total <= nmax:
        per = [list(itertools.combinations(range(len(x)), int(f.sum()))) for x, f in groups]
        null = []
        for combo in itertools.product(*per):
            a = []
            for (x, _), c in zip(groups, combo):
                m = np.zeros(len(x), bool); m[list(c)] = True; a.append(m)
            null.append(stat(a))
        null = np.array(null)
    else:
        null = np.array([stat([rng.permutation(f) for _, f in groups]) for _ in range(nmax if len(df) < 200 else 20000)])
    return obs, float((null >= obs - 1e-12).mean()), len(groups)


def unstrat_perm(df, n=100000, rng=np.random.RandomState(2)):
    df = df[df.flip.notna()]; x = df.delta.values; f = df.flip.values.astype(bool)
    if f.all() or not f.any():
        return np.nan, np.nan
    obs = x[f].mean() - x[~f].mean(); null = np.array([(lambda p: x[p].mean() - x[~p].mean())(rng.permutation(f)) for _ in range(n)])
    return obs, float((null >= obs - 1e-12).mean())


def patient_macro(pooled, pat, n=100000, rng=np.random.RandomState(3)):
    """P2: per patient, mean Delta over flip cells - over preserved cells (within task); sign-flip null, equal weight per task."""
    fl = pooled.set_index("cell").flip; pat = pat[pat.cell.isin(fl.dropna().index)].copy(); pat["flip"] = pat.cell.map(fl).astype(bool)
    per_task = []
    for t, g in pat.groupby("task"):
        a = g[g.flip].groupby("pid").delta.mean(); b = g[~g.flip].groupby("pid").delta.mean(); common = a.index.intersection(b.index)
        if len(common) >= 5:
            per_task.append((t, (a[common] - b[common]).values))
    if not per_task:
        return np.nan, np.nan, []
    obs = np.mean([d.mean() for _, d in per_task])
    null = np.array([np.mean([(d * rng.choice([-1, 1], d.size)).mean() for _, d in per_task]) for _ in range(n)])
    return obs, float((null >= obs - 1e-12).mean()), [(t, len(d), d.mean()) for t, d in per_task]


def describe(df, label):
    df = df[df.flip.notna()]; s = df[df.p_holm < 0.05]
    pred = np.where(df.flip.astype(bool), 1, -1); hit = pred == np.sign(df.delta)
    ps = np.where(s.flip.astype(bool), 1, -1); hs = ps == np.sign(s.delta)
    o1 = strat_perm(df); o2 = unstrat_perm(df)
    rho = spearmanr(-df.agree, df.delta) if "agree" in df and len(df) > 3 else (np.nan, np.nan)
    return [f"### {label}: {len(df)} cells ({int(df.flip.astype(bool).sum())} flip / {int((~df.flip.astype(bool)).sum())} preserved), {len(s)} Holm-significant",
            f"- P1 stratified permutation: S = {o1[0]:+.2f} Dice pts over {o1[2]} task(s) with both cell types, one-sided p = {o1[1]:.4g}",
            f"- unstratified: mean Delta flip - preserved = {o2[0]:+.2f}, p = {o2[1]:.4g}; mean Delta flip {df[df.flip.astype(bool)].delta.mean():+.2f}, preserved {df[~df.flip.astype(bool)].delta.mean():+.2f}",
            f"- sign accuracy all cells: rule {int(hit.sum())}/{len(df)} vs always-helps {int((df.delta > 0).sum())}/{len(df)}; significant: rule {int(hs.sum())}/{len(s)} vs always-helps {int((s.delta > 0).sum())}/{len(s)}"
            + (f"; Spearman(-agreement, Delta) = {rho[0]:+.2f} (p={rho[1]:.2g})" if "agree" in df else ""), ""]


def score():
    pooled, perlab = load_cells("texture_rule_cells.csv"), load_cells("texture_rule_cells_perlabel.csv")
    pooled["labels"] = pooled["labels"].astype(str); perlab["label"] = perlab.label.astype(str)
    pat = load_cells("texture_rule_cells_patient.csv"); pol = pol_table()
    pooled, perlab = add_features(pooled, perlab, pol)
    pooled.to_csv(DATA / "texture_rule_cells_scored.csv", index=False); perlab.to_csv(DATA / "texture_rule_cells_perlabel_scored.csv", index=False)
    L = ["# Texture rule: target polarity flip (rung 4.5 flat -> rung 5 real), pre-registered in texture_rule_polarity.py", ""]
    for name, sel in (("CONFIRMATORY (all tasks but BraTS)", lambda d: d[d.task != "brats2024-glioma"]), ("DERIVATION (BraTS)", lambda d: d[d.task == "brats2024-glioma"]), ("ALL tasks", lambda d: d)):
        L += [f"## {name}", ""] + describe(sel(pooled), "pooled cells") + describe(sel(perlab), "per-label cells")
        o = patient_macro(sel(pooled), pat[pat.task.isin(sel(pooled).task.unique())])
        L += [f"- P2 patient-level macro: S2 = {o[0]:+.2f}, one-sided p = {o[1]:.4g}; tasks: " + "; ".join(f"{t} n={n} {m:+.2f}" for t, n, m in o[2]), ""]
    L += ["## Pooled cells", "", "| task | cell | n | labels | flip frac | agreement | flip | Δ real-flat | p Holm |", "|---|---|--:|---|--:|--:|:--:|--:|--:|"]
    for r in pooled.sort_values(["task", "cell"]).itertuples():
        L.append(f"| {r.task} | {r.cell} | {r.n} | {r.labels if len(r.labels) < 20 else str(len(r.labels.split(','))) + ' labels'} | {r.flip_frac:.2f} | {r.agree:+.2f} | {'FLIP' if r.flip == True else ('pres' if r.flip == False else '?')} | {r.delta:+.2f} | {r.p_holm:.2g}{'*' if r.p_holm < 0.05 else ''} |")
    L += ["", "## Polarity d per contrast key and label", "", "```", pol.pivot_table(index="key", columns="label", values="d").round(2).to_string(), "```"]
    (TAB / "texture_rule_polarity.md").write_text("\n".join(L)); print("\n".join(L[:40]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("stage", choices=["extract", "cells", "score"]); ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    {"extract": lambda: extract(a.nproc), "cells": build_cells, "score": score}[a.stage]()
