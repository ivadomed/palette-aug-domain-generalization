#!/usr/bin/env python
"""
texture_rule_shared.py -- SHARED-TEXTURE rule for the texture gain (rung 4.5 flat -> rung 5 real), cell level.  Design written and
committed 2026-10-15 BEFORE the feature below was computed for any contrast (the commit is the time stamp).

Why this rule (FINDINGS 49): 8 of the 12 two-way pairs are the same class in both directions (appearance tasks help both ways,
interface tasks neither way), so a train/eval RATIO (antisymmetric) is capped; the rule must give the same answer both ways. The
analyst HAS seen every cell's Delta (56 cells) and FINDINGS 47-49; the feature has not been computed. Proposed to Paul and accepted.

Feature per (contrast key, label, case): TEXTURE DISTINCTNESS D = how differently the target's interior texture looks from its
surrounding ring, with the mean intensity and the boundary edge removed.
  z = foreground-z-scored native image (cross_dataset_fillswap_model.prep).  All filtering IN-PLANE only (two finest axes;
  sigma 0 along the slice axis) so slice thickness does not enter.  Side-separated normalised convolution (target T and outside
  O = foreground minus target are smoothed separately, never across the boundary, so the step edge cannot leak in):
    hp = z - G_side(z)          (local mean removed, sigma 2 mm)
    E  = sqrt(G_side(hp^2))     (local texture energy, sigma 2 mm)
  interior = target voxels > 1 in-plane voxel from the boundary (all target voxels if < 50 remain); ring = outside voxels with
  in-plane-voxel < distance <= 6 mm (mm-scaled EDT) inside the foreground; label skipped if < 100 target or < 50 ring voxels.
  D = folded AUC of E, interior vs ring = max(AUC, 1 - AUC) in [0.5, 1] (rougher OR smoother than the ring both count as
  "looks different").  Also stored: signed AUC (secondary) and intensity AUC of z (diagnostic only).
  Per key and label: mean of D over <= 40 cases (texture_rule_polarity.manifest cases; BraTS = the 40 internal_ramp patients,
  regions 1-4; AMOS labels remapped to the CHAOS ids; mandible-only sets: label 1).
Cell score: D_train(cell) / D_eval(cell) = mean of D(key, label) over the cell's labels (texture_rule_cells.csv "labels", equal
  weight, as the cell's Delta averages labels);  S = min(D_train, D_eval)   (symmetric: the same both ways).
Outcome: Delta real - flat (Dice points); HELPS iff Delta > 1.  Cells: texture_rule_polarity.load_cells (56; pancreas excluded,
  pelvis + HCC HELD OUT).
Tests (frozen):
  STEP 0 (feature only, run before any outcome is joined): within-task share of the variance of S over the 56 cells
     (= 1 - between-task SS / total SS).  STOP RULE: if < 0.20, S is (almost) a task label and cannot add cell-level power; the
     report then stops there and no outcome is scored.
  P1 PRIMARY: mean over the 6 tasks of the within-task Spearman(S, Delta), one-sided (higher S -> larger gain); null = contrast
     keys permuted WITHIN each task (a key's per-label D values move together; D_train, D_eval, S recomputed), 20k draws.
     Kill: p >= 0.05.  (Within-task only: a one-class task cannot score by its base rate, the FINDINGS 49 problem.)
  P2: binary rule HELPS iff S > theta, theta chosen leave-one-task-out (max hits on the other 5 tasks, ties -> lowest theta);
     out-of-fold hits vs the task label (45/56) and, inside the tasks with both classes, vs the within-task majority.
  P3 HELD-OUT confirmation: pelvis (2 cells) + HCC (4 cells), loaded only here: theta fitted on all 56 cells; predicted vs
     observed class, hits / 6, and Spearman(S, Delta) on the 6 (descriptive; n is small).  Pancreas (6 cells): features computed,
     NOT scored until Paul confirms it may serve as a second confirmation set.
  Secondary (reported, no kill): P1 statistic for the signed AUC version and for D_eval alone; across-task Spearman(S, Delta).
POST-HOC CANDIDATE (added 2026-10-15 after the primary S was seen on 41 of the 56 cells -- BraTS, open-ms, on-harmony, breast --
  where S ran the wrong way across tasks and the signed secondary ran the right way; committed BEFORE the signed score was computed
  on chaos, toothfairy2 or the held-out cells): S_signed = min(signed AUC train, signed AUC eval) (target ROUGHER than its ring on
  both contrasts).  Reported with the same P1 / P2 / P3 machinery as S; its dev-set p is optimistic (41/56 cells seen); its
  confirmatory evidence is (a) the 15 chaos + toothfairy2 cells alone (within-task rho, LOTO hits) and (b) the 6 held-out cells.
  Caveat: the held-out cells' Delta values were seen in FINDINGS 47/48 before they were held out (task means pelvis -1.2, HCC +1.3).
Stages:  extract -> outputs/data/shared_tex_label_<key>.csv ;  score -> outputs/tables/texture_rule_shared.md
"""
from __future__ import annotations
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import rankdata, spearmanr
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import texture_rule_polarity as T  # noqa: E402
import cross_dataset_fillswap_model as xd  # noqa: E402
DATA, TAB = T.DATA, T.TAB; SIG_MM = 2.0; RING_MM = 6.0; MAXC = 40; NPERM = 20000; CUT = 1.0
HELDOUT = {"totalseg-pelvic", "atlas-liver-hcc"}; PANCREAS = {"pansegdata"}
RNG = np.random.RandomState(50)


def _fn(k):
    return DATA / f"shared_tex_label_{k.replace(':', '_').replace('/', '_')}.csv"


def _auc(a, b):
    a, b = xd._sub(a), xd._sub(b); r = rankdata(np.concatenate([a, b]))
    return (r[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size)


def _gside(x, w, sig):
    return gaussian_filter(x * w, sig) / np.maximum(gaussian_filter(w, sig), 1e-6)


def tex_feats(z, fg, mask, zooms, axes, pad_mm=15.0):
    if mask.sum() < 100:
        return None
    idx = np.where(mask); pad = [int(np.ceil(pad_mm / zooms[a])) for a in range(3)]
    sl = tuple(slice(max(0, i.min() - p), i.max() + p + 1) for i, p in zip(idx, pad))
    m = mask[sl]; f = fg[sl] | m; zz = z[sl].astype(np.float32); o = f & ~m
    sig = [0.0, 0.0, 0.0]
    for a in axes:
        sig[a] = SIG_MM / zooms[a]
    mw, ow = m.astype(np.float32), o.astype(np.float32)
    hp = np.where(m, zz - _gside(zz, mw, sig), np.where(o, zz - _gside(zz, ow, sig), 0)).astype(np.float32)
    e = np.where(m, np.sqrt(np.maximum(_gside(hp ** 2, mw, sig), 0)), np.sqrt(np.maximum(_gside(hp ** 2, ow, sig), 0)))
    s_in = max(zooms[a] for a in axes)
    din = distance_transform_edt(m, sampling=zooms); dout = distance_transform_edt(~m, sampling=zooms)
    inner = m & (din > s_in)
    if inner.sum() < 50:
        inner = m
    ring = o & (dout > s_in) & (dout <= RING_MM)
    if ring.sum() < 50:
        return None
    a_tex = _auc(e[inner], e[ring]); a_int = _auc(zz[inner], zz[ring])
    return max(a_tex, 1 - a_tex), a_tex, a_int, int(m.sum())


def case_task(task):
    key, cid, ip, lp, mode = task; rows = []
    try:
        if mode == "brats":
            lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int); c = key.split(":")[1]
            im = nib.load(str(Path(ip) / f"sub-{cid}_{xd.BSUF[c]}"))
        else:
            im = nib.load(str(ip)); lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
            if mode == "amos":
                lbl = sum((lbl == a) * b for a, b in T.AMOS_MAP.items())
        zooms = tuple(float(v) for v in im.header.get_zooms()[:3])
        z, _, fg, axes = xd.prep(np.asarray(im.dataobj), zooms)
        labs = [1] if mode is True else [int(v) for v in np.unique(lbl) if v > 0]
        for lab in labs:
            o = tex_feats(z, fg, lbl == lab, zooms, axes)
            if o:
                rows.append((key, cid, lab, o[3], o[0], o[1], o[2]))
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {key} {cid}: {e!r}", flush=True)
    return rows


def all_keys():
    c = pd.read_csv(DATA / "texture_rule_cells.csv", usecols=["train_key", "eval_key"])   # keys only, no outcome read
    return sorted(set(c.train_key) | set(c.eval_key))


def extract(nproc, only=None):
    man = T.manifest(); tasks = []
    for k in (only or all_keys()):
        if _fn(k).exists():
            continue
        if k.startswith("brats:"):
            tasks += [(k, p, xd.BIDS / f"sub-{p}" / "anat", xd.LABDIR / f"{p}.nii.gz", "brats") for p in xd.brats_patients()[:MAXC]]
            continue
        mode, fn = man[k]; cs = fn()[:MAXC]; print(f"{k}: {len(cs)} cases", flush=True)
        tasks += [(k, c, i, l, mode) for c, i, l in cs]
    with Pool(nproc) as pool:
        res = [r for rs in pool.imap_unordered(case_task, tasks, chunksize=1) for r in rs]
    df = pd.DataFrame(res, columns=["key", "case", "label", "n_vox", "D", "auc_tex", "auc_int"])
    for k, g in df.groupby("key"):
        g.to_csv(_fn(k), index=False); print(k, len(g), "rows", flush=True)


# ---------------- scoring ----------------
def key_label_table(col):
    rows = [pd.read_csv(_fn(k)) for k in all_keys() if _fn(k).exists()]
    d = pd.concat(rows); return d.groupby(["key", "label"])[col].mean()


def cell_scores(c, kl, kmap=None):
    """D_train, D_eval per cell = mean over the cell's labels; kmap optionally relabels keys (permutation null)."""
    kmap = kmap or {}; tr, ev = [], []
    for r in c.itertuples():
        labs = [int(x) for x in str(r.labels).split(",")]
        for key, out in ((kmap.get(r.train_key, r.train_key), tr), (kmap.get(r.eval_key, r.eval_key), ev)):
            v = [kl.get((key, l), np.nan) for l in labs]; out.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
    return np.array(tr), np.array(ev)


def within_rho(c, s):
    return np.nanmean([spearmanr(s[ix], c.delta.values[ix]).correlation for ix in c.groupby("task").indices.values()])


def perm_within(c, kl, score_fn, obs, n=NPERM):
    keys_by_task = {t: sorted(set(g.train_key) | set(g.eval_key)) for t, g in c.groupby("task")}; nl = np.empty(n)
    for i in range(n):
        kmap = {}
        for ks in keys_by_task.values():
            kmap.update(dict(zip(ks, RNG.permutation(ks))))
        nl[i] = within_rho(c, score_fn(*cell_scores(c, kl, kmap)))
    return float((nl >= obs - 1e-12).mean())


def score():
    kl = key_label_table("D"); c = T.load_cells("texture_rule_cells.csv"); assert not c.task.isin(HELDOUT | PANCREAS).any()
    dtr, dev = cell_scores(c, kl); s = np.minimum(dtr, dev); c["D_train"], c["D_eval"], c["S"] = dtr, dev, s
    miss = c[c.S.isna()]
    L = ["# Shared-texture rule S = min(D_train, D_eval) vs texture gain (rung 4.5 flat -> 5 real); design frozen in texture_rule_shared.py", "",
         f"Cells: {len(c)} ({c.task.nunique()} tasks; pancreas excluded, pelvis + HCC held out); cells without S: {len(miss)} {list(miss.cell)}", ""]
    ok = c.S.notna(); cc = c[ok]
    tot = ((cc.S - cc.S.mean()) ** 2).sum(); btw = sum(len(g) * (g.S.mean() - cc.S.mean()) ** 2 for _, g in cc.groupby("task"))
    share = 1 - btw / tot
    L += [f"## STEP 0 (feature only): within-task share of the variance of S = {share:.3f} (stop rule < 0.20)", "",
          "| task | n cells | S mean | S min | S max |", "|---|--:|--:|--:|--:|"]
    L += [f"| {t} | {len(g)} | {g.S.mean():.3f} | {g.S.min():.3f} | {g.S.max():.3f} |" for t, g in cc.groupby("task")]
    if share < 0.20:
        L += ["", "**STOP: S is (almost) constant within tasks, i.e. a task label; no outcome scored.**"]
        (TAB / "texture_rule_shared.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return
    y = (cc.delta > CUT).values
    obs = within_rho(cc, cc.S.values); p1 = perm_within(cc, kl, np.minimum, obs)
    L += ["", f"## P1 (primary): mean within-task Spearman(S, Δ) = {obs:+.3f}, one-sided within-task key-permutation p = {p1:.4g} ({NPERM} draws); kill p >= 0.05", "",
          "- per task: " + "; ".join(f"{t} {spearmanr(g.S, g.delta).correlation:+.2f} (n {len(g)})" for t, g in cc.groupby("task")),
          f"- across all cells (confounded with the task): Spearman(S, Δ) = {spearmanr(cc.S, cc.delta).correlation:+.3f}"]
    # P2
    def best_theta(d):
        th = np.sort(d.S.unique()); hits = [((d.S > t) == (d.delta > CUT)).sum() for t in np.r_[th.min() - 1, th]]
        return np.r_[th.min() - 1, th][int(np.argmax(hits))]
    pred = pd.Series(False, index=cc.index); ths = {}
    for t in cc.task.unique():
        th = best_theta(cc[cc.task != t]); ths[t] = th; te = cc.task == t; pred[te] = cc.S[te] > th
    hit = (pred.values == y); lab = cc.task.isin({"brats2024-glioma", "open-ms", "ispy2"}).values
    mix = cc.groupby("task").filter(lambda g: 0 < (g.delta > CUT).sum() < len(g)).index
    maj = int(cc.loc[mix].groupby("task").apply(lambda g: max((g.delta > CUT).sum(), (g.delta <= CUT).sum())).sum())
    L += ["", f"## P2: leave-one-task-out threshold, out-of-fold hits {hit.sum()}/{len(cc)} (task label {(lab == y).sum()}, never helps {(~y).sum()})",
          f"- tasks with both classes: {int(hit[cc.index.get_indexer(mix)].sum())}/{len(mix)} vs within-task majority {maj}/{len(mix)}",
          "- theta per held-out task: " + "; ".join(f"{t} {v:.3f}" for t, v in ths.items())]
    # secondary
    kls = key_label_table("auc_tex"); a_tr, a_ev = cell_scores(cc, kls); ss = np.minimum(a_tr, a_ev)
    L += ["", "## Secondary (no kill)",
          f"- signed version (target rougher than ring), S' = min(signed AUC train, eval): mean within-task rho {within_rho(cc, ss):+.3f}, p = {perm_within(cc, kls, np.minimum, within_rho(cc, ss), 5000):.3g}",
          f"- D_eval alone: mean within-task rho {within_rho(cc, cc.D_eval.values):+.3f}, p = {perm_within(cc, kl, lambda a, b: b, within_rho(cc, cc.D_eval.values), 5000):.3g}"]
    # post-hoc candidate: signed
    sa, sb = cell_scores(cc, kls); cc["S_signed"] = np.minimum(sa, sb); os_ = within_rho(cc, cc.S_signed.values)
    p_s = perm_within(cc, kls, np.minimum, os_)
    ps = pd.Series(False, index=cc.index); cs = cc.rename(columns={"S": "S_prim", "S_signed": "S"})
    for t in cc.task.unique():
        te = cc.task == t; ps[te] = cs.S[te] > best_theta(cs[cc.task != t])
    hs = ps.values == y; new = cc.task.isin({"chaos", "toothfairy2"}).values
    L += ["", "## POST-HOC CANDIDATE S_signed = min(signed AUC train, eval) (target rougher than ring on both)", "",
          f"- all 56: mean within-task Spearman {os_:+.3f}, within-task key-permutation p = {p_s:.4g} (optimistic: chosen after 41 cells); across-task Spearman {spearmanr(cc.S_signed, cc.delta).correlation:+.3f}",
          f"- LOTO hits {hs.sum()}/{len(cc)} (task label {(lab == y).sum()}); tasks with both classes {int(hs[cc.index.get_indexer(mix)].sum())}/{len(mix)} vs majority {maj}",
          f"- UNSEEN-AT-CHOICE cells (chaos + toothfairy2, n = {new.sum()}): LOTO hits {hs[new].sum()}/{new.sum()} (never helps {(~y[new]).sum()}); "
          f"chaos within-task Spearman {spearmanr(cc.S_signed[cc.task == 'chaos'], cc.delta[cc.task == 'chaos']).correlation:+.2f}",
          "- per task mean S_signed: " + "; ".join(f"{t} {g.S_signed.mean():.3f}" for t, g in cc.groupby("task"))]
    # P3 held out
    th_all = best_theta(cc); h = pd.read_csv(DATA / "texture_rule_cells.csv"); h = h[h.task.isin(HELDOUT)].reset_index(drop=True)
    htr, hev = cell_scores(h, kl); h["S"] = np.minimum(htr, hev); h["pred_helps"] = h.S > th_all; h["helps"] = h.delta > CUT
    L += ["", f"## P3: HELD-OUT pelvis + HCC (theta fitted on all {len(cc)} cells = {th_all:.3f})", "",
          f"- hits {int((h.pred_helps == h.helps).sum())}/{len(h)}; Spearman(S, Δ) = {spearmanr(h.S, h.delta).correlation:+.2f} (descriptive, n = {len(h)})", "",
          "| task | cell | S | predicted | Δ | HELPS |", "|---|---|--:|:-:|--:|:-:|"]
    L += [f"| {r.task} | {r.cell} | {r.S:.3f} | {'helps' if r.pred_helps else 'no'} | {r.delta:+.2f} | {'y' if r.helps else ''} |" for r in h.itertuples()]
    th_s = best_theta(cs); ha, hb = cell_scores(h, kls); h["S_signed"] = np.minimum(ha, hb); h["pred_s"] = h.S_signed > th_s
    L += ["", f"- POST-HOC CANDIDATE on the held-out cells (theta from all 56 = {th_s:.3f}): hits {int((h.pred_s == h.helps).sum())}/{len(h)}; "
          f"Spearman(S_signed, Δ) = {spearmanr(h.S_signed, h.delta).correlation:+.2f}; " + "; ".join(f"{r.cell} {r.S_signed:.3f} ({'helps' if r.pred_s else 'no'})" for r in h.itertuples())]
    L += ["", "Pancreas: not touched (no features computed, no cells scored).", "",
          "## Cells", "", "| task | cell | D train | D eval | S | Δ | HELPS | LOTO pred |", "|---|---|--:|--:|--:|--:|:-:|:-:|"]
    L += [f"| {r.task} | {r.cell} | {r.D_train:.3f} | {r.D_eval:.3f} | {r.S:.3f} | {r.delta:+.2f} | {'y' if r.delta > CUT else ''} | {'helps' if pred[r.Index] else ''} |"
          for r in cc.sort_values(["task", "S"]).itertuples()]
    cc.assign(pred_loto=pred, pred_loto_signed=ps).to_csv(DATA / "texture_rule_shared_cells.csv", index=False)
    (TAB / "texture_rule_shared.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["extract", "score"]); ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--only", default=None); a = ap.parse_args()
    extract(a.nproc, a.only.split(",") if a.only else None) if a.stage == "extract" else score()
