#!/usr/bin/env python
"""
texture_rule_search.py -- which (train contrast, eval contrast) cells gain from texture (rung 4.5 flat -> rung 5 real)?
After the pre-registered polarity rule was killed (texture_rule_polarity.py, FINDINGS 48). Design fixed and committed
2026-10-15 BEFORE the hand features of the new keys (pancreas, pelvis, HCC, amos, sliver07; texture_rule_hand.py) were read.
Cells and Delta: outputs/data/texture_rule_cells.csv / _patient.csv (texture_rule_polarity.py `cells`, 68 pooled cells = every
paper ladder's OOD items). Per-key features: xds_full_hand_shard*.csv (2026-10-02 keys) + xds_full_hand_extra.csv (new keys),
mean over <= 20 cases; BraTS from bigfeat (per region), pooled per patient as the n_R-weighted mean over regions (approximation:
the pooled-label features of the other tasks are computed on the union mask, not averaged).

PART A -- zero-selection test of the four frozen percentile rules (frozen_rule_20261002 = tex_q<=.75 & vis_q<=.5,
  frozen_rule_brats_breast_20261002, frozen_rule_brats_20261007, frozen_rule_brats_breast_20261007), exactly as rule_common:
  vis = con_auc(eval), tex = R_hp_std(eval), mism = |R_mean(eval) - R_mean(train)|, percentile rank within task over that task's
  cells. Reported on cells OUTSIDE each rule's derivation tasks (BraTS; BraTS + breast). They were tuned on the 4 -> 5 step and
  BraTS regions, so this is a transfer test to the texture step.
PART B -- leave-one-TASK-out search over a FIXED family (no other rule is ever scored):
  base features x = vis (con_auc), d (con_d, signed target-vs-ring contrast), tex (R_hp_std), texr (log R_hp_std/ring_hp_std),
  acf (mean R_acf lag1 over the 3 axes), mean (R_mean) ; forms: eval, train, eval - train (mean: |eval - train| only)
  -> 16 candidates. Candidate score = s * (x - median_train_folds(x)) with s = sign of the training-fold Spearman(x, Delta).
  In each fold (one task held out) the candidate with the highest training-fold Spearman is selected; the held-out task's cells get
  its score. Metrics: pooled out-of-fold Spearman(score, Delta) over all cells; sign accuracy (score >= 0 -> texture helps) on all
  cells and on Holm-significant cells vs "always helps".
  Null: the WHOLE pipeline (selection included) rerun on Delta permuted across cells (1000x) and permuted within task (1000x);
  p = share of null Spearman >= observed. The finally reported rule = the candidate selected on ALL tasks, with its in-sample
  stats and the patient-level macro test (cells grouped by its sign, texture_rule_polarity.patient_macro).
Output: outputs/tables/texture_rule_search.md
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import rule_common as R  # noqa: E402
import texture_rule_polarity as T  # noqa: E402
DATA = T.DATA; TAB = T.TAB
COLS = ["con_auc", "con_d", "R_hp_std", "ring_hp_std", "R_mean", "R_acf_ax0_lag1", "R_acf_ax1_lag1", "R_acf_ax2_lag1"]
FROZEN = {"brats 2026-10-02": (dict(combo=[["tex", "lo", 0.75], ["vis", "lo", 0.5]], ops=["&"]), {"brats2024-glioma"}),
          "brats+breast 2026-10-02": (json.load(open(DATA / "frozen_rule_brats_breast_20261002.json")), {"brats2024-glioma", "ispy2"}),
          "brats 2026-10-07": (json.load(open(DATA / "frozen_rule_brats_20261007.json")), {"brats2024-glioma"}),
          "brats+breast 2026-10-07": (json.load(open(DATA / "frozen_rule_brats_breast_20261007.json")), {"brats2024-glioma", "ispy2"})}
RNG = np.random.RandomState(7); NPERM = 1000


def key_table():
    h = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "xds_full_hand_shard*.csv")) + [str(DATA / "xds_full_hand_extra.csv")]])
    k = h.groupby("key")[COLS].mean()
    b = pd.concat([pd.read_csv(f, usecols=["patient", "contrast", "region", "n_R"] + COLS) for f in glob.glob(str(DATA / "bigfeat_shard*.csv"))]).drop_duplicates(["patient", "contrast", "region"])
    bp = b.groupby(["contrast", "patient"]).apply(lambda g: pd.Series(np.average(g[COLS].fillna(g[COLS].mean()), axis=0, weights=g.n_R), index=COLS))
    for c, g in bp.groupby(level=0):
        k.loc[f"brats:{c}"] = g.mean()
    k["acf"] = k[["R_acf_ax0_lag1", "R_acf_ax1_lag1", "R_acf_ax2_lag1"]].mean(axis=1)
    k["texr"] = np.log(k.R_hp_std / k.ring_hp_std)
    return k.rename(columns={"con_auc": "vis", "con_d": "d", "R_hp_std": "tex", "R_mean": "mean"})


def cell_features(c, k):
    c = c[c.train_key.isin(k.index) & c.eval_key.isin(k.index)].copy()
    for f in ("vis", "d", "tex", "texr", "acf"):
        c[f"{f}_eval"] = k.loc[c.eval_key, f].values; c[f"{f}_train"] = k.loc[c.train_key, f].values; c[f"{f}_gap"] = c[f"{f}_eval"] - c[f"{f}_train"]
    c["mean_absgap"] = np.abs(k.loc[c.eval_key, "mean"].values - k.loc[c.train_key, "mean"].values)
    return c


CANDS = [f"{f}_{form}" for f in ("vis", "d", "tex", "texr", "acf") for form in ("eval", "train", "gap")] + ["mean_absgap"]


def fit(tr):
    best = None
    for cnd in CANDS:
        x = tr[cnd].values; ok = np.isfinite(x)
        if ok.sum() < 5:
            continue
        rho = spearmanr(x[ok], tr.delta.values[ok]).correlation
        if np.isfinite(rho) and (best is None or abs(rho) > best[1]):
            best = (cnd, abs(rho), np.sign(rho), np.nanmedian(x) if not cnd.endswith("gap") else 0.0)
    return best


def loto(c, delta):
    c = c.assign(delta=delta); score = pd.Series(np.nan, index=c.index); picks = {}
    for t in c.task.unique():
        tr, te = c[c.task != t], c[c.task == t]
        cnd, _, s, th = fit(tr); picks[t] = cnd
        score[te.index] = s * (te[cnd] - th)
    ok = score.notna()
    return spearmanr(score[ok], c.delta[ok]).correlation, score, picks


def acc(score, c):
    pred = np.where(score >= 0, 1, -1); y = np.sign(c.delta.values); s = (c.p_holm < 0.05).values
    return int((pred == y).sum()), len(c), int(((pred == y) & s).sum()), int(s.sum()), int(((y > 0) & s).sum()), int((y > 0).sum())


def main():
    k = key_table(); cells = T.load_cells("texture_rule_cells.csv"); pat = T.load_cells("texture_rule_cells_patient.csv")
    c = cell_features(cells, k)
    L = ["# Texture rule search (rung 4.5 flat -> rung 5 real), design frozen in texture_rule_search.py", "",
         f"Cells with features: {len(c)} / {len(cells)} (missing keys: {sorted((set(cells.train_key) | set(cells.eval_key)) - set(k.index))})", ""]
    # ---- Part A
    L += ["## Part A: frozen percentile rules (zero selection), cells outside each rule's derivation tasks", "",
          "| rule | cells | hits all (always-helps) | hits sig (always-helps) | Spearman(pred, Δ) | stratified perm p | patient macro p |", "|---|--:|--:|--:|--:|--:|--:|"]
    cf = c.rename(columns={"vis_eval": "vis", "tex_eval": "tex"}).assign(mism=c.mean_absgap, fam=c.task, region="target")
    for name, (rule, deriv) in FROZEN.items():
        d = cf[~cf.task.isin(deriv)].copy()
        for col in ("vis", "tex", "mism"):
            d[col + "_q"] = d.groupby("fam")[col].rank(pct=True)
        pred = R.apply_rule(d, rule["combo"], rule["ops"])
        h, n, hs, ns, ahs, ah = acc(pred.astype(float), d)
        rho = spearmanr(pred, d.delta).correlation if len(set(pred)) > 1 else np.nan
        dd = d.assign(flip=pred > 0); sp = T.strat_perm(dd); pm = T.patient_macro(dd, pat)
        L.append(f"| {name}: `noise wins iff {R.rule_text(rule['combo'], rule['ops'])}` | {n} | {h} ({ah}) | {hs}/{ns} ({ahs}) | {rho:+.2f} | {sp[1]:.3g} | {pm[1]:.3g} |")
    # ---- Part B
    obs, score, picks = loto(c, c.delta.values)
    h, n, hs, ns, ahs, ah = acc(score.values, c)
    nul_u = np.array([loto(c, RNG.permutation(c.delta.values))[0] for _ in range(NPERM)])
    nul_s = []
    for _ in range(NPERM):
        dl = c.delta.copy()
        for _, idx in c.groupby("task").groups.items():
            dl[idx] = RNG.permutation(dl[idx].values)
        nul_s.append(loto(c, dl.values)[0])
    nul_s = np.array(nul_s)
    full = fit(c)
    fs = full[2] * (c[full[0]] - full[3]); fh = acc(fs.values, c); pm = T.patient_macro(c.assign(flip=fs.values >= 0), pat)
    L += ["", "## Part B: leave-one-task-out search over 16 single-feature rules", "",
          f"- out-of-fold Spearman(score, Δ) = {obs:+.3f}; whole-pipeline null p = {(nul_u >= obs).mean():.3g} (Δ permuted across cells), {(nul_s >= obs).mean():.3g} (within task)",
          f"- out-of-fold sign accuracy: all cells {h}/{n} (always-helps {ah}); Holm-significant {hs}/{ns} (always-helps {ahs})",
          "- selected per held-out task: " + "; ".join(f"{t}: {p}" for t, p in picks.items()),
          f"- rule selected on all tasks: {full[0]} (sign {int(full[2]):+d}, threshold {full[3]:.3g}), in-sample |Spearman| {full[1]:.2f}; sign accuracy {fh[0]}/{fh[1]}, significant {fh[2]}/{fh[3]} (always-helps {fh[4]}); "
          f"patient-level macro (cells grouped by its sign) p = {pm[1]:.3g}", "",
          "| candidate | Spearman with Δ, all cells | without BraTS |", "|---|--:|--:|"]
    for cnd in CANDS:
        ok = c[cnd].notna(); nb = ok & (c.task != "brats2024-glioma")
        L.append(f"| {cnd} | {spearmanr(c[cnd][ok], c.delta[ok]).correlation:+.2f} | {spearmanr(c[cnd][nb], c.delta[nb]).correlation:+.2f} |")
    L += ["", "## Cells", "", "| task | cell | Δ | p Holm | " + " | ".join(CANDS) + " |", "|---|---|--:|--:|" + "--:|" * len(CANDS)]
    for r in c.sort_values(["task", "cell"]).itertuples():
        L.append(f"| {r.task} | {r.cell} | {r.delta:+.2f} | {r.p_holm:.2g} | " + " | ".join(f"{getattr(r, x):.3g}" for x in CANDS) + " |")
    c.to_csv(DATA / "texture_rule_search_cells.csv", index=False)
    (TAB / "texture_rule_search.md").write_text("\n".join(L)); print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
