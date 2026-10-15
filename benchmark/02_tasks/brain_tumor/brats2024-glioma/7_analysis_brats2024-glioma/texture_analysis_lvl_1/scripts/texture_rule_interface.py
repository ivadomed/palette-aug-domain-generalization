#!/usr/bin/env python
"""
texture_rule_interface.py -- is the texture gain (rung 4.5 flat -> rung 5 real) governed by how sharply the target's interface is
marked by an intensity edge on the TRAINING contrast?  (Paul's lead, 2026-10-15.)  Definition, hypothesis and tests written and
committed BEFORE any interface value was computed or looked at.

Unit: TRAINING SETTING = (task, training contrast), 19 settings, instead of the task (10) used by the paper's appearance-vs-interface
split. A measured quantity replaces the conceptual label.

Metric (per contrast key, from the existing hand features: xds_full_hand_shard*.csv + xds_full_hand_extra.csv, BraTS bigfeat; 1 mm
grid, z-scored foreground, gradient magnitude |grad z|; R = target eroded by 1 voxel, shell = the target's outer 1-voxel layer,
ring = 1-5 voxels outside):
  ETR (edge-to-texture ratio) = mean |grad| on the boundary shell / mean(mean |grad| inside R, mean |grad| in the ring)
     shell gradient = bnd_grad_ratio x brain_gm_mean (bnd_grad_ratio is stored relative to the whole-foreground mean gradient).
  ETR >> 1: the interface is a gradient that stands out above the texture on both sides (Canny-type edge-vs-noise contrast);
  ETR ~ 1: the boundary is not marked by an edge stronger than the surrounding texture, so the target must be recognised by its
  appearance. Per case log(ETR), mean over <= 20 cases (BraTS: per patient, regions combined by n_R weight; approximation, the
  BraTS shells include internal sub-region boundaries).
  Secondary metric: log(bnd_grad_ratio) (boundary gradient vs whole-image mean gradient).
Outcome per setting: Delta_setting = mean over the setting's OOD cells (texture_rule_cells.csv) of Delta real-flat (Dice points),
  equal weight per cell.
H1 (primary): Spearman(log ETR of the training contrast, Delta_setting) < 0 (sharper interface -> smaller texture gain).
  Test: permutation of the metric across the 19 settings, 200k draws, one-sided.  Kill: p >= 0.05.
Robustness (reported whatever H1 gives): leave-one-task-out rho; the same at the task level (10 tasks, metric averaged over a task's
  settings); cell level (68 cells, the cell carries its setting's metric; null permutes metrics across settings so a setting's cells
  move together); secondary metric; eval-contrast and min(train, eval) versions (exploratory).
H2 (two-sided rationale, added 2026-10-15 before any metric value was computed; Paul: "interface defined => texture helps not at all
  or only a little; appearance defined => texture does help"):
  class: INTERFACE-defined iff log ETR(train) > median over the 19 settings, else APPEARANCE-defined.
  bound: "a little" = 2 Dice points.
  per-setting prediction, 95 % CI of Delta_setting from a patient bootstrap (resample patients within each of the setting's
  cells, 10k draws): INTERFACE -> upper bound < 2;  APPEARANCE -> lower bound > 2 (tightened from > 0 the same day, Paul,
  still before any metric value was read).
  H2b: same with bound 1 Dice on BOTH sides (interface upper CI < 1, appearance lower CI > 1), chosen by Paul blind to results,
  AFTER the analyst had run a preliminary H2 on 14 of 19 settings (texture_rule_interface_PRELIM_14settings.md); the 5 then-missing
  settings (pancreas x2, pelvis x2, HCC) were unseen. Reported next to H2, neither replaces the other.
  global tests: (a) rank-sum Delta(appearance) > Delta(interface) over settings, permutation of the class labels (200k), one-sided;
  (b) number of settings meeting their prediction, null = class labels permuted (same draws).  Kill: (a) p >= 0.05.
  Comparison row (not a test of the metric): the same with the paper's conceptual task labels.
METRIC SWITCH (Paul, 2026-10-15, after the 14-setting preliminary on ETR, before the edge metric below was computed): "a pure edge
  metric, don't try to measure texture". PRIMARY metric for H1, H2 and H2b is now
    EDGE = log mean |grad z| on the target's boundary shell (= bnd_grad_ratio x brain_gm_mean; z = foreground-z-scored image on the
    1 mm grid, so units are global-SD per mm; no target-interior or ring term).
  Higher EDGE = the interface is a strong intensity step = INTERFACE-defined. H1: Spearman(EDGE train, Delta) < 0. H2/H2b classes:
  EDGE above the median of the settings -> interface. ETR results are kept in the report as the superseded metric. The analyst had seen
  the outcomes and the ETR / bnd_grad_ratio values on 14 settings, not EDGE.
Output: outputs/tables/texture_rule_interface.md
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import texture_rule_polarity as T  # noqa: E402
DATA = T.DATA; TAB = T.TAB; NPERM = 200000; RNG = np.random.RandomState(11)
C = ["bnd_grad_ratio", "brain_gm_mean", "R_gm_mean", "ring_gm_mean"]


def per_case(df):
    df = df.copy()
    df["log_etr"] = np.log(df.bnd_grad_ratio * df.brain_gm_mean / ((df.R_gm_mean + df.ring_gm_mean) / 2))
    df["log_bgr"] = np.log(df.bnd_grad_ratio)
    df["edge"] = np.log(df.bnd_grad_ratio * df.brain_gm_mean)
    return df


def key_metrics():
    h = per_case(pd.concat([pd.read_csv(f, usecols=["key", "case"] + C) for f in glob.glob(str(DATA / "xds_full_hand_shard*.csv")) + glob.glob(str(DATA / "xds_full_hand_extra.csv"))]))
    k = h.groupby("key")[["log_etr", "log_bgr", "edge"]].mean()
    k["n_cases"] = h.groupby("key").case.nunique()
    b = pd.concat([pd.read_csv(f, usecols=["patient", "contrast", "region", "n_R"] + C) for f in glob.glob(str(DATA / "bigfeat_shard*.csv"))]).drop_duplicates(["patient", "contrast", "region"])
    b = per_case(b).dropna(subset=["log_etr"])
    bp = b.groupby(["contrast", "patient"]).apply(lambda g: pd.Series({m: np.average(g[m], weights=g.n_R) for m in ("log_etr", "log_bgr", "edge")}))
    for c, g in bp.groupby(level=0):
        k.loc[f"brats:{c}", ["log_etr", "log_bgr", "edge", "n_cases"]] = [g.log_etr.mean(), g.log_bgr.mean(), g.edge.mean(), len(g)]
    return k


def perm_rho(x, y, n=NPERM):
    x, y = np.asarray(x, float), np.asarray(y, float); obs = spearmanr(x, y).correlation
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); null = np.empty(n)
    for i in range(n):
        null[i] = np.corrcoef(RNG.permutation(rx), ry)[0, 1]
    return obs, float((null <= obs + 1e-12).mean())


def cell_perm(c, col, n=20000):
    """cells carry their setting's metric; permute metrics across settings (a setting's cells move together)."""
    st = c.groupby("train_key")[col].first(); keys = st.index.values; obs = spearmanr(c[col], c.delta).correlation; null = np.empty(n)
    for i in range(n):
        mp = dict(zip(keys, RNG.permutation(st.values))); null[i] = spearmanr(c.train_key.map(mp), c.delta).correlation
    return obs, float((null <= obs + 1e-12).mean())


def main():
    k = key_metrics(); cells = T.load_cells("texture_rule_cells.csv")
    for col in ("log_etr", "log_bgr", "edge"):
        cells[col] = cells.train_key.map(k[col]); cells[col + "_eval"] = cells.eval_key.map(k[col])
    cells["edge_min"] = np.minimum(cells.edge, cells.edge_eval)
    st = cells.groupby(["task", "train_key"]).agg(delta=("delta", "mean"), n_cells=("delta", "size"), log_etr=("log_etr", "first"), log_bgr=("log_bgr", "first"), edge=("edge", "first")).reset_index()
    miss = st[st.edge.isna()]
    st = st.dropna(subset=["edge"])
    L = ["# Interface sharpness of the training contrast vs texture gain (rung 4.5 flat -> 5 real); design frozen in texture_rule_interface.py", "",
         f"Settings with a metric: {len(st)} (missing: {', '.join(miss.train_key) or 'none'})", ""]
    r1 = perm_rho(st.edge, st.delta)
    L += [f"## H1 (primary): Spearman(EDGE train = log boundary |grad z|, Δ setting) = {r1[0]:+.3f}, one-sided permutation p = {r1[1]:.4g} (n = {len(st)} settings)", ""]
    lo = [(t, spearmanr(st[st.task != t].edge, st[st.task != t].delta).correlation) for t in st.task.unique()]
    L += ["- leave-one-task-out rho: " + "; ".join(f"-{t} {r:+.2f}" for t, r in lo)]
    tk = st.groupby("task")[["edge", "delta"]].mean(); r2 = perm_rho(tk.edge, tk.delta)
    L += [f"- task level (n = {len(tk)}): rho = {r2[0]:+.3f}, p = {r2[1]:.4g}"]
    cc = cells.dropna(subset=["edge"]); r3 = cell_perm(cc, "edge")
    L += [f"- cell level (n = {len(cc)} cells, setting-block permutation): rho = {r3[0]:+.3f}, p = {r3[1]:.4g}"]
    r4 = perm_rho(st.log_bgr, st.delta); r5 = perm_rho(st.log_etr, st.delta)
    L += [f"- secondary metric log(bnd_grad_ratio): rho = {r4[0]:+.3f}, p = {r4[1]:.4g}"]
    L += [f"- superseded metric log ETR (edge-to-texture, the 14-setting preliminary): rho = {r5[0]:+.3f}, p = {r5[1]:.4g}"]
    for col in ("edge_eval", "edge_min"):
        ok = cells[col].notna(); L += [f"- exploratory, cell level, {col}: rho = {spearmanr(cells[col][ok], cells.delta[ok]).correlation:+.3f}"]
    L += h2(st, cells, 2.0) + h2(st, cells, 1.0)
    L += ["", "| task | training setting | n OOD cells | EDGE (train) | log ETR | log bnd/brain grad | Δ real-flat (mean over cells) |", "|---|---|--:|--:|--:|--:|--:|"]
    for r in st.sort_values("edge").itertuples():
        L.append(f"| {r.task} | {r.train_key} | {r.n_cells} | {r.edge:+.3f} | {r.log_etr:+.3f} | {r.log_bgr:+.3f} | {r.delta:+.2f} |")
    L += ["", "## Metric per contrast key", "", "```", k.round(3).to_string(), "```"]
    cells.to_csv(DATA / "texture_rule_interface_cells.csv", index=False); st.to_csv(DATA / "texture_rule_interface_settings.csv", index=False)
    (TAB / "texture_rule_interface.md").write_text("\n".join(L)); print("\n".join(L))


CONCEPT_INTERFACE = {"chaos", "on-harmony", "toothfairy2", "pansegdata", "totalseg-pelvic"}


def setting_ci(task, tkey, pat, cells, n=10000, rng=np.random.RandomState(5)):
    cs = cells[(cells.train_key == tkey)].cell.tolist(); arrs = [pat[pat.cell == c].delta.values for c in cs]
    arrs = [a for a in arrs if len(a)]
    boots = np.mean([[rng.choice(a, a.size).mean() for _ in range(n)] for a in arrs], axis=0)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def h2_block(st, cls, name, BOUND):
    x = st.delta.values; a = (cls == "appearance")
    def u(lbl):
        xa, xi = x[lbl], x[~lbl]
        return np.mean([(p > q) + 0.5 * (p == q) for p in xa for q in xi]) if xa.size and xi.size else np.nan
    def met(lbl):
        return int((((lbl) & (st.lo.values > BOUND)) | ((~lbl) & (st.hi.values < BOUND))).sum())
    obs_u, obs_m = u(a), met(a); nu, nm = [], []
    for _ in range(20000):
        p = RNG.permutation(a); nu.append(u(p)); nm.append(met(p))
    nu, nm = np.array(nu), np.array(nm)
    return [f"### {name}: {int(a.sum())} appearance / {int((~a).sum())} interface settings",
            f"- (a) P(Δ appearance > Δ interface) = {obs_u:.3f}, one-sided permutation p = {(nu >= obs_u - 1e-12).mean():.4g}",
            f"- (b) settings meeting their prediction: {obs_m}/{len(st)} (interface: upper CI < {BOUND:g}; appearance: lower CI > {BOUND:g}), permutation p = {(nm >= obs_m).mean():.4g}", ""]


def h2(st, cells, BOUND):
    pat = T.load_cells("texture_rule_cells_patient.csv")
    st = st.copy(); ci = [setting_ci(r.task, r.train_key, pat, cells) for r in st.itertuples()]
    st["lo"], st["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    med = st.edge.median(); st["cls"] = np.where(st.edge > med, "interface", "appearance")
    L = ["", f"## H2{'' if BOUND == 2 else 'b'} (two-sided, bound {BOUND:g}): interface (EDGE > median {med:+.3f}) -> Δ upper CI < {BOUND:g}; appearance -> Δ lower CI > {BOUND:g}", ""]
    L += h2_block(st, st.cls.values, "measured classes (EDGE, median split)", BOUND)
    cst = st.assign(cls=np.where(st.task.isin(CONCEPT_INTERFACE), "interface", "appearance"))
    L += h2_block(cst, cst.cls.values, "comparison: the paper's conceptual task labels", BOUND)
    L += ["| setting | EDGE | measured class | conceptual class | Δ | 95 % CI | prediction met (measured) |", "|---|--:|---|---|--:|---|:--:|"]
    for r, cc in zip(st.sort_values("edge").itertuples(), cst.loc[st.sort_values("edge").index].cls):
        met = (r.cls == "appearance" and r.lo > BOUND) or (r.cls == "interface" and r.hi < BOUND)
        L.append(f"| {r.train_key} | {r.edge:+.3f} | {r.cls} | {cc} | {r.delta:+.2f} | [{r.lo:+.2f}, {r.hi:+.2f}] | {'YES' if met else 'no'} |")
    return L


if __name__ == "__main__":
    main()
