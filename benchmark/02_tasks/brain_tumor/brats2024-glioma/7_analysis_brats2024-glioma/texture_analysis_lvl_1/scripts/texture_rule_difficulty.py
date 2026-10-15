#!/usr/bin/env python
"""
texture_rule_difficulty.py -- does the BASELINE'S IN-DOMAIN DIFFICULTY of a training setting separate where texture helps
(rung 4.5 flat -> rung 5 real)?  (Paul, 2026-10-15.)  Design written and committed BEFORE any in-domain Dice was read.

Why: the paper's appearance/interface label is conceptual (reviewer risk: labels chosen after the results). In-domain difficulty is
a MEASURED quantity every results table already reports; appearance-defined targets are expected to be the hard ones.
Unit: TRAINING SETTING (task, training contrast) -- difficulty is a property of the setting, constant over its eval cells.
  Dev: 14 settings (chaos x2, on-harmony x3, open-ms x2, BraTS x4, ispy2 x2, toothfairy2 x1).
  HELD OUT: pelvis ct, pelvis mri, HCC t1w (3 settings, 6 cells), scored once below.  Pancreas: not read at all.
Difficulty: Dice_ID = baseline (rung 1, plain nnU-Net, the ladder's own run key) Dice on the TRAINING contrast's own test group,
  folds 0-2 averaged, labels averaged per case, same-patient variants merged (ladder_ood_common._patient_key), mean over patients.
Outcome: Delta_setting = mean over the setting's OOD cells (texture_rule_cells.csv) of Delta real - flat (Dice points), equal
  weight per cell; HELPS iff Delta_setting > 1.
Tests (frozen):
  H1 PRIMARY: Spearman(Dice_ID, Delta_setting) < 0 over the 14 dev settings (harder -> larger texture gain); permutation of
     Dice_ID across settings, 200k draws, one-sided.  Kill: p >= 0.05.
  Reported with H1: the same with Dice_ID permuted only WITHIN task (between-task structure kept; asks whether difficulty tracks
     the gain inside a task); task level (6 tasks, means over settings, exact 720-permutation p).
  H2 binary: HARD iff Dice_ID < median of the 14 dev settings -> predict HELPS.  Setting hits /14 vs the task label
     (appearance = brats, open-ms, ispy2) and vs never-helps; cell level (each cell inherits its setting's class) hits /56 vs
     task label and never-helps (descriptive: cells of a setting are not independent).
  H3 HELD OUT: the 3 held-out settings with the dev median threshold: predicted vs observed class (settings and their 6 cells),
     plus Dice_ID and Delta listed.  Caveat: the held-out cells' Delta values were seen in FINDINGS 47/48 (task means pelvis -1.2,
     HCC +1.3) before they were held out; the in-domain Dice values were not computed before this commit.
Output: outputs/tables/texture_rule_difficulty.md, outputs/data/texture_rule_difficulty_settings.csv
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import texture_rule_polarity as T  # noqa: E402
from ladder_ood_common import _patient_key  # noqa: E402
DATA, TAB = T.DATA, T.TAB; CUT = 1.0; NPERM = 200000; RNG = np.random.RandomState(51)
HELDOUT = {"totalseg-pelvic", "atlas-liver-hcc"}; APPEARANCE = {"brats2024-glioma", "open-ms", "ispy2"}


def dice_id():
    rows = []
    for s in T.ladder_specs():
        if s["task"] in T.EXCLUDE_TASKS - HELDOUT:   # pancreas: never read
            continue
        mroot = s["srcs"][0][0]; d = json.load(open(Path(mroot) / "ablations/ladder_series.json"))
        assert d["labels"][0].startswith("baseline"), d["labels"][0]
        base = Path(d["run_keys"][0]).name.removeprefix("nnUNet_").removeprefix("auglab_"); run = T._runs(mroot, base)
        pc = T._per_case(run).reset_index(); pc = pc[pc.group == s["train"]]
        g = pc.groupby("case").dice.mean().reset_index(); g["pid"] = [_patient_key(c, s["train"]) for c in g.case]
        rows.append(dict(task=s["task"], train_key=f"{T.ALIAS.get(s['task'], s['task'])}:{s['train']}", run=run.name,
                         dice_id=100 * g.groupby("pid").dice.mean().mean(), n_pat=g.pid.nunique()))
    return pd.DataFrame(rows)


def perm_p(x, y, groups=None, n=NPERM):
    obs = spearmanr(x, y).correlation; x = np.asarray(x, float); nl = np.empty(n)
    for i in range(n):
        if groups is None:
            xp = RNG.permutation(x)
        else:
            xp = x.copy()
            for g in np.unique(groups):
                ix = np.where(groups == g)[0]; xp[ix] = x[RNG.permutation(ix)]
        nl[i] = spearmanr(xp, y).correlation
    return obs, float((nl <= obs + 1e-12).mean())


def main():
    di = dice_id(); allc = pd.read_csv(DATA / "texture_rule_cells.csv"); allc = allc[~allc.task.isin(T.EXCLUDE_TASKS - HELDOUT)]
    st = allc.groupby(["task", "train_key"]).agg(delta=("delta", "mean"), n_cells=("delta", "size")).reset_index().merge(di, on=["task", "train_key"], how="left")
    assert st.dice_id.notna().all(), st[st.dice_id.isna()]
    dev = st[~st.task.isin(HELDOUT)].reset_index(drop=True); ho = st[st.task.isin(HELDOUT)].reset_index(drop=True)
    assert len(dev) == 14, len(dev)
    r1 = perm_p(dev.dice_id, dev.delta); r1s = perm_p(dev.dice_id, dev.delta, dev.task.values, n=20000)
    tk = dev.groupby("task")[["dice_id", "delta"]].mean(); obs_t = spearmanr(tk.dice_id, tk.delta).correlation
    nl_t = [spearmanr(np.array(p), tk.delta).correlation for p in itertools.permutations(tk.dice_id.values)]
    p_t = float(np.mean(np.array(nl_t) <= obs_t + 1e-12))
    med = dev.dice_id.median(); dev["hard"] = dev.dice_id < med; dev["helps"] = dev.delta > CUT; lab = dev.task.isin(APPEARANCE)
    c = T.load_cells("texture_rule_cells.csv").merge(dev[["train_key", "hard"]], on="train_key"); cy = c.delta > CUT; clab = c.task.isin(APPEARANCE)
    L = ["# Baseline in-domain difficulty vs texture gain (rung 4.5 flat -> 5 real); design frozen in texture_rule_difficulty.py", "",
         f"Dev: {len(dev)} training settings, {len(c)} cells; held out: {len(ho)} settings ({int(ho.n_cells.sum())} cells); pancreas not read.", "",
         f"## H1 (primary): Spearman(Dice_ID, Δ setting) = {r1[0]:+.3f}, one-sided permutation p = {r1[1]:.4g} (n = 14 settings; kill p >= 0.05)", "",
         f"- within-task permutation (between-task structure kept): p = {r1s[1]:.4g}",
         f"- task level (n = {len(tk)}): rho = {obs_t:+.3f}, exact p = {p_t:.4g}", "",
         f"## H2: HARD iff Dice_ID < dev median ({med:.1f}) -> predict HELPS", "",
         f"- settings: {(dev.hard == dev.helps).sum()}/14 (task label {(lab == dev.helps).sum()}, never helps {(~dev.helps).sum()})",
         f"- cells (inherit their setting's class, descriptive): {(c.hard == cy).sum()}/{len(c)} (task label {(clab == cy).sum()}, never helps {(~cy).sum()})", ""]
    ho["hard"] = ho.dice_id < med; ho["helps"] = ho.delta > CUT
    hc = allc[allc.task.isin(HELDOUT)].merge(ho[["train_key", "hard"]], on="train_key")
    L += ["## H3: HELD OUT (dev median threshold)", "",
          f"- settings {(ho.hard == ho.helps).sum()}/{len(ho)}; cells {(hc.hard == (hc.delta > CUT)).sum()}/{len(hc)}", ""]
    L += ["| set | task | training setting | baseline in-domain Dice | patients | Δ real-flat (setting mean) | n cells | HARD | HELPS | task label |",
          "|---|---|---|--:|--:|--:|--:|:-:|:-:|---|"]
    for name, d in (("dev", dev.sort_values("dice_id")), ("HELD OUT", ho.sort_values("dice_id"))):
        L += [f"| {name} | {r.task} | {r.train_key} | {r.dice_id:.1f} | {r.n_pat} | {r.delta:+.2f} | {r.n_cells} | {'y' if r.hard else ''} | {'y' if r.helps else ''} | "
              f"{'appearance' if r.task in APPEARANCE | {'atlas-liver-hcc'} else 'interface'} |" for r in d.itertuples()]
    pd.concat([dev.assign(set="dev"), ho.assign(set="heldout")]).to_csv(DATA / "texture_rule_difficulty_settings.csv", index=False)
    (TAB / "texture_rule_difficulty.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
