#!/usr/bin/env python
"""
fill_swap_decomposition.py (Paul, 2026-10-09): the rung 4 -> 5 fill swap is TWO steps,
    4 -> 4.5  noise removal   (Voronoi NOISE fill -> FLAT fill, region means; PALETTE label step kept)
    4.5 -> 5  texture addition (FLAT fill -> REAL intensities)
and "texture hurts" may really be "removing noise hurts". This script puts the three per-cell tables side by side
(outputs/data/visibility_sign_rule_cells_{lblvor,noiseremoval,flatfill}.csv, written by visibility_sign_rule.py with
RUNG4=lblvor / RUNG4=lblvor REAL_ARM=flatfill / RUNG4=flatfill; same cells, same cases, Holm within dataset per step),
classifies which step carries each significant 4->5 effect, and re-scores the candidate sign rules against EACH step:
  - visibility gap (vis_eval - vis_train; frozen 2026-10-07; |gap| < 0.02 = no call, counted as a miss here)
  - the four frozen percentile rules of 2026-10-02 / 2026-10-07 (rule_common: vis / tex / mism percentiles within family)
  - the internal-ramp rule (R_eval < R_train -> noise wins; ramp_out_of_sample per-key R, pooled over labels)
Coverage: pelvis has no hand-feature / ramp extraction (visibility only); ispy1 and acrin6698 have no ramp extraction.
Output: outputs/tables/fill_swap_decomposition.md, outputs/data/fill_swap_decomposition_cells.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import rule_common as R  # noqa: E402
DATA = THIS.parent / "outputs/data"; TAB = THIS.parent / "outputs/tables"
STEPS = {"full": ("lblvor", "4 -> 5"), "noise": ("noiseremoval", "4 -> 4.5 (noise removal)"), "tex": ("flatfill", "4.5 -> 5 (texture)")}
KEY = ["dataset", "train_key", "eval_key", "label"]
FAM = {"ispy2": "breast", "duke-breast-mri": "breast", "ispy1": "breast", "acrin6698": "breast", "chaos": "chaos", "open-ms": "openms",
       "hanseg": "toothfairy2", "pddca": "toothfairy2", "totalseg-pelvic": "pelvis"}
NOCALL = 0.02


def load():
    dfs = {}
    for s, (tag, _) in STEPS.items():
        d = pd.read_csv(DATA / f"visibility_sign_rule_cells_{tag}.csv")
        d["sig"] = d.p_holm < 0.05
        dfs[s] = d[KEY + ["n", "delta", "p_holm", "sig"] + (["label_id", "vis_train", "vis_eval", "gap"] if s == "full" else [])].rename(
            columns={c: f"{c}_{s}" for c in ("n", "delta", "p_holm", "sig")})
    m = dfs["full"].merge(dfs["noise"], on=KEY, how="inner").merge(dfs["tex"], on=KEY, how="inner")
    m["fam"] = m.dataset.map(FAM)
    return m


def star(d, p):
    return f"{d:+.2f}{'*' if p < 0.05 else ''}"


def carrier(r):
    """Which step carries a significant 4->5 effect (same sign & significant)."""
    if not r.sig_full:
        if r.sig_noise and r.sig_tex and np.sign(r.delta_noise) != np.sign(r.delta_tex):
            return "cancel (both steps sig, opposite)"
        if r.sig_noise:
            return "n.s. total; noise-removal step sig"
        if r.sig_tex:
            return "n.s. total; texture step sig"
        return ""
    s = np.sign(r.delta_full)
    a = r.sig_noise and np.sign(r.delta_noise) == s
    b = r.sig_tex and np.sign(r.delta_tex) == s
    if a and b:
        return "both"
    if a:
        return "noise removal"
    if b:
        return "texture"
    return "neither step sig alone"


# ---------------------------------------------------------------- rule predictions (+1 real/next rung wins, -1 previous rung wins)
def pred_visibility(m):
    p = np.where(m.gap >= 0, 1, -1).astype(float)
    p[np.abs(m.gap.values) < NOCALL] = 0   # no call
    return p


def ramp_R():
    """Pooled R per key (voxel-weighted over labels per case, mean over cases) and per (key, label)."""
    out_pool, out_lab = {}, {}
    for f in sorted(DATA.glob("ramp_oos_*.csv")):
        if f.name in ("ramp_oos_pairs.csv", "ramp_oos_spacing.csv"):
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        key = f"{d.dataset.iloc[0]}:{d.contrast.iloc[0]}"
        pc = d.groupby("case").apply(lambda g: np.average(g.R_adj, weights=g.n_vox))
        out_pool[key] = float(pc.mean())
        for lab, g in d.groupby("label"):
            out_lab[(key, str(lab))] = float(g.groupby("case").R_adj.mean().mean())
    return out_pool, out_lab


def rkey(k):
    return k.replace("_uniap", "_uni")   # R was computed on the *_uni duke volumes (lesion interiors identical)


def pred_ramp(m, pool, lab):
    p = []
    for r in m.itertuples():
        if r.label == "POOLED":
            a, b = pool.get(rkey(r.train_key)), pool.get(rkey(r.eval_key))
        else:
            a, b = lab.get((rkey(r.train_key), str(r.label_id))), lab.get((rkey(r.eval_key), str(r.label_id)))
        p.append(np.nan if a is None or b is None else (1.0 if b >= a else -1.0))
    return np.array(p)


FROZEN = {
    "frozen brats 2026-10-02 (tex_q<=.75 & vis_q<=.5)": dict(combo=[["tex", "lo", 0.75], ["vis", "lo", 0.5]], ops=["&"]),
    "frozen brats+breast 2026-10-02": json.load(open(DATA / "frozen_rule_brats_breast_20261002.json")),
    "frozen brats 2026-10-07": json.load(open(DATA / "frozen_rule_brats_20261007.json")),
    "frozen brats+breast 2026-10-07": json.load(open(DATA / "frozen_rule_brats_breast_20261007.json")),
}


def pred_frozen(m):
    """Percentile rules need the hand features (pooled label per key); pelvis keys are absent -> NaN."""
    kf = R.key_features()
    pooled = m[m.label == "POOLED"].copy()
    ok = pooled.train_key.isin(kf.index) & pooled.eval_key.isin(kf.index)
    cells = pd.DataFrame(dict(fam=pooled.fam[ok], train=pooled.train_key[ok], eval=pooled.eval_key[ok], delta=pooled.delta_full[ok],
                              name=pooled.train_key[ok] + "->" + pooled.eval_key[ok]))
    fe = R.feats(cells, kf)
    out = {}
    for name, rule in FROZEN.items():
        p = pd.Series(np.nan, index=m.index)
        p.loc[fe.index] = R.apply_rule(fe, rule["combo"], rule["ops"]).astype(float)
        out[name] = p.values
    return out


def score(pred, delta, sig):
    pred = np.asarray(pred, float); y = np.sign(delta.values); sig = sig.values.astype(bool)
    have = ~np.isnan(pred)
    call = have & (pred != 0)
    loss = (y < 0) & sig
    s = dict(n=int(have.sum()), nocall=int((have & (pred == 0)).sum()), hits=int(((pred == y) & call).sum()),
             n_sig=int((sig & have).sum()), hits_sig=int(((pred == y) & call & sig).sum()),
             n_loss=int((loss & have).sum()), caught=int(((pred < 0) & loss).sum()), false_alarm=int(((pred < 0) & sig & ~loss).sum()),
             always_sig=int(((y > 0) & sig & have).sum()))
    if call.sum() >= 4 and len(np.unique(pred[call])) > 1:
        s["rho"] = spearmanr(pred[call], delta.values[call]).correlation
    else:
        s["rho"] = np.nan
    return s


def fmt(s):
    rho = "" if np.isnan(s["rho"]) else f"{s['rho']:+.2f}"
    return f"{s['hits']}/{s['n']} ({s['nocall']} no call) | {s['hits_sig']}/{s['n_sig']} | {s['caught']}/{s['n_loss']} | {s['false_alarm']} | {s['always_sig']}/{s['n_sig']} | {rho}"


def main():
    m = load()
    m["carrier"] = [carrier(r) for r in m.itertuples()]
    pool, lab = ramp_R(); print("ramp keys:", sorted(pool))
    preds = {"visibility gap (frozen 2026-10-07)": pred_visibility(m), "internal ramp (R_eval >= R_train -> real)": pred_ramp(m, pool, lab)}
    preds.update(pred_frozen(m))
    for k, v in preds.items():
        m["pred_" + k.split(" ")[0] + ("_" + k.split(" ")[1] if k.startswith("frozen") else "")] = v
    m.to_csv(DATA / "fill_swap_decomposition_cells.csv", index=False)

    L = ["# Fill swap decomposed: 4 -> 4.5 (noise removal) and 4.5 -> 5 (texture addition)", "",
         "Δ in Dice points, next rung minus previous rung, mean over the same test cases in all three columns; `*` = Holm-significant within dataset "
         "(Holm run separately per step). Available: abdomen, breast (+duke/ispy1/acrin6698), mandible (+hanseg/pddca), pelvis, open-ms. "
         "Missing (no rung 4.5): BraTS, on-harmony, pancreas.", ""]
    for pooled, title in ((True, "Pooled cells (labels averaged per case)"), (False, "Per-label cells")):
        d = m[(m.label == "POOLED") == pooled]
        L += [f"## {title}: {len(d)} cells", ""]
        # headline counts
        for s, (_, txt) in STEPS.items():
            sg = d[d[f"sig_{s}"]]
            L.append(f"- {txt}: {len(sg)} significant = {int((sg[f'delta_{s}'] > 0).sum())} gains / {int((sg[f'delta_{s}'] < 0).sum())} losses; "
                     f"median Δ over all cells {d[f'delta_{s}'].median():+.2f}")
        r1 = spearmanr(d.delta_noise, d.delta_tex); r2 = spearmanr(d.delta_noise, d.delta_full); r3 = spearmanr(d.delta_tex, d.delta_full)
        L += [f"- Spearman(noise-removal Δ, texture Δ) = {r1.correlation:+.2f} (p={r1.pvalue:.2g}); with the total: noise removal {r2.correlation:+.2f}, texture {r3.correlation:+.2f}",
              "- Share of the total carried by each step, over cells with a significant total: "
              + (lambda sg: f"noise removal median {np.median(sg.delta_noise / sg.delta_full):.0%}, texture {np.median(sg.delta_tex / sg.delta_full):.0%}" if len(sg) else "n/a")(d[d.sig_full]),
              ""]
        vc = d[d.sig_full].carrier.value_counts()
        L += ["Which step carries the significant 4 -> 5 effects: " + ", ".join(f"{k} {v}" for k, v in vc.items()), ""]
        gl = d[d.sig_full & (d.delta_full < 0)]
        L += [f"Significant 4 -> 5 LOSSES ({len(gl)}): " + "; ".join(f"{r.train_key}->{r.eval_key}{'' if pooled else ':' + r.label} {r.delta_full:+.1f} = noise {star(r.delta_noise, r.p_holm_noise)} + tex {star(r.delta_tex, r.p_holm_tex)}" for r in gl.itertuples()), ""]
        tl = d[d.sig_tex & (d.delta_tex < 0)]; nl = d[d.sig_noise & (d.delta_noise < 0)]
        L += [f"Significant TEXTURE losses ({len(tl)}): " + "; ".join(f"{r.train_key}->{r.eval_key}{'' if pooled else ':' + r.label} {r.delta_tex:+.1f} (noise step {star(r.delta_noise, r.p_holm_noise)})" for r in tl.itertuples()),
              f"Significant NOISE-REMOVAL losses ({len(nl)}): " + "; ".join(f"{r.train_key}->{r.eval_key}{'' if pooled else ':' + r.label} {r.delta_noise:+.1f} (texture step {star(r.delta_tex, r.p_holm_tex)})" for r in nl.itertuples()), ""]
        # rules per step
        L += ["### Sign rules scored against each step", "",
              "hits counted on sign; a no-call (visibility |gap| < 0.02) or a missing feature counts as a miss / is excluded (n). "
              "'losses caught' = significant negative cells the rule called 'previous rung wins'; 'always next' = the trivial rule, on the same cells.", ""]
        for s, (_, txt) in STEPS.items():
            L += [f"**{txt}**", "", "| rule | hits all (no call) | hits sig | losses caught | false alarms | always next, sig | Spearman(pred, Δ) |", "|---|--:|--:|--:|--:|--:|--:|"]
            for k, v in preds.items():
                if not pooled and k.startswith("frozen"):
                    continue
                L.append(f"| {k} | {fmt(score(v[d.index], d[f'delta_{s}'], d[f'sig_{s}']))} |")
            L.append("")
        # cell table
        L += ["<details><summary>per-cell table</summary>", "", "| dataset | train | eval | label | n | Δ 4->5 | Δ 4->4.5 noise | Δ 4.5->5 tex | carrier | vis gap |", "|---|---|---|---|--:|--:|--:|--:|---|--:|"]
        for r in d.sort_values(["dataset", "train_key", "eval_key", "label"]).itertuples():
            L.append(f"| {r.dataset} | {r.train_key} | {r.eval_key} | {r.label} | {r.n_full} | {star(r.delta_full, r.p_holm_full)} | {star(r.delta_noise, r.p_holm_noise)} | {star(r.delta_tex, r.p_holm_tex)} | {r.carrier} | {r.gap:+.3f} |")
        L += ["", "</details>", ""]
    out = TAB / "fill_swap_decomposition.md"; out.write_text("\n".join(L)); print("\n".join(L)); print("wrote", out)


if __name__ == "__main__":
    main()
