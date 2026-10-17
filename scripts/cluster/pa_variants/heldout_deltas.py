#!/usr/bin/env python
"""Held-out deltas (exact, recomputed from the symlinked metrics with the tables' own cell function; p-values read from the tables) vs the TamIA control from the tables written by write_configs.py + aggregate_from_config.py.
Per setting and arm: held-out Dice / HD95 (= `all` of vs_ctrl_ho, equal weight per held-out contrast), delta vs ctrl,
p (ctrl better, from vs_ctrl_ho) and p (arm better, the ctrl row of vs_<arm>_ho), training-contrast Dice delta (vs_ctrl),
then the mean over the settings present and the screen rules (rule 2: round-1 candidate; rule 3: extension).
usage: python heldout_deltas.py <agg root> <run_ids.tsv>   -> prints + writes <agg root>/HELDOUT_DELTAS.md"""
import collections, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/00_commun_scripts/00_03_evaluate"))
sys.path.insert(0, str(ROOT / "benchmark/00_commun_scripts/00_00_utils"))
from aggregate_from_config import load_run, cross_fold_class_mean  # noqa: E402 -- same cell math as the tables

REFARM = os.environ.get("PA_REF_ARM", "ctrl")   # the control the deltas are taken against (round 3: ctrl81)
LABEL = {"onh_t1w": "Brain", "openms_flair": "MS", "chaos_t2spir": "Abdomen"}


def table(md, metric):
    """run id -> {column: value} from one '## Dice' / '## HD95' block of an aggregate summary."""
    txt = Path(md).read_text(); blk = txt.split(f"## {metric}")[1].split("\n## ")[0]
    rows = [l for l in blk.splitlines() if l.startswith("|")]
    hdr = [c.strip().strip("*") for c in rows[0].strip("|").split("|")]
    out = {}
    for l in rows[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        run = re.sub(r"\s*\(Ours\)$", "", cells[0].replace("**", "")).strip()
        vals = {}
        for h, c in zip(hdr[1:], cells[1:]):
            c = c.replace("**", "")
            try: vals[h] = float(c)
            except ValueError: vals[h] = None
        out[run] = vals
    return out


def main():
    agg, tsv = Path(sys.argv[1]), Path(sys.argv[2])
    rid = collections.defaultdict(dict); arms = []
    for l in tsv.read_text().splitlines()[1:]:
        if not l.strip() or l.startswith("#"): continue
        s, f, r, p, a = l.split("\t")[:5]; rid[s][a] = r
        if a not in arms and a != REFARM: arms.append(a)
    lines = [f"| setting | arm | held-out Dice | Δ vs {REFARM} | p ref>arm | p arm>ref | held-out HD95 | Δ mm | p ref>arm | p arm>ref | train Dice Δ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    D = collections.defaultdict(dict)
    for s in LABEL:
        if s not in rid or not (agg / s / f"vs_{REFARM}_ho/01_results_summary.md").is_file(): continue
        c = rid[s][REFARM]
        ho = {m: table(agg / s / f"vs_{REFARM}_ho/01_results_summary.md", m) for m in ("Dice", "HD95 mm")}
        held = [h for h in ho["Dice"][c] if h not in ("folds", "all", "sig. vs ref")]
        for run in ho["Dice"]:   # replace the rounded `all` cells by exact held-out means
            d = load_run(agg / s / "metrics" / f"auglab_{run}")
            if not d.get("dice"): continue
            ho["Dice"][run]["all"] = 100 * sum(cross_fold_class_mean(d, "dice", h) for h in held) / len(held)
            ho["HD95 mm"][run]["all"] = sum(cross_fold_class_mean(d, "hd95", h) for h in held) / len(held)
        al = table(agg / s / f"vs_{REFARM}/01_results_summary.md", "Dice")
        ind = [h for h in al[c] if h not in ("folds", "all", "sig. vs ref") and h not in ho["Dice"][c]][0]
        lines.append(f"| {LABEL[s]} | {REFARM} (ref) | {ho['Dice'][c]['all']:.1f} | | | | {ho['HD95 mm'][c]['all']:.1f} | | | | ({ind} {al[c][ind]:.1f}) |")
        for a in arms:
            r = rid[s].get(a)
            if r not in ho["Dice"] or ho["Dice"][r]["all"] is None: continue
            rev = agg / s / f"vs_{a}_ho/01_results_summary.md"
            rv = {m: table(rev, m)[c]["sig. vs ref"] if rev.is_file() else None for m in ("Dice", "HD95 mm")}
            dd = ho["Dice"][r]["all"] - ho["Dice"][c]["all"]; dh = ho["HD95 mm"][r]["all"] - ho["HD95 mm"][c]["all"]
            D[a][s] = (dd, dh)
            f = lambda x: "—" if x is None else (f"**{x:.3f}**" if x < 0.05 else f"{x:.3f}")
            lines.append(f"| {LABEL[s]} | {a} | {ho['Dice'][r]['all']:.1f} | {dd:+.2f} | {f(ho['Dice'][r]['sig. vs ref'])} | {f(rv['Dice'])} | "
                         f"{ho['HD95 mm'][r]['all']:.1f} | {dh:+.2f} | {f(ho['HD95 mm'][r]['sig. vs ref'])} | {f(rv['HD95 mm'])} | "
                         f"{al[r][ind] - al[c][ind]:+.1f} |")
    lines += ["", "| arm | settings | mean held-out ΔDice | min ΔDice | mean ΔHD95 mm | MS ΔDice | rule 2 (r1 candidate) | rule 3 (extend) |", "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if not D[a]: continue
        v = list(D[a].values()); md = sum(x[0] for x in v) / len(v); mn = min(x[0] for x in v); mh = sum(x[1] for x in v) / len(v)
        ms = D[a].get("openms_flair", (None,))[0]
        r2 = "yes" if (md >= -0.5 and ms is not None and ms >= -2.0) else "no"
        r3 = "yes" if (md >= 1.0 and mn >= -2.0 and mh <= 2.0) else "no"
        lines.append(f"| {a} | {'+'.join(LABEL[s] for s in D[a])} | {md:+.2f} | {mn:+.2f} | {mh:+.2f} | "
                     f"{'—' if ms is None else f'{ms:+.2f}'} | {r2} | {r3}{'' if len(D[a]) == 3 else ' (partial)'} |")
    txt = "\n".join(lines) + "\n"; (agg / "HELDOUT_DELTAS.md").write_text(txt); print(txt)


if __name__ == "__main__":
    main()
