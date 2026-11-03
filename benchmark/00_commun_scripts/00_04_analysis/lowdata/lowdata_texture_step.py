#!/usr/bin/env python3
"""
Texture step (flat fill -> PALETTE real-fill texture, ladder rungs 4.5 -> 5) as a function of the training-set size.

Question: is the texture gain of an interface-bounded task still ~0 when the task is made HARD by training on few cases
(so that its flat-fill OOD Dice falls into the range of the appearance-defined tasks)? Each data level N is a pair of
runs trained with the SAME low-data subset (LowDataMixin, RUN_ID marker `_lowdata_n<NN>_<arm>_`), arm `flat` and arm
`texture`, otherwise identical to the full-data rung-4.5 / rung-5 runs, which are reported alongside with the same code.

Per level and metric (dice, hd95):
  - flat / texture value on the held-out contrasts: per held-out contrast, mean over patients of each patient's metric
    averaged over labels and folds 0-2 (ladder_ood_common.load_case_means, the ladder engine's case-first estimand);
    contrasts averaged with equal weight. In-domain value reported the same way (shows the difficulty manipulation).
  - delta = texture - flat on the patients present in both runs (per contrast; equal weight per contrast).
  - p = stat_tests.macro_perm: contrast-stratified paired sign-flip test of that delta over patients (two-sided),
    + the engine's pooled paired Wilcoxon (patients merged across contrasts by ladder_ood_common._patient_key) for
    reference. Within-task tests, no correction across levels.
The table also gives each value in the main tables' aggregation ("table": per fold, mean over every finite (case, label)
value of the contrast, then mean over folds; aggregate_from_config.load_run + cross_fold_class_mean), which is what the
paper's tables print (full-data Pelvis MRI-trained flat fill on CT: 72.7 table vs 68.5 case-first). The test is always
the patient-level one above (case-first units), as in the ladder engine.

Config (yaml): task, dataset, metrics_root (may use ${PROJECT_ROOT}), ood_contrasts, in_domain, full_data {flat, texture,
n_train}, lowdata_dir (subdir of metrics_root holding the low-data metric dirs), output (relative to metrics_root).
A level is reported once both arms have eval_all.csv in folds 0-2; otherwise it is listed as pending.

  .venv/bin/python lowdata_texture_step.py <config.yaml>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
from stat_tests import macro_perm, wilcoxon_p, fmt_p  # noqa: E402
from ladder_ood_common import load_case_means, _paired_cases, _merge_patient_pairs  # noqa: E402
from aggregate_from_config import load_run, cross_fold_class_mean  # noqa: E402

LEVEL_RE = re.compile(r"_lowdata_n0*(\d+)_(flat|texture)_")
METRICS = (("dice", True, 100.0), ("hd95", False, 1.0))


def complete(run_dir: Path) -> bool:
    return all((run_dir / f"fold{k}" / "eval_all.csv").exists() for k in (0, 1, 2))


def discover(lowdata_root: Path) -> dict:
    """{N: {arm: run_dir}}; aborts on two runs for one (N, arm): a stale duplicate would be averaged silently."""
    levels: dict = {}
    for d in sorted(lowdata_root.glob("*")) if lowdata_root.is_dir() else []:
        m = LEVEL_RE.search(d.name)
        if not (d.is_dir() and m):
            continue
        n, arm = int(m.group(1)), m.group(2)
        if arm in levels.setdefault(n, {}):
            sys.exit(f"ERROR: two {arm} runs for N={n}: {levels[n][arm].name} and {d.name} (move one aside)")
        levels[n][arm] = d
    return levels


def step(flat_dir: Path, tex_dir: Path, contrasts, metric: str, higher: bool, scale: float, ns: str) -> dict:
    f = load_case_means(flat_dir, metric)
    t = load_case_means(tex_dir, metric)
    pairs = {c: _paired_cases(f.get(c, {}), t.get(c, {}), ns=ns) for c in contrasts}
    if not all(pairs.values()):
        return {}
    flat_v = np.mean([np.mean([p[0] for p in pairs[c].values()]) for c in contrasts]) * scale
    tex_v = np.mean([np.mean([p[1] for p in pairs[c].values()]) for c in contrasts]) * scale
    diffs = [np.array([p[1] - p[0] for p in pairs[c].values()]) for c in contrasts]
    delta, p_two, _ = macro_perm(diffs, higher_better=higher, scale=scale)
    x, y = _merge_patient_pairs(pairs)
    fr, tr = load_run(flat_dir), load_run(tex_dir)
    flat_t = np.mean([cross_fold_class_mean(fr, metric, c) for c in contrasts]) * scale
    tex_t = np.mean([cross_fold_class_mean(tr, metric, c) for c in contrasts]) * scale
    return {"flat_table": flat_t, "texture_table": tex_t, "delta_table": tex_t - flat_t,"flat": flat_v, "texture": tex_v, "delta": delta, "p_signflip": p_two,
            "p_wilcoxon": wilcoxon_p(x, y), "n": int(len(x))}


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    root = Path(cfg["metrics_root"].replace("${PROJECT_ROOT}", str(REPO)))
    ns, ood, ind = cfg["dataset"], list(cfg["ood_contrasts"]), list(cfg["in_domain"])
    fd = cfg["full_data"]
    rows = [("full", fd["n_train"], root / fd["flat"], root / fd["texture"])]
    pending = []
    for n, arms in sorted(discover(root / cfg["lowdata_dir"]).items()):
        if set(arms) == {"flat", "texture"} and all(complete(d) for d in arms.values()):
            rows.append((f"N={n}", n, arms["flat"], arms["texture"]))
        else:
            pending.append(f"N={n} ({', '.join(f'{a}: {d.name} complete={complete(d)}' for a, d in sorted(arms.items()))})")

    out = [f"# Texture step vs training-set size -- {cfg['task']}", "",
           f"Held-out contrasts: {', '.join(ood)}; in-domain: {', '.join(ind)}. Folds 0-2, checkpoint_best. "
           "Case-first means (patient = mean over labels and folds). delta = texture - flat; p = contrast-stratified "
           "paired sign-flip over patients (stat_tests.macro_perm, two-sided); Wilcoxon for reference. Raw p, "
           "within-task.", ""]
    for metric, higher, scale in METRICS:
        out += [f"## {metric.upper()}", "",
                "| level | train cases/fold | OOD flat | OOD texture | OOD delta | p sign-flip | p Wilcoxon | n patients "
                "| OOD flat / texture / delta (table) | in-domain flat | in-domain texture | in-domain delta | p (in-domain) "
                "| in-domain flat / texture (table) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, n_train, fdir, tdir in rows:
            o = step(fdir, tdir, ood, metric, higher, scale, ns)
            i = step(fdir, tdir, ind, metric, higher, scale, ns)
            if not o:
                out.append(f"| {name} | {n_train} | missing | | | | | | | | | | | |")
                continue
            ii = (f"{i['flat']:.1f} | {i['texture']:.1f} | {i['delta']:+.2f} | {fmt_p(i['p_signflip'])} | "
                  f"{i['flat_table']:.1f} / {i['texture_table']:.1f}" if i else "- | - | - | - | -")
            out.append(f"| {name} | {n_train} | {o['flat']:.1f} | {o['texture']:.1f} | {o['delta']:+.2f} | "
                       f"{fmt_p(o['p_signflip'])} | {fmt_p(o['p_wilcoxon'])} | {o['n']} | "
                       f"{o['flat_table']:.1f} / {o['texture_table']:.1f} / {o['delta_table']:+.2f} | {ii} |")
        out.append("")
    out += ["Runs:", ""] + [f"- {name}: flat `{f.name}`, texture `{t.name}`" for name, _, f, t in rows]
    if pending:
        out += ["", "Pending (not reported):", ""] + [f"- {p}" for p in pending]
    dest = root / cfg["output"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
