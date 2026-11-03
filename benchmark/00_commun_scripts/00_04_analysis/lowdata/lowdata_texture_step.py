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

GROUPED (cross-dataset) mode, opt-in: a config with `ood_groups` (and usually `extra_ood_sources`) pools the held-out cases
by TRUE contrast exactly like the ladder engine's grouped mode (ladder_ood_common.run_ladder_grouped, same source format
{metrics_root, run_subdir?, items?} and the same run-key rewriting _src_key / resolve_run_dir, so `ablations/<run id>`
full-data keys work): group value = mean over every paired case of the group's member labels ("<contrast>" own,
"<dataset>/<item>" for a source); OOD = equal-weight mean over groups. Test: sign-flip over PATIENTS blocked across groups
(stat_tests.macro_perm_design, one stratum per group, a patient seen in several groups -- CHAOS's T1-out / T2-SPIR are the
same patients -- flips as one unit; two-sided), pooled patient-merged Wilcoxon for reference. Table aggregation: per fold,
mean over every finite (case, label) value of the group's members, then mean over folds, then mean over groups. A low-data
run is looked up in every source at `<lowdata_dir>/<category>_<RUN_ID>` (rewritten by the source's run_subdir, e.g.
lowdata/t2like/auglab_<RUN_ID>); a level is pending until both arms are complete in EVERY source. Without `ood_groups` the
code path is the original one (Pelvis output unchanged).

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
from ladder_ood_common import (_labelled_case_values, _labelled_case_rows, _patient_key, _merge_by,  # noqa: E402
                               _src_root, _src_key, resolve_run_dir)
from stat_tests import macro_perm_design  # noqa: E402
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


def _own_labels(groups: dict) -> list:
    return sorted({lbl for m in groups.values() for lbl in m if "/" not in lbl})


def grouped_step(root: Path, sources: list, groups: dict, fkey: str, tkey: str, metric: str, higher: bool,
                 scale: float) -> dict:
    """Grouped-mode counterpart of step() (see the module docstring). {} if a member label is missing in either arm."""
    own = _own_labels(groups)
    f = _labelled_case_values(root, own, sources, fkey, metric)
    t = _labelled_case_values(root, own, sources, tkey, metric)
    pairs, missing = {}, []
    for g, members in groups.items():
        pg = {}
        for lbl in members:
            a, b = f.get(lbl, {}), t.get(lbl, {})
            common = [k for k in sorted(set(a) & set(b)) if np.isfinite(a[k]) and np.isfinite(b[k])]
            if not common:
                missing.append(lbl)
            pg.update({f"{lbl}\u00a7{k}": (float(a[k]), float(b[k])) for k in common})
        pairs[g] = pg
    if missing:
        return {"missing": missing}
    K = len(groups)
    flat_v = np.mean([np.mean([p[0] for p in pairs[g].values()]) for g in groups]) * scale
    tex_v = np.mean([np.mean([p[1] for p in pairs[g].values()]) for g in groups]) * scale

    def unit(k):
        lbl, case = k.split("\u00a7", 1)
        return _patient_key(case, lbl)

    entries = [(gi, g, unit(k), (p[1] - p[0]) / len(pairs[g]))
               for gi, g in enumerate(groups) for k, p in pairs[g].items()]
    delta, p_two, _ = macro_perm_design(entries, K, higher_better=higher, scale=scale)
    allp = {k: v for g in groups for k, v in pairs[g].items()}
    x, y = _merge_by(allp, unit)

    def table(key):
        rows = _labelled_case_rows(root, own, sources, key, metric)
        gv = []
        for members in groups.values():
            per_fold = {}
            for lbl in members:
                for cv in rows.get(lbl, {}).values():
                    for (_, fold), v in cv.items():
                        per_fold.setdefault(fold, []).append(v)
            gv.append(np.mean([np.mean(v) for v in per_fold.values()]) if per_fold else float("nan"))
        return float(np.mean(gv)) * scale

    ft, tt = table(fkey), table(tkey)
    per_group = {g: (np.mean([p[0] for p in pairs[g].values()]) * scale,
                     np.mean([p[1] for p in pairs[g].values()]) * scale, len(pairs[g])) for g in groups}
    return {"flat_table": ft, "texture_table": tt, "delta_table": tt - ft, "flat": flat_v, "texture": tex_v,
            "delta": delta, "p_signflip": p_two, "p_wilcoxon": wilcoxon_p(x, y), "n": int(len(x)),
            "per_group": per_group}


def _complete_everywhere(root: Path, sources: list, key: str) -> bool:
    dirs = [resolve_run_dir(root, key)] + [resolve_run_dir(_src_root(s), _src_key(s, key)) for s in sources]
    return all(complete(d) for d in dirs)


def main_grouped(cfg: dict, root: Path) -> None:
    ns_groups = {g: list(m) for g, m in cfg["ood_groups"].items()}
    ind_groups = {c: [c] for c in cfg["in_domain"]}
    sources = []
    for src in cfg.get("extra_ood_sources", []):
        src = dict(src)
        src["metrics_root"] = Path(str(src["metrics_root"]).replace("${PROJECT_ROOT}", str(REPO)))
        sources.append(src)
    fd = cfg["full_data"]
    ld = cfg["lowdata_dir"]
    rows = [("full", fd["n_train"], fd["flat"], fd["texture"])]
    pending = []
    for n, arms in sorted(discover(root / ld).items()):
        keys = {a: f"{ld}/{d.name}" for a, d in arms.items()}
        if set(arms) == {"flat", "texture"} and all(_complete_everywhere(root, sources, k) for k in keys.values()):
            rows.append((f"N={n}", n, keys["flat"], keys["texture"]))
        else:
            pending.append(f"N={n} (" + ", ".join(f"{a}: {arms[a].name} complete_in_all_sources="
                                                  f"{_complete_everywhere(root, sources, keys[a])}"
                                                  for a in sorted(arms)) + ")")
    grp_txt = "; ".join(f"{g} = {' + '.join(m)}" for g, m in ns_groups.items())
    out = [f"# Texture step vs training-set size -- {cfg['task']}", "",
           f"Held-out contrast groups (pooled by true contrast, equal weight per group): {grp_txt}. In-domain: "
           f"{', '.join(cfg['in_domain'])}. Folds 0-2, checkpoint_best. Case-first means (patient = mean over labels and "
           "folds), cases pooled within a group. delta = texture - flat; p = sign-flip over patients, group-stratified, a "
           "patient present in several groups flipping as one unit (stat_tests.macro_perm_design, two-sided); patient-merged "
           "Wilcoxon for reference. Raw p, within-task.", ""]
    for metric, higher, scale in METRICS:
        out += [f"## {metric.upper()}", "",
                "| level | train cases/fold | OOD flat | OOD texture | OOD delta | p sign-flip | p Wilcoxon | n patients "
                "| OOD flat / texture / delta (table) | in-domain flat | in-domain texture | in-domain delta | p (in-domain) "
                "| in-domain flat / texture (table) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        detail = []
        for name, n_train, fk, tk in rows:
            o = grouped_step(root, sources, ns_groups, fk, tk, metric, higher, scale)
            i = grouped_step(root, [], ind_groups, fk, tk, metric, higher, scale)
            if "missing" in o:
                out.append(f"| {name} | {n_train} | missing: {', '.join(o['missing'])} | | | | | | | | | | | |")
                continue
            ii = (f"{i['flat']:.1f} | {i['texture']:.1f} | {i['delta']:+.2f} | {fmt_p(i['p_signflip'])} | "
                  f"{i['flat_table']:.1f} / {i['texture_table']:.1f}" if "missing" not in i else "- | - | - | - | -")
            out.append(f"| {name} | {n_train} | {o['flat']:.1f} | {o['texture']:.1f} | {o['delta']:+.2f} | "
                       f"{fmt_p(o['p_signflip'])} | {fmt_p(o['p_wilcoxon'])} | {o['n']} | "
                       f"{o['flat_table']:.1f} / {o['texture_table']:.1f} / {o['delta_table']:+.2f} | {ii} |")
            detail.append(f"| {name} | " + " | ".join(
                f"{fv:.1f} / {tv:.1f} ({tv - fv:+.2f}, n={nc})" for fv, tv, nc in o["per_group"].values()) + " |")
        out += ["", f"Per held-out group, flat / texture (delta, paired cases) -- {metric}:", "",
                "| level | " + " | ".join(ns_groups) + " |", "|---|" + "---|" * len(ns_groups)] + detail + [""]
    out += ["Runs (keys relative to the metrics root; sources rewrite them with their run_subdir):", ""]
    out += [f"- {name}: flat `{f}`, texture `{t}`" for name, _, f, t in rows]
    out += ["", "Sources: " + ", ".join(f"`{_src_root(s)}`" + (f" (run_subdir {s['run_subdir']})" if s.get("run_subdir")
                                                                 else "") for s in sources)]
    if pending:
        out += ["", "Pending (not reported):", ""] + [f"- {p}" for p in pending]
    dest = root / cfg["output"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n-> {dest}")


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    if cfg.get("ood_groups"):
        return main_grouped(cfg, Path(cfg["metrics_root"].replace("${PROJECT_ROOT}", str(REPO))))
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
