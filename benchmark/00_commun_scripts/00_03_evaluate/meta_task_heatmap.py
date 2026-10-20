#!/usr/bin/env python3
"""
Cross-dataset TASK-LEVEL heatmap — one overall conclusion across all 4 datasets.

One level up from combined_modality_summary.py: that script pools a dataset's TWO
training modalities into one row per method (one column per tested contrast, plus an
`all` column). This script pools ALL 4 DATASETS' combined_modality tables into one
row per method, one COLUMN PER DATASET ("task": brats2024-glioma, chaos, on-harmony,
open-ms — each task's cell = that dataset's own `all` value, i.e. already averaged
over its tested contrasts AND its two training modalities), plus an `overall` column
(equal weight per task) and an inline `sig. vs ref` column — the single "which method
wins overall" table.

Each task's config is one of the existing `*_combined_01_results.yaml` files (chaos/
on-harmony/brats2024-glioma/open-ms) — reused as-is, not re-specified, so this script
can never drift from the per-dataset combined tables it's built on top of.

Statistics (2026-10-20): the cross-task column is a SETTING-level test. A setting =
one training modality of one task; its score is the task's `all` estimand restricted
to that modality's models (setting_value). For each competitor, `settings won` counts
the settings where the ref scores better, and p is the one-sided exact sign test over
settings (stat_tests.sign_test), Holm-corrected across the non-ref competitors.
Patients are the unit only WITHIN a task (the ★/▼ marks: task_design_entries +
macro_perm_design, one task at a time). A patient-pooled test across tasks (the
pre-2026-10-20 column) is deliberately no longer available: it answers a different
question and was far too optimistic.

Meta-config YAML:
  title: "Cross-dataset — task-level heatmap (7-method suite)"
  output_dir: "${PROJECT_ROOT}/benchmark/01_commun_results"
  output_prefix: "meta_task_heatmap"
  ref: auglabAug_v26_6_2_train050_val000
  tasks:
    - name: brats2024-glioma
      dataset: brats2024-glioma
      config: "${PROJECT_ROOT}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/brats_combined_01_results.yaml"
    - name: chaos
      dataset: chaos
      config: "${PROJECT_ROOT}/benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/configs/chaos_combined_01_results.yaml"
    - ...

Opt-in OOD-only mode (default off; without it the output is unchanged):
  ood_only: true          # top level
  tasks:
    - name: healthy-spine-tum
      ...
      ood_exclude:        # optional per task: training modality -> contrast groups
        ct: [spinegan_ct] #   (or flat columns) that are NOT out-of-domain for it
  With `ood_only: true`, every training modality's own models are dropped from the
  contrast groups that are not out-of-domain for that modality; the other
  modalities' models stay in those groups (BraTS's t1n column keeps the t2w/t2f/t1c
  models' cases and loses the t1n model's). Groups per modality, in priority order:
  the task's `ood_exclude[<modality>]`, else the combined config's `in_domain_group`
  (HCC: CE_T1w_all), else the group/column named like the modality. Each group
  expands to all its leaf columns. A listed group or column that the reference
  run's data for that modality does not contain is a hard error, so a misspelt
  name can never silently leave the in-domain cases in. Cells, the within-task
  tests and the pooled test are then computed exactly as in the default mode on
  what remains (a group left empty drops out of the task's equal-weight mean and
  of its test design).

Usage:
  python meta_task_heatmap.py <meta_config.yaml>
Writes: {output_dir}/{output_prefix}_summary.md, _heatmap_dice.png, _heatmap_hd95.png
"""
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "00_00_utils"))
from stat_tests import holm, macro_perm_design, sign_test  # noqa: E402

# Reuse the exact per-modality pooling machinery — one implementation, no drift
# from the per-dataset combined_contrasts tables this rolls up.
def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_agg = _load_sibling("aggregate_from_config")
_cms = _load_sibling("combined_modality_summary")


def _group_columns(node) -> set:
    """Every raw column under a contrast_groups node (dicts recursed)."""
    if isinstance(node, dict) and "column" not in node:
        return set().union(*(_group_columns(v) for v in node.values())) if node else set()
    return {c for c, _ in _agg._leaf_refs(node)}


def apply_ood_exclusions(task: dict, cfg: dict, modalities: list, contrast_groups: dict,
                         ref_key: str) -> list:
    """OOD-only mode: set each modality's `exclude_columns` (read by
    combined_modality_summary's loaders) to the columns of its non-OOD groups.
    Returns human-readable lines describing what was excluded."""
    override = task.get("ood_exclude") or {}
    unknown = set(override) - {m["name"] for m in modalities}
    if unknown:
        sys.exit(f"task {task['name']}: ood_exclude names unknown modalities {sorted(unknown)}")
    notes = []
    for mod in modalities:
        name = mod["name"]
        if name in override:
            groups = list(override[name])
        elif cfg.get("in_domain_group"):
            groups = [cfg["in_domain_group"]]
        else:
            groups = [name]
        present = set()
        run_id = mod["runs"].get(ref_key)
        if run_id:
            present = set(_agg.load_run_cases(mod["sources"], run_id, "dice"))
        cols = set()
        for g in groups:
            if contrast_groups and g in contrast_groups:
                gcols = _group_columns(contrast_groups[g])
            elif contrast_groups:
                sys.exit(f"task {task['name']}: modality {name}: '{g}' is not a contrast group "
                         f"({sorted(contrast_groups)})")
            else:
                gcols = {g}
            if not gcols & present:
                sys.exit(f"task {task['name']}: modality {name}: excluded group '{g}' has no "
                         f"data for the reference run (columns present: {sorted(present)})")
            cols |= gcols
        mod["exclude_columns"] = cols
        notes.append(f"{task['name']} / {name}-trained: excluded {', '.join(groups)}"
                     f" (columns {', '.join(sorted(cols & present))})")
    return notes


def load_task_modalities(task: dict, project_root: str, ood_only: bool = False,
                         ref_key: str = None, notes: list = None) -> tuple:
    """Load one task's own *_combined_01_results.yaml `modalities:` list, with
    METRICS_ROOT set to that task's dataset so its ${METRICS_ROOT}-relative
    metrics_dir paths expand correctly (same pattern as meta_significance.py).
    Returns (modalities, contrast_groups) — contrast_groups is that task's own
    opt-in hierarchical-pooling tree (None if not configured / disabled), read
    straight from the same combined config so this task-level rollup can never
    silently disagree with what that dataset's own combined table shows."""
    cfg_path = Path(os.path.expandvars(task["config"]))
    if not cfg_path.exists():
        print(f"  skip task {task['name']}: config not found {cfg_path}", file=sys.stderr)
        return [], None
    # Derive METRICS_ROOT from the config's own path rather than a hardcoded
    # "benchmark/<dataset>/..." template: active datasets live at
    # benchmark/02_tasks/<task>/<dataset>/ (4 levels below repo root) while
    # archived ones live at benchmark/03_archive/<dataset>/ (3 levels) -- a
    # fixed hop-count can't be right for both (same reasoning as
    # common_env.sh's PROJECT_ROOT fix). Every config here sits at
    # <dataset_root>/5_scripts_<dataset>/06_evaluate/configs/<file>.yaml, so
    # its 5_scripts_<dataset> dir's parent IS <dataset_root>, regardless of depth.
    dataset_root = cfg_path.parents[2].parent
    os.environ["METRICS_ROOT"] = str(dataset_root / f"8_results_{task['dataset']}" / "02_metrics")
    cfg = yaml.safe_load(cfg_path.read_text())
    modalities = []
    for m in cfg["modalities"]:
        if "sources" in m:
            sources = [
                {**src, "metrics_dir": Path(os.path.expandvars(src["metrics_dir"]))}
                for src in m["sources"]
            ]
        else:
            sources = [{"metrics_dir": Path(os.path.expandvars(m["metrics_dir"]))}]
        modalities.append({
            "name": m["name"],
            "sources": sources,
            "runs": m.get("runs", {}),
        })
    contrast_groups = cfg.get("contrast_groups") if cfg.get("use_contrast_groups", True) else None
    if ood_only:
        lines = apply_ood_exclusions(task, cfg, modalities, contrast_groups, ref_key)
        if notes is not None:
            notes += lines
    return modalities, contrast_groups


def task_method_value(modalities: list, method_key: str, metric: str,
                      contrast_groups: dict = None) -> float:
    """This task's `all` cell for one method: cross-fold-class mean per tested
    contrast (pooled over both training modalities), then averaged over
    contrasts, equal weight per contrast — IDENTICAL estimand to
    combined_modality_summary.py's `all` column (same contrast_groups tree,
    when that task's own combined config declares one, so this rollup's
    displayed number always matches the per-dataset table it's built on)."""
    data, _ = _cms.load_combined_run(modalities, method_key)
    if contrast_groups:
        grp_vals = [_agg.resolve_group_value(node, data, metric) for node in contrast_groups.values()]
        finite = [v for v in grp_vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")
    contrasts = sorted(data.get(metric, {}))
    vals = [_agg.cross_fold_class_mean(data, metric, c) for c in contrasts]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def task_design_entries(modalities: list, ref_key: str, comp_key: str, metric: str,
                        contrast_groups: dict) -> list:
    """ONE task's design for its WITHIN-task patient-level sign-flip test
    (macro_perm_design(entries, 1, ...)); the ★/▼ marks.

    Same aggregate_from_config.build_design as the per-task combined tables
    (a patient flips as ONE unit across every contrast group and training modality
    it appears in), entries rescaled by 1/K so the statistic is the task's
    displayed `all` difference. Every entry is in stratum 0: since 2026-10-20 there
    is no task index, so designs of several tasks cannot be stacked into one
    patient-pooled cross-task test. Across tasks, use cross_setting_test."""
    ref_raw = _cms.load_combined_cases_raw(modalities, ref_key, metric)
    comp_raw = _cms.load_combined_cases_raw(modalities, comp_key, metric)
    groups = contrast_groups or {c: c for c in sorted(set(ref_raw) | set(comp_raw))}
    entries, K = _agg.build_design(groups, ref_raw, comp_raw, _agg.DEFAULT_UNIT_SCOPE)
    if not K:
        return []
    return [(0, leaf, unit, c / K) for _, leaf, unit, c in entries]


def setting_value(task: dict, modality: dict, method_key: str, metric: str) -> float:
    """One SETTING's score: the task's `all` estimand restricted to one training
    modality's models (same contrast groups, external cohorts and, in OOD-only mode,
    the same exclusions). NaN if the method has no run for that modality."""
    if not modality["runs"].get(method_key):
        return float("nan")
    return task_method_value([modality], method_key, metric, task["contrast_groups"])


def setting_labels(tasks: list) -> list:
    return [f"{t['name']} {m['name']}" for t in tasks for m in t["modalities"]]


def setting_scores(tasks: list, method_key: str, metric: str) -> list:
    """Scores of one method on every setting of every task, in setting_labels order."""
    return [setting_value(t, m, method_key, metric) for t in tasks for m in t["modalities"]]


def cross_setting_test(tasks: list, a_key: str, b_key: str, metric: str,
                       two_sided: bool = False) -> tuple:
    """THE cross-task comparison: exact sign test over settings that `a_key` scores
    better than `b_key` (one-sided unless two_sided). -> (wins, n non-tied, ties, p)."""
    return sign_test(setting_scores(tasks, a_key, metric), setting_scores(tasks, b_key, metric),
                     metric == "dice", two_sided)


def _per_task_sig(tasks, ref_key, competitors, metric, higher_better):
    """{competitor: [(verdict, p_win, p_lose) per task]} -- each task is its own single
    stratum (patient-level units, within the task only), Holm across the
    competitors within the task, separately per direction."""
    out = {c: [] for c in competitors}
    for i, t in enumerate(tasks):
        pw, pl = [], []
        for comp in competitors:
            ents = task_design_entries(t["modalities"], ref_key, comp, metric,
                                       t["contrast_groups"])
            if ents:
                pw.append(macro_perm_design(ents, 1, higher_better)[2])
                pl.append(macro_perm_design(ents, 1, not higher_better)[2])
            else:
                pw.append(float("nan")); pl.append(float("nan"))
        hw, hl = holm(pw), holm(pl)
        for comp, a, b in zip(competitors, hw, hl):
            v = "win" if a < 0.05 else "lose" if b < 0.05 else "ns"
            out[comp].append((v, a, b))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta_config")
    args = ap.parse_args()

    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    cfg_path = Path(args.meta_config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    title = cfg.get("title", "Cross-dataset task-level heatmap")
    output_prefix = cfg.get("output_prefix", "meta_task_heatmap")
    out_dir = Path(os.path.expandvars(cfg["output_dir"]))
    ref_key = cfg["ref"]

    ood_only = bool(cfg.get("ood_only", False))
    ood_notes = []
    tasks = []
    for t in cfg["tasks"]:
        modalities, contrast_groups = load_task_modalities(t, project_root, ood_only,
                                                           ref_key, ood_notes)
        if modalities:
            tasks.append({"name": t["name"], "modalities": modalities,
                          "contrast_groups": contrast_groups})
    if len(tasks) < 2:
        sys.exit("meta_task_heatmap.py expects >= 2 tasks with resolvable configs.")

    method_order = cfg.get("method_order") or list(tasks[0]["modalities"][0]["runs"].keys())
    if ref_key not in method_order:
        sys.exit(f"`ref: {ref_key}` is not among the method keys: {method_order}")

    # Reference-only rows (val100 mirror): shown, but excluded from bold/best, Holm,
    # significance and any count -- they are not part of the analysis.
    ref_only = [k for k in method_order if "val100" in k]
    method_order = [k for k in method_order if k not in ref_only] + ref_only
    n_analysis = len(method_order) - len(ref_only)

    task_names = [t["name"] for t in tasks]
    cols = task_names + ["overall"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {title}", "",
        f"Generated: {now}  |  Methods: {len(method_order)}  |  Tasks: {', '.join(task_names)}",
        "",
        "Each task column = that dataset's own `all` value (already averaged over its "
        "tested contrasts AND its two training modalities — see its "
        "`combined_contrasts/01_results_summary.md`). `overall` = equal-weight average "
        f"across the {len(tasks)} tasks that have data for that method (a task missing "
        "for a given method — e.g. no HD95, or a method not run on that dataset — is "
        "excluded, not counted as 0). **Bold** = best per column. Across tasks the unit is the "
        "SETTING (one training modality of one task): `settings won by ref` = settings where "
        f"`{_agg._display_key(ref_key)}` scores better than that row, out of the non-tied settings "
        "both were run on; `sig. vs ref` = one-sided exact sign test over those settings, "
        "Holm-corrected across competitors (blank on the ref's own row); **bold** = p < 0.05. "
        "**Per task:** ★ = Ours significantly better than that row *within that task* "
        "(Holm over the competitors, one-sided p<0.05), ▼ = significantly worse, none = n.s.; "
        "last row counts the ★. The val100 mirror is listed last for reference only — "
        "excluded from bold, `overall` ranking, significance and counts.", "",
    ]
    if ood_only:
        lines += ["**OOD-only mode:** each training modality's models are scored on the "
                  "contrast groups that are out-of-domain for it only; the other "
                  "modalities' models still fill those groups. Excluded per setting:", ""]
        lines += [f"- {n}" for n in ood_notes] + [""]

    matrices = {}
    sig_by_metric = {}
    won_by_metric = {}
    task_sig = {}
    for metric, heading, prec in (("dice", "Dice ↑", 4), ("hd95", "HD95 mm ↓", 2)):
        mat = np.full((len(method_order), len(cols)), np.nan)
        for i, key in enumerate(method_order):
            per_task = [task_method_value(t["modalities"], key, metric,
                                          contrast_groups=t["contrast_groups"]) for t in tasks]
            for j, v in enumerate(per_task):
                mat[i, j] = v
            finite = [v for v in per_task if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

        higher_better = metric == "dice"
        competitors = [k for k in method_order[:n_analysis] if k != ref_key]
        task_sig[metric] = _per_task_sig(tasks, ref_key, competitors, metric, higher_better)
        # cross-task: settings are the unit (sign test over settings, Holm across competitors)
        st = {comp: cross_setting_test(tasks, ref_key, comp, metric) for comp in competitors}
        hp = holm([st[c][3] for c in competitors])
        sig_by_metric[metric] = {ref_key: float("nan"), **dict(zip(competitors, hp))}
        won_by_metric[metric] = {c: st[c][:2] for c in competitors}

        best = _agg._best_per_col(mat[:n_analysis], metric)
        sig = sig_by_metric[metric]
        won = won_by_metric[metric]
        ts = task_sig[metric]
        lines += [
            f"## {heading}", "",
            "| method | " + " | ".join(cols) + " | settings won by ref | sig. vs ref |",
            "|" + "---|" * (len(cols) + 3),
        ]
        for i, key in enumerate(method_order):
            cells = []
            for j in range(len(cols)):
                s = _agg._fmt_val(mat[i, j], prec)
                if s != "—" and best[j] == i:
                    s = f"**{s}**"
                if j < len(task_names) and key != ref_key and key not in ref_only:
                    s += {"win": " ★", "lose": " ▼"}.get(ts[key][j][0], "")
                cells.append(s)
            sig_cell = "(ref only)" if key in ref_only else "—" if key == ref_key else _agg._fmt_sig(sig.get(key, float("nan")))
            won_cell = f"{won[key][0]}/{won[key][1]}" if key in won else ""
            lines.append(f"| {_agg._format_label_md(_agg._display_key(key))} | "
                         + " | ".join(cells) + f" | {won_cell} | {sig_cell} |")
        wins = [sum(ts[k][j][0] == "win" for k in competitors) for j in range(len(task_names))]
        lines.append("| **Ours sig. wins** | " + " | ".join(f"{w}/{len(competitors)}" for w in wins)
                     + " | | | |")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{output_prefix}_summary.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    _save_heatmaps(matrices, method_order, sig_by_metric, ref_key, cols, out_dir,
                   output_prefix, title, task_sig, n_analysis)


def _save_heatmaps(matrices, method_order, sig_by_metric, ref_key, cols, out_dir: Path,
                   prefix: str, title: str, task_sig=None, n_analysis=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping heatmaps: {e})", file=sys.stderr)
        return

    ylabels = [_agg._format_label_mathtext(_agg._display_key(k)) for k in method_order]

    specs = (("dice", "Dice ↑", 4, "RdBu"), ("hd95", "HD95 mm ↓", 2, "RdBu_r"))
    for metric, heading, prec, cmap in specs:
        mat = matrices.get(metric)
        if mat is None:
            continue
        sig = sig_by_metric[metric]
        best = _agg._best_per_col(mat[:n_analysis], metric)
        ts = task_sig[metric]

        n_rows, n_data_cols = mat.shape
        n_cols = n_data_cols + 1
        plot_cols = cols + ["sig. vs ref"]
        full = np.full((n_rows, n_cols), np.nan)
        full[:, :n_data_cols] = mat

        fig_w = max(8, 1.4 * n_cols + 5)
        fig_h = max(3, 0.55 * n_rows + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        masked = np.ma.masked_invalid(full)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(plot_cols, rotation=30, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(f"{title} — {heading}", pad=10)

        for i in range(n_rows):
            for j in range(n_data_cols):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v * 100:.1f}" if prec == 4 else f"{v:.1f}"
                    r, g, b, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    fw = "bold" if best[j] == i else "normal"
                    key = method_order[i]
                    if j < n_data_cols - 1 and key in ts:
                        txt += {"win": " ★", "lose": " ▼"}.get(ts[key][j][0], "")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                           color="white" if lum < 0.5 else "black", fontweight=fw)

        for i, key in enumerate(method_order):
            if key == ref_key or i >= n_analysis:
                continue
            p = sig.get(key, float("nan"))
            if not np.isfinite(p):
                continue
            txt = f"{p:.1e}" if p < 1e-4 else f"{p:.3f}"
            starred = p < 0.05
            ax.text(n_data_cols, i, txt, ha="center", va="center", fontsize=9,
                   color="crimson" if starred else "gray",
                   fontweight="bold" if starred else "normal")
        if n_analysis < n_rows:
            ax.axhline(n_analysis - 0.5, color="black", linewidth=1.6)
            ax.text(n_data_cols, n_analysis, "ref only", ha="center", va="center",
                    fontsize=8, color="gray", style="italic")
        ax.axvline(n_data_cols - 0.5, color="black", linewidth=0.8)
        ax.axvline(n_data_cols - 1.5, color="black", linewidth=0.6, linestyle="--")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        wins = [sum(ts[k][j][0] == "win" for k in ts) for j in range(n_data_cols - 1)]
        ax.set_xlabel("task (dataset)  |  sig. vs ref = sign test over settings (Holm)\n★/▼ = Ours sig. better/worse "
                      "within task (Holm, p<0.05).  Ours sig. wins per task: "
                      + ", ".join(f"{w}/{len(ts)}" for w in wins)
                      + "\nrows below the thick line (val100) are reference only — not in the analysis")
        ax.set_ylabel("method")
        fig.tight_layout()

        p_out = out_dir / f"{prefix}_heatmap_{metric}.png"
        fig.savefig(p_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p_out}")


if __name__ == "__main__":
    main()
