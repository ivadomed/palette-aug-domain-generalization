"""pansegdata t1wce-trained causal-ablation ladder (baseline -> +kmeans -> +label_remap -> +voronoi noise fill -> v26_6_2 real fill ->
+AugLab val000 -> +AugLab val100). Thin wrapper over the shared 00_03_evaluate/ladder_from_roster.py (rungs come from the roster
pin file, no timestamps here), which calls ladder_ood_common.run_ladder. OOD = every held-out contrast group: t2w (+ TotalSeg T2-like), TotalSeg T1 GRE
and MSD CT (the T1-CE training contrast itself is the only in-domain group).
Usage: .venv/bin/python 06_10_ladder_summary_t1wce.py"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/pancreas_disease/pansegdata
sys.path.insert(0, str(DATASET_ROOT.parents[2] / "00_commun_scripts" / "00_03_evaluate"))
from ladder_from_roster import run  # noqa: E402

# 2026-10-07: + external cohort totalsegmri-pancreas (TotalSegmentator MRI v2), pooled by TRUE held-out contrast (grouped mode, OOD-only report, patient-merged significance).
# Only the item that is genuinely held-out for this training direction enters: item t2like here. The other item (t1gre) is same-contrast (or not provably different) ->
# cross-dataset evidence only, it stays in totalsegmri-pancreas's OWN ladders (06_1X there), never in an OOD bucket. [t1gre part SUPERSEDED 2026-10-22, below]
# 2026-10-07 (2): + msd-pancreas item ct (portal-venous CT, a new modality = genuinely held-out for both directions, own group "ct").
# 2026-10-22 (user decision): + TotalSeg item t1gre as its own OOD group. T1 GRE was kept out above as "not provably different" from
# T1-CE; it is OUT-of-domain for the T1-CE-trained models: their Plain nnU-Net scores 81.4 Dice on PanSegData T1-CE vs 36.5 on T1 GRE,
# a contrast shift, not a cohort shift. This also matches the shared default of meta_task_heatmap.apply_ood_exclusions (only the group
# named like the training modality is in-domain). Pre-change outputs: ablations/*.bak_20261022_pre_t1gre_ood.
TS_ROOT = DATASET_ROOT.parent / "totalsegmri-pancreas" / "8_results_totalsegmri-pancreas/02_metrics/pansegdata_model/t1wce"
EXTRA_OOD_SOURCES = [{"metrics_root": TS_ROOT, "run_subdir": "t2like"}, {"metrics_root": DATASET_ROOT.parent / "msd-pancreas" / "8_results_msd-pancreas/02_metrics/pansegdata_model/t1wce", "run_subdir": "ct"},
                     {"metrics_root": TS_ROOT, "run_subdir": "t1gre"}]
OOD_GROUPS = {"t2w": ["t2w", "totalsegmri-pancreas/t2like"], "ct": ["msd-pancreas/ct"], "t1gre": ["totalsegmri-pancreas/t1gre"]}

if __name__ == "__main__":
    run(dataset_root=DATASET_ROOT, model_type="pansegdata_model", contrast="t1wce",
        ood_contrasts=["t2w"], task_name="pansegdata T1WCE (own-model)", extra_ood_sources=EXTRA_OOD_SOURCES, ood_groups=OOD_GROUPS)
