#!/usr/bin/env python3
"""
atlas-liver-hcc causal-ablation ladder (CE-T1w-trained model), GROUPED mode
(2026-10-14): same engine and estimand as every other paper ladder
(ladder_ood_common.run_ladder(ood_groups=...) -> run_ladder_grouped): OOD only,
pooled by TRUE held-out contrast with equal weight per contrast, paired Wilcoxon
on patients (a patient scored under several contrasts is one unit).

Training contrast: ATLAS CE-T1w, a mix of arterial/portal/delayed phases plus a few
pre-contrast scans. ATLAS's own test set (12 cases) has only that contrast, so it
contributes nothing to the OOD pool. Held-out contrasts come from the external
lld-mmri-hcc cohort (LLD-MMRI, HCC tumour masks only): T2WI, DWI, Dixon in-phase and
out-phase. Its CE phases (ce-pre/art/pv/del) and all of liverhccseg (CE phases only)
are the training contrast family -> cross-dataset only, kept out of the pool by the
source's "items" filter (same rule as the breast ladders: same contrast != OOD).
lld-mmri-malignant (non-HCC malignant lesions of the same cohort) is not pooled: a
different pathology than the HCC the model is trained on.

Until 2026-10-13 this ladder used run_ladder_cross_dataset (case-weighted pool incl.
the CE phases; 171 cases); that version and its outputs are in git history /
ablations/*.bak_20261014_pre_grouped. Rung 4 = label_voronoi noise fill (lblvor,
2026-10-08), rung 5 = PALETTE alone on val000 (2026-10-07). Rung 4.5 (flat fill,
atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_flatfill_20261010_073719) is
not a ladder rung; paper/scripts/compute_flatfill_decomposition.py swaps it in.

Usage:
  .venv/bin/python 06_13_ladder_summary.py
"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/liver_hcc/atlas-liver-hcc
TASK_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(TASK_ROOT.parents[1] / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder  # noqa: E402

METRICS_ROOT = DATASET_ROOT / "8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"
IN_DOMAIN = "t1w"
OOD_CONTRASTS = []          # the own test set holds only the training contrast

LLD_ROOT = TASK_ROOT / "lld-mmri-hcc/8_results_lld-mmri-hcc/02_metrics/atlas_liver_hcc_model/t1w"
OOD_ITEMS = ["t2wi", "dwi", "inphase", "outphase"]
EXTRA_OOD_SOURCES = [{"metrics_root": LLD_ROOT, "items": OOD_ITEMS}]
OOD_GROUPS = {c: [f"lld-mmri-hcc/{c}"] for c in OOD_ITEMS}

TS = "20260804_062605"
RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     f"atlas-liver-hcc_t1w_baseline_{TS}"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/atlas-liver-hcc_t1w_baseline_kmeans_20260829_165445"),
    ("+label_remap", "+ label remap",
     "ablations/atlas-liver-hcc_t1w_baseline_kmeans_label_remap_20260829_165445"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_lblvor_20261008_095133"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/atlas-liver-hcc_t1w_v26_6_2_train050_val000_20261007_141144"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     f"atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_{TS}"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     f"atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_{TS}"),
]

if __name__ == "__main__":
    run_ladder(task_name="atlas-liver-hcc T1w (CE)", contrast_label="t1w",
               metrics_root=METRICS_ROOT, ablations_root=ABLATIONS_ROOT,
               in_domain=IN_DOMAIN, ood_contrasts=OOD_CONTRASTS, rungs=RUNGS,
               extra_ood_sources=EXTRA_OOD_SOURCES, ood_groups=OOD_GROUPS)
