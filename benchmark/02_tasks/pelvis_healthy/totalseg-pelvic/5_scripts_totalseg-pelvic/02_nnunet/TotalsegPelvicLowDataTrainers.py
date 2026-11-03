# Registration file for the LOW-DATA experiment on totalseg-pelvic (additive, separate from TotalsegPelvicTrainers.py so
# no result-producing file changes). nnU-Net's recursive_find_python_class only searches the installed nnunetv2 package,
# so copy this there on every machine that trains or predicts these runs:
#   cp benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/02_nnunet/TotalsegPelvicLowDataTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
"""
Low-data variant of the totalseg-pelvic AugLab trainer: LowDataMixin
(benchmark/00_commun_scripts/00_01_train/lowdata_mixin.py) layered on nnUNetTrainerTotalsegPelvicAugLabDefault, the
trainer of both arms of the texture step (rung 4.5 flat fill, 04_34; rung 5 PALETTE alone val000, 04_29). No behaviour
change unless RUN_ID carries `_lowdata_n<NN>`; then each fold keeps N training subjects (fixed-seed, nested subsets,
identical for every arm), validation unchanged. Project root from NNUNET_PROJECT_ROOT, as TotalsegPelvicTrainers.py.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "benchmark", "02_tasks", "pelvis_healthy", "totalseg-pelvic", "5_scripts_totalseg-pelvic")
    _commun = os.path.join(_root, "benchmark", "00_commun_scripts", "00_01_train")
    for _p in (_root, _scripts, _commun):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from lowdata_mixin import LowDataMixin
    from totalseg_pelvic.trainers.auglab_default import nnUNetTrainerTotalsegPelvicAugLabDefault

    class nnUNetTrainerTotalsegPelvicAugLabDefaultLowData(LowDataMixin, nnUNetTrainerTotalsegPelvicAugLabDefault):
        pass

except Exception as _e:  # pragma: no cover -- discovery must not hard-crash nnU-Net import
    print(f"[TotalsegPelvicLowDataTrainers] import failed: {_e}")
