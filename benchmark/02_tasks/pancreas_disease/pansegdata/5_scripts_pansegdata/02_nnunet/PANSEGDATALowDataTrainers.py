# Registration file for the LOW-DATA texture-step experiment on pansegdata (additive, separate from PANSEGDATATrainers.py so
# no result-producing file changes). nnU-Net's recursive_find_python_class only searches the installed nnunetv2 package,
# so copy this there on every machine that trains or predicts these runs:
#   cp benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/02_nnunet/PANSEGDATALowDataTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
"""
Low-data variant of the pansegdata AugLab trainer: LowDataMixin (benchmark/00_commun_scripts/00_01_train/lowdata_mixin.py)
layered on nnUNetTrainerPANSEGDATAAugLabDefault, the trainer of both arms of the texture step (rung 4.5 flat fill, 04_33 /
04_34; rung 5 PALETTE alone val000, 04_18 / 04_22). No behaviour change unless RUN_ID carries `_lowdata_n<NN>`; then each
fold keeps N training subjects (fixed-seed, nested subsets, identical for every arm), validation unchanged. Project root
from NNUNET_PROJECT_ROOT, as PANSEGDATATrainers.py.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "benchmark", "02_tasks", "pancreas_disease", "pansegdata", "5_scripts_pansegdata")
    _commun = os.path.join(_root, "benchmark", "00_commun_scripts", "00_01_train")
    for _p in (_root, _scripts, _commun):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from lowdata_mixin import LowDataMixin
    from pansegdata.trainers.auglab_default import nnUNetTrainerPANSEGDATAAugLabDefault

    class nnUNetTrainerPANSEGDATAAugLabDefaultLowData(LowDataMixin, nnUNetTrainerPANSEGDATAAugLabDefault):
        pass

except Exception as _e:  # pragma: no cover -- discovery must not hard-crash nnU-Net import
    print(f"[PANSEGDATALowDataTrainers] import failed: {_e}")
