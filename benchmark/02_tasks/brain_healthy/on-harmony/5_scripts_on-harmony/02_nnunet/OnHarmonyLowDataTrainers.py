# Registration file for the LOW-DATA texture-step experiment on on-harmony (additive, separate from OnHarmonyTrainers.py so
# no result-producing file changes). nnU-Net's recursive_find_python_class only searches the installed nnunetv2 package,
# so copy this there on every machine that trains or predicts these runs:
#   cp benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony/02_nnunet/OnHarmonyLowDataTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
"""
Low-data variant of the on-harmony AugLab trainer: LowDataMixin (benchmark/00_commun_scripts/00_01_train/lowdata_mixin.py)
layered on nnUNetTrainerOnHarmonyAugLabDefault, the trainer of both arms of the texture step (rung 4.5 flat fill, 04_62 /
04_63; rung 5 PALETTE alone val000, 04_53 / 04_55). The training keys are SESSIONS (51-68 per fold) of only 12 subjects, so
the subset takes N sessions of N DIFFERENT subjects (LOWDATA_SUBJECT_RE, subject = the BIDS sub-<id> prefix). No behaviour
change unless RUN_ID carries `_lowdata_n<NN>`. Project root from NNUNET_PROJECT_ROOT, as OnHarmonyTrainers.py.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "benchmark", "02_tasks", "brain_healthy", "on-harmony", "5_scripts_on-harmony")
    _commun = os.path.join(_root, "benchmark", "00_commun_scripts", "00_01_train")
    for _p in (_root, _scripts, _commun):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from lowdata_mixin import LowDataMixin
    from on_harmony.trainers.auglab_default import nnUNetTrainerOnHarmonyAugLabDefault

    class nnUNetTrainerOnHarmonyAugLabDefaultLowData(LowDataMixin, nnUNetTrainerOnHarmonyAugLabDefault):
        LOWDATA_SUBJECT_RE = r"^(sub-[^_]+)_"

except Exception as _e:  # pragma: no cover -- discovery must not hard-crash nnU-Net import
    print(f"[OnHarmonyLowDataTrainers] import failed: {_e}")
