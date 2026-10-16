"""nnUNetv2_train for the training-latency probe (scripts/cluster/latency_probe/launch.sh).

Identical to the `nnUNetv2_train` console entry point except that the post-training
`perform_actual_validation` (sliding-window prediction of every validation case + NIfTI export)
is skipped: the probe only reads the per-epoch "Epoch time" lines nnU-Net writes DURING training,
and the full validation pass would add ~10-30 min per run and thousands of files on a nearly full
scratch. Nothing that happens inside an epoch is changed (same trainer, same data loader, same
augmentation, same per-epoch online validation). Usage: same arguments as nnUNetv2_train.
"""
import sys

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _skip_actual_validation(self, *args, **kwargs):
    self.print_to_log_file("[latency-probe] perform_actual_validation skipped (timing probe)")


nnUNetTrainer.perform_actual_validation = _skip_actual_validation

from nnunetv2.run.run_training import run_training_entry  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "nnUNetv2_train"
    run_training_entry()
