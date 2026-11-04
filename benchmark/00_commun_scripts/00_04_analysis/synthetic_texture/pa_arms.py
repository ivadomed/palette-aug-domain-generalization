"""The two PA fills, taken from the AugLab code itself (no reimplementation).

texture fill = auglab.transforms.gpu.fromSeg.RandomV26_6_2ContrastGPU            (paper rung 5, PA alone)
flat fill    = auglab.transforms.gpu.palette_noisefill.RandomV26_6_2NoiseFillContrastGPU
               with noise_std_range=[0, 0], label_fill_noise=False, label_voronoi=False (paper rung 4.5)
Both use the kwargs of the paper's configs
  transform_params_gpu_v26_6_2_synth_spatialDA_train050.json        (texture)
  transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json (flat)
which differ only in the class and the three noise-fill kwargs. ONE deviation, same in both arms:
blur_sigmas=[0.0] (the 3D separable blur zero-pads the depth axis, so on a depth-1 image it
scales intensities by the kernel centre weight before the label step; see README).
The AugLab package is imported from a pinned copy (env AUGLAB_PIN, default below).
"""
from __future__ import annotations

import os
import sys

import torch

AUGLAB_PIN = os.environ.get("AUGLAB_PIN", "/scratch/paulh/synthetic_texture/auglab_pin")
if AUGLAB_PIN not in sys.path:
    sys.path.insert(0, AUGLAB_PIN)

from auglab.transforms.gpu.fromSeg import RandomV26_6_2ContrastGPU  # noqa: E402
from auglab.transforms.gpu.palette_noisefill import RandomV26_6_2NoiseFillContrastGPU  # noqa: E402

PA_KW = dict(
    c_choices=[2, 3, 4, 5, 6],
    s_choices=[2, 3, 4, 5, 6, 7, 8, 9, 10],
    blur_sigmas=[0.0],  # deviation (paper: [0,0,0,0.3,0.5,0.8]); see module docstring
    dark_threshold=0.01,
    n_kmeans_subsample=10000,
    skip_parcellation_prob=0.1,
    skip_sub_parc_prob=0.4,
    alpha_magnitude_range=[0.5, 2.0],
    label_remap_prob=0.5,
    min_label_voxels=4,
    label_classes=None,
)
P_SYNTH = 0.5  # "train050": per-sample probability of the synthetic image


def make_transform(arm):
    if arm == "texture":
        return RandomV26_6_2ContrastGPU(**PA_KW)
    if arm == "flat":
        return RandomV26_6_2NoiseFillContrastGPU(noise_std_range=[0.0, 0.0], label_fill_noise=False,
                                                 label_voronoi=False, **PA_KW)
    if arm == "plain":
        return None
    raise ValueError(arm)


def zscore_fg(x):
    """Per-sample foreground (x > 0) z-score, background 0. x: [B,1,H,W]."""
    fg = (x > 0).float()
    n = fg.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
    m = (x * fg).sum(dim=(1, 2, 3), keepdim=True) / n
    v = ((x - m) ** 2 * fg).sum(dim=(1, 2, 3), keepdim=True) / n
    return (x - m) / (v.sqrt() + 1e-7) * fg


@torch.no_grad()
def apply_arm(tf, x, y, p=P_SYNTH):
    """x: raw images [B,1,H,W] (background 0), y: labels [B,1,H,W] long.
    Every sample is foreground z-scored; with prob p a sample is replaced by the PA output
    (which AugLab already z-scores over its own foreground)."""
    out = zscore_fg(x)
    if tf is None:
        return out
    sel = torch.rand(x.shape[0], device=x.device) < p
    if sel.any():
        xs = x[sel].unsqueeze(2)  # [b,1,1,H,W] depth-1 volume
        ys = y[sel].unsqueeze(2)
        syn = tf.apply_transform(xs, {"seg": ys}, {})
        out[sel] = syn[:, :, 0]
    return out
