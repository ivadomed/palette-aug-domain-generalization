# Visual proof v19: weight-level evidence of fine-scale kernel content (FINDINGS 46; pre-registered: PREREG.md)

## Part A: DCT order spectra of every conv layer, 15 training settings

15 settings x arms noise / flat / real x folds 0, 1, 2 + 3 random-init networks per setting; 22 distinct layer names; kernel shapes: 3x3x3 (298 layers), 1x3x3 (32 layers). Near-dead kernels skipped: 0 of 195753600.

### A0 sanity: random-init energy per atom

- kernel 1x3x3: 184320 kernels, atom energy ratio min 0.996 max 1.003 (expected 1.000)
- kernel 1x3x3 (first conv only): 576 kernels, atom energy ratio min 0.917 max 1.097 (expected 1.000)
- kernel 3x3x3: 48752640 kernels, atom energy ratio min 0.999 max 1.000 (expected 1.000)
- kernel 3x3x3 (first conv only): 864 kernels, atom energy ratio min 0.818 max 1.170 (expected 1.000)

### Per stage, across settings (unit = training setting; pooled quantity = HOF / HOF_random of the same layer)

delta = fold-mean ratio real - flat per setting. Wilcoxon one-sided (real > flat, pre-registered direction); sign test one-sided. Separated = 3 real folds > 3 flat folds on raw HOF within the setting (chance 0.75 per stage of 15).

| stage | ratio noise / flat / real (median over settings) | real > flat (fold mean) | delta median | Wilcoxon p | sign p | separated real>flat / flat>real | non-BraTS: real > flat, p | 23-cell weighted real > flat | real > noise | flat > noise | 1-DC real > flat | median-HOF real > flat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | 0.734 / 0.614 / 0.605 | 11/15 | +0.0288 | 0.02 | 0.06 | 2 / 0 | 9/13, 0.03 | 18/23 | 2/15 (p 0.007) | 0/15 (p 6e-05) | 14/15 (p 6e-05) | 10/15 (p 0.02) |
| c2 | 0.432 / 0.436 / 0.477 | 12/15 | +0.0356 | 0.001 | 0.02 | 6 / 0 | 10/13, 0.004 | 19/23 | 12/15 (p 0.01) | 7/15 (p 0.3) | 15/15 (p 3e-05) | 12/15 (p 0.0004) |
| E1 | 0.503 / 0.541 / 0.525 | 14/15 | +0.0273 | 0.0002 | 0.0005 | 6 / 0 | 12/13, 0.0009 | 21/23 | 14/15 (p 0.0001) | 13/15 (p 0.0003) | 15/15 (p 3e-05) | 14/15 (p 0.0002) |
| E2 | 0.520 / 0.518 / 0.563 | 12/15 | +0.0406 | 0.0008 | 0.02 | 8 / 0 | 10/13, 0.003 | 19/23 | 11/15 (p 0.003) | 8/15 (p 1) | 14/15 (p 0.0003) | 12/15 (p 0.0006) |
| E3 | 0.555 / 0.478 / 0.520 | 13/15 | +0.0200 | 0.003 | 0.004 | 6 / 1 | 11/13, 0.007 | 21/23 | 7/15 (p 0.8) | 6/15 (p 0.06) | 13/15 (p 0.003) | 13/15 (p 0.003) |
| E4 | 0.515 / 0.448 / 0.478 | 10/15 | +0.0189 | 0.02 | 0.2 | 4 / 1 | 9/13, 0.02 | 17/23 | 9/15 (p 1) | 4/15 (p 0.05) | 9/15 (p 0.2) | 10/15 (p 0.02) |
| E5 | 0.470 / 0.445 / 0.497 | 12/15 | +0.0196 | 0.006 | 0.02 | 6 / 2 | 12/13, 0.002 | 18/23 | 12/15 (p 0.1) | 9/15 (p 0.7) | 10/15 (p 0.06) | 12/15 (p 0.006) |
| D4 | 0.341 / 0.322 / 0.350 | 12/15 | +0.0147 | 0.003 | 0.02 | 4 / 0 | 12/13, 0.0009 | 18/23 | 9/15 (p 0.3) | 6/15 (p 0.2) | 8/15 (p 0.2) | 12/15 (p 0.003) |
| D3 | 0.495 / 0.475 / 0.506 | 11/15 | +0.0104 | 0.002 | 0.06 | 6 / 0 | 9/13, 0.007 | 18/23 | 12/15 (p 0.005) | 9/15 (p 1) | 11/15 (p 0.002) | 12/15 (p 0.001) |
| D2 | 0.623 / 0.625 / 0.620 | 10/15 | +0.0234 | 0.01 | 0.2 | 5 / 1 | 8/13, 0.02 | 17/23 | 12/15 (p 0.003) | 10/15 (p 0.03) | 12/15 (p 0.001) | 11/15 (p 0.01) |
| D1 | 0.606 / 0.625 / 0.618 | 13/15 | +0.0222 | 0.004 | 0.004 | 4 / 0 | 11/13, 0.01 | 20/23 | 14/15 (p 0.001) | 10/15 (p 0.05) | 14/15 (p 6e-05) | 13/15 (p 0.002) |
| D0 | 0.519 / 0.542 / 0.563 | 11/15 | +0.0114 | 0.08 | 0.06 | 1 / 0 | 9/13, 0.2 | 19/23 | 11/15 (p 0.02) | 10/15 (p 0.07) | 10/15 (p 0.009) | 10/15 (p 0.07) |

### First conv (c1) per setting: raw HOF per fold (random-init mean in brackets)

| setting | kernel | random | noise f0/f1/f2 | flat f0/f1/f2 | real f0/f1/f2 | real - flat (fold mean) | separated |
|---|---|---|---|---|---|---|---|
| glioma_t2w | 3x3x3 | 0.618 | 0.241/0.209/0.321 | 0.211/0.150/0.187 | 0.191/0.200/0.210 | +0.0178 | overlap |
| glioma_t1n | 3x3x3 | 0.618 | 0.304/0.275/0.237 | 0.126/0.129/0.209 | 0.155/0.176/0.209 | +0.0253 | overlap |
| ms_flair | 3x3x3 | 0.622 | 0.531/0.518/0.533 | 0.461/0.442/0.474 | 0.432/0.451/0.471 | -0.0074 | overlap |
| ms_t1w | 3x3x3 | 0.622 | 0.388/0.446/0.470 | 0.522/0.346/0.409 | 0.463/0.428/0.446 | +0.0201 | overlap |
| brain_t1w | 3x3x3 | 0.604 | 0.380/0.399/0.449 | 0.400/0.416/0.352 | 0.430/0.560/0.564 | +0.1286 | real>flat |
| brain_t2w | 3x3x3 | 0.604 | 0.448/0.497/0.432 | 0.371/0.431/0.369 | 0.387/0.449/0.417 | +0.0276 | overlap |
| abd_t1in | 1x3x3 | 0.340 | 0.290/0.267/0.339 | 0.210/0.179/0.239 | 0.190/0.197/0.262 | +0.0073 | overlap |
| abd_t2spir | 1x3x3 | 0.361 | 0.333/0.278/0.326 | 0.253/0.219/0.255 | 0.312/0.239/0.288 | +0.0370 | overlap |
| mand_cbct | 3x3x3 | 0.633 | 0.567/0.460/0.395 | 0.320/0.268/0.320 | 0.415/0.349/0.349 | +0.0683 | real>flat |
| breast_t1wce | 3x3x3 | 0.622 | 0.301/0.351/0.407 | 0.310/0.303/0.375 | 0.312/0.294/0.413 | +0.0103 | overlap |
| breast_t2w | 1x3x3 | 0.346 | 0.297/0.173/0.292 | 0.172/0.151/0.145 | 0.181/0.138/0.100 | -0.0167 | overlap |
| pelvis_mri | 1x3x3 | 0.339 | 0.294/0.259/0.327 | 0.261/0.208/0.268 | 0.265/0.203/0.285 | +0.0057 | overlap |
| pelvis_ct | 3x3x3 | 0.615 | 0.316/0.291/0.319 | 0.316/0.275/0.293 | 0.314/0.258/0.282 | -0.0098 | overlap |
| pancreas_t1wce | 1x3x3 | 0.323 | 0.205/0.169/0.207 | 0.182/0.160/0.147 | 0.222/0.141/0.203 | +0.0259 | overlap |
| pancreas_t2w | 1x3x3 | 0.323 | 0.252/0.242/0.281 | 0.228/0.181/0.217 | 0.221/0.221/0.144 | -0.0132 | overlap |

### First-conv order profile (energy fraction per DCT order, mean over kernels, folds and the 3x3x3-first-conv settings)

| arm | order 0 | order 1 | order 2 | order 3 | order 4 | order 5 | order 6 | HOF |
|---|---|---|---|---|---|---|---|---|
| noise | 0.101 | 0.317 | 0.193 | 0.153 | 0.135 | 0.064 | 0.037 | 0.388 |
| flat | 0.105 | 0.311 | 0.258 | 0.168 | 0.101 | 0.037 | 0.019 | 0.325 |
| real | 0.079 | 0.315 | 0.250 | 0.172 | 0.113 | 0.046 | 0.026 | 0.356 |
| rand | 0.043 | 0.111 | 0.228 | 0.254 | 0.214 | 0.110 | 0.039 | 0.617 |
| iid expectation | 0.037 | 0.111 | 0.222 | 0.259 | 0.222 | 0.111 | 0.037 | 0.630 |

## Part B: channel-selective first-stage kernel smoothing (BraTS)

Cells: glioma_t2w_t1n (n = 70), glioma_t2w_t2f (n = 70), glioma_t1n_t2f (n = 70), glioma_t1n_t2w (n = 70). Arms noise / flat / real, folds 0, 1, 2 (fold-mean). Selection from each model's own weights (conv 1: kernel HOF; conv 2: mean HOF of the output channel's input kernels); SMOOTH = projection onto DCT orders 0-1, norm kept. E = J(natural) - J(modified), J = edema log-odds, mean over GT edema; D_ch = E_sel - mean of 20 random-channel draws; D_pert = E_sel - mean of 20 equal-angle random perturbations of the same kernels. Headline c12, k = 8. Wilcoxon two-sided over cases.

### Plumbing gates

(a) manual batched stage-0 forward with unmodified weights vs the network's own forward: max |logit diff| = 0.00e+00 (gate < 1e-3: pass); restore check after all variants of a fold: max |diff| = 0.00e+00.

(b) all 32 conv-1 kernels smoothed (orders 0-1), median over cases:

| cell | arm | E target | Dice natural | Dice after |
|---|---|---|---|---|
| glioma_t2w_t1n | noise | -0.206 | 0.605 | 0.636 |
| glioma_t2w_t1n | flat | -1.160 | 0.433 | 0.569 |
| glioma_t2w_t1n | real | -1.195 | 0.456 | 0.599 |
| glioma_t2w_t2f | noise | +0.821 | 0.741 | 0.721 |
| glioma_t2w_t2f | flat | +0.627 | 0.734 | 0.717 |
| glioma_t2w_t2f | real | +0.168 | 0.769 | 0.761 |
| glioma_t1n_t2f | noise | -0.493 | 0.259 | 0.313 |
| glioma_t1n_t2f | flat | +0.001 | 0.424 | 0.428 |
| glioma_t1n_t2f | real | +0.387 | 0.632 | 0.599 |
| glioma_t1n_t2w | noise | -0.553 | 0.583 | 0.663 |
| glioma_t1n_t2w | flat | +0.280 | 0.694 | 0.689 |
| glioma_t1n_t2w | real | +0.432 | 0.745 | 0.729 |

### Input-blur equivalence check (first 5 sorted cases per cell; corr of the edema log-odds map over body voxels, per fold; median [min, max] over cases x folds)

| comparison | sigma | noise | flat | real | max interior |dJ| (median) |
|---|---|---|---|---|---|
| full_support_kxs_vs_blurred_input | 0.75 | 0.9999 [0.9995, 1.0000] | 0.9999 [0.9995, 1.0000] | 0.9999 [0.9995, 1.0000] | 5.32e-01 |
| truncated_kxs_3x3x3_vs_blurred_input | 0.75 | 0.9766 [0.9115, 0.9966] | 0.9734 [0.9165, 0.9934] | 0.9760 [0.9116, 0.9935] | 9.22e+00 |
| dct_order01_all32_vs_blurred_input | 0.5 | 0.9122 [0.8091, 0.9784] | 0.9182 [0.7511, 0.9809] | 0.8936 [0.8050, 0.9729] | 1.49e+01 |
| dct_order01_all32_vs_blurred_input | 0.75 | 0.9306 [0.8336, 0.9838] | 0.9330 [0.8467, 0.9853] | 0.9093 [0.8501, 0.9782] | 1.41e+01 |
| dct_order01_all32_vs_blurred_input | 1.0 | 0.9420 [0.8574, 0.9859] | 0.9377 [0.8961, 0.9870] | 0.9260 [0.8787, 0.9824] | 1.30e+01 |
| dct_order01_all32_vs_blurred_input | 1.5 | 0.8999 [0.8022, 0.9727] | 0.8774 [0.7826, 0.9656] | 0.8904 [0.7928, 0.9693] | 1.54e+01 |
| dct_order01_all32_vs_natural | 0.0 | 0.9015 [0.8037, 0.9763] | 0.9052 [0.7222, 0.9790] | 0.8888 [0.7947, 0.9713] | 1.53e+01 |

### Selection (mean over folds): HOF of selected vs all output channels; kernels zeroed by the smoothing guard

| cell | arm | k | conv1 HOF sel / all | conv2 mean HOF sel / all | zeroed c1 / c2 |
|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | 4 | 0.915 / 0.257 | 0.291 / 0.144 | 0 / 0 |
| glioma_t2w_t1n | noise | 8 | 0.739 / 0.257 | 0.240 / 0.144 | 0 / 0 |
| glioma_t2w_t1n | noise | 16 | 0.471 / 0.257 | 0.191 / 0.144 | 0 / 0 |
| glioma_t2w_t1n | flat | 4 | 0.802 / 0.183 | 0.222 / 0.113 | 0 / 0 |
| glioma_t2w_t1n | flat | 8 | 0.517 / 0.183 | 0.185 / 0.113 | 0 / 0 |
| glioma_t2w_t1n | flat | 16 | 0.317 / 0.183 | 0.152 / 0.113 | 0 / 0 |
| glioma_t2w_t1n | real | 4 | 0.792 / 0.200 | 0.296 / 0.146 | 0 / 0 |
| glioma_t2w_t1n | real | 8 | 0.543 / 0.200 | 0.252 / 0.146 | 0 / 0 |
| glioma_t2w_t1n | real | 16 | 0.347 / 0.200 | 0.204 / 0.146 | 0 / 0 |
| glioma_t2w_t2f | noise | 4 | 0.915 / 0.257 | 0.291 / 0.144 | 0 / 0 |
| glioma_t2w_t2f | noise | 8 | 0.739 / 0.257 | 0.240 / 0.144 | 0 / 0 |
| glioma_t2w_t2f | noise | 16 | 0.471 / 0.257 | 0.191 / 0.144 | 0 / 0 |
| glioma_t2w_t2f | flat | 4 | 0.802 / 0.183 | 0.222 / 0.113 | 0 / 0 |
| glioma_t2w_t2f | flat | 8 | 0.517 / 0.183 | 0.185 / 0.113 | 0 / 0 |
| glioma_t2w_t2f | flat | 16 | 0.317 / 0.183 | 0.152 / 0.113 | 0 / 0 |
| glioma_t2w_t2f | real | 4 | 0.792 / 0.200 | 0.296 / 0.146 | 0 / 0 |
| glioma_t2w_t2f | real | 8 | 0.543 / 0.200 | 0.252 / 0.146 | 0 / 0 |
| glioma_t2w_t2f | real | 16 | 0.347 / 0.200 | 0.204 / 0.146 | 0 / 0 |
| glioma_t1n_t2f | noise | 4 | 0.951 / 0.272 | 0.310 / 0.141 | 0 / 0 |
| glioma_t1n_t2f | noise | 8 | 0.786 / 0.272 | 0.255 / 0.141 | 0 / 0 |
| glioma_t1n_t2f | noise | 16 | 0.501 / 0.272 | 0.199 / 0.141 | 0 / 0 |
| glioma_t1n_t2f | flat | 4 | 0.561 / 0.155 | 0.188 / 0.093 | 0 / 0 |
| glioma_t1n_t2f | flat | 8 | 0.383 / 0.155 | 0.160 / 0.093 | 0 / 0 |
| glioma_t1n_t2f | flat | 16 | 0.253 / 0.155 | 0.131 / 0.093 | 0 / 0 |
| glioma_t1n_t2f | real | 4 | 0.819 / 0.180 | 0.264 / 0.116 | 0 / 0 |
| glioma_t1n_t2f | real | 8 | 0.526 / 0.180 | 0.216 / 0.116 | 0 / 0 |
| glioma_t1n_t2f | real | 16 | 0.317 / 0.180 | 0.170 / 0.116 | 0 / 0 |
| glioma_t1n_t2w | noise | 4 | 0.951 / 0.272 | 0.310 / 0.141 | 0 / 0 |
| glioma_t1n_t2w | noise | 8 | 0.786 / 0.272 | 0.255 / 0.141 | 0 / 0 |
| glioma_t1n_t2w | noise | 16 | 0.501 / 0.272 | 0.199 / 0.141 | 0 / 0 |
| glioma_t1n_t2w | flat | 4 | 0.561 / 0.155 | 0.188 / 0.093 | 0 / 0 |
| glioma_t1n_t2w | flat | 8 | 0.383 / 0.155 | 0.160 / 0.093 | 0 / 0 |
| glioma_t1n_t2w | flat | 16 | 0.253 / 0.155 | 0.131 / 0.093 | 0 / 0 |
| glioma_t1n_t2w | real | 4 | 0.819 / 0.180 | 0.264 / 0.116 | 0 / 0 |
| glioma_t1n_t2w | real | 8 | 0.526 / 0.180 | 0.216 / 0.116 | 0 / 0 |
| glioma_t1n_t2w | real | 16 | 0.317 / 0.180 | 0.170 / 0.116 | 0 / 0 |

### Headline (c12, k = 8): median [IQR] over cases

| cell | arm | E sel | E random-ch (mean) | D_ch | E pert (mean) | D_pert | Dice nat | after sel | after random-ch | after pert | E ring sel | E bg sel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +0.015 [-0.260, +0.425] | -0.035 [-0.085, +0.047] | +0.095 [-0.223, +0.482] | -0.189 [-0.406, +0.098] | +0.270 [+0.084, +0.476] | 0.605 | 0.625 | 0.619 | 0.627 | -0.196 | -0.196 |
| glioma_t2w_t1n | flat | -1.727 [-2.860, -1.092] | -0.627 [-1.032, -0.409] | -1.139 [-1.768, -0.503] | -1.618 [-2.489, -0.997] | -0.151 [-0.597, +0.111] | 0.433 | 0.595 | 0.498 | 0.581 | -0.903 | -0.117 |
| glioma_t2w_t1n | real | -0.935 [-1.861, -0.498] | -0.559 [-1.033, -0.348] | -0.465 [-1.004, -0.002] | -0.338 [-1.003, +0.661] | -1.131 [-1.574, -0.457] | 0.456 | 0.591 | 0.499 | 0.505 | -0.701 | +0.241 |
| glioma_t2w_t2f | noise | +0.808 [+0.569, +1.206] | +0.282 [+0.185, +0.391] | +0.558 [+0.373, +0.916] | +0.843 [+0.520, +1.212] | +0.039 [-0.178, +0.496] | 0.741 | 0.729 | 0.727 | 0.723 | -0.085 | -0.427 |
| glioma_t2w_t2f | flat | +1.421 [+1.008, +1.894] | +0.264 [+0.146, +0.381] | +1.197 [+0.863, +1.541] | +1.589 [+1.098, +2.089] | -0.218 [-0.647, +0.221] | 0.734 | 0.730 | 0.734 | 0.674 | +0.335 | -0.150 |
| glioma_t2w_t2f | real | +0.787 [+0.248, +1.293] | +0.080 [-0.147, +0.161] | +0.702 [+0.308, +1.091] | +2.378 [+1.532, +3.403] | -1.569 [-2.150, -1.112] | 0.769 | 0.755 | 0.761 | 0.682 | -0.276 | +0.049 |
| glioma_t1n_t2f | noise | -1.902 [-2.735, -1.108] | -0.571 [-0.898, -0.348] | -1.367 [-1.917, -0.762] | -1.087 [-1.874, -0.372] | -0.968 [-1.315, -0.588] | 0.259 | 0.473 | 0.331 | 0.366 | -0.297 | +0.281 |
| glioma_t1n_t2f | flat | +0.133 [-0.508, +0.865] | -0.620 [-0.894, -0.279] | +0.768 [+0.105, +1.269] | +0.707 [-0.047, +1.332] | -0.572 [-1.115, -0.018] | 0.424 | 0.425 | 0.500 | 0.382 | -0.360 | -0.676 |
| glioma_t1n_t2f | real | +0.917 [+0.335, +1.654] | +0.172 [-0.181, +0.387] | +0.794 [+0.169, +1.336] | +3.815 [+2.849, +5.537] | -2.901 [-4.531, -1.474] | 0.632 | 0.567 | 0.632 | 0.349 | -0.880 | -0.576 |
| glioma_t1n_t2w | noise | -0.684 [-1.401, -0.324] | -0.313 [-0.603, -0.161] | -0.395 [-0.904, -0.172] | -0.227 [-0.628, +0.050] | -0.566 [-0.949, -0.182] | 0.583 | 0.665 | 0.637 | 0.634 | -0.322 | +0.295 |
| glioma_t1n_t2w | flat | +0.240 [-0.109, +0.628] | +0.105 [-0.010, +0.208] | +0.125 [-0.134, +0.524] | +0.890 [+0.459, +1.436] | -0.674 [-0.908, -0.344] | 0.694 | 0.689 | 0.697 | 0.662 | -0.341 | -0.388 |
| glioma_t1n_t2w | real | +0.403 [+0.010, +0.761] | +0.045 [-0.040, +0.104] | +0.329 [+0.026, +0.677] | +1.942 [+1.320, +3.173] | -1.492 [-2.837, -0.930] | 0.745 | 0.715 | 0.744 | 0.637 | -0.430 | -0.197 |

### D_ch (median over cases) per conv set and k; Dice after selective smoothing in brackets

| cell | arm | c1 k=4 | c1 k=8 | c1 k=16 | c2 k=4 | c2 k=8 | c2 k=16 | c12 k=4 | c12 k=8 | c12 k=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | -0.164 (0.63) | -0.145 (0.64) | -0.070 (0.63) | +0.214 (0.61) | +0.141 (0.61) | +0.057 (0.61) | +0.117 (0.62) | +0.095 (0.62) | +0.130 (0.63) |
| glioma_t2w_t1n | flat | -0.733 (0.52) | -0.823 (0.54) | -0.543 (0.56) | -0.336 (0.48) | -0.412 (0.52) | -0.274 (0.54) | -1.118 (0.60) | -1.139 (0.59) | -0.611 (0.62) |
| glioma_t2w_t1n | real | -0.817 (0.58) | -0.710 (0.59) | -0.437 (0.59) | -0.525 (0.54) | -0.385 (0.55) | -0.331 (0.59) | -0.495 (0.56) | -0.465 (0.59) | -0.213 (0.60) |
| glioma_t2w_t2f | noise | +0.670 (0.72) | +0.563 (0.72) | +0.396 (0.72) | +0.247 (0.73) | +0.211 (0.73) | +0.164 (0.73) | +0.793 (0.72) | +0.558 (0.73) | +0.268 (0.73) |
| glioma_t2w_t2f | flat | +0.448 (0.72) | +0.387 (0.72) | +0.214 (0.72) | +0.105 (0.75) | +0.202 (0.73) | +0.164 (0.73) | +1.678 (0.68) | +1.197 (0.73) | +0.751 (0.72) |
| glioma_t2w_t2f | real | +0.074 (0.77) | +0.085 (0.77) | +0.112 (0.77) | +0.035 (0.77) | +0.122 (0.77) | +0.089 (0.76) | +0.513 (0.77) | +0.702 (0.76) | +0.552 (0.74) |
| glioma_t1n_t2f | noise | -0.268 (0.29) | -0.282 (0.30) | -0.215 (0.29) | -1.143 (0.41) | -1.442 (0.47) | -1.079 (0.47) | -1.270 (0.40) | -1.367 (0.47) | -1.074 (0.51) |
| glioma_t1n_t2f | flat | +0.035 (0.40) | -0.005 (0.41) | -0.059 (0.41) | +0.114 (0.47) | -0.238 (0.52) | +0.040 (0.52) | +0.229 (0.42) | +0.768 (0.42) | +0.935 (0.39) |
| glioma_t1n_t2f | real | +0.158 (0.62) | +0.118 (0.59) | -0.095 (0.60) | +0.166 (0.62) | +0.326 (0.61) | +0.357 (0.61) | +0.622 (0.59) | +0.794 (0.57) | +1.107 (0.45) |
| glioma_t1n_t2w | noise | -0.172 (0.63) | -0.327 (0.65) | -0.296 (0.65) | -0.547 (0.65) | -0.819 (0.67) | -0.547 (0.67) | -0.324 (0.65) | -0.395 (0.67) | -0.265 (0.68) |
| glioma_t1n_t2w | flat | +0.350 (0.69) | +0.311 (0.69) | +0.247 (0.69) | -0.003 (0.69) | -0.005 (0.70) | +0.056 (0.70) | +0.125 (0.69) | +0.125 (0.69) | +0.332 (0.68) |
| glioma_t1n_t2w | real | +0.248 (0.73) | +0.361 (0.73) | +0.221 (0.73) | +0.009 (0.74) | +0.064 (0.74) | +0.144 (0.72) | +0.302 (0.72) | +0.329 (0.71) | +0.382 (0.72) |

### Arm contrasts on D_ch (paired over cases) and per-arm controls

| cell | set | k | real - flat median | p | real > flat % | real - noise | p | flat - noise | p |
|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | c1 | 8 | +0.062 | 0.9 | 54 | -0.506 | 1e-07 | -0.619 | 5e-08 |
| glioma_t2w_t1n | c2 | 8 | +0.073 | 0.5 | 57 | -0.577 | 4e-11 | -0.504 | 1e-11 |
| glioma_t2w_t1n | c12 | 4 | +0.629 | 3e-07 | 77 | -0.572 | 6e-09 | -1.352 | 4e-11 |
| glioma_t2w_t1n | c12 | 8 | +0.699 | 9e-07 | 77 | -0.503 | 3e-06 | -1.308 | 3e-11 |
| glioma_t2w_t1n | c12 | 16 | +0.458 | 3e-06 | 74 | -0.344 | 0.001 | -0.912 | 2e-11 |
| glioma_t2w_t2f | c1 | 8 | -0.317 | 2e-08 | 16 | -0.471 | 3e-09 | -0.204 | 0.0008 |
| glioma_t2w_t2f | c2 | 8 | -0.127 | 0.002 | 29 | -0.082 | 0.1 | +0.050 | 0.3 |
| glioma_t2w_t2f | c12 | 4 | -1.213 | 2e-11 | 9 | -0.327 | 8e-05 | +0.939 | 1e-07 |
| glioma_t2w_t2f | c12 | 8 | -0.572 | 0.0009 | 24 | +0.151 | 0.2 | +0.602 | 2e-05 |
| glioma_t2w_t2f | c12 | 16 | -0.225 | 0.1 | 34 | +0.386 | 3e-05 | +0.517 | 4e-06 |
| glioma_t1n_t2f | c1 | 8 | +0.044 | 0.4 | 56 | +0.457 | 0.0002 | +0.194 | 0.002 |
| glioma_t1n_t2f | c2 | 8 | +0.558 | 3e-12 | 93 | +1.821 | 4e-13 | +1.240 | 4e-13 |
| glioma_t1n_t2f | c12 | 4 | +0.330 | 5e-05 | 73 | +2.029 | 8e-13 | +1.522 | 1e-11 |
| glioma_t1n_t2f | c12 | 8 | +0.062 | 0.3 | 54 | +2.336 | 1e-12 | +2.054 | 3e-12 |
| glioma_t1n_t2f | c12 | 16 | +0.316 | 0.002 | 67 | +2.471 | 4e-13 | +1.878 | 2e-12 |
| glioma_t1n_t2w | c1 | 8 | +0.055 | 0.5 | 56 | +0.764 | 4e-09 | +0.707 | 5e-11 |
| glioma_t1n_t2w | c2 | 8 | +0.081 | 0.08 | 59 | +0.854 | 3e-11 | +0.819 | 9e-13 |
| glioma_t1n_t2w | c12 | 4 | +0.088 | 0.04 | 63 | +0.658 | 2e-10 | +0.529 | 8e-10 |
| glioma_t1n_t2w | c12 | 8 | +0.236 | 0.007 | 69 | +0.931 | 7e-10 | +0.676 | 3e-09 |
| glioma_t1n_t2w | c12 | 16 | +0.048 | 0.4 | 56 | +0.644 | 3e-10 | +0.467 | 5e-10 |

(rows: every set at k = 8 and every k for c12; everything in kspec_B_controls.csv)

## Verdicts (pre-registered rules)

- PA1 (c1: real > flat by fold mean in >= 2/3 of settings AND Wilcoxon p < 0.01): NOT SUPPORTED (kill condition): 11/15, p 0.02, separated 2 (flat > real 0); non-BraTS 9/13 p 0.03.
- PA2 (c2 AND E1 by the same rule): SUPPORTED: c2 12/15 p 0.001 separated 6; E1 14/15 p 0.0002 separated 6.
- PA3 (exploratory, decoder): D4 real > flat 12/15 p 0.003; D3 real > flat 11/15 p 0.002; D2 real > flat 10/15 p 0.01; D1 real > flat 13/15 p 0.004; D0 real > flat 11/15 p 0.08.
- PA4 (noise arm, reported): c1 real > noise 2/15 (p 0.007), flat > noise 0/15 (p 6e-05); c2 real > noise 12/15 (p 0.01), flat > noise 7/15 (p 0.3); E1 real > noise 14/15 (p 0.0001), flat > noise 13/15 (p 0.0003).
- PB1 (c12, k = 8; D_ch real > flat, p < 0.01, >= 3/4 cells): NOT SUPPORTED (kill condition): 2/4 (glioma_t2w_t1n r-f +0.699 p 9e-07; glioma_t2w_t2f r-f -0.572 p 0.0009; glioma_t1n_t2f r-f +0.062 p 0.3; glioma_t1n_t2w r-f +0.236 p 0.007). Real > noise (reported): 2/4 (glioma_t2w_t1n r-n -0.503 p 3e-06; glioma_t2w_t2f r-n +0.151 p 0.2; glioma_t1n_t2f r-n +2.336 p 1e-12; glioma_t1n_t2w r-n +0.931 p 7e-10).
  - same rule at c1, k = 4: real > flat 0/4
  - same rule at c1, k = 8: real > flat 0/4
  - same rule at c1, k = 16: real > flat 0/4
  - same rule at c2, k = 4: real > flat 0/4
  - same rule at c2, k = 8: real > flat 1/4
  - same rule at c2, k = 16: real > flat 2/4
  - same rule at c12, k = 4: real > flat 2/4
  - same rule at c12, k = 16: real > flat 2/4
- PB2 (real arm, E_sel > equal-angle perturbation, >= 3/4 cells): NOT SUPPORTED: 0/4 (glioma_t2w_t1n -1.131 p 2e-09; glioma_t2w_t2f -1.569 p 4e-13; glioma_t1n_t2f -2.901 p 5e-13; glioma_t1n_t2w -1.492 p 1e-12).
- PB3 (real arm, Dice drop selective > random-channel, >= 3/4 cells): SUPPORTED: 3/4 (glioma_t2w_t1n -0.029 (mean -0.0401) p 8e-05; glioma_t2w_t2f +0.011 (mean +0.0144) p 1e-06; glioma_t1n_t2f +0.063 (mean +0.0779) p 2e-09; glioma_t1n_t2w +0.016 (mean +0.0197) p 2e-05).
- PB4: by eye on kspec_examples_<cell>.png (FINDINGS 46).
