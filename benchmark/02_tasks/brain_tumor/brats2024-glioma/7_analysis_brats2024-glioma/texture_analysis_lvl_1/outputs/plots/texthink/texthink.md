# What the model thinks about texture (FINDINGS 42, pre-registered: PREREG.md): texture-evidence maps by activation patching

Cells: glioma_t2w_t1n (n = 70; x_scr n = 70), glioma_t2w_t2f (n = 70; x_scr n = 70), glioma_t1n_t2f (n = 70; x_scr n = 70), glioma_t1n_t2w (n = 70; x_scr n = 70). Arms noise / flat / real, folds 0-2. E_L = J(x) - J(x with the activation at L taken from the counterfactual run), J = per-voxel log-odds of the target class (edema), maps fold-mean, scalars read on the fold-mean maps; E_full = J(x) - J(x_cf). Three counterfactuals: x_eq (texture-energy map flattened, local patterns kept) -> E; x_le (fine structure phase-scrambled, local energy map kept) -> S; x_scr (added after the first full run on request: fine structure phase-scrambled, body-masked, global body energy rescaled, no local gain; same phase field as x_le; not iterated) -> E_scr. Positive = removing that property LOWERS the target belief, i.e. it raised it. Target = GT target voxels in the patch; ring = dilation 10 minus target, inside the body; background = body minus dilation 10. d3 / d4 equal or approach E_full by construction and are not read.
Counterfactual construction (x_eq, x_le): TT_ITER = 2 (image-level re-flattening: after the pre-registered construction, 2 outer passes recompute the image's own fine band (x' - G1 x') * body and re-apply the same gain rule to it, because the sigma-1 highpass is not idempotent and the pre-registered x_eq left part of the original energy map in the image's fine band; chosen on the smoke QC before any evidence value was read).

## QC (medians over cases; out = cases outside the smoke gates: edge ratio [0.8, 1.2], x_eq contrast [0.8, 1.25])

| cell | Emap target/body contrast orig | after x_eq | after x_le | after x_scr | edge ratio eq [min-max, out] | le [min-max, out] | scr [min-max, out] | x_eq contrast out | body fine energy eq | le | scr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | 0.505 | 0.986 | 0.510 | 0.888 | 1.029 [0.97-1.23, 1] | 0.936 [0.90-0.99, 0] | 0.937 [0.89-1.07, 0] | 0 | 1.070 | 0.980 | 0.704 |
| glioma_t2w_t2f | 0.794 | 0.985 | 0.808 | 1.042 | 1.020 [0.93-1.17, 0] | 0.921 [0.89-0.97, 0] | 0.917 [0.88-0.96, 0] | 0 | 1.067 | 0.993 | 0.706 |
| glioma_t1n_t2f | 0.794 | 0.985 | 0.808 | 1.042 | 1.020 [0.93-1.17, 0] | 0.921 [0.89-0.97, 0] | 0.917 [0.88-0.96, 0] | 0 | 1.067 | 0.993 | 0.706 |
| glioma_t1n_t2w | 0.414 | 0.979 | 0.420 | 0.912 | 1.071 [0.92-1.23, 3] | 0.931 [0.90-0.95, 0] | 0.939 [0.89-0.98, 0] | 0 | 1.056 | 0.998 | 0.705 |

## E (x_eq: energy map flattened, pattern kept): target mean, median over cases per location

| cell | arm | full | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +1.676 | +0.005 | +0.126 | +0.254 | +0.667 | +0.232 | +0.014 | +0.265 | +1.177 | +1.575 | +1.698 | +1.676 |
| glioma_t2w_t1n | flat | +2.279 | +0.021 | +0.230 | +0.249 | +1.157 | +0.679 | +0.032 | +0.759 | +1.980 | +2.186 | +2.320 | +2.279 |
| glioma_t2w_t1n | real | +3.452 | +0.041 | +0.298 | +0.300 | +1.711 | +0.574 | +0.008 | +0.615 | +2.814 | +3.134 | +3.415 | +3.452 |
| glioma_t2w_t2f | noise | +0.925 | +0.007 | +0.062 | +0.122 | +0.258 | +0.101 | +0.003 | +0.106 | +0.406 | +0.739 | +0.909 | +0.925 |
| glioma_t2w_t2f | flat | +1.728 | -0.015 | +0.026 | +0.185 | +0.637 | +0.365 | +0.015 | +0.421 | +1.238 | +1.629 | +1.701 | +1.728 |
| glioma_t2w_t2f | real | +2.614 | +0.037 | +0.110 | +0.231 | +0.815 | +0.452 | +0.006 | +0.455 | +2.039 | +2.461 | +2.581 | +2.614 |
| glioma_t1n_t2f | noise | -0.563 | +0.001 | -0.117 | -0.164 | -0.164 | -0.100 | -0.022 | -0.108 | -0.352 | -0.487 | -0.559 | -0.563 |
| glioma_t1n_t2f | flat | +0.374 | +0.019 | +0.029 | -0.037 | +0.001 | +0.128 | +0.017 | +0.164 | +0.255 | +0.281 | +0.359 | +0.374 |
| glioma_t1n_t2f | real | +0.859 | +0.022 | +0.044 | +0.113 | +0.159 | +0.156 | +0.021 | +0.181 | +0.441 | +0.695 | +0.819 | +0.859 |
| glioma_t1n_t2w | noise | +0.452 | -0.008 | +0.034 | +0.061 | +0.286 | -0.068 | -0.004 | -0.068 | +0.292 | +0.374 | +0.445 | +0.452 |
| glioma_t1n_t2w | flat | +1.054 | +0.028 | +0.119 | +0.164 | +0.409 | +0.159 | +0.020 | +0.196 | +0.648 | +0.866 | +0.998 | +1.054 |
| glioma_t1n_t2w | real | +1.985 | +0.030 | +0.124 | +0.326 | +0.603 | +0.238 | +0.018 | +0.326 | +1.198 | +1.777 | +1.962 | +1.985 |

## S (x_le: pattern scrambled, local energy map kept): target mean, median over cases per location

| cell | arm | full | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +0.235 | -0.006 | +0.050 | +0.064 | +0.045 | +0.008 | +0.003 | +0.016 | +0.123 | +0.202 | +0.237 | +0.235 |
| glioma_t2w_t1n | flat | -0.824 | +0.007 | +0.011 | -0.148 | -0.637 | -0.399 | -0.007 | -0.401 | -0.828 | -0.849 | -0.831 | -0.824 |
| glioma_t2w_t1n | real | -0.331 | +0.010 | -0.029 | +0.011 | -0.235 | -0.067 | +0.004 | -0.037 | -0.420 | -0.376 | -0.355 | -0.331 |
| glioma_t2w_t2f | noise | +0.084 | -0.013 | -0.021 | -0.009 | +0.020 | +0.041 | +0.004 | +0.040 | +0.077 | +0.122 | +0.090 | +0.084 |
| glioma_t2w_t2f | flat | -0.569 | +0.006 | -0.110 | -0.187 | -0.278 | -0.151 | -0.011 | -0.121 | -0.351 | -0.560 | -0.588 | -0.569 |
| glioma_t2w_t2f | real | -0.600 | +0.021 | -0.100 | -0.199 | -0.190 | -0.218 | -0.013 | -0.221 | -0.392 | -0.528 | -0.625 | -0.600 |
| glioma_t1n_t2f | noise | +0.219 | +0.003 | +0.024 | -0.041 | +0.030 | +0.153 | -0.000 | +0.148 | +0.251 | +0.198 | +0.224 | +0.219 |
| glioma_t1n_t2f | flat | +0.013 | +0.008 | -0.016 | -0.160 | -0.037 | +0.078 | -0.007 | +0.126 | +0.001 | +0.005 | +0.007 | +0.013 |
| glioma_t1n_t2f | real | +0.779 | +0.008 | -0.054 | -0.064 | +0.345 | +0.317 | +0.007 | +0.362 | +0.700 | +0.778 | +0.774 | +0.779 |
| glioma_t1n_t2w | noise | +0.468 | -0.002 | +0.047 | +0.025 | +0.132 | +0.181 | +0.006 | +0.154 | +0.374 | +0.434 | +0.461 | +0.468 |
| glioma_t1n_t2w | flat | +0.746 | -0.003 | -0.000 | +0.109 | +0.325 | +0.134 | +0.003 | +0.176 | +0.603 | +0.725 | +0.751 | +0.746 |
| glioma_t1n_t2w | real | +0.913 | +0.006 | +0.016 | +0.127 | +0.344 | +0.158 | +0.021 | +0.186 | +0.657 | +0.845 | +0.900 | +0.913 |

## E_scr (x_scr: pattern scrambled, global body energy only): target mean, median over cases per location

| cell | arm | full | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +2.065 | +0.003 | +0.163 | +0.360 | +0.734 | +0.439 | +0.043 | +0.517 | +1.421 | +1.926 | +2.097 | +2.065 |
| glioma_t2w_t1n | flat | +2.619 | +0.026 | +0.214 | +0.213 | +0.936 | +0.842 | +0.063 | +0.938 | +2.350 | +2.521 | +2.620 | +2.619 |
| glioma_t2w_t1n | real | +4.043 | +0.038 | +0.250 | +0.400 | +1.306 | +0.853 | +0.037 | +0.899 | +2.916 | +3.564 | +4.047 | +4.043 |
| glioma_t2w_t2f | noise | +0.363 | +0.013 | +0.005 | +0.052 | +0.169 | +0.119 | +0.008 | +0.142 | +0.319 | +0.367 | +0.368 | +0.363 |
| glioma_t2w_t2f | flat | +0.683 | +0.030 | -0.085 | +0.128 | +0.089 | +0.135 | +0.007 | +0.166 | +0.353 | +0.678 | +0.594 | +0.683 |
| glioma_t2w_t2f | real | +0.616 | +0.050 | +0.002 | +0.067 | +0.155 | -0.003 | +0.003 | +0.031 | +0.352 | +0.456 | +0.477 | +0.616 |
| glioma_t1n_t2f | noise | -0.902 | -0.015 | -0.127 | -0.223 | -0.271 | -0.244 | -0.056 | -0.292 | -0.577 | -0.801 | -0.882 | -0.902 |
| glioma_t1n_t2f | flat | -0.151 | +0.011 | -0.006 | -0.060 | -0.020 | +0.020 | +0.027 | +0.042 | +0.006 | -0.213 | -0.174 | -0.151 |
| glioma_t1n_t2f | real | +1.271 | +0.031 | -0.026 | +0.148 | +0.307 | +0.194 | +0.026 | +0.328 | +0.857 | +1.275 | +1.198 | +1.271 |
| glioma_t1n_t2w | noise | +0.389 | -0.008 | +0.020 | -0.017 | +0.145 | +0.023 | +0.001 | +0.034 | +0.201 | +0.340 | +0.403 | +0.389 |
| glioma_t1n_t2w | flat | +1.840 | +0.021 | +0.136 | +0.297 | +0.704 | +0.250 | +0.027 | +0.463 | +1.238 | +1.651 | +1.809 | +1.840 |
| glioma_t1n_t2w | real | +3.588 | +0.043 | +0.158 | +0.544 | +0.899 | +0.440 | +0.048 | +0.602 | +2.340 | +3.377 | +3.532 | +3.588 |

## E (x_eq: energy map flattened, pattern kept): per cell and arm, median [IQR] over cases

| cell | arm | target full | target s2 | target s3 | ring full | background full | frac > 1 nat (target) | share (s2+s3)/full | ratio to S_target full (x_le) | ratio s3 |
|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +1.676 [+0.785, +3.188] | +0.254 [+0.114, +0.400] | +0.667 [+0.386, +1.281] | +0.479 [+0.061, +0.960] | -0.136 [-0.274, -0.074] | 0.590 | 0.57 | 7.14 | 14.68 |
| glioma_t2w_t1n | flat | +2.279 [+1.518, +3.821] | +0.249 [+0.132, +0.376] | +1.157 [+0.674, +1.859] | +0.379 [-0.157, +0.845] | -0.236 [-0.376, -0.113] | 0.656 | 0.63 | -2.77 | -1.81 |
| glioma_t2w_t1n | real | +3.452 [+1.728, +4.813] | +0.300 [+0.109, +0.456] | +1.711 [+0.896, +2.159] | +0.421 [-0.361, +1.094] | -0.511 [-0.715, -0.353] | 0.729 | 0.58 | -10.44 | -7.27 |
| glioma_t2w_t2f | noise | +0.925 [+0.084, +1.414] | +0.122 [-0.041, +0.280] | +0.258 [+0.031, +0.526] | +0.404 [+0.044, +0.860] | -0.050 [-0.155, +0.088] | 0.437 | 0.54 | 11.03 | 12.94 |
| glioma_t2w_t2f | flat | +1.728 [+0.718, +3.148] | +0.185 [+0.004, +0.382] | +0.637 [+0.248, +1.094] | +0.503 [+0.096, +1.283] | -0.007 [-0.138, +0.148] | 0.596 | 0.44 | -3.04 | -2.29 |
| glioma_t2w_t2f | real | +2.614 [+1.242, +4.471] | +0.231 [+0.013, +0.373] | +0.815 [+0.362, +1.224] | +0.768 [+0.060, +2.041] | -0.156 [-0.408, -0.002] | 0.681 | 0.40 | -4.35 | -4.28 |
| glioma_t1n_t2f | noise | -0.563 [-1.061, -0.102] | -0.164 [-0.270, -0.044] | -0.164 [-0.396, +0.066] | +0.364 [+0.125, +0.825] | +0.431 [+0.313, +0.619] | 0.120 | 0.59 | -2.58 | -5.50 |
| glioma_t1n_t2f | flat | +0.374 [-0.320, +1.055] | -0.037 [-0.132, +0.089] | +0.001 [-0.197, +0.354] | +0.338 [-0.090, +0.817] | +0.226 [+0.061, +0.371] | 0.286 | 0.47 | 28.82 | -0.04 |
| glioma_t1n_t2f | real | +0.859 [+0.348, +1.326] | +0.113 [-0.015, +0.257] | +0.159 [-0.159, +0.437] | +0.323 [-0.144, +0.648] | -0.004 [-0.084, +0.092] | 0.415 | 0.47 | 1.10 | 0.46 |
| glioma_t1n_t2w | noise | +0.452 [-0.000, +0.830] | +0.061 [-0.047, +0.213] | +0.286 [+0.071, +0.561] | +0.470 [+0.179, +0.721] | +0.166 [+0.101, +0.313] | 0.343 | 0.67 | 0.97 | 2.16 |
| glioma_t1n_t2w | flat | +1.054 [+0.688, +1.973] | +0.164 [+0.082, +0.299] | +0.409 [+0.186, +0.709] | +0.330 [-0.046, +0.890] | -0.176 [-0.393, -0.065] | 0.466 | 0.47 | 1.41 | 1.26 |
| glioma_t1n_t2w | real | +1.985 [+1.172, +2.986] | +0.326 [+0.197, +0.505] | +0.603 [+0.358, +0.941] | +0.651 [+0.156, +1.292] | -0.139 [-0.262, -0.053] | 0.622 | 0.53 | 2.17 | 1.75 |

Ratio = ratio of the medians over cases (primary for P2; per-case ratios with |S| > 1e-3 in texthink_cells.csv). Share = median over cases of the per-case (X_s2 + X_s3) / X_full (|X_full| > 1e-3).

## S (x_le: pattern scrambled, local energy map kept): per cell and arm, median [IQR] over cases

| cell | arm | target full | target s2 | target s3 | ring full | background full | frac > 1 nat (target) | share (s2+s3)/full | ratio to S_target full (x_le) | ratio s3 |
|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +0.235 [-0.170, +0.533] | +0.064 [-0.039, +0.148] | +0.045 [-0.183, +0.202] | -0.254 [-0.478, -0.044] | -0.310 [-0.446, -0.207] | 0.265 | 0.63 | 1.00 | 1.00 |
| glioma_t2w_t1n | flat | -0.824 [-2.209, -0.235] | -0.148 [-0.283, +0.009] | -0.637 [-1.081, -0.269] | -0.540 [-1.092, -0.086] | +0.031 [-0.098, +0.137] | 0.197 | 0.61 | 1.00 | 1.00 |
| glioma_t2w_t1n | real | -0.331 [-1.575, +0.415] | +0.011 [-0.218, +0.198] | -0.235 [-0.967, +0.096] | -0.108 [-0.577, +0.359] | +0.223 [+0.131, +0.398] | 0.266 | 0.74 | 1.00 | 1.00 |
| glioma_t2w_t2f | noise | +0.084 [-0.124, +0.365] | -0.009 [-0.052, +0.055] | +0.020 [-0.076, +0.180] | -0.698 [-1.000, -0.476] | -0.541 [-0.826, -0.377] | 0.146 | 0.67 | 1.00 | 1.00 |
| glioma_t2w_t2f | flat | -0.569 [-0.927, -0.193] | -0.187 [-0.249, -0.079] | -0.278 [-0.755, -0.075] | -0.958 [-1.361, -0.657] | -0.533 [-0.747, -0.243] | 0.117 | 0.77 | 1.00 | 1.00 |
| glioma_t2w_t2f | real | -0.600 [-1.087, -0.082] | -0.199 [-0.284, -0.107] | -0.190 [-0.578, +0.068] | -1.301 [-1.649, -0.989] | -0.162 [-0.396, +0.035] | 0.122 | 0.70 | 1.00 | 1.00 |
| glioma_t1n_t2f | noise | +0.219 [-0.263, +0.679] | -0.041 [-0.124, +0.030] | +0.030 [-0.152, +0.297] | -0.580 [-1.045, -0.336] | -0.958 [-1.194, -0.738] | 0.262 | 0.32 | 1.00 | 1.00 |
| glioma_t1n_t2f | flat | +0.013 [-0.606, +0.759] | -0.160 [-0.295, -0.002] | -0.037 [-0.384, +0.196] | -0.778 [-1.293, -0.140] | -0.968 [-1.406, -0.638] | 0.274 | 0.39 | 1.00 | 1.00 |
| glioma_t1n_t2f | real | +0.779 [+0.149, +1.529] | -0.064 [-0.184, +0.087] | +0.345 [+0.032, +0.705] | -0.665 [-1.184, +0.073] | -0.428 [-0.667, -0.307] | 0.461 | 0.44 | 1.00 | 1.00 |
| glioma_t1n_t2w | noise | +0.468 [+0.161, +0.826] | +0.025 [-0.053, +0.141] | +0.132 [-0.029, +0.341] | -0.209 [-0.380, +0.052] | -0.474 [-0.623, -0.303] | 0.327 | 0.41 | 1.00 | 1.00 |
| glioma_t1n_t2w | flat | +0.746 [+0.412, +1.094] | +0.109 [+0.024, +0.220] | +0.325 [+0.123, +0.520] | -0.202 [-0.594, +0.133] | -0.748 [-0.971, -0.597] | 0.435 | 0.57 | 1.00 | 1.00 |
| glioma_t1n_t2w | real | +0.913 [+0.644, +1.264] | +0.127 [+0.033, +0.214] | +0.344 [+0.131, +0.580] | -0.021 [-0.393, +0.216] | -0.240 [-0.332, -0.126] | 0.458 | 0.55 | 1.00 | 1.00 |

Ratio = ratio of the medians over cases (primary for P2; per-case ratios with |S| > 1e-3 in texthink_cells.csv). Share = median over cases of the per-case (X_s2 + X_s3) / X_full (|X_full| > 1e-3).

## E_scr (x_scr: pattern scrambled, global body energy only): per cell and arm, median [IQR] over cases

| cell | arm | target full | target s2 | target s3 | ring full | background full | frac > 1 nat (target) | share (s2+s3)/full | ratio to S_target full (x_le) | ratio s3 |
|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | +2.065 [+0.952, +3.745] | +0.360 [+0.128, +0.566] | +0.734 [+0.323, +1.374] | +0.134 [-0.289, +0.832] | -0.592 [-0.789, -0.374] | 0.618 | 0.57 | 8.80 | 16.16 |
| glioma_t2w_t1n | flat | +2.619 [+1.117, +4.208] | +0.213 [+0.065, +0.429] | +0.936 [+0.192, +1.344] | +0.242 [-0.579, +0.977] | -0.425 [-0.685, -0.158] | 0.668 | 0.47 | -3.18 | -1.47 |
| glioma_t2w_t1n | real | +4.043 [+1.915, +5.758] | +0.400 [+0.169, +0.698] | +1.306 [+0.821, +2.426] | +0.602 [-0.517, +1.726] | -0.394 [-0.712, -0.212] | 0.741 | 0.55 | -12.23 | -5.55 |
| glioma_t2w_t2f | noise | +0.363 [+0.167, +1.029] | +0.052 [-0.112, +0.209] | +0.169 [+0.014, +0.420] | -0.361 [-0.816, +0.015] | -0.490 [-0.764, -0.346] | 0.367 | 0.59 | 4.33 | 8.50 |
| glioma_t2w_t2f | flat | +0.683 [-0.258, +1.795] | +0.128 [-0.080, +0.302] | +0.089 [-0.242, +0.791] | -0.748 [-1.293, -0.240] | -0.135 [-0.394, +0.064] | 0.444 | 0.56 | -1.20 | -0.32 |
| glioma_t2w_t2f | real | +0.616 [-0.201, +2.017] | +0.067 [-0.071, +0.244] | +0.155 [-0.198, +0.669] | -1.249 [-2.041, -0.502] | -0.184 [-0.399, +0.105] | 0.436 | 0.44 | -1.03 | -0.82 |
| glioma_t1n_t2f | noise | -0.902 [-1.383, -0.225] | -0.223 [-0.346, -0.069] | -0.271 [-0.572, +0.098] | -0.872 [-1.354, -0.597] | -1.029 [-1.493, -0.700] | 0.151 | 0.57 | -4.13 | -9.11 |
| glioma_t1n_t2f | flat | -0.151 [-0.872, +1.320] | -0.060 [-0.295, +0.241] | -0.020 [-0.558, +0.419] | -0.777 [-1.695, -0.099] | -1.064 [-1.402, -0.438] | 0.339 | 0.48 | -11.67 | 0.55 |
| glioma_t1n_t2f | real | +1.271 [+0.045, +2.474] | +0.148 [-0.118, +0.335] | +0.307 [-0.157, +0.799] | -0.955 [-1.703, -0.099] | -0.546 [-0.795, -0.306] | 0.556 | 0.50 | 1.63 | 0.89 |
| glioma_t1n_t2w | noise | +0.389 [-0.349, +1.089] | -0.017 [-0.195, +0.143] | +0.145 [-0.324, +0.437] | -0.424 [-0.748, -0.003] | -0.729 [-1.011, -0.471] | 0.384 | 0.67 | 0.83 | 1.10 |
| glioma_t1n_t2w | flat | +1.840 [+1.201, +3.364] | +0.297 [+0.178, +0.490] | +0.704 [+0.247, +1.156] | -0.367 [-1.054, +0.524] | -1.558 [-1.967, -1.110] | 0.636 | 0.53 | 2.47 | 2.17 |
| glioma_t1n_t2w | real | +3.588 [+2.251, +5.055] | +0.544 [+0.390, +0.738] | +0.899 [+0.459, +1.548] | +0.376 [-0.328, +1.870] | -0.490 [-0.874, -0.223] | 0.766 | 0.41 | 3.93 | 2.61 |

Ratio = ratio of the medians over cases (primary for P2; per-case ratios with |S| > 1e-3 in texthink_cells.csv). Share = median over cases of the per-case (X_s2 + X_s3) / X_full (|X_full| > 1e-3).

## Arm contrasts on the target mean (paired over cases, two-sided Wilcoxon)

| counterfactual | cell | loc | real - flat median | p | real > flat (% cases) | real - noise median | p | flat - noise median | p |
|---|---|---|---|---|---|---|---|---|---|
| eq | glioma_t2w_t1n | full | +0.827 | 3e-05 | 69 | +1.405 | 1e-05 | +0.665 | 0.005 |
| eq | glioma_t2w_t1n | s2 | +0.056 | 0.2 | 59 | +0.108 | 0.2 | +0.015 | 1 |
| eq | glioma_t2w_t1n | s3 | +0.279 | 0.008 | 61 | +0.669 | 1e-06 | +0.445 | 0.0002 |
| eq | glioma_t2w_t2f | full | +0.630 | 2e-08 | 86 | +1.781 | 6e-11 | +0.879 | 2e-09 |
| eq | glioma_t2w_t2f | s2 | +0.008 | 0.4 | 51 | +0.087 | 0.0001 | +0.093 | 4e-05 |
| eq | glioma_t2w_t2f | s3 | +0.133 | 0.003 | 66 | +0.401 | 2e-08 | +0.264 | 2e-06 |
| eq | glioma_t1n_t2f | full | +0.435 | 7e-05 | 70 | +1.295 | 2e-12 | +0.786 | 1e-08 |
| eq | glioma_t1n_t2f | s2 | +0.127 | 3e-06 | 73 | +0.250 | 1e-09 | +0.099 | 1e-05 |
| eq | glioma_t1n_t2f | s3 | +0.076 | 0.2 | 60 | +0.215 | 5e-05 | +0.161 | 0.0008 |
| eq | glioma_t1n_t2w | full | +0.591 | 6e-10 | 89 | +1.678 | 7e-13 | +0.732 | 2e-10 |
| eq | glioma_t1n_t2w | s2 | +0.177 | 2e-11 | 96 | +0.287 | 2e-12 | +0.112 | 3e-05 |
| eq | glioma_t1n_t2w | s3 | +0.178 | 0.0002 | 70 | +0.318 | 2e-07 | +0.077 | 0.01 |
| le | glioma_t2w_t1n | full | +0.562 | 9e-05 | 74 | -0.470 | 0.0004 | -0.915 | 3e-10 |
| le | glioma_t2w_t1n | s2 | +0.159 | 4e-05 | 70 | -0.005 | 0.2 | -0.192 | 5e-08 |
| le | glioma_t2w_t1n | s3 | +0.325 | 0.0007 | 70 | -0.324 | 0.0003 | -0.667 | 8e-10 |
| le | glioma_t2w_t2f | full | -0.018 | 0.5 | 47 | -0.721 | 2e-09 | -0.646 | 1e-09 |
| le | glioma_t2w_t2f | s2 | -0.032 | 0.1 | 40 | -0.204 | 5e-11 | -0.158 | 2e-08 |
| le | glioma_t2w_t2f | s3 | +0.065 | 0.5 | 54 | -0.227 | 2e-06 | -0.333 | 1e-07 |
| le | glioma_t1n_t2f | full | +0.892 | 3e-08 | 80 | +0.578 | 1e-05 | -0.122 | 0.2 |
| le | glioma_t1n_t2f | s2 | +0.116 | 4e-06 | 70 | +0.015 | 1 | -0.086 | 1e-05 |
| le | glioma_t1n_t2f | s3 | +0.451 | 1e-09 | 87 | +0.284 | 1e-05 | -0.084 | 0.04 |
| le | glioma_t1n_t2w | full | +0.151 | 0.4 | 60 | +0.468 | 3e-05 | +0.463 | 0.0001 |
| le | glioma_t1n_t2w | s2 | -0.007 | 0.9 | 47 | +0.112 | 0.0001 | +0.075 | 0.0004 |
| le | glioma_t1n_t2w | s3 | +0.016 | 0.5 | 57 | +0.230 | 2e-05 | +0.181 | 0.0006 |
| scr | glioma_t2w_t1n | full | +1.283 | 3e-07 | 77 | +1.691 | 3e-05 | +0.377 | 0.3 |
| scr | glioma_t2w_t1n | s2 | +0.170 | 0.0001 | 67 | +0.080 | 0.09 | -0.122 | 0.03 |
| scr | glioma_t2w_t1n | s3 | +0.559 | 2e-05 | 73 | +0.525 | 0.0001 | -0.088 | 0.9 |
| scr | glioma_t2w_t2f | full | +0.181 | 0.5 | 57 | +0.232 | 0.09 | +0.271 | 0.06 |
| scr | glioma_t2w_t2f | s2 | -0.038 | 0.08 | 39 | +0.030 | 0.2 | +0.072 | 0.005 |
| scr | glioma_t2w_t2f | s3 | +0.079 | 0.5 | 57 | +0.022 | 0.9 | -0.128 | 0.6 |
| scr | glioma_t1n_t2f | full | +1.188 | 3e-08 | 80 | +1.959 | 2e-11 | +0.944 | 2e-06 |
| scr | glioma_t1n_t2f | s2 | +0.114 | 8e-06 | 74 | +0.328 | 8e-10 | +0.142 | 4e-05 |
| scr | glioma_t1n_t2f | s3 | +0.239 | 2e-06 | 76 | +0.446 | 4e-06 | +0.185 | 0.09 |
| scr | glioma_t1n_t2w | full | +1.263 | 3e-08 | 79 | +3.193 | 1e-12 | +1.723 | 1e-12 |
| scr | glioma_t1n_t2w | s2 | +0.257 | 5e-11 | 93 | +0.603 | 2e-12 | +0.362 | 2e-11 |
| scr | glioma_t1n_t2w | s3 | +0.204 | 0.005 | 70 | +0.863 | 2e-10 | +0.661 | 2e-10 |

## Units (x_eq displacement, s2 / s3): fraction of channels with relative displacement > 0.2 (median over cases), median displacement, top-5 Spearman with the residualised fine-energy map (fold 0; mean of the 5 per case, then median; raw = not residualised)

| cell | arm | loc | n ch | frac > 0.2 | median disp | top-1 disp | rho top-5 (resid) | |rho| top-5 | rho top-1 | rho top-5 raw |
|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t2w_t1n | noise | s2 | 128 | 0.992 | 0.437 | 0.862 | -0.017 | 0.109 | +0.007 | -0.121 |
| glioma_t2w_t1n | flat | s2 | 128 | 1.000 | 0.465 | 0.808 | -0.041 | 0.141 | +0.073 | -0.115 |
| glioma_t2w_t1n | real | s2 | 128 | 1.000 | 0.491 | 0.793 | -0.074 | 0.107 | -0.080 | -0.059 |
| glioma_t2w_t1n | noise | s3 | 256 | 1.000 | 0.356 | 0.556 | -0.000 | 0.077 | -0.052 | -0.004 |
| glioma_t2w_t1n | flat | s3 | 256 | 1.000 | 0.442 | 0.610 | -0.041 | 0.075 | -0.071 | -0.056 |
| glioma_t2w_t1n | real | s3 | 256 | 1.000 | 0.533 | 0.756 | +0.029 | 0.096 | +0.022 | +0.008 |
| glioma_t2w_t2f | noise | s2 | 128 | 1.000 | 0.543 | 0.928 | -0.018 | 0.142 | +0.080 | -0.075 |
| glioma_t2w_t2f | flat | s2 | 128 | 1.000 | 0.528 | 0.899 | -0.039 | 0.135 | +0.032 | -0.062 |
| glioma_t2w_t2f | real | s2 | 128 | 1.000 | 0.569 | 0.877 | -0.127 | 0.144 | -0.142 | -0.172 |
| glioma_t2w_t2f | noise | s3 | 256 | 1.000 | 0.472 | 0.685 | -0.015 | 0.089 | -0.122 | -0.029 |
| glioma_t2w_t2f | flat | s3 | 256 | 1.000 | 0.508 | 0.695 | +0.025 | 0.072 | +0.020 | -0.027 |
| glioma_t2w_t2f | real | s3 | 256 | 1.000 | 0.577 | 0.780 | +0.009 | 0.085 | -0.023 | -0.046 |
| glioma_t1n_t2f | noise | s2 | 128 | 1.000 | 0.526 | 0.873 | -0.027 | 0.148 | -0.215 | -0.074 |
| glioma_t1n_t2f | flat | s2 | 128 | 1.000 | 0.459 | 0.830 | +0.048 | 0.222 | +0.261 | +0.007 |
| glioma_t1n_t2f | real | s2 | 128 | 1.000 | 0.495 | 0.704 | -0.034 | 0.151 | +0.148 | -0.151 |
| glioma_t1n_t2f | noise | s3 | 256 | 1.000 | 0.445 | 0.601 | +0.058 | 0.093 | +0.098 | +0.035 |
| glioma_t1n_t2f | flat | s3 | 256 | 1.000 | 0.425 | 0.562 | -0.033 | 0.086 | -0.119 | -0.076 |
| glioma_t1n_t2f | real | s3 | 256 | 1.000 | 0.480 | 0.667 | +0.007 | 0.069 | +0.020 | -0.023 |
| glioma_t1n_t2w | noise | s2 | 128 | 1.000 | 0.581 | 1.009 | +0.019 | 0.174 | -0.196 | +0.003 |
| glioma_t1n_t2w | flat | s2 | 128 | 1.000 | 0.575 | 1.410 | -0.126 | 0.165 | -0.114 | -0.245 |
| glioma_t1n_t2w | real | s2 | 128 | 1.000 | 0.590 | 1.552 | -0.003 | 0.073 | -0.056 | -0.031 |
| glioma_t1n_t2w | noise | s3 | 256 | 1.000 | 0.481 | 0.705 | +0.043 | 0.081 | +0.080 | +0.047 |
| glioma_t1n_t2w | flat | s3 | 256 | 1.000 | 0.508 | 0.646 | -0.032 | 0.095 | -0.073 | -0.047 |
| glioma_t1n_t2w | real | s3 | 256 | 1.000 | 0.570 | 0.741 | +0.022 | 0.078 | +0.001 | +0.006 |

## Verdicts (pre-registered P1-P4, read on x_eq as pre-registered; P1-P3 also read on x_le and x_scr on request. Operational rules fixed before reading the full-run numbers: P1 paired real - flat and real - noise median > 0 with two-sided Wilcoxon p < 0.01, kill if real <= flat in >= 2 cells; P2 ratio of medians; P3 median per-case share; P4 'similar' = |median paired difference of the fraction| < 0.1 at s2 and s3)

- P1 on eq (PRE-REGISTERED reading): SUPPORTED: real > flat (p < 0.01) in 4/4 cells, real > noise in 4/4. real - flat full: glioma_t2w_t1n +0.827 (p 3e-05), glioma_t2w_t2f +0.630 (p 2e-08), glioma_t1n_t2f +0.435 (p 7e-05), glioma_t1n_t2w +0.591 (p 6e-10).
- P1 on le: NOT SUPPORTED: real > flat (p < 0.01) in 2/4 cells, real > noise in 2/4 (need >= 3 each); real <= flat in 1/4. real - flat full: glioma_t2w_t1n +0.562 (p 9e-05), glioma_t2w_t2f -0.018 (p 0.5), glioma_t1n_t2f +0.892 (p 3e-08), glioma_t1n_t2w +0.151 (p 0.4).
- P1 on scr (expected to show the ordering, per the Dice follow-up): SUPPORTED: real > flat (p < 0.01) in 3/4 cells, real > noise in 3/4. real - flat full: glioma_t2w_t1n +1.283 (p 3e-07), glioma_t2w_t2f +0.181 (p 0.5), glioma_t1n_t2f +1.188 (p 3e-08), glioma_t1n_t2w +1.263 (p 3e-08).
- P2 on eq (real-arm X_target / S_target full, ratio of medians > 2) (PRE-REGISTERED reading): NOT SUPPORTED in 1/4 cells (glioma_t2w_t1n -10.44, glioma_t2w_t2f -4.35, glioma_t1n_t2f 1.10, glioma_t1n_t2w 2.17).
- P2 on scr (real-arm X_target / S_target full, ratio of medians > 2): NOT SUPPORTED in 1/4 cells (glioma_t2w_t1n -12.23, glioma_t2w_t2f -1.03, glioma_t1n_t2f 1.63, glioma_t1n_t2w 3.93).
- P3 on eq (real-arm median (X_s2 + X_s3) / X_full > 0.5) (PRE-REGISTERED reading): NOT SUPPORTED in 2/4 cells (glioma_t2w_t1n 0.58, glioma_t2w_t2f 0.40, glioma_t1n_t2f 0.47, glioma_t1n_t2w 0.53).
- P3 on le (real-arm median (X_s2 + X_s3) / X_full > 0.5): SUPPORTED in 3/4 cells (glioma_t2w_t1n 0.74, glioma_t2w_t2f 0.70, glioma_t1n_t2f 0.44, glioma_t1n_t2w 0.55).
- P3 on scr (real-arm median (X_s2 + X_s3) / X_full > 0.5): NOT SUPPORTED in 2/4 cells (glioma_t2w_t1n 0.55, glioma_t2w_t2f 0.44, glioma_t1n_t2f 0.50, glioma_t1n_t2w 0.41).
- P4 (units, x_eq): SUPPORTED: |real - flat| fraction of channels with displacement > 0.2 < 0.1 at both s2 and s3 in 4/4 cells; real >= flat + 0.1 at both in 0/4 (median paired difference: glioma_t2w_t1n s2 +0.000, s3 +0.000; glioma_t2w_t2f s2 +0.000, s3 +0.000; glioma_t1n_t2f s2 +0.000, s3 +0.000; glioma_t1n_t2w s2 +0.000, s3 +0.000).
