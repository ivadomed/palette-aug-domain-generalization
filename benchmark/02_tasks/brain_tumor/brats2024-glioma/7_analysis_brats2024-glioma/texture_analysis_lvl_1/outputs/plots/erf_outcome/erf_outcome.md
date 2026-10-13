# Per-voxel evidence by outcome pattern across the three fill arms (FINDINGS section 36)

Outcome per arm from the fold-mean log-odds of the voxel's ground-truth class: C > +2.0, F < -2.0, A ambiguous (counted, not explained). Pattern order: noise, flat, real. Seeds: up to 2 per (pattern, depth tertile) target voxels + 2 per pattern ring-background voxels (false positives), max 40 per case. Each arm explained through the gradient of its OWN predicted class at the voxel (`pred`); `gt` = gradient of the ground-truth class. Units: per-case median over seeds, then median over cases; Wilcoxon over cases.

## 1. Outcome patterns of the target voxels (share of definitive voxels per cell; in brackets: cases contributing)

| cell | definitive voxels | ambiguous share | CCC | CCF | CFC | CFF | FCC | FCF | FFC | FFF |
|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 1477855 | 0.57 | 0.131 (60) | 0.000 (22) | 0.002 (31) | 0.000 (18) | 0.088 (59) | 0.001 (26) | 0.137 (68) | 0.640 (70) |
| glioma_t1n_t2w | 1515427 | 0.56 | 0.623 (67) | 0.001 (35) | 0.000 (26) | 0.001 (36) | 0.131 (66) | 0.001 (38) | 0.026 (62) | 0.217 (70) |
| glioma_t2w_t2f | 2579882 | 0.26 | 0.918 (68) | 0.003 (35) | 0.001 (49) | 0.004 (35) | 0.004 (55) | 0.000 (28) | 0.004 (58) | 0.066 (69) |
| glioma_t2w_t1n | 1776262 | 0.49 | 0.256 (62) | 0.006 (46) | 0.016 (58) | 0.131 (62) | 0.001 (48) | 0.001 (36) | 0.007 (55) | 0.583 (70) |
| ms_flair_t2w | 81770 | 0.33 | 0.208 (8) | 0.004 (8) | 0.014 (7) | 0.003 (7) | 0.003 (7) | 0.001 (5) | 0.032 (7) | 0.735 (8) |
| ms_flair_t1w | 88236 | 0.27 | 0.065 (6) | 0.002 (6) | 0.004 (5) | 0.003 (5) | 0.004 (6) | 0.002 (5) | 0.020 (5) | 0.901 (8) |
| ms_t1w_flair | 69366 | 0.46 | 0.547 (8) | 0.001 (5) | 0.104 (8) | 0.009 (8) | 0.003 (5) | 0.000 (5) | 0.051 (8) | 0.285 (8) |
| ms_t1w_t2w | 72240 | 0.43 | 0.391 (8) | 0.002 (5) | 0.118 (8) | 0.009 (8) | 0.003 (6) | 0.000 (3) | 0.061 (8) | 0.416 (8) |
| breast_t1wce_t2w | 1143932 | 0.25 | 0.384 (40) | 0.005 (15) | 0.000 (2) | 0.003 (8) | 0.001 (11) | 0.000 (11) | 0.000 (5) | 0.607 (40) |
| breast_t2w_t1wce | 432657 | 0.34 | 0.503 (39) | 0.001 (16) | 0.001 (11) | 0.000 (13) | 0.002 (18) | 0.000 (12) | 0.003 (14) | 0.490 (40) |
| brain_t1w_t2w | 3292767 | 0.17 | 0.962 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.037 (8) |
| brain_t2w_t1w | 3139839 | 0.21 | 0.984 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.000 (8) | 0.000 (7) | 0.016 (8) |
| abd_t1in_t2spir | 107321 | 0.22 | 0.936 (4) | 0.000 (3) | 0.000 (1) | 0.000 (4) | 0.003 (4) | 0.000 (4) | 0.001 (3) | 0.059 (4) |
| abd_t1in_ct | 666839 | 0.09 | 0.976 (20) | 0.002 (3) | 0.000 (1) | 0.000 (2) | 0.000 (8) | 0.000 (4) | 0.000 (2) | 0.022 (20) |
| abd_t2spir_t1in | 313180 | 0.13 | 0.955 (4) | 0.000 (3) | 0.000 (4) | 0.000 (4) | 0.015 (4) | 0.001 (4) | 0.000 (3) | 0.028 (4) |
| abd_t2spir_t1out | 314435 | 0.13 | 0.947 (4) | 0.000 (2) | 0.000 (3) | 0.000 (3) | 0.004 (4) | 0.000 (4) | 0.001 (4) | 0.048 (4) |
| abd_t2spir_ct | 1217318 | 0.10 | 0.979 (20) | 0.000 (4) | 0.000 (5) | 0.002 (6) | 0.000 (6) | 0.000 (2) | 0.000 (4) | 0.018 (19) |
| mand_cbct_hanseg_ct | 4347592 | 0.10 | 0.851 (40) | 0.000 (12) | 0.000 (14) | 0.000 (19) | 0.000 (25) | 0.000 (9) | 0.000 (21) | 0.149 (40) |
| mand_cbct_pddca_ct | 4191126 | 0.07 | 0.942 (40) | 0.000 (11) | 0.000 (20) | 0.000 (22) | 0.000 (24) | 0.000 (8) | 0.000 (12) | 0.058 (40) |
| pelvis_mri_ct | 1799505 | 0.37 | 0.863 (25) | 0.000 (1) | 0.000 (1) | 0.000 (5) | 0.000 (20) | 0.000 (19) | 0.000 (2) | 0.136 (26) |
| pelvis_ct_mri | 908506 | 0.31 | 0.753 (21) | 0.007 (11) | 0.001 (4) | 0.001 (6) | 0.008 (21) | 0.003 (11) | 0.002 (15) | 0.226 (24) |
| pancreas_t1wce_t2w | 400636 | 0.24 | 0.779 (40) | 0.014 (39) | 0.004 (36) | 0.021 (38) | 0.005 (36) | 0.005 (38) | 0.002 (36) | 0.170 (40) |
| pancreas_t2w_t1wce | 438356 | 0.22 | 0.791 (40) | 0.008 (40) | 0.005 (38) | 0.013 (40) | 0.005 (38) | 0.005 (38) | 0.002 (37) | 0.171 (40) |

## 2. Does a model use something different where ONLY it succeeds? Within-arm comparison of its own-decision gradient, pattern 'only me' vs 'all correct' (same cases, matched by sampling per depth tertile)

### real: FFC - CCC, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.0036 | 0.22 | 3.2e-05 |
| glioma_t1n_t2w | 62 | -0.0015 | 0.35 | 0.14 |
| glioma_t2w_t2f | 57 | -0.0095 | 0.14 | 2.5e-09 |
| glioma_t2w_t1n | 51 | -0.0004 | 0.47 | 0.45 |
| ms_flair_t2w | 7 | -0.0116 | 0.29 | 0.38 |
| ms_flair_t1w | 5 | +0.0220 | 0.80 | 0.19 |
| ms_t1w_flair | 8 | +0.0171 | 0.75 | 0.055 |
| ms_t1w_t2w | 8 | +0.0061 | 0.62 | 0.84 |
| breast_t1wce_t2w | 5 | -0.0040 | 0.40 | 0.31 |
| breast_t2w_t1wce | 14 | -0.0027 | 0.14 | 0.017 |
| brain_t1w_t2w | 8 | -0.0078 | 0.38 | 0.11 |
| brain_t2w_t1w | 7 | -0.0079 | 0.29 | 0.16 |
| abd_t1in_t2spir | 3 | +0.0029 | 0.67 | nan |
| abd_t2spir_t1in | 3 | +0.0068 | 0.67 | nan |
| abd_t2spir_t1out | 4 | -0.0041 | 0.50 | nan |
| abd_t2spir_ct | 4 | +0.0014 | 0.50 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0118 | 0.90 | 4.1e-05 |
| mand_cbct_pddca_ct | 12 | +0.0137 | 0.83 | 0.0049 |
| pelvis_ct_mri | 14 | -0.0043 | 0.29 | 0.025 |
| pancreas_t1wce_t2w | 36 | +0.0002 | 0.53 | 0.9 |
| pancreas_t2w_t1wce | 37 | +0.0015 | 0.65 | 0.012 |
pooled over cells: median of cell medians -0.0004, cells with d > 0: 10 / 21, p < 0.05 same sign: 3 +, 4 -

### real: FFC - CCC, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.0068 | 0.42 | 0.077 |
| glioma_t1n_t2w | 62 | -0.0034 | 0.48 | 0.61 |
| glioma_t2w_t2f | 57 | -0.0241 | 0.32 | 0.00047 |
| glioma_t2w_t1n | 51 | -0.0194 | 0.47 | 0.23 |
| ms_flair_t2w | 7 | -0.0366 | 0.14 | 0.047 |
| ms_flair_t1w | 5 | -0.0243 | 0.40 | 0.44 |
| ms_t1w_flair | 8 | -0.0094 | 0.25 | 0.38 |
| ms_t1w_t2w | 8 | -0.0202 | 0.25 | 0.039 |
| breast_t1wce_t2w | 5 | -0.0222 | 0.00 | 0.062 |
| breast_t2w_t1wce | 14 | -0.0040 | 0.36 | 0.76 |
| brain_t1w_t2w | 8 | -0.1530 | 0.00 | 0.0078 |
| brain_t2w_t1w | 7 | -0.2040 | 0.00 | 0.016 |
| abd_t1in_t2spir | 3 | -0.2708 | 0.00 | nan |
| abd_t2spir_t1in | 3 | -0.2428 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.2517 | 0.25 | nan |
| abd_t2spir_ct | 4 | -0.1965 | 0.00 | nan |
| mand_cbct_hanseg_ct | 21 | +0.1582 | 0.95 | 1.9e-06 |
| mand_cbct_pddca_ct | 12 | +0.1717 | 0.92 | 0.0015 |
| pelvis_ct_mri | 14 | -0.0084 | 0.43 | 0.3 |
| pancreas_t1wce_t2w | 36 | -0.0199 | 0.31 | 0.0041 |
| pancreas_t2w_t1wce | 37 | -0.0001 | 0.49 | 0.3 |
pooled over cells: median of cell medians -0.0202, cells with d > 0: 2 / 21, p < 0.05 same sign: 2 +, 6 -

### real: FFC - CCC, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.0076 | 0.40 | 0.035 |
| glioma_t1n_t2w | 62 | -0.0080 | 0.39 | 0.086 |
| glioma_t2w_t2f | 57 | -0.0011 | 0.47 | 0.63 |
| glioma_t2w_t1n | 51 | -0.0006 | 0.45 | 0.51 |
| ms_flair_t2w | 7 | -0.0739 | 0.29 | 0.38 |
| ms_flair_t1w | 5 | -0.0329 | 0.20 | 0.19 |
| ms_t1w_flair | 8 | -0.0560 | 0.38 | 0.11 |
| ms_t1w_t2w | 8 | -0.0317 | 0.12 | 0.016 |
| breast_t1wce_t2w | 5 | -0.0062 | 0.20 | 0.12 |
| breast_t2w_t1wce | 14 | -0.0146 | 0.36 | 0.71 |
| brain_t1w_t2w | 8 | -0.0094 | 0.25 | 0.31 |
| brain_t2w_t1w | 7 | -0.0024 | 0.43 | 0.58 |
| abd_t1in_t2spir | 3 | +0.0557 | 1.00 | nan |
| abd_t2spir_t1in | 3 | -0.0014 | 0.33 | nan |
| abd_t2spir_t1out | 4 | +0.0127 | 0.75 | nan |
| abd_t2spir_ct | 4 | +0.0011 | 0.50 | nan |
| mand_cbct_hanseg_ct | 21 | -0.0152 | 0.24 | 0.0019 |
| mand_cbct_pddca_ct | 12 | -0.0159 | 0.17 | 0.0049 |
| pelvis_ct_mri | 14 | +0.0035 | 0.57 | 0.86 |
| pancreas_t1wce_t2w | 36 | +0.0036 | 0.61 | 0.17 |
| pancreas_t2w_t1wce | 37 | +0.0087 | 0.76 | 0.017 |
pooled over cells: median of cell medians -0.0062, cells with d > 0: 6 / 21, p < 0.05 same sign: 1 +, 4 -

### real: FFC - CCC, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.0141 | 0.42 | 0.07 |
| glioma_t1n_t2w | 62 | -0.0110 | 0.42 | 0.24 |
| glioma_t2w_t2f | 57 | +0.0026 | 0.54 | 0.3 |
| glioma_t2w_t1n | 51 | -0.0064 | 0.37 | 0.13 |
| ms_flair_t2w | 7 | -0.0766 | 0.29 | 0.16 |
| ms_flair_t1w | 5 | -0.0305 | 0.40 | 0.31 |
| ms_t1w_flair | 8 | -0.0606 | 0.38 | 0.11 |
| ms_t1w_t2w | 8 | -0.0466 | 0.12 | 0.023 |
| breast_t1wce_t2w | 5 | -0.0073 | 0.40 | 0.44 |
| breast_t2w_t1wce | 14 | -0.0170 | 0.36 | 0.36 |
| brain_t1w_t2w | 8 | -0.0026 | 0.38 | 0.55 |
| brain_t2w_t1w | 7 | +0.0029 | 0.57 | 0.94 |
| abd_t1in_t2spir | 3 | +0.0476 | 1.00 | nan |
| abd_t2spir_t1in | 3 | +0.0035 | 0.67 | nan |
| abd_t2spir_t1out | 4 | +0.0276 | 1.00 | nan |
| abd_t2spir_ct | 4 | +0.0001 | 0.50 | nan |
| mand_cbct_hanseg_ct | 21 | -0.0206 | 0.19 | 0.001 |
| mand_cbct_pddca_ct | 12 | -0.0199 | 0.08 | 0.0015 |
| pelvis_ct_mri | 14 | +0.0034 | 0.50 | 1 |
| pancreas_t1wce_t2w | 36 | -0.0011 | 0.47 | 0.87 |
| pancreas_t2w_t1wce | 37 | +0.0109 | 0.68 | 0.059 |
pooled over cells: median of cell medians -0.0064, cells with d > 0: 8 / 21, p < 0.05 same sign: 0 +, 3 -

### real: FFC - CCC, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.0126 | 0.42 | 0.15 |
| glioma_t1n_t2w | 62 | +0.0100 | 0.58 | 0.27 |
| glioma_t2w_t2f | 57 | +0.0314 | 0.75 | 0.00012 |
| glioma_t2w_t1n | 51 | +0.0123 | 0.73 | 0.0071 |
| ms_flair_t2w | 7 | +0.0278 | 0.86 | 0.3 |
| ms_flair_t1w | 5 | -0.0117 | 0.40 | 1 |
| ms_t1w_flair | 8 | +0.0219 | 0.62 | 0.25 |
| ms_t1w_t2w | 8 | -0.0166 | 0.38 | 0.74 |
| breast_t1wce_t2w | 5 | +0.0741 | 0.60 | 0.31 |
| breast_t2w_t1wce | 14 | -0.0054 | 0.36 | 0.81 |
| brain_t1w_t2w | 8 | +0.0098 | 0.75 | 0.15 |
| brain_t2w_t1w | 7 | +0.0280 | 0.71 | 0.078 |
| abd_t1in_t2spir | 3 | -0.0576 | 0.33 | nan |
| abd_t2spir_t1in | 3 | +0.1146 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.1616 | 0.75 | nan |
| abd_t2spir_ct | 4 | -0.0598 | 0.25 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0112 | 0.67 | 0.26 |
| mand_cbct_pddca_ct | 12 | +0.0134 | 0.67 | 0.68 |
| pelvis_ct_mri | 14 | +0.0165 | 0.57 | 0.5 |
| pancreas_t1wce_t2w | 36 | +0.0004 | 0.50 | 0.86 |
| pancreas_t2w_t1wce | 37 | +0.0162 | 0.62 | 0.16 |
pooled over cells: median of cell medians +0.0123, cells with d > 0: 15 / 21, p < 0.05 same sign: 2 +, 0 -

### flat: FCF - CCC, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 24 | -0.0002 | 0.46 | 0.75 |
| glioma_t1n_t2w | 38 | -0.0020 | 0.39 | 0.26 |
| glioma_t2w_t2f | 28 | -0.0092 | 0.29 | 0.019 |
| glioma_t2w_t1n | 35 | -0.0010 | 0.40 | 0.14 |
| ms_flair_t2w | 5 | -0.0324 | 0.00 | 0.062 |
| ms_flair_t1w | 5 | -0.0300 | 0.00 | 0.062 |
| ms_t1w_flair | 4 | +0.0090 | 0.75 | nan |
| ms_t1w_t2w | 3 | +0.0024 | 0.67 | nan |
| breast_t1wce_t2w | 11 | -0.0009 | 0.45 | 0.58 |
| breast_t2w_t1wce | 12 | -0.0005 | 0.42 | 0.27 |
| brain_t1w_t2w | 8 | -0.0105 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.0067 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0027 | 0.50 | nan |
| abd_t1in_ct | 4 | +0.0002 | 0.50 | nan |
| abd_t2spir_t1in | 4 | -0.0008 | 0.50 | nan |
| abd_t2spir_t1out | 4 | +0.0117 | 0.75 | nan |
| mand_cbct_hanseg_ct | 9 | +0.0293 | 0.78 | 0.02 |
| mand_cbct_pddca_ct | 8 | +0.0057 | 0.88 | 0.016 |
| pelvis_mri_ct | 19 | -0.0129 | 0.00 | 3.8e-06 |
| pelvis_ct_mri | 11 | -0.0027 | 0.27 | 0.1 |
| pancreas_t1wce_t2w | 38 | -0.0019 | 0.18 | 4.6e-07 |
| pancreas_t2w_t1wce | 38 | +0.0002 | 0.58 | 0.21 |
pooled over cells: median of cell medians -0.0009, cells with d > 0: 8 / 22, p < 0.05 same sign: 2 +, 5 -

### flat: FCF - CCC, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 24 | -0.0052 | 0.46 | 0.62 |
| glioma_t1n_t2w | 38 | -0.0118 | 0.42 | 0.11 |
| glioma_t2w_t2f | 28 | -0.0059 | 0.39 | 0.6 |
| glioma_t2w_t1n | 35 | -0.0403 | 0.29 | 9.2e-05 |
| ms_flair_t2w | 5 | -0.0688 | 0.00 | 0.062 |
| ms_flair_t1w | 5 | -0.0239 | 0.20 | 0.12 |
| ms_t1w_flair | 4 | -0.0106 | 0.25 | nan |
| ms_t1w_t2w | 3 | -0.0064 | 0.33 | nan |
| breast_t1wce_t2w | 11 | -0.0116 | 0.18 | 0.019 |
| breast_t2w_t1wce | 12 | -0.0009 | 0.42 | 1 |
| brain_t1w_t2w | 8 | -0.2042 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.2219 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.1485 | 0.00 | nan |
| abd_t1in_ct | 4 | -0.2282 | 0.00 | nan |
| abd_t2spir_t1in | 4 | -0.2467 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.1935 | 0.00 | nan |
| mand_cbct_hanseg_ct | 9 | +0.1438 | 0.89 | 0.0078 |
| mand_cbct_pddca_ct | 8 | +0.1064 | 1.00 | 0.0078 |
| pelvis_mri_ct | 19 | -0.1234 | 0.05 | 7.6e-06 |
| pelvis_ct_mri | 11 | -0.0477 | 0.36 | 0.1 |
| pancreas_t1wce_t2w | 38 | -0.0071 | 0.39 | 0.048 |
| pancreas_t2w_t1wce | 38 | -0.0166 | 0.32 | 0.01 |
pooled over cells: median of cell medians -0.0203, cells with d > 0: 2 / 22, p < 0.05 same sign: 2 +, 7 -

### flat: FCF - CCC, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 24 | +0.0210 | 0.79 | 0.0022 |
| glioma_t1n_t2w | 38 | +0.0122 | 0.63 | 0.17 |
| glioma_t2w_t2f | 28 | -0.0118 | 0.32 | 0.077 |
| glioma_t2w_t1n | 35 | -0.0065 | 0.40 | 0.11 |
| ms_flair_t2w | 5 | +0.0407 | 0.60 | 0.31 |
| ms_flair_t1w | 5 | +0.0084 | 0.60 | 0.44 |
| ms_t1w_flair | 4 | -0.0450 | 0.25 | nan |
| ms_t1w_t2w | 3 | +0.0572 | 1.00 | nan |
| breast_t1wce_t2w | 11 | -0.0359 | 0.27 | 0.024 |
| breast_t2w_t1wce | 12 | -0.0745 | 0.00 | 0.00049 |
| brain_t1w_t2w | 8 | +0.0054 | 0.62 | 0.64 |
| brain_t2w_t1w | 8 | -0.0063 | 0.38 | 0.25 |
| abd_t1in_t2spir | 4 | +0.0254 | 1.00 | nan |
| abd_t1in_ct | 4 | -0.0079 | 0.25 | nan |
| abd_t2spir_t1in | 4 | -0.0289 | 0.00 | nan |
| abd_t2spir_t1out | 4 | +0.0260 | 0.50 | nan |
| mand_cbct_hanseg_ct | 9 | -0.0050 | 0.44 | 1 |
| mand_cbct_pddca_ct | 8 | -0.0174 | 0.12 | 0.016 |
| pelvis_mri_ct | 19 | -0.0220 | 0.11 | 0.00034 |
| pelvis_ct_mri | 11 | +0.0136 | 0.64 | 0.9 |
| pancreas_t1wce_t2w | 38 | -0.0090 | 0.16 | 8.7e-06 |
| pancreas_t2w_t1wce | 38 | +0.0007 | 0.58 | 0.23 |
pooled over cells: median of cell medians -0.0057, cells with d > 0: 10 / 22, p < 0.05 same sign: 1 +, 5 -

### flat: FCF - CCC, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 24 | +0.0390 | 0.79 | 0.00065 |
| glioma_t1n_t2w | 38 | +0.0343 | 0.63 | 0.048 |
| glioma_t2w_t2f | 28 | -0.0024 | 0.46 | 0.27 |
| glioma_t2w_t1n | 35 | -0.0085 | 0.37 | 0.025 |
| ms_flair_t2w | 5 | +0.0610 | 0.80 | 0.19 |
| ms_flair_t1w | 5 | +0.0090 | 0.60 | 0.44 |
| ms_t1w_flair | 4 | -0.0639 | 0.25 | nan |
| ms_t1w_t2w | 3 | +0.0535 | 1.00 | nan |
| breast_t1wce_t2w | 11 | -0.0412 | 0.18 | 0.014 |
| breast_t2w_t1wce | 12 | -0.0902 | 0.00 | 0.00049 |
| brain_t1w_t2w | 8 | +0.0108 | 0.62 | 0.31 |
| brain_t2w_t1w | 8 | +0.0020 | 0.62 | 1 |
| abd_t1in_t2spir | 4 | +0.0279 | 0.75 | nan |
| abd_t1in_ct | 4 | -0.0098 | 0.50 | nan |
| abd_t2spir_t1in | 4 | -0.0132 | 0.00 | nan |
| abd_t2spir_t1out | 4 | +0.0281 | 0.50 | nan |
| mand_cbct_hanseg_ct | 9 | -0.0085 | 0.22 | 0.36 |
| mand_cbct_pddca_ct | 8 | -0.0277 | 0.00 | 0.0078 |
| pelvis_mri_ct | 19 | -0.0574 | 0.05 | 7.6e-06 |
| pelvis_ct_mri | 11 | +0.0324 | 0.64 | 0.64 |
| pancreas_t1wce_t2w | 38 | -0.0169 | 0.18 | 2.8e-06 |
| pancreas_t2w_t1wce | 38 | +0.0043 | 0.68 | 0.026 |
pooled over cells: median of cell medians -0.0002, cells with d > 0: 11 / 22, p < 0.05 same sign: 3 +, 6 -

### flat: FCF - CCC, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 24 | +0.0313 | 0.79 | 0.00037 |
| glioma_t1n_t2w | 38 | +0.0359 | 0.76 | 0.00034 |
| glioma_t2w_t2f | 28 | +0.0212 | 0.71 | 0.0044 |
| glioma_t2w_t1n | 35 | +0.0120 | 0.63 | 0.17 |
| ms_flair_t2w | 5 | -0.0514 | 0.20 | 0.12 |
| ms_flair_t1w | 5 | +0.0217 | 0.60 | 0.62 |
| ms_t1w_flair | 4 | -0.0054 | 0.50 | nan |
| ms_t1w_t2w | 3 | -0.1089 | 0.33 | nan |
| breast_t1wce_t2w | 11 | -0.0050 | 0.45 | 0.32 |
| breast_t2w_t1wce | 12 | -0.0018 | 0.50 | 0.85 |
| brain_t1w_t2w | 8 | +0.0235 | 0.75 | 0.46 |
| brain_t2w_t1w | 8 | +0.0273 | 0.75 | 0.25 |
| abd_t1in_t2spir | 4 | -0.0260 | 0.25 | nan |
| abd_t1in_ct | 4 | +0.0472 | 0.75 | nan |
| abd_t2spir_t1in | 4 | +0.0600 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.1118 | 0.75 | nan |
| mand_cbct_hanseg_ct | 9 | -0.0009 | 0.44 | 1 |
| mand_cbct_pddca_ct | 8 | +0.0127 | 0.50 | 0.74 |
| pelvis_mri_ct | 19 | -0.0411 | 0.32 | 0.011 |
| pelvis_ct_mri | 11 | +0.0175 | 0.73 | 0.37 |
| pancreas_t1wce_t2w | 38 | -0.0053 | 0.42 | 0.4 |
| pancreas_t2w_t1wce | 38 | +0.0227 | 0.66 | 0.093 |
pooled over cells: median of cell medians +0.0151, cells with d > 0: 13 / 22, p < 0.05 same sign: 3 +, 1 -

### noise: CFF - CCC, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 17 | -0.0006 | 0.41 | 0.43 |
| glioma_t1n_t2w | 36 | +0.0018 | 0.64 | 0.12 |
| glioma_t2w_t2f | 34 | +0.0002 | 0.50 | 0.88 |
| glioma_t2w_t1n | 59 | -0.0009 | 0.44 | 0.23 |
| ms_flair_t2w | 7 | -0.0131 | 0.43 | 0.38 |
| ms_flair_t1w | 5 | -0.0063 | 0.20 | 0.31 |
| ms_t1w_flair | 8 | +0.0085 | 0.75 | 0.46 |
| ms_t1w_t2w | 8 | -0.0049 | 0.38 | 0.95 |
| breast_t1wce_t2w | 8 | -0.0130 | 0.25 | 0.11 |
| breast_t2w_t1wce | 12 | +0.0010 | 0.58 | 0.38 |
| brain_t1w_t2w | 8 | -0.0052 | 0.38 | 0.38 |
| brain_t2w_t1w | 8 | -0.0014 | 0.38 | 0.95 |
| abd_t1in_t2spir | 4 | +0.0030 | 0.50 | nan |
| abd_t2spir_t1in | 4 | +0.0036 | 0.50 | nan |
| abd_t2spir_t1out | 3 | -0.0011 | 0.33 | nan |
| abd_t2spir_ct | 6 | -0.0006 | 0.50 | 0.69 |
| mand_cbct_hanseg_ct | 19 | +0.0032 | 0.53 | 0.47 |
| mand_cbct_pddca_ct | 22 | +0.0206 | 0.77 | 0.0011 |
| pelvis_mri_ct | 5 | -0.0061 | 0.20 | 0.19 |
| pelvis_ct_mri | 6 | -0.0023 | 0.33 | 0.56 |
| pancreas_t1wce_t2w | 37 | -0.0010 | 0.43 | 0.14 |
| pancreas_t2w_t1wce | 40 | -0.0006 | 0.40 | 0.46 |
pooled over cells: median of cell medians -0.0007, cells with d > 0: 8 / 22, p < 0.05 same sign: 1 +, 0 -

### noise: CFF - CCC, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 17 | -0.0221 | 0.18 | 0.0021 |
| glioma_t1n_t2w | 36 | -0.0259 | 0.33 | 0.0086 |
| glioma_t2w_t2f | 34 | +0.0028 | 0.53 | 0.72 |
| glioma_t2w_t1n | 59 | -0.0001 | 0.49 | 0.76 |
| ms_flair_t2w | 7 | -0.0352 | 0.43 | 0.3 |
| ms_flair_t1w | 5 | -0.0225 | 0.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0096 | 0.38 | 0.31 |
| ms_t1w_t2w | 8 | -0.0142 | 0.38 | 0.25 |
| breast_t1wce_t2w | 8 | -0.0016 | 0.50 | 1 |
| breast_t2w_t1wce | 12 | -0.0087 | 0.33 | 0.73 |
| brain_t1w_t2w | 8 | -0.1497 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1004 | 0.12 | 0.039 |
| abd_t1in_t2spir | 4 | -0.2064 | 0.25 | nan |
| abd_t2spir_t1in | 4 | -0.1032 | 0.25 | nan |
| abd_t2spir_t1out | 3 | -0.2257 | 0.00 | nan |
| abd_t2spir_ct | 6 | -0.1017 | 0.00 | 0.031 |
| mand_cbct_hanseg_ct | 19 | +0.1052 | 0.68 | 0.016 |
| mand_cbct_pddca_ct | 22 | +0.1296 | 0.68 | 0.0029 |
| pelvis_mri_ct | 5 | -0.0290 | 0.20 | 0.62 |
| pelvis_ct_mri | 6 | -0.0677 | 0.00 | 0.031 |
| pancreas_t1wce_t2w | 37 | -0.0194 | 0.24 | 0.00016 |
| pancreas_t2w_t1wce | 40 | -0.0276 | 0.20 | 3.8e-05 |
pooled over cells: median of cell medians -0.0242, cells with d > 0: 3 / 22, p < 0.05 same sign: 2 +, 8 -

### noise: CFF - CCC, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 17 | +0.0238 | 0.59 | 0.17 |
| glioma_t1n_t2w | 36 | +0.0103 | 0.64 | 0.067 |
| glioma_t2w_t2f | 34 | +0.0246 | 0.68 | 0.012 |
| glioma_t2w_t1n | 59 | -0.0025 | 0.49 | 0.15 |
| ms_flair_t2w | 7 | -0.0725 | 0.14 | 0.3 |
| ms_flair_t1w | 5 | +0.0865 | 1.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0013 | 0.50 | 0.95 |
| ms_t1w_t2w | 8 | +0.1054 | 0.75 | 0.078 |
| breast_t1wce_t2w | 8 | -0.1273 | 0.00 | 0.0078 |
| breast_t2w_t1wce | 12 | -0.0185 | 0.33 | 0.38 |
| brain_t1w_t2w | 8 | -0.0032 | 0.50 | 0.84 |
| brain_t2w_t1w | 8 | -0.0053 | 0.38 | 0.84 |
| abd_t1in_t2spir | 4 | +0.0156 | 0.75 | nan |
| abd_t2spir_t1in | 4 | -0.0152 | 0.25 | nan |
| abd_t2spir_t1out | 3 | +0.0047 | 0.67 | nan |
| abd_t2spir_ct | 6 | -0.0044 | 0.50 | 0.56 |
| mand_cbct_hanseg_ct | 19 | -0.0098 | 0.21 | 0.011 |
| mand_cbct_pddca_ct | 22 | -0.0037 | 0.45 | 0.32 |
| pelvis_mri_ct | 5 | -0.0204 | 0.40 | 0.81 |
| pelvis_ct_mri | 6 | -0.0118 | 0.33 | 0.56 |
| pancreas_t1wce_t2w | 37 | -0.0038 | 0.35 | 0.0079 |
| pancreas_t2w_t1wce | 40 | -0.0103 | 0.38 | 0.0019 |
pooled over cells: median of cell medians -0.0038, cells with d > 0: 7 / 22, p < 0.05 same sign: 1 +, 4 -

### noise: CFF - CCC, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 17 | +0.0167 | 0.59 | 0.26 |
| glioma_t1n_t2w | 36 | +0.0039 | 0.50 | 0.18 |
| glioma_t2w_t2f | 34 | +0.0311 | 0.65 | 0.0073 |
| glioma_t2w_t1n | 59 | -0.0037 | 0.44 | 0.37 |
| ms_flair_t2w | 7 | -0.0391 | 0.14 | 0.3 |
| ms_flair_t1w | 5 | +0.1232 | 1.00 | 0.062 |
| ms_t1w_flair | 8 | +0.0035 | 0.50 | 0.84 |
| ms_t1w_t2w | 8 | +0.1121 | 0.62 | 0.11 |
| breast_t1wce_t2w | 8 | -0.1225 | 0.00 | 0.0078 |
| breast_t2w_t1wce | 12 | -0.0222 | 0.33 | 0.34 |
| brain_t1w_t2w | 8 | -0.0006 | 0.50 | 0.84 |
| brain_t2w_t1w | 8 | -0.0074 | 0.38 | 0.95 |
| abd_t1in_t2spir | 4 | +0.0692 | 0.75 | nan |
| abd_t2spir_t1in | 4 | -0.0283 | 0.25 | nan |
| abd_t2spir_t1out | 3 | +0.0284 | 0.67 | nan |
| abd_t2spir_ct | 6 | +0.0070 | 0.50 | 1 |
| mand_cbct_hanseg_ct | 19 | -0.0099 | 0.21 | 0.0082 |
| mand_cbct_pddca_ct | 22 | -0.0043 | 0.45 | 0.26 |
| pelvis_mri_ct | 5 | -0.0333 | 0.40 | 0.62 |
| pelvis_ct_mri | 6 | -0.0225 | 0.33 | 0.44 |
| pancreas_t1wce_t2w | 37 | -0.0075 | 0.30 | 0.0056 |
| pancreas_t2w_t1wce | 40 | -0.0103 | 0.40 | 0.0031 |
pooled over cells: median of cell medians -0.0040, cells with d > 0: 9 / 22, p < 0.05 same sign: 1 +, 4 -

### noise: CFF - CCC, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 17 | -0.0043 | 0.47 | 0.78 |
| glioma_t1n_t2w | 36 | +0.0159 | 0.64 | 0.11 |
| glioma_t2w_t2f | 34 | +0.0160 | 0.62 | 0.14 |
| glioma_t2w_t1n | 59 | -0.0026 | 0.41 | 0.27 |
| ms_flair_t2w | 7 | +0.0037 | 0.57 | 0.94 |
| ms_flair_t1w | 5 | -0.0286 | 0.20 | 0.31 |
| ms_t1w_flair | 8 | +0.0382 | 0.75 | 0.055 |
| ms_t1w_t2w | 8 | +0.0364 | 0.62 | 0.15 |
| breast_t1wce_t2w | 8 | +0.0288 | 0.75 | 0.15 |
| breast_t2w_t1wce | 12 | +0.0259 | 0.67 | 0.34 |
| brain_t1w_t2w | 8 | +0.0189 | 0.62 | 0.95 |
| brain_t2w_t1w | 8 | +0.0146 | 0.62 | 0.31 |
| abd_t1in_t2spir | 4 | +0.0003 | 0.50 | nan |
| abd_t2spir_t1in | 4 | +0.0340 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.2340 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.1117 | 1.00 | 0.031 |
| mand_cbct_hanseg_ct | 19 | -0.0069 | 0.47 | 0.95 |
| mand_cbct_pddca_ct | 22 | -0.0248 | 0.41 | 0.54 |
| pelvis_mri_ct | 5 | -0.0573 | 0.20 | 0.12 |
| pelvis_ct_mri | 6 | +0.0244 | 0.67 | 0.84 |
| pancreas_t1wce_t2w | 37 | +0.0095 | 0.57 | 0.81 |
| pancreas_t2w_t1wce | 40 | -0.0266 | 0.28 | 0.077 |
pooled over cells: median of cell medians +0.0152, cells with d > 0: 15 / 22, p < 0.05 same sign: 1 +, 0 -


## 3. At the SAME voxels, what does each arm rely on? Own-decision gradients, real - flat and real - noise, per pattern

### FFC (only real correct): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.0025 | 0.85 | 4.3e-09 |
| glioma_t1n_t2w | 62 | +0.0052 | 0.85 | 7.6e-10 |
| glioma_t2w_t2f | 58 | +0.0022 | 0.79 | 7.7e-05 |
| glioma_t2w_t1n | 55 | +0.0050 | 0.84 | 7.8e-08 |
| ms_flair_t2w | 7 | -0.0040 | 0.43 | 0.22 |
| ms_flair_t1w | 5 | +0.0061 | 1.00 | 0.062 |
| ms_t1w_flair | 8 | +0.0027 | 0.75 | 0.25 |
| ms_t1w_t2w | 8 | +0.0005 | 0.50 | 0.55 |
| breast_t1wce_t2w | 5 | +0.0048 | 0.80 | 0.19 |
| breast_t2w_t1wce | 14 | -0.0001 | 0.36 | 0.54 |
| brain_t1w_t2w | 8 | +0.0001 | 0.50 | 0.64 |
| brain_t2w_t1w | 7 | +0.0008 | 0.57 | 0.94 |
| abd_t1in_t2spir | 3 | -0.0078 | 0.00 | nan |
| abd_t2spir_t1in | 3 | +0.0064 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0031 | 0.75 | nan |
| abd_t2spir_ct | 4 | -0.0005 | 0.25 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0012 | 0.52 | 0.71 |
| mand_cbct_pddca_ct | 12 | -0.0002 | 0.50 | 0.57 |
| pelvis_ct_mri | 15 | +0.0035 | 0.67 | 0.064 |
| pancreas_t1wce_t2w | 36 | +0.0012 | 0.67 | 0.0039 |
| pancreas_t2w_t1wce | 37 | -0.0017 | 0.35 | 0.0096 |
pooled over cells: median of cell medians +0.0012, cells with d > 0: 15 / 21, p < 0.05 same sign: 5 +, 1 -

### FFC (only real correct): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.0003 | 0.53 | 0.31 |
| glioma_t1n_t2w | 62 | +0.0064 | 0.89 | 7.6e-10 |
| glioma_t2w_t2f | 58 | +0.0005 | 0.53 | 0.51 |
| glioma_t2w_t1n | 55 | -0.0001 | 0.49 | 0.89 |
| ms_flair_t2w | 7 | -0.0279 | 0.00 | 0.016 |
| ms_flair_t1w | 5 | -0.0061 | 0.40 | 0.62 |
| ms_t1w_flair | 8 | -0.0033 | 0.38 | 0.38 |
| ms_t1w_t2w | 8 | +0.0025 | 0.50 | 0.84 |
| breast_t1wce_t2w | 5 | -0.0098 | 0.00 | 0.062 |
| breast_t2w_t1wce | 14 | -0.0072 | 0.14 | 0.00061 |
| brain_t1w_t2w | 8 | -0.0103 | 0.00 | 0.0078 |
| brain_t2w_t1w | 7 | -0.0061 | 0.00 | 0.016 |
| abd_t1in_t2spir | 3 | -0.0084 | 0.00 | nan |
| abd_t2spir_t1in | 3 | +0.0061 | 0.67 | nan |
| abd_t2spir_t1out | 4 | +0.0016 | 0.75 | nan |
| abd_t2spir_ct | 4 | +0.0028 | 0.75 | nan |
| mand_cbct_hanseg_ct | 21 | -0.0185 | 0.05 | 1.9e-06 |
| mand_cbct_pddca_ct | 12 | -0.0153 | 0.08 | 0.0015 |
| pelvis_ct_mri | 15 | +0.0059 | 0.80 | 0.00085 |
| pancreas_t1wce_t2w | 36 | -0.0003 | 0.47 | 0.96 |
| pancreas_t2w_t1wce | 37 | -0.0007 | 0.35 | 0.19 |
pooled over cells: median of cell medians -0.0007, cells with d > 0: 8 / 21, p < 0.05 same sign: 2 +, 6 -

### FFC (only real correct): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.0196 | 0.94 | 3.4e-11 |
| glioma_t1n_t2w | 62 | +0.0054 | 0.61 | 0.026 |
| glioma_t2w_t2f | 58 | +0.0059 | 0.74 | 6.6e-05 |
| glioma_t2w_t1n | 55 | +0.0179 | 0.84 | 2.8e-08 |
| ms_flair_t2w | 7 | -0.0067 | 0.29 | 0.16 |
| ms_flair_t1w | 5 | +0.0116 | 1.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0000 | 0.50 | 0.55 |
| ms_t1w_t2w | 8 | +0.0019 | 0.62 | 0.2 |
| breast_t1wce_t2w | 5 | +0.0117 | 1.00 | 0.062 |
| breast_t2w_t1wce | 14 | -0.0017 | 0.36 | 0.76 |
| brain_t1w_t2w | 8 | +0.0522 | 0.75 | 0.15 |
| brain_t2w_t1w | 7 | -0.0273 | 0.29 | 0.69 |
| abd_t1in_t2spir | 3 | +0.0090 | 0.67 | nan |
| abd_t2spir_t1in | 3 | +0.0392 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0374 | 1.00 | nan |
| abd_t2spir_ct | 4 | -0.0264 | 0.00 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0387 | 0.90 | 0.00029 |
| mand_cbct_pddca_ct | 12 | +0.0561 | 1.00 | 0.00049 |
| pelvis_ct_mri | 15 | -0.0096 | 0.40 | 0.25 |
| pancreas_t1wce_t2w | 36 | +0.0055 | 0.67 | 0.0051 |
| pancreas_t2w_t1wce | 37 | -0.0044 | 0.35 | 0.047 |
pooled over cells: median of cell medians +0.0059, cells with d > 0: 14 / 21, p < 0.05 same sign: 7 +, 1 -

### FFC (only real correct): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.0162 | 0.84 | 2.5e-08 |
| glioma_t1n_t2w | 62 | +0.0227 | 0.82 | 1.4e-08 |
| glioma_t2w_t2f | 58 | -0.0051 | 0.33 | 0.033 |
| glioma_t2w_t1n | 55 | -0.0010 | 0.47 | 0.57 |
| ms_flair_t2w | 7 | -0.0439 | 0.00 | 0.016 |
| ms_flair_t1w | 5 | +0.0029 | 0.60 | 0.31 |
| ms_t1w_flair | 8 | -0.0099 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.0093 | 0.25 | 0.039 |
| breast_t1wce_t2w | 5 | -0.0321 | 0.00 | 0.062 |
| breast_t2w_t1wce | 14 | -0.0437 | 0.00 | 0.00012 |
| brain_t1w_t2w | 8 | +0.0474 | 0.75 | 0.11 |
| brain_t2w_t1w | 7 | -0.0362 | 0.29 | 0.69 |
| abd_t1in_t2spir | 3 | +0.0020 | 0.67 | nan |
| abd_t2spir_t1in | 3 | +0.0660 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0655 | 0.75 | nan |
| abd_t2spir_ct | 4 | +0.0768 | 1.00 | nan |
| mand_cbct_hanseg_ct | 21 | -0.1458 | 0.00 | 9.5e-07 |
| mand_cbct_pddca_ct | 12 | -0.1475 | 0.00 | 0.00049 |
| pelvis_ct_mri | 15 | +0.0196 | 0.67 | 0.17 |
| pancreas_t1wce_t2w | 36 | -0.0077 | 0.25 | 0.0041 |
| pancreas_t2w_t1wce | 37 | -0.0065 | 0.38 | 0.13 |
pooled over cells: median of cell medians -0.0051, cells with d > 0: 9 / 21, p < 0.05 same sign: 2 +, 8 -

### FFC (only real correct): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.0764 | 1.00 | 7.6e-13 |
| glioma_t1n_t2w | 62 | +0.0723 | 1.00 | 7.6e-12 |
| glioma_t2w_t2f | 58 | +0.0548 | 1.00 | 3.5e-11 |
| glioma_t2w_t1n | 55 | +0.0505 | 1.00 | 1.1e-10 |
| ms_flair_t2w | 7 | +0.0191 | 0.57 | 0.3 |
| ms_flair_t1w | 5 | -0.0189 | 0.40 | 0.44 |
| ms_t1w_flair | 8 | -0.0530 | 0.12 | 0.023 |
| ms_t1w_t2w | 8 | -0.0167 | 0.38 | 0.64 |
| breast_t1wce_t2w | 5 | +0.0280 | 1.00 | 0.062 |
| breast_t2w_t1wce | 14 | +0.0801 | 1.00 | 0.00012 |
| brain_t1w_t2w | 8 | +0.0179 | 0.88 | 0.016 |
| brain_t2w_t1w | 7 | +0.0159 | 1.00 | 0.016 |
| abd_t1in_t2spir | 3 | +0.0450 | 1.00 | nan |
| abd_t2spir_t1in | 3 | -0.0249 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0105 | 0.00 | nan |
| abd_t2spir_ct | 4 | -0.0331 | 0.25 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0375 | 1.00 | 9.5e-07 |
| mand_cbct_pddca_ct | 12 | +0.0471 | 1.00 | 0.00049 |
| pelvis_ct_mri | 15 | +0.0526 | 1.00 | 6.1e-05 |
| pancreas_t1wce_t2w | 36 | +0.0352 | 0.97 | 5.8e-11 |
| pancreas_t2w_t1wce | 37 | +0.0080 | 0.59 | 0.14 |
pooled over cells: median of cell medians +0.0280, cells with d > 0: 15 / 21, p < 0.05 same sign: 11 +, 1 -

### FFC (only real correct): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.1355 | 1.00 | 7.6e-13 |
| glioma_t1n_t2w | 62 | +0.1258 | 1.00 | 7.6e-12 |
| glioma_t2w_t2f | 58 | +0.0834 | 1.00 | 3.5e-11 |
| glioma_t2w_t1n | 55 | +0.0833 | 0.98 | 1.3e-10 |
| ms_flair_t2w | 7 | -0.0102 | 0.43 | 0.94 |
| ms_flair_t1w | 5 | -0.0739 | 0.20 | 0.19 |
| ms_t1w_flair | 8 | +0.0446 | 0.50 | 0.31 |
| ms_t1w_t2w | 8 | +0.1000 | 0.88 | 0.016 |
| breast_t1wce_t2w | 5 | +0.0633 | 0.80 | 0.12 |
| breast_t2w_t1wce | 14 | -0.0120 | 0.29 | 0.1 |
| brain_t1w_t2w | 8 | +0.0513 | 1.00 | 0.0078 |
| brain_t2w_t1w | 7 | +0.0252 | 1.00 | 0.016 |
| abd_t1in_t2spir | 3 | +0.0851 | 1.00 | nan |
| abd_t2spir_t1in | 3 | +0.0249 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0214 | 1.00 | nan |
| abd_t2spir_ct | 4 | +0.0495 | 0.75 | nan |
| mand_cbct_hanseg_ct | 21 | -0.0168 | 0.05 | 8.4e-05 |
| mand_cbct_pddca_ct | 12 | -0.0078 | 0.25 | 0.042 |
| pelvis_ct_mri | 15 | +0.1259 | 1.00 | 6.1e-05 |
| pancreas_t1wce_t2w | 36 | -0.0011 | 0.39 | 0.35 |
| pancreas_t2w_t1wce | 37 | +0.0157 | 0.78 | 4.4e-05 |
pooled over cells: median of cell medians +0.0446, cells with d > 0: 15 / 21, p < 0.05 same sign: 9 +, 2 -

### FFC (only real correct): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.0740 | 0.99 | 8e-13 |
| glioma_t1n_t2w | 62 | +0.0698 | 0.97 | 8.8e-12 |
| glioma_t2w_t2f | 58 | +0.0444 | 0.97 | 6.2e-11 |
| glioma_t2w_t1n | 55 | +0.0443 | 0.98 | 1.2e-10 |
| ms_flair_t2w | 7 | +0.0194 | 0.57 | 0.3 |
| ms_flair_t1w | 5 | -0.0166 | 0.40 | 0.44 |
| ms_t1w_flair | 8 | -0.0581 | 0.12 | 0.016 |
| ms_t1w_t2w | 8 | -0.0150 | 0.38 | 0.74 |
| breast_t1wce_t2w | 5 | +0.0225 | 1.00 | 0.062 |
| breast_t2w_t1wce | 14 | +0.0623 | 0.93 | 0.00024 |
| brain_t1w_t2w | 8 | +0.0149 | 0.75 | 0.039 |
| brain_t2w_t1w | 7 | +0.0131 | 1.00 | 0.016 |
| abd_t1in_t2spir | 3 | +0.0463 | 1.00 | nan |
| abd_t2spir_t1in | 3 | -0.0256 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0078 | 0.25 | nan |
| abd_t2spir_ct | 4 | -0.0299 | 0.25 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0367 | 1.00 | 9.5e-07 |
| mand_cbct_pddca_ct | 12 | +0.0418 | 1.00 | 0.00049 |
| pelvis_ct_mri | 15 | +0.0461 | 0.87 | 0.00085 |
| pancreas_t1wce_t2w | 36 | +0.0339 | 0.94 | 3.2e-09 |
| pancreas_t2w_t1wce | 37 | +0.0056 | 0.57 | 0.62 |
pooled over cells: median of cell medians +0.0225, cells with d > 0: 15 / 21, p < 0.05 same sign: 11 +, 1 -

### FFC (only real correct): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.1709 | 1.00 | 7.6e-13 |
| glioma_t1n_t2w | 62 | +0.1347 | 1.00 | 7.6e-12 |
| glioma_t2w_t2f | 58 | +0.0906 | 1.00 | 3.5e-11 |
| glioma_t2w_t1n | 55 | +0.0994 | 0.96 | 1.3e-10 |
| ms_flair_t2w | 7 | -0.0109 | 0.43 | 1 |
| ms_flair_t1w | 5 | -0.0833 | 0.20 | 0.19 |
| ms_t1w_flair | 8 | +0.0402 | 0.50 | 0.38 |
| ms_t1w_t2w | 8 | +0.1011 | 0.88 | 0.016 |
| breast_t1wce_t2w | 5 | +0.0602 | 0.80 | 0.12 |
| breast_t2w_t1wce | 14 | -0.0192 | 0.21 | 0.03 |
| brain_t1w_t2w | 8 | +0.0402 | 1.00 | 0.0078 |
| brain_t2w_t1w | 7 | +0.0260 | 1.00 | 0.016 |
| abd_t1in_t2spir | 3 | +0.0595 | 1.00 | nan |
| abd_t2spir_t1in | 3 | +0.0199 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0450 | 1.00 | nan |
| abd_t2spir_ct | 4 | +0.0326 | 0.75 | nan |
| mand_cbct_hanseg_ct | 21 | -0.0156 | 0.05 | 0.00013 |
| mand_cbct_pddca_ct | 12 | -0.0072 | 0.25 | 0.077 |
| pelvis_ct_mri | 15 | +0.1244 | 1.00 | 6.1e-05 |
| pancreas_t1wce_t2w | 36 | +0.0014 | 0.53 | 0.9 |
| pancreas_t2w_t1wce | 37 | +0.0195 | 0.76 | 1.6e-05 |
pooled over cells: median of cell medians +0.0402, cells with d > 0: 16 / 21, p < 0.05 same sign: 9 +, 2 -

### FFC (only real correct): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | -0.0009 | 0.49 | 0.9 |
| glioma_t1n_t2w | 62 | +0.0022 | 0.61 | 0.029 |
| glioma_t2w_t2f | 58 | -0.0032 | 0.38 | 0.02 |
| glioma_t2w_t1n | 55 | +0.0115 | 0.73 | 6.6e-06 |
| ms_flair_t2w | 7 | -0.0335 | 0.14 | 0.078 |
| ms_flair_t1w | 5 | -0.0170 | 0.40 | 0.31 |
| ms_t1w_flair | 8 | -0.0166 | 0.25 | 0.64 |
| ms_t1w_t2w | 8 | +0.0067 | 0.62 | 0.64 |
| breast_t1wce_t2w | 5 | +0.0742 | 1.00 | 0.062 |
| breast_t2w_t1wce | 14 | -0.0174 | 0.21 | 0.0085 |
| brain_t1w_t2w | 8 | -0.0065 | 0.25 | 0.38 |
| brain_t2w_t1w | 7 | -0.0183 | 0.00 | 0.016 |
| abd_t1in_t2spir | 3 | +0.0068 | 0.67 | nan |
| abd_t2spir_t1in | 3 | +0.0055 | 0.67 | nan |
| abd_t2spir_t1out | 4 | -0.0226 | 0.00 | nan |
| abd_t2spir_ct | 4 | +0.0132 | 0.50 | nan |
| mand_cbct_hanseg_ct | 21 | +0.0927 | 1.00 | 9.5e-07 |
| mand_cbct_pddca_ct | 12 | +0.1068 | 1.00 | 0.00049 |
| pelvis_ct_mri | 15 | +0.0287 | 0.93 | 0.0067 |
| pancreas_t1wce_t2w | 36 | +0.0042 | 0.69 | 0.024 |
| pancreas_t2w_t1wce | 37 | -0.0237 | 0.11 | 6.5e-09 |
pooled over cells: median of cell medians +0.0022, cells with d > 0: 11 / 21, p < 0.05 same sign: 6 +, 4 -

### FFC (only real correct): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 68 | -0.1092 | 0.00 | 7.6e-13 |
| glioma_t1n_t2w | 62 | -0.1137 | 0.00 | 7.6e-12 |
| glioma_t2w_t2f | 58 | -0.1016 | 0.00 | 3.5e-11 |
| glioma_t2w_t1n | 55 | -0.0940 | 0.00 | 1.1e-10 |
| ms_flair_t2w | 7 | -0.1957 | 0.00 | 0.016 |
| ms_flair_t1w | 5 | -0.0467 | 0.00 | 0.062 |
| ms_t1w_flair | 8 | -0.1725 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.1793 | 0.00 | 0.0078 |
| breast_t1wce_t2w | 5 | -0.2106 | 0.00 | 0.062 |
| breast_t2w_t1wce | 14 | -0.1093 | 0.00 | 0.00012 |
| brain_t1w_t2w | 8 | -0.1136 | 0.00 | 0.0078 |
| brain_t2w_t1w | 7 | -0.1256 | 0.00 | 0.016 |
| abd_t1in_t2spir | 3 | -0.0811 | 0.00 | nan |
| abd_t2spir_t1in | 3 | -0.0404 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.1115 | 0.00 | nan |
| abd_t2spir_ct | 4 | -0.0931 | 0.00 | nan |
| mand_cbct_hanseg_ct | 21 | -0.1897 | 0.00 | 9.5e-07 |
| mand_cbct_pddca_ct | 12 | -0.2992 | 0.00 | 0.00049 |
| pelvis_ct_mri | 15 | -0.0691 | 0.00 | 6.1e-05 |
| pancreas_t1wce_t2w | 36 | -0.0386 | 0.06 | 2.9e-10 |
| pancreas_t2w_t1wce | 37 | -0.0596 | 0.03 | 1.5e-10 |
pooled over cells: median of cell medians -0.1093, cells with d > 0: 0 / 21, p < 0.05 same sign: 0 +, 15 -

### FCF (only flat correct): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.0027 | 0.65 | 0.041 |
| glioma_t1n_t2w | 38 | +0.0032 | 0.63 | 0.016 |
| glioma_t2w_t2f | 28 | -0.0024 | 0.29 | 0.056 |
| glioma_t2w_t1n | 35 | +0.0029 | 0.63 | 0.0034 |
| ms_flair_t2w | 5 | -0.0017 | 0.40 | 0.81 |
| ms_flair_t1w | 5 | +0.0010 | 0.60 | 1 |
| ms_t1w_flair | 4 | +0.0151 | 0.75 | nan |
| ms_t1w_t2w | 3 | +0.0061 | 1.00 | nan |
| breast_t1wce_t2w | 11 | -0.0002 | 0.45 | 0.32 |
| breast_t2w_t1wce | 12 | +0.0010 | 0.67 | 0.092 |
| brain_t1w_t2w | 8 | +0.0020 | 0.75 | 0.11 |
| brain_t2w_t1w | 8 | -0.0000 | 0.50 | 0.84 |
| abd_t1in_t2spir | 4 | -0.0023 | 0.25 | nan |
| abd_t1in_ct | 4 | -0.0082 | 0.00 | nan |
| abd_t2spir_t1in | 4 | +0.0024 | 0.75 | nan |
| abd_t2spir_t1out | 4 | -0.0031 | 0.25 | nan |
| mand_cbct_hanseg_ct | 9 | +0.0111 | 0.78 | 0.02 |
| mand_cbct_pddca_ct | 8 | +0.0102 | 1.00 | 0.0078 |
| pelvis_mri_ct | 19 | -0.0018 | 0.16 | 0.00052 |
| pelvis_ct_mri | 11 | -0.0013 | 0.36 | 0.7 |
| pancreas_t1wce_t2w | 38 | +0.0009 | 0.68 | 0.028 |
| pancreas_t2w_t1wce | 38 | +0.0011 | 0.63 | 0.13 |
pooled over cells: median of cell medians +0.0010, cells with d > 0: 13 / 22, p < 0.05 same sign: 6 +, 1 -

### FCF (only flat correct): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.0014 | 0.54 | 0.32 |
| glioma_t1n_t2w | 38 | +0.0022 | 0.61 | 0.012 |
| glioma_t2w_t2f | 28 | -0.0000 | 0.50 | 0.97 |
| glioma_t2w_t1n | 35 | -0.0035 | 0.37 | 0.093 |
| ms_flair_t2w | 5 | -0.0146 | 0.20 | 0.62 |
| ms_flair_t1w | 5 | -0.0022 | 0.20 | 0.62 |
| ms_t1w_flair | 4 | -0.0056 | 0.50 | nan |
| ms_t1w_t2w | 3 | -0.0056 | 0.00 | nan |
| breast_t1wce_t2w | 11 | -0.0095 | 0.00 | 0.00098 |
| breast_t2w_t1wce | 12 | -0.0051 | 0.25 | 0.016 |
| brain_t1w_t2w | 8 | -0.0103 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.0058 | 0.12 | 0.016 |
| abd_t1in_t2spir | 4 | -0.0095 | 0.00 | nan |
| abd_t1in_ct | 4 | -0.0166 | 0.00 | nan |
| abd_t2spir_t1in | 4 | -0.0015 | 0.50 | nan |
| abd_t2spir_t1out | 4 | -0.0036 | 0.25 | nan |
| mand_cbct_hanseg_ct | 9 | -0.0298 | 0.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | -0.0150 | 0.12 | 0.023 |
| pelvis_mri_ct | 19 | -0.0047 | 0.00 | 3.8e-06 |
| pelvis_ct_mri | 11 | +0.0005 | 0.55 | 0.97 |
| pancreas_t1wce_t2w | 38 | +0.0002 | 0.53 | 0.66 |
| pancreas_t2w_t1wce | 38 | -0.0002 | 0.45 | 0.7 |
pooled over cells: median of cell medians -0.0049, cells with d > 0: 4 / 22, p < 0.05 same sign: 1 +, 7 -

### FCF (only flat correct): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.0158 | 0.88 | 9.1e-06 |
| glioma_t1n_t2w | 38 | +0.0051 | 0.61 | 0.02 |
| glioma_t2w_t2f | 28 | +0.0018 | 0.54 | 0.96 |
| glioma_t2w_t1n | 35 | +0.0138 | 0.77 | 0.00063 |
| ms_flair_t2w | 5 | -0.0110 | 0.40 | 0.31 |
| ms_flair_t1w | 5 | +0.0061 | 0.80 | 0.44 |
| ms_t1w_flair | 4 | +0.0038 | 0.75 | nan |
| ms_t1w_t2w | 3 | +0.0119 | 0.67 | nan |
| breast_t1wce_t2w | 11 | +0.0110 | 0.91 | 0.0098 |
| breast_t2w_t1wce | 12 | +0.0000 | 0.50 | 0.85 |
| brain_t1w_t2w | 8 | -0.0431 | 0.38 | 0.64 |
| brain_t2w_t1w | 8 | -0.0071 | 0.50 | 0.84 |
| abd_t1in_t2spir | 4 | -0.0047 | 0.25 | nan |
| abd_t1in_ct | 4 | +0.0004 | 0.75 | nan |
| abd_t2spir_t1in | 4 | +0.0266 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0113 | 0.75 | nan |
| mand_cbct_hanseg_ct | 9 | +0.0496 | 1.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | +0.0597 | 1.00 | 0.0078 |
| pelvis_mri_ct | 19 | +0.0011 | 0.53 | 0.65 |
| pelvis_ct_mri | 11 | -0.0030 | 0.36 | 0.7 |
| pancreas_t1wce_t2w | 38 | +0.0111 | 0.89 | 1.4e-07 |
| pancreas_t2w_t1wce | 38 | +0.0014 | 0.55 | 0.44 |
pooled over cells: median of cell medians +0.0045, cells with d > 0: 17 / 22, p < 0.05 same sign: 7 +, 0 -

### FCF (only flat correct): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.0082 | 0.73 | 0.011 |
| glioma_t1n_t2w | 38 | +0.0105 | 0.71 | 0.00064 |
| glioma_t2w_t2f | 28 | -0.0087 | 0.32 | 0.2 |
| glioma_t2w_t1n | 35 | -0.0020 | 0.46 | 0.62 |
| ms_flair_t2w | 5 | -0.0347 | 0.00 | 0.062 |
| ms_flair_t1w | 5 | +0.0090 | 0.80 | 0.19 |
| ms_t1w_flair | 4 | -0.0267 | 0.00 | nan |
| ms_t1w_t2w | 3 | -0.0269 | 0.00 | nan |
| breast_t1wce_t2w | 11 | -0.0220 | 0.09 | 0.002 |
| breast_t2w_t1wce | 12 | -0.0488 | 0.00 | 0.00049 |
| brain_t1w_t2w | 8 | -0.0007 | 0.38 | 0.55 |
| brain_t2w_t1w | 8 | -0.0089 | 0.12 | 0.2 |
| abd_t1in_t2spir | 4 | -0.0095 | 0.25 | nan |
| abd_t1in_ct | 4 | +0.0523 | 1.00 | nan |
| abd_t2spir_t1in | 4 | +0.0623 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0338 | 0.75 | nan |
| mand_cbct_hanseg_ct | 9 | -0.1325 | 0.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | -0.0924 | 0.00 | 0.0078 |
| pelvis_mri_ct | 19 | -0.0309 | 0.00 | 3.8e-06 |
| pelvis_ct_mri | 11 | +0.0073 | 0.55 | 0.83 |
| pancreas_t1wce_t2w | 38 | -0.0069 | 0.24 | 0.00016 |
| pancreas_t2w_t1wce | 38 | -0.0009 | 0.45 | 0.4 |
pooled over cells: median of cell medians -0.0078, cells with d > 0: 7 / 22, p < 0.05 same sign: 2 +, 6 -

### FCF (only flat correct): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.0567 | 0.96 | 6e-08 |
| glioma_t1n_t2w | 38 | +0.0628 | 1.00 | 7.3e-12 |
| glioma_t2w_t2f | 28 | +0.0474 | 0.96 | 1.5e-08 |
| glioma_t2w_t1n | 35 | +0.0516 | 1.00 | 5.8e-11 |
| ms_flair_t2w | 5 | +0.0862 | 1.00 | 0.062 |
| ms_flair_t1w | 5 | +0.0607 | 1.00 | 0.062 |
| ms_t1w_flair | 4 | +0.1218 | 1.00 | nan |
| ms_t1w_t2w | 3 | +0.0021 | 0.67 | nan |
| breast_t1wce_t2w | 11 | +0.0703 | 0.73 | 0.024 |
| breast_t2w_t1wce | 12 | +0.0892 | 1.00 | 0.00049 |
| brain_t1w_t2w | 8 | +0.0095 | 0.88 | 0.016 |
| brain_t2w_t1w | 8 | +0.0112 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0381 | 1.00 | nan |
| abd_t1in_ct | 4 | +0.0652 | 1.00 | nan |
| abd_t2spir_t1in | 4 | -0.0105 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0175 | 0.00 | nan |
| mand_cbct_hanseg_ct | 9 | +0.0253 | 1.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | +0.0419 | 1.00 | 0.0078 |
| pelvis_mri_ct | 19 | +0.0253 | 1.00 | 3.8e-06 |
| pelvis_ct_mri | 11 | +0.0277 | 0.82 | 0.054 |
| pancreas_t1wce_t2w | 38 | +0.0401 | 1.00 | 7.3e-12 |
| pancreas_t2w_t1wce | 38 | +0.0284 | 0.92 | 3.3e-09 |
pooled over cells: median of cell medians +0.0410, cells with d > 0: 20 / 22, p < 0.05 same sign: 13 +, 0 -

### FCF (only flat correct): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.1242 | 1.00 | 3e-08 |
| glioma_t1n_t2w | 38 | +0.1077 | 1.00 | 7.3e-12 |
| glioma_t2w_t2f | 28 | +0.0714 | 1.00 | 7.5e-09 |
| glioma_t2w_t1n | 35 | +0.0667 | 0.94 | 2.9e-10 |
| ms_flair_t2w | 5 | +0.0039 | 0.60 | 1 |
| ms_flair_t1w | 5 | +0.0201 | 0.80 | 0.12 |
| ms_t1w_flair | 4 | +0.0801 | 1.00 | nan |
| ms_t1w_t2w | 3 | +0.0710 | 1.00 | nan |
| breast_t1wce_t2w | 11 | +0.0697 | 0.91 | 0.002 |
| breast_t2w_t1wce | 12 | +0.0080 | 0.58 | 0.73 |
| brain_t1w_t2w | 8 | +0.0185 | 0.88 | 0.016 |
| brain_t2w_t1w | 8 | +0.0331 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0795 | 1.00 | nan |
| abd_t1in_ct | 4 | +0.1659 | 1.00 | nan |
| abd_t2spir_t1in | 4 | +0.0303 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0104 | 1.00 | nan |
| mand_cbct_hanseg_ct | 9 | -0.0160 | 0.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | -0.0246 | 0.00 | 0.0078 |
| pelvis_mri_ct | 19 | +0.1185 | 0.95 | 7.6e-06 |
| pelvis_ct_mri | 11 | +0.0984 | 1.00 | 0.00098 |
| pancreas_t1wce_t2w | 38 | -0.0024 | 0.42 | 0.5 |
| pancreas_t2w_t1wce | 38 | +0.0261 | 0.79 | 2.6e-07 |
pooled over cells: median of cell medians +0.0499, cells with d > 0: 19 / 22, p < 0.05 same sign: 10 +, 2 -

### FCF (only flat correct): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.0522 | 0.96 | 2.1e-06 |
| glioma_t1n_t2w | 38 | +0.0557 | 0.95 | 3.6e-11 |
| glioma_t2w_t2f | 28 | +0.0360 | 0.96 | 1.5e-08 |
| glioma_t2w_t1n | 35 | +0.0454 | 0.97 | 1.2e-10 |
| ms_flair_t2w | 5 | +0.0681 | 1.00 | 0.062 |
| ms_flair_t1w | 5 | +0.0555 | 1.00 | 0.062 |
| ms_t1w_flair | 4 | +0.1285 | 0.75 | nan |
| ms_t1w_t2w | 3 | +0.0212 | 0.67 | nan |
| breast_t1wce_t2w | 11 | +0.0741 | 0.73 | 0.014 |
| breast_t2w_t1wce | 12 | +0.0888 | 1.00 | 0.00049 |
| brain_t1w_t2w | 8 | +0.0025 | 0.62 | 0.31 |
| brain_t2w_t1w | 8 | +0.0081 | 0.88 | 0.016 |
| abd_t1in_t2spir | 4 | +0.0393 | 1.00 | nan |
| abd_t1in_ct | 4 | +0.0593 | 0.75 | nan |
| abd_t2spir_t1in | 4 | -0.0145 | 0.25 | nan |
| abd_t2spir_t1out | 4 | -0.0090 | 0.25 | nan |
| mand_cbct_hanseg_ct | 9 | +0.0238 | 1.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | +0.0395 | 1.00 | 0.0078 |
| pelvis_mri_ct | 19 | +0.0617 | 1.00 | 3.8e-06 |
| pelvis_ct_mri | 11 | +0.0327 | 0.73 | 0.083 |
| pancreas_t1wce_t2w | 38 | +0.0340 | 1.00 | 7.3e-12 |
| pancreas_t2w_t1wce | 38 | +0.0303 | 0.89 | 8.3e-08 |
pooled over cells: median of cell medians +0.0394, cells with d > 0: 20 / 22, p < 0.05 same sign: 12 +, 0 -

### FCF (only flat correct): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.1367 | 1.00 | 3e-08 |
| glioma_t1n_t2w | 38 | +0.1162 | 1.00 | 7.3e-12 |
| glioma_t2w_t2f | 28 | +0.0727 | 1.00 | 7.5e-09 |
| glioma_t2w_t1n | 35 | +0.0756 | 0.97 | 1.1e-09 |
| ms_flair_t2w | 5 | +0.0098 | 0.80 | 0.62 |
| ms_flair_t1w | 5 | +0.0122 | 0.80 | 0.31 |
| ms_t1w_flair | 4 | +0.0801 | 1.00 | nan |
| ms_t1w_t2w | 3 | +0.0807 | 1.00 | nan |
| breast_t1wce_t2w | 11 | +0.0375 | 0.91 | 0.0068 |
| breast_t2w_t1wce | 12 | +0.0197 | 0.67 | 0.3 |
| brain_t1w_t2w | 8 | +0.0155 | 0.88 | 0.055 |
| brain_t2w_t1w | 8 | +0.0344 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0771 | 1.00 | nan |
| abd_t1in_ct | 4 | +0.1605 | 1.00 | nan |
| abd_t2spir_t1in | 4 | +0.0247 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0193 | 1.00 | nan |
| mand_cbct_hanseg_ct | 9 | -0.0159 | 0.11 | 0.0078 |
| mand_cbct_pddca_ct | 8 | -0.0186 | 0.00 | 0.0078 |
| pelvis_mri_ct | 19 | +0.0635 | 0.84 | 0.0039 |
| pelvis_ct_mri | 11 | +0.1029 | 0.91 | 0.002 |
| pancreas_t1wce_t2w | 38 | +0.0003 | 0.53 | 0.37 |
| pancreas_t2w_t1wce | 38 | +0.0287 | 0.92 | 2.7e-08 |
pooled over cells: median of cell medians +0.0360, cells with d > 0: 20 / 22, p < 0.05 same sign: 9 +, 2 -

### FCF (only flat correct): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | -0.0068 | 0.38 | 0.099 |
| glioma_t1n_t2w | 38 | -0.0061 | 0.39 | 0.11 |
| glioma_t2w_t2f | 28 | +0.0000 | 0.50 | 0.9 |
| glioma_t2w_t1n | 35 | +0.0024 | 0.54 | 0.75 |
| ms_flair_t2w | 5 | -0.0315 | 0.00 | 0.062 |
| ms_flair_t1w | 5 | -0.0175 | 0.00 | 0.062 |
| ms_t1w_flair | 4 | -0.0329 | 0.00 | nan |
| ms_t1w_t2w | 3 | -0.0026 | 0.33 | nan |
| breast_t1wce_t2w | 11 | +0.0691 | 1.00 | 0.00098 |
| breast_t2w_t1wce | 12 | -0.0274 | 0.17 | 0.0034 |
| brain_t1w_t2w | 8 | +0.0016 | 0.50 | 0.74 |
| brain_t2w_t1w | 8 | -0.0171 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0085 | 0.25 | nan |
| abd_t1in_ct | 4 | -0.0028 | 0.25 | nan |
| abd_t2spir_t1in | 4 | -0.0076 | 0.25 | nan |
| abd_t2spir_t1out | 4 | -0.0126 | 0.25 | nan |
| mand_cbct_hanseg_ct | 9 | +0.0930 | 1.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | +0.1017 | 1.00 | 0.0078 |
| pelvis_mri_ct | 19 | -0.0228 | 0.16 | 0.002 |
| pelvis_ct_mri | 11 | +0.0316 | 0.82 | 0.014 |
| pancreas_t1wce_t2w | 38 | +0.0064 | 0.68 | 0.0057 |
| pancreas_t2w_t1wce | 38 | -0.0163 | 0.08 | 3.9e-09 |
pooled over cells: median of cell medians -0.0064, cells with d > 0: 8 / 22, p < 0.05 same sign: 5 +, 4 -

### FCF (only flat correct): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 26 | -0.0994 | 0.00 | 3e-08 |
| glioma_t1n_t2w | 38 | -0.1117 | 0.00 | 7.3e-12 |
| glioma_t2w_t2f | 28 | -0.1005 | 0.00 | 7.5e-09 |
| glioma_t2w_t1n | 35 | -0.0897 | 0.00 | 5.8e-11 |
| ms_flair_t2w | 5 | -0.2186 | 0.00 | 0.062 |
| ms_flair_t1w | 5 | -0.0702 | 0.00 | 0.062 |
| ms_t1w_flair | 4 | -0.1990 | 0.00 | nan |
| ms_t1w_t2w | 3 | -0.2100 | 0.00 | nan |
| breast_t1wce_t2w | 11 | -0.2622 | 0.00 | 0.00098 |
| breast_t2w_t1wce | 12 | -0.1388 | 0.00 | 0.00049 |
| brain_t1w_t2w | 8 | -0.1061 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1407 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0783 | 0.00 | nan |
| abd_t1in_ct | 4 | -0.0709 | 0.00 | nan |
| abd_t2spir_t1in | 4 | -0.0736 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0898 | 0.00 | nan |
| mand_cbct_hanseg_ct | 9 | -0.1954 | 0.00 | 0.0039 |
| mand_cbct_pddca_ct | 8 | -0.2595 | 0.00 | 0.0078 |
| pelvis_mri_ct | 19 | -0.1219 | 0.00 | 3.8e-06 |
| pelvis_ct_mri | 11 | -0.0686 | 0.00 | 0.00098 |
| pancreas_t1wce_t2w | 38 | -0.0361 | 0.05 | 3.1e-08 |
| pancreas_t2w_t1wce | 38 | -0.0551 | 0.00 | 7.3e-12 |
pooled over cells: median of cell medians -0.1033, cells with d > 0: 0 / 22, p < 0.05 same sign: 0 +, 14 -

### CFF (only noise correct): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.0032 | 0.89 | 0.00033 |
| glioma_t1n_t2w | 36 | +0.0029 | 0.81 | 0.00023 |
| glioma_t2w_t2f | 34 | -0.0005 | 0.47 | 0.42 |
| glioma_t2w_t1n | 62 | +0.0033 | 0.82 | 8.7e-08 |
| ms_flair_t2w | 7 | -0.0042 | 0.29 | 0.22 |
| ms_flair_t1w | 5 | +0.0001 | 0.60 | 0.81 |
| ms_t1w_flair | 8 | +0.0034 | 0.75 | 0.15 |
| ms_t1w_t2w | 8 | +0.0050 | 0.75 | 0.31 |
| breast_t1wce_t2w | 8 | +0.0001 | 0.50 | 0.84 |
| breast_t2w_t1wce | 13 | +0.0006 | 0.62 | 0.74 |
| brain_t1w_t2w | 8 | -0.0010 | 0.38 | 0.95 |
| brain_t2w_t1w | 8 | +0.0014 | 0.62 | 0.64 |
| abd_t1in_t2spir | 4 | -0.0070 | 0.00 | nan |
| abd_t2spir_t1in | 4 | +0.0014 | 0.75 | nan |
| abd_t2spir_t1out | 3 | +0.0001 | 0.67 | nan |
| abd_t2spir_ct | 6 | +0.0010 | 0.67 | 0.56 |
| mand_cbct_hanseg_ct | 19 | +0.0010 | 0.63 | 0.17 |
| mand_cbct_pddca_ct | 22 | +0.0028 | 0.64 | 0.074 |
| pelvis_mri_ct | 5 | -0.0038 | 0.20 | 0.12 |
| pelvis_ct_mri | 6 | +0.0004 | 0.50 | 0.69 |
| pancreas_t1wce_t2w | 37 | +0.0006 | 0.65 | 0.022 |
| pancreas_t2w_t1wce | 40 | +0.0001 | 0.60 | 0.54 |
pooled over cells: median of cell medians +0.0006, cells with d > 0: 17 / 22, p < 0.05 same sign: 4 +, 0 -

### CFF (only noise correct): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | -0.0004 | 0.50 | 0.77 |
| glioma_t1n_t2w | 36 | +0.0011 | 0.64 | 0.02 |
| glioma_t2w_t2f | 34 | -0.0096 | 0.12 | 1.2e-07 |
| glioma_t2w_t1n | 62 | -0.0057 | 0.08 | 5.8e-09 |
| ms_flair_t2w | 7 | -0.0230 | 0.14 | 0.031 |
| ms_flair_t1w | 5 | -0.0039 | 0.20 | 0.31 |
| ms_t1w_flair | 8 | +0.0012 | 0.62 | 0.74 |
| ms_t1w_t2w | 8 | +0.0034 | 0.62 | 0.84 |
| breast_t1wce_t2w | 8 | -0.0210 | 0.12 | 0.016 |
| breast_t2w_t1wce | 13 | -0.0151 | 0.00 | 0.00024 |
| brain_t1w_t2w | 8 | -0.0058 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.0079 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0152 | 0.25 | nan |
| abd_t2spir_t1in | 4 | -0.0050 | 0.25 | nan |
| abd_t2spir_t1out | 3 | +0.0069 | 0.67 | nan |
| abd_t2spir_ct | 6 | +0.0027 | 0.83 | 0.44 |
| mand_cbct_hanseg_ct | 19 | -0.0127 | 0.32 | 0.011 |
| mand_cbct_pddca_ct | 22 | -0.0178 | 0.05 | 1.2e-05 |
| pelvis_mri_ct | 5 | +0.0030 | 0.60 | 0.81 |
| pelvis_ct_mri | 6 | -0.0053 | 0.33 | 0.56 |
| pancreas_t1wce_t2w | 37 | -0.0003 | 0.46 | 0.32 |
| pancreas_t2w_t1wce | 40 | +0.0013 | 0.70 | 0.0012 |
pooled over cells: median of cell medians -0.0052, cells with d > 0: 7 / 22, p < 0.05 same sign: 2 +, 9 -

### CFF (only noise correct): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.0303 | 0.83 | 0.00011 |
| glioma_t1n_t2w | 36 | -0.0016 | 0.47 | 0.68 |
| glioma_t2w_t2f | 34 | +0.0042 | 0.59 | 0.16 |
| glioma_t2w_t1n | 62 | +0.0191 | 0.84 | 1.4e-09 |
| ms_flair_t2w | 7 | -0.0048 | 0.29 | 0.38 |
| ms_flair_t1w | 5 | +0.0083 | 0.60 | 0.31 |
| ms_t1w_flair | 8 | -0.0003 | 0.50 | 0.95 |
| ms_t1w_t2w | 8 | +0.0026 | 0.75 | 0.38 |
| breast_t1wce_t2w | 8 | +0.0190 | 0.88 | 0.039 |
| breast_t2w_t1wce | 13 | -0.0004 | 0.46 | 0.79 |
| brain_t1w_t2w | 8 | -0.0058 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0003 | 0.62 | 0.95 |
| abd_t1in_t2spir | 4 | +0.0131 | 0.75 | nan |
| abd_t2spir_t1in | 4 | +0.0375 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0311 | 1.00 | nan |
| abd_t2spir_ct | 6 | -0.0167 | 0.33 | 0.31 |
| mand_cbct_hanseg_ct | 19 | +0.0371 | 0.89 | 1.9e-05 |
| mand_cbct_pddca_ct | 22 | +0.0415 | 0.91 | 0.00026 |
| pelvis_mri_ct | 5 | -0.0080 | 0.20 | 0.31 |
| pelvis_ct_mri | 6 | +0.0052 | 0.67 | 1 |
| pancreas_t1wce_t2w | 37 | +0.0056 | 0.73 | 0.00056 |
| pancreas_t2w_t1wce | 40 | -0.0022 | 0.45 | 0.36 |
pooled over cells: median of cell medians +0.0047, cells with d > 0: 14 / 22, p < 0.05 same sign: 6 +, 1 -

### CFF (only noise correct): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.0201 | 0.72 | 0.0028 |
| glioma_t1n_t2w | 36 | +0.0136 | 0.69 | 0.0011 |
| glioma_t2w_t2f | 34 | -0.0137 | 0.18 | 7.4e-05 |
| glioma_t2w_t1n | 62 | -0.0017 | 0.47 | 0.5 |
| ms_flair_t2w | 7 | -0.0359 | 0.00 | 0.016 |
| ms_flair_t1w | 5 | +0.0033 | 0.60 | 0.31 |
| ms_t1w_flair | 8 | -0.0040 | 0.25 | 0.25 |
| ms_t1w_t2w | 8 | -0.0077 | 0.00 | 0.0078 |
| breast_t1wce_t2w | 8 | -0.0403 | 0.12 | 0.039 |
| breast_t2w_t1wce | 13 | -0.0411 | 0.00 | 0.00024 |
| brain_t1w_t2w | 8 | +0.1305 | 0.88 | 0.016 |
| brain_t2w_t1w | 8 | -0.1312 | 0.25 | 0.078 |
| abd_t1in_t2spir | 4 | -0.0010 | 0.50 | nan |
| abd_t2spir_t1in | 4 | +0.0656 | 0.75 | nan |
| abd_t2spir_t1out | 3 | +0.0319 | 0.67 | nan |
| abd_t2spir_ct | 6 | +0.0771 | 1.00 | 0.031 |
| mand_cbct_hanseg_ct | 19 | -0.0840 | 0.05 | 1.9e-05 |
| mand_cbct_pddca_ct | 22 | -0.1405 | 0.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | +0.0788 | 0.80 | 0.62 |
| pelvis_ct_mri | 6 | -0.0163 | 0.00 | 0.031 |
| pancreas_t1wce_t2w | 37 | -0.0045 | 0.32 | 0.083 |
| pancreas_t2w_t1wce | 40 | +0.0025 | 0.55 | 0.52 |
pooled over cells: median of cell medians -0.0028, cells with d > 0: 9 / 22, p < 0.05 same sign: 4 +, 8 -

### CFF (only noise correct): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.0765 | 1.00 | 7.6e-06 |
| glioma_t1n_t2w | 36 | +0.0650 | 0.97 | 1.5e-10 |
| glioma_t2w_t2f | 34 | +0.0454 | 0.94 | 1.6e-09 |
| glioma_t2w_t1n | 62 | +0.0565 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 7 | +0.0718 | 1.00 | 0.016 |
| ms_flair_t1w | 5 | -0.0215 | 0.40 | 0.81 |
| ms_t1w_flair | 8 | +0.0287 | 0.75 | 0.74 |
| ms_t1w_t2w | 8 | -0.0171 | 0.50 | 0.64 |
| breast_t1wce_t2w | 8 | +0.0341 | 0.88 | 0.15 |
| breast_t2w_t1wce | 13 | +0.0908 | 1.00 | 0.00024 |
| brain_t1w_t2w | 8 | +0.0208 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0200 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0469 | 1.00 | nan |
| abd_t2spir_t1in | 4 | -0.0204 | 0.00 | nan |
| abd_t2spir_t1out | 3 | -0.0105 | 0.33 | nan |
| abd_t2spir_ct | 6 | -0.0531 | 0.00 | 0.031 |
| mand_cbct_hanseg_ct | 19 | +0.0452 | 1.00 | 3.8e-06 |
| mand_cbct_pddca_ct | 22 | +0.0266 | 1.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | +0.0216 | 1.00 | 0.062 |
| pelvis_ct_mri | 6 | +0.0360 | 1.00 | 0.031 |
| pancreas_t1wce_t2w | 37 | +0.0278 | 0.95 | 3.6e-10 |
| pancreas_t2w_t1wce | 40 | +0.0275 | 0.88 | 3.2e-09 |
pooled over cells: median of cell medians +0.0283, cells with d > 0: 17 / 22, p < 0.05 same sign: 13 +, 1 -

### CFF (only noise correct): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.1053 | 1.00 | 7.6e-06 |
| glioma_t1n_t2w | 36 | +0.1149 | 1.00 | 2.9e-11 |
| glioma_t2w_t2f | 34 | +0.0714 | 1.00 | 1.2e-10 |
| glioma_t2w_t1n | 62 | +0.0791 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 7 | +0.0059 | 0.71 | 0.47 |
| ms_flair_t1w | 5 | -0.0024 | 0.40 | 0.81 |
| ms_t1w_flair | 8 | +0.1087 | 0.88 | 0.039 |
| ms_t1w_t2w | 8 | +0.0452 | 0.75 | 0.11 |
| breast_t1wce_t2w | 8 | +0.0972 | 0.88 | 0.016 |
| breast_t2w_t1wce | 13 | -0.0179 | 0.38 | 0.95 |
| brain_t1w_t2w | 8 | +0.0647 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0535 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0994 | 1.00 | nan |
| abd_t2spir_t1in | 4 | +0.0488 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0399 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.0110 | 0.67 | 0.31 |
| mand_cbct_hanseg_ct | 19 | -0.0234 | 0.05 | 2.7e-05 |
| mand_cbct_pddca_ct | 22 | -0.0232 | 0.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | +0.0881 | 1.00 | 0.062 |
| pelvis_ct_mri | 6 | +0.1734 | 1.00 | 0.031 |
| pancreas_t1wce_t2w | 37 | -0.0126 | 0.14 | 1.2e-05 |
| pancreas_t2w_t1wce | 40 | +0.0290 | 1.00 | 1.8e-12 |
pooled over cells: median of cell medians +0.0512, cells with d > 0: 17 / 22, p < 0.05 same sign: 10 +, 3 -

### CFF (only noise correct): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.0745 | 1.00 | 7.6e-06 |
| glioma_t1n_t2w | 36 | +0.0603 | 0.94 | 4.1e-10 |
| glioma_t2w_t2f | 34 | +0.0331 | 0.91 | 2.4e-08 |
| glioma_t2w_t1n | 62 | +0.0511 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 7 | +0.0724 | 1.00 | 0.016 |
| ms_flair_t1w | 5 | -0.0352 | 0.40 | 1 |
| ms_t1w_flair | 8 | +0.0309 | 0.75 | 0.74 |
| ms_t1w_t2w | 8 | -0.0088 | 0.50 | 0.84 |
| breast_t1wce_t2w | 8 | +0.0275 | 0.75 | 0.38 |
| breast_t2w_t1wce | 13 | +0.0671 | 0.92 | 0.00049 |
| brain_t1w_t2w | 8 | +0.0188 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0167 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0301 | 1.00 | nan |
| abd_t2spir_t1in | 4 | -0.0038 | 0.25 | nan |
| abd_t2spir_t1out | 3 | -0.0049 | 0.33 | nan |
| abd_t2spir_ct | 6 | -0.0527 | 0.00 | 0.031 |
| mand_cbct_hanseg_ct | 19 | +0.0397 | 1.00 | 3.8e-06 |
| mand_cbct_pddca_ct | 22 | +0.0241 | 1.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | +0.0229 | 1.00 | 0.062 |
| pelvis_ct_mri | 6 | +0.0408 | 0.83 | 0.062 |
| pancreas_t1wce_t2w | 37 | +0.0206 | 0.84 | 1.6e-06 |
| pancreas_t2w_t1wce | 40 | +0.0208 | 0.78 | 1.8e-07 |
pooled over cells: median of cell medians +0.0258, cells with d > 0: 17 / 22, p < 0.05 same sign: 12 +, 1 -

### CFF (only noise correct): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | +0.1193 | 1.00 | 7.6e-06 |
| glioma_t1n_t2w | 36 | +0.1207 | 1.00 | 2.9e-11 |
| glioma_t2w_t2f | 34 | +0.0730 | 1.00 | 1.2e-10 |
| glioma_t2w_t1n | 62 | +0.0874 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 7 | +0.0081 | 0.71 | 0.22 |
| ms_flair_t1w | 5 | -0.0022 | 0.40 | 1 |
| ms_t1w_flair | 8 | +0.1236 | 0.88 | 0.039 |
| ms_t1w_t2w | 8 | +0.0316 | 0.75 | 0.25 |
| breast_t1wce_t2w | 8 | +0.0830 | 0.75 | 0.055 |
| breast_t2w_t1wce | 13 | +0.0004 | 0.54 | 0.74 |
| brain_t1w_t2w | 8 | +0.0633 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0599 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0223 | 1.00 | nan |
| abd_t2spir_t1in | 4 | +0.0665 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0559 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.0034 | 0.50 | 1 |
| mand_cbct_hanseg_ct | 19 | -0.0168 | 0.11 | 9.5e-05 |
| mand_cbct_pddca_ct | 22 | -0.0225 | 0.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | +0.0396 | 0.80 | 0.19 |
| pelvis_ct_mri | 6 | +0.1707 | 1.00 | 0.031 |
| pancreas_t1wce_t2w | 37 | -0.0098 | 0.19 | 0.00068 |
| pancreas_t2w_t1wce | 40 | +0.0386 | 1.00 | 1.8e-12 |
pooled over cells: median of cell medians +0.0478, cells with d > 0: 18 / 22, p < 0.05 same sign: 9 +, 3 -

### CFF (only noise correct): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | -0.0087 | 0.28 | 0.048 |
| glioma_t1n_t2w | 36 | +0.0028 | 0.53 | 0.68 |
| glioma_t2w_t2f | 34 | +0.0065 | 0.68 | 0.075 |
| glioma_t2w_t1n | 62 | +0.0070 | 0.77 | 1.4e-06 |
| ms_flair_t2w | 7 | -0.0593 | 0.00 | 0.016 |
| ms_flair_t1w | 5 | -0.0210 | 0.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0231 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.0013 | 0.25 | 0.55 |
| breast_t1wce_t2w | 8 | +0.0829 | 1.00 | 0.0078 |
| breast_t2w_t1wce | 13 | -0.0268 | 0.15 | 0.027 |
| brain_t1w_t2w | 8 | -0.0048 | 0.25 | 0.15 |
| brain_t2w_t1w | 8 | -0.0227 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0056 | 0.50 | nan |
| abd_t2spir_t1in | 4 | +0.0026 | 0.50 | nan |
| abd_t2spir_t1out | 3 | -0.0156 | 0.33 | nan |
| abd_t2spir_ct | 6 | +0.0038 | 0.50 | 1 |
| mand_cbct_hanseg_ct | 19 | +0.0928 | 0.95 | 7.6e-06 |
| mand_cbct_pddca_ct | 22 | +0.1056 | 1.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | -0.0291 | 0.00 | 0.062 |
| pelvis_ct_mri | 6 | +0.0267 | 0.83 | 0.062 |
| pancreas_t1wce_t2w | 37 | +0.0027 | 0.59 | 0.29 |
| pancreas_t2w_t1wce | 40 | -0.0194 | 0.00 | 1.8e-12 |
pooled over cells: median of cell medians -0.0030, cells with d > 0: 10 / 22, p < 0.05 same sign: 4 +, 6 -

### CFF (only noise correct): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 18 | -0.1253 | 0.00 | 7.6e-06 |
| glioma_t1n_t2w | 36 | -0.1101 | 0.00 | 2.9e-11 |
| glioma_t2w_t2f | 34 | -0.0964 | 0.00 | 1.2e-10 |
| glioma_t2w_t1n | 62 | -0.0993 | 0.00 | 7.6e-12 |
| ms_flair_t2w | 7 | -0.2387 | 0.00 | 0.016 |
| ms_flair_t1w | 5 | -0.0693 | 0.00 | 0.062 |
| ms_t1w_flair | 8 | -0.1738 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.1934 | 0.00 | 0.0078 |
| breast_t1wce_t2w | 8 | -0.2697 | 0.00 | 0.0078 |
| breast_t2w_t1wce | 13 | -0.1739 | 0.00 | 0.00024 |
| brain_t1w_t2w | 8 | -0.1062 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1176 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.1419 | 0.00 | nan |
| abd_t2spir_t1in | 4 | -0.0692 | 0.00 | nan |
| abd_t2spir_t1out | 3 | -0.1082 | 0.00 | nan |
| abd_t2spir_ct | 6 | -0.0708 | 0.00 | 0.031 |
| mand_cbct_hanseg_ct | 19 | -0.1873 | 0.00 | 3.8e-06 |
| mand_cbct_pddca_ct | 22 | -0.2772 | 0.00 | 4.8e-07 |
| pelvis_mri_ct | 5 | -0.1165 | 0.00 | 0.062 |
| pelvis_ct_mri | 6 | -0.0832 | 0.00 | 0.031 |
| pancreas_t1wce_t2w | 37 | -0.0414 | 0.05 | 2e-10 |
| pancreas_t2w_t1wce | 40 | -0.0586 | 0.00 | 1.8e-12 |
pooled over cells: median of cell medians -0.1133, cells with d > 0: 0 / 22, p < 0.05 same sign: 0 +, 17 -

### CCC (all correct): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.0039 | 0.82 | 5.5e-09 |
| glioma_t1n_t2w | 67 | +0.0035 | 0.81 | 3.2e-07 |
| glioma_t2w_t2f | 68 | -0.0004 | 0.43 | 0.48 |
| glioma_t2w_t1n | 62 | +0.0024 | 0.77 | 4.4e-07 |
| ms_flair_t2w | 8 | -0.0062 | 0.38 | 0.64 |
| ms_flair_t1w | 6 | -0.0213 | 0.17 | 0.094 |
| ms_t1w_flair | 8 | +0.0012 | 0.62 | 0.31 |
| ms_t1w_t2w | 8 | +0.0092 | 0.75 | 0.039 |
| breast_t1wce_t2w | 40 | -0.0010 | 0.42 | 0.16 |
| breast_t2w_t1wce | 39 | -0.0014 | 0.33 | 0.0057 |
| brain_t1w_t2w | 8 | -0.0040 | 0.38 | 0.2 |
| brain_t2w_t1w | 8 | -0.0024 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0003 | 0.50 | nan |
| abd_t1in_ct | 20 | +0.0017 | 0.65 | 0.28 |
| abd_t2spir_t1in | 4 | +0.0055 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0034 | 1.00 | nan |
| abd_t2spir_ct | 20 | +0.0005 | 0.60 | 0.73 |
| mand_cbct_hanseg_ct | 40 | -0.0010 | 0.20 | 2e-05 |
| mand_cbct_pddca_ct | 40 | -0.0008 | 0.38 | 0.094 |
| pelvis_mri_ct | 25 | -0.0013 | 0.36 | 0.045 |
| pelvis_ct_mri | 21 | -0.0024 | 0.29 | 0.038 |
| pancreas_t1wce_t2w | 40 | -0.0009 | 0.25 | 0.00048 |
| pancreas_t2w_t1wce | 40 | +0.0002 | 0.65 | 0.21 |
pooled over cells: median of cell medians -0.0004, cells with d > 0: 11 / 23, p < 0.05 same sign: 4 +, 6 -

### CCC (all correct): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.0023 | 0.67 | 0.0013 |
| glioma_t1n_t2w | 67 | +0.0046 | 0.79 | 2.3e-08 |
| glioma_t2w_t2f | 68 | -0.0036 | 0.37 | 0.0018 |
| glioma_t2w_t1n | 62 | -0.0042 | 0.21 | 1.1e-06 |
| ms_flair_t2w | 8 | -0.0334 | 0.00 | 0.0078 |
| ms_flair_t1w | 6 | -0.0055 | 0.33 | 0.31 |
| ms_t1w_flair | 8 | -0.0049 | 0.25 | 0.039 |
| ms_t1w_t2w | 8 | -0.0003 | 0.50 | 0.74 |
| breast_t1wce_t2w | 40 | -0.0267 | 0.03 | 5.5e-12 |
| breast_t2w_t1wce | 39 | -0.0076 | 0.10 | 2.8e-09 |
| brain_t1w_t2w | 8 | -0.0151 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.0058 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0052 | 0.00 | nan |
| abd_t1in_ct | 20 | -0.0015 | 0.40 | 0.28 |
| abd_t2spir_t1in | 4 | +0.0067 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0054 | 1.00 | nan |
| abd_t2spir_ct | 20 | +0.0015 | 0.80 | 0.0032 |
| mand_cbct_hanseg_ct | 40 | -0.0182 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.0162 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | -0.0049 | 0.08 | 8.3e-07 |
| pelvis_ct_mri | 21 | -0.0007 | 0.48 | 0.68 |
| pancreas_t1wce_t2w | 40 | -0.0007 | 0.35 | 0.07 |
| pancreas_t2w_t1wce | 40 | -0.0004 | 0.38 | 0.62 |
pooled over cells: median of cell medians -0.0042, cells with d > 0: 5 / 23, p < 0.05 same sign: 3 +, 11 -

### CCC (all correct): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.0181 | 0.95 | 7.6e-11 |
| glioma_t1n_t2w | 67 | -0.0002 | 0.49 | 0.78 |
| glioma_t2w_t2f | 68 | -0.0000 | 0.50 | 0.26 |
| glioma_t2w_t1n | 62 | +0.0131 | 0.77 | 5.6e-06 |
| ms_flair_t2w | 8 | -0.0053 | 0.38 | 0.64 |
| ms_flair_t1w | 6 | -0.0166 | 0.17 | 0.22 |
| ms_t1w_flair | 8 | +0.0005 | 0.50 | 0.95 |
| ms_t1w_t2w | 8 | +0.0060 | 0.62 | 0.2 |
| breast_t1wce_t2w | 40 | +0.0105 | 0.80 | 3.5e-06 |
| breast_t2w_t1wce | 39 | -0.0053 | 0.33 | 0.016 |
| brain_t1w_t2w | 8 | -0.0110 | 0.12 | 0.15 |
| brain_t2w_t1w | 8 | +0.0013 | 0.62 | 0.84 |
| abd_t1in_t2spir | 4 | +0.0104 | 0.75 | nan |
| abd_t1in_ct | 20 | +0.0198 | 0.80 | 0.0017 |
| abd_t2spir_t1in | 4 | +0.0917 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0512 | 1.00 | nan |
| abd_t2spir_ct | 20 | -0.0261 | 0.15 | 6.3e-05 |
| mand_cbct_hanseg_ct | 40 | -0.0211 | 0.07 | 1.6e-10 |
| mand_cbct_pddca_ct | 40 | -0.0029 | 0.42 | 0.11 |
| pelvis_mri_ct | 25 | -0.0145 | 0.16 | 3.8e-05 |
| pelvis_ct_mri | 21 | -0.0179 | 0.24 | 0.009 |
| pancreas_t1wce_t2w | 40 | +0.0186 | 0.95 | 3.5e-11 |
| pancreas_t2w_t1wce | 40 | +0.0015 | 0.60 | 0.4 |
pooled over cells: median of cell medians +0.0005, cells with d > 0: 12 / 23, p < 0.05 same sign: 5 +, 5 -

### CCC (all correct): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.0185 | 0.82 | 1.2e-07 |
| glioma_t1n_t2w | 67 | +0.0170 | 0.81 | 1.6e-08 |
| glioma_t2w_t2f | 68 | -0.0075 | 0.34 | 0.00028 |
| glioma_t2w_t1n | 62 | +0.0003 | 0.50 | 0.51 |
| ms_flair_t2w | 8 | -0.0476 | 0.00 | 0.0078 |
| ms_flair_t1w | 6 | +0.0076 | 0.67 | 0.31 |
| ms_t1w_flair | 8 | -0.0115 | 0.12 | 0.023 |
| ms_t1w_t2w | 8 | -0.0080 | 0.38 | 0.25 |
| breast_t1wce_t2w | 40 | -0.0388 | 0.07 | 1.6e-09 |
| breast_t2w_t1wce | 39 | -0.0455 | 0.00 | 3.6e-12 |
| brain_t1w_t2w | 8 | -0.0275 | 0.12 | 0.055 |
| brain_t2w_t1w | 8 | -0.0024 | 0.50 | 0.84 |
| abd_t1in_t2spir | 4 | +0.0114 | 0.50 | nan |
| abd_t1in_ct | 20 | +0.1080 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | +0.1258 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0856 | 1.00 | nan |
| abd_t2spir_ct | 20 | +0.1155 | 1.00 | 1.9e-06 |
| mand_cbct_hanseg_ct | 40 | -0.1947 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.1914 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | -0.0133 | 0.28 | 0.048 |
| pelvis_ct_mri | 21 | -0.0204 | 0.29 | 0.076 |
| pancreas_t1wce_t2w | 40 | -0.0023 | 0.42 | 0.7 |
| pancreas_t2w_t1wce | 40 | +0.0021 | 0.57 | 0.28 |
pooled over cells: median of cell medians -0.0024, cells with d > 0: 10 / 23, p < 0.05 same sign: 4 +, 8 -

### CCC (all correct): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.0735 | 1.00 | 1.6e-11 |
| glioma_t1n_t2w | 67 | +0.0671 | 1.00 | 1.1e-12 |
| glioma_t2w_t2f | 68 | +0.0454 | 0.99 | 8.4e-13 |
| glioma_t2w_t1n | 62 | +0.0520 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 8 | +0.0512 | 0.88 | 0.023 |
| ms_flair_t1w | 6 | +0.0250 | 0.50 | 0.44 |
| ms_t1w_flair | 8 | -0.0166 | 0.25 | 0.25 |
| ms_t1w_t2w | 8 | +0.0168 | 0.62 | 0.95 |
| breast_t1wce_t2w | 40 | +0.0323 | 0.82 | 1.3e-05 |
| breast_t2w_t1wce | 39 | +0.0470 | 0.92 | 4e-10 |
| brain_t1w_t2w | 8 | +0.0123 | 0.88 | 0.023 |
| brain_t2w_t1w | 8 | +0.0085 | 0.88 | 0.023 |
| abd_t1in_t2spir | 4 | +0.0474 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0713 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | -0.0141 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0157 | 0.25 | nan |
| abd_t2spir_ct | 20 | -0.0404 | 0.00 | 1.9e-06 |
| mand_cbct_hanseg_ct | 40 | +0.0324 | 1.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | +0.0365 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | +0.0168 | 0.80 | 5.4e-05 |
| pelvis_ct_mri | 21 | +0.0455 | 0.95 | 4.8e-06 |
| pancreas_t1wce_t2w | 40 | +0.0228 | 0.97 | 3.6e-12 |
| pancreas_t2w_t1wce | 40 | +0.0094 | 0.80 | 0.00018 |
pooled over cells: median of cell medians +0.0323, cells with d > 0: 19 / 23, p < 0.05 same sign: 16 +, 1 -

### CCC (all correct): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.1496 | 1.00 | 1.6e-11 |
| glioma_t1n_t2w | 67 | +0.1321 | 1.00 | 1.1e-12 |
| glioma_t2w_t2f | 68 | +0.1057 | 1.00 | 7.6e-13 |
| glioma_t2w_t1n | 62 | +0.0780 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 8 | -0.0599 | 0.12 | 0.055 |
| ms_flair_t1w | 6 | +0.0327 | 1.00 | 0.031 |
| ms_t1w_flair | 8 | +0.1302 | 1.00 | 0.0078 |
| ms_t1w_t2w | 8 | +0.1827 | 1.00 | 0.0078 |
| breast_t1wce_t2w | 40 | +0.0202 | 0.68 | 0.0019 |
| breast_t2w_t1wce | 39 | +0.0033 | 0.51 | 0.78 |
| brain_t1w_t2w | 8 | +0.0789 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0507 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0818 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0942 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | +0.0228 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0351 | 1.00 | nan |
| abd_t2spir_ct | 20 | +0.0140 | 0.60 | 0.35 |
| mand_cbct_hanseg_ct | 40 | -0.0288 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.0261 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | +0.1143 | 1.00 | 6e-08 |
| pelvis_ct_mri | 21 | +0.1250 | 1.00 | 9.5e-07 |
| pancreas_t1wce_t2w | 40 | -0.0137 | 0.15 | 1.8e-07 |
| pancreas_t2w_t1wce | 40 | +0.0004 | 0.53 | 0.97 |
pooled over cells: median of cell medians +0.0507, cells with d > 0: 19 / 23, p < 0.05 same sign: 13 +, 3 -

### CCC (all correct): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.0752 | 1.00 | 1.6e-11 |
| glioma_t1n_t2w | 67 | +0.0644 | 1.00 | 1.1e-12 |
| glioma_t2w_t2f | 68 | +0.0372 | 0.99 | 9.1e-13 |
| glioma_t2w_t1n | 62 | +0.0437 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 8 | +0.0500 | 0.88 | 0.016 |
| ms_flair_t1w | 6 | +0.0218 | 0.50 | 0.44 |
| ms_t1w_flair | 8 | -0.0080 | 0.25 | 0.31 |
| ms_t1w_t2w | 8 | +0.0152 | 0.62 | 0.95 |
| breast_t1wce_t2w | 40 | +0.0307 | 0.75 | 0.002 |
| breast_t2w_t1wce | 39 | +0.0251 | 0.74 | 4e-06 |
| brain_t1w_t2w | 8 | +0.0117 | 0.88 | 0.039 |
| brain_t2w_t1w | 8 | +0.0076 | 0.88 | 0.055 |
| abd_t1in_t2spir | 4 | +0.0446 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0622 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | -0.0095 | 0.25 | nan |
| abd_t2spir_t1out | 4 | -0.0201 | 0.25 | nan |
| abd_t2spir_ct | 20 | -0.0328 | 0.10 | 1.3e-05 |
| mand_cbct_hanseg_ct | 40 | +0.0269 | 1.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | +0.0253 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | +0.0197 | 0.80 | 0.001 |
| pelvis_ct_mri | 21 | +0.0404 | 0.95 | 1.8e-05 |
| pancreas_t1wce_t2w | 40 | +0.0130 | 0.88 | 5e-09 |
| pancreas_t2w_t1wce | 40 | +0.0064 | 0.72 | 0.0032 |
pooled over cells: median of cell medians +0.0251, cells with d > 0: 19 / 23, p < 0.05 same sign: 15 +, 1 -

### CCC (all correct): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.1701 | 1.00 | 1.6e-11 |
| glioma_t1n_t2w | 67 | +0.1453 | 1.00 | 1.1e-12 |
| glioma_t2w_t2f | 68 | +0.1126 | 1.00 | 7.6e-13 |
| glioma_t2w_t1n | 62 | +0.0955 | 1.00 | 7.6e-12 |
| ms_flair_t2w | 8 | -0.0571 | 0.12 | 0.055 |
| ms_flair_t1w | 6 | +0.0343 | 0.83 | 0.062 |
| ms_t1w_flair | 8 | +0.1313 | 1.00 | 0.0078 |
| ms_t1w_t2w | 8 | +0.1798 | 1.00 | 0.0078 |
| breast_t1wce_t2w | 40 | +0.0121 | 0.62 | 0.073 |
| breast_t2w_t1wce | 39 | +0.0069 | 0.59 | 0.34 |
| brain_t1w_t2w | 8 | +0.0782 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0527 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0778 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0922 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | +0.0231 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0368 | 0.75 | nan |
| abd_t2spir_ct | 20 | +0.0135 | 0.65 | 0.31 |
| mand_cbct_hanseg_ct | 40 | -0.0234 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.0174 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | +0.0891 | 1.00 | 6e-08 |
| pelvis_ct_mri | 21 | +0.1261 | 1.00 | 9.5e-07 |
| pancreas_t1wce_t2w | 40 | -0.0122 | 0.20 | 7.2e-05 |
| pancreas_t2w_t1wce | 40 | +0.0038 | 0.65 | 0.21 |
pooled over cells: median of cell medians +0.0527, cells with d > 0: 19 / 23, p < 0.05 same sign: 11 +, 3 -

### CCC (all correct): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.0101 | 0.22 | 1.5e-06 |
| glioma_t1n_t2w | 67 | +0.0024 | 0.57 | 0.32 |
| glioma_t2w_t2f | 68 | -0.0026 | 0.40 | 0.095 |
| glioma_t2w_t1n | 62 | +0.0098 | 0.85 | 1.8e-07 |
| ms_flair_t2w | 8 | -0.0630 | 0.00 | 0.0078 |
| ms_flair_t1w | 6 | -0.0341 | 0.00 | 0.031 |
| ms_t1w_flair | 8 | -0.0296 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.0010 | 0.50 | 0.55 |
| breast_t1wce_t2w | 40 | +0.0577 | 0.97 | 3.6e-12 |
| breast_t2w_t1wce | 39 | -0.0207 | 0.18 | 1.2e-06 |
| brain_t1w_t2w | 8 | -0.0034 | 0.12 | 0.023 |
| brain_t2w_t1w | 8 | -0.0139 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0083 | 0.75 | nan |
| abd_t1in_ct | 20 | -0.0152 | 0.10 | 9.5e-06 |
| abd_t2spir_t1in | 4 | +0.0061 | 0.75 | nan |
| abd_t2spir_t1out | 4 | -0.0149 | 0.00 | nan |
| abd_t2spir_ct | 20 | -0.0040 | 0.40 | 0.29 |
| mand_cbct_hanseg_ct | 40 | +0.0870 | 1.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | +0.1037 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | -0.0199 | 0.04 | 3e-07 |
| pelvis_ct_mri | 21 | +0.0246 | 0.81 | 0.00013 |
| pancreas_t1wce_t2w | 40 | +0.0043 | 0.62 | 0.094 |
| pancreas_t2w_t1wce | 40 | -0.0225 | 0.00 | 1.8e-12 |
pooled over cells: median of cell medians -0.0034, cells with d > 0: 9 / 23, p < 0.05 same sign: 5 +, 10 -

### CCC (all correct): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 60 | -0.1208 | 0.00 | 1.6e-11 |
| glioma_t1n_t2w | 67 | -0.1163 | 0.00 | 1.1e-12 |
| glioma_t2w_t2f | 68 | -0.1038 | 0.00 | 7.6e-13 |
| glioma_t2w_t1n | 62 | -0.0951 | 0.00 | 7.6e-12 |
| ms_flair_t2w | 8 | -0.2269 | 0.00 | 0.0078 |
| ms_flair_t1w | 6 | -0.0590 | 0.00 | 0.031 |
| ms_t1w_flair | 8 | -0.1441 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.1503 | 0.00 | 0.0078 |
| breast_t1wce_t2w | 40 | -0.2243 | 0.00 | 1.8e-12 |
| breast_t2w_t1wce | 39 | -0.1338 | 0.03 | 7.3e-12 |
| brain_t1w_t2w | 8 | -0.1096 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1130 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.1048 | 0.00 | nan |
| abd_t1in_ct | 20 | -0.1082 | 0.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | -0.0811 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0938 | 0.00 | nan |
| abd_t2spir_ct | 20 | -0.1062 | 0.00 | 1.9e-06 |
| mand_cbct_hanseg_ct | 40 | -0.2017 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.2900 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | -0.1185 | 0.00 | 6e-08 |
| pelvis_ct_mri | 21 | -0.0613 | 0.00 | 9.5e-07 |
| pancreas_t1wce_t2w | 40 | -0.0406 | 0.03 | 1.3e-11 |
| pancreas_t2w_t1wce | 40 | -0.0588 | 0.03 | 3.6e-12 |
pooled over cells: median of cell medians -0.1096, cells with d > 0: 0 / 23, p < 0.05 same sign: 0 +, 20 -

### FFF (all fail): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.0008 | 0.64 | 0.0007 |
| glioma_t1n_t2w | 70 | +0.0021 | 0.79 | 1.1e-07 |
| glioma_t2w_t2f | 69 | +0.0006 | 0.65 | 0.0048 |
| glioma_t2w_t1n | 70 | +0.0029 | 0.89 | 1.5e-09 |
| ms_flair_t2w | 8 | +0.0045 | 0.62 | 0.25 |
| ms_flair_t1w | 8 | +0.0090 | 0.88 | 0.016 |
| ms_t1w_flair | 8 | -0.0065 | 0.38 | 0.25 |
| ms_t1w_t2w | 8 | +0.0117 | 1.00 | 0.0078 |
| breast_t1wce_t2w | 40 | +0.0012 | 0.68 | 0.068 |
| breast_t2w_t1wce | 40 | +0.0000 | 0.50 | 0.76 |
| brain_t1w_t2w | 8 | -0.0001 | 0.50 | 0.74 |
| brain_t2w_t1w | 8 | -0.0020 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0010 | 0.50 | nan |
| abd_t1in_ct | 20 | -0.0060 | 0.10 | 0.00013 |
| abd_t2spir_t1in | 4 | +0.0016 | 0.50 | nan |
| abd_t2spir_t1out | 4 | +0.0001 | 0.50 | nan |
| abd_t2spir_ct | 19 | +0.0020 | 0.79 | 0.002 |
| mand_cbct_hanseg_ct | 40 | -0.0006 | 0.45 | 0.99 |
| mand_cbct_pddca_ct | 40 | +0.0002 | 0.55 | 0.48 |
| pelvis_mri_ct | 26 | -0.0011 | 0.23 | 0.00066 |
| pelvis_ct_mri | 24 | +0.0008 | 0.58 | 0.21 |
| pancreas_t1wce_t2w | 40 | +0.0001 | 0.53 | 0.54 |
| pancreas_t2w_t1wce | 40 | -0.0006 | 0.38 | 0.028 |
pooled over cells: median of cell medians +0.0002, cells with d > 0: 15 / 23, p < 0.05 same sign: 7 +, 4 -

### FFF (all fail): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | -0.0008 | 0.40 | 0.057 |
| glioma_t1n_t2w | 70 | +0.0010 | 0.67 | 0.0067 |
| glioma_t2w_t2f | 69 | -0.0014 | 0.32 | 0.0029 |
| glioma_t2w_t1n | 70 | -0.0015 | 0.30 | 0.0055 |
| ms_flair_t2w | 8 | -0.0170 | 0.12 | 0.039 |
| ms_flair_t1w | 8 | -0.0027 | 0.50 | 0.38 |
| ms_t1w_flair | 8 | -0.0022 | 0.38 | 0.95 |
| ms_t1w_t2w | 8 | +0.0057 | 0.75 | 0.15 |
| breast_t1wce_t2w | 40 | -0.0259 | 0.03 | 5.5e-12 |
| breast_t2w_t1wce | 40 | -0.0076 | 0.00 | 1.8e-12 |
| brain_t1w_t2w | 8 | -0.0131 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.0067 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0085 | 0.00 | nan |
| abd_t1in_ct | 20 | -0.0147 | 0.05 | 9.5e-06 |
| abd_t2spir_t1in | 4 | -0.0018 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0000 | 0.50 | nan |
| abd_t2spir_ct | 19 | -0.0007 | 0.42 | 0.98 |
| mand_cbct_hanseg_ct | 40 | -0.0303 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.0310 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 26 | -0.0055 | 0.04 | 1.5e-07 |
| pelvis_ct_mri | 24 | +0.0007 | 0.58 | 0.079 |
| pancreas_t1wce_t2w | 40 | -0.0008 | 0.30 | 0.1 |
| pancreas_t2w_t1wce | 40 | -0.0005 | 0.42 | 0.42 |
pooled over cells: median of cell medians -0.0022, cells with d > 0: 3 / 23, p < 0.05 same sign: 1 +, 11 -

### FFF (all fail): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.0149 | 0.94 | 1.1e-11 |
| glioma_t1n_t2w | 70 | +0.0015 | 0.57 | 0.092 |
| glioma_t2w_t2f | 69 | +0.0028 | 0.72 | 0.00016 |
| glioma_t2w_t1n | 70 | +0.0152 | 0.87 | 3.9e-10 |
| ms_flair_t2w | 8 | +0.0064 | 0.62 | 0.11 |
| ms_flair_t1w | 8 | +0.0063 | 1.00 | 0.0078 |
| ms_t1w_flair | 8 | -0.0073 | 0.25 | 0.039 |
| ms_t1w_t2w | 8 | +0.0044 | 0.88 | 0.11 |
| breast_t1wce_t2w | 40 | +0.0064 | 0.88 | 6.4e-06 |
| breast_t2w_t1wce | 40 | +0.0004 | 0.57 | 0.5 |
| brain_t1w_t2w | 8 | -0.0052 | 0.25 | 0.31 |
| brain_t2w_t1w | 8 | +0.0006 | 0.50 | 0.74 |
| abd_t1in_t2spir | 4 | +0.0093 | 0.75 | nan |
| abd_t1in_ct | 20 | +0.0024 | 0.55 | 0.35 |
| abd_t2spir_t1in | 4 | +0.0396 | 0.75 | nan |
| abd_t2spir_t1out | 4 | +0.0266 | 1.00 | nan |
| abd_t2spir_ct | 19 | -0.0280 | 0.05 | 7.6e-06 |
| mand_cbct_hanseg_ct | 40 | +0.0078 | 0.65 | 0.033 |
| mand_cbct_pddca_ct | 40 | +0.0103 | 0.70 | 0.0086 |
| pelvis_mri_ct | 26 | -0.0069 | 0.12 | 0.00022 |
| pelvis_ct_mri | 24 | +0.0033 | 0.62 | 0.33 |
| pancreas_t1wce_t2w | 40 | +0.0076 | 0.82 | 1.2e-06 |
| pancreas_t2w_t1wce | 40 | -0.0007 | 0.45 | 0.19 |
pooled over cells: median of cell medians +0.0044, cells with d > 0: 18 / 23, p < 0.05 same sign: 8 +, 3 -

### FFF (all fail): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.0148 | 0.84 | 1.2e-09 |
| glioma_t1n_t2w | 70 | +0.0108 | 0.73 | 4.2e-07 |
| glioma_t2w_t2f | 69 | -0.0095 | 0.25 | 5.6e-06 |
| glioma_t2w_t1n | 70 | -0.0089 | 0.36 | 0.014 |
| ms_flair_t2w | 8 | -0.0183 | 0.00 | 0.0078 |
| ms_flair_t1w | 8 | +0.0032 | 0.88 | 0.023 |
| ms_t1w_flair | 8 | -0.0068 | 0.12 | 0.2 |
| ms_t1w_t2w | 8 | -0.0059 | 0.38 | 0.2 |
| breast_t1wce_t2w | 40 | -0.0343 | 0.03 | 5.5e-12 |
| breast_t2w_t1wce | 40 | -0.0338 | 0.00 | 1.8e-12 |
| brain_t1w_t2w | 8 | -0.0139 | 0.12 | 0.023 |
| brain_t2w_t1w | 8 | -0.0094 | 0.25 | 0.46 |
| abd_t1in_t2spir | 4 | +0.0007 | 0.50 | nan |
| abd_t1in_ct | 20 | +0.0556 | 0.90 | 3.6e-05 |
| abd_t2spir_t1in | 4 | +0.0686 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0844 | 1.00 | nan |
| abd_t2spir_ct | 19 | +0.0481 | 0.95 | 7.6e-06 |
| mand_cbct_hanseg_ct | 40 | -0.1638 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.1650 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 26 | -0.0191 | 0.19 | 0.019 |
| pelvis_ct_mri | 24 | +0.0015 | 0.58 | 0.51 |
| pancreas_t1wce_t2w | 40 | -0.0056 | 0.20 | 0.00025 |
| pancreas_t2w_t1wce | 40 | -0.0008 | 0.42 | 0.48 |
pooled over cells: median of cell medians -0.0059, cells with d > 0: 9 / 23, p < 0.05 same sign: 5 +, 10 -

### FFF (all fail): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.0667 | 1.00 | 3.6e-13 |
| glioma_t1n_t2w | 70 | +0.0685 | 0.99 | 3.7e-13 |
| glioma_t2w_t2f | 69 | +0.0497 | 1.00 | 5.2e-13 |
| glioma_t2w_t1n | 70 | +0.0531 | 1.00 | 3.6e-13 |
| ms_flair_t2w | 8 | +0.0279 | 0.88 | 0.023 |
| ms_flair_t1w | 8 | +0.0417 | 1.00 | 0.0078 |
| ms_t1w_flair | 8 | -0.0046 | 0.50 | 0.55 |
| ms_t1w_t2w | 8 | +0.0167 | 0.62 | 0.55 |
| breast_t1wce_t2w | 40 | +0.0657 | 0.97 | 3.6e-12 |
| breast_t2w_t1wce | 40 | +0.0635 | 1.00 | 1.8e-12 |
| brain_t1w_t2w | 8 | +0.0094 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0117 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0703 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0629 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | -0.0142 | 0.25 | nan |
| abd_t2spir_t1out | 4 | -0.0197 | 0.00 | nan |
| abd_t2spir_ct | 19 | -0.0401 | 0.00 | 3.8e-06 |
| mand_cbct_hanseg_ct | 40 | +0.0351 | 1.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | +0.0287 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 26 | +0.0156 | 0.88 | 1.3e-06 |
| pelvis_ct_mri | 24 | +0.0540 | 0.96 | 3.6e-07 |
| pancreas_t1wce_t2w | 40 | +0.0350 | 1.00 | 1.8e-12 |
| pancreas_t2w_t1wce | 40 | +0.0167 | 0.80 | 1.8e-05 |
pooled over cells: median of cell medians +0.0350, cells with d > 0: 19 / 23, p < 0.05 same sign: 17 +, 1 -

### FFF (all fail): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.1191 | 1.00 | 3.6e-13 |
| glioma_t1n_t2w | 70 | +0.0972 | 1.00 | 3.6e-13 |
| glioma_t2w_t2f | 69 | +0.0819 | 1.00 | 5.2e-13 |
| glioma_t2w_t1n | 70 | +0.0710 | 1.00 | 3.6e-13 |
| ms_flair_t2w | 8 | -0.0072 | 0.38 | 0.95 |
| ms_flair_t1w | 8 | +0.0103 | 0.62 | 0.25 |
| ms_t1w_flair | 8 | +0.0843 | 0.75 | 0.039 |
| ms_t1w_t2w | 8 | +0.1088 | 0.75 | 0.078 |
| breast_t1wce_t2w | 40 | +0.0430 | 0.90 | 1.8e-08 |
| breast_t2w_t1wce | 40 | -0.0380 | 0.17 | 2e-05 |
| brain_t1w_t2w | 8 | +0.0468 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0306 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0890 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.1419 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | +0.0257 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0123 | 0.50 | nan |
| abd_t2spir_ct | 19 | +0.0259 | 0.84 | 0.0071 |
| mand_cbct_hanseg_ct | 40 | -0.0156 | 0.05 | 9.1e-12 |
| mand_cbct_pddca_ct | 40 | -0.0171 | 0.05 | 9.1e-12 |
| pelvis_mri_ct | 26 | +0.1082 | 1.00 | 3e-08 |
| pelvis_ct_mri | 24 | +0.1228 | 1.00 | 1.2e-07 |
| pancreas_t1wce_t2w | 40 | -0.0082 | 0.28 | 0.0031 |
| pancreas_t2w_t1wce | 40 | +0.0166 | 0.82 | 7.3e-08 |
pooled over cells: median of cell medians +0.0430, cells with d > 0: 18 / 23, p < 0.05 same sign: 13 +, 4 -

### FFF (all fail): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.0572 | 1.00 | 3.6e-13 |
| glioma_t1n_t2w | 70 | +0.0597 | 0.99 | 4.2e-13 |
| glioma_t2w_t2f | 69 | +0.0403 | 1.00 | 5.2e-13 |
| glioma_t2w_t1n | 70 | +0.0432 | 1.00 | 3.6e-13 |
| ms_flair_t2w | 8 | +0.0247 | 0.88 | 0.039 |
| ms_flair_t1w | 8 | +0.0346 | 0.88 | 0.016 |
| ms_t1w_flair | 8 | +0.0072 | 0.50 | 0.74 |
| ms_t1w_t2w | 8 | +0.0396 | 0.75 | 0.2 |
| breast_t1wce_t2w | 40 | +0.0691 | 0.97 | 5.5e-12 |
| breast_t2w_t1wce | 40 | +0.0344 | 0.95 | 6e-11 |
| brain_t1w_t2w | 8 | +0.0067 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0084 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0609 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0582 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | -0.0033 | 0.50 | nan |
| abd_t2spir_t1out | 4 | -0.0103 | 0.00 | nan |
| abd_t2spir_ct | 19 | -0.0370 | 0.00 | 3.8e-06 |
| mand_cbct_hanseg_ct | 40 | +0.0331 | 1.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | +0.0255 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 26 | +0.0251 | 0.88 | 1.3e-05 |
| pelvis_ct_mri | 24 | +0.0543 | 0.96 | 2.4e-07 |
| pancreas_t1wce_t2w | 40 | +0.0301 | 1.00 | 1.8e-12 |
| pancreas_t2w_t1wce | 40 | +0.0139 | 0.75 | 0.00021 |
pooled over cells: median of cell medians +0.0331, cells with d > 0: 20 / 23, p < 0.05 same sign: 17 +, 1 -

### FFF (all fail): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.1369 | 1.00 | 3.6e-13 |
| glioma_t1n_t2w | 70 | +0.1033 | 1.00 | 3.6e-13 |
| glioma_t2w_t2f | 69 | +0.0858 | 1.00 | 5.2e-13 |
| glioma_t2w_t1n | 70 | +0.0830 | 1.00 | 3.6e-13 |
| ms_flair_t2w | 8 | -0.0031 | 0.38 | 0.95 |
| ms_flair_t1w | 8 | +0.0095 | 0.62 | 0.25 |
| ms_t1w_flair | 8 | +0.0796 | 0.75 | 0.039 |
| ms_t1w_t2w | 8 | +0.1049 | 0.75 | 0.078 |
| breast_t1wce_t2w | 40 | +0.0350 | 0.78 | 2e-06 |
| breast_t2w_t1wce | 40 | -0.0385 | 0.15 | 1.9e-06 |
| brain_t1w_t2w | 8 | +0.0415 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0357 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0691 | 0.75 | nan |
| abd_t1in_ct | 20 | +0.1311 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | +0.0301 | 0.75 | nan |
| abd_t2spir_t1out | 4 | +0.0019 | 0.50 | nan |
| abd_t2spir_ct | 19 | +0.0300 | 0.84 | 0.02 |
| mand_cbct_hanseg_ct | 40 | -0.0139 | 0.05 | 2.5e-11 |
| mand_cbct_pddca_ct | 40 | -0.0159 | 0.05 | 7.8e-11 |
| pelvis_mri_ct | 26 | +0.0621 | 0.88 | 2.6e-06 |
| pelvis_ct_mri | 24 | +0.1334 | 1.00 | 1.2e-07 |
| pancreas_t1wce_t2w | 40 | -0.0021 | 0.40 | 0.13 |
| pancreas_t2w_t1wce | 40 | +0.0225 | 0.97 | 5.5e-12 |
pooled over cells: median of cell medians +0.0357, cells with d > 0: 18 / 23, p < 0.05 same sign: 13 +, 3 -

### FFF (all fail): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | -0.0044 | 0.34 | 0.0017 |
| glioma_t1n_t2w | 70 | +0.0035 | 0.59 | 0.055 |
| glioma_t2w_t2f | 69 | -0.0032 | 0.43 | 0.94 |
| glioma_t2w_t1n | 70 | +0.0066 | 0.67 | 7.5e-05 |
| ms_flair_t2w | 8 | -0.0484 | 0.12 | 0.016 |
| ms_flair_t1w | 8 | -0.0174 | 0.25 | 0.055 |
| ms_t1w_flair | 8 | -0.0257 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.0077 | 0.12 | 0.078 |
| breast_t1wce_t2w | 40 | +0.0651 | 1.00 | 1.8e-12 |
| breast_t2w_t1wce | 40 | -0.0210 | 0.15 | 2.9e-06 |
| brain_t1w_t2w | 8 | -0.0015 | 0.38 | 0.74 |
| brain_t2w_t1w | 8 | -0.0185 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0071 | 0.75 | nan |
| abd_t1in_ct | 20 | -0.0081 | 0.20 | 0.024 |
| abd_t2spir_t1in | 4 | +0.0034 | 0.75 | nan |
| abd_t2spir_t1out | 4 | -0.0135 | 0.25 | nan |
| abd_t2spir_ct | 19 | +0.0091 | 0.63 | 0.35 |
| mand_cbct_hanseg_ct | 40 | +0.0881 | 1.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | +0.0921 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 26 | -0.0155 | 0.08 | 6.2e-06 |
| pelvis_ct_mri | 24 | +0.0153 | 0.79 | 0.0011 |
| pancreas_t1wce_t2w | 40 | +0.0058 | 0.68 | 0.017 |
| pancreas_t2w_t1wce | 40 | -0.0198 | 0.00 | 1.8e-12 |
pooled over cells: median of cell medians -0.0032, cells with d > 0: 10 / 23, p < 0.05 same sign: 6 +, 8 -

### FFF (all fail): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | -0.1086 | 0.00 | 3.6e-13 |
| glioma_t1n_t2w | 70 | -0.1246 | 0.00 | 3.6e-13 |
| glioma_t2w_t2f | 69 | -0.1010 | 0.00 | 5.2e-13 |
| glioma_t2w_t1n | 70 | -0.0911 | 0.00 | 3.6e-13 |
| ms_flair_t2w | 8 | -0.2257 | 0.00 | 0.0078 |
| ms_flair_t1w | 8 | -0.0655 | 0.00 | 0.0078 |
| ms_t1w_flair | 8 | -0.1601 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.2039 | 0.00 | 0.0078 |
| breast_t1wce_t2w | 40 | -0.2539 | 0.00 | 1.8e-12 |
| breast_t2w_t1wce | 40 | -0.1468 | 0.00 | 1.8e-12 |
| brain_t1w_t2w | 8 | -0.1108 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1182 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0947 | 0.00 | nan |
| abd_t1in_ct | 20 | -0.1079 | 0.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | -0.0783 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0898 | 0.00 | nan |
| abd_t2spir_ct | 19 | -0.0571 | 0.00 | 3.8e-06 |
| mand_cbct_hanseg_ct | 40 | -0.1982 | 0.00 | 1.8e-12 |
| mand_cbct_pddca_ct | 40 | -0.2683 | 0.00 | 1.8e-12 |
| pelvis_mri_ct | 26 | -0.1139 | 0.00 | 3e-08 |
| pelvis_ct_mri | 24 | -0.0690 | 0.00 | 1.2e-07 |
| pancreas_t1wce_t2w | 40 | -0.0413 | 0.05 | 6e-11 |
| pancreas_t2w_t1wce | 40 | -0.0523 | 0.00 | 1.8e-12 |
pooled over cells: median of cell medians -0.1086, cells with d > 0: 0 / 23, p < 0.05 same sign: 0 +, 20 -

### CCF (real fails, others correct): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.0046 | 0.59 | 0.054 |
| glioma_t1n_t2w | 35 | +0.0025 | 0.66 | 0.026 |
| glioma_t2w_t2f | 35 | -0.0029 | 0.31 | 0.033 |
| glioma_t2w_t1n | 45 | +0.0019 | 0.62 | 0.00094 |
| ms_flair_t2w | 8 | +0.0015 | 0.62 | 0.55 |
| ms_flair_t1w | 6 | +0.0011 | 0.67 | 0.16 |
| ms_t1w_flair | 5 | +0.0048 | 0.60 | 0.31 |
| ms_t1w_t2w | 5 | +0.0169 | 1.00 | 0.062 |
| breast_t1wce_t2w | 15 | +0.0019 | 0.67 | 0.39 |
| breast_t2w_t1wce | 16 | -0.0001 | 0.50 | 0.9 |
| brain_t1w_t2w | 8 | +0.0010 | 0.75 | 0.55 |
| brain_t2w_t1w | 8 | +0.0007 | 0.50 | 0.64 |
| abd_t1in_t2spir | 3 | -0.0035 | 0.00 | nan |
| abd_t1in_ct | 3 | -0.0037 | 0.00 | nan |
| abd_t2spir_t1in | 3 | -0.0004 | 0.33 | nan |
| abd_t2spir_ct | 4 | -0.0043 | 0.00 | nan |
| mand_cbct_hanseg_ct | 12 | +0.0081 | 0.92 | 0.0024 |
| mand_cbct_pddca_ct | 11 | +0.0084 | 0.82 | 0.0068 |
| pelvis_ct_mri | 11 | -0.0050 | 0.27 | 0.024 |
| pancreas_t1wce_t2w | 39 | +0.0002 | 0.54 | 0.96 |
| pancreas_t2w_t1wce | 40 | +0.0004 | 0.55 | 0.28 |
pooled over cells: median of cell medians +0.0010, cells with d > 0: 14 / 21, p < 0.05 same sign: 4 +, 2 -

### CCF (real fails, others correct): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.0019 | 0.59 | 0.41 |
| glioma_t1n_t2w | 35 | +0.0043 | 0.63 | 0.014 |
| glioma_t2w_t2f | 35 | -0.0042 | 0.23 | 0.0017 |
| glioma_t2w_t1n | 45 | -0.0071 | 0.22 | 1.4e-06 |
| ms_flair_t2w | 8 | -0.0249 | 0.12 | 0.016 |
| ms_flair_t1w | 6 | +0.0000 | 0.50 | 1 |
| ms_t1w_flair | 5 | -0.0060 | 0.40 | 0.62 |
| ms_t1w_t2w | 5 | +0.0080 | 1.00 | 0.062 |
| breast_t1wce_t2w | 15 | -0.0186 | 0.00 | 6.1e-05 |
| breast_t2w_t1wce | 16 | -0.0161 | 0.00 | 3.1e-05 |
| brain_t1w_t2w | 8 | -0.0066 | 0.12 | 0.016 |
| brain_t2w_t1w | 8 | -0.0070 | 0.12 | 0.016 |
| abd_t1in_t2spir | 3 | -0.0141 | 0.00 | nan |
| abd_t1in_ct | 3 | -0.0074 | 0.00 | nan |
| abd_t2spir_t1in | 3 | -0.0010 | 0.33 | nan |
| abd_t2spir_ct | 4 | -0.0050 | 0.25 | nan |
| mand_cbct_hanseg_ct | 12 | -0.0078 | 0.25 | 0.064 |
| mand_cbct_pddca_ct | 11 | -0.0069 | 0.36 | 0.52 |
| pelvis_ct_mri | 11 | -0.0068 | 0.00 | 0.00098 |
| pancreas_t1wce_t2w | 39 | -0.0008 | 0.23 | 0.00036 |
| pancreas_t2w_t1wce | 40 | +0.0007 | 0.62 | 0.11 |
pooled over cells: median of cell medians -0.0066, cells with d > 0: 5 / 21, p < 0.05 same sign: 1 +, 9 -

### CCF (real fails, others correct): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.0206 | 0.86 | 1.2e-05 |
| glioma_t1n_t2w | 35 | -0.0005 | 0.43 | 0.88 |
| glioma_t2w_t2f | 35 | +0.0002 | 0.51 | 0.43 |
| glioma_t2w_t1n | 45 | +0.0161 | 0.84 | 3.8e-07 |
| ms_flair_t2w | 8 | -0.0006 | 0.50 | 0.95 |
| ms_flair_t1w | 6 | +0.0107 | 1.00 | 0.031 |
| ms_t1w_flair | 5 | +0.0018 | 0.60 | 1 |
| ms_t1w_t2w | 5 | +0.0170 | 1.00 | 0.062 |
| breast_t1wce_t2w | 15 | +0.0100 | 0.87 | 0.0067 |
| breast_t2w_t1wce | 16 | +0.0021 | 0.56 | 0.21 |
| brain_t1w_t2w | 8 | -0.0973 | 0.12 | 0.2 |
| brain_t2w_t1w | 8 | -0.0063 | 0.38 | 0.84 |
| abd_t1in_t2spir | 3 | +0.0127 | 1.00 | nan |
| abd_t1in_ct | 3 | -0.0238 | 0.00 | nan |
| abd_t2spir_t1in | 3 | +0.0322 | 1.00 | nan |
| abd_t2spir_ct | 4 | -0.0333 | 0.00 | nan |
| mand_cbct_hanseg_ct | 12 | +0.0682 | 1.00 | 0.00049 |
| mand_cbct_pddca_ct | 11 | +0.0583 | 0.82 | 0.0049 |
| pelvis_ct_mri | 11 | +0.0019 | 0.55 | 0.64 |
| pancreas_t1wce_t2w | 39 | +0.0111 | 0.90 | 5.6e-07 |
| pancreas_t2w_t1wce | 40 | +0.0005 | 0.53 | 0.49 |
pooled over cells: median of cell medians +0.0021, cells with d > 0: 15 / 21, p < 0.05 same sign: 7 +, 0 -

### CCF (real fails, others correct): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.0120 | 0.82 | 0.00069 |
| glioma_t1n_t2w | 35 | +0.0063 | 0.74 | 0.0053 |
| glioma_t2w_t2f | 35 | -0.0131 | 0.31 | 0.007 |
| glioma_t2w_t1n | 45 | -0.0046 | 0.47 | 0.95 |
| ms_flair_t2w | 8 | -0.0419 | 0.00 | 0.0078 |
| ms_flair_t1w | 6 | +0.0057 | 0.67 | 0.44 |
| ms_t1w_flair | 5 | -0.0205 | 0.00 | 0.062 |
| ms_t1w_t2w | 5 | -0.0052 | 0.20 | 0.31 |
| breast_t1wce_t2w | 15 | -0.0236 | 0.07 | 0.00012 |
| breast_t2w_t1wce | 16 | -0.0540 | 0.00 | 3.1e-05 |
| brain_t1w_t2w | 8 | -0.0900 | 0.12 | 0.2 |
| brain_t2w_t1w | 8 | -0.0206 | 0.38 | 0.84 |
| abd_t1in_t2spir | 3 | -0.0016 | 0.33 | nan |
| abd_t1in_ct | 3 | +0.0676 | 1.00 | nan |
| abd_t2spir_t1in | 3 | +0.0645 | 1.00 | nan |
| abd_t2spir_ct | 4 | +0.0394 | 0.50 | nan |
| mand_cbct_hanseg_ct | 12 | -0.1143 | 0.17 | 0.0049 |
| mand_cbct_pddca_ct | 11 | -0.0746 | 0.18 | 0.0068 |
| pelvis_ct_mri | 11 | -0.0197 | 0.36 | 0.067 |
| pancreas_t1wce_t2w | 39 | -0.0064 | 0.26 | 0.0022 |
| pancreas_t2w_t1wce | 40 | +0.0055 | 0.60 | 0.14 |
pooled over cells: median of cell medians -0.0064, cells with d > 0: 7 / 21, p < 0.05 same sign: 2 +, 7 -

### CCF (real fails, others correct): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.0627 | 0.95 | 9.5e-07 |
| glioma_t1n_t2w | 35 | +0.0615 | 0.97 | 1.5e-09 |
| glioma_t2w_t2f | 35 | +0.0449 | 1.00 | 5.8e-11 |
| glioma_t2w_t1n | 45 | +0.0559 | 1.00 | 5.7e-14 |
| ms_flair_t2w | 8 | +0.0584 | 1.00 | 0.0078 |
| ms_flair_t1w | 6 | +0.0446 | 1.00 | 0.031 |
| ms_t1w_flair | 5 | +0.0184 | 0.60 | 0.81 |
| ms_t1w_t2w | 5 | -0.0380 | 0.00 | 0.062 |
| breast_t1wce_t2w | 15 | +0.0310 | 0.87 | 0.0043 |
| breast_t2w_t1wce | 16 | +0.0706 | 1.00 | 3.1e-05 |
| brain_t1w_t2w | 8 | +0.0132 | 0.75 | 0.11 |
| brain_t2w_t1w | 8 | +0.0155 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 3 | +0.0362 | 1.00 | nan |
| abd_t1in_ct | 3 | +0.0603 | 1.00 | nan |
| abd_t2spir_t1in | 3 | -0.0377 | 0.00 | nan |
| abd_t2spir_ct | 4 | -0.0198 | 0.00 | nan |
| mand_cbct_hanseg_ct | 12 | +0.0321 | 1.00 | 0.00049 |
| mand_cbct_pddca_ct | 11 | +0.0409 | 1.00 | 0.00098 |
| pelvis_ct_mri | 11 | +0.0407 | 0.91 | 0.002 |
| pancreas_t1wce_t2w | 39 | +0.0377 | 0.97 | 1.1e-11 |
| pancreas_t2w_t1wce | 40 | +0.0268 | 0.90 | 4.6e-10 |
pooled over cells: median of cell medians +0.0377, cells with d > 0: 18 / 21, p < 0.05 same sign: 14 +, 0 -

### CCF (real fails, others correct): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.1178 | 1.00 | 4.8e-07 |
| glioma_t1n_t2w | 35 | +0.1181 | 1.00 | 5.8e-11 |
| glioma_t2w_t2f | 35 | +0.0763 | 1.00 | 5.8e-11 |
| glioma_t2w_t1n | 45 | +0.0872 | 1.00 | 5.7e-14 |
| ms_flair_t2w | 8 | -0.0160 | 0.38 | 0.84 |
| ms_flair_t1w | 6 | +0.0452 | 0.83 | 0.22 |
| ms_t1w_flair | 5 | +0.0280 | 0.60 | 1 |
| ms_t1w_t2w | 5 | +0.1331 | 1.00 | 0.062 |
| breast_t1wce_t2w | 15 | +0.0253 | 0.93 | 0.00085 |
| breast_t2w_t1wce | 16 | -0.0325 | 0.31 | 0.034 |
| brain_t1w_t2w | 8 | +0.0656 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0252 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 3 | +0.0931 | 1.00 | nan |
| abd_t1in_ct | 3 | +0.2063 | 1.00 | nan |
| abd_t2spir_t1in | 3 | +0.0207 | 0.67 | nan |
| abd_t2spir_ct | 4 | +0.0036 | 0.75 | nan |
| mand_cbct_hanseg_ct | 12 | -0.0226 | 0.17 | 0.034 |
| mand_cbct_pddca_ct | 11 | -0.0200 | 0.00 | 0.00098 |
| pelvis_ct_mri | 11 | +0.1361 | 1.00 | 0.00098 |
| pancreas_t1wce_t2w | 39 | -0.0027 | 0.31 | 0.035 |
| pancreas_t2w_t1wce | 40 | +0.0310 | 0.88 | 1.9e-09 |
pooled over cells: median of cell medians +0.0310, cells with d > 0: 16 / 21, p < 0.05 same sign: 9 +, 4 -

### CCF (real fails, others correct): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.0517 | 0.91 | 4.8e-06 |
| glioma_t1n_t2w | 35 | +0.0514 | 0.94 | 2.2e-07 |
| glioma_t2w_t2f | 35 | +0.0320 | 0.91 | 2.5e-09 |
| glioma_t2w_t1n | 45 | +0.0492 | 1.00 | 5.7e-14 |
| ms_flair_t2w | 8 | +0.0553 | 1.00 | 0.0078 |
| ms_flair_t1w | 6 | +0.0530 | 1.00 | 0.031 |
| ms_t1w_flair | 5 | +0.0140 | 0.60 | 0.81 |
| ms_t1w_t2w | 5 | -0.0573 | 0.40 | 0.31 |
| breast_t1wce_t2w | 15 | +0.0275 | 0.73 | 0.018 |
| breast_t2w_t1wce | 16 | +0.0502 | 1.00 | 3.1e-05 |
| brain_t1w_t2w | 8 | +0.0095 | 0.75 | 0.2 |
| brain_t2w_t1w | 8 | +0.0119 | 0.88 | 0.016 |
| abd_t1in_t2spir | 3 | +0.0387 | 1.00 | nan |
| abd_t1in_ct | 3 | +0.0567 | 1.00 | nan |
| abd_t2spir_t1in | 3 | -0.0242 | 0.00 | nan |
| abd_t2spir_ct | 4 | -0.0112 | 0.00 | nan |
| mand_cbct_hanseg_ct | 12 | +0.0306 | 1.00 | 0.00049 |
| mand_cbct_pddca_ct | 11 | +0.0389 | 1.00 | 0.00098 |
| pelvis_ct_mri | 11 | +0.0502 | 1.00 | 0.00098 |
| pancreas_t1wce_t2w | 39 | +0.0307 | 0.97 | 6.9e-11 |
| pancreas_t2w_t1wce | 40 | +0.0274 | 0.88 | 2.7e-09 |
pooled over cells: median of cell medians +0.0320, cells with d > 0: 18 / 21, p < 0.05 same sign: 14 +, 0 -

### CCF (real fails, others correct): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.1280 | 1.00 | 4.8e-07 |
| glioma_t1n_t2w | 35 | +0.1334 | 1.00 | 5.8e-11 |
| glioma_t2w_t2f | 35 | +0.0804 | 1.00 | 5.8e-11 |
| glioma_t2w_t1n | 45 | +0.0918 | 1.00 | 5.7e-14 |
| ms_flair_t2w | 8 | -0.0136 | 0.38 | 0.74 |
| ms_flair_t1w | 6 | +0.0449 | 0.83 | 0.22 |
| ms_t1w_flair | 5 | +0.0213 | 0.60 | 1 |
| ms_t1w_t2w | 5 | +0.0947 | 0.80 | 0.12 |
| breast_t1wce_t2w | 15 | +0.0123 | 0.67 | 0.15 |
| breast_t2w_t1wce | 16 | -0.0424 | 0.31 | 0.034 |
| brain_t1w_t2w | 8 | +0.0644 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0300 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 3 | +0.0862 | 1.00 | nan |
| abd_t1in_ct | 3 | +0.1963 | 1.00 | nan |
| abd_t2spir_t1in | 3 | +0.0237 | 0.67 | nan |
| abd_t2spir_ct | 4 | +0.0069 | 0.50 | nan |
| mand_cbct_hanseg_ct | 12 | -0.0197 | 0.17 | 0.052 |
| mand_cbct_pddca_ct | 11 | -0.0188 | 0.00 | 0.00098 |
| pelvis_ct_mri | 11 | +0.1490 | 1.00 | 0.00098 |
| pancreas_t1wce_t2w | 39 | -0.0027 | 0.41 | 0.51 |
| pancreas_t2w_t1wce | 40 | +0.0431 | 0.93 | 1.3e-10 |
pooled over cells: median of cell medians +0.0431, cells with d > 0: 16 / 21, p < 0.05 same sign: 8 +, 2 -

### CCF (real fails, others correct): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | -0.0123 | 0.18 | 0.0015 |
| glioma_t1n_t2w | 35 | -0.0046 | 0.43 | 0.19 |
| glioma_t2w_t2f | 35 | -0.0006 | 0.49 | 0.37 |
| glioma_t2w_t1n | 45 | +0.0096 | 0.64 | 0.0006 |
| ms_flair_t2w | 8 | -0.0369 | 0.12 | 0.039 |
| ms_flair_t1w | 6 | -0.0183 | 0.33 | 0.22 |
| ms_t1w_flair | 5 | -0.0277 | 0.00 | 0.062 |
| ms_t1w_t2w | 5 | -0.0193 | 0.20 | 0.12 |
| breast_t1wce_t2w | 15 | +0.0599 | 1.00 | 6.1e-05 |
| breast_t2w_t1wce | 16 | -0.0048 | 0.44 | 0.6 |
| brain_t1w_t2w | 8 | +0.0018 | 0.75 | 0.46 |
| brain_t2w_t1w | 8 | -0.0212 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 3 | +0.0168 | 1.00 | nan |
| abd_t1in_ct | 3 | +0.0049 | 0.67 | nan |
| abd_t2spir_t1in | 3 | -0.0162 | 0.33 | nan |
| abd_t2spir_ct | 4 | +0.0039 | 0.75 | nan |
| mand_cbct_hanseg_ct | 12 | +0.1035 | 1.00 | 0.00049 |
| mand_cbct_pddca_ct | 11 | +0.0976 | 1.00 | 0.00098 |
| pelvis_ct_mri | 11 | +0.0381 | 1.00 | 0.00098 |
| pancreas_t1wce_t2w | 39 | +0.0056 | 0.64 | 0.027 |
| pancreas_t2w_t1wce | 40 | -0.0257 | 0.03 | 2.5e-10 |
pooled over cells: median of cell medians -0.0006, cells with d > 0: 10 / 21, p < 0.05 same sign: 6 +, 4 -

### CCF (real fails, others correct): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 22 | -0.1114 | 0.00 | 4.8e-07 |
| glioma_t1n_t2w | 35 | -0.1227 | 0.00 | 5.8e-11 |
| glioma_t2w_t2f | 35 | -0.1027 | 0.00 | 5.8e-11 |
| glioma_t2w_t1n | 45 | -0.0985 | 0.00 | 5.7e-14 |
| ms_flair_t2w | 8 | -0.2309 | 0.00 | 0.0078 |
| ms_flair_t1w | 6 | -0.0665 | 0.00 | 0.031 |
| ms_t1w_flair | 5 | -0.1811 | 0.00 | 0.062 |
| ms_t1w_t2w | 5 | -0.2124 | 0.00 | 0.062 |
| breast_t1wce_t2w | 15 | -0.2468 | 0.00 | 6.1e-05 |
| breast_t2w_t1wce | 16 | -0.1480 | 0.00 | 3.1e-05 |
| brain_t1w_t2w | 8 | -0.1043 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1282 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 3 | -0.0893 | 0.00 | nan |
| abd_t1in_ct | 3 | -0.0731 | 0.00 | nan |
| abd_t2spir_t1in | 3 | -0.0622 | 0.00 | nan |
| abd_t2spir_ct | 4 | -0.0637 | 0.00 | nan |
| mand_cbct_hanseg_ct | 12 | -0.1905 | 0.00 | 0.00049 |
| mand_cbct_pddca_ct | 11 | -0.2478 | 0.00 | 0.00098 |
| pelvis_ct_mri | 11 | -0.0656 | 0.00 | 0.00098 |
| pancreas_t1wce_t2w | 39 | -0.0430 | 0.08 | 2.5e-10 |
| pancreas_t2w_t1wce | 40 | -0.0544 | 0.00 | 1.8e-12 |
pooled over cells: median of cell medians -0.1043, cells with d > 0: 0 / 21, p < 0.05 same sign: 0 +, 15 -

### CFC (flat fails): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.0045 | 0.84 | 1.5e-05 |
| glioma_t1n_t2w | 26 | +0.0021 | 0.73 | 0.08 |
| glioma_t2w_t2f | 49 | +0.0030 | 0.78 | 5.8e-06 |
| glioma_t2w_t1n | 57 | +0.0043 | 0.88 | 2.1e-08 |
| ms_flair_t2w | 6 | +0.0054 | 0.67 | 0.44 |
| ms_flair_t1w | 5 | +0.0088 | 1.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0016 | 0.25 | 0.74 |
| ms_t1w_t2w | 8 | -0.0007 | 0.38 | 0.95 |
| breast_t2w_t1wce | 11 | +0.0005 | 0.55 | 0.9 |
| brain_t1w_t2w | 8 | -0.0006 | 0.50 | 0.46 |
| brain_t2w_t1w | 8 | -0.0019 | 0.25 | 0.11 |
| abd_t2spir_t1in | 4 | +0.0058 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0034 | 0.67 | nan |
| abd_t2spir_ct | 5 | +0.0007 | 0.80 | 0.31 |
| mand_cbct_hanseg_ct | 14 | +0.0027 | 0.79 | 0.22 |
| mand_cbct_pddca_ct | 20 | +0.0033 | 0.85 | 0.0083 |
| pelvis_ct_mri | 4 | +0.0078 | 1.00 | nan |
| pancreas_t1wce_t2w | 36 | -0.0001 | 0.47 | 0.79 |
| pancreas_t2w_t1wce | 38 | -0.0010 | 0.24 | 0.001 |
pooled over cells: median of cell medians +0.0027, cells with d > 0: 13 / 19, p < 0.05 same sign: 4 +, 1 -

### CFC (flat fails): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.0006 | 0.52 | 0.11 |
| glioma_t1n_t2w | 26 | +0.0011 | 0.69 | 0.084 |
| glioma_t2w_t2f | 49 | -0.0033 | 0.27 | 0.00044 |
| glioma_t2w_t1n | 57 | -0.0028 | 0.39 | 0.0068 |
| ms_flair_t2w | 6 | -0.0239 | 0.00 | 0.031 |
| ms_flair_t1w | 5 | +0.0008 | 0.60 | 1 |
| ms_t1w_flair | 8 | -0.0078 | 0.25 | 0.078 |
| ms_t1w_t2w | 8 | -0.0009 | 0.38 | 0.46 |
| breast_t2w_t1wce | 11 | -0.0074 | 0.00 | 0.00098 |
| brain_t1w_t2w | 8 | -0.0087 | 0.12 | 0.016 |
| brain_t2w_t1w | 8 | -0.0097 | 0.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0026 | 0.75 | nan |
| abd_t2spir_t1out | 3 | +0.0013 | 0.67 | nan |
| abd_t2spir_ct | 5 | -0.0009 | 0.20 | 0.19 |
| mand_cbct_hanseg_ct | 14 | -0.0195 | 0.07 | 0.0017 |
| mand_cbct_pddca_ct | 20 | -0.0185 | 0.00 | 1.9e-06 |
| pelvis_ct_mri | 4 | +0.0007 | 0.50 | nan |
| pancreas_t1wce_t2w | 36 | -0.0013 | 0.28 | 0.012 |
| pancreas_t2w_t1wce | 38 | -0.0006 | 0.47 | 0.66 |
pooled over cells: median of cell medians -0.0013, cells with d > 0: 6 / 19, p < 0.05 same sign: 0 +, 9 -

### CFC (flat fails): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.0194 | 0.94 | 2.9e-07 |
| glioma_t1n_t2w | 26 | +0.0051 | 0.73 | 0.043 |
| glioma_t2w_t2f | 49 | +0.0056 | 0.65 | 0.0094 |
| glioma_t2w_t1n | 57 | +0.0132 | 0.88 | 6.7e-08 |
| ms_flair_t2w | 6 | +0.0009 | 0.50 | 0.84 |
| ms_flair_t1w | 5 | +0.0260 | 1.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0060 | 0.12 | 0.2 |
| ms_t1w_t2w | 8 | +0.0016 | 0.62 | 0.55 |
| breast_t2w_t1wce | 11 | +0.0033 | 0.55 | 1 |
| brain_t1w_t2w | 8 | -0.0362 | 0.25 | 0.25 |
| brain_t2w_t1w | 8 | +0.0078 | 0.50 | 0.74 |
| abd_t2spir_t1in | 4 | +0.0409 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.2888 | 1.00 | nan |
| abd_t2spir_ct | 5 | -0.0785 | 0.00 | 0.062 |
| mand_cbct_hanseg_ct | 14 | +0.0395 | 1.00 | 0.00012 |
| mand_cbct_pddca_ct | 20 | +0.0388 | 1.00 | 1.9e-06 |
| pelvis_ct_mri | 4 | +0.0650 | 1.00 | nan |
| pancreas_t1wce_t2w | 36 | +0.0023 | 0.56 | 0.79 |
| pancreas_t2w_t1wce | 38 | -0.0037 | 0.45 | 0.074 |
pooled over cells: median of cell medians +0.0056, cells with d > 0: 15 / 19, p < 0.05 same sign: 6 +, 0 -

### CFC (flat fails): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.0210 | 0.68 | 0.00098 |
| glioma_t1n_t2w | 26 | +0.0175 | 0.81 | 0.00028 |
| glioma_t2w_t2f | 49 | -0.0095 | 0.24 | 0.00066 |
| glioma_t2w_t1n | 57 | -0.0055 | 0.44 | 0.55 |
| ms_flair_t2w | 6 | -0.0301 | 0.00 | 0.031 |
| ms_flair_t1w | 5 | +0.0019 | 0.80 | 0.44 |
| ms_t1w_flair | 8 | -0.0130 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.0029 | 0.12 | 0.016 |
| breast_t2w_t1wce | 11 | -0.0466 | 0.00 | 0.00098 |
| brain_t1w_t2w | 8 | -0.0089 | 0.25 | 0.2 |
| brain_t2w_t1w | 8 | -0.0036 | 0.38 | 0.55 |
| abd_t2spir_t1in | 4 | +0.0501 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.1055 | 1.00 | nan |
| abd_t2spir_ct | 5 | +0.0937 | 1.00 | 0.062 |
| mand_cbct_hanseg_ct | 14 | -0.1472 | 0.07 | 0.00024 |
| mand_cbct_pddca_ct | 20 | -0.1486 | 0.00 | 1.9e-06 |
| pelvis_ct_mri | 4 | +0.0028 | 0.75 | nan |
| pancreas_t1wce_t2w | 36 | -0.0087 | 0.22 | 0.00055 |
| pancreas_t2w_t1wce | 38 | +0.0031 | 0.58 | 0.83 |
pooled over cells: median of cell medians -0.0036, cells with d > 0: 8 / 19, p < 0.05 same sign: 2 +, 8 -

### CFC (flat fails): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.0830 | 1.00 | 9.3e-10 |
| glioma_t1n_t2w | 26 | +0.0764 | 0.92 | 3e-07 |
| glioma_t2w_t2f | 49 | +0.0527 | 0.98 | 1.1e-14 |
| glioma_t2w_t1n | 57 | +0.0543 | 1.00 | 5.1e-11 |
| ms_flair_t2w | 6 | +0.0143 | 0.67 | 0.44 |
| ms_flair_t1w | 5 | +0.0120 | 0.80 | 0.44 |
| ms_t1w_flair | 8 | -0.0045 | 0.25 | 0.31 |
| ms_t1w_t2w | 8 | -0.0425 | 0.38 | 0.15 |
| breast_t2w_t1wce | 11 | +0.1013 | 1.00 | 0.00098 |
| brain_t1w_t2w | 8 | +0.0156 | 0.88 | 0.039 |
| brain_t2w_t1w | 8 | +0.0146 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | -0.0135 | 0.50 | nan |
| abd_t2spir_t1out | 3 | -0.0051 | 0.33 | nan |
| abd_t2spir_ct | 5 | -0.0564 | 0.20 | 0.12 |
| mand_cbct_hanseg_ct | 14 | +0.0373 | 1.00 | 0.00012 |
| mand_cbct_pddca_ct | 20 | +0.0291 | 0.95 | 3.8e-06 |
| pelvis_ct_mri | 4 | +0.0427 | 1.00 | nan |
| pancreas_t1wce_t2w | 36 | +0.0340 | 0.94 | 6e-06 |
| pancreas_t2w_t1wce | 38 | +0.0181 | 0.79 | 1.9e-05 |
pooled over cells: median of cell medians +0.0181, cells with d > 0: 14 / 19, p < 0.05 same sign: 11 +, 0 -

### CFC (flat fails): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.1263 | 1.00 | 9.3e-10 |
| glioma_t1n_t2w | 26 | +0.1257 | 1.00 | 3e-08 |
| glioma_t2w_t2f | 49 | +0.0844 | 1.00 | 3.6e-15 |
| glioma_t2w_t1n | 57 | +0.0862 | 1.00 | 5.1e-11 |
| ms_flair_t2w | 6 | -0.0502 | 0.17 | 0.062 |
| ms_flair_t1w | 5 | -0.0318 | 0.40 | 1 |
| ms_t1w_flair | 8 | +0.1534 | 0.88 | 0.016 |
| ms_t1w_t2w | 8 | +0.1035 | 0.88 | 0.055 |
| breast_t2w_t1wce | 11 | +0.0010 | 0.55 | 0.97 |
| brain_t1w_t2w | 8 | +0.0415 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0325 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0441 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0370 | 1.00 | nan |
| abd_t2spir_ct | 5 | +0.0972 | 1.00 | 0.062 |
| mand_cbct_hanseg_ct | 14 | -0.0094 | 0.36 | 0.24 |
| mand_cbct_pddca_ct | 20 | -0.0143 | 0.10 | 2.7e-05 |
| pelvis_ct_mri | 4 | +0.1210 | 1.00 | nan |
| pancreas_t1wce_t2w | 36 | -0.0033 | 0.39 | 0.085 |
| pancreas_t2w_t1wce | 38 | +0.0216 | 0.82 | 1e-06 |
pooled over cells: median of cell medians +0.0415, cells with d > 0: 14 / 19, p < 0.05 same sign: 8 +, 1 -

### CFC (flat fails): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.0866 | 0.97 | 1.9e-09 |
| glioma_t1n_t2w | 26 | +0.0686 | 0.96 | 4.2e-07 |
| glioma_t2w_t2f | 49 | +0.0477 | 0.94 | 1.5e-13 |
| glioma_t2w_t1n | 57 | +0.0493 | 1.00 | 5.1e-11 |
| ms_flair_t2w | 6 | +0.0196 | 0.67 | 0.44 |
| ms_flair_t1w | 5 | +0.0168 | 0.80 | 0.62 |
| ms_t1w_flair | 8 | -0.0136 | 0.50 | 0.46 |
| ms_t1w_t2w | 8 | -0.0140 | 0.38 | 0.38 |
| breast_t2w_t1wce | 11 | +0.0559 | 1.00 | 0.00098 |
| brain_t1w_t2w | 8 | +0.0128 | 0.88 | 0.078 |
| brain_t2w_t1w | 8 | +0.0090 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0010 | 0.50 | nan |
| abd_t2spir_t1out | 3 | -0.0078 | 0.33 | nan |
| abd_t2spir_ct | 5 | -0.0477 | 0.20 | 0.19 |
| mand_cbct_hanseg_ct | 14 | +0.0303 | 1.00 | 0.00012 |
| mand_cbct_pddca_ct | 20 | +0.0276 | 0.95 | 3.8e-06 |
| pelvis_ct_mri | 4 | +0.0426 | 0.75 | nan |
| pancreas_t1wce_t2w | 36 | +0.0301 | 0.92 | 6.7e-06 |
| pancreas_t2w_t1wce | 38 | +0.0104 | 0.66 | 0.0073 |
pooled over cells: median of cell medians +0.0196, cells with d > 0: 15 / 19, p < 0.05 same sign: 10 +, 0 -

### CFC (flat fails): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.1441 | 1.00 | 9.3e-10 |
| glioma_t1n_t2w | 26 | +0.1371 | 1.00 | 3e-08 |
| glioma_t2w_t2f | 49 | +0.0862 | 1.00 | 3.6e-15 |
| glioma_t2w_t1n | 57 | +0.1007 | 1.00 | 5.1e-11 |
| ms_flair_t2w | 6 | -0.0462 | 0.17 | 0.062 |
| ms_flair_t1w | 5 | -0.0265 | 0.40 | 1 |
| ms_t1w_flair | 8 | +0.1399 | 0.88 | 0.016 |
| ms_t1w_t2w | 8 | +0.1111 | 0.88 | 0.055 |
| breast_t2w_t1wce | 11 | -0.0086 | 0.45 | 0.83 |
| brain_t1w_t2w | 8 | +0.0356 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0342 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0527 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0331 | 1.00 | nan |
| abd_t2spir_ct | 5 | +0.0872 | 1.00 | 0.062 |
| mand_cbct_hanseg_ct | 14 | -0.0065 | 0.43 | 0.46 |
| mand_cbct_pddca_ct | 20 | -0.0114 | 0.15 | 0.0001 |
| pelvis_ct_mri | 4 | +0.1307 | 1.00 | nan |
| pancreas_t1wce_t2w | 36 | -0.0010 | 0.44 | 0.75 |
| pancreas_t2w_t1wce | 38 | +0.0254 | 0.89 | 3.1e-08 |
pooled over cells: median of cell medians +0.0356, cells with d > 0: 13 / 19, p < 0.05 same sign: 8 +, 1 -

### CFC (flat fails): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | -0.0034 | 0.45 | 0.12 |
| glioma_t1n_t2w | 26 | +0.0042 | 0.58 | 0.12 |
| glioma_t2w_t2f | 49 | +0.0016 | 0.53 | 0.71 |
| glioma_t2w_t1n | 57 | +0.0078 | 0.68 | 8e-05 |
| ms_flair_t2w | 6 | -0.0183 | 0.33 | 0.44 |
| ms_flair_t1w | 5 | -0.0140 | 0.40 | 0.81 |
| ms_t1w_flair | 8 | -0.0195 | 0.12 | 0.016 |
| ms_t1w_t2w | 8 | +0.0056 | 0.62 | 0.74 |
| breast_t2w_t1wce | 11 | -0.0167 | 0.18 | 0.042 |
| brain_t1w_t2w | 8 | -0.0002 | 0.50 | 0.95 |
| brain_t2w_t1w | 8 | -0.0194 | 0.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0260 | 0.75 | nan |
| abd_t2spir_t1out | 3 | -0.0223 | 0.33 | nan |
| abd_t2spir_ct | 5 | +0.0101 | 0.80 | 0.31 |
| mand_cbct_hanseg_ct | 14 | +0.0926 | 1.00 | 0.00012 |
| mand_cbct_pddca_ct | 20 | +0.0990 | 1.00 | 1.9e-06 |
| pelvis_ct_mri | 4 | +0.0032 | 0.50 | nan |
| pancreas_t1wce_t2w | 36 | +0.0051 | 0.64 | 0.064 |
| pancreas_t2w_t1wce | 38 | -0.0219 | 0.03 | 1.5e-11 |
pooled over cells: median of cell medians +0.0016, cells with d > 0: 10 / 19, p < 0.05 same sign: 3 +, 4 -

### CFC (flat fails): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 31 | -0.1195 | 0.00 | 9.3e-10 |
| glioma_t1n_t2w | 26 | -0.1176 | 0.00 | 3e-08 |
| glioma_t2w_t2f | 49 | -0.1050 | 0.00 | 3.6e-15 |
| glioma_t2w_t1n | 57 | -0.0954 | 0.00 | 5.1e-11 |
| ms_flair_t2w | 6 | -0.2029 | 0.00 | 0.031 |
| ms_flair_t1w | 5 | -0.0445 | 0.00 | 0.062 |
| ms_t1w_flair | 8 | -0.1237 | 0.00 | 0.0078 |
| ms_t1w_t2w | 8 | -0.1487 | 0.00 | 0.0078 |
| breast_t2w_t1wce | 11 | -0.1385 | 0.00 | 0.00098 |
| brain_t1w_t2w | 8 | -0.1123 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1240 | 0.00 | 0.0078 |
| abd_t2spir_t1in | 4 | -0.0715 | 0.00 | nan |
| abd_t2spir_t1out | 3 | -0.1237 | 0.00 | nan |
| abd_t2spir_ct | 5 | -0.0905 | 0.00 | 0.062 |
| mand_cbct_hanseg_ct | 14 | -0.2000 | 0.00 | 0.00012 |
| mand_cbct_pddca_ct | 20 | -0.2620 | 0.00 | 1.9e-06 |
| pelvis_ct_mri | 4 | -0.0568 | 0.00 | nan |
| pancreas_t1wce_t2w | 36 | -0.0413 | 0.03 | 5.8e-11 |
| pancreas_t2w_t1wce | 38 | -0.0632 | 0.00 | 7.3e-12 |
pooled over cells: median of cell medians -0.1176, cells with d > 0: 0 / 19, p < 0.05 same sign: 0 +, 14 -

### FCC (noise fails): real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.0026 | 0.86 | 3e-08 |
| glioma_t1n_t2w | 66 | +0.0046 | 0.83 | 3e-08 |
| glioma_t2w_t2f | 55 | +0.0013 | 0.65 | 0.012 |
| glioma_t2w_t1n | 48 | +0.0022 | 0.71 | 7.8e-05 |
| ms_flair_t2w | 7 | -0.0006 | 0.43 | 0.69 |
| ms_flair_t1w | 6 | -0.0161 | 0.17 | 0.094 |
| ms_t1w_flair | 5 | -0.0052 | 0.00 | 0.062 |
| ms_t1w_t2w | 6 | +0.0085 | 1.00 | 0.031 |
| breast_t1wce_t2w | 11 | -0.0001 | 0.45 | 0.97 |
| breast_t2w_t1wce | 18 | -0.0007 | 0.44 | 0.37 |
| brain_t1w_t2w | 8 | +0.0000 | 0.50 | 0.95 |
| brain_t2w_t1w | 8 | -0.0009 | 0.25 | 0.055 |
| abd_t1in_t2spir | 4 | -0.0034 | 0.25 | nan |
| abd_t1in_ct | 8 | -0.0044 | 0.25 | 0.11 |
| abd_t2spir_t1in | 4 | +0.0018 | 0.75 | nan |
| abd_t2spir_t1out | 4 | +0.0011 | 0.75 | nan |
| abd_t2spir_ct | 6 | +0.0009 | 0.67 | 1 |
| mand_cbct_hanseg_ct | 25 | +0.0025 | 0.80 | 0.0051 |
| mand_cbct_pddca_ct | 24 | +0.0017 | 0.75 | 0.056 |
| pelvis_mri_ct | 20 | -0.0009 | 0.25 | 0.053 |
| pelvis_ct_mri | 21 | -0.0005 | 0.48 | 0.76 |
| pancreas_t1wce_t2w | 36 | +0.0003 | 0.53 | 0.37 |
| pancreas_t2w_t1wce | 38 | -0.0002 | 0.50 | 0.74 |
pooled over cells: median of cell medians +0.0000, cells with d > 0: 12 / 23, p < 0.05 same sign: 6 +, 0 -

### FCC (noise fails): real - noise, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.0007 | 0.53 | 0.058 |
| glioma_t1n_t2w | 66 | +0.0067 | 0.91 | 2.6e-11 |
| glioma_t2w_t2f | 55 | +0.0016 | 0.67 | 0.006 |
| glioma_t2w_t1n | 48 | -0.0007 | 0.35 | 0.038 |
| ms_flair_t2w | 7 | -0.0134 | 0.43 | 0.22 |
| ms_flair_t1w | 6 | -0.0067 | 0.00 | 0.031 |
| ms_t1w_flair | 5 | -0.0073 | 0.00 | 0.062 |
| ms_t1w_t2w | 6 | -0.0023 | 0.33 | 0.16 |
| breast_t1wce_t2w | 11 | -0.0136 | 0.00 | 0.00098 |
| breast_t2w_t1wce | 18 | -0.0064 | 0.11 | 0.00014 |
| brain_t1w_t2w | 8 | -0.0134 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.0116 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0169 | 0.00 | nan |
| abd_t1in_ct | 8 | -0.0100 | 0.12 | 0.055 |
| abd_t2spir_t1in | 4 | +0.0012 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0027 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.0029 | 0.83 | 0.062 |
| mand_cbct_hanseg_ct | 25 | -0.0203 | 0.04 | 1.8e-07 |
| mand_cbct_pddca_ct | 24 | -0.0207 | 0.08 | 6e-07 |
| pelvis_mri_ct | 20 | -0.0026 | 0.30 | 0.09 |
| pelvis_ct_mri | 21 | +0.0049 | 0.81 | 0.0014 |
| pancreas_t1wce_t2w | 36 | +0.0001 | 0.53 | 0.86 |
| pancreas_t2w_t1wce | 38 | -0.0008 | 0.47 | 0.09 |
pooled over cells: median of cell medians -0.0026, cells with d > 0: 8 / 23, p < 0.05 same sign: 3 +, 8 -

### FCC (noise fails): real - flat, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.0256 | 0.95 | 1.4e-10 |
| glioma_t1n_t2w | 66 | +0.0035 | 0.64 | 0.024 |
| glioma_t2w_t2f | 55 | +0.0065 | 0.65 | 0.0038 |
| glioma_t2w_t1n | 48 | +0.0100 | 0.71 | 0.0025 |
| ms_flair_t2w | 7 | -0.0084 | 0.29 | 0.3 |
| ms_flair_t1w | 6 | -0.0172 | 0.17 | 0.44 |
| ms_t1w_flair | 5 | -0.0006 | 0.40 | 0.62 |
| ms_t1w_t2w | 6 | +0.0074 | 0.83 | 0.16 |
| breast_t1wce_t2w | 11 | +0.0196 | 0.73 | 0.083 |
| breast_t2w_t1wce | 18 | -0.0013 | 0.50 | 0.58 |
| brain_t1w_t2w | 8 | -0.0035 | 0.12 | 0.11 |
| brain_t2w_t1w | 8 | -0.0013 | 0.38 | 0.64 |
| abd_t1in_t2spir | 4 | +0.0001 | 0.50 | nan |
| abd_t1in_ct | 8 | +0.0288 | 0.75 | 0.15 |
| abd_t2spir_t1in | 4 | +0.0236 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0254 | 1.00 | nan |
| abd_t2spir_ct | 6 | -0.0332 | 0.33 | 0.16 |
| mand_cbct_hanseg_ct | 25 | +0.0335 | 1.00 | 6e-08 |
| mand_cbct_pddca_ct | 24 | +0.0563 | 1.00 | 1.2e-07 |
| pelvis_mri_ct | 20 | -0.0042 | 0.35 | 0.18 |
| pelvis_ct_mri | 21 | -0.0022 | 0.38 | 0.26 |
| pancreas_t1wce_t2w | 36 | +0.0106 | 0.81 | 2.5e-07 |
| pancreas_t2w_t1wce | 38 | +0.0011 | 0.58 | 0.4 |
pooled over cells: median of cell medians +0.0035, cells with d > 0: 14 / 23, p < 0.05 same sign: 7 +, 0 -

### FCC (noise fails): real - noise, inside_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.0258 | 0.88 | 8e-10 |
| glioma_t1n_t2w | 66 | +0.0219 | 0.82 | 4.8e-08 |
| glioma_t2w_t2f | 55 | -0.0099 | 0.35 | 0.0094 |
| glioma_t2w_t1n | 48 | -0.0066 | 0.38 | 0.11 |
| ms_flair_t2w | 7 | -0.0258 | 0.14 | 0.031 |
| ms_flair_t1w | 6 | +0.0017 | 0.50 | 0.84 |
| ms_t1w_flair | 5 | -0.0174 | 0.00 | 0.062 |
| ms_t1w_t2w | 6 | -0.0182 | 0.00 | 0.031 |
| breast_t1wce_t2w | 11 | -0.0304 | 0.09 | 0.0029 |
| breast_t2w_t1wce | 18 | -0.0296 | 0.00 | 7.6e-06 |
| brain_t1w_t2w | 8 | +0.1508 | 0.75 | 0.078 |
| brain_t2w_t1w | 8 | +0.0107 | 0.50 | 0.74 |
| abd_t1in_t2spir | 4 | -0.0154 | 0.00 | nan |
| abd_t1in_ct | 8 | +0.1355 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0798 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0452 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.0743 | 1.00 | 0.031 |
| mand_cbct_hanseg_ct | 25 | -0.1422 | 0.00 | 6e-08 |
| mand_cbct_pddca_ct | 24 | -0.1438 | 0.00 | 1.2e-07 |
| pelvis_mri_ct | 20 | -0.0196 | 0.20 | 0.0083 |
| pelvis_ct_mri | 21 | +0.0132 | 0.62 | 0.3 |
| pancreas_t1wce_t2w | 36 | -0.0130 | 0.19 | 9.2e-05 |
| pancreas_t2w_t1wce | 38 | -0.0063 | 0.37 | 0.031 |
pooled over cells: median of cell medians -0.0066, cells with d > 0: 10 / 23, p < 0.05 same sign: 4 +, 10 -

### FCC (noise fails): real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.0680 | 1.00 | 2.4e-11 |
| glioma_t1n_t2w | 66 | +0.0713 | 1.00 | 1.6e-12 |
| glioma_t2w_t2f | 55 | +0.0510 | 1.00 | 1.1e-10 |
| glioma_t2w_t1n | 48 | +0.0507 | 1.00 | 7.1e-15 |
| ms_flair_t2w | 7 | +0.0788 | 0.71 | 0.11 |
| ms_flair_t1w | 6 | +0.0454 | 0.83 | 0.22 |
| ms_t1w_flair | 5 | +0.0016 | 0.60 | 0.81 |
| ms_t1w_t2w | 6 | -0.0545 | 0.17 | 0.094 |
| breast_t1wce_t2w | 11 | +0.0285 | 0.82 | 0.0098 |
| breast_t2w_t1wce | 18 | +0.0809 | 1.00 | 7.6e-06 |
| brain_t1w_t2w | 8 | +0.0134 | 0.88 | 0.016 |
| brain_t2w_t1w | 8 | +0.0126 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0489 | 1.00 | nan |
| abd_t1in_ct | 8 | +0.0582 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | -0.0197 | 0.25 | nan |
| abd_t2spir_t1out | 4 | -0.0080 | 0.50 | nan |
| abd_t2spir_ct | 6 | -0.0401 | 0.17 | 0.062 |
| mand_cbct_hanseg_ct | 25 | +0.0320 | 1.00 | 6e-08 |
| mand_cbct_pddca_ct | 24 | +0.0308 | 1.00 | 1.2e-07 |
| pelvis_mri_ct | 20 | +0.0192 | 0.80 | 0.0027 |
| pelvis_ct_mri | 21 | +0.0331 | 0.86 | 2.4e-05 |
| pancreas_t1wce_t2w | 36 | +0.0389 | 0.97 | 7.3e-10 |
| pancreas_t2w_t1wce | 38 | +0.0173 | 0.84 | 6.6e-06 |
pooled over cells: median of cell medians +0.0320, cells with d > 0: 19 / 23, p < 0.05 same sign: 15 +, 0 -

### FCC (noise fails): real - noise, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.1518 | 1.00 | 2.4e-11 |
| glioma_t1n_t2w | 66 | +0.1469 | 1.00 | 1.6e-12 |
| glioma_t2w_t2f | 55 | +0.0922 | 1.00 | 1.1e-10 |
| glioma_t2w_t1n | 48 | +0.0816 | 1.00 | 7.1e-15 |
| ms_flair_t2w | 7 | -0.0063 | 0.43 | 0.94 |
| ms_flair_t1w | 6 | +0.0283 | 0.67 | 0.69 |
| ms_t1w_flair | 5 | +0.0333 | 0.60 | 1 |
| ms_t1w_t2w | 6 | +0.1566 | 1.00 | 0.031 |
| breast_t1wce_t2w | 11 | +0.0551 | 1.00 | 0.00098 |
| breast_t2w_t1wce | 18 | -0.0192 | 0.44 | 0.8 |
| brain_t1w_t2w | 8 | +0.0602 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0411 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0752 | 1.00 | nan |
| abd_t1in_ct | 8 | +0.1364 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0402 | 1.00 | nan |
| abd_t2spir_t1out | 4 | +0.0328 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.0422 | 1.00 | 0.031 |
| mand_cbct_hanseg_ct | 25 | -0.0171 | 0.20 | 0.00019 |
| mand_cbct_pddca_ct | 24 | -0.0152 | 0.04 | 2.4e-07 |
| pelvis_mri_ct | 20 | +0.0924 | 1.00 | 1.9e-06 |
| pelvis_ct_mri | 21 | +0.1338 | 0.95 | 1.9e-06 |
| pancreas_t1wce_t2w | 36 | -0.0015 | 0.44 | 0.41 |
| pancreas_t2w_t1wce | 38 | +0.0186 | 0.76 | 0.00047 |
pooled over cells: median of cell medians +0.0422, cells with d > 0: 18 / 23, p < 0.05 same sign: 13 +, 2 -

### FCC (noise fails): real - flat, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.0630 | 1.00 | 2.4e-11 |
| glioma_t1n_t2w | 66 | +0.0627 | 0.98 | 1.9e-12 |
| glioma_t2w_t2f | 55 | +0.0428 | 1.00 | 1.1e-10 |
| glioma_t2w_t1n | 48 | +0.0398 | 0.98 | 1.4e-14 |
| ms_flair_t2w | 7 | +0.0680 | 0.71 | 0.11 |
| ms_flair_t1w | 6 | +0.0479 | 0.83 | 0.22 |
| ms_t1w_flair | 5 | +0.0037 | 0.60 | 0.81 |
| ms_t1w_t2w | 6 | -0.0399 | 0.17 | 0.16 |
| breast_t1wce_t2w | 11 | +0.0358 | 0.82 | 0.024 |
| breast_t2w_t1wce | 18 | +0.0433 | 0.94 | 0.00033 |
| brain_t1w_t2w | 8 | +0.0097 | 0.88 | 0.016 |
| brain_t2w_t1w | 8 | +0.0086 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0408 | 1.00 | nan |
| abd_t1in_ct | 8 | +0.0514 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | -0.0175 | 0.50 | nan |
| abd_t2spir_t1out | 4 | +0.0112 | 0.75 | nan |
| abd_t2spir_ct | 6 | -0.0178 | 0.33 | 0.31 |
| mand_cbct_hanseg_ct | 25 | +0.0313 | 1.00 | 6e-08 |
| mand_cbct_pddca_ct | 24 | +0.0234 | 1.00 | 1.2e-07 |
| pelvis_mri_ct | 20 | +0.0274 | 0.80 | 0.014 |
| pelvis_ct_mri | 21 | +0.0313 | 0.76 | 0.00024 |
| pancreas_t1wce_t2w | 36 | +0.0390 | 0.97 | 5.9e-08 |
| pancreas_t2w_t1wce | 38 | +0.0203 | 0.76 | 0.00014 |
pooled over cells: median of cell medians +0.0313, cells with d > 0: 20 / 23, p < 0.05 same sign: 15 +, 0 -

### FCC (noise fails): real - noise, fine_local

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.1761 | 1.00 | 2.4e-11 |
| glioma_t1n_t2w | 66 | +0.1679 | 1.00 | 1.6e-12 |
| glioma_t2w_t2f | 55 | +0.0990 | 1.00 | 1.1e-10 |
| glioma_t2w_t1n | 48 | +0.0849 | 1.00 | 7.1e-15 |
| ms_flair_t2w | 7 | -0.0062 | 0.43 | 0.94 |
| ms_flair_t1w | 6 | +0.0307 | 0.67 | 0.56 |
| ms_t1w_flair | 5 | +0.0320 | 0.60 | 1 |
| ms_t1w_t2w | 6 | +0.1540 | 1.00 | 0.031 |
| breast_t1wce_t2w | 11 | +0.0377 | 0.91 | 0.0049 |
| breast_t2w_t1wce | 18 | -0.0173 | 0.39 | 0.8 |
| brain_t1w_t2w | 8 | +0.0531 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0439 | 1.00 | 0.0078 |
| abd_t1in_t2spir | 4 | +0.0798 | 1.00 | nan |
| abd_t1in_ct | 8 | +0.1328 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | +0.0372 | 0.75 | nan |
| abd_t2spir_t1out | 4 | +0.0439 | 1.00 | nan |
| abd_t2spir_ct | 6 | +0.0383 | 1.00 | 0.031 |
| mand_cbct_hanseg_ct | 25 | -0.0162 | 0.24 | 0.0013 |
| mand_cbct_pddca_ct | 24 | -0.0149 | 0.08 | 2.3e-06 |
| pelvis_mri_ct | 20 | +0.0433 | 0.80 | 0.0049 |
| pelvis_ct_mri | 21 | +0.1420 | 0.95 | 6.7e-06 |
| pancreas_t1wce_t2w | 36 | +0.0017 | 0.56 | 1 |
| pancreas_t2w_t1wce | 38 | +0.0270 | 0.82 | 5.4e-05 |
pooled over cells: median of cell medians +0.0433, cells with d > 0: 19 / 23, p < 0.05 same sign: 13 +, 2 -

### FCC (noise fails): real - flat, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | -0.0066 | 0.24 | 0.00027 |
| glioma_t1n_t2w | 66 | +0.0044 | 0.58 | 0.068 |
| glioma_t2w_t2f | 55 | -0.0023 | 0.44 | 0.28 |
| glioma_t2w_t1n | 48 | +0.0066 | 0.62 | 0.00086 |
| ms_flair_t2w | 7 | -0.0422 | 0.14 | 0.031 |
| ms_flair_t1w | 6 | -0.0287 | 0.00 | 0.031 |
| ms_t1w_flair | 5 | -0.0194 | 0.00 | 0.062 |
| ms_t1w_t2w | 6 | -0.0018 | 0.33 | 0.44 |
| breast_t1wce_t2w | 11 | +0.0702 | 1.00 | 0.00098 |
| breast_t2w_t1wce | 18 | -0.0142 | 0.28 | 0.099 |
| brain_t1w_t2w | 8 | -0.0051 | 0.25 | 0.38 |
| brain_t2w_t1w | 8 | -0.0194 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0016 | 0.50 | nan |
| abd_t1in_ct | 8 | -0.0124 | 0.25 | 0.055 |
| abd_t2spir_t1in | 4 | +0.0007 | 0.75 | nan |
| abd_t2spir_t1out | 4 | -0.0118 | 0.00 | nan |
| abd_t2spir_ct | 6 | -0.0057 | 0.17 | 0.44 |
| mand_cbct_hanseg_ct | 25 | +0.0824 | 1.00 | 6e-08 |
| mand_cbct_pddca_ct | 24 | +0.1013 | 1.00 | 1.2e-07 |
| pelvis_mri_ct | 20 | -0.0113 | 0.15 | 0.00026 |
| pelvis_ct_mri | 21 | +0.0245 | 0.86 | 0.00029 |
| pancreas_t1wce_t2w | 36 | +0.0048 | 0.64 | 0.16 |
| pancreas_t2w_t1wce | 38 | -0.0230 | 0.08 | 2.4e-10 |
pooled over cells: median of cell medians -0.0051, cells with d > 0: 8 / 23, p < 0.05 same sign: 5 +, 6 -

### FCC (noise fails): real - noise, align

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 59 | -0.1125 | 0.00 | 2.4e-11 |
| glioma_t1n_t2w | 66 | -0.1118 | 0.00 | 1.6e-12 |
| glioma_t2w_t2f | 55 | -0.1062 | 0.00 | 1.1e-10 |
| glioma_t2w_t1n | 48 | -0.0943 | 0.00 | 7.1e-15 |
| ms_flair_t2w | 7 | -0.1934 | 0.00 | 0.016 |
| ms_flair_t1w | 6 | -0.0565 | 0.00 | 0.031 |
| ms_t1w_flair | 5 | -0.1745 | 0.00 | 0.062 |
| ms_t1w_t2w | 6 | -0.1737 | 0.00 | 0.031 |
| breast_t1wce_t2w | 11 | -0.2389 | 0.00 | 0.00098 |
| breast_t2w_t1wce | 18 | -0.1307 | 0.00 | 7.6e-06 |
| brain_t1w_t2w | 8 | -0.1030 | 0.00 | 0.0078 |
| brain_t2w_t1w | 8 | -0.1181 | 0.00 | 0.0078 |
| abd_t1in_t2spir | 4 | -0.0952 | 0.00 | nan |
| abd_t1in_ct | 8 | -0.1109 | 0.00 | 0.0078 |
| abd_t2spir_t1in | 4 | -0.0816 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0823 | 0.00 | nan |
| abd_t2spir_ct | 6 | -0.0794 | 0.00 | 0.031 |
| mand_cbct_hanseg_ct | 25 | -0.1981 | 0.00 | 6e-08 |
| mand_cbct_pddca_ct | 24 | -0.2990 | 0.00 | 1.2e-07 |
| pelvis_mri_ct | 20 | -0.1165 | 0.00 | 1.9e-06 |
| pelvis_ct_mri | 21 | -0.0703 | 0.00 | 9.5e-07 |
| pancreas_t1wce_t2w | 36 | -0.0356 | 0.06 | 2.9e-10 |
| pancreas_t2w_t1wce | 38 | -0.0656 | 0.00 | 7.3e-12 |
pooled over cells: median of cell medians -0.1109, cells with d > 0: 0 / 23, p < 0.05 same sign: 0 +, 19 -


## 4. Ring background voxels (false-positive patterns; C = correctly background), own-decision gradients, fine_share and local_share

### ring FFC: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 34 | +0.0612 | 0.97 | 3.5e-10 |
| glioma_t1n_t2w | 30 | +0.0606 | 1.00 | 1.9e-09 |
| glioma_t2w_t2f | 52 | +0.0527 | 0.94 | 1e-09 |
| glioma_t2w_t1n | 41 | +0.0497 | 1.00 | 9.1e-13 |
| ms_flair_t2w | 8 | +0.0994 | 1.00 | 0.0078 |
| ms_flair_t1w | 6 | +0.0693 | 1.00 | 0.031 |
| ms_t1w_flair | 5 | +0.0469 | 0.80 | 0.31 |
| ms_t1w_t2w | 5 | +0.0817 | 0.80 | 0.12 |
| breast_t1wce_t2w | 17 | +0.0307 | 0.76 | 0.0067 |
| breast_t2w_t1wce | 20 | +0.0886 | 1.00 | 1.9e-06 |
| abd_t1in_ct | 8 | +0.0607 | 1.00 | 0.0078 |
| abd_t2spir_t1in | 4 | -0.0187 | 0.00 | nan |
| abd_t2spir_t1out | 3 | +0.0076 | 1.00 | nan |
| abd_t2spir_ct | 12 | -0.0315 | 0.17 | 0.0093 |
| mand_cbct_hanseg_ct | 3 | +0.0251 | 1.00 | nan |
| mand_cbct_pddca_ct | 5 | +0.0405 | 1.00 | 0.062 |
| pelvis_ct_mri | 7 | +0.0486 | 1.00 | 0.016 |
| pancreas_t1wce_t2w | 37 | +0.0419 | 0.97 | 1.6e-09 |
| pancreas_t2w_t1wce | 34 | +0.0281 | 0.94 | 1.2e-09 |
pooled over cells: median of cell medians +0.0486, cells with d > 0: 17 / 19, p < 0.05 same sign: 12 +, 1 -

### ring FFC: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 34 | +0.0004 | 0.50 | 0.88 |
| glioma_t1n_t2w | 30 | +0.0000 | 0.50 | 0.67 |
| glioma_t2w_t2f | 52 | +0.0010 | 0.54 | 0.51 |
| glioma_t2w_t1n | 41 | +0.0028 | 0.68 | 0.0096 |
| ms_flair_t2w | 8 | +0.0026 | 0.62 | 1 |
| ms_flair_t1w | 6 | +0.0025 | 0.83 | 0.16 |
| ms_t1w_flair | 5 | +0.0320 | 1.00 | 0.062 |
| ms_t1w_t2w | 5 | +0.0114 | 0.80 | 0.31 |
| breast_t1wce_t2w | 17 | -0.0011 | 0.35 | 0.24 |
| breast_t2w_t1wce | 20 | +0.0017 | 0.65 | 0.15 |
| abd_t1in_ct | 8 | -0.0111 | 0.25 | 0.039 |
| abd_t2spir_t1in | 4 | +0.0036 | 0.50 | nan |
| abd_t2spir_t1out | 3 | -0.0044 | 0.00 | nan |
| abd_t2spir_ct | 12 | +0.0011 | 0.67 | 0.73 |
| mand_cbct_hanseg_ct | 3 | -0.0054 | 0.00 | nan |
| mand_cbct_pddca_ct | 5 | -0.0009 | 0.40 | 0.81 |
| pelvis_ct_mri | 7 | -0.0025 | 0.14 | 0.078 |
| pancreas_t1wce_t2w | 37 | +0.0005 | 0.59 | 0.091 |
| pancreas_t2w_t1wce | 34 | +0.0009 | 0.59 | 0.2 |
pooled over cells: median of cell medians +0.0009, cells with d > 0: 13 / 19, p < 0.05 same sign: 1 +, 1 -

### ring FCF: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 40 | +0.0677 | 0.97 | 5.5e-12 |
| glioma_t1n_t2w | 19 | +0.0718 | 1.00 | 3.8e-06 |
| glioma_t2w_t2f | 46 | +0.0495 | 0.98 | 1.4e-13 |
| glioma_t2w_t1n | 37 | +0.0602 | 1.00 | 1.5e-11 |
| ms_flair_t2w | 5 | +0.0463 | 0.60 | 0.31 |
| ms_t1w_flair | 8 | -0.0114 | 0.50 | 0.64 |
| ms_t1w_t2w | 8 | -0.1116 | 0.12 | 0.023 |
| breast_t2w_t1wce | 7 | +0.0748 | 1.00 | 0.016 |
| mand_cbct_hanseg_ct | 6 | +0.0499 | 1.00 | 0.031 |
| mand_cbct_pddca_ct | 4 | +0.0518 | 1.00 | nan |
| pelvis_ct_mri | 5 | +0.0588 | 1.00 | 0.062 |
| pancreas_t1wce_t2w | 29 | +0.0368 | 0.97 | 2.8e-06 |
| pancreas_t2w_t1wce | 35 | +0.0197 | 0.69 | 0.00063 |
pooled over cells: median of cell medians +0.0499, cells with d > 0: 11 / 13, p < 0.05 same sign: 8 +, 1 -

### ring FCF: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 40 | +0.0047 | 0.85 | 2.3e-07 |
| glioma_t1n_t2w | 19 | +0.0026 | 0.63 | 0.12 |
| glioma_t2w_t2f | 46 | +0.0031 | 0.78 | 5.1e-07 |
| glioma_t2w_t1n | 37 | +0.0056 | 0.76 | 7.4e-06 |
| ms_flair_t2w | 5 | -0.0095 | 0.20 | 0.12 |
| ms_t1w_flair | 8 | +0.0016 | 0.62 | 0.95 |
| ms_t1w_t2w | 8 | +0.0024 | 0.62 | 0.74 |
| breast_t2w_t1wce | 7 | +0.0020 | 0.71 | 0.16 |
| mand_cbct_hanseg_ct | 6 | +0.0015 | 0.67 | 0.44 |
| mand_cbct_pddca_ct | 4 | +0.0036 | 1.00 | nan |
| pelvis_ct_mri | 5 | +0.0064 | 0.80 | 0.12 |
| pancreas_t1wce_t2w | 29 | -0.0001 | 0.48 | 0.51 |
| pancreas_t2w_t1wce | 35 | -0.0016 | 0.37 | 0.22 |
pooled over cells: median of cell medians +0.0024, cells with d > 0: 10 / 13, p < 0.05 same sign: 3 +, 0 -

### ring CFF: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 53 | +0.0672 | 1.00 | 2.4e-10 |
| glioma_t1n_t2w | 64 | +0.0621 | 0.98 | 3.9e-12 |
| glioma_t2w_t2f | 44 | +0.0544 | 0.98 | 2.3e-13 |
| glioma_t2w_t1n | 45 | +0.0530 | 1.00 | 5.7e-14 |
| ms_flair_t2w | 7 | +0.0349 | 0.86 | 0.3 |
| ms_flair_t1w | 4 | +0.0227 | 1.00 | nan |
| ms_t1w_flair | 5 | -0.0003 | 0.40 | 1 |
| ms_t1w_t2w | 5 | -0.0454 | 0.20 | 0.62 |
| breast_t1wce_t2w | 18 | +0.0471 | 0.89 | 5.3e-05 |
| breast_t2w_t1wce | 16 | +0.0517 | 1.00 | 3.1e-05 |
| abd_t1in_ct | 12 | +0.0576 | 1.00 | 0.00049 |
| abd_t2spir_t1in | 4 | -0.0017 | 0.25 | nan |
| abd_t2spir_t1out | 4 | -0.0220 | 0.25 | nan |
| abd_t2spir_ct | 15 | -0.0306 | 0.13 | 0.00031 |
| mand_cbct_hanseg_ct | 12 | +0.0530 | 1.00 | 0.00049 |
| mand_cbct_pddca_ct | 11 | +0.0556 | 1.00 | 0.00098 |
| pelvis_mri_ct | 4 | +0.0040 | 0.50 | nan |
| pelvis_ct_mri | 15 | +0.0398 | 1.00 | 6.1e-05 |
| pancreas_t1wce_t2w | 35 | +0.0420 | 1.00 | 5.8e-11 |
| pancreas_t2w_t1wce | 31 | +0.0191 | 0.71 | 0.0044 |
pooled over cells: median of cell medians +0.0409, cells with d > 0: 15 / 20, p < 0.05 same sign: 12 +, 1 -

### ring CFF: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 53 | +0.0046 | 0.81 | 8e-06 |
| glioma_t1n_t2w | 64 | +0.0032 | 0.77 | 7e-06 |
| glioma_t2w_t2f | 44 | +0.0019 | 0.64 | 0.046 |
| glioma_t2w_t1n | 45 | +0.0022 | 0.69 | 0.00052 |
| ms_flair_t2w | 7 | +0.0005 | 0.57 | 1 |
| ms_flair_t1w | 4 | -0.0116 | 0.00 | nan |
| ms_t1w_flair | 5 | +0.0017 | 0.60 | 1 |
| ms_t1w_t2w | 5 | +0.0121 | 0.80 | 0.19 |
| breast_t1wce_t2w | 18 | -0.0003 | 0.44 | 0.58 |
| breast_t2w_t1wce | 16 | +0.0012 | 0.75 | 0.14 |
| abd_t1in_ct | 12 | -0.0113 | 0.17 | 0.0049 |
| abd_t2spir_t1in | 4 | -0.0061 | 0.00 | nan |
| abd_t2spir_t1out | 4 | -0.0042 | 0.25 | nan |
| abd_t2spir_ct | 15 | -0.0038 | 0.47 | 0.14 |
| mand_cbct_hanseg_ct | 12 | -0.0001 | 0.42 | 0.73 |
| mand_cbct_pddca_ct | 11 | +0.0017 | 0.73 | 0.15 |
| pelvis_mri_ct | 4 | -0.0017 | 0.25 | nan |
| pelvis_ct_mri | 15 | +0.0014 | 0.73 | 0.012 |
| pancreas_t1wce_t2w | 35 | +0.0002 | 0.60 | 0.21 |
| pancreas_t2w_t1wce | 31 | +0.0002 | 0.52 | 0.81 |
pooled over cells: median of cell medians +0.0004, cells with d > 0: 12 / 20, p < 0.05 same sign: 5 +, 1 -

### ring CCF: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 65 | +0.0738 | 0.98 | 2.9e-12 |
| glioma_t1n_t2w | 56 | +0.0700 | 0.96 | 2.6e-10 |
| glioma_t2w_t2f | 46 | +0.0516 | 1.00 | 2.8e-14 |
| glioma_t2w_t1n | 47 | +0.0533 | 1.00 | 1.4e-14 |
| ms_flair_t2w | 8 | +0.0227 | 0.62 | 0.25 |
| ms_flair_t1w | 4 | -0.0234 | 0.25 | nan |
| ms_t1w_flair | 7 | -0.0417 | 0.29 | 0.38 |
| ms_t1w_t2w | 7 | -0.0735 | 0.14 | 0.031 |
| breast_t1wce_t2w | 11 | +0.0562 | 0.73 | 0.067 |
| breast_t2w_t1wce | 14 | +0.0745 | 1.00 | 0.00012 |
| abd_t1in_ct | 7 | +0.0732 | 1.00 | 0.016 |
| abd_t2spir_t1out | 3 | -0.0153 | 0.33 | nan |
| abd_t2spir_ct | 8 | -0.0357 | 0.12 | 0.11 |
| mand_cbct_hanseg_ct | 14 | +0.0583 | 1.00 | 0.00012 |
| mand_cbct_pddca_ct | 8 | +0.0517 | 1.00 | 0.0078 |
| pelvis_ct_mri | 15 | +0.0509 | 0.87 | 0.00031 |
| pancreas_t1wce_t2w | 33 | +0.0404 | 1.00 | 2.3e-10 |
| pancreas_t2w_t1wce | 39 | +0.0116 | 0.69 | 0.021 |
pooled over cells: median of cell medians +0.0513, cells with d > 0: 13 / 18, p < 0.05 same sign: 11 +, 1 -

### ring CCF: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 65 | +0.0056 | 0.91 | 2.6e-09 |
| glioma_t1n_t2w | 56 | +0.0053 | 0.88 | 3.1e-07 |
| glioma_t2w_t2f | 46 | +0.0026 | 0.72 | 7.8e-05 |
| glioma_t2w_t1n | 47 | +0.0039 | 0.91 | 7e-09 |
| ms_flair_t2w | 8 | +0.0014 | 0.50 | 0.46 |
| ms_flair_t1w | 4 | -0.0015 | 0.50 | nan |
| ms_t1w_flair | 7 | +0.0055 | 0.57 | 1 |
| ms_t1w_t2w | 7 | +0.0057 | 0.71 | 0.38 |
| breast_t1wce_t2w | 11 | +0.0025 | 0.64 | 0.46 |
| breast_t2w_t1wce | 14 | +0.0008 | 0.64 | 0.46 |
| abd_t1in_ct | 7 | +0.0007 | 0.57 | 1 |
| abd_t2spir_t1out | 3 | +0.0005 | 0.67 | nan |
| abd_t2spir_ct | 8 | +0.0009 | 0.62 | 0.55 |
| mand_cbct_hanseg_ct | 14 | +0.0004 | 0.64 | 0.33 |
| mand_cbct_pddca_ct | 8 | +0.0029 | 1.00 | 0.0078 |
| pelvis_ct_mri | 15 | +0.0038 | 0.87 | 0.018 |
| pancreas_t1wce_t2w | 33 | +0.0007 | 0.70 | 0.09 |
| pancreas_t2w_t1wce | 39 | -0.0015 | 0.28 | 0.0047 |
pooled over cells: median of cell medians +0.0019, cells with d > 0: 16 / 18, p < 0.05 same sign: 6 +, 1 -

### ring CFC: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 40 | +0.0615 | 0.97 | 3.6e-12 |
| glioma_t1n_t2w | 39 | +0.0568 | 1.00 | 3.6e-12 |
| glioma_t2w_t2f | 38 | +0.0601 | 1.00 | 7.3e-12 |
| glioma_t2w_t1n | 39 | +0.0529 | 0.97 | 2.5e-11 |
| ms_flair_t2w | 7 | +0.0737 | 1.00 | 0.016 |
| ms_flair_t1w | 4 | +0.0365 | 0.75 | nan |
| ms_t1w_flair | 7 | +0.0909 | 0.86 | 0.11 |
| ms_t1w_t2w | 5 | -0.0602 | 0.00 | 0.062 |
| breast_t1wce_t2w | 19 | +0.0545 | 0.95 | 2.7e-05 |
| breast_t2w_t1wce | 18 | +0.0692 | 1.00 | 7.6e-06 |
| abd_t1in_ct | 7 | +0.0638 | 1.00 | 0.016 |
| abd_t2spir_t1in | 4 | -0.0323 | 0.00 | nan |
| abd_t2spir_t1out | 4 | +0.0036 | 0.75 | nan |
| abd_t2spir_ct | 19 | -0.0224 | 0.16 | 0.00097 |
| mand_cbct_hanseg_ct | 8 | +0.0365 | 1.00 | 0.0078 |
| mand_cbct_pddca_ct | 13 | +0.0402 | 1.00 | 0.00024 |
| pelvis_ct_mri | 11 | +0.0362 | 0.91 | 0.002 |
| pancreas_t1wce_t2w | 35 | +0.0337 | 0.94 | 2.9e-10 |
| pancreas_t2w_t1wce | 37 | +0.0204 | 0.81 | 1.4e-05 |
pooled over cells: median of cell medians +0.0402, cells with d > 0: 16 / 19, p < 0.05 same sign: 13 +, 1 -

### ring CFC: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 40 | +0.0021 | 0.70 | 0.0076 |
| glioma_t1n_t2w | 39 | +0.0020 | 0.59 | 0.082 |
| glioma_t2w_t2f | 38 | +0.0006 | 0.53 | 0.64 |
| glioma_t2w_t1n | 39 | +0.0028 | 0.69 | 0.046 |
| ms_flair_t2w | 7 | +0.0049 | 0.71 | 0.078 |
| ms_flair_t1w | 4 | +0.0067 | 0.75 | nan |
| ms_t1w_flair | 7 | +0.0098 | 0.71 | 0.11 |
| ms_t1w_t2w | 5 | +0.0197 | 1.00 | 0.062 |
| breast_t1wce_t2w | 19 | -0.0015 | 0.21 | 0.0028 |
| breast_t2w_t1wce | 18 | +0.0002 | 0.56 | 0.83 |
| abd_t1in_ct | 7 | -0.0052 | 0.00 | 0.016 |
| abd_t2spir_t1in | 4 | -0.0009 | 0.50 | nan |
| abd_t2spir_t1out | 4 | +0.0005 | 0.50 | nan |
| abd_t2spir_ct | 19 | +0.0005 | 0.63 | 0.8 |
| mand_cbct_hanseg_ct | 8 | +0.0000 | 0.50 | 0.95 |
| mand_cbct_pddca_ct | 13 | +0.0003 | 0.54 | 0.68 |
| pelvis_ct_mri | 11 | -0.0024 | 0.36 | 0.083 |
| pancreas_t1wce_t2w | 35 | +0.0003 | 0.57 | 0.58 |
| pancreas_t2w_t1wce | 37 | +0.0008 | 0.54 | 0.77 |
pooled over cells: median of cell medians +0.0005, cells with d > 0: 15 / 19, p < 0.05 same sign: 2 +, 2 -

### ring FCC: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 41 | +0.0672 | 0.98 | 1.8e-12 |
| glioma_t1n_t2w | 51 | +0.0676 | 0.98 | 5.5e-10 |
| glioma_t2w_t2f | 54 | +0.0574 | 0.98 | 1.7e-10 |
| glioma_t2w_t1n | 63 | +0.0535 | 0.98 | 5.4e-12 |
| ms_flair_t2w | 8 | +0.1071 | 1.00 | 0.0078 |
| ms_flair_t1w | 6 | +0.0488 | 0.83 | 0.22 |
| ms_t1w_flair | 8 | -0.0055 | 0.38 | 0.95 |
| ms_t1w_t2w | 8 | -0.0123 | 0.38 | 0.64 |
| breast_t1wce_t2w | 9 | +0.0577 | 1.00 | 0.0039 |
| breast_t2w_t1wce | 22 | +0.0703 | 0.95 | 9.5e-07 |
| abd_t1in_t2spir | 3 | +0.0707 | 1.00 | nan |
| abd_t1in_ct | 14 | +0.0593 | 1.00 | 0.00012 |
| abd_t2spir_t1in | 3 | -0.0460 | 0.00 | nan |
| abd_t2spir_t1out | 3 | +0.0021 | 0.67 | nan |
| abd_t2spir_ct | 14 | -0.0547 | 0.14 | 0.00061 |
| mand_cbct_hanseg_ct | 7 | +0.0399 | 1.00 | 0.016 |
| mand_cbct_pddca_ct | 7 | +0.0284 | 1.00 | 0.016 |
| pelvis_mri_ct | 7 | +0.0148 | 0.71 | 0.3 |
| pelvis_ct_mri | 9 | +0.0510 | 0.89 | 0.0078 |
| pancreas_t1wce_t2w | 36 | +0.0358 | 0.97 | 5.8e-11 |
| pancreas_t2w_t1wce | 38 | +0.0269 | 0.79 | 0.00016 |
pooled over cells: median of cell medians +0.0488, cells with d > 0: 17 / 21, p < 0.05 same sign: 13 +, 1 -

### ring FCC: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 41 | +0.0032 | 0.71 | 0.00048 |
| glioma_t1n_t2w | 51 | +0.0018 | 0.71 | 0.00018 |
| glioma_t2w_t2f | 54 | +0.0014 | 0.63 | 0.058 |
| glioma_t2w_t1n | 63 | +0.0017 | 0.78 | 3.2e-06 |
| ms_flair_t2w | 8 | -0.0011 | 0.12 | 0.039 |
| ms_flair_t1w | 6 | +0.0019 | 0.67 | 1 |
| ms_t1w_flair | 8 | +0.0022 | 0.50 | 0.74 |
| ms_t1w_t2w | 8 | -0.0026 | 0.38 | 0.84 |
| breast_t1wce_t2w | 9 | +0.0008 | 0.56 | 1 |
| breast_t2w_t1wce | 22 | +0.0004 | 0.55 | 0.57 |
| abd_t1in_t2spir | 3 | -0.0014 | 0.33 | nan |
| abd_t1in_ct | 14 | -0.0034 | 0.29 | 0.15 |
| abd_t2spir_t1in | 3 | +0.0094 | 1.00 | nan |
| abd_t2spir_t1out | 3 | +0.0024 | 0.67 | nan |
| abd_t2spir_ct | 14 | +0.0042 | 0.64 | 0.091 |
| mand_cbct_hanseg_ct | 7 | +0.0002 | 0.71 | 0.69 |
| mand_cbct_pddca_ct | 7 | +0.0022 | 0.86 | 0.031 |
| pelvis_mri_ct | 7 | -0.0004 | 0.43 | 0.38 |
| pelvis_ct_mri | 9 | +0.0002 | 0.56 | 0.43 |
| pancreas_t1wce_t2w | 36 | +0.0009 | 0.64 | 0.013 |
| pancreas_t2w_t1wce | 38 | +0.0009 | 0.58 | 0.4 |
pooled over cells: median of cell medians +0.0009, cells with d > 0: 16 / 21, p < 0.05 same sign: 5 +, 1 -

### ring FFF: real - flat, fine_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 63 | +0.0582 | 0.94 | 5.7e-11 |
| glioma_t1n_t2w | 68 | +0.0488 | 0.91 | 2e-11 |
| glioma_t2w_t2f | 67 | +0.0455 | 0.99 | 1.7e-12 |
| glioma_t2w_t1n | 63 | +0.0498 | 1.00 | 5.2e-12 |
| ms_flair_t2w | 8 | +0.0128 | 0.62 | 0.38 |
| ms_flair_t1w | 5 | -0.0174 | 0.40 | 0.81 |
| ms_t1w_flair | 8 | -0.0373 | 0.25 | 0.64 |
| ms_t1w_t2w | 8 | -0.0336 | 0.38 | 0.55 |
| breast_t1wce_t2w | 40 | +0.0331 | 0.82 | 4.1e-05 |
| breast_t2w_t1wce | 39 | +0.0641 | 1.00 | 3.6e-12 |
| brain_t1w_t2w | 8 | +0.0230 | 1.00 | 0.0078 |
| brain_t2w_t1w | 8 | +0.0086 | 0.88 | 0.016 |
| abd_t1in_t2spir | 4 | +0.0358 | 1.00 | nan |
| abd_t1in_ct | 20 | +0.0511 | 1.00 | 1.9e-06 |
| abd_t2spir_t1in | 4 | +0.0019 | 0.50 | nan |
| abd_t2spir_t1out | 4 | -0.0104 | 0.00 | nan |
| abd_t2spir_ct | 20 | -0.0399 | 0.05 | 3.8e-06 |
| mand_cbct_hanseg_ct | 40 | +0.0322 | 0.95 | 1.8e-11 |
| mand_cbct_pddca_ct | 40 | +0.0322 | 1.00 | 1.8e-12 |
| pelvis_mri_ct | 25 | +0.0189 | 0.84 | 7.5e-05 |
| pelvis_ct_mri | 20 | +0.0419 | 0.95 | 3.8e-06 |
| pancreas_t1wce_t2w | 39 | +0.0366 | 0.97 | 2.5e-11 |
| pancreas_t2w_t1wce | 38 | +0.0179 | 0.79 | 0.0016 |
pooled over cells: median of cell medians +0.0322, cells with d > 0: 18 / 23, p < 0.05 same sign: 15 +, 1 -

### ring FFF: real - flat, local_share

| cell | n cases | median d | share > 0 | p |
|---|---|---|---|---|
| glioma_t1n_t2f | 63 | +0.0050 | 0.63 | 0.0021 |
| glioma_t1n_t2w | 68 | -0.0001 | 0.49 | 0.84 |
| glioma_t2w_t2f | 67 | -0.0006 | 0.45 | 0.15 |
| glioma_t2w_t1n | 63 | +0.0026 | 0.59 | 0.0016 |
| ms_flair_t2w | 8 | -0.0131 | 0.25 | 0.15 |
| ms_flair_t1w | 5 | -0.0113 | 0.00 | 0.062 |
| ms_t1w_flair | 8 | -0.0013 | 0.38 | 0.95 |
| ms_t1w_t2w | 8 | +0.0014 | 0.50 | 0.55 |
| breast_t1wce_t2w | 40 | -0.0009 | 0.47 | 0.077 |
| breast_t2w_t1wce | 39 | -0.0004 | 0.46 | 0.066 |
| brain_t1w_t2w | 8 | +0.0023 | 0.62 | 0.38 |
| brain_t2w_t1w | 8 | +0.0000 | 0.50 | 0.95 |
| abd_t1in_t2spir | 4 | +0.0029 | 0.75 | nan |
| abd_t1in_ct | 20 | -0.0059 | 0.20 | 0.00085 |
| abd_t2spir_t1in | 4 | +0.0009 | 0.50 | nan |
| abd_t2spir_t1out | 4 | -0.0038 | 0.50 | nan |
| abd_t2spir_ct | 20 | +0.0011 | 0.55 | 0.35 |
| mand_cbct_hanseg_ct | 40 | -0.0001 | 0.50 | 0.36 |
| mand_cbct_pddca_ct | 40 | -0.0003 | 0.47 | 0.96 |
| pelvis_mri_ct | 25 | -0.0009 | 0.48 | 0.98 |
| pelvis_ct_mri | 20 | +0.0017 | 0.60 | 0.28 |
| pancreas_t1wce_t2w | 39 | -0.0001 | 0.49 | 0.72 |
| pancreas_t2w_t1wce | 38 | +0.0004 | 0.55 | 0.73 |
pooled over cells: median of cell medians -0.0001, cells with d > 0: 10 / 23, p < 0.05 same sign: 2 +, 1 -

