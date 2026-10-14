# Grain advantage vs Dice gain of real fill, per cell

Grain delta = median over cases of grain(real) - grain(flat or noise) (fine-detail share of the input gradient of the arm's own decision; p2 = stride-4 path, p3 = stride-8 path, full = whole gradient), from `pathgrad_depth_localfine_cells.csv` (p2/p3, reproduced exactly from `<cell>/seeds.csv`) and `<cell>/seeds.csv` (full). Dice delta = mean over the grain cases of Dice(real) - Dice(flat or noise), Dice per case = mean over its labels and folds 0-2 (ladder unit), x100. Sources per arm: `grain_vs_dice_sources.csv`.

## Per cell

| cell | kind | n | Dice noise | Dice flat | Dice real | dDice r-f (p) | dDice r-n (p) | grain r-f p2 | p3 | full | grain r-n p2 | p3 | full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | appe | 70 | 29.7 | 33.9 | 41.2 | +7.24 (2e-10) | +11.41 (1e-12) | +0.065 | +0.068 | +0.077 | +0.125 | +0.097 | +0.133 |
| glioma_t1n_t2w | appe | 70 | 38.4 | 43.5 | 45.5 | +1.97 (0.0003) | +7.12 (1e-10) | +0.068 | +0.073 | +0.067 | +0.111 | +0.098 | +0.115 |
| glioma_t2w_t2f | appe | 69 | 45.2 | 46.2 | 48.3 | +2.19 (0.001) | +3.12 (0.0003) | +0.051 | +0.059 | +0.051 | +0.076 | +0.078 | +0.087 |
| glioma_t2w_t1n | appe | 70 | 41.5 | 33.4 | 37.0 | +3.68 (0.0001) | -4.42 (0.0003) | +0.051 | +0.058 | +0.054 | +0.061 | +0.065 | +0.079 |
| ms_flair_t2w | appe | 8 | 20.7 | 22.2 | 32.8 | +10.58 (0.008) | +12.01 (0.008) | +0.071 | +0.071 | +0.061 | +0.049 | +0.046 | -0.011 |
| ms_flair_t1w | appe | 8 | 12.0 | 18.2 | 21.8 | +3.55 (0.1) | +9.77 (0.02) | +0.039 | +0.046 | +0.036 | +0.016 | +0.014 | +0.010 |
| ms_t1w_flair | appe | 8 | 34.6 | 33.2 | 41.9 | +8.73 (0.008) | +7.34 (0.008) | -0.011 | -0.035 | +0.001 | +0.175 | +0.137 | +0.108 |
| ms_t1w_t2w | appe | 8 | 30.3 | 26.1 | 38.2 | +12.09 (0.008) | +7.82 (0.008) | -0.049 | -0.075 | -0.025 | +0.157 | +0.130 | +0.125 |
| breast_t1wce_t2w | appe | 40 | 29.7 | 29.7 | 30.6 | +0.89 (0.09) | +0.87 (0.08) | +0.078 | +0.054 | +0.046 | +0.220 | +0.214 | +0.048 |
| breast_t2w_t1wce | appe | 40 | 49.1 | 48.5 | 53.0 | +4.44 (4e-05) | +3.86 (0.0008) | +0.065 | +0.086 | +0.080 | -0.018 | +0.002 | -0.018 |
| brain_t1w_t2w | inte | 8 | 87.6 | 87.6 | 87.6 | +0.03 (0.2) | +0.01 (1) | +0.022 | +0.032 | +0.013 | +0.099 | +0.106 | +0.050 |
| brain_t2w_t1w | inte | 8 | 90.4 | 89.3 | 89.7 | +0.37 (0.008) | -0.70 (0.008) | +0.020 | +0.026 | +0.014 | +0.037 | +0.028 | +0.032 |
| abd_t1in_t2spir | inte | 4 | 86.2 | 87.5 | 87.7 | +0.29 () | +1.58 () | +0.055 | +0.069 | +0.042 | +0.173 | +0.150 | +0.104 |
| abd_t1in_ct | inte | 20 | 88.6 | 90.5 | 90.8 | +0.29 (0.3) | +2.25 (2e-06) | +0.076 | +0.086 | +0.066 | +0.171 | +0.181 | +0.138 |
| abd_t2spir_t1in | inte | 4 | 74.6 | 83.4 | 83.6 | +0.19 () | +8.93 () | -0.020 | -0.024 | -0.020 | +0.034 | +0.030 | +0.025 |
| abd_t2spir_t1out | inte | 4 | 80.3 | 86.2 | 87.7 | +1.50 () | +7.33 () | -0.003 | -0.016 | -0.003 | +0.029 | +0.023 | +0.031 |
| abd_t2spir_ct | inte | 20 | 87.6 | 88.7 | 88.9 | +0.19 (0.5) | +1.31 (0.04) | -0.033 | -0.052 | -0.038 | +0.037 | +0.030 | +0.023 |
| mand_cbct_hanseg_ct | inte | 40 | 83.7 | 83.4 | 83.6 | +0.16 (0.5) | -0.07 (0.4) | +0.045 | +0.056 | +0.036 | -0.030 | -0.054 | -0.021 |
| mand_cbct_pddca_ct | inte | 40 | 80.8 | 81.6 | 82.4 | +0.77 (0.02) | +1.66 (8e-06) | +0.047 | +0.062 | +0.037 | -0.026 | -0.043 | -0.022 |
| pelvis_mri_ct | inte | 26 | 73.1 | 77.0 | 73.2 | -3.76 (2e-05) | +0.06 (0.4) | +0.013 | +0.006 | +0.019 | +0.131 | +0.188 | +0.109 |
| pelvis_ct_mri | inte | 24 | 37.8 | 41.2 | 43.9 | +2.63 (0.03) | +6.11 (0.004) | +0.035 | +0.035 | +0.038 | +0.119 | +0.109 | +0.122 |
| pancreas_t1wce_t2w | inte | 40 | 72.2 | 69.1 | 68.9 | -0.19 (1) | -3.21 (1e-07) | +0.034 | +0.040 | +0.038 | +0.001 | -0.006 | -0.002 |
| pancreas_t2w_t1wce | inte | 40 | 72.0 | 72.0 | 72.0 | -0.03 (0.01) | -0.04 (0.2) | +0.025 | +0.026 | +0.023 | +0.022 | +0.006 | +0.024 |

Glioma, SNFH only (the grain target class): glioma_t1n_t2f dDice r-f +14.94, r-n +26.55; glioma_t1n_t2w dDice r-f +5.57, r-n +16.25; glioma_t2w_t2f dDice r-f +3.58, r-n +3.45; glioma_t2w_t1n dDice r-f +5.06, r-n -11.03

## Correlations (grain delta vs Dice delta)

| comparison | path | unit | n | Spearman (p) | Pearson (p) | leave-one-task-out Spearman |
|---|---|---|---|---|---|---|
| real - flat | p2 | cell | 23 | +0.22 (0.31) | -0.10 (0.66) | +0.11 .. +0.52 |
| real - flat | p2 | task mean | 8 | +0.00 (1) | -0.01 (0.98) |  |
| real - flat | p2 | cell, Dice on all test cases | 23 | +0.22 (0.3) | -0.12 (0.6) |  |
| real - flat | p3 | cell | 23 | +0.20 (0.35) | -0.17 (0.43) | +0.08 .. +0.51 |
| real - flat | p3 | task mean | 8 | +0.05 (0.91) | -0.18 (0.67) |  |
| real - flat | p3 | cell, Dice on all test cases | 23 | +0.20 (0.35) | -0.20 (0.37) |  |
| real - flat | full | cell | 23 | +0.30 (0.16) | +0.06 (0.79) |  |
| real - flat | full | task mean | 8 | +0.19 (0.65) | +0.12 (0.77) |  |
| real - flat | full | cell, Dice on all test cases | 23 | +0.30 (0.16) | +0.04 (0.87) |  |
| real - noise | p2 | cell | 23 | +0.18 (0.42) | +0.13 (0.54) | +0.02 .. +0.34 |
| real - noise | p2 | task mean | 8 | +0.57 (0.14) | +0.58 (0.13) |  |
| real - noise | p2 | cell, Dice on all test cases | 23 | +0.16 (0.46) | +0.14 (0.54) |  |
| real - noise | p3 | cell | 23 | +0.15 (0.51) | +0.06 (0.8) | -0.05 .. +0.30 |
| real - noise | p3 | task mean | 8 | +0.50 (0.21) | +0.46 (0.25) |  |
| real - noise | p3 | cell, Dice on all test cases | 23 | +0.13 (0.54) | +0.06 (0.77) |  |
| real - noise | full | cell | 23 | +0.20 (0.36) | +0.16 (0.46) |  |
| real - noise | full | task mean | 8 | +0.67 (0.071) | +0.48 (0.23) |  |
| real - noise | full | cell, Dice on all test cases | 23 | +0.20 (0.36) | +0.17 (0.43) |  |

Cells are not independent: 23 cells come from 17 models; the 4 glioma cells share 2 models and one test cohort, the abdomen cells 2 models. The task-mean rows (n = 8) are the conservative unit.

## Sign agreement (cell level, Dice on the grain cases)

- real - flat, p2: grain+/Dice+ 15, grain+/Dice- 3, grain-/Dice+ 5, grain-/Dice- 0 (agree 15/23, Fisher p 1)
- real - flat, p3: grain+/Dice+ 15, grain+/Dice- 3, grain-/Dice+ 5, grain-/Dice- 0 (agree 15/23, Fisher p 1)
- real - flat, full: grain+/Dice+ 16, grain+/Dice- 3, grain-/Dice+ 4, grain-/Dice- 0 (agree 16/23, Fisher p 1)
- real - noise, p2: grain+/Dice+ 16, grain+/Dice- 4, grain-/Dice+ 2, grain-/Dice- 1 (agree 17/23, Fisher p 0.54)
- real - noise, p3: grain+/Dice+ 17, grain+/Dice- 3, grain-/Dice+ 1, grain-/Dice- 2 (agree 19/23, Fisher p 0.11)
- real - noise, full: grain+/Dice+ 15, grain+/Dice- 3, grain-/Dice+ 3, grain-/Dice- 2 (agree 17/23, Fisher p 0.29)
