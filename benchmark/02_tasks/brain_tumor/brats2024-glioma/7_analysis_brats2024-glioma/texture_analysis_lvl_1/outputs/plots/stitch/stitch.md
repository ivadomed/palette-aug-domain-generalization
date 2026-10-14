# Model stitching flat <-> real (FINDINGS 40, pre-registered: PREREG.md)

Non-calibration cases per cell: glioma_t1n_t2f 50, glioma_t1n_t2w 50, glioma_t2w_t2f 50, glioma_t2w_t1n 50. Stitch = ridge W'x + b fitted on body voxels of the first 20 cases (label-free, lambda = 1e-3 tr(Sxx)/C). Dice = target Dice on the plan patch, fold-mean softmax over the 3 fold pairs. FR = flat source -> real target, RF = real -> flat, FF / RR = cross-fold self-stitch (k -> k+1); pen = parent - self-stitch Dice (median over cases). g = (FR - flat)/(real - flat): per-case median over cases with |real - flat| > 0.02, and cell-level (medians) when |median gap| > 0.05. g_RF = (real - RF)/(real - flat) = fraction of the gap LOST by R->F. g_corr = g with the stitch penalty of the target network added back (FR + (real - RR), RF - (flat - FF)). R2 columns = in-sample calibration R^2 (mean over folds; E = mean over s0..s5).

## Verdicts (Wilcoxon over cases reported; thresholds on medians as pre-registered; variant b is exploratory)

| cell | variant | P1 | P2 | P3 |
|---|---|---|---|---|
| glioma_t1n_t2f | full | FAILS | KILLED | HOLDS |
| glioma_t1n_t2w | full | FAILS | KILLED | FAILS |
| glioma_t2w_t2f | full | FAILS | NOT EVALUABLE | NOT EVALUABLE |
| glioma_t2w_t1n | full | FAILS | NOT EVALUABLE | NOT EVALUABLE |

## glioma_t1n_t2f: parents flat 0.408 / real 0.622 (median gap +0.214, n = 50, cases with |gap| > 0.02: 45)

### map applied to the whole patch (pre-registered)

| loc | FR | RF | FF | RR | pen FF | pen RR | g FR (case) | g FR (cell) | g_corr FR (cell) | g_RF loss (cell) | p FR-flat | p FR-real | p FF-flat | p RR-real | p RF-real | R2 FR / RF / FF / RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | 0.357 | 0.454 | 0.361 | 0.464 | +0.037 | +0.124 | -0.02 | -0.24 | 0.50 | 0.78 | 0.4 | 4e-09 | 0.0003 | 6e-09 | 2e-09 | 0.54 / 0.54 / 0.55 / 0.54 |
| D0 | 0.577 | 0.356 | 0.341 | 0.604 | +0.050 | +0.023 | 0.85 | 0.79 | 0.88 | 1.24 | 6e-09 | 0.0001 | 4e-08 | 4e-05 | 2e-09 | 0.66 / 0.68 / 0.69 / 0.65 |
| D1 | 0.531 | 0.438 | 0.328 | 0.558 | +0.058 | +0.040 | 0.55 | 0.57 | 0.87 | 0.86 | 6e-07 | 6e-06 | 1e-05 | 0.0001 | 1e-09 | 0.41 / 0.41 / 0.41 / 0.41 |
| D2 | 0.522 | 0.524 | 0.397 | 0.596 | -0.001 | +0.000 | 0.52 | 0.53 | 0.65 | 0.46 | 3e-07 | 4e-05 | 0.6 | 0.6 | 3e-09 | 0.36 / 0.35 / 0.34 / 0.37 |
| D3 | 0.500 | 0.615 | 0.430 | 0.641 | -0.028 | -0.021 | 0.42 | 0.43 | 0.34 | 0.03 | 3e-07 | 4e-06 | 0.0002 | 0.002 | 0.0008 | 0.38 / 0.36 / 0.35 / 0.39 |
| D4 | 0.363 | 0.331 | 0.298 | 0.524 | +0.094 | +0.087 | -0.16 | -0.21 | 0.25 | 1.36 | 3e-08 | 2e-09 | 2e-09 | 2e-09 | 2e-09 | 0.35 / 0.31 / 0.30 / 0.37 |

- P1 (self-stitch penalty < 0.03 at every location): FAILS (worst median penalty 0.124)
- P2 (late re-wiring): KILLED (g D4 -0.21, D3 0.43, FR_D4 - real -0.259, p 2e-09; penalty-corrected g D4 0.25); secondary: E recovers -0.24
- P3 (symmetry): HOLDS (R->F at D4 loses 1.36 of the gap; penalty-corrected 1.87)

### Stitch fit R^2 per location: in-sample (calibration, mean over folds) / held-out (non-calibration cases, 1 - sum SSE / sum SST pooled over cases and folds)

| pair | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FR | 0.75 / 0.75 | 0.51 / 0.50 | 0.43 / 0.42 | 0.26 / 0.24 | 0.42 / 0.30 | 0.84 / 0.64 | 0.66 / 0.51 | 0.41 / 0.36 | 0.36 / 0.33 | 0.38 / 0.34 | 0.35 / 0.30 |
| RF | 0.76 / 0.75 | 0.52 / 0.51 | 0.43 / 0.43 | 0.27 / 0.25 | 0.43 / 0.30 | 0.86 / 0.68 | 0.68 / 0.55 | 0.41 / 0.36 | 0.35 / 0.32 | 0.36 / 0.32 | 0.31 / 0.26 |
| FF | 0.73 / 0.72 | 0.53 / 0.52 | 0.45 / 0.44 | 0.27 / 0.26 | 0.45 / 0.32 | 0.86 / 0.69 | 0.69 / 0.56 | 0.41 / 0.37 | 0.34 / 0.32 | 0.35 / 0.32 | 0.30 / 0.24 |
| RR | 0.77 / 0.77 | 0.54 / 0.53 | 0.43 / 0.42 | 0.26 / 0.24 | 0.41 / 0.28 | 0.84 / 0.67 | 0.65 / 0.50 | 0.41 / 0.36 | 0.37 / 0.33 | 0.39 / 0.35 | 0.37 / 0.32 |

## glioma_t1n_t2w: parents flat 0.665 / real 0.724 (median gap +0.059, n = 50, cases with |gap| > 0.02: 42)

### map applied to the whole patch (pre-registered)

| loc | FR | RF | FF | RR | pen FF | pen RR | g FR (case) | g FR (cell) | g_corr FR (cell) | g_RF loss (cell) | p FR-flat | p FR-real | p FF-flat | p RR-real | p RF-real | R2 FR / RF / FF / RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | 0.606 | 0.644 | 0.610 | 0.664 | +0.090 | +0.084 | -1.16 | -1.00 | 0.02 | 1.36 | 2e-13 | 1e-09 | 2e-13 | 2e-09 | 1e-09 | 0.55 / 0.56 / 0.56 / 0.56 |
| D0 | 0.717 | 0.664 | 0.661 | 0.715 | +0.014 | +0.013 | 0.77 | 0.88 | 1.02 | 1.02 | 5e-06 | 2e-05 | 0.0004 | 0.0003 | 4e-11 | 0.70 / 0.72 / 0.71 / 0.68 |
| D1 | 0.688 | 0.677 | 0.651 | 0.690 | +0.030 | +0.030 | 0.07 | 0.39 | 0.97 | 0.80 | 1 | 1e-06 | 1e-05 | 3e-05 | 4e-08 | 0.44 / 0.44 / 0.43 / 0.43 |
| D2 | 0.669 | 0.691 | 0.639 | 0.704 | +0.018 | +0.010 | -0.01 | 0.08 | 0.42 | 0.56 | 0.3 | 1e-10 | 0.0003 | 0.006 | 0.0001 | 0.39 / 0.38 / 0.36 / 0.38 |
| D3 | 0.662 | 0.699 | 0.656 | 0.697 | +0.002 | +0.012 | -0.06 | -0.05 | 0.40 | 0.43 | 0.08 | 2e-10 | 0.5 | 0.02 | 0.2 | 0.42 / 0.40 / 0.38 / 0.41 |
| D4 | 0.644 | 0.712 | 0.642 | 0.710 | +0.040 | +0.035 | -0.45 | -0.36 | -0.12 | 0.21 | 2e-08 | 3e-08 | 7e-09 | 5e-08 | 5e-08 | 0.39 / 0.35 / 0.33 / 0.38 |

- P1 (self-stitch penalty < 0.03 at every location): FAILS (worst median penalty 0.090)
- P2 (late re-wiring): KILLED (g D4 -0.36, D3 -0.05, FR_D4 - real -0.080, p 3e-08; penalty-corrected g D4 -0.12); secondary: E recovers -1.00
- P3 (symmetry): FAILS (R->F at D4 loses 0.21 of the gap; penalty-corrected 0.60)

### Stitch fit R^2 per location: in-sample (calibration, mean over folds) / held-out (non-calibration cases, 1 - sum SSE / sum SST pooled over cases and folds)

| pair | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FR | 0.76 / 0.75 | 0.53 / 0.52 | 0.44 / 0.43 | 0.27 / 0.25 | 0.47 / 0.34 | 0.87 / 0.71 | 0.70 / 0.55 | 0.44 / 0.39 | 0.39 / 0.36 | 0.42 / 0.38 | 0.39 / 0.35 |
| RF | 0.76 / 0.75 | 0.53 / 0.52 | 0.44 / 0.44 | 0.28 / 0.27 | 0.48 / 0.35 | 0.87 / 0.72 | 0.72 / 0.58 | 0.44 / 0.39 | 0.38 / 0.35 | 0.40 / 0.36 | 0.35 / 0.28 |
| FF | 0.73 / 0.72 | 0.54 / 0.53 | 0.45 / 0.45 | 0.29 / 0.27 | 0.49 / 0.36 | 0.87 / 0.71 | 0.71 / 0.58 | 0.43 / 0.39 | 0.36 / 0.33 | 0.38 / 0.34 | 0.33 / 0.27 |
| RR | 0.77 / 0.77 | 0.56 / 0.55 | 0.44 / 0.43 | 0.27 / 0.25 | 0.46 / 0.32 | 0.85 / 0.70 | 0.68 / 0.53 | 0.43 / 0.38 | 0.38 / 0.36 | 0.41 / 0.38 | 0.38 / 0.34 |

## glioma_t2w_t2f: parents flat 0.714 / real 0.762 (median gap +0.048, n = 50, cases with |gap| > 0.02: 37)

### map applied to the whole patch (pre-registered)

| loc | FR | RF | FF | RR | pen FF | pen RR | g FR (case) | g FR (cell) | g_corr FR (cell) | g_RF loss (cell) | p FR-flat | p FR-real | p FF-flat | p RR-real | p RF-real | R2 FR / RF / FF / RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | 0.713 | 0.706 | 0.687 | 0.731 | +0.027 | +0.029 | 0.04 | nan | nan | nan | 0.03 | 5e-09 | 1e-05 | 9e-08 | 2e-09 | 0.55 / 0.57 / 0.56 / 0.56 |
| D0 | 0.759 | 0.724 | 0.714 | 0.756 | +0.003 | +0.000 | 0.89 | nan | nan | nan | 1e-05 | 0.7 | 0.2 | 0.3 | 3e-09 | 0.73 / 0.74 / 0.74 / 0.72 |
| D1 | 0.740 | 0.733 | 0.705 | 0.753 | +0.009 | +0.014 | 0.57 | nan | nan | nan | 0.01 | 0.0009 | 0.03 | 0.003 | 1e-07 | 0.45 / 0.44 / 0.44 / 0.45 |
| D2 | 0.702 | 0.733 | 0.700 | 0.745 | -0.001 | +0.011 | 0.22 | nan | nan | nan | 0.3 | 3e-06 | 1 | 0.0002 | 1e-07 | 0.40 / 0.40 / 0.40 / 0.38 |
| D3 | 0.699 | 0.734 | 0.705 | 0.743 | +0.014 | +0.021 | -0.06 | nan | nan | nan | 0.1 | 2e-08 | 0.03 | 8e-05 | 2e-06 | 0.47 / 0.45 / 0.46 / 0.45 |
| D4 | 0.735 | 0.752 | 0.722 | 0.720 | -0.002 | +0.035 | 0.30 | nan | nan | nan | 0.8 | 7e-08 | 1 | 1e-05 | 0.002 | 0.44 / 0.40 / 0.42 / 0.35 |

- P1 (self-stitch penalty < 0.03 at every location): FAILS (worst median penalty 0.035)
- P2 (late re-wiring): NOT EVALUABLE (|median gap| 0.048 <= 0.05; raw Dice above)
- P3 (symmetry): NOT EVALUABLE (|median gap| 0.048 <= 0.05; raw Dice above)

### Stitch fit R^2 per location: in-sample (calibration, mean over folds) / held-out (non-calibration cases, 1 - sum SSE / sum SST pooled over cases and folds)

| pair | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FR | 0.76 / 0.76 | 0.52 / 0.51 | 0.42 / 0.41 | 0.27 / 0.25 | 0.48 / 0.35 | 0.88 / 0.74 | 0.73 / 0.60 | 0.45 / 0.39 | 0.40 / 0.35 | 0.47 / 0.41 | 0.44 / 0.39 |
| RF | 0.78 / 0.78 | 0.55 / 0.53 | 0.42 / 0.41 | 0.28 / 0.26 | 0.49 / 0.36 | 0.89 / 0.75 | 0.74 / 0.62 | 0.44 / 0.38 | 0.40 / 0.35 | 0.45 / 0.40 | 0.40 / 0.35 |
| FF | 0.75 / 0.75 | 0.52 / 0.51 | 0.43 / 0.42 | 0.28 / 0.26 | 0.50 / 0.37 | 0.90 / 0.77 | 0.74 / 0.61 | 0.44 / 0.38 | 0.40 / 0.36 | 0.46 / 0.42 | 0.42 / 0.38 |
| RR | 0.79 / 0.79 | 0.54 / 0.52 | 0.42 / 0.42 | 0.27 / 0.25 | 0.48 / 0.34 | 0.88 / 0.73 | 0.72 / 0.58 | 0.45 / 0.39 | 0.38 / 0.34 | 0.45 / 0.41 | 0.35 / 0.29 |

## glioma_t2w_t1n: parents flat 0.433 / real 0.479 (median gap +0.045, n = 50, cases with |gap| > 0.02: 42)

### map applied to the whole patch (pre-registered)

| loc | FR | RF | FF | RR | pen FF | pen RR | g FR (case) | g FR (cell) | g_corr FR (cell) | g_RF loss (cell) | p FR-flat | p FR-real | p FF-flat | p RR-real | p RF-real | R2 FR / RF / FF / RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | 0.178 | 0.240 | 0.214 | 0.207 | +0.134 | +0.149 | -1.32 | nan | nan | nan | 2e-09 | 3e-09 | 2e-09 | 3e-09 | 3e-08 | 0.57 / 0.58 / 0.58 / 0.58 |
| D0 | 0.449 | 0.303 | 0.330 | 0.427 | +0.038 | +0.030 | 0.79 | nan | nan | nan | 0.03 | 0.0004 | 1e-07 | 2e-07 | 4e-09 | 0.76 / 0.76 / 0.75 / 0.75 |
| D1 | 0.348 | 0.324 | 0.242 | 0.383 | +0.101 | +0.074 | -0.05 | nan | nan | nan | 7e-05 | 3e-08 | 9e-09 | 1e-08 | 3e-09 | 0.42 / 0.42 / 0.41 / 0.42 |
| D2 | 0.464 | 0.475 | 0.387 | 0.532 | +0.008 | -0.025 | 0.41 | nan | nan | nan | 1e-06 | 0.4 | 0.09 | 8e-05 | 0.006 | 0.29 / 0.30 / 0.31 / 0.28 |
| D3 | 0.460 | 0.518 | 0.464 | 0.512 | -0.033 | -0.035 | 0.25 | nan | nan | nan | 2e-07 | 0.1 | 4e-07 | 2e-06 | 1e-07 | 0.30 / 0.30 / 0.32 / 0.29 |
| D4 | 0.201 | 0.124 | 0.110 | 0.160 | +0.199 | +0.256 | -0.81 | nan | nan | nan | 2e-09 | 1e-08 | 2e-09 | 2e-09 | 2e-09 | 0.29 / 0.24 / 0.28 / 0.22 |

- P1 (self-stitch penalty < 0.03 at every location): FAILS (worst median penalty 0.256)
- P2 (late re-wiring): NOT EVALUABLE (|median gap| 0.045 <= 0.05; raw Dice above)
- P3 (symmetry): NOT EVALUABLE (|median gap| 0.045 <= 0.05; raw Dice above)

### Stitch fit R^2 per location: in-sample (calibration, mean over folds) / held-out (non-calibration cases, 1 - sum SSE / sum SST pooled over cases and folds)

| pair | s0 | s1 | s2 | s3 | s4 | s5 | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FR | 0.76 / 0.76 | 0.53 / 0.52 | 0.43 / 0.43 | 0.28 / 0.26 | 0.50 / 0.39 | 0.90 / 0.79 | 0.76 / 0.65 | 0.42 / 0.38 | 0.29 / 0.28 | 0.30 / 0.29 | 0.29 / 0.29 |
| RF | 0.78 / 0.78 | 0.56 / 0.54 | 0.45 / 0.45 | 0.29 / 0.27 | 0.51 / 0.40 | 0.89 / 0.77 | 0.76 / 0.66 | 0.42 / 0.37 | 0.30 / 0.29 | 0.30 / 0.29 | 0.24 / 0.24 |
| FF | 0.76 / 0.75 | 0.53 / 0.52 | 0.46 / 0.45 | 0.29 / 0.28 | 0.51 / 0.40 | 0.90 / 0.79 | 0.75 / 0.65 | 0.41 / 0.37 | 0.31 / 0.29 | 0.32 / 0.31 | 0.28 / 0.27 |
| RR | 0.80 / 0.80 | 0.55 / 0.54 | 0.44 / 0.43 | 0.28 / 0.26 | 0.50 / 0.39 | 0.90 / 0.79 | 0.75 / 0.64 | 0.42 / 0.38 | 0.28 / 0.27 | 0.29 / 0.28 | 0.22 / 0.19 |
