# Scale-band scrambling: the reliance spectrum (visual proof v14, FINDINGS 41)

Design: PREREG.md (this directory). Cells: glioma_t1n_t2f, glioma_t1n_t2w, glioma_t2w_t2f, glioma_t2w_t1n; cases: glioma_t1n_t2f 70, glioma_t1n_t2w 70, glioma_t2w_t2f 70, glioma_t2w_t1n 70; seeds per band: 3. Bands (Gaussian sigma in voxels): fine = x - G1, mid = G1 - G2, coarse = G2 - G4, base = G4, allfine = fine + mid; *_le = local fine-energy map matched (Gaussian sigma 2 gain, applied twice) before the global rescale. x^(-b) = x - body*x_b + scr(x_b): same amplitude spectrum, random Hermitian phases, body-masked, body energy restored. Dice = target Dice on the plan patch (fold-mean softmax); r_b = Dice(x) - Dice(x^(-b)) (seed mean, then median over cases); rel = r_b / Dice(x) (cases with Dice(x) > 0.05). Differences real - flat etc. are per-case differences of r_b, Wilcoxon over cases.

## glioma_t1n_t2f (natural Dice noise 0.259 / flat 0.424 / real 0.632)

| band | r_b noise | r_b flat | r_b real | rel noise | rel flat | rel real | real-flat (p) | real-noise (p) | noise-flat (p) | rho(pred,dJ) noise / flat / real | edge ratio | tex ratio | tex contrast before -> after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fine | -0.029 | +0.041 | +0.119 | -0.19 | +0.11 | +0.23 | +0.066 (3e-07) | +0.156 (6e-10) | -0.073 (1e-05) | 0.41 / -0.05 / -0.07 | 0.91 | 1.07 |  |
| fine_le | +0.037 | +0.045 | +0.092 | +0.19 | +0.12 | +0.16 | +0.043 (3e-05) | +0.051 (0.0006) | +0.000 (0.6) | 0.45 / 0.12 / 0.05 | 0.91 | 0.96 |  |
| fine_eq | -0.038 | +0.010 | +0.031 | -0.16 | +0.03 | +0.06 | +0.011 (0.02) | +0.072 (1e-10) | -0.049 (1e-07) | 0.65 / 0.74 / 0.71 | 1.02 | 1.13 | 0.64 -> 0.84 |
| fine_eq_scr | -0.041 | +0.035 | +0.118 | -0.23 | +0.13 | +0.23 | +0.072 (1e-07) | +0.181 (3e-10) | -0.082 (2e-06) | 0.39 / -0.04 / -0.05 | 0.91 | 1.08 | 0.64 -> 1.00 |
| mid | +0.015 | +0.048 | +0.061 | +0.12 | +0.13 | +0.10 | +0.015 (0.08) | +0.038 (0.03) | -0.013 (0.1) | 0.26 / 0.33 / 0.44 | 0.83 | 0.83 |  |
| mid_le | -0.007 | -0.035 | -0.009 | -0.04 | -0.10 | -0.02 | +0.020 (0.1) | -0.010 (0.03) | +0.028 (8e-05) | 0.58 / 0.67 / 0.78 | 0.83 | 0.88 |  |
| coarse | +0.040 | +0.021 | +0.049 | +0.16 | +0.05 | +0.07 | +0.026 (0.05) | +0.001 (0.4) | +0.010 (0.02) | 0.15 / -0.08 / 0.19 | 0.71 | 0.93 |  |
| base | -0.007 | +0.028 | +0.126 | -0.02 | +0.05 | +0.22 | +0.071 (3e-07) | +0.122 (4e-11) | -0.039 (0.001) | 0.22 / 0.26 / 0.37 | 0.70 | 0.98 |  |
| allfine | -0.004 | +0.054 | +0.236 | +0.06 | +0.24 | +0.36 | +0.148 (3e-10) | +0.212 (1e-10) | -0.033 (0.007) | 0.26 / 0.23 / 0.28 | 0.77 | 1.25 |  |
| allfine_le | +0.080 | +0.113 | +0.215 | +0.44 | +0.35 | +0.36 | +0.062 (0.0005) | +0.073 (0.0001) | -0.022 (0.1) | 0.36 / 0.32 / 0.29 | 0.76 | 1.31 |  |
| allfine_eq | -0.028 | +0.027 | +0.041 | -0.12 | +0.07 | +0.08 | +0.019 (0.007) | +0.078 (1e-10) | -0.048 (8e-08) | 0.46 / 0.57 / 0.68 | 1.01 | 1.05 | 0.64 -> 0.65 |
| allfine_eq_scr | -0.017 | +0.051 | +0.242 | +0.01 | +0.16 | +0.37 | +0.162 (1e-10) | +0.232 (5e-11) | -0.057 (0.001) | 0.27 / 0.24 / 0.29 | 0.77 | 1.30 | 0.64 -> 1.00 |

## glioma_t1n_t2w (natural Dice noise 0.583 / flat 0.694 / real 0.745)

| band | r_b noise | r_b flat | r_b real | rel noise | rel flat | rel real | real-flat (p) | real-noise (p) | noise-flat (p) | rho(pred,dJ) noise / flat / real | edge ratio | tex ratio | tex contrast before -> after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fine | +0.013 | +0.148 | +0.207 | +0.02 | +0.22 | +0.29 | +0.055 (8e-05) | +0.182 (5e-13) | -0.137 (1e-12) | 0.23 / -0.02 / -0.31 | 0.93 | 1.56 |  |
| fine_le | +0.021 | +0.061 | +0.039 | +0.04 | +0.10 | +0.05 | -0.016 (0.0001) | +0.014 (0.01) | -0.042 (1e-05) | 0.51 / 0.35 / 0.17 | 0.93 | 0.98 |  |
| fine_eq | +0.002 | +0.038 | +0.057 | +0.01 | +0.06 | +0.08 | +0.012 (0.04) | +0.055 (2e-09) | -0.037 (2e-08) | 0.64 / 0.64 / 0.66 | 1.05 | 1.56 | 0.26 -> 0.71 |
| fine_eq_scr | +0.013 | +0.145 | +0.207 | +0.02 | +0.21 | +0.30 | +0.059 (5e-05) | +0.188 (4e-13) | -0.146 (1e-12) | 0.20 / -0.05 / -0.30 | 0.93 | 1.57 | 0.26 -> 0.86 |
| mid | +0.094 | +0.153 | +0.149 | +0.16 | +0.22 | +0.21 | +0.008 (0.5) | +0.070 (4e-06) | -0.058 (6e-06) | 0.01 / -0.04 / -0.07 | 0.89 | 0.93 |  |
| mid_le | +0.052 | +0.061 | +0.040 | +0.09 | +0.09 | +0.05 | -0.017 (0.0001) | -0.007 (0.06) | -0.011 (0.1) | 0.22 / 0.08 / 0.24 | 0.86 | 0.85 |  |
| coarse | +0.158 | +0.126 | +0.143 | +0.27 | +0.18 | +0.20 | +0.024 (0.0002) | +0.003 (0.3) | +0.026 (0.05) | 0.21 / 0.18 / 0.44 | 0.83 | 0.97 |  |
| base | +0.234 | +0.139 | +0.183 | +0.47 | +0.23 | +0.25 | +0.038 (6e-06) | -0.024 (0.1) | +0.078 (7e-06) | -0.04 / 0.29 / 0.34 | 0.80 | 1.02 |  |
| allfine | +0.276 | +0.390 | +0.450 | +0.48 | +0.64 | +0.70 | +0.091 (3e-07) | +0.233 (9e-11) | -0.153 (1e-08) | 0.24 / -0.25 / -0.28 | 0.90 | 1.90 |  |
| allfine_le | +0.183 | +0.300 | +0.249 | +0.36 | +0.47 | +0.37 | -0.038 (0.003) | +0.072 (2e-05) | -0.129 (1e-08) | 0.18 / -0.14 / -0.28 | 0.84 | 1.24 |  |
| allfine_eq | +0.039 | +0.045 | +0.052 | +0.07 | +0.07 | +0.07 | +0.006 (0.3) | +0.015 (0.005) | -0.011 (0.03) | 0.43 / 0.77 / 0.67 | 1.13 | 1.63 | 0.26 -> 0.74 |
| allfine_eq_scr | +0.283 | +0.391 | +0.457 | +0.48 | +0.65 | +0.70 | +0.086 (1e-07) | +0.222 (2e-10) | -0.135 (6e-08) | 0.25 / -0.21 / -0.31 | 0.89 | 1.99 | 0.26 -> 0.97 |

## glioma_t2w_t2f (natural Dice noise 0.741 / flat 0.734 / real 0.769)

| band | r_b noise | r_b flat | r_b real | rel noise | rel flat | rel real | real-flat (p) | real-noise (p) | noise-flat (p) | rho(pred,dJ) noise / flat / real | edge ratio | tex ratio | tex contrast before -> after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fine | -0.012 | +0.026 | +0.050 | -0.02 | +0.03 | +0.06 | +0.028 (2e-06) | +0.069 (2e-11) | -0.042 (4e-09) | -0.20 / 0.09 / -0.22 | 0.91 | 1.07 |  |
| fine_le | +0.062 | +0.035 | +0.045 | +0.09 | +0.06 | +0.06 | +0.013 (0.2) | -0.017 (0.0003) | +0.025 (2e-07) | 0.79 / 0.51 / 0.50 | 0.91 | 0.96 |  |
| fine_eq | -0.021 | +0.024 | +0.041 | -0.03 | +0.03 | +0.05 | +0.012 (0.0009) | +0.062 (4e-11) | -0.047 (2e-10) | 0.76 / 0.67 / 0.73 | 1.02 | 1.13 | 0.64 -> 0.84 |
| fine_eq_scr | -0.014 | +0.026 | +0.049 | -0.02 | +0.04 | +0.07 | +0.027 (3e-07) | +0.071 (2e-11) | -0.041 (2e-09) | -0.19 / 0.07 / -0.21 | 0.91 | 1.08 | 0.64 -> 1.00 |
| mid | +0.030 | +0.036 | +0.034 | +0.04 | +0.05 | +0.04 | -0.002 (0.2) | -0.001 (1) | -0.005 (0.4) | 0.29 / 0.24 / 0.08 | 0.83 | 0.83 |  |
| mid_le | +0.036 | +0.026 | +0.028 | +0.05 | +0.03 | +0.04 | -0.002 (0.7) | -0.008 (0.0003) | +0.007 (0.0006) | 0.54 / 0.41 / 0.62 | 0.83 | 0.88 |  |
| coarse | +0.056 | +0.051 | +0.041 | +0.08 | +0.07 | +0.06 | -0.007 (0.06) | -0.012 (0.001) | +0.008 (0.02) | 0.04 / 0.45 / 0.50 | 0.71 | 0.93 |  |
| base | +0.115 | +0.098 | +0.100 | +0.17 | +0.14 | +0.13 | -0.002 (0.7) | -0.030 (0.001) | +0.019 (0.02) | 0.38 / 0.43 / 0.42 | 0.70 | 0.98 |  |
| allfine | +0.061 | +0.179 | +0.268 | +0.09 | +0.25 | +0.36 | +0.087 (1e-07) | +0.193 (8e-12) | -0.096 (2e-10) | -0.12 / 0.08 / -0.14 | 0.77 | 1.25 |  |
| allfine_le | +0.161 | +0.193 | +0.174 | +0.20 | +0.26 | +0.22 | +0.006 (0.6) | +0.019 (0.06) | -0.032 (0.002) | 0.42 / 0.26 / 0.24 | 0.76 | 1.31 |  |
| allfine_eq | -0.017 | +0.018 | +0.036 | -0.02 | +0.02 | +0.04 | +0.016 (0.0005) | +0.049 (4e-09) | -0.033 (2e-07) | 0.82 / 0.65 / 0.79 | 1.01 | 1.05 | 0.64 -> 0.65 |
| allfine_eq_scr | +0.068 | +0.186 | +0.286 | +0.11 | +0.25 | +0.40 | +0.093 (2e-07) | +0.195 (9e-12) | -0.105 (2e-10) | -0.15 / 0.06 / -0.16 | 0.77 | 1.30 | 0.64 -> 1.00 |

## glioma_t2w_t1n (natural Dice noise 0.605 / flat 0.433 / real 0.456)

| band | r_b noise | r_b flat | r_b real | rel noise | rel flat | rel real | real-flat (p) | real-noise (p) | noise-flat (p) | rho(pred,dJ) noise / flat / real | edge ratio | tex ratio | tex contrast before -> after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fine | +0.165 | +0.186 | +0.264 | +0.30 | +0.57 | +0.67 | +0.063 (2e-06) | +0.071 (0.001) | -0.020 (0.3) | -0.17 / -0.17 / -0.21 | 0.93 | 1.35 |  |
| fine_le | -0.003 | -0.084 | -0.018 | -0.01 | -0.22 | -0.06 | +0.064 (1e-06) | -0.003 (0.2) | +0.057 (6e-07) | 0.60 / 0.46 / 0.34 | 0.92 | 0.94 |  |
| fine_eq | +0.098 | +0.178 | +0.210 | +0.18 | +0.55 | +0.59 | +0.040 (0.005) | +0.123 (7e-07) | -0.089 (0.0002) | 0.77 / 0.80 / 0.64 | 1.02 | 1.46 | 0.34 -> 0.77 |
| fine_eq_scr | +0.166 | +0.192 | +0.268 | +0.31 | +0.59 | +0.67 | +0.062 (2e-06) | +0.076 (0.001) | -0.028 (0.3) | -0.18 / -0.18 / -0.20 | 0.93 | 1.37 | 0.34 -> 0.85 |
| mid | +0.144 | +0.127 | +0.105 | +0.35 | +0.38 | +0.28 | +0.005 (0.8) | -0.050 (0.0001) | +0.039 (0.002) | 0.16 / 0.28 / 0.20 | 0.89 | 0.92 |  |
| mid_le | +0.056 | +0.026 | +0.027 | +0.10 | +0.07 | +0.07 | +0.012 (0.06) | -0.019 (0.01) | +0.036 (0.0002) | 0.35 / 0.51 / 0.36 | 0.85 | 0.86 |  |
| coarse | +0.128 | +0.039 | +0.034 | +0.21 | +0.12 | +0.10 | +0.004 (0.9) | -0.085 (2e-09) | +0.069 (6e-09) | 0.13 / 0.27 / 0.24 | 0.84 | 0.99 |  |
| base | +0.123 | +0.061 | +0.064 | +0.21 | +0.19 | +0.14 | -0.024 (0.04) | -0.057 (9e-06) | +0.054 (0.0009) | 0.10 / 0.61 / 0.54 | 0.91 | 1.02 |  |
| allfine | +0.412 | +0.255 | +0.377 | +0.84 | +0.74 | +0.78 | +0.059 (4e-06) | -0.046 (0.06) | +0.102 (5e-05) | 0.11 / 0.17 / -0.03 | 0.93 | 1.66 |  |
| allfine_le | +0.248 | +0.080 | +0.092 | +0.42 | +0.24 | +0.21 | -0.009 (0.9) | -0.110 (3e-06) | +0.118 (2e-07) | 0.21 / 0.17 / 0.10 | 0.84 | 1.14 |  |
| allfine_eq | +0.122 | +0.170 | +0.198 | +0.26 | +0.51 | +0.53 | +0.039 (0.02) | +0.040 (0.002) | -0.041 (0.09) | 0.58 / 0.76 / 0.60 | 1.07 | 1.69 | 0.34 -> 0.90 |
| allfine_eq_scr | +0.414 | +0.244 | +0.357 | +0.82 | +0.72 | +0.76 | +0.060 (3e-06) | -0.039 (0.04) | +0.098 (5e-05) | 0.15 / 0.17 / -0.01 | 0.90 | 1.76 | 0.34 -> 0.96 |

## Pre-registered verdicts (alpha 0.01, Wilcoxon over cases)

Read on both scrambling variants: **fine_le** (local fine-energy map matched, primary: destroys structure only) and **fine** (global body energy only; also changes the local heterogeneity level). The specificity check uses coarse and base (one variant each).

### Band fine_le (PRIMARY, local-energy matched)

- glioma_t1n_t2f: r_fine_le real - flat +0.043 (p 3e-05) -> real > flat; coarse/base real - flat significantly positive: base
- glioma_t1n_t2w: r_fine_le real - flat -0.016 (p 0.0001) -> not significant / not positive; coarse/base real - flat significantly positive: coarse, base
- glioma_t2w_t2f: r_fine_le real - flat +0.013 (p 0.2) -> not significant / not positive; coarse/base real - flat significantly positive: none
- glioma_t2w_t1n: r_fine_le real - flat +0.064 (p 1e-06) -> real > flat; coarse/base real - flat significantly positive: none

**P1 (fine_le, reliance ordering):** NOT SUPPORTED (inconclusive): r_fine_le real > flat significant in 2/4 cells (need >= 3); real <= flat in 1/4.

**P2 (fine_le, noise arm r <= flat):** VIOLATED (noise significantly greater than flat in 2/4 cells; 'not greater' is read as no significant positive noise - flat difference). glioma_t1n_t2f +0.000 (p 0.6); glioma_t1n_t2w -0.042 (p 1e-05); glioma_t2w_t2f +0.025 (p 2e-07); glioma_t2w_t1n +0.057 (p 6e-07)

**P3 (fine_le, gradient link, real arm, Spearman > 0.5):** 1/4 cells with a defined rho pass -> PARTLY. glioma_t1n_t2f 0.05; glioma_t1n_t2w 0.17; glioma_t2w_t2f 0.50; glioma_t2w_t1n 0.34

**QC (fine_le, GT-boundary edge ratio within [0.8, 1.2]):** PASS. glioma_t1n_t2f edge 0.91, target texture ratio 0.96; glioma_t1n_t2w edge 0.93, target texture ratio 0.98; glioma_t2w_t2f edge 0.91, target texture ratio 0.96; glioma_t2w_t1n edge 0.92, target texture ratio 0.94

### Band fine (global-energy variant)

- glioma_t1n_t2f: r_fine real - flat +0.066 (p 3e-07) -> real > flat; coarse/base real - flat significantly positive: base
- glioma_t1n_t2w: r_fine real - flat +0.055 (p 8e-05) -> real > flat; coarse/base real - flat significantly positive: coarse, base
- glioma_t2w_t2f: r_fine real - flat +0.028 (p 2e-06) -> real > flat; coarse/base real - flat significantly positive: none
- glioma_t2w_t1n: r_fine real - flat +0.063 (p 2e-06) -> real > flat; coarse/base real - flat significantly positive: none

**P1 (fine, reliance ordering):** PARTLY: r_fine real > flat in 4/4 cells, but coarse/base show the same ordering in some of them (not fine-scale specific).

**P2 (fine, noise arm r <= flat):** HOLDS (noise significantly greater than flat in 0/4 cells; 'not greater' is read as no significant positive noise - flat difference). glioma_t1n_t2f -0.073 (p 1e-05); glioma_t1n_t2w -0.137 (p 1e-12); glioma_t2w_t2f -0.042 (p 4e-09); glioma_t2w_t1n -0.020 (p 0.3)

**P3 (fine, gradient link, real arm, Spearman > 0.5):** 0/4 cells with a defined rho pass -> NOT SUPPORTED: the gradient-based sensitivity results (sections 33-37) do not predict reliance. glioma_t1n_t2f -0.07; glioma_t1n_t2w -0.31; glioma_t2w_t2f -0.22; glioma_t2w_t1n -0.21

**QC (fine, GT-boundary edge ratio within [0.8, 1.2]):** PASS. glioma_t1n_t2f edge 0.91, target texture ratio 1.07; glioma_t1n_t2w edge 0.93, target texture ratio 1.56; glioma_t2w_t2f edge 0.91, target texture ratio 1.07; glioma_t2w_t1n edge 0.93, target texture ratio 1.35

## Addendum (PREREG_addendum.md, set after the first smoke tests / after the 4-cell result): structure vs energy map

Crossed factors: fine = structure destroyed, energy spread evenly over the body; fine_le = structure destroyed, local energy map kept; fine_eq = structure kept, map flattened (deterministic, 1 seed); fine_eq_scr = structure destroyed AND map flattened. Same for allfine. Texture contrast = mean fine energy in the target / in the body (natural -> after).

- glioma_t1n_t2f: r_fine_eq real - flat +0.011 (p 0.02), real - noise +0.072 (p 1e-10)
- glioma_t1n_t2w: r_fine_eq real - flat +0.012 (p 0.04), real - noise +0.055 (p 2e-09)
- glioma_t2w_t2f: r_fine_eq real - flat +0.012 (p 0.0009), real - noise +0.062 (p 4e-11)
- glioma_t2w_t1n: r_fine_eq real - flat +0.040 (p 0.005), real - noise +0.123 (p 7e-07)

**P4 (structure kept, map flattened: r_fine_eq real > flat in >= 3/4 cells and real > noise):** KILLED: real - flat on fine_eq not significantly positive in 2/4 cells.

- glioma_t1n_t2f: real - flat on fine_eq_scr +0.072 (p 1e-07) vs on fine +0.066; difference +0.004 (p 0.0004) -> does not hold. noise: r(eq_scr) - r(fine) -0.012 (p 1e-12), additivity residual r(eq_scr) - [r(le) + r(eq)] -0.047; flat: r(eq_scr) - r(fine) -0.002 (p 9e-05), additivity residual r(eq_scr) - [r(le) + r(eq)] -0.014; real: r(eq_scr) - r(fine) +0.001 (p 0.3), additivity residual r(eq_scr) - [r(le) + r(eq)] +0.002
- glioma_t1n_t2w: real - flat on fine_eq_scr +0.059 (p 5e-05) vs on fine +0.055; difference +0.004 (p 0.0004) -> does not hold. noise: r(eq_scr) - r(fine) -0.002 (p 0.03), additivity residual r(eq_scr) - [r(le) + r(eq)] -0.023; flat: r(eq_scr) - r(fine) +0.000 (p 0.7), additivity residual r(eq_scr) - [r(le) + r(eq)] +0.030; real: r(eq_scr) - r(fine) +0.002 (p 2e-05), additivity residual r(eq_scr) - [r(le) + r(eq)] +0.081
- glioma_t2w_t2f: real - flat on fine_eq_scr +0.027 (p 3e-07) vs on fine +0.028; difference +0.002 (p 0.0001) -> does not hold. noise: r(eq_scr) - r(fine) -0.002 (p 6e-06), additivity residual r(eq_scr) - [r(le) + r(eq)] -0.056; flat: r(eq_scr) - r(fine) +0.000 (p 0.9), additivity residual r(eq_scr) - [r(le) + r(eq)] -0.042; real: r(eq_scr) - r(fine) +0.002 (p 0.0001), additivity residual r(eq_scr) - [r(le) + r(eq)] -0.018
- glioma_t2w_t1n: real - flat on fine_eq_scr +0.062 (p 2e-06) vs on fine +0.063; difference +0.001 (p 0.5) -> holds. noise: r(eq_scr) - r(fine) +0.000 (p 0.5), additivity residual r(eq_scr) - [r(le) + r(eq)] +0.051; flat: r(eq_scr) - r(fine) +0.000 (p 0.2), additivity residual r(eq_scr) - [r(le) + r(eq)] +0.059; real: r(eq_scr) - r(fine) +0.000 (p 0.1), additivity residual r(eq_scr) - [r(le) + r(eq)] +0.069

**P5 (fine_eq_scr ~ fine: flattening explains the global-scrambling effect):** NOT SUPPORTED (1/4 cells).

**QC (addendum variants; real-arm rows, values are identical across arms):**

- glioma_t1n_t2f: fine edge 0.91 tex 1.07; fine_le edge 0.91 tex 0.96; fine_eq edge 1.02 tex 1.13 contrast 0.64 -> 0.84; fine_eq_scr edge 0.91 tex 1.08 contrast 0.64 -> 1.00; allfine_eq edge 1.01 tex 1.05 contrast 0.64 -> 0.65; allfine_eq_scr edge 0.77 tex 1.30 contrast 0.64 -> 1.00
- glioma_t1n_t2w: fine edge 0.93 tex 1.56; fine_le edge 0.93 tex 0.98; fine_eq edge 1.05 tex 1.56 contrast 0.26 -> 0.71; fine_eq_scr edge 0.93 tex 1.57 contrast 0.26 -> 0.86; allfine_eq edge 1.13 tex 1.63 contrast 0.26 -> 0.74; allfine_eq_scr edge 0.89 tex 1.99 contrast 0.26 -> 0.97
- glioma_t2w_t2f: fine edge 0.91 tex 1.07; fine_le edge 0.91 tex 0.96; fine_eq edge 1.02 tex 1.13 contrast 0.64 -> 0.84; fine_eq_scr edge 0.91 tex 1.08 contrast 0.64 -> 1.00; allfine_eq edge 1.01 tex 1.05 contrast 0.64 -> 0.65; allfine_eq_scr edge 0.77 tex 1.30 contrast 0.64 -> 1.00
- glioma_t2w_t1n: fine edge 0.93 tex 1.35; fine_le edge 0.92 tex 0.94; fine_eq edge 1.02 tex 1.46 contrast 0.34 -> 0.77; fine_eq_scr edge 0.93 tex 1.37 contrast 0.34 -> 0.85; allfine_eq edge 1.07 tex 1.69 contrast 0.34 -> 0.90; allfine_eq_scr edge 0.90 tex 1.76 contrast 0.34 -> 0.96

