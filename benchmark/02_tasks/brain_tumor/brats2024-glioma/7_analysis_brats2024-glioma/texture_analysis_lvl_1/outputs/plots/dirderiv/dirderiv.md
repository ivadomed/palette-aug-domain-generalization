# Directional derivative along the image's own fine detail (FINDINGS section 38): is the texture that is there used by the decision?

**DEGENERATE GLOBALLY (audit 2026-10-13, `--check`): nnU-Net is instance-normed, so f(a x) = f(x) and <G, x> = 0 exactly (measured cos(G, x) = 0.0000). Hence <G, hp> = -<G, lp>, texfrac = 0.5 by identity, attr is not an attribution (gradient x input sums to zero for any split of the image) and the phase-randomised null is biased against the real hp. The GLOBAL tables below (cos_hp, cos_lp, texfrac, attr, cos_null, above chance, per pattern) are WITHDRAWN. Only the _local / _mid / _in variants (inner product restricted to a ball or region, which breaks the identity) are valid first-order derivatives; the gradient itself matches finite differences (r 0.999). See FINDINGS 38.**

Seeds = section-36 voxels; gradient of each arm's OWN predicted class, fold-mean. hp = x - gaussian(x, sigma) over the body; <G, hp> = d/d eps f(x + eps hp) at 0. Units: per-case median over seeds, median over cases, Wilcoxon over cases.

## cos_hp_s1: cos(G, hp1): signed alignment of the decision's sensitivity with the image's own sigma-1 fine detail

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.000 | +0.000 | +0.000 | +0.000 | 0.71 | 1.5e-06 | +0.000 | 0.69 | 0.00038 |
| glioma_t1n_t2w | Glioma | 70 | +0.000 | +0.000 | +0.000 | -0.000 | 0.39 | 0.022 | -0.000 | 0.31 | 0.00055 |
| glioma_t2w_t2f | Glioma | 69 | +0.000 | +0.000 | +0.000 | -0.000 | 0.41 | 0.14 | -0.000 | 0.45 | 0.14 |
| glioma_t2w_t1n | Glioma | 70 | -0.000 | +0.000 | +0.000 | -0.000 | 0.31 | 0.001 | +0.000 | 0.57 | 0.0078 |

real > flat in 1 / 4 cells (p < 0.05: 1 +, 2 -); real > noise in 2 / 4 (p < 0.05: 2 +, 1 -)

## abscos_hp_s1: |cos(G, hp1)|

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.001 | +0.001 | +0.001 | -0.000 | 0.36 | 0.023 | -0.000 | 0.47 | 0.27 |
| glioma_t1n_t2w | Glioma | 70 | +0.001 | +0.000 | +0.000 | -0.000 | 0.40 | 0.018 | -0.000 | 0.11 | 3e-11 |
| glioma_t2w_t2f | Glioma | 69 | +0.001 | +0.000 | +0.000 | +0.000 | 0.51 | 0.54 | -0.000 | 0.14 | 2.7e-07 |
| glioma_t2w_t1n | Glioma | 70 | +0.000 | +0.001 | +0.000 | -0.000 | 0.34 | 0.014 | -0.000 | 0.43 | 0.18 |

real > flat in 1 / 4 cells (p < 0.05: 0 +, 3 -); real > noise in 0 / 4 (p < 0.05: 0 +, 2 -)

## cos_null_s1: |cos| chance level (phase-randomised hp1)

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.001 | +0.001 | +0.001 | -0.000 | 0.27 | 2.9e-06 | -0.000 | 0.13 | 3.7e-10 |
| glioma_t1n_t2w | Glioma | 70 | +0.001 | +0.001 | +0.001 | -0.000 | 0.43 | 0.023 | -0.000 | 0.07 | 4.9e-12 |
| glioma_t2w_t2f | Glioma | 69 | +0.001 | +0.001 | +0.001 | -0.000 | 0.14 | 9.7e-11 | -0.000 | 0.06 | 2.3e-12 |
| glioma_t2w_t1n | Glioma | 70 | +0.001 | +0.001 | +0.001 | -0.000 | 0.19 | 3.6e-08 | -0.000 | 0.03 | 5.2e-13 |

real > flat in 0 / 4 cells (p < 0.05: 0 +, 4 -); real > noise in 0 / 4 (p < 0.05: 0 +, 4 -)

## cos_hp_s2: cos(G, hp2), sigma 2

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.000 | +0.000 | +0.000 | -0.000 | 0.50 | 0.87 | -0.000 | 0.36 | 0.0044 |
| glioma_t1n_t2w | Glioma | 70 | +0.000 | +0.000 | -0.000 | -0.000 | 0.37 | 0.0069 | -0.000 | 0.21 | 1.8e-07 |
| glioma_t2w_t2f | Glioma | 69 | +0.000 | +0.000 | -0.000 | -0.000 | 0.39 | 0.075 | -0.000 | 0.43 | 0.07 |
| glioma_t2w_t1n | Glioma | 70 | -0.000 | +0.000 | -0.000 | -0.000 | 0.26 | 2.8e-05 | +0.000 | 0.60 | 0.13 |

real > flat in 0 / 4 cells (p < 0.05: 0 +, 2 -); real > noise in 1 / 4 (p < 0.05: 0 +, 2 -)

## cos_lp_s1: cos(G, lp1): alignment with the smooth content

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | -0.000 | -0.000 | -0.000 | -0.000 | 0.29 | 2.4e-06 | -0.000 | 0.31 | 0.00038 |
| glioma_t1n_t2w | Glioma | 70 | -0.000 | -0.000 | -0.000 | +0.000 | 0.63 | 0.025 | +0.000 | 0.70 | 0.00077 |
| glioma_t2w_t2f | Glioma | 69 | -0.000 | -0.000 | -0.000 | +0.000 | 0.59 | 0.13 | +0.000 | 0.54 | 0.16 |
| glioma_t2w_t1n | Glioma | 70 | +0.000 | -0.000 | -0.000 | +0.000 | 0.69 | 0.0014 | -0.000 | 0.40 | 0.0053 |

real > flat in 3 / 4 cells (p < 0.05: 2 +, 1 -); real > noise in 2 / 4 (p < 0.05: 1 +, 2 -)

## texfrac_s1: share of the directional sensitivity along the texture vs the smooth content

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.499 | +0.501 | +0.503 | +0.002 | 0.77 | 9.3e-08 | +0.005 | 0.86 | 2.6e-09 |
| glioma_t1n_t2w | Glioma | 70 | +0.500 | +0.500 | +0.500 | +0.000 | 0.57 | 0.084 | -0.000 | 0.49 | 0.96 |
| glioma_t2w_t2f | Glioma | 69 | +0.501 | +0.498 | +0.498 | +0.000 | 0.52 | 0.7 | -0.004 | 0.20 | 1.2e-07 |
| glioma_t2w_t1n | Glioma | 70 | +0.504 | +0.499 | +0.496 | -0.004 | 0.23 | 2.8e-06 | -0.010 | 0.11 | 5.5e-11 |

real > flat in 3 / 4 cells (p < 0.05: 1 +, 1 -); real > noise in 1 / 4 (p < 0.05: 1 +, 2 -)

## attr_s1: <G, hp1> / margin: first-order fraction of the decision margin carried by the texture present

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.028 | +0.036 | +0.119 | +0.112 | 0.71 | 2.7e-06 | +0.098 | 0.74 | 2.2e-06 |
| glioma_t1n_t2w | Glioma | 70 | +0.024 | +0.026 | +0.005 | -0.033 | 0.34 | 0.005 | -0.045 | 0.33 | 0.0022 |
| glioma_t2w_t2f | Glioma | 69 | +0.021 | +0.024 | +0.020 | -0.011 | 0.42 | 0.29 | -0.007 | 0.48 | 0.48 |
| glioma_t2w_t1n | Glioma | 70 | -0.011 | +0.109 | +0.034 | -0.066 | 0.36 | 0.016 | +0.036 | 0.59 | 0.058 |

real > flat in 1 / 4 cells (p < 0.05: 1 +, 2 -); real > noise in 2 / 4 (p < 0.05: 1 +, 1 -)

## attr_s2: <G, hp2> / margin

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | +0.079 | +0.054 | +0.032 | +0.007 | 0.53 | 0.96 | -0.009 | 0.47 | 0.5 |
| glioma_t1n_t2w | Glioma | 70 | +0.086 | +0.005 | -0.028 | -0.061 | 0.33 | 0.0076 | -0.122 | 0.26 | 2e-07 |
| glioma_t2w_t2f | Glioma | 69 | +0.012 | +0.036 | -0.004 | -0.035 | 0.42 | 0.035 | -0.044 | 0.35 | 0.025 |
| glioma_t2w_t1n | Glioma | 70 | -0.016 | +0.112 | -0.017 | -0.154 | 0.31 | 0.0001 | +0.018 | 0.54 | 0.82 |

real > flat in 1 / 4 cells (p < 0.05: 0 +, 3 -); real > noise in 1 / 4 (p < 0.05: 0 +, 2 -)

## cos_hp_s1_local: cos(G, hp1) within r_loc of the voxel

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | -0.022 | -0.029 | -0.041 | -0.008 | 0.44 | 0.14 | -0.018 | 0.37 | 0.054 |
| glioma_t1n_t2w | Glioma | 70 | +0.043 | -0.006 | -0.034 | -0.039 | 0.29 | 2.7e-07 | -0.073 | 0.27 | 1e-05 |
| glioma_t2w_t2f | Glioma | 69 | -0.036 | -0.034 | -0.055 | -0.007 | 0.48 | 0.21 | -0.015 | 0.46 | 0.59 |
| glioma_t2w_t1n | Glioma | 70 | -0.187 | -0.012 | -0.070 | -0.050 | 0.26 | 1.9e-05 | +0.113 | 0.76 | 0.00015 |

real > flat in 0 / 4 cells (p < 0.05: 0 +, 2 -); real > noise in 1 / 4 (p < 0.05: 1 +, 1 -)

## attr_s1_local: attr within r_loc

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | -0.005 | -0.021 | -0.024 | -0.009 | 0.46 | 0.47 | -0.021 | 0.30 | 0.02 |
| glioma_t1n_t2w | Glioma | 70 | +0.013 | -0.005 | -0.029 | -0.028 | 0.17 | 2.9e-07 | -0.036 | 0.17 | 1.8e-08 |
| glioma_t2w_t2f | Glioma | 69 | -0.016 | -0.053 | -0.069 | -0.023 | 0.42 | 0.01 | -0.064 | 0.36 | 0.0002 |
| glioma_t2w_t1n | Glioma | 70 | -0.084 | -0.014 | -0.119 | -0.103 | 0.21 | 3e-07 | -0.026 | 0.41 | 0.099 |

real > flat in 0 / 4 cells (p < 0.05: 0 +, 3 -); real > noise in 0 / 4 (p < 0.05: 0 +, 3 -)

## cos_hp_s1_mid: cos(G, hp1) within 3 r_loc

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | -0.005 | +0.000 | +0.000 | +0.003 | 0.56 | 0.49 | +0.007 | 0.67 | 0.0067 |
| glioma_t1n_t2w | Glioma | 70 | +0.012 | +0.002 | -0.003 | -0.006 | 0.26 | 2.1e-06 | -0.015 | 0.21 | 3.8e-07 |
| glioma_t2w_t2f | Glioma | 69 | -0.004 | -0.006 | -0.006 | +0.001 | 0.55 | 0.72 | -0.004 | 0.46 | 0.55 |
| glioma_t2w_t1n | Glioma | 70 | -0.022 | +0.005 | -0.012 | -0.016 | 0.20 | 1.4e-08 | +0.014 | 0.63 | 0.0073 |

real > flat in 2 / 4 cells (p < 0.05: 0 +, 2 -); real > noise in 2 / 4 (p < 0.05: 2 +, 1 -)

## cos_hp_s1_in: cos(G, hp1) inside the target region

| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | Glioma | 70 | -0.002 | +0.000 | +0.001 | +0.002 | 0.60 | 0.1 | +0.002 | 0.66 | 0.00076 |
| glioma_t1n_t2w | Glioma | 70 | +0.002 | +0.001 | -0.000 | -0.001 | 0.41 | 0.021 | -0.003 | 0.33 | 0.00015 |
| glioma_t2w_t2f | Glioma | 69 | -0.003 | -0.001 | -0.001 | -0.000 | 0.48 | 0.15 | +0.000 | 0.51 | 0.79 |
| glioma_t2w_t1n | Glioma | 70 | -0.006 | +0.003 | -0.002 | -0.006 | 0.19 | 5.2e-07 | +0.002 | 0.63 | 0.063 |

real > flat in 1 / 4 cells (p < 0.05: 0 +, 2 -); real > noise in 3 / 4 (p < 0.05: 1 +, 1 -)

## Above chance: |cos(G, hp1)| minus its phase-randomised null, per arm (per-case medians; Wilcoxon over cases)

| cell | cases | noise d (p) | flat d (p) | real d (p) |
|---|---|---|---|---|
| glioma_t1n_t2f | 70 | -0.000 (1e-08) | -0.000 (0.004) | -0.000 (0.004) |
| glioma_t1n_t2w | 70 | -0.001 (6e-13) | -0.001 (4e-13) | -0.001 (4e-13) |
| glioma_t2w_t2f | 69 | -0.000 (5e-07) | -0.000 (4e-11) | -0.000 (6e-09) |
| glioma_t2w_t1n | 70 | -0.000 (8e-12) | -0.000 (7e-07) | -0.000 (4e-05) |

## Per outcome pattern (noise, flat, real): cos_hp_s1 and attr_s1 per arm, real - flat (p), cells with >= 3 cases

### CCC

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 60 | +0.000/+0.001/+0.000 | -0.000 (0.04) | -0.000 (0.2) | +0.081/+0.135/+0.081 | -0.051 (0.03) | +0.025 (0.4) |
| glioma_t1n_t2w | 67 | -0.000/-0.000/+0.000 | -0.000 (0.5) | -0.000 (0.5) | -0.011/-0.001/+0.002 | -0.005 (0.9) | +0.005 (0.6) |
| glioma_t2w_t2f | 68 | +0.001/+0.000/-0.000 | -0.000 (5e-05) | -0.001 (8e-13) | +0.049/+0.000/-0.013 | -0.032 (6e-05) | -0.095 (8e-13) |
| glioma_t2w_t1n | 62 | -0.000/-0.000/-0.000 | +0.000 (0.1) | -0.000 (0.0009) | -0.021/-0.243/-0.226 | +0.018 (0.4) | -0.202 (1e-08) |

### FFC

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 68 | +0.000/-0.001/+0.001 | +0.002 (2e-11) | +0.001 (7e-06) | +0.036/-0.379/+0.457 | +0.856 (1e-10) | +0.440 (2e-08) |
| glioma_t1n_t2w | 62 | +0.001/+0.000/-0.000 | -0.000 (0.03) | -0.001 (4e-05) | +0.102/+0.105/-0.108 | -0.117 (0.1) | -0.263 (0.001) |
| glioma_t2w_t2f | 58 | -0.000/+0.000/-0.000 | -0.000 (0.9) | +0.000 (0.6) | -0.032/+0.015/-0.041 | -0.025 (0.7) | -0.026 (0.9) |
| glioma_t2w_t1n | 55 | +0.000/+0.000/-0.000 | -0.000 (2e-05) | -0.000 (9e-05) | +0.113/+0.246/-0.241 | -0.500 (3e-05) | -0.273 (9e-05) |

### FCF

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 26 | +0.000/+0.000/-0.000 | -0.001 (0.03) | -0.000 (0.06) | +0.039/+0.255/-0.182 | -0.649 (0.03) | -0.282 (0.03) |
| glioma_t1n_t2w | 38 | -0.000/+0.000/-0.000 | -0.000 (0.04) | +0.000 (0.9) | -0.035/+0.100/-0.105 | -0.349 (0.01) | -0.093 (0.1) |
| glioma_t2w_t2f | 28 | +0.000/+0.000/-0.000 | -0.000 (0.5) | -0.000 (0.6) | -0.044/+0.100/-0.021 | -0.136 (0.5) | -0.019 (0.9) |
| glioma_t2w_t1n | 35 | +0.000/-0.000/-0.000 | -0.000 (0.8) | -0.000 (0.7) | +0.033/-0.039/-0.032 | -0.189 (0.9) | -0.050 (0.8) |

### CFF

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 18 | -0.000/-0.000/-0.000 | -0.000 (0.4) | -0.000 (0.8) | -0.121/-0.167/-0.228 | -0.190 (0.3) | -0.129 (0.6) |
| glioma_t1n_t2w | 36 | -0.000/+0.000/+0.000 | -0.000 (0.1) | +0.000 (0.03) | -0.085/+0.189/+0.112 | -0.114 (0.3) | +0.223 (0.03) |
| glioma_t2w_t2f | 34 | -0.000/+0.000/+0.001 | +0.000 (0.2) | +0.001 (8e-05) | -0.080/+0.445/+0.583 | +0.275 (0.007) | +0.879 (1e-06) |
| glioma_t2w_t1n | 62 | -0.000/+0.001/+0.001 | -0.000 (0.2) | +0.001 (8e-11) | -0.156/+0.612/+0.644 | +0.012 (0.2) | +0.756 (6e-11) |

### CCF

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 22 | +0.000/+0.000/+0.000 | -0.000 (0.4) | -0.000 (1) | +0.101/+0.237/+0.320 | +0.054 (1) | +0.114 (0.4) |
| glioma_t1n_t2w | 35 | +0.000/-0.000/-0.000 | +0.000 (0.7) | -0.000 (0.6) | +0.072/-0.004/-0.059 | +0.027 (0.6) | -0.127 (0.6) |
| glioma_t2w_t2f | 35 | -0.000/-0.000/+0.000 | +0.000 (0.001) | +0.000 (0.004) | -0.023/-0.067/+0.483 | +0.451 (0.0006) | +0.464 (0.0002) |
| glioma_t2w_t1n | 45 | -0.000/-0.000/+0.000 | +0.001 (1e-09) | +0.001 (6e-05) | -0.032/-0.424/+0.686 | +1.069 (2e-09) | +0.817 (3e-07) |

### CFC

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 31 | +0.000/-0.000/+0.000 | +0.001 (0.004) | +0.001 (0.007) | +0.069/-0.200/+0.161 | +0.475 (0.02) | +0.070 (0.06) |
| glioma_t1n_t2w | 26 | -0.000/-0.000/-0.000 | -0.000 (0.1) | -0.000 (0.6) | -0.046/+0.032/-0.155 | -0.148 (0.2) | -0.025 (0.3) |
| glioma_t2w_t2f | 49 | +0.000/+0.000/-0.000 | -0.000 (0.03) | -0.001 (4e-08) | +0.119/+0.206/-0.100 | -0.384 (0.01) | -0.477 (3e-10) |
| glioma_t2w_t1n | 57 | -0.000/+0.001/-0.001 | -0.001 (4e-10) | -0.000 (0.2) | -0.105/+0.626/-0.724 | -1.416 (4e-10) | -0.572 (1e-08) |

### FCC

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 59 | +0.000/+0.001/+0.001 | -0.000 (0.2) | +0.001 (1e-05) | +0.048/+0.391/+0.211 | -0.185 (0.009) | +0.227 (1e-05) |
| glioma_t1n_t2w | 66 | +0.000/+0.000/-0.000 | -0.000 (0.1) | -0.000 (0.007) | +0.106/-0.002/-0.018 | -0.027 (0.4) | -0.148 (0.005) |
| glioma_t2w_t2f | 55 | -0.000/-0.000/+0.000 | -0.000 (0.6) | +0.000 (0.02) | -0.086/-0.008/+0.008 | +0.008 (0.4) | +0.117 (0.04) |
| glioma_t2w_t1n | 48 | +0.000/+0.000/+0.000 | -0.000 (0.6) | -0.000 (0.5) | +0.067/+0.024/+0.027 | -0.081 (0.9) | +0.014 (0.8) |

### FFF

| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |
|---|---|---|---|---|---|---|---|
| glioma_t1n_t2f | 70 | +0.000/-0.001/-0.000 | -0.000 (0.4) | -0.001 (2e-08) | +0.027/-0.081/-0.164 | -0.060 (0.05) | -0.197 (1e-07) |
| glioma_t1n_t2w | 70 | +0.000/+0.000/+0.000 | +0.000 (0.2) | -0.000 (0.2) | +0.045/+0.056/+0.073 | +0.043 (0.04) | +0.039 (0.01) |
| glioma_t2w_t2f | 69 | +0.000/+0.000/+0.000 | +0.000 (0.4) | +0.000 (0.3) | +0.012/+0.065/+0.107 | +0.041 (0.03) | +0.090 (0.008) |
| glioma_t2w_t1n | 70 | +0.000/+0.000/+0.000 | -0.000 (0.6) | +0.000 (0.9) | +0.069/+0.116/+0.142 | +0.026 (0.05) | +0.053 (0.0002) |

