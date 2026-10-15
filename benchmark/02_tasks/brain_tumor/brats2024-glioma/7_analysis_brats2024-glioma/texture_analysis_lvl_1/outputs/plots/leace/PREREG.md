# Pre-registration: LEACE erasure of the local texture-energy subspace inside the network (visual proof v17, FINDINGS 44), written 2026-10-14 before any extraction

Goal: the published bridge from "decodable" (section 31: residualised fine texture energy is linearly decodable from every stage of every arm) to "used"
(LEACE concept erasure, Belrose et al. 2023; literature memo `LITERATURE_TEXTURE_EVIDENCE_FORMS.md` A9 and section 5 item 1). Question: does the real-fill
model's edema decision rest on the linearly decodable local texture-energy subspace more than the flat-fill and noise-fill models' decisions do?
Wording throughout: "local texture-energy subspace", never "texture pattern".

## Cells, arms, cases
- Cells (model trained on the first contrast, tested on the second): glioma_t2w_t1n, glioma_t2w_t2f, glioma_t1n_t2f, glioma_t1n_t2w; 70 test cases per cell,
  sorted by name. FIT cases = the first 20; EVALUATION cases = the other 50 (the v13 stitch split). All numbers and tests are read on the 50 evaluation cases.
- Arms: noise (rung 4, `*lblvor*`), flat (rung 4.5, `*flatfill*`), real (rung 5, `*v26_6_2_train050_val000*`), folds 0-2; plus the architectural floor
  "init" = the real arm's network with every module re-initialised by `reset_parameters` (torch seed 1000 + k for "fold" k), with its OWN LEACE fit.
- Input: the v10 plan patch per case (`visual_proof_switch.load_case`), natural test images only, no input intervention. fp32, no autocast.

## Concept Z (image-derived, mask-free, label-free)
Per case, at full patch resolution (FINDINGS 31 descriptors, `visual_proof_texture_content.descriptors`): I = G2(x) (local mean, sigma 2), E = |grad G2(x)|
(sigma 2), Tf = sqrt(G2((x - G1(x))^2)) (fine texture energy). Residualised texture r = log(Tf + 1e-6) minus its per-case OLS fit on [1, I, log(E + 1e-6)]
over the case's body voxels (FINDINGS 31 `residualise`). Order of operations, fixed: (1) descriptors and residualisation at full resolution over body voxels;
(2) downsample to the stage's grid as the body-masked average over the stage's cell, area(r * body) / area(body) (`interpolate(mode="area")`); stage body
mask = area(body) >= 0.5; (3) per case, quantile-bin the stage-resolution values over the case's stage-body voxels into K = 8 bins -> one-hot Z (8 columns).
Control concept Zc: the same with I (not residualised), K = 8.

## LEACE (closed form, Belrose et al. 2023)
Per cell x arm x fold x stage L, moments accumulated in float64 over the stage-body voxels of the 20 FIT cases: mu = mean(h), Sigma = Cov(h),
Sigma_hz = Cov(h, Z). W = (Sigma + lam I)^(-1/2), W+ = (Sigma + lam I)^(1/2) (eigendecomposition), lam = 1e-3 trace(Sigma) / C. Orthonormal basis U of
colspace(W Sigma_hz) from its SVD, rank r = number of singular values > 1e-6 x the largest (r <= K - 1 = 7 because the one-hot columns sum to 1).
P = U U^T; eraser A = W+ P W; erasure h' = h - A (h - mu), applied per voxel along channels, at EVERY voxel of the patch (body and background alike),
one fixed (mu, A) per cell x arm x fold x stage. Sanity check logged at fit time: in-sample |Cov(h', Z)| / |Cov(h, Z)| ~ 0.
Stages L: skips s2, s3, s4 and decoder stages d1, d2, d3 (section 42: texture evidence enters at s3 and d1 / d2). Resolutions: s2 = d2 (1/4), s3 = d1 (1/8),
s4 (1/16), d3 (1/2).
Semantics (stated, not a choice made after the fact):
- Skip erasure = SKIP-PATH erasure: s_j is erased only in the tensor concatenated into the decoder; the encoder's own downstream path (s_{j+1} .. s5 ->
  bottleneck) is computed from the unerased activation and still carries the concept (same semantics as the section 42 patching).
- Decoder erasure: the output of decoder stage d_s (post-nonlinearity, the tensor that feeds the next transposed convolution) is erased.
- Erasure is applied AFTER the nonlinearity. The next block normalises per instance and per channel, so the part of an erasure that is a per-case constant
  shift of a channel is largely absorbed; the per-voxel effect maps are therefore read as effects on the spatially varying part of the representation.

## Controls (per cell x arm x fold x stage)
- (a) RANDOM subspace of the same rank r: U_rand = orthonormal basis of colspace(W G), G ~ N(0, 1) of shape C x r (20 draws, seeded by cell, arm, fold, stage,
  draw), A_rand = W+ U_rand U_rand^T W. In whitened space every direction has unit variance, so a random rank-r subspace removes the same whitened variance
  as the texture subspace: this is the fair equal-rank control (Elazar 2021; Rozanova 2023 caution about high-variance random controls). Reported: mean over
  draws, and the 95th percentile over the 20 draws (with 20 draws this is the 19th-20th order statistic, i.e. close to the maximum; named as such).
- (b) INTENSITY erasure: the same LEACE with Zc (its own rank r_int <= 7).
- (c) PROBE-AT-CHANCE (gate, read BEFORE any effect number): ridge probes (lam = 1e-3 trace / C, no CV) fitted on the FIT-case moments, before erasure on h
  and REFITTED after erasure on h' (closed form: Sigma' = (I - A) Sigma (I - A)^T, Cov(h', y) = (I - A) Cov(h, y), mean unchanged), scored as held-out
  R^2 = 1 - SSE / SST on the 50 EVALUATION cases (SST around the pooled evaluation mean). Gate target = the BIN INDEX of the residualised texture
  (a fixed linear function of the one-hot Z, the quantity LEACE removes): held-out R^2 after texture erasure <= 0.05 x its pre-erasure value OR <= 0.02
  absolute, in every cell x arm x stage (trained arms), else LEACE is not doing its job and the fit is fixed before any result is read. Reported alongside,
  NOT gated: the continuous residual Tf (LEACE does not remove the within-bin variance, so its R^2 need not reach chance), the intensity bin index after
  intensity erasure, and the texture bin index / continuous after random erasure (draw 0; expected to stay near the pre value).
- (d) RANDOM-INIT FLOOR: arm "init" (see above), same erasures and controls with its own fit.
- (e) IN identity: nothing extra; erasure after the nonlinearity, see Semantics.

## Outcomes (per evaluation case; fold-mean over folds 0-2)
J(v) = edema (class 2) log-odds at v, fold-mean of the per-fold log-odds (as in section 43 primary, target class for all arms). E_L(v) = J(v) - J_L'(v)
(texture-erasure effect map; positive = the texture-energy subspace at L pushes the edema belief UP); E_L^rand(v) = mean over the 20 draws;
E_L^int(v) = intensity-erasure effect. Scalars per case (on fold-mean maps): mean of E over GT edema voxels (target), over the peritumoural ring (dilation 10
minus target, in body), over far background (body minus dilation 10); fraction of target voxels with E > 1 nat; target Dice on the patch (argmax of the
fold-mean softmax == class 2 vs GT == 2) before and after erasure (texture, intensity, random draw 0). One value per case, no median over seeds.
Per draw, the target mean of the fold-mean random effect is kept (for the mean, the 95th percentile and the depth-profile band).

## Tests and predictions (Wilcoxon signed-rank, two-sided, over the 50 evaluation cases per cell, p < 0.01, direction by the median paired difference)
- P1 (headline stage d2, chosen a priori; every stage reported): target-mean E_d2 real > flat in >= 3/4 cells AND real > noise in 4/4 cells.
- P2 (real arm, d2): texture erasure exceeds random-subspace erasure, i.e. per-case E_tex - E_rand_p95 > 0 (paired, p < 0.01) in 4/4 cells; and
  E_tex - E_int > 0 (p < 0.01) in >= 3/4 cells. Also reported: E_tex - E_rand_mean and the fraction of cases with E_tex > E_rand_p95.
- P3 (depth): the real arm's texture-specific profile, median over cases of (E_tex - E_rand_mean) per stage, peaks (argmax over s2, s3, s4, d1, d2, d3)
  at s3, d1 or d2 in >= 3/4 cells. Raw real profile and real - flat profile reported.
- P4 (readability, by eye, no test): in the example figure the real arm's edema prediction visibly shrinks / collapses after texture erasure at d2 and not
  after random erasure, and the flat arm's moves less.
- Kill: P1 fails -> the real-fill model's decision does not rest more on the linearly decodable texture-energy subspace; report and stop.
  Probe gate fails -> fix the fit (K, lam, stage-body definition) on the smoke / fit data only, before any effect number is read.

## Visuals (patient slices: NOT committed; kept on Vulcan only)
Per cell, two evaluation cases (closest to the cell median and to the 75th percentile of the per-case real - flat target E_d2), slice = the axial slice
with the most GT edema in the patch; rows = real / flat / noise; columns = image | prediction (edema = argmax class 2, GT edema contour in lime) |
prediction after texture erasure at d2 | prediction after random erasure at d2 (ONE draw, draw 0; the mean over draws would wash out) | E_d2 painted on the
image (orange = the texture-energy subspace supported the edema call, cyan = opposed; shared symmetric scale = 99th percentile of |E| over the three arms;
floor-subtracted: zero where |E| <= |E_d2| of the init arm). Side panel: depth profile of the median target E per stage, one curve per arm, random-erasure
band (2.5-97.5 percentile over the 20 draws, median over cases) shaded, init floor dotted. Summary figure without patient data: per cell, bars of median
target E_d2 for the three arms with the random (mean, 95th percentile) and intensity controls and the init floor -> `leace_summary.png` (committable).

## Outputs
`outputs/plots/leace/{PREREG.md, leace.md, leace_cells.csv, leace_depth.csv, leace_controls.csv, leace_probe.csv, leace_summary.png/pdf,
leace_examples_<cell>.png (patient data), <cell>/fit/*.npz, <cell>/<case>_leace.csv, <case>_leace_maps.npz (patient data)}`.

## Smoke (before the run; not read for results)
1 cell, 3 fit + 2 evaluation cases, fold 0, 2 random draws, into a separate smoke dir: checks the fit sanity (in-sample Cov(h', Z) ~ 0), that decoder
re-entry without erasure reproduces the natural logits, the probe gate mechanics and the completeness of the outputs.
