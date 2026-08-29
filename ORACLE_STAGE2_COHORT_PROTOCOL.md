# CHARIS V2 Oracle Stage-2 Cohort Protocol — freeze 1

## Status and purpose
This is a post-primary, nondeployable diagnostic oracle analysis. The frozen V2 equations, primary predictions, preprocessing, segmentation, burn-in, reference construction, and primary scoring remain unchanged. Stage 2 asks whether a small patient-specific parameter set can rescue late-record dynamics after calibration on early invasive ICP.

This analysis is **not** a primary validation result and must not be described as calibration-free. It uses invasive ICP for parameter selection.

## Parameters profiled
The mechanistic equations are unchanged. Four quantities are personalized jointly:

- segment initialization pressure `P0`: 3 to 18 mmHg (linear search coordinate),
- `R_o` multiplier relative to 526.3 mmHg*s/mL: 0.03 to 20 (log10 search coordinate),
- `k_E` multiplier relative to 0.11 1/mL: 0.25 to 4 (log10 search coordinate),
- `G` multiplier relative to 1.5: 0.10 to 4 (log10 search coordinate).

The lower `R_o` bound is wider than Stage 1 because Stage 1 frequently selected its 0.1x lower boundary. This widening is explicitly exploratory and data-informed; parameter values from Stage 2 are therefore not physiological estimates.

All other V2 parameters remain frozen. No output offset is included in the Stage-2 four-parameter fit. `P0` is the mechanistic state initialization at each retained segment, not an additive observation offset.

## Replay and validation
The Stage-1 fast replay is reused: frozen 1-Hz `A_map` and analytic `A_dot`, cubic Hermite interpolation between 1-Hz samples, classical RK4 with 0.5-s substeps, full autoregulation, and the same segment resets. Before any oracle optimization is accepted, the population setting (`P0=9.5`, all multipliers 1) must reproduce the stored frozen V2 prediction with:

- mean absolute replay difference <= 0.02 mmHg,
- 99th-percentile absolute replay difference <= 0.15 mmHg,
- maximum absolute replay difference <= 0.5 mmHg.

Failure of this gate fails the patient job.

## Optimization
Objective: mean absolute error (MAE) against invasive ICP on the fit subset. Search is deterministic SciPy differential evolution with:

- transformed coordinates described above,
- `init='sobol'`,
- `seed=20260829`,
- `popsize=5`,
- `maxiter=10`,
- `tol=0`, `atol=0`,
- `polish=False`,
- `updating='immediate'`, `workers=1`.

The fixed evaluation budget is intentional. The population parameter setting is evaluated separately and retained if it beats the optimizer result. No search settings are changed after seeing Stage-2 results.

## Two diagnostic fits
1. **Whole-record oracle:** fit the four parameters using every primary score-eligible paired ICP row. This is an upper bound only.
2. **Early-25% -> late-75% transfer:** fit on the chronological first 25% of paired rows and evaluate, without refitting, on the remaining 75%.

The split is identical in definition to Stage 1. No segment-specific fitting is allowed.

## Frozen evaluation outputs
For whole and transfer fits report absolute MAE/RMSE/bias/Pearson/Spearman. For the late 75% additionally report the frozen 5/15/30-minute trend metrics and sustained elevated-ICP endpoints at >22 mmHg (primary) and >20 mmHg (secondary). The first 60 s after the transfer boundary are naturally excluded from strict trailing-median trend windows.

Patient-level parameter locations and near-boundary selections are reported. Cohort summaries use equal patient weight with 10,000 patient bootstrap resamples and seed 20260829.

## Interpretation lock
- Large MAE improvement with late Pearson/Spearman and trend metrics still near zero supports a structural/observability limitation rather than a simple population-parameter mismatch.
- Stable positive late tracking across patients after early calibration would support patient-specific parameter mismatch as a material contributor.
- Stage 2 is diagnostic and cannot rescue or revise the frozen primary V2 claim.
- No equation change or CHARIS-driven tuning may be relabeled as V2 primary; any subsequent mechanistic change is V2.1 exploratory.
