# V2 Cohort Analysis Freeze 1

**Timing:** specified after the CHARIS6 technical first run and before running the frozen V2 pipeline across the remaining CHARIS cohort. No V2 model, preprocessing, runtime-solver, or patient scoring rule is changed here.

## Unit of inference

The patient/record is the inferential unit. 1-Hz samples and overlapping trend pairs are never treated as independent patients for cohort confidence intervals.

All 13 records are tabulated. A metric is summarized across every patient for which that metric is mathematically evaluable; the number of evaluable patients is always reported. No patient is excluded because of poor V2 performance.

## Absolute ICP cohort summaries

For V2, B0, and B1 patient-level MAE/RMSE/bias are retained. The cohort report gives:

- median and IQR of patient-level metrics;
- equal-patient mean and a 95% percentile bootstrap interval using 10,000 patient-level resamples;
- fixed bootstrap seed `20260829`.

Paired patient-level MAE differences are reported as:

- `V2 MAE - B0 MAE`;
- `V2 MAE - B1 MAE`;
- number of patients in which V2 has lower MAE than each baseline.

## Trend cohort summaries

For 5, 15, and 30 minute frozen horizons, summarize patient-level:

- delta-MAE;
- Spearman association;
- concordance correlation coefficient;
- direction accuracy.

No lag optimization or patient-specific temporal alignment is added at cohort stage.

## Elevated ICP

At 22 mmHg primary and 20 mmHg secondary thresholds, report:

- number of patients containing at least one evaluable sustained reference event;
- total sustained reference events and detected events, descriptively pooled;
- macro patient-level event sensitivity among patients with reference events;
- median patient-level AUROC/AUPRC among patients in which the time-point classification metric is defined.

The pooled event count is descriptive and is not treated as an independent-patient confidence interval.

## Missingness / coverage

Report score-eligible hours, paired hours, and reference-QC exclusions per patient and cohort totals.

## Interpretation

This script produces descriptive and patient-level uncertainty summaries only. It does not automatically modify V2 or tune thresholds. Any mechanistic change after inspecting the cohort is a new named model version (V2.1 or later).
