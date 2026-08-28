# V2 Scoring Freeze 1

**Frozen after CHARIS6 blind prediction hash, before loading invasive ICP values.**

## Reference alignment

The invasive ICP channel is sample-synchronous with ABP in the same WFDB file.
No cross-correlation, lag optimization, or post-hoc temporal shifting is allowed.

For each frozen 1-Hz prediction timestamp `t`, reference ICP is the robust slow-time
summary of the raw 50-Hz ICP samples in the centered interval

`[t-0.5 s, t+0.5 s)`.

The primary summary is the **median**. At least 40 of 50 finite in-domain samples
are required. A deliberately broad predeclared reference sensor-domain gate
`-50 <= ICP <= 150 mmHg` is used only to reject gross nonphysiologic/sensor values.
No prediction-dependent or patient-specific ICP exclusion is allowed.

Only rows already marked `valid_for_score=1` by the blind predictor are eligible.
The reference cannot create new prediction segments or alter burn-in.

## Absolute ICP endpoints

Primary descriptive metrics per patient:
- bias = mean(prediction - reference)
- MAE
- RMSE
- median absolute error
- Pearson and Spearman association (descriptive only)
- Bland-Altman mean difference and 95% limits of agreement

No overlapping 1-Hz sample is treated as an independent patient for confidence
intervals. For the full cohort, inference is patient-level; CHARIS6 alone is a
technical first run and receives no pseudo-population CI.

## Trend endpoints

Within each frozen prediction segment, define a 60-s trailing median for both
prediction and reference. For horizons `h = 5, 15, 30 min`:

`Delta_h(t) = median60(t+h) - median60(t)`.

Pairs must stay in the same segment and both endpoints must be score-eligible.
Report MAE of Delta, Spearman correlation, concordance correlation coefficient,
and direction accuracy. Direction accuracy is primary without a deadband;
secondary direction accuracy uses only reference changes with `|Delta| >= 2 mmHg`.
No lag fitting is allowed.

## Elevated-ICP endpoint

Primary threshold: reference ICP `>22 mmHg` sustained for at least `300 s`.
Secondary threshold: `>20 mmHg` sustained for at least `300 s`.

Report time-point AUROC/AUPRC for the continuous prediction, plus thresholded
sustained-event sensitivity, false alarms/hour and time-to-detection when the
patient contains evaluable events. Event definitions are not altered after
seeing prevalence.

## Blind baselines

B0: constant `9.5 mmHg`.

B1: frozen static mechanistic equilibrium with `C=C_an`, population parameters,
and each already-frozen `A_map_mmHg`; no ICP is used to produce B1.

Supervised MAP-only / MAP+RR baselines are deferred to the full patient-level
nested training protocol and are not trained on CHARIS6 alone.

## Missingness

Metrics use pairwise rows with a valid blind prediction and valid 1-Hz reference.
The number and fraction excluded by reference QC are always reported.

## Version rule

Any change to these rules after the first invasive ICP value is loaded creates
`V2.1` and does not overwrite V2 results.

## Sustained-event matching detail (pre-unlock addendum)

At 1 Hz, a sustained run is >=300 consecutive score-eligible seconds above the
specified threshold within one frozen segment. A reference sustained event is
counted detected if at least one predicted sustained run temporally overlaps
it. A predicted sustained run that overlaps no reference sustained event is a
false-alarm event. False alarms/hour uses total score-eligible hours as the
denominator. For each detected reference event, time-to-detection is
`max(0, first_overlapping_predicted_run_start - reference_event_start)`.
