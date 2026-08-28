# ICP V2 — Baseline Protocol Freeze

Frozen before invasive ICP scoring.

## Unsupervised / no-ICP baselines

### B0: population constant
\[
\hat P(t)=9.5~mmHg
\]

Purpose: absolute-ICP sanity baseline. It has no trend information.

### B1: fixed-C static mechanistic equilibrium
At each reconstructed slow MAP value, solve the frozen pressure-balance equation
with `C=C_an`, population parameters, and no autoregulatory dynamics.

Purpose: asks whether V2 dynamics/autoregulation add value beyond a static
mechanistic pressure mapping.

## Supervised baselines (created only after label unlock)

These are not deployable without training data and are kept separate from the
blind V2 prediction.

### B2: MAP-only ridge
Nested leave-one-patient-out training. Features are derived only from the frozen
`A(t)` representation and its predeclared lagged summaries. All scaling and
regularization selection occur inside the training patients.

### B3: MAP + RR ridge
Same nested patient-level protocol, with ECG RR/HR features added after the ECG
QC/detector specification is frozen.

No normalization may use the held-out patient's invasive ICP.
No overlapping window is treated as an independent patient.
