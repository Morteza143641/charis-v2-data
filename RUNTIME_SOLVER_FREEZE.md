# V2 Runtime Solver Freeze 1

**Timing:** created after inspecting ABP-only computational behavior of CHARIS6, but before loading or scoring invasive ICP.

The mathematical model, population parameters, beat detection, MAP definition, causal A(t), artifact gates, segmentation, initialization, burn-in and 1-Hz output grid are unchanged.

The original reference implementation used SciPy DOP853 with `rtol=1e-8`, `atol=1e-10`, `max_step=0.5 s`. On a 46-hour record, adaptive integration is computationally impractical because `dA/dt` changes at every completed arterial beat.

For execution only, V2 uses classical fixed-step RK4 with:

- `dt = 0.02 s` (the native CHARIS 50-Hz sample interval),
- float64 arithmetic,
- the exact same V2 right-hand-side equations,
- the exact same continuous causal `A(t)` representation,
- independent initialization of each frozen valid segment.

This is a numerical-runtime substitution, not a model or preprocessing change.

## Acceptance gate

Before invasive ICP is read:

1. synthetic dynamic tests must agree with frozen DOP853 to max absolute ICP difference `<0.005 mmHg`;
2. selected ABP-only CHARIS6 segments must agree with DOP853 to max absolute ICP difference `<0.005 mmHg` and max compliance difference `<5e-5 mL/mmHg`;
3. all states remain finite and positive and `A-P` remains positive.

If these gates fail, this runtime solver is rejected. No tolerance is adjusted after ICP scoring.
