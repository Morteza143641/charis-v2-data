# CHARIS V2 mechanistic ICP pipeline

This repository stores the public CHARIS release assets and a reproducible implementation of the frozen V2 mechanistic ICP experiment.

## Methodological separation

The workflow has separate jobs:

1. **blind prediction** — downloads a record and loads **ABP only**, creates frozen V2 predictions and blind B0/B1 baselines, hashes and uploads them as an Actions artifact;
2. **scoring** — starts only after blind artifacts exist, downloads the same record, loads invasive ICP only for the frozen reference/scoring protocol, and uploads per-patient results;
3. **cohort summary** — summarizes patient-level results without treating 1-Hz samples as independent patients.

The primary model, preprocessing, solver and scoring specifications are recorded in the `*_FREEZE.md` files. Changes after the V2 unlock require a new model version rather than overwriting V2.

## Release data

Release `v1` contains `charis1` through `charis13` `.dat`/`.hea` pairs. The GitHub Actions workflow verifies each downloaded release asset against GitHub's release-asset SHA-256 digest before processing.

## Local tests

```bash
python -m pip install -r requirements.txt
pytest -q
```

The workflow can also be started manually from **Actions → CHARIS V2 frozen cohort → Run workflow**.
