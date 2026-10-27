# Cloud-Assignment Evaluation Utilities

This repository now includes a calibrated evaluation runner for trajectory-imputation assessment.

## Calibrated protocol runner

Run:

```bash
python calibrated_protocol.py \
  --data-root data_corrupted \
  --checkpoint lstm_baseline.pt \
  --output-dir reports/calibrated_protocol \
  --test-seeds 17 18 19 20
```

### Fixed protocol settings

- Input window: `30 x 7` at `10 Hz`
- Label horizon: `50 x 2`
- Penetration rates: `10%, 20%, 30%`
- Metrics: Euclidean ADE/FDE (metres), masked-cell reconstruction MAE
- Lane-change criterion minimum: `n >= 200`

### Outputs (machine-readable + artifacts)

- `cv_baseline_suite.csv` (CV dead-reckoning across conditions/MPRs, mean ± seed SD)
- `calibration_report.json` and `calibration_report.md` (pre-registered pass/fail/inconclusive criteria)
- `clean_lstm_overlays.png` (20 random clean-window overlays)
- `table3_matched_control_factors.csv` (+ conditional `table3_matched_control_metrics.csv`)
- `week7_garnn_reconstruction.json` (if reconstruction arrays are present)
- `docs/sfm_graph_attention_argument.md`
- `docs/what_we_could_not_verify.md`
- `run_manifest.json`

### Branch B behavior

If any determinate calibration criterion fails, the runner marks `branch_b_required` and executes:

- normalization audit,
- x/y ordering check,
- 100-window overfit memorization check,
- repair specification output.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover metric correctness, CV behavior, sigma-ratio criterion behavior (including `n<200` inconclusive), matched-control structure, and GARNN temporal branch tensor behavior.
