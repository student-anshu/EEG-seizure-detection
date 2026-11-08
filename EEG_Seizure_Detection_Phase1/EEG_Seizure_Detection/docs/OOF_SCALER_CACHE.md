# Inner OOF scaler statistics cache

## Purpose

The first outer fold holds out `patient_01`. Its remaining 22 patients are
used for inner model development. This cache stores per-recording raw EEG
channel sufficient statistics so each patient-held-out inner fold can construct
its own training-only channel scaler without repeatedly flattening all samples.

## Contents

- `evaluation/oof_scaler_cache.py`: population count/mean/M2 calculation,
  parallel-Welford merge, scaler finalization, and fold-specific `log_std`
  reconstruction.
- `scripts/build_oof_recording_stats_cache.py`: resumable per-recording
  channel statistics under `cache/oof_recording_stats_v1`.
- `window_channel_standard_deviations` and
  `log_std_from_raw_standard_deviations`: compact per-window raw standard
  deviations and fold-specific `log_std` reconstruction.
- `scripts/build_oof_window_std_cache.py`: resumable per-recording raw-std
  artifacts under `cache/oof_window_std_v1`.
- `scripts/run_inner_patient_oof.py`: resumable 22-patient OOF Logistic
  Regression runner and score artifacts under `cache/oof_patient_scores_v1`.

The builder processes only the first outer fold's 22 training patients. It
excludes the outer held-out patient before opening processed recordings.
Existing per-recording artifacts are reused only when their source signature
and array structure validate. A changed processed source invalidates its
corresponding artifact.

## Run

From the EEG Python project root:

```powershell
python -m pytest tests/test_oof_scaler_cache.py -q
python -m scripts.build_oof_recording_stats_cache --audit ..\\..\\docs\\full_annotation_audit.csv
python -m scripts.build_oof_window_std_cache --audit ..\\..\\docs\\full_annotation_audit.csv
```

The builder reads existing processed NPZ windows; it does not require original
EDF files, run Welch spectral extraction, train a model, or read outer-test
recordings. It can be safely rerun after interruption. Do not commit generated
cache files.

## OOF fold requirements

For each of the 22 inner held-out patients:
1. Merge statistics from the other 21 patients only.
2. Finalize the channel scaler using population variance and epsilon
   `1e-6`, matching `FoldChannelStandardizer`.
3. Use the compact per-window raw channel-std cache to reconstruct all 17
   `log_std` columns under that fold scaler. The other nine features are
   affine-scale invariant; preserve them only after row-identity and numerical
   tests. This also preserves the 422 low-variance validation windows clipped
   at the `log(1e-12)` floor.
4. Apply the SAME fold scaler to every recording in that OOF fold, including
   both the 21 training patients and the held-out patient. Never mix features
   normalized with different patient-fold scalers in one model fit.
5. Fit the model feature scaler on the 21 training patients only, train on
   those patients, and score only the held-out inner patient.

The OOF score collection and threshold-selection protocol must be completed
and frozen before the outer `patient_01` test labels are accessed. The cache
builder is not itself an OOF evaluation and does not claim model performance.

## OOF runner safeguards and restart behavior

- The raw-std cache is built once from existing processed NPZ windows; it does
  not need EDF files or spectral feature extraction.
- For each OOF patient, channel statistics from that entire patient are
  excluded. The same remaining-21-patient channel scaler is applied to every
  recording in that fold. The model-level `StandardScaler` is then fit on the
  21 training patients only.
- Fold feature matrices are built one at a time. A per-recording progress
  checkpoint allows the current fold's feature matrix to resume after an
  interruption. Once its score artifact is committed, temporary fold matrices
  are removed.
- Each completed patient score artifact is committed atomically. A restart
  reuses valid completed folds and resumes the first incomplete fold.
- The pooled OOF F2 threshold is provisional. The 22 probabilities come from
  distinct fold models and may differ in calibration; event-level operating
  points and score-transfer behavior must be reviewed before any threshold is
  frozen.
- OOF results are inner model-development evidence only. The outer
  `patient_01` recordings are never opened by these builders or runner.

The first recording-statistics cache has been built locally by the user for all
608 outer-training recordings. The focused statistics tests passed (8/8).
The raw window-std cache and 22-fold OOF runner still require local execution
and verification. Do not interpret committed code as a completed experiment.

After the second builder finishes, verify its manifest before training:
- `complete` is `true`.
- `recording_count` is `608`.
- `excluded_outer_test_patient` is `patient_01`.

Only then launch the 22-fold training runner:

```powershell
python -m scripts.run_inner_patient_oof --audit ..\\..\\docs\\full_annotation_audit.csv
```

Before OOF fitting, the runner compares reconstructed `log_std` against the
existing real inner-validation feature cache under the original scaler. It checks
all validation windows, requires maximum absolute error at most `5e-4`, and
requires zero mismatches at the `log(1e-12)` floor. If this real-data audit
fails, the runner stops before fitting any OOF model; do not bypass the check.
