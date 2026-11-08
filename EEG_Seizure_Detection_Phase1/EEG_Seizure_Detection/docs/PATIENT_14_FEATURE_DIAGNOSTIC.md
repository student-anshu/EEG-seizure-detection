# Patient 14 Feature Diagnostic

The patient-14 diagnostic is a read-only investigation of the weak OOF fold. It does not fit a model, change existing score artifacts, select features, reverse feature directions, or tune thresholds. Patient_01 remains excluded.

Run from `EEG_Seizure_Detection`:

```powershell
python -m scripts.diagnose_patient14_features --audit ..\..\docs\full_annotation_audit.csv
```

The script uses the patient-14 fold's channel scaler, fitted from the other 21 inner patients, and the existing processed NPZ windows. It verifies recording identity, OOF offsets, window counts, and label alignment before computing diagnostics.

## Outputs

- `results/diagnostics/patient_14_feature_diagnostic_v1.json`: cohort counts, per-recording summary, feature-level recording-macro summaries, and invariance audit summary.
- `results/diagnostics/patient_14_feature_separation.csv`: within-recording AUC, direction, and positive/negative medians for every channel-feature pair in each seizure-containing recording.
- `results/diagnostics/patient_14_affine_invariance.csv`: numerical differences for the nine features expected to be invariant to per-channel affine scaling, comparing patient-14-fold recomputation with the existing first-fold cache.

## Interpretation safeguards

- AUC below 0.5 means that feature values tend to be lower in seizure windows in that recording. It is not permission to reverse the feature or prediction score.
- Recording-level positive-window counts are small, so AUC estimates can be noisy. Do not treat a single feature or recording as conclusive.
- The nine non-`log_std` features are mathematically invariant to ideal per-channel affine scaling. Float32 arithmetic can still introduce numerical differences; inspect the reported error distribution before deciding whether cache reuse is sufficiently accurate.
- This is exploratory diagnostics only. Do not use patient-14 labels to select features, train models, or set patient-specific thresholds.
