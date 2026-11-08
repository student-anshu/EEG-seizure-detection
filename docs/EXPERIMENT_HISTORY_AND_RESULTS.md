# EEG Seizure Detection — Experiment History and Results

**Status date:** 6 November 2026  
**Working repository:** https://github.com/shivkumar15/EEG  
**Working branch:** `feature/cross-patient-feature-stability-audit`

This document records the completed model-development work and the evidence available at the current handoff. It is a research record, not a claim of clinical performance. Distinguish local artifacts/results from files committed to Git: large NPZ caches, checkpoints and result JSON files are generally local and are not expected to be in the repository.

## 1. Research task and non-negotiable safeguards

- Dataset: CHB-MIT Scalp EEG, PhysioNet v1.0.0, DOI `10.13026/C2K01R`.
- Task: binary EEG-window **seizure detection**, not prediction of future seizures.
- Academic research prototype only; not a medical device and not for diagnosis or treatment decisions.
- Primary cohort: 683 recordings, 185 seizure events, 23 independent patients. CHB01 and CHB21 are grouped as one person (`patient_01`). Three incompatible-montage recordings are excluded: CHB12 recordings 27, 28 and 29.
- 17 preferred channels, 256 Hz, 10-second windows, 5-second stride (50% overlap).
- Full processed cohort audit: 704,836 windows (702,280 normal; 2,556 seizure), 683/683 valid recordings.
- Never randomly split overlapping windows across train/test.
- Never fit scalers, feature selectors, model parameters, thresholds or alarm policy using an outer-test patient.
- `patient_01` is the outer-test patient for the first LOPO fold. It is sealed from feature selection, tuning, calibration and development decisions. Historical first-fold test metrics below are retained as a one-time descriptive result; do not use those labels to improve the model.
- Report window-level and event-level metrics separately. Accuracy alone is misleading under severe class imbalance.
- False alarms/hour denominator is non-seizure exposure: recording duration minus the union of clipped seizure intervals. Do not use raw recording exposure as the denominator and do not double-subtract overlapping annotations.

## 2. Cohort and split foundation

The primary cohort is 683 recordings / 23 patients. The full audit contains 686 EDF recordings and 198 events; three incompatible-montage CHB12 files are excluded from the primary cohort, leaving 185 events. The full annotation audit and signal inventory are repository documentation files:
- `docs/full_annotation_audit.csv`
- `docs/signal_inventory.csv`

The inventory matched all 683 primary recordings and provides exact durations. Total raw primary-cohort exposure is 979.9286 hours. This is **not** the non-seizure exposure; annotated seizure intervals must be unioned and subtracted.

Outer evaluation design: 23 Leave-One-Patient-Out folds. For the first outer fold, `patient_01` is held out; the other 22 patients (608 recordings) form the inner cohort. The first-fold patient validation split contains 17 fit patients and 5 validation patients. A separate 22-fold patient-grouped OOF pipeline was later run across all 22 inner patients, excluding patient_01.

## 3. Legacy 221-feature Logistic Regression first-fold baseline

**Purpose:** original feature-based baseline/reference, not the final chosen model.

- 13 features/channel × 17 channels = 221 features/window.
- 10-second windows, 5-second stride, 256 Hz.
- Logistic Regression.
- Threshold selected using inner-validation F1/F2 procedure: `0.9984662542771952`.
- Runner: `scripts/run_first_baseline_fold.py`.
- Result artifact (local): `results/baselines/first_fold_patient_01_f1_f2_v1.json`.
- Checkpoint (local): `results/baselines/checkpoints/first_fold_patient_01_logistic_regression_f1_f2_v1`.
- Score artifacts (local): `inner_validation_scores.npz`, `outer_test_scores.npz`.
- Runtime reported: approximately 6 hours.

### Window-level results

| Metric | Inner validation | Outer test patient_01 |
|---|---:|---:|
| AUROC | 0.78980 | 0.90428 |
| AUPRC | 0.01895 | 0.03039 |
| Sensitivity | 45.33% | 2.04% |
| Specificity | 97.04% | 99.88% |
| Precision | 3.52% | 4.55% |
| F1 | 0.06537 | 0.02817 |
| F2 | 0.13436 | 0.02294 |
| TP / FP / FN / TN | — | 3 / 63 / 144 / 52,549 |
| Windows | 216,005 | 52,759 |

Outer-test positive windows: 147. High AUROC did not translate to useful thresholded seizure detection.

### Event-level results

Source: `scripts/evaluate_first_fold_events.py`; result:
`results/baselines/first_fold_patient_01_event_metrics_v2.json`.

- Inner validation: 139 recordings, 29 events; 19 detected / 10 missed; event sensitivity 65.52%; mean delay 15.74 s; 1,327 false-alarm episodes; non-seizure exposure 1,078,457 s; 4.429 false alarms/hour.
- Outer test patient_01: 75 recordings, 11 events; 1 detected / 10 missed; event sensitivity 9.09%; mean delay 58 s; 40 false-alarm episodes; non-seizure exposure 263,536 s; 0.546 false alarms/hour.

This is a first-fold result only, not a 23-patient performance estimate. It is a warning about patient-level generalization.

## 4. Robust 170-feature Logistic Regression first-fold experiment

**Purpose:** a more scale-robust feature representation, evaluated against the legacy reference. It is not a proven winner.

Feature set: 10 features/channel × 17 channels = 170 features:
1. log standard deviation (`log_std`)
2. normalized line length
3. skewness
4. excess kurtosis
5. relative delta power
6. relative theta power
7. relative alpha power
8. relative beta power
9. relative gamma power
10. spectral entropy

Relative spectral shape and normalized temporal shape were intended to reduce amplitude sensitivity. Features were generated using vectorized Welch PSD. The first-fold robust model used Logistic Regression, chunked feature processing, and a scaler fitted on inner-fit only; the same fitted model generated validation and outer-test scores.

- Runner: `scripts/run_robust_feature_first_fold.py` (confirm exact CLI in the checked-out source before rerunning).
- Result (local): `results/baselines/robust_feature_first_fold_v1.json`.
- Score artifacts (local): `results/baselines/robust_feature_first_fold_v1/logistic_regression/validation_scores.npz` and `outer_test_scores.npz`.
- Threshold selected on inner validation by F2: `0.9852270300513666`.
- Runtime reported: 4,296 seconds (~71.6 minutes).

### Window-level results

| Metric | Inner validation | Outer test patient_01 |
|---|---:|---:|
| AUROC | 0.92644 | 0.90639 |
| AUPRC | 0.04014 | 0.02433 |
| Sensitivity | 47.28% | 6.80% |
| Specificity | 98.12% | 99.45% |
| Precision | 5.65% | 3.36% |
| F1 | 0.10098 | 0.04494 |
| F2 | 0.19119 | 0.05643 |
| TP / FP / FN / TN | 243 / 4,056 / 271 / 211,435 | 10 / 288 / 137 / 52,324 |
| Windows | 216,005 | 52,759 |

### Event-level results

Source: `scripts/evaluate_robust_first_fold_events.py`; result:
`results/baselines/robust_feature_first_fold_event_metrics_v1.json`.

- Inner validation: 139 recordings, 29 events; 27 detected / 2 missed; event sensitivity 93.10%; mean delay 14.15 s; 1,272 false-alarm episodes; non-seizure exposure 1,078,457 s; 4.246 false alarms/hour.
- Outer test patient_01: 75 recordings, 11 events; 4 detected / 7 missed; event sensitivity 36.36%; mean delay 36.5 s; 167 false-alarm episodes; non-seizure exposure 263,536 s; 2.281 false alarms/hour.

Comparison with legacy on patient_01: event detections improved from 1/11 to 4/11, but false alarms rose from 40 to 167 and FA/hour from 0.546 to 2.281. Three events were recovered; no legacy-detected event regressed. False alarms were concentrated in a few recordings, but cause is unknown. This does not establish that robust is superior overall.

**Decision:** preserve both as research references. Do not continue tuning against patient_01. The robust 170-feature model is the provisional baseline candidate for application integration only after its saved inference artifacts and preprocessing contract are verified.

## 5. Alarm-policy exploration

An exploratory minimum-positive-duration / gap grid was evaluated on inner validation only. It did not meet the research operating target of at least 90% event sensitivity and at most 1.0 false alarm/hour:

| Minimum alarm duration | Event sensitivity | FA/hour range across tested merge gaps |
|---|---:|---:|
| 0 s | 93.10% | 4.246 to 2.701 |
| 15 s | 86.21% | 2.267 to 1.505 |
| 20 s | 62.07% | 1.188 to 0.751 |
| 30 s | 34.48% | 0.624 to 0.404 |

Do not adopt a 30-second minimum-duration filter merely to reduce false alarms; it loses too many events. The 30-second merge gap was exploratory only. Future alarm policy must be validated using patient-grouped OOF data, but further accuracy optimization is now deferred while the complete application is built.

## 6. Patient-grouped inner OOF Logistic Regression

**Purpose:** evaluate ranking/generalization across the 22 development patients without using patient_01.

- Script: `scripts/run_inner_patient_oof.py`.
- 22 held-out-patient folds; each fold excludes its held-out patient from channel scaler and model fitting.
- 608 recordings, 652,077 windows; 2,409 seizure-positive windows.
- 170 robust features.
- Fold channel standardization is patient-grouped; model StandardScaler and Logistic Regression are fitted on the other 21 patients only.
- Raw standard-deviation cache and fold-specific log_std reconstruction are used to reproduce the fold feature values.
- Completed report (local): `cache/oof_patient_scores_v1/report.json`.
- Per-fold scores (local): `cache/oof_patient_scores_v1/folds/patient_XX.npz`.
- Provisional pooled OOF threshold: `0.98188316822052`, selected using OOF labels. This threshold and its metrics are descriptive/optimistic, not an independent test; fold score scales differ and may not be calibrated comparably.

### Pooled event-level results

Evaluator: `scripts/evaluate_inner_oof_events.py`. Output:
`results/baselines/inner_oof_event_metrics_v1.json`.

At threshold 0.5:
- 174 events; 162 detected / 12 missed; 93.10% event sensitivity.
- Mean delay 5.78 s; median delay 0 s.
- 29,146 false-alarm episodes.
- Non-seizure exposure 3,253,082 s.
- 32.254 false alarms/hour.

At provisional threshold 0.981883:
- 174 events; 75 detected / 99 missed; 43.10% event sensitivity.
- Mean delay 17.36 s; median delay 8 s.
- 3,164 false-alarm episodes.
- Same 3,253,082 s non-seizure exposure.
- 3.501 false alarms/hour.

Neither operating point meets both the research target (≥90% event sensitivity and ≤1 FA/hour). The pooled threshold was selected from the same OOF labels and is not an unbiased estimate.

### Patient-wise ranking

Macro AUROC 0.8148; macro AUPRC 0.1419. Ranking varies substantially:
- Stronger AUROC examples: patient_07 0.9925, patient_10 0.9733, patient_11 0.9708, patient_03 0.9639.
- Weak AUROC examples: patient_14 0.4401, patient_15 0.5265, patient_13 0.5361, patient_20 0.6404, patient_12 0.6434.
- Calibration or threshold tuning cannot repair weak ranking by itself.

At threshold 0.981883, several patients have very low or zero event sensitivity. Only 3 patients meet both ≥90% event sensitivity and ≤1 FA/hour. Do not quote pooled metrics without patient-wise variability.

## 7. Feature and patient stability diagnostics

### Original cross-patient distribution audit

- Script: `scripts/audit_cross_patient_feature_stability.py`.
- Test: `tests/test_cross_patient_feature_stability.py`.
- Doc: `docs/CROSS_PATIENT_FEATURE_STABILITY_AUDIT.md`.
- User execution: 22 patients, 7,480 patient-feature-class rows.
- Descriptive median absolute peer-IQR shift ~0.54; p90 ~1.51.
- Within-patient paired seizure vs non-seizure median shifts: log_std median +1.33 non-seizure-IQR units and positive direction in 89.04% of patient/channel comparisons. Other feature directions were mixed; excess kurtosis had median −0.11 and positive direction in 34.49%.
- Channel median absolute shifts were modest (largest examples around FZ-CZ 0.45, CZ-PZ 0.42).
- Exploratory patient-average shift vs OOF AUROC association: Spearman about −0.333 (n=22). Not causal or inferential evidence.
- Conclusion: patient variation exists; no feature removal justified.

### Fold-aware feature stability audit

- Script: `scripts/audit_fold_aware_feature_stability.py`.
- Tests: `tests/test_fold_aware_feature_stability.py`; user reported 4 passed in 2.50 s.
- Doc: `docs/FOLD_AWARE_FEATURE_STABILITY_AUDIT.md`.
- User execution completed: 22 folds, 7,480 patient-feature-class rows.
- It reconstructs each patient's matrix with the exact fold-specific patient-excluded channel scaler, compares patients under the same fold transform, and uses equal-weight peer-patient medians/IQRs.
- Reconstructs log_std from raw per-window channel standard deviations; reuses affine-invariant robust features.
- Existing OOF AUROC/AUPRC are context only.
- `patient_01` excluded before processed data access.
- Report flags: diagnostic_only=true, models_trained=false, thresholds_tuned=false, oof_scores_modified=false, outer_test_patient_loaded=false.
- Summary: 7,480 finite absolute peer-IQR shifts; median 0.5395; p90 1.5141.
- Patient ranking in the report matches the OOF report. No feature selection/removal or model changes are justified by this descriptive audit.
- Output (local): `results/diagnostics/fold_aware_feature_stability_v1/report.json` and `fold_aware_patient_feature_distributions.csv`. The CSV has not yet been reviewed in detail in the assistant conversation; do not claim feature/channel-level review is complete.

## 8. Patient 13/14/15 model-contribution diagnostics

- Script: `scripts/audit_patient14_model_contributions.py` generalized with `--patient`.
- Audits completed for patient_13, patient_14 and patient_15; reconstructed probabilities matched OOF probabilities exactly (maximum difference 0).
- Output JSON and per-recording/channel-feature CSVs were created locally for each patient.
- Contribution directions varied substantially between patients. There is no stable universal seizure signature in the fitted model contributions.
- Example aggregated positive-minus-negative logit contributions for patient_14: beta −1.610, theta −0.612, gamma −0.511, delta −0.305, kurtosis −0.214, alpha +0.019, skewness +0.297, entropy +0.373, log_std +0.667, normalized line length +1.376.
- These are additive fitted-logit contributions, not causal or physiological effects. Overlapping windows are correlated; small seizure-positive supports increase uncertainty.
- The user's ZIP of patient 13/14/15 audit outputs was reported at the local project root. Its contents are not verified by this document.
- Decision: do not remove or reverse features based on these diagnostics.

## 9. Current scientific conclusion

The existing model pipeline is a usable **research baseline**, not a high-performing or clinically reliable detector. It demonstrates that:
- Feature-based Logistic Regression can rank seizure windows reasonably for some patients.
- Patient-level generalization is highly variable.
- Thresholded event sensitivity and false alarms trade off sharply.
- Some patients have very weak ranking, which calibration cannot fix.
- The 170-feature robust representation is not an established winner over the legacy 221-feature reference.

No additional accuracy experiment is required before building the complete software system. Preserve all existing outputs and methods. Accuracy improvement is deferred to a future research milestone and must remain documented, reproducible and patient-wise.


## 10. Software and artifact integrity verification

These are user-reported local verification outputs recorded in the project work history. They are software checks, not evidence of clinical or model generalization performance.

- Full repository suite after OOF event evaluator changes: **213 passed, 4 subtests passed, in 3.87 seconds**.
- Fold-aware feature stability focused tests: **4 passed in 2.50 seconds**.
- Raw-standard-deviation reconstruction audit: 3,672,085 values checked; maximum absolute error 2.3841858e-07; zero values over tolerance; zero floor mismatches.
- Patient 13/14/15 contribution audit reconstruction: maximum probability difference 0 for each audited patient.
- Patient 14 feature diagnostic affine-invariance check: maximum absolute difference 7.6293945e-06; no values exceeded the stated 1e-4 tolerance.
- OOF artifact integrity: 22 fold files, 22 patient folds, 652,077 windows, 2,409 positive windows; patient_01 excluded.
- Fold-aware audit completed: 22 folds, 7,480 patient-feature-class rows; report flags confirm no model training, no threshold tuning, no OOF score modification and no patient_01 access.

Do not infer that these tests establish a reusable deployable model. The next artifact-readiness audit must separately verify model serialization, scaler(s), feature ordering and end-to-end inference parity.
