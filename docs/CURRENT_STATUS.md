# Current Project Status

**Last reconciled:** 6 November 2026  
**Working repository:** https://github.com/shivkumar15/EEG  
**Active branch:** `feature/cross-patient-feature-stability-audit`  
**Current milestone:** Research handoff complete; saved model/scaler and cached-feature parity have been verified locally; raw processed-EEG inference parity remains pending before formal baseline freeze and FastAPI integration.

> This file supersedes older status statements in this document that identify a different active branch, describe event evaluation as pending, or recommend more accuracy experiments as the immediate next step. Full experimental history is in [EXPERIMENT_HISTORY_AND_RESULTS.md](EXPERIMENT_HISTORY_AND_RESULTS.md).

## Project objective and current CTO direction

Complete an end-to-end academic EEG seizure-detection application using the existing CHB-MIT work. The priority is now **complete, reproducible system delivery**, not endless model accuracy optimization. Accuracy is not ignored: every completed experiment, metric and limitation remains documented; further model improvement is deferred to a future research milestone and is not a blocker for backend/frontend completion unless a genuine inference blocker is discovered.

This is seizure detection, not prediction of future seizures. This is not a medical device and must not be used for diagnosis or treatment decisions.

## Verified cohort and preprocessing

- Dataset: CHB-MIT Scalp EEG, PhysioNet v1.0.0, DOI `10.13026/C2K01R`.
- Full archive audit: 686 EDF recordings, 198 events, 24 subject folders.
- Primary cohort: 683 recordings, 185 events, 23 independent patients.
- CHB01 and CHB21 are grouped as the same person (`patient_01`).
- Excluded incompatible-montage recordings: CHB12 recordings 27, 28 and 29.
- 17 preferred channels, 256 Hz; 10-second windows, 5-second stride.
- Full processed integrity audit: 683/683 valid; 704,836 windows (702,280 normal; 2,556 seizure).
- Exact duration inventory matches 683/683 primary recordings. Raw exposure is 979.9286 hours; FA/hour must use non-seizure exposure after subtracting the union of clipped seizure annotations.

The original EDF files were deleted locally after earlier guidance. Existing processed NPZs, metadata, caches and audit artifacts remain. **Do not ask the user to download/reprocess EDFs as a routine step.** Verify local artifact availability before using any file.

## Research work completed

1. Dataset annotation/cohort audit and processed data integrity audit.
2. Patient-wise LOPO split construction and leakage checks.
3. Fold-wise preprocessing and feature iteration.
4. Legacy 221-feature Logistic Regression first-fold baseline.
5. Robust 170-feature Logistic Regression first-fold baseline.
6. Saved-score window/event evaluation for both first-fold experiments.
7. 22-fold patient-grouped inner OOF Logistic Regression across 22 development patients (patient_01 excluded).
8. Pooled and patient-wise OOF event evaluation.
9. Alarm duration/gap exploration on development validation data.
10. Original cross-patient feature-distribution audit.
11. Fold-aware feature stability audit (22 folds, 7,480 patient-feature-class rows).
12. Patient 13/14/15 model-contribution diagnostics.
13. Numerical/cache integrity checks and regression tests.

See [Experiment History and Results](EXPERIMENT_HISTORY_AND_RESULTS.md) for exact metrics, paths, interpretations and limitations.

## Key model findings (research baseline, not clinical performance)

- Legacy 221-feature first-fold outer test patient_01: window sensitivity 2.04%; event sensitivity 1/11 (9.09%); 0.546 false alarms/hour.
- Robust 170-feature first-fold outer test patient_01: window sensitivity 6.80%; event sensitivity 4/11 (36.36%); 2.281 false alarms/hour.
- Inner 22-patient OOF at provisional threshold 0.981883: event sensitivity 43.10% (75/174), 3.501 false alarms/hour.
- Inner OOF at threshold 0.5: event sensitivity 93.10% but 32.254 false alarms/hour.
- Macro OOF AUROC 0.8148, macro AUPRC 0.1419; patient-wise ranking varies substantially.
- Neither OOF operating point meets the exploratory research target of ≥90% event sensitivity and ≤1 FA/hour.
- The pooled OOF threshold was selected using OOF labels; its metrics are descriptive/optimistic. Fold scores may not share calibrated scales.
- The 170-feature robust model is **not proven superior** to the 221-feature legacy reference. It is only the provisional application candidate pending artifact/inference verification.

## Local artifact verification update — 6 November 2026

The following checks were completed in the primary developer's local environment and reported in the project work history. They have not yet been independently reproduced by a fresh teammate clone:

- Robust 170-feature Logistic Regression model deserialized successfully.
- Fold-specific scaler checkpoint restored and finalized successfully.
- Validation feature cache checked: 216,005 rows × 170 features.
- Cached-feature → model → archived validation-score parity passed for 216,005 rows; maximum absolute probability difference approximately 2.33e-15.
- This establishes parity from the saved feature matrix onward only. **Raw processed EEG → preprocessing → scaler → feature extraction → model parity remains pending.**
- The formal model baseline is not frozen. Do not describe the inference service as complete or deployable.
- Local-only artifacts still require a verified transfer bundle and checksum manifest. See [Teammate Onboarding](TEAMMATE_ONBOARDING.md) and [Artifact Transfer Guide](ARTIFACT_TRANSFER_GUIDE.md).

## Current model decision

Accuracy optimization is paused. Do not launch further training, feature selection, threshold tuning or alarm-policy tuning simply to improve a metric.

The robust 170-feature Logistic Regression is the provisional candidate for application integration because it is the latest developed pipeline. The legacy 221-feature model remains a comparison reference. **The baseline is not yet formally frozen**: first verify the actual saved model artifact, model scaler, feature order, channel order, preprocessing configuration, threshold provenance and inference path. Do not retrain automatically if an artifact is missing; inspect existing checkpoints and decide safely.

`patient_01` remains sealed. Historical first-fold test metrics are preserved as descriptive results only; never use patient_01 for future tuning, calibration, feature selection or development decisions.

## Exact next milestone: raw processed-EEG inference parity

Use existing processed artifacts only; no model training and no EDF download.

1. Obtain/verify the approved local artifact bundle using [Artifact Transfer Guide](ARTIFACT_TRANSFER_GUIDE.md).
2. Confirm exact channel order, sample rate, filtering, normalization, windowing and robust feature ordering from source.
3. Select an approved non-sealed development recording and run the complete processed-NPZ → scaler → feature extraction → model path.
4. Compare generated features and probabilities with the matching reference cache/scores; record tolerances and hashes.
5. Do not use patient_01 for tuning, threshold selection, calibration or development decisions.
6. Only after raw-recording parity passes, write `docs/MODEL_BASELINE_FREEZE.md` with artifact hashes, package versions, input/output contract and limitations.
7. Then implement FastAPI health/model-info/inference endpoints.

## Following milestones

- **Backend:** modular FastAPI service, health/model-info/inference endpoints, request/response schemas, input validation, exact shared preprocessing and event aggregation.
- **Frontend:** React + Vite + TypeScript dashboard. Inspect existing `feature/react-dashboard-foundation` branch / PR #1 before reuse. It currently contains mock data only and is not connected to a model.
- **Integration:** end-to-end tests, invalid-input handling, model metadata, event timeline and research disclaimer.
- **Final delivery:** reproducible setup, documentation, demo, deployment/security review and academic report.

Do not claim that the full 23-fold outer LOPO evaluation, deep learning, explainability, FastAPI inference or React-to-model integration is complete.

## Research safeguards

- Never randomly split overlapping EEG windows across train/test.
- Never fit transforms, selectors, thresholds or hyperparameters on outer-test patients.
- Keep patient-wise evaluation and report window/event metrics separately.
- Use non-seizure exposure for FA/hour; merge overlapping seizure annotations before subtracting.
- Preserve raw data provenance; do not commit raw EDFs, runtime caches, virtual environments or large artifacts.
- Do not present this academic prototype as a diagnostic tool.

## Read next

1. [Teammate Onboarding](TEAMMATE_ONBOARDING.md)
2. [Artifact Transfer Guide](ARTIFACT_TRANSFER_GUIDE.md)
3. [Complete Project Handoff](PROJECT_HANDOFF.md)
2. [Experiment History and Results](EXPERIMENT_HISTORY_AND_RESULTS.md)
3. [Project Master Plan](PROJECT_MASTER_PLAN.md)
4. [AI / Teammate Handoff](AI_HANDOFF.md)
5. [Fold-aware Feature Stability Audit](FOLD_AWARE_FEATURE_STABILITY_AUDIT.md)
6. [Event-level Evaluation Semantics](../EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/docs/EVENT_LEVEL_EVALUATION.md)
