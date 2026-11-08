# EEG Seizure Detection — Project Master Plan

**Working repository:** https://github.com/shivkumar15/EEG  
**Original source repository:** https://github.com/student-anshu/EEG-seizure-detection  
**Current branch:** `feature/cross-patient-feature-stability-audit`  
**Current milestone:** complete raw processed-EEG inference parity; then freeze the baseline and implement FastAPI.

## 1. Research and product objective

Build a complete academic EEG seizure-detection application using the CHB-MIT Scalp EEG dataset. The model detects seizure activity in EEG windows; it does not predict future seizures. This is a research prototype, not a medical device, and must not be used for diagnosis or treatment decisions.

The current delivery priority is a complete, reproducible system. Model accuracy is not ignored: completed results and limitations must remain visible and future improvements documented. However, endless model optimization is no longer a blocker for completing backend, frontend and integration.

## 2. Verified dataset and cohort

- CHB-MIT Scalp EEG Database, PhysioNet v1.0.0, DOI `10.13026/C2K01R`.
- Full audit: 686 EDF recordings, 198 events, 24 subject folders.
- Primary cohort: 683 recordings, 185 events, 23 independent patients.
- CHB01 and CHB21 grouped as the same individual (`patient_01`).
- Excluded incompatible-montage recordings: CHB12 recordings 27, 28 and 29.
- Sampling rate 256 Hz; preferred montage 17 channels.
- 10-second windows with 5-second stride.
- Full processed cohort: 704,836 windows (702,280 normal, 2,556 seizure), 683/683 valid.
- Duration inventory matches all 683 primary recordings; non-seizure exposure is duration minus the union of clipped seizure annotations.

The user no longer has the original EDF files locally. Existing processed NPZs, metadata, caches and result artifacts remain. Do not ask for a fresh EDF download or full preprocessing run unless a separately justified future task truly requires it.

## 3. Research methodology and safeguards

Preferred research design was 23-fold patient-wise Leave-One-Patient-Out evaluation, with inner patient-grouped fit/validation. No random window-level splitting is permitted because windows overlap and share patient/recording characteristics.

- All transforms, feature selection, class weighting, model parameters, thresholds and alarm rules must be selected without the outer-test patient.
- `patient_01` remains sealed for all future development and tuning. Its historical first-fold test metrics are preserved as a descriptive one-time evaluation.
- Report window-level and event-level metrics separately.
- Metrics include sensitivity, specificity, precision, F1/F2, AUROC, AUPRC, confusion counts, event sensitivity, detection delay, false-alarm episodes and false alarms per non-seizure hour.
- False alarms/hour denominator is exact non-seizure exposure; union overlapping seizure annotations before subtracting.
- OOF pooled thresholds selected on OOF labels are descriptive/optimistic, not independent test performance. Scores from separately trained folds may not share a calibrated scale.

## 4. Completed model experiments

The exact feature definitions, runner paths, artifact paths, thresholds, tables and caveats are in [Experiment History and Results](EXPERIMENT_HISTORY_AND_RESULTS.md).

### Legacy 221-feature Logistic Regression
- First-fold model: 13 features/channel × 17 channels, LR.
- patient_01 outer-test window sensitivity 2.04%; event sensitivity 1/11 (9.09%); 0.546 FA/hour.
- Retain as historical baseline/reference.

### Robust 170-feature Logistic Regression
- 10 robust features/channel × 17 channels, LR.
- patient_01 outer-test window sensitivity 6.80%; event sensitivity 4/11 (36.36%); 2.281 FA/hour.
- It is a provisional application candidate only; not proven superior.

### 22-patient inner OOF
- 22 held-out-patient folds; 608 recordings; 652,077 windows; patient_01 excluded.
- At provisional threshold 0.981883: 75/174 events detected (43.10%), 3.501 FA/hour.
- At 0.5: 162/174 events detected (93.10%), 32.254 FA/hour.
- Macro AUROC 0.8148, macro AUPRC 0.1419, with large patient variation.
- Neither operating point meets the exploratory research target ≥90% event sensitivity and ≤1 FA/hour.

### Feature stability and contribution diagnostics
- Original and fold-aware feature stability audits completed.
- Fold-aware audit: 22 folds, 7,480 patient-feature-class rows; median absolute peer-IQR shift 0.5395, p90 1.5141.
- Patient 13/14/15 model contribution audits completed; directions vary.
- No feature removal or further tuning justified. Diagnostics are descriptive, not causal.

## 5. Current model freeze policy

Local verification update (6 November 2026): the robust model deserialized, the fold-specific scaler checkpoint restored/finalized, and cached-feature-to-model parity passed locally for 216,005 validation rows (maximum absolute probability difference approximately 2.33e-15). These checks do not verify raw processed-EEG preprocessing-to-score parity and have not yet been independently reproduced by a teammate. The formal baseline remains unfrozen. See [Teammate Onboarding](TEAMMATE_ONBOARDING.md) and [Artifact Transfer Guide](ARTIFACT_TRANSFER_GUIDE.md) for setup and local artifact requirements.

Accuracy experiments are paused. The 170-feature robust LR is the provisional model candidate for application integration; the legacy 221-feature LR remains the reference. Do not claim the robust model is better overall.

Before calling the model frozen, verify the actual serialized model, all required scaler(s), feature/channel order, preprocessing configuration, inference function, threshold provenance and reproducible smoke test. If the saved model artifact is incomplete, inspect existing completed checkpoints and export safely without retraining. Document the final artifact inventory in `docs/MODEL_BASELINE_FREEZE.md`.

## 6. System architecture

```text
React + Vite + TypeScript UI
          |
     REST / JSON API
          |
      FastAPI app
          |
   Inference service
     |           |
Preprocessing  Model loader
     |           |
Feature extraction -> LR model
          |
 Event/alarm aggregation
          |
 Versioned JSON response
```

### Backend
- Python 3.11.
- FastAPI + Pydantic.
- Modular routes, schemas, services and configuration.
- Model loading at application startup; avoid reloading per request.
- Validate file type, sample rate, channels, duration and finite signal values.
- Use exactly the verified training preprocessing and feature pipeline.
- Provide health, model metadata and inference endpoints.
- Response includes model version, threshold/config version, window timestamps/scores, detected intervals, event summaries and limitations.
- Do not describe outputs as diagnosis.

### Frontend
- React + Vite + TypeScript + Tailwind.
- Inspect existing `feature/react-dashboard-foundation` branch / PR #1 before reusing. It contains illustrative mock data only and is not connected to a trained model.
- Dashboard: project status, supported EEG input, signal/channel visualization where feasible, window score timeline, event intervals, model metadata, evaluation metrics and clear research disclaimer.
- Mock data must be explicitly labelled until connected to actual backend inference.

## 7. Execution roadmap

### Gate 1 — raw processed-EEG inference parity (next)
- Obtain and hash-verify the approved local artifact bundle; do not assume caches or processed NPZs are in a fresh clone.
- Verify exact feature/channel ordering and preprocessing contract from source.
- Use an approved non-sealed development recording to run processed NPZ → scaler → robust feature extraction → model.
- Compare generated features and probabilities with matching reference cache/scores using explicit tolerances.
- Record artifact hashes, software versions, input identity, commands and outputs.
- Do not tune, calibrate or select features using patient_01.
- Create `MODEL_BASELINE_FREEZE.md` only after end-to-end inference parity passes.

### Gate 2 — FastAPI backend
- Define supported input format and API contract.
- Implement health and model-info routes.
- Implement inference service using the frozen pipeline.
- Implement event aggregation.
- Add unit and API integration tests.
- Validate error cases and preprocessing parity.

### Gate 3 — React frontend
- Inspect existing dashboard foundation.
- Build app shell, routing, upload/input flow and results views.
- Integrate real API responses.
- Display signal, window scores, event timeline and model metadata.

### Gate 4 — end-to-end integration
- Known sample inference test.
- Invalid file, missing channel, wrong sample rate, empty data and non-finite data tests.
- Performance and memory check.
- Reproducible setup, demo and deployment guide.

### Gate 5 — final research/submission package
- Preserve all experimental results and limitations.
- Write methods, cohort, preprocessing, model, evaluation and limitations sections.
- Explain that full 23-fold outer evaluation, deep learning and clinical validation are not complete unless later performed.
- Accuracy improvement becomes a separate future research roadmap, not a completion blocker.

## 8. Out of scope for immediate work

- More Logistic Regression threshold hunting.
- Further feature deletion/reversal based on patient diagnostics.
- More alarm smoothing/grid search.
- CNN, CNN-BiLSTM, attention or model zoo experiments.
- New full 23-fold training run solely to increase metrics.
- Clinical claims or medical-device deployment.

These may be revisited as future research if time and evidence justify them.

## 9. Engineering and handoff rules

1. Read [Project Handoff](PROJECT_HANDOFF.md) and [Current Status](CURRENT_STATUS.md) before code changes.
2. Inspect the actual source, tests and local artifact availability; do not infer from filenames alone.
3. Do not retrain automatically.
4. Do not delete processed data, caches or checkpoints during cleanup.
5. Do not ask the user to download EDFs again as a default step.
6. Make small, tested changes and work one milestone at a time.
7. Record exact commands and observed outputs; never invent results.
8. Update current status, development log and relevant docs after each milestone.
9. Keep active working repository separate from original team repository; do not sync upstream without explicit approval.
10. Use beginner-friendly Hinglish and exact PowerShell commands when guiding the user.
