# EEG Seizure Detection — Complete Project Handoff

**Last reconciled:** 6 November 2026  
**Working repository:** https://github.com/shivkumar15/EEG  
**Original repository:** https://github.com/student-anshu/EEG-seizure-detection  
**Active branch:** `feature/cross-patient-feature-stability-audit`  
**User's Windows repository root:** `D:\2026-THE ENDGAME\byteBazzar\EEG`  
**Nested Python project:** `EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection`  
**Python environment:** repository-root `.venv311` (Python 3.11)

> **Read this file first in every new ChatGPT conversation or developer handoff.** Then read `CURRENT_STATUS.md`, `EXPERIMENT_HISTORY_AND_RESULTS.md`, and `PROJECT_MASTER_PLAN.md`. This file is the operational entry point; experiment-specific numbers and methodology are in the experiment history.

## 1. Project goal

Complete an end-to-end academic EEG seizure-detection application using the existing CHB-MIT dataset and the model work already performed. The complete system should support the research workflow from EEG input through preprocessing/inference to clearly presented model estimates and event summaries.

The goal has changed from continuing to optimize model accuracy to **finishing a complete, reproducible, honestly evaluated project**. Accuracy is not ignored: all existing results and limitations remain documented, and model improvement is a future research task. Do not make further accuracy experiments a blocker for backend/frontend completion unless a genuine technical issue prevents inference.

This is seizure **detection**, not future seizure prediction. This is not a medical device and must never be presented as diagnosis or treatment guidance.

## 2. CTO-approved execution policy

1. Freeze the existing research baseline after verifying which trained artifact can actually be loaded for inference.
2. Do not launch more model training or accuracy tuning merely to chase a higher score.
3. Preserve the legacy 221-feature baseline, robust 170-feature experiment, OOF scores and all audit results as research records.
4. Do not claim the 170-feature model is scientifically superior; its first-fold event sensitivity improved but false alarms increased, and patient-level OOF ranking is inconsistent.
5. Build the backend around the exact preprocessing and feature contract of the selected saved model. Never create a look-alike preprocessing implementation.
6. Build the React frontend against a documented API contract and use mock data only until real inference is connected and verified.
7. Keep all metrics and limitations visible in project documentation. Never hide poor results.
8. Preserve the sealed outer-test patient_01. No tuning, calibration, feature selection or model selection based on its labels.
9. Never ask the user to download/reprocess EDFs as a routine step. The original EDF files were deleted locally after earlier guidance. Existing processed NPZs, metadata, caches and audit artifacts remain available locally. The current integration work should first use existing saved artifacts and verify what is actually present.
10. Never claim a file, test, command or deployment was executed unless its result was observed.

## 3. Dataset and research foundation

- CHB-MIT Scalp EEG, PhysioNet v1.0.0, DOI `10.13026/C2K01R`.
- 683 primary recordings, 185 seizure events, 23 independent patients.
- CHB01 and CHB21 grouped as patient_01.
- Excluded incompatible-montage files: CHB12 recordings 27, 28, 29.
- 17-channel montage; 256 Hz.
- 10-second windows; 5-second stride.
- 704,836 windows: 702,280 normal and 2,556 seizure.
- Full audit: `docs/full_annotation_audit.csv`; duration inventory: `docs/signal_inventory.csv`.
- Exact duration inventory matches all 683 primary recordings. False alarms/hour uses non-seizure exposure, not raw recording duration.
- Original EDF data is not required for the current documentation and inference integration tasks. Do not delete processed data or caches.

## 4. Work completed

### Data and evaluation foundation
- Dataset annotation and cohort audits.
- Full processed cohort integrity audit.
- Patient-wise LOPO split construction and leakage checks.
- Fit-only normalization and streaming feature iteration.
- 221-feature legacy baseline.
- 170-feature robust baseline.
- First-fold event-level evaluators with exact recording-duration inventory.
- 22-fold patient-grouped inner OOF Logistic Regression.
- OOF event-level evaluation.
- Original and fold-aware cross-patient feature stability audits.
- Patient 13/14/15 model-contribution diagnostics.
- Unit tests for feature extraction, data alignment, OOF caches, event metrics, feature audits and numerical reconstruction.

### Research baseline and its limitations
See `docs/EXPERIMENT_HISTORY_AND_RESULTS.md` for complete metric tables and exact counts.

Key facts:
- Legacy 221-feature first-fold outer test patient_01: window sensitivity 2.04%; event sensitivity 1/11 (9.09%); 0.546 FA/hour.
- Robust 170-feature first-fold outer test patient_01: window sensitivity 6.80%; event sensitivity 4/11 (36.36%); 2.281 FA/hour.
- Inner 22-patient OOF at provisional threshold 0.981883: event sensitivity 43.10% (75/174), 3.501 FA/hour.
- At OOF threshold 0.5: 93.10% event sensitivity but 32.254 FA/hour.
- Macro OOF AUROC 0.8148, macro AUPRC 0.1419; patient variability is substantial, including very weak ranking for patients 13, 14 and 15.
- Research target used for exploration: ≥90% event sensitivity at ≤1 false alarm/hour. It is not a clinical target and current results do not meet it.
- OOF threshold was selected on OOF labels and is not an independent final test. Fold scores may not share a calibrated scale.

### Recent stability audit
- Fold-aware audit completed for 22 patients / 7,480 patient-feature-class rows.
- Median absolute peer-IQR shift 0.5395; p90 1.5141.
- Diagnostic only; no model training, feature selection, threshold tuning or OOF score modification.
- The detailed CSV remains to be inspected if a feature/channel-level review is ever needed. This review is not a prerequisite for application completion.

## 5. Current exact stopping point

**Model research:** paused after the fold-aware feature-stability audit. No further accuracy optimization is currently authorized/required for project completion.

**Baseline decision:** the 170-feature robust Logistic Regression is the provisional application candidate because it is the more recent developed pipeline, but it is not a proven winner. The 221-feature legacy model remains the reference comparator. Do not mark the model frozen until the next artifact audit confirms a complete, reusable inference artifact and exact preprocessing configuration.

**Local verification update (6 November 2026):** the robust model was deserialized, the fold scaler checkpoint was restored/finalized, the validation cache was verified, and cached-feature-to-model parity passed for 216,005 validation rows (maximum absolute probability difference about 2.33e-15). This is local verification only; raw processed-EEG-to-model inference parity remains pending and the baseline is not formally frozen.

**Current immediate task:** complete raw processed-recording inference parity and confirm the approved artifact transfer using local files only:
- obtain and hash-verify the approved local artifact bundle (see `docs/ARTIFACT_TRANSFER_GUIDE.md`);
- confirm exact 170-feature order, channel order, input sampling rate, filters, window size/stride, normalization and threshold provenance;
- run the complete processed-NPZ → scaler → robust feature extractor → model path on an approved non-sealed development recording;
- compare generated feature rows and probabilities with the matching reference cache/scores using explicit tolerances;
- record the input recording identity, artifact hashes, software versions, exact commands and outputs;
- do not retrain, and do not use patient_01 for tuning, calibration, feature selection or development decisions.
If a required deployable model artifact does not exist, determine whether an already completed model checkpoint can be exported safely. Do not retrain automatically.

**Only after that audit:** freeze a versioned model package and start the FastAPI backend.

## Teammate onboarding and artifact transfer

New contributors must start with [`TEAMMATE_ONBOARDING.md`](TEAMMATE_ONBOARDING.md), then read [`ARTIFACT_TRANSFER_GUIDE.md`](ARTIFACT_TRANSFER_GUIDE.md). A fresh clone does not guarantee access to processed NPZs, runtime caches or all local artifacts. Transfer is not considered complete until the receiver verifies the SHA-256 manifest and records the accepted files.

## 6. Planned system architecture

Recommended first implementation:

```text
React + Vite + TypeScript dashboard
              |
        HTTP / JSON API
              |
        FastAPI backend
              |
     Inference service layer
       |              |
Preprocessing      Model loader
       |              |
Feature extraction -> Logistic Regression
              |
      Prediction post-processing
              |
      Event / alarm aggregation
              |
  JSON response + metrics metadata
```

Keep the initial backend modular:
- `app/main.py`: FastAPI app and lifespan.
- `app/api/routes/`: health, model info, inference.
- `app/schemas/`: Pydantic request/response contracts.
- `app/services/`: preprocessing, feature extraction, model inference, event aggregation.
- `app/core/`: settings, paths, logging and validation.
- `models/eeg_baseline_v1/`: versioned model artifacts and model card (only after verified export).
- `tests/`: unit and API integration tests.

Do not assume an uploaded EDF can be handled by the API until the required reader, annotation policy, montage mapping and preprocessing contract are explicitly implemented. Define a supported input format and validation behavior. A prediction response must include model version, threshold/config version, window timestamps, scores, alarm/event summaries and limitations. Avoid diagnosis language.

Frontend proposal: React + Vite + TypeScript + Tailwind. Use the existing React dashboard branch/PR as a reference, but inspect it before merging or reusing. It currently contains illustrative mock data and is not connected to real inference. Dashboard should show EEG signal/selected channels where supported, per-window score timeline, detected event intervals, model metadata and an explicit research disclaimer.

## 7. Immediate next steps — do these in order

### Step A — model artifact readiness audit (no training)
1. Inspect local `results/baselines`, `cache`, and any model/checkpoint directories.
2. Identify candidate robust 170-feature model, scaler and feature metadata.
3. Read the exact training runner and serialization/loading code.
4. Verify feature order and preprocessing parity.
5. Perform a smoke test with a non-sealed development recording or saved feature rows.
6. Produce `docs/MODEL_BASELINE_FREEZE.md` with artifact inventory, exact run contract, checksums, limitations and status.
7. Freeze the baseline only if all required components are reproducible.

### Step B — FastAPI backend
1. Agree API request/response schema and supported input.
2. Implement health and model-info endpoints.
3. Implement validated inference service using the frozen pipeline.
4. Add event aggregation and clear response metadata.
5. Add unit/API tests and verify no preprocessing mismatch.
6. Do not claim clinical reliability.

### Step C — React frontend
1. Inspect existing React branch/PR.
2. Build dashboard shell and routes.
3. Integrate health/model-info/inference endpoints.
4. Display signal, window scores and event timeline from actual backend response.
5. Keep mock data visibly labelled until connected.

### Step D — integration, packaging and final documentation
- End-to-end tests with known sample input.
- Error handling for invalid file, missing channels, sampling-rate mismatch, empty data and inference failure.
- Reproducible setup instructions and environment versions.
- Deployment plan, security review and demo script.
- Research limitations and all measured results included in final report.

## 8. Local PowerShell setup

From repository root:

```powershell
cd "D:\2026-THE ENDGAME\byteBazzar\EEG"
.\.venv311\Scripts\Activate.ps1
python --version
cd .\EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection
```

The project environment is Python 3.11. Do not switch to Python 3.14 for the current scientific stack without testing dependency compatibility.

Useful verification command:

```powershell
python -m pytest -q
```

Use focused test modules before the full suite when changing a specific component. Record actual output.

## 9. Important code and artifact map

Project code root:
`EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/`

- `scripts/run_first_baseline_fold.py` — legacy 221-feature first-fold baseline.
- `scripts/run_robust_feature_first_fold.py` — robust 170-feature first-fold baseline (verify exact filename/options in checkout before running).
- `scripts/run_inner_patient_oof.py` — 22-fold inner patient-grouped OOF training.
- `scripts/evaluate_first_fold_events.py` — legacy saved-score event evaluation.
- `scripts/evaluate_robust_first_fold_events.py` — robust saved-score event evaluation.
- `scripts/evaluate_inner_oof_events.py` — inner OOF saved-score event evaluation.
- `scripts/audit_cross_patient_feature_stability.py` — original descriptive stability audit.
- `scripts/audit_fold_aware_feature_stability.py` — fold-aware descriptive stability audit.
- `scripts/audit_patient14_model_contributions.py` — generalized per-patient contribution audit.
- `evaluation/event_metrics.py` — event matching and false-alarm semantics.
- `evaluation/recording_inventory.py` — exact recording duration join.
- `feature_selection/robust_feature_extraction.py` — 170-feature extractor.
- `docs/EVENT_LEVEL_EVALUATION.md` — event metric semantics.
- `docs/CROSS_PATIENT_FEATURE_STABILITY_AUDIT.md` — original audit protocol.
- `docs/FOLD_AWARE_FEATURE_STABILITY_AUDIT.md` — fold-aware audit protocol.

Local (not necessarily Git-tracked) artifacts:
- `cache/oof_patient_scores_v1/report.json`
- `cache/oof_patient_scores_v1/folds/patient_XX.npz`
- `results/baselines/inner_oof_event_metrics_v1.json`
- `results/diagnostics/fold_aware_feature_stability_v1/report.json`
- `results/diagnostics/fold_aware_feature_stability_v1/fold_aware_patient_feature_distributions.csv`
- robust/legacy baseline result and score artifacts listed in `EXPERIMENT_HISTORY_AND_RESULTS.md`.

Never assume a local artifact is present solely because it is named here; verify its existence before use.

## 10. New ChatGPT session starter prompt

Copy/paste this in a new conversation:

> You are the CTO and senior ML/software architect for my EEG Seizure Detection final-year project. First read these repository documents in this order: `docs/PROJECT_HANDOFF.md`, `docs/CURRENT_STATUS.md`, `docs/EXPERIMENT_HISTORY_AND_RESULTS.md`, and `docs/PROJECT_MASTER_PLAN.md`. Treat them as the source of truth and inspect the actual repository code before proposing changes.
>
> The goal is now to finish a complete, reproducible EEG seizure-detection application, not to keep chasing accuracy. Model accuracy is not ignored: preserve all existing experiments, results and limitations in documentation, but defer further optimization unless a real technical blocker prevents integration. Do not retrain automatically.
>
> The immediate task is a local model artifact/inference-readiness audit for the existing robust 170-feature Logistic Regression candidate. Verify the actual checkpoint, scaler, feature order, preprocessing contract and inference path. The legacy 221-feature model remains a reference; robust is not proven superior. Freeze only after a reproducible smoke test. Then implement FastAPI and integrate the React dashboard.
>
> Strict safeguards: patient_01 is sealed and must never be used for tuning, feature selection, calibration or development. Never randomly split overlapping EEG windows. Do not ask me to download/reprocess EDF files; existing processed NPZs and caches remain. Do not invent results or claim local commands/tests have run unless I provide output or you actually execute them. This is an academic detection prototype, not a clinical device. Explain steps in beginner-friendly Hinglish and give exact PowerShell commands when I need to run something.
>
> After reading the docs, summarize the current state briefly and begin Step A: inspect the repository's actual inference/checkpoint code and prepare the artifact-readiness audit. Do not start model training.

## 11. Handoff rules

- Read docs before coding; inspect source and tests before modifying.
- One milestone at a time.
- Make small changes with tests.
- Update `CURRENT_STATUS.md`, `DEVELOPMENT_LOG.md` and relevant technical docs after each milestone.
- Distinguish planned, implemented, locally tested, scientifically evaluated and deployed.
- Preserve all research outputs; do not delete caches or processed files during cleanup.
- Do not push to the original team repository unless explicitly instructed.
