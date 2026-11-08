# AI / Teammate Handoff

**New teammate entry point:** [Teammate Onboarding](TEAMMATE_ONBOARDING.md), then [Artifact Transfer Guide](ARTIFACT_TRANSFER_GUIDE.md).  
**Read first for project context:** [Complete Project Handoff](PROJECT_HANDOFF.md).  
Then read [Current Project Status](CURRENT_STATUS.md), [Experiment History and Results](EXPERIMENT_HISTORY_AND_RESULTS.md), and [Project Master Plan](PROJECT_MASTER_PLAN.md).

## Repository and environment

- Active working repository: https://github.com/shivkumar15/EEG
- Current working branch: `feature/cross-patient-feature-stability-audit`
- Original source/team repository: https://github.com/student-anshu/EEG-seizure-detection (do not modify/synchronize automatically).
- User Windows repository root: `D:\2026-THE ENDGAME\byteBazzar\EEG`
- Python environment: repository-root `.venv311`, Python 3.11.
- Nested code directory: `EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection`.

## Project goal and priority

The CTO-approved goal is to complete a working, reproducible EEG seizure-detection application. Do not keep the project blocked by endless accuracy experiments. Preserve and report all actual results and limitations; future accuracy improvement remains a documented research task. Do not launch new training or threshold/feature tuning unless a genuine technical blocker makes it necessary.

This is seizure detection, not future seizure prediction. Academic prototype only; not for diagnosis or treatment decisions.

## Current stopping point

Research baseline development is paused after:
- legacy 221-feature LR first-fold baseline;
- robust 170-feature LR first-fold baseline;
- first-fold event evaluation for both;
- 22-patient inner OOF LR and event evaluation;
- alarm policy exploration;
- original and fold-aware feature stability audits;
- patient_13/14/15 contribution diagnostics.

Exact metrics and caveats are in `EXPERIMENT_HISTORY_AND_RESULTS.md`.

The 170-feature robust Logistic Regression is the provisional integration candidate, not a proven winner. The 221-feature model remains the reference. Do not mark the baseline frozen until the saved model, scaler(s), feature order and complete preprocessing/inference contract have been verified.

## Local verification update — 6 November 2026

The robust model deserialized, the fold scaler checkpoint restored/finalized, and the validation cache was checked locally. Cached-feature-to-model parity passed for 216,005 validation rows (maximum absolute probability difference about 2.33e-15). This is not raw-EEG inference parity and has not been independently reproduced by a fresh teammate clone. Local-only artifact transfer must be verified using `ARTIFACT_TRANSFER_GUIDE.md`.

## Immediate task

**Complete raw processed-recording inference parity — no training.**

Obtain the approved artifact bundle and inspect local `results/baselines`, `cache` and model/checkpoint directories. Verify:
1. bundle hashes and checkpoint manifests;
2. exact feature and channel order;
3. sampling frequency, filtering, normalization, window length/stride and feature extraction;
4. inference function uses exactly the training transformations;
5. threshold and event policy provenance;
6. full processed-NPZ → scaler → feature extraction → model parity on an approved non-sealed development recording.

Only after raw-recording parity passes, write `docs/MODEL_BASELINE_FREEZE.md` with artifact paths, hashes, package versions, input/output contract, reproducibility evidence and limitations. If an artifact is missing, inspect completed checkpoints and determine a safe export path. Do not automatically retrain.

Do not use patient_01 for tuning, calibration, feature selection or development. Existing historical outer-test results may be retained as descriptive metrics only.

## After artifact verification

1. Freeze a versioned baseline package.
2. Implement modular FastAPI health, model-info and inference endpoints.
3. Ensure API preprocessing exactly matches the verified training pipeline.
4. Define validated request/response schemas and event-level aggregation.
5. Inspect/reuse the React + Vite + TypeScript dashboard foundation (separate branch `feature/react-dashboard-foundation`, PR #1); it currently uses illustrative mock data and is not connected to real inference.
6. Integrate frontend with actual backend output and test end to end.
7. Finish reproducibility, deployment, security and academic documentation.

## Scientific and engineering rules

- Never randomly split overlapping windows across train/test.
- Never fit scalers, feature selectors, thresholds or hyperparameters on outer-test patients.
- Do not confuse window metrics with seizure-event metrics.
- False alarms/hour = false-alarm episodes / non-seizure exposure hours; non-seizure exposure subtracts the union of clipped seizure intervals from recording duration.
- OOF threshold metrics are not independent test metrics; fold score scales may differ.
- Do not call the robust model better based only on first-fold results.
- Do not request EDF redownload/reprocessing; existing processed NPZs/caches remain locally. Verify file presence before use.
- Do not commit raw EDFs, runtime caches, virtual environments or large model files.
- Never claim commands/tests were run unless execution output was observed.
- Distinguish planned, implemented, locally tested, scientifically evaluated and deployed.

## PowerShell startup

```powershell
cd "D:\2026-THE ENDGAME\byteBazzar\EEG"
.\.venv311\Scripts\Activate.ps1
python --version
cd .\EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection
python -m pytest -q
```

Run focused tests first when making a targeted change. Explain actions in beginner-friendly Hinglish and provide exact commands. Work one milestone at a time and update `CURRENT_STATUS.md`, `DEVELOPMENT_LOG.md` and relevant technical docs after each milestone.
