# Cross-Subject Epileptic Seizure Detection Using EEG Signals

Academic EEG seizure-detection project using the CHB-MIT Scalp EEG Database. The current research implementation includes cohort auditing, signal preprocessing, seizure annotation parsing, window segmentation, feature extraction, classical baseline experiments, patient-wise OOF evaluation and feature-stability diagnostics.

> **Disclaimer:** This is an academic research prototype, not a medical device. It must not be used for diagnosis or treatment decisions.

## Repositories

- Active working repository: https://github.com/shivkumar15/EEG
- Original source repository: https://github.com/student-anshu/EEG-seizure-detection

## Start here — teammate onboarding and project handoff

New teammates should begin with [Teammate Onboarding](docs/TEAMMATE_ONBOARDING.md) and [Artifact Transfer Guide](docs/ARTIFACT_TRANSFER_GUIDE.md). A new developer or ChatGPT session should then read these files in order:

1. [Complete Project Handoff](docs/PROJECT_HANDOFF.md) — authoritative context, exact stopping point and next actions.
2. [Current Project Status](docs/CURRENT_STATUS.md) — current milestone and verified state.
3. [Experiment History and Results](docs/EXPERIMENT_HISTORY_AND_RESULTS.md) — completed experiments, metrics, artifacts and limitations.
4. [Project Master Plan](docs/PROJECT_MASTER_PLAN.md) — system architecture and completion roadmap.
5. [AI / Teammate Handoff](docs/AI_HANDOFF.md) — operating rules for future contributors.
6. [Development Log](docs/DEVELOPMENT_LOG.md) — chronological implementation history.

## Current direction

Local verification update (6 November 2026): the robust model loads, the fold scaler checkpoint restores, and cached-feature-to-model parity was reported passing locally for 216,005 validation rows (maximum absolute probability difference approximately 2.33e-15). Raw processed EEG-to-model inference parity remains pending; these local checks are not a formal baseline freeze.

Model accuracy optimization is paused. The priority is to complete a reproducible end-to-end application while preserving all actual model results and limitations. The 170-feature robust Logistic Regression is a provisional integration candidate, not a proven winner; the legacy 221-feature model remains a reference. The next task is to complete raw processed-recording inference parity and document the artifact transfer before freezing a versioned baseline and implementing FastAPI.

Do not retrain automatically, do not use sealed outer-test patient_01 for tuning, and do not ask for EDF redownload/reprocessing as a routine step. See the handoff documents for details.

## Project layout

```text
EEG/
├── .venv311/                         # Python 3.11 environment (local)
├── EEG_Seizure_Detection_Phase1/
│   ├── dataset/                       # Original EDF data was local; do not assume present
│   └── EEG_Seizure_Detection/         # Python project code
├── docs/                              # Research and handoff documentation
└── .gitignore
```

## Environment (PowerShell)

From repository root:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
cd .\EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The user's current environment already exists at repository root as `.venv311`; do not recreate it unless missing. Original EDFs are not required for the immediate model-artifact audit if existing processed artifacts suffice.

## Research protocol essentials

- Primary cohort: 683 recordings, 185 seizure events, 23 independent patients.
- CHB01 and CHB21 grouped as one individual; three incompatible-montage CHB12 recordings excluded.
- 17 channels, 256 Hz, 10-second windows, 5-second stride.
- Never randomly split overlapping windows across train/test.
- Never fit transforms, feature selection, thresholds or hyperparameters on outer-test patients.
- Report window and event metrics separately.
- False alarms/hour uses non-seizure exposure, not raw recording duration.
- Patient_01 remains sealed for future development/tuning.
