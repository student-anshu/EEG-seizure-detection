# EEG Seizure Detection

Academic research workspace for EEG-window seizure detection.

> This is an academic prototype, not a medical device. Do not use it for diagnosis or treatment decisions.

## Repositories
- Active working copy: https://github.com/shivkumar15/EEG
- Original source repository: https://github.com/student-anshu/EEG-seizure-detection

## Current status
**Phase 0 — repository audit and correctness.** Validate seizure annotations and dataset paths before beginning model training.

## Layout
```text
EEG_Seizure_Detection_Phase1/
├── EEG_Seizure_Detection/
│   ├── main.py
│   ├── config.py
│   ├── preprocessing/
│   ├── feature_selection/
│   └── requirements.txt
└── dataset/
    └── chbmit/
```

## Read first
1. [Project Master Plan](docs/PROJECT_MASTER_PLAN.md)
2. [AI / Teammate Handoff](docs/AI_HANDOFF.md)
3. [Development Log](docs/DEVELOPMENT_LOG.md)
4. [Upstream Change Log](docs/CHANGELOG_FOR_UPSTREAM.md)

## Research direction
Binary EEG-window seizure detection; SVM/Random Forest and 1D CNN baselines; proposed CNN + BiLSTM + temporal attention; patient-wise/LOSO evaluation; explainability after validation; FastAPI backend; React + Vite + TypeScript + Tailwind frontend.

Do not randomly split overlapping windows across train/test. Fit transformations, feature/channel selection, thresholds and hyperparameters using training/validation subjects only. Previously reported dataset counts remain unverified until reproduced.

Work in this repository first. The original repository is not automatically modified; every change must be recorded with exact upstream instructions.
