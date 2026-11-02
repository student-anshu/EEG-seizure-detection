# AI / Teammate Handoff

Read this file and `PROJECT_MASTER_PLAN.md` before changing code.

## Repository roles
- Active working copy: https://github.com/shivkumar15/EEG
- Original source/upstream: https://github.com/student-anshu/EEG-seizure-detection
- Develop here first. Do not edit upstream automatically. Give the user exact changes to mirror upstream.

## Goal and stage
Binary classification of EEG windows for seizure activity (seizure detection, not future prediction). Academic prototype only; not for clinical use.

**Current stage: Phase 0 — audit and correctness.** Do not train models until annotation labels, file paths, data inventory and leakage-safe split strategy are verified.

## Existing code layout
`EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/` contains `main.py`, `config.py`, `preprocessing/`, `feature_selection/`, `check_processed_data.py` and `requirements.txt`.

Dataset is a sibling directory: `EEG_Seizure_Detection_Phase1/dataset/chbmit/`.

## High-priority risks to verify
1. `preprocessing/annotations.py` currently searches numbered annotation keys (`Seizure 1 Start Time`, etc.). CHB-MIT summaries may use unnumbered `Seizure Start Time` and `Seizure End Time`. Add robust parsing and explicit tests/failure behavior.
2. `config.py` currently derives the dataset path below the code folder, while the repository tree places dataset beside the code folder. Confirm and correct.
3. Normalize import/package conventions and document one canonical invocation.
4. Verify whether `feature_selection/channel_selection.py` is connected to the active pipeline.
5. Never randomly split overlapping windows. Fit all transforms and selection inside training folds only.
6. Raw EDF binaries are currently present. Do not add more large data until storage and provenance policy is decided.

## Unverified prior report
A prior report stated 8 CHB01 EDFs, 5,497 windows (5,397 non-seizure and 100 seizure), 17 channels, 256 Hz, 10-second windows and 5-second stride. Do not describe this as reproduced until rerun and audited.

## Required workflow for every code change
1. Inspect source and callers.
2. Explain the issue and expected behavior.
3. Make the smallest safe change.
4. Add/update tests.
5. Run tests or state why they could not run.
6. Update `DEVELOPMENT_LOG.md`.
7. Update `CHANGELOG_FOR_UPSTREAM.md` with exact paths and copy-ready patch/instructions.
8. Tell the user what changed and what must be mirrored to the original repository.

## Target architecture
Python signal processing/ML; SVM/RF and 1D CNN baselines; proposed CNN + BiLSTM + temporal attention; patient-wise/LOSO evaluation; SHAP after validation; FastAPI; React + Vite + TypeScript + Tailwind. See master plan.

Explain in beginner-friendly steps. Work one phase at a time. Never claim tests, metrics, training or an upstream commit succeeded unless verified.
