# Teammate Onboarding Guide

**Last reviewed:** 6 November 2026  
**Repository:** https://github.com/shivkumar15/EEG  
**Branch:** `feature/cross-patient-feature-stability-audit`  
**Project type:** Academic EEG seizure-detection prototype (not a medical device)

This guide is the practical entry point for a teammate joining the project. It describes the repository state and the known local-artifact limitations. It does not imply that a fresh clone contains the original processed EEG data, runtime caches, or every model artifact.

## 1. Read in this order

1. [Project Handoff](PROJECT_HANDOFF.md) — project objective, decisions, safeguards and architecture.
2. [Current Status](CURRENT_STATUS.md) — verified work and the immediate engineering milestone.
3. [Experiment History and Results](EXPERIMENT_HISTORY_AND_RESULTS.md) — exact metrics and scientific limitations.
4. [Project Master Plan](PROJECT_MASTER_PLAN.md) — backend/frontend delivery plan.
5. [AI / Teammate Handoff](AI_HANDOFF.md) — rules for contributors and future sessions.
6. [Artifact Transfer Guide](ARTIFACT_TRANSFER_GUIDE.md) — what is and is not in Git, and how to transfer local artifacts.
7. [Development Log](DEVELOPMENT_LOG.md) — dated implementation history.
8. [Fold-aware Feature Stability Audit](FOLD_AWARE_FEATURE_STABILITY_AUDIT.md) and [Event-level Evaluation](../EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/docs/EVENT_LEVEL_EVALUATION.md) — technical research details.

## 2. Clone and checkout the working branch

Run in PowerShell:

```powershell
git clone https://github.com/shivkumar15/EEG.git
Set-Location .\EEG
git fetch origin
git switch feature/cross-patient-feature-stability-audit
git pull --ff-only origin feature/cross-patient-feature-stability-audit
git status -sb
```

Do not work on the original repository `student-anshu/EEG-seizure-detection`. Do not merge or synchronize branches automatically. At the last repository comparison recorded on 6 November 2026, the working branch and `main` had diverged (working branch 220 commits ahead and 1 behind); re-check before any merge/rebase.

## 3. Python environment

Use Python 3.11 for the current scientific stack. From repository root:

```powershell
py -3.11 --version
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python --version
Set-Location .\EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection
```

Before installing dependencies, check which dependency file is present:

```powershell
Get-ChildItem -Name requirements*.txt, pyproject.toml, setup.cfg
```

Install using the file actually present in this checkout. If `requirements.txt` exists:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then inspect available tests and run them only after dependencies are installed:

```powershell
Get-ChildItem .\tests -Filter "test_*.py" | Measure-Object
python -m pytest -q
```

Record the Python version, exact command and complete observed output in the handoff. No fresh-clone test result is claimed by this document.

## 4. What the project has completed

The documented research foundation includes:
- CHB-MIT cohort audit: 683 primary recordings, 185 seizure events and 23 independent patients.
- Processed cohort integrity audit: 704,836 windows (702,280 normal; 2,556 seizure).
- Patient-wise nested evaluation safeguards and fold-specific normalization.
- Legacy 221-feature and robust 170-feature Logistic Regression research baselines.
- First-fold window/event metrics, 22-patient inner OOF evaluation and alarm-policy exploration.
- Cross-patient and fold-aware feature-stability audits, plus patient 13/14/15 diagnostics.

These are research milestones, not proof of clinical performance. Exact numbers and caveats are in `EXPERIMENT_HISTORY_AND_RESULTS.md`.

## 5. Current engineering status — read carefully

| Area | Status |
|---|---|
| Research experiments and diagnostics | Completed and documented |
| Serialized robust model loading | Verified locally in the primary developer environment |
| Scaler checkpoint restore/finalize | Verified locally in the primary developer environment |
| Validation-cache integrity and labels | Verified locally |
| Cached-feature → model → archived-score parity | Passed locally for 216,005 validation windows; max absolute probability difference about 2.33e-15 |
| Raw processed EEG → preprocessing → scaler → features → model parity | **Pending** |
| Formal model baseline freeze/package | Pending |
| FastAPI backend and inference endpoints | Pending |
| React dashboard real API integration | Pending; existing dashboard foundation is mock-data-only |
| End-to-end tests and deployment | Pending |

The cached-feature parity result verifies the model on an already-built feature matrix. It does **not** establish that a fresh recording is transformed into exactly that matrix. Do not describe the complete raw-EEG inference pipeline as verified.

The local verification results above have not been independently reproduced by a fresh teammate clone. Ask the owner for the exact verification output and artifact inventory before relying on them.

## 6. Scientific and data safeguards

- This is seizure **detection**, not future seizure prediction.
- The project is an academic prototype, not a diagnostic or treatment device.
- Never randomly split overlapping EEG windows across train/test.
- Never fit scalers, selectors, thresholds, hyperparameters or alarm policies on outer-test patients.
- `patient_01` is sealed for future tuning, calibration, feature selection and development decisions. Historical first-fold test metrics may remain in reports as descriptive results only.
- Report window-level and event-level metrics separately.
- False alarms/hour must use non-seizure exposure; subtract the union of clipped seizure intervals, not raw recording duration.
- OOF threshold results selected on OOF labels are descriptive/optimistic, not independent final-test estimates.
- Do not claim the 170-feature robust model is superior to the legacy 221-feature model.

## 7. Source-code map

Python project root: `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/`

- `config.py` — channels, sampling/filter/window configuration.
- `preprocessing/signal_processing.py` — filtering and notch processing.
- `preprocessing/segmentation.py` — windowing and label overlap.
- `preprocessing/normalization.py` — fold-fit channel standardizer and checkpoint state.
- `data_management/fold_data.py` — patient-wise fold preparation and streaming transforms/features.
- `feature_selection/robust_feature_extraction.py` — robust 170-feature extraction.
- `modeling/` — model fitting and experiment checkpoint support.
- `evaluation/event_metrics.py` — event-level metrics and false-alarm semantics.
- `scripts/run_robust_feature_first_fold.py` — robust first-fold experiment runner.
- `scripts/run_inner_patient_oof.py` — patient-grouped inner OOF experiment.
- `scripts/audit_fold_aware_feature_stability.py` — descriptive fold-aware audit.
- `tests/` — unit and regression tests.

Inspect the current source before relying on this map; filenames and implementation may evolve.

## 8. First assigned engineering task

Do not retrain. Do not tune against patient_01.

1. Read `ARTIFACT_TRANSFER_GUIDE.md` and obtain the approved local artifact bundle from the project owner.
2. Verify artifact hashes and manifests before loading anything.
3. Inspect the robust feature extractor, channel order, filters, sampling rate, windowing and normalization contract.
4. Complete raw processed-recording inference parity on a permitted development recording, comparing generated features/scores with the existing validation cache/reference.
5. Record the exact input recording identity, artifact hashes, tolerances, command and output. Do not expose patient_01 to development/tuning.
6. Only after parity passes, prepare `MODEL_BASELINE_FREEZE.md`.
7. Then implement FastAPI health/model-info/inference endpoints against the frozen contract.
8. Integrate the React dashboard with real API responses; mock data must remain visibly labelled until then.

## 9. Git hygiene

- Use the active working branch and inspect `git status -sb` before and after work.
- Never use `git add .` blindly.
- Do not commit raw EDFs, processed data, caches, virtual environments, secrets or unapproved large artifacts.
- Keep research outputs and code changes traceable; update `CURRENT_STATUS.md` and `DEVELOPMENT_LOG.md` after meaningful milestones.
- Do not overwrite teammates' work or merge the diverged `main` branch without review.
- Distinguish implemented, locally verified, committed, teammate-reproduced and deployed work.

## 10. Handoff completion checklist

- [ ] Fresh clone and branch checkout completed.
- [ ] Python 3.11 environment and dependencies installed.
- [ ] Tests run by teammate; exact output recorded.
- [ ] Required local artifacts received and hashes verified.
- [ ] Raw processed EEG inference parity passed.
- [ ] Versioned model package and model card frozen.
- [ ] FastAPI endpoints and tests completed.
- [ ] React dashboard connected to actual backend output.
- [ ] End-to-end tests, limitations, setup and deployment guide completed.
