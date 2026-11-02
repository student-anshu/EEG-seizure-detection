# Upstream Change Log — Mirror to Original Repository

**Working copy:** https://github.com/shivkumar15/EEG  
**Original repository:** https://github.com/student-anshu/EEG-seizure-detection

This ledger tracks changes developed here that the team must also apply to the original repository.

## 2026-11-02 — Documentation baseline (working copy)

### Files added
- `README.md` — repository overview and navigation.
- `docs/PROJECT_MASTER_PLAN.md` — research architecture, roadmap, evaluation protocol and working rules.
- `docs/DEVELOPMENT_LOG.md` — verified repository baseline and progress.
- `docs/AI_HANDOFF.md` — context for teammates and future AI sessions.
- `docs/CHANGELOG_FOR_UPSTREAM.md` — this upstream mirror ledger.

### Code/data changes
None. No source code, dataset or model artifact was changed in this documentation-only update.

### What to mirror upstream
Copy these five documentation paths/files to the original repository. There is no code patch for this update.

## 2026-11-02 — Phase 0, Task 1: annotation parser correction

### Files changed in working copy
- Updated `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/preprocessing/annotations.py`.
- Added `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/test_annotations.py`.
- Added `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/__init__.py` (empty file).

### Behavior change
- Supports unnumbered CHB-MIT fields (`Seizure Start Time`, `Seizure End Time`) and numbered variants.
- Verifies annotation count against `Number of Seizures in File`.
- Returns `[]` for an explicitly zero-seizure recording.
- Raises clear errors for missing/mismatched annotations, invalid intervals, missing summary files and EDF files not listed in the summary.

### Verified tests (user's local Windows checkout)
- `python -m unittest discover -s tests -v` — **6 tests passed**.
- `python verify_annotations.py` — **4 actual CHB-MIT recording checks passed**:
  - `chb01_03.edf` → `[(2996.0, 3036.0)]`
  - `chb01_05.edf` → `[]`
  - `chb03_01.edf` → `[(362.0, 414.0)]`
  - `chb03_05.edf` → `[]`

The temporary `verify_annotations.py` was created locally for verification and is not required in the upstream mirror.

### Mirror to original repository
Copy these three files from the working repository to the exact same relative paths in `student-anshu/EEG-seizure-detection`:
1. `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/preprocessing/annotations.py`
2. `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/test_annotations.py`
3. `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/__init__.py`

Then run `python -m unittest discover -s tests -v` from the original repository's `EEG_Seizure_Detection` directory. Do not mark upstream as updated until that is confirmed.

## Pending implementation changes
- Fix dataset path resolution for the sibling `dataset` directory.
- Normalize import/package execution conventions.
- Add tests and validate preprocessing, channel handling and normalization.
- Correct `.gitignore` scope and decide raw EDF storage policy.


## Phase 0, Task 2: dataset path correction

### Files changed in working copy
- Updated `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/config.py`.
- Added `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/test_config_paths.py`.

### Behavior change
- `DATASET_DIR` now resolves to `PROJECT_ROOT.parent / "dataset" / "chbmit"`, matching the actual layout where `dataset` and `EEG_Seizure_Detection` are sibling directories.

### Verification status
Implementation and two path tests are committed in the working repository. Local test execution and `python main.py --inspect` are **pending user verification**.

### Mirror to original repository
Copy both files to the same relative paths in `student-anshu/EEG-seizure-detection`. Then, from the original repository's `EEG_Seizure_Detection` directory, run:
```powershell
python -m unittest discover -s tests -v
python main.py --inspect
```
Do not mark the upstream copy complete until verified.
