# Development Log

Record verified actions only. Planned work is not marked complete.

## 2026-11-02 — Working repository established
- Active working repository: https://github.com/shivkumar15/EEG
- Original source repository: https://github.com/student-anshu/EEG-seizure-detection
- Connected GitHub identity verified as `shivkumar15`.
- Write permission to `shivkumar15/EEG` verified.
- Default branch: `main`.
- Repository contains `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection` and sibling `EEG_Seizure_Detection_Phase1/dataset/chbmit` paths with CHB01–CHB03 folders.
- Development and verification are being performed in the working repository first. The original team repository has not yet been synchronized or tested.

## Phase 0 — audit and correctness

### Task 1: CHB-MIT annotation parser — VERIFIED
- Updated `preprocessing/annotations.py` to support numbered and unnumbered seizure time fields, validate declared seizure counts, reject invalid intervals, return an empty list only for explicitly zero-seizure recordings, and report missing EDF/summary mappings.
- Added `tests/test_annotations.py` (6 tests) and `tests/__init__.py`.
- User ran the unit tests: 6 passed.
- Four actual summary-file checks passed:
  - `chb01_03.edf` → `[(2996.0, 3036.0)]`
  - `chb01_05.edf` → `[]`
  - `chb03_01.edf` → `[(362.0, 414.0)]`
  - `chb03_05.edf` → `[]`
- Temporary `verify_annotations.py` was removed and is not part of the project.

### Task 2: Dataset path correction — VERIFIED
- `config.py` resolves `DATASET_DIR` as `PROJECT_ROOT.parent / "dataset" / "chbmit"`.
- Added `tests/test_config_paths.py` (2 tests).
- User verified that `python main.py --inspect` finds 10 EDFs: CHB01 (8), CHB02 (1), CHB03 (1).

### Task 3: Import/execution convention — VERIFIED
- Updated `check_processed_data.py` to use `from config import PROCESSED_DIR`, consistent with running project scripts from the `EEG_Seizure_Detection` directory.
- User ran `python -m unittest discover -s tests -v`: 8 tests passed at that stage.
- `python check_processed_data.py` completed successfully before processed outputs existed.

### Task 4: Signal-processing hardening — VERIFIED
- Hardened `preprocessing/signal_processing.py` with validation for array shape, finite values, sampling frequency and filter cutoffs.
- Retained the fourth-order Butterworth bandpass (default 0.5–40 Hz), with Nyquist-aware upper cutoff.
- Used second-order-section forward-backward filtering for the optional notch stage; notch is skipped when disabled or at/above Nyquist.
- Added clear errors for recordings too short for zero-phase filtering.
- Retained median/IQR normalization and documented that it uses statistics from the complete supplied recording (offline, non-causal).
- Added `tests/test_signal_processing.py` (10 tests).
- User ran the complete suite at that stage: 18 tests passed.

### Task 5: Segmentation, channel selection and channel matching — VERIFIED
- Hardened `preprocessing/segmentation.py` with input and interval validation; windows use half-open intervals `[start, end)` and positive labels for any overlap with a seizure annotation.
- Hardened `feature_selection/channel_selection.py` and documented that channel ranking must be fitted on training-fold data only.
- Updated `main.py` channel matching to be case/whitespace insensitive, added a constrained EEG-like fallback, and expanded per-window metadata with end time, sampling frequency and selected channels.
- Added segmentation, channel-selection and channel-matching tests.
- User ran `python -m unittest discover -s tests -v`: **35 tests passed, 0 failures**.
- User ran `python main.py --inspect`: 10 EDF files found.
- All 10 inspected recordings had 256 Hz sampling and the same 17 selected channels. No resampling was needed for this inspected subset.
- The selected montage did not include `T8-P8` because that channel was not present under the expected name in these files; the actual selected channel list was consistent across the 10 files.

### Task 6: Git ignore scope and raw EDF storage — VERIFIED
- Added a repository-root `.gitignore` to exclude virtual environments, Python caches, generated logs, IDE files, OS metadata, and the local CHB-MIT dataset directory.
- Commit `e5c27c0` added the root ignore rules; commit `abfd931` removed the already-tracked dataset files from the Git index and pushed the removal to `shivkumar15/EEG`.
- The removal commit staged and removed 10 EDF recordings and 3 matching subject summary files from the current Git tree. The files remain on the user's computer; `Test-Path` returned `True` for `chb01_03.edf`.
- `git check-ignore` confirmed that the raw EDF path is ignored. The user also verified a clean working tree after the push.
- Do not copy raw EDF files or subject summary files into the original team repository. Team members should obtain the dataset independently from the official PhysioNet source and retain the source/version/license notice.
- Dataset provenance verified from the official PhysioNet CHB-MIT v1.0.0 page: DOI `10.13026/C2K01R`; license listed as Open Data Commons Attribution License (ODC-By) v1.0. The ODC-By license governs database rights and expressly does not automatically clear separate rights in individual contents. Do not treat the database license as a blanket legal conclusion about every individual recording.
- For public redistribution of the database or a derivative database, comply with the ODC-By notice/license requirements. For public use of produced works, include the required source/license notice. Project policy is to keep raw recordings out of Git and cite PhysioNet in project documentation and public outputs.
- The ten EDFs and three summary files are no longer tracked in the current Git tree, but remain in earlier Git history. No history rewrite was performed.

### Task 7: Clean preprocessing verification — VERIFIED FOR THE CURRENT 10-EDF SUBSET
- User inspected all 10 EDF headers and annotation summaries. Each had 256 Hz sampling, 23 raw channels, and 17 selected channels. Nine recordings were 3,600 seconds and `chb01_26.edf` was 2,325 seconds.
- All 10 EDFs loaded and their annotation intervals were parsed. Eight recordings contained seizure intervals; two declared no seizures.
- Controlled single-recording run:
  - `chb01_03.edf` produced `X=(719, 17, 2560)`, 709 normal and 10 seizure windows.
  - Its NPZ and CSV were checked: finite values, 17 channels, 256 Hz, 719 rows, no missing CSV values, and labels/start times matched.
- Full run: `python main.py --all`. User output showed all 10 recordings saved and no `SKIPPED` lines.
- Final generated-file validation found 10 NPZ files and 10 metadata CSV files. Every file pair passed shape, finite-value, binary-label, row-count, label-match, timestamp-match and missing-value checks.
- Verified aggregate:
  - **10 recordings**
  - **6,935 windows**
  - **6,823 normal windows**
  - **112 seizure windows**
- Per subject:
  - CHB01: 8 recordings, 5,497 windows (5,397 normal; 100 seizure).
  - CHB02: 1 recording, 719 windows (719 normal; 0 seizure).
  - CHB03: 1 recording, 719 windows (707 normal; 12 seizure).
- The seizure-window proportion in this selected subset is approximately 1.61%.
- Previously reported CHB01 counts were reproduced exactly. These are counts for the current selected subset, not the complete CHB-MIT corpus.

## Current verified limitations and methodology notes
- Only 10 EDF recordings from 3 subjects are currently present in this working subset. This is enough to verify the pipeline, not to claim reliable patient-independent performance.
- Windows overlap by 50%; never randomly split windows across train/test. Patient-wise/grouped splits are required.
- Current `robust_normalize` calculates median/IQR over the entire recording. It is an offline per-recording transform, not causal real-time normalization. Its use in evaluation must be reviewed to prevent inappropriate information leakage and to ensure inference consistency.
- Channel selection and feature ranking must be fitted using training-fold data only.
- No resampling is currently implemented. The 10 inspected recordings all use 256 Hz; re-check sampling rates when expanding the cohort.
- `main.py --all` catches per-file exceptions and prints `SKIPPED`; future cohort runs must record and audit skips explicitly.
- This is an academic seizure-detection prototype, not a medical device or diagnostic system.

## Task 8 — Final Phase 0 validation
- [x] Dataset provenance and storage policy documented using the official PhysioNet CHB-MIT v1.0.0 page and ODC-By v1.0 license information.
- [x] README dataset paths, environment setup, repository links, and documentation links corrected to match the verified layout.
- [x] User reran the full unit test suite: **35 tests passed, 0 failures**.
- [x] User reran `python main.py --inspect`: 10 EDF recordings found (CHB01: 8; CHB02: 1; CHB03: 1).
- [x] Removed 10 EDF files and 3 subject summary files from current Git tracking while preserving local files; verified ignore behavior and clean working tree.
- [ ] Synchronize the verified code folder and documentation to the original team repository, preserving its dataset and excluding virtual environments/generated outputs.
- [ ] Run tests and `python main.py --inspect` in the original repository. Do not claim upstream completion until the user confirms the results.

## Current phase
**Phase 0 — Tasks 1–7 verified; Task 8 final synchronization and original-repository verification pending.**

## Next actions
- Prepare and review the original repository backup and synchronization diff.
- Do not begin model training until the original repository has been synchronized and verified, and the cohort/split design is explicitly defined.
