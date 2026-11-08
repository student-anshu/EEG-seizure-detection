# Development Log

This log distinguishes confirmed repository state from local work that has not yet been pushed.

## 2026-11-03 — Current project checkpoint

### Phase 0 — dataset audit and methodology
- CHB-MIT dataset source: PhysioNet v1.0.0, DOI `10.13026/C2K01R`; preserve attribution and ODC-By license notice.
- Full archive audit: 686 EDF recordings, 198 seizure events, 24 subject folders; 0 EDF annotation parsing failures.
- All 686 EDF headers inspected; all use 256 Hz. Preferred 17-channel montage is available in 683/686 recordings.
- CHB01 and CHB21 are the same individual and are grouped as `patient_01`; the cohort therefore has 23 independent patients.
- Three CHB12 recordings are excluded from the primary cohort due to incompatible montage: `chb12_27.edf`, `chb12_28.edf`, `chb12_29.edf`. These contain 13 annotated seizure events.
- Primary cohort: 683 recordings, 185 seizure events, 23 independent patients.
- Full audit manifest: `docs/full_annotation_audit.csv`.
- The original 42.62 GB ZIP archive was removed after extraction to recover disk space. The extracted EDF dataset remains locally available. Do not delete it.

### Phase 1 — full-cohort preprocessing: COMPLETE
- Resumable `--prepare-cohort` command was added and hardened in commits:
  - `cfc0d772001825e5cb413013d45fe7a1ee8939f3`
  - `01f6d64159fad31c0f46f18b41aa3c50abeaee40`
- Latest recovery run: 65 recordings processed, 618 valid outputs skipped, 0 failed.
- Full integrity audit: 683/683 recordings checked and valid; 0 invalid.
- Total windows: 704,836: 702,280 normal and 2,556 seizure.
- Verification command: `python .\\verify_full_cohort.py`.
- This validates saved outputs and metadata against implemented integrity checks; it is not proof of model performance.

### Phase 2 — patient-wise evaluation splits: VERIFIED
- 23 outer Leave-One-Patient-Out folds.
- Every eligible recording is in the test set exactly once; no patient leakage between training and test.
- Inner validation: 17 fit patients, 5 validation patients, 1 outer test patient per fold.
- Split inspection scripts: `python .\\scripts\\inspect_lopo_splits.py` and `python .\\scripts\\inspect_nested_validation.py`.
- These scripts verify split construction; they do not train models or generate performance metrics.

### Phase 3 — preprocessing and full-cohort integrity: COMPLETE
- Fold-wise standardization is implemented in `data_management/fold_data.py`.
- `prepare_lopo_fold()` fits the standardizer using inner-fit patients only; the same fitted transform is applied to validation and outer-test data.
- `PreparedFold.iter_split("fit"|"validation"|"test")` streams recordings one at a time.
- Model evaluation must use this fold-wise path and must not fit transforms on validation/test patients.

### Phase 4 — baseline model development: IN PROGRESS
Current milestone: **4.1 — feature extraction and scientific validation**.

#### Feature extraction implementation: PUSHED AND VERIFIED ON GITHUB MAIN
- Source: `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/feature_selection/feature_extraction.py`.
- Focused tests: `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/test_feature_extraction.py`.
- Both files were pushed and verified on main at commit `00670a5121e2c739bbe827f05ea16b8817b99c8c`.
- Implements 13 features per channel: mean, standard deviation, variance, RMS, line length, skewness, excess kurtosis, delta/theta/alpha/beta/gamma band powers, and spectral entropy.
- Frequency bands: delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–40 Hz.
- Input shape: `(n_windows, n_channels, n_samples)`; output shape: `(n_windows, n_channels * 13)`, float32, channel-major ordering.
- Uses Welch PSD and SciPy trapezoidal integration. Constant/near-constant signal skewness and kurtosis are set to zero to avoid NaN output.
- Full regression suite was rerun by the user after synchronization: **95 tests passed, 0 failures** in 0.470 seconds.
- This is a unit/regression result, not yet evidence that frequency-band estimates are scientifically validated.
- Synthetic frequency validation script: `scripts/validate_frequency_features.py` (local; successful run reported by user).
- **Synthetic pure-tone validation: 5/5 passed** — 2 Hz correctly dominant in delta, 6 Hz in theta, 10 Hz in alpha, 20 Hz in beta, and 35 Hz in gamma.
- This confirms expected band dominance for clean in-band pure tones at 256 Hz. It does not validate noisy/mixed EEG, boundary frequencies, or clinical performance.

#### Extended synthetic robustness validation — completed diagnostic run
- Added `scripts/validate_frequency_robustness.py`; user pulled commit `bc3324b` and ran it locally.
- Ten near-boundary tones (0.75, 3.5, 4.5, 7.5, 8.5, 12.5, 13.5, 29.5, 30.5, 39.5 Hz) produced sensible dominant-band results. At several midpoints, adjacent bands had nearly equal power, as expected from finite resolution and spectral leakage.
- Mixed 2 Hz + 10 Hz test passed: delta and alpha components were both represented.
- Noisy 10 Hz test passed: alpha remained dominant after Gaussian noise (standard deviation 0.3; seed 42), with finite powers.
- Welch comparison: 1-second segments provide 1 Hz resolution; 2-second segments provide 0.5 Hz resolution.
- For the diagnostic mixed 2 Hz + 10 Hz signal, delta power was 0.4166667 using the existing 1-second bin-only integration and 0.4895833 with interpolated band edges. At 2-second Welch, both methods gave 0.5 for this test.
- These are synthetic diagnostics only and do not establish performance on real EEG.

#### Candidate spectral improvement — verified by user locally
- Updated `feature_selection/feature_extraction.py` in commit `5487307915967a6b7c5cfd9fed745db97d2670fd`.
- Welch now uses up to 2-second segments (`nperseg=min(signal_length, round(2 * sfreq))`), providing 0.5 Hz bin spacing at 256 Hz when the window is at least 2 seconds long.
- Band powers now integrate from each configured lower and upper edge, interpolating PSD values at both edges before trapezoidal integration.
- Updated `tests/test_feature_extraction.py` in commit `f3d54ed5b5547287a1a4be1c41e5747340b6f1e1` with synthetic expected-power checks: 2 Hz unit-amplitude sine should yield about 0.5 delta power; 10 Hz amplitude-0.5 sine should yield about 0.125 alpha power.
- User ran both synthetic validation scripts after pulling the changes. Pure-tone tests: 5/5 passed; ten boundary-near frequencies produced sensible dominant-band results; mixed 2 Hz + 10 Hz check passed; noisy 10 Hz signal retained alpha dominance with finite band powers.
- User ran the full regression suite after the changes: **97 tests passed, 0 failures** in 0.468 seconds.
- These results verify the tested synthetic cases and software regression suite; they do not establish real-EEG seizure detection performance.

#### Still pending in milestone 4.1
- Review the scientific assumptions and spectral entropy definition.
- Integrate feature extraction into the fold-wise data pipeline and add integration tests.
- Fold-standardized input changes the scale of absolute band-power features; do not describe these as calibrated physical EEG power in microvolt-squared units.
- Integrate feature extraction into the fold-wise data pipeline only after scientific checks pass.

#### Not started
- Feature-based SVM and Random Forest baselines.
- 1D CNN, CNN + BiLSTM, attention and ablation experiments.
- Final window/event-level model evaluation.

### React dashboard
- Separate branch: `feature/react-dashboard-foundation`.
- Commit: `2a08eb9a921db5d84e953510539b6ecdc5e5733d`.
- Open PR #1: https://github.com/shivkumar15/EEG/pull/1
- React 19 + TypeScript + Vite foundation uses clearly labelled illustrative mock data.
- Local frontend build has not been confirmed. PR is not merged; dashboard is not connected to a trained model or inference API.

## Current milestone order
1. Synthetic frequency-band validation.
2. Review/freeze feature definitions.
3. Integrate fold-wise feature generation.
4. Train SVM and Random Forest baselines with nested patient-wise folds.
5. Record fold-wise predictions and metrics before moving to deep learning.

## Important scientific safeguards
- Do not randomly split overlapping windows across train/test.
- Do not fit scalers, feature selectors, thresholds or hyperparameters using outer-test patients.
- Channel ranking must be performed inside the training fold only.
- Report window-level and event-level evaluation separately.
- This is an academic research prototype, not a medical device or diagnostic tool.


### 2026-11-03 — Milestone 4.2: fold-wise feature integration VERIFIED LOCALLY
- Added `RecordingFeatures` and `PreparedFold.iter_features(split)` in `data_management/fold_data.py`.
- The iterator builds on `iter_split()`: each recording is loaded, transformed using the standardizer fitted on inner-fit patients, and then passed to deterministic feature extraction.
- It streams one recording at a time and yields the feature matrix together with the recording identity, labels, window start timestamps and sampling frequency.
- Expected matrix shape is `(n_windows, 221)`: 17 configured channels × 13 features per channel. Matrices are checked for float32 dtype and finite values.
- Integration tests cover fit/validation/test split iteration, 221-column feature shape, finite float32 values, label/timestamp alignment, and retention of recording identity.
- GitHub main commits: implementation `5100a3cbc08d44c3c437f23f49229549abc5ecaf`; tests `372afa09b362c165b9b79c7a394815f6f246175a`.
- User pulled main successfully. Focused command `python -m unittest discover -s tests -p "test_fold_data.py" -v`: **5 tests passed, 0 failures** in 0.521 seconds.
- Full command `python -m unittest discover -s tests -v`: **98 tests passed, 0 failures** in 0.874 seconds.
- These are user-reported local test results. They verify software behavior on the test fixtures, not seizure-detection performance on real EEG.

#### Next milestone
- Review and freeze feature definitions, then implement the baseline training/evaluation workflow using the patient-wise nested folds and recording-at-a-time feature iterator.
- Do not randomly split overlapping windows. Hyperparameter selection must remain inside inner fit/validation patients; outer-test patients are reserved for final fold evaluation.


## 2026-11-04 — Documentation reconciliation

- Reconciled `docs/PROJECT_MASTER_PLAN.md`, `docs/CURRENT_STATUS.md` and `docs/AI_HANDOFF.md` with the current 683-recording, 23-patient research cohort and Phase 4.5.2 checkpoint implementation.
- Clarified the distinction between completed data/split/feature infrastructure and pending real-EEG model evaluation.
- Recorded that model-stage compile, focused tests, full regression tests and controlled runner interruption/restart verification still require confirmation from the user's local environment.
- Corrected the handoff instructions to use the repository-root `.venv311` Python 3.11 environment and run tests from the nested `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection` directory.
- No model training was started and no experiment metrics were added.


---

## 2026-11-06 — Research baseline handoff and system-completion strategy

### CTO decision
- Accuracy optimization is paused; the project priority is now a complete, reproducible, honestly evaluated application.
- Existing model experiments, OOF results and diagnostic audits remain part of the research record. Further accuracy improvement is a future research task, not a blocker for backend/frontend completion.
- The robust 170-feature Logistic Regression is only a provisional integration candidate. The legacy 221-feature Logistic Regression remains the reference; neither is declared a proven winner.
- No additional training or tuning is authorized merely to improve metrics. First verify whether an existing saved model/checkpoint can be used reproducibly for inference.

### Documentation created/reconciled
- Added `docs/EXPERIMENT_HISTORY_AND_RESULTS.md` with the 221-feature and 170-feature first-fold results, event-level metrics, 22-patient OOF results, alarm-policy exploration, feature stability audits, patient contribution diagnostics, artifact paths and scientific caveats.
- Added `docs/PROJECT_HANDOFF.md` as the authoritative new-chat/developer handoff, including exact stopping point, local environment, file map, next actions and a copy/paste new-chat prompt.
- Rewrote `docs/CURRENT_STATUS.md`, `docs/AI_HANDOFF.md` and `docs/PROJECT_MASTER_PLAN.md` to remove stale next-step guidance and align the project with the approved system-completion strategy.
- Updated repository `README.md` so new contributors are directed to the handoff and experiment history first.
- The user asked that a new ChatGPT session reading the repository should understand what was done, why work paused, and where to resume.

### Next milestone
- Perform a local, no-training model artifact/inference-readiness audit.
- Verify the serialized model, model scaler(s), feature/channel order, exact preprocessing contract, threshold provenance and smoke-test inference path.
- Create `docs/MODEL_BASELINE_FREEZE.md` only after the audit.
- Then implement FastAPI and connect the React dashboard.

### Verification state
- Documentation files were written to branch `feature/cross-patient-feature-stability-audit` through GitHub.
- Documentation commits:
  - Experiment history: `a94e9f0f5b3e00c1c1a38b0ef3129b33f7c7d086`
  - Project handoff: `65b1955fdc01945d0bb75375f3005104aa815173`
  - Current status: `141bd6f1b0fd13970d2ce1f162460715adf17ad9`
  - AI handoff: `03d83476003d2b275e01fd002b86810523187365`
  - Master plan: `6777ab130d5957e2a797edfd3d1fe89e86a0dcd6`
  - README: `f2e745fd4b5051146ecfb374e583cab5a92c9a80`
- This documentation update did not run local tests, train a model, modify model scores or change dataset artifacts.


---

## 2026-11-06 — Teammate handoff and local inference verification reconciliation

### Local checks reported by the project owner
- Robust 170-feature Logistic Regression model deserialized successfully in the primary developer's environment.
- Fold-specific scaler checkpoint restored and finalized successfully.
- Validation feature cache checked: 216,005 rows × 170 features.
- Cached-feature → model → archived validation-score parity passed for 216,005 validation rows; maximum absolute probability difference approximately 2.33e-15.
- These checks cover saved features onward only. Raw processed EEG → preprocessing → scaler → robust feature extraction → model parity remains pending.
- These local checks have not yet been independently reproduced by a fresh teammate clone. They are not a formal model freeze or deployment verification.

### Teammate-facing documentation
- Added `docs/TEAMMATE_ONBOARDING.md` with repository/branch setup, Python 3.11 environment instructions, documentation reading order, source map, scientific safeguards, current progress and first assigned task.
- Added `docs/ARTIFACT_TRANSFER_GUIDE.md` describing local-only artifact categories, source inventory, SHA-256 verification, transfer acceptance and receiver checks.
- Updated README, CURRENT_STATUS, PROJECT_HANDOFF and AI_HANDOFF to point teammates to the new guides and distinguish cached-feature parity from pending raw-EEG inference parity.
- Artifact transfer itself is not confirmed. The guide intentionally leaves bundle identity, hashes and receiver acceptance pending until a real transfer is performed.

### Next milestone
- Verify exact channel/filter/window/feature contract from source.
- Complete raw processed-recording inference parity using an approved non-sealed development recording and already processed NPZ data.
- Do not retrain or tune against patient_01.
- Freeze the model only after parity passes; then implement FastAPI and connect the React dashboard.

### Scope of this documentation update
- No model training, retraining, threshold tuning or data reprocessing was performed.
- No local test suite was run as part of this documentation update.
