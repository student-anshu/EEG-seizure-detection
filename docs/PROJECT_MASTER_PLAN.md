# EEG Seizure Detection — Project Master Plan

**Working repository:** https://github.com/shivkumar15/EEG  
**Original source repository:** https://github.com/student-anshu/EEG-seizure-detection  
**Current status:** Phase 0 — final audit and synchronization preparation.

## Objective
Build an academic binary EEG-window seizure detection system. This is seizure detection, not prediction of a future seizure. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Research architecture
Proposed primary model: EEG preprocessing and channel harmonization → training-fold-only channel selection → multichannel 1D CNN → BiLSTM → temporal attention → Dense sigmoid output.

Required baselines and ablations: feature-based SVM/Random Forest, 1D CNN, CNN+BiLSTM, and CNN+BiLSTM+attention. This architecture is a research hypothesis; compare all models under identical subject-wise splits.

## Current verified data subset
The working repository currently contains 10 selected EDF recordings from CHB01–CHB03:
- CHB01: 8 EDFs.
- CHB02: 1 EDF.
- CHB03: 1 EDF.

All 10 inspected recordings have 256 Hz sampling, 23 raw channels, and the same 17 selected EEG channels. Nine recordings are 3,600 seconds; `chb01_26.edf` is 2,325 seconds. No resampling was needed for this inspected subset.

Verified preprocessing settings:
- 10-second windows.
- 5-second stride (50% overlap).
- Fourth-order Butterworth bandpass: 0.5–40 Hz.
- Optional 60 Hz notch (skipped if at/above Nyquist).
- Median/IQR normalization computed over each complete recording (offline, non-causal).

Verified generated dataset:
| Subject | Recordings | Windows | Normal | Seizure |
|---|---:|---:|---:|---:|
| CHB01 | 8 | 5,497 | 5,397 | 100 |
| CHB02 | 1 | 719 | 719 | 0 |
| CHB03 | 1 | 719 | 707 | 12 |
| **Total** | **10** | **6,935** | **6,823** | **112** |

All 10 NPZ files and 10 metadata CSV files passed checks for expected dimensions, finite EEG values, binary labels, matching row counts, matching labels/timestamps and no missing CSV values. The current selected subset has approximately 1.61% seizure windows.

**Scope warning:** This is a small, selected subset covering only three subjects. It is suitable for pipeline verification, not for claiming robust patient-independent seizure detection or generalization.

## Data and preprocessing rules
- Parse real CHB-MIT seizure start/end annotation formats correctly. Never silently convert annotation failures into non-seizure labels.
- Retain subject ID, EDF filename, recording ID, window start/end, sampling rate, channels and label in metadata.
- Never randomly split overlapping windows from the same patient across train and test.
- Fit normalization, channel/feature selection, class weights, thresholds and hyperparameters using training/validation subjects only. Never fit them on a held-out test subject.
- Record exclusions and their reasons.
- Re-check sampling rates and channel availability when expanding beyond the currently inspected 10 EDFs.
- Current whole-recording median/IQR normalization is offline and non-causal. Revisit its suitability for the chosen evaluation protocol and make inference preprocessing consistent with training.

## Evaluation
Preferred protocol: Leave-One-Subject-Out (LOSO), with patient-wise validation inside the training subjects. If cohort size makes this unstable, report limitations and use an explicitly defined grouped split; do not use random window splits.

Report sensitivity, specificity, precision, F1, AUROC, AUPRC, confusion matrix, class support, per-subject metrics and fold-wise variability/confidence intervals where feasible. Report window-level and event-level metrics separately. Define event matching, gap merging and alarm refractory period before reporting false alarms per hour.

## Explainability
Add SHAP or another justified method only after model selection and evaluation are stable. Document output being explained, reference samples and limitations. Explanations are not clinical evidence.

## Application architecture
- Python for signal processing, training and inference.
- MNE, NumPy, SciPy, pandas, scikit-learn and one consistently selected deep-learning framework (TensorFlow/Keras is the current proposal); SHAP after validation.
- FastAPI backend.
- React + Vite + TypeScript + Tailwind frontend; Plotly for charts.
- File-based metadata/results initially; add a database only for a concrete requirement.
- Inference preprocessing must match validated training preprocessing.
- UI must call outputs model estimates, not diagnoses.

## Execution roadmap
### Phase 0 — audit and correctness
- [x] Verify layout, imports and paths.
- [x] Correct CHB-MIT annotation parsing against actual summaries.
- [x] Add tests for annotation parsing and edge cases.
- [x] Validate channel matching, filtering, segmentation and normalization behavior for the current subset.
- [x] Reproduce CHB01 processing counts from a clean processed-output run.
- [x] Finalize raw EDF provenance/license and storage/distribution policy; keep raw EDFs out of Git and preserve local copies.
- [x] Correct README dataset path, repository links, documentation links, and setup examples to match the verified folder layout.
- [ ] Synchronize to the original repository and complete its clean-checkout/test verification.

### Phase 1 — multi-subject preparation
- [ ] Expand and document a justified subject/recording cohort.
- [ ] Create subject/recording inventory and annotation audit.
- [ ] Process the defined cohort reproducibly.
- [ ] Export windows with traceable metadata.
- [ ] Generate subject/class distribution and quality-control report.

### Phase 2 — leakage-safe protocol
- [ ] Build subject-wise/LOSO split generator.
- [ ] Add automated leakage assertions.
- [ ] Fit channel selection and transforms on training folds only.
- [ ] Freeze split manifests.

### Phase 3 — baselines
- [ ] Feature-based SVM/Random Forest baseline.
- [ ] 1D CNN baseline.
- [ ] Save configurations, seeds, checkpoints and fold predictions.

### Phase 4 — proposed model
- [ ] Implement CNN + BiLSTM + temporal attention.
- [ ] Run ablations with identical splits.
- [ ] Tune only on training/validation subjects.

### Phase 5 — evaluation
- [ ] Complete LOSO evaluation.
- [ ] Produce window-level and event-level metrics.
- [ ] Analyze errors per subject/recording.
- [ ] Document limitations and uncertainty.

### Phase 6 — explainability
- [ ] Add justified SHAP/explanation workflow.
- [ ] Document limitations and compare explanations across folds.

### Phase 7 — FastAPI
- [ ] Define versioned request/response contracts.
- [ ] Load validated model and matching preprocessing artifacts.
- [ ] Add validation, health endpoint, logging and tests.

### Phase 8 — React frontend
- [ ] Build signal view, upload/sample workflow and prediction display.
- [ ] Show probability, threshold, metadata and clear non-diagnostic disclaimer.
- [ ] Integrate API and test end to end.

### Phase 9 — final validation/submission
- [ ] Reproduce results from documented commands.
- [ ] Verify clean installation and setup.
- [ ] Prepare report, presentation, diagrams and demo.
- [ ] Document dataset provenance, ethics, limits and contributions.

## Dataset provenance and distribution policy
- Source: [CHB-MIT Scalp EEG Database, PhysioNet v1.0.0](https://physionet.org/content/chbmit/1.0.0/).
- DOI: `10.13026/C2K01R`.
- PhysioNet lists the [Open Data Commons Attribution License v1.0 (ODC-By)](https://physionet.org/content/chbmit/view-license/1.0.0/).
- ODC-By governs rights in the database; the license text says it does not automatically cover separate rights in individual contents. This project records the dataset's stated database license but does not make a blanket legal determination about every recording.
- Raw EDFs and subject summary files are excluded from the current Git tree. Local copies remain available to the developer, and the root `.gitignore` prevents accidental recommit.
- Do not copy raw EDFs or summary files into the original team repository. Team members should download the dataset from PhysioNet independently and preserve attribution/license notices.
- If the database or a derivative database is publicly conveyed, follow ODC-By notice and license requirements. If produced works are publicly used, include the source/license notice required by the license. Cite PhysioNet in reports and project documentation.
- Commit `abfd931` removed 10 EDFs and 3 summary files from current Git tracking. Earlier Git history still contains the prior versions; no history rewrite was performed.

## Remaining audit items
1. The active `main.py --all` loop catches per-file exceptions and prints `SKIPPED`; future larger-cohort runs must record and audit skips explicitly.
2. The selected montage's `T8-P8` channel was not present under the expected name in the inspected recordings; the 17 selected channels were otherwise consistent.
3. Channel ranking exists as a helper and must be integrated only inside training folds, never globally before splitting.
4. The original `student-anshu/EEG-seizure-detection` repository remains untouched until synchronization and verification are explicitly completed.

## Upstream synchronization policy
Develop and test in `shivkumar15/EEG` first. Once Phase 0 is complete, back up the original code folder, copy the verified code folder, preserve the original dataset, and exclude virtual environments and generated outputs. Review the diff, commit/push to the original repository, then run the complete tests and `python main.py --inspect` there. Never claim the original repository was updated until confirmed.

## Working rules
One phase at a time. Do not train before annotation correctness, cohort design and leakage-safe splitting are established. Every change needs a test or explanation of why testing is unavailable. Distinguish verified results from plans, assumptions and unverified reports.
