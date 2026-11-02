# EEG Seizure Detection — Project Master Plan

**Working repository:** https://github.com/shivkumar15/EEG  
**Original source repository:** https://github.com/student-anshu/EEG-seizure-detection  
**Current status:** Phase 0 — audit and correctness.

## Objective
Build an academic binary EEG-window seizure detection system. This is seizure detection, not prediction of a future seizure. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Research architecture
Proposed primary model: EEG preprocessing and channel harmonization → training-fold-only channel selection → multichannel 1D CNN → BiLSTM → temporal attention → Dense sigmoid output.

Required baselines and ablations: feature-based SVM/Random Forest, 1D CNN, CNN+BiLSTM, and CNN+BiLSTM+attention. This architecture is a research hypothesis; compare all models under identical subject-wise splits.

## Data and preprocessing
Initial dataset: CHB-MIT Scalp EEG. The current repository contains selected EDF and summary files under CHB01–CHB03; completeness and processed outputs have not been established. Current code settings include 10-second windows, 5-second stride, 0.5–40 Hz bandpass and 60 Hz notch. Validate these choices against the data and sampling rate.

Rules:
- Parse real CHB-MIT seizure start/end annotation formats correctly. Never silently convert annotation failures into non-seizure labels.
- Retain subject ID, EDF filename, recording ID, window start/end, sampling rate, channels and label in metadata.
- Never randomly split overlapping windows from the same patient across train and test.
- Fit normalization, channel/feature selection, class weights, thresholds and hyperparameters using training/validation subjects only. Never fit them on a held-out test subject.
- Record exclusions and their reasons.

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

React is a proposed direction; no completed React application was found in the inspected tree.

## Execution roadmap
### Phase 0 — audit and correctness (CURRENT)
- [ ] Verify layout, imports and paths.
- [ ] Correct CHB-MIT annotation parsing against actual summaries.
- [ ] Add tests for seizure/non-seizure recordings, missing/malformed annotations and interval boundaries.
- [ ] Validate channel matching, filtering, resampling and normalization policy.
- [ ] Correct .gitignore scope for dataset/generated artifacts.
- [ ] Decide raw EDF storage/distribution policy.
- [ ] Reproduce previously reported CHB01 processing counts from a clean run.

### Phase 1 — multi-subject preparation
- [ ] Create subject/recording inventory and annotation audit.
- [ ] Process a defined cohort reproducibly.
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

## Audit findings to verify before code changes
1. Current parser uses numbered keys such as `Seizure 1 Start Time`; CHB-MIT summaries may use unnumbered `Seizure Start Time` / `Seizure End Time`. The parser in this repository still uses numbered-only regexes.
2. `config.py` derives `dataset` beneath the code folder, but the tree places `dataset` beside `EEG_Seizure_Detection` under `EEG_Seizure_Detection_Phase1`. Confirm and correct.
3. README setup commands and Python import/package conventions need verification against actual run paths.
4. Channel selection exists as a separate module; integration into the active pipeline is not yet established.
5. Earlier reported CHB01 counts: 8 EDFs, 5,497 windows (5,397 normal; 100 seizure), 17 channels, 256 Hz, 10s windows and 5s stride. These are **reported but not independently reproduced**.
6. Raw EDF binaries exist in the repository. Review provenance, dataset terms, repository size and storage before adding more.

## Upstream synchronization policy
Develop and test in `shivkumar15/EEG` first. The original `student-anshu/EEG-seizure-detection` repository remains untouched unless a separate successful upstream update is explicitly performed. For every code change, record exact paths, rationale, test commands/results and a copy-ready patch in `docs/CHANGELOG_FOR_UPSTREAM.md`. Never claim an upstream change was made unless verified. Do not sync secrets, local environments, generated data, raw EEG or model weights without an explicit decision.

## Working rules
One phase at a time. Do not train before annotation correctness is established. Every change needs a test or explanation of why testing is unavailable. Distinguish verified results from plans, assumptions and unverified reports. Explain each change and exactly what the user must mirror upstream.
