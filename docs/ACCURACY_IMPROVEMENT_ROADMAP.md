# EEG Seizure Detection — Accuracy Improvement and Experiment Roadmap

**Repository:** `shivkumar15/EEG`  
**Planning status:** Proposed execution baseline; no new model experiment is authorized until Gate 0 is passed.  
**Research task:** Cross-patient seizure detection from CHB-MIT scalp EEG windows. This is not seizure prediction and is not a medical device.

## 1. Why the project is not ready for another accuracy experiment

The data foundation is substantially complete:

- Primary cohort: 683 recordings, 23 independent patients, 185 annotated seizure events.
- 704,836 ten-second windows with five-second stride: 702,280 normal and 2,556 seizure windows (about 0.36% positive).
- Patient-wise 23-fold LOPO split construction and fit-only fold normalization have been implemented and audited.
- The original feature representation has 221 features per window (17 channels × 13 features).

The first-fold Logistic Regression baseline (held-out patient_01) produced:

| Metric | Outer-test result |
|---|---:|
| AUROC | 0.9043 |
| AUPRC | 0.0304 |
| Sensitivity | 2.04% |
| Specificity | 99.88% |
| Precision | 4.55% |
| F1 | 0.0282 |
| TP / FP / FN / TN | 3 / 63 / 144 / 52,549 |

This is a weak operating point: the ranking metric is materially better than chance, but the selected threshold misses almost all positive windows. It does **not** prove that the model is accurate, nor does it establish why sensitivity is low. The validation and test score distributions differ; possible explanations include patient heterogeneity, model score instability, and threshold transfer. These are hypotheses to test, not established causes.

A patient-balanced validation threshold sweep did not produce a meaningful improvement: its tiny macro-F2 gain came with lower sensitivity and lower pooled F2. Threshold-only tuning is therefore paused.

A second feature experiment (`relative_spectral_shape_v1`, 170 features/window) has been added as a candidate. Its full-data runtime and scientific performance have not yet been established. It is **not** the new baseline and must not be launched as an unbounded full-cohort run.

## 2. Research principles (non-negotiable)

1. **Patient independence:** no patient may appear in both training and evaluation partitions of a fold. Overlapping windows must never be randomly split across train and test.
2. **Outer-test isolation:** outer-test labels cannot influence features, normalization, channel selection, class weights, hyperparameters, calibration, threshold, or alarm post-processing.
3. **One comparison protocol:** candidate models use the same cohort, outer folds, label rules, and reporting metrics.
4. **Event-level evaluation:** window metrics are necessary but insufficient. A clinically meaningful research report must also count detected seizure events and false alarms per hour.
5. **No metric gaming:** accuracy and pooled window F1 are not primary objectives under approximately 0.36% positive-window prevalence. Report patient-level results and the full operating curve.
6. **Evidence labels:** separate unit-tested, synthetic-tested, first-fold-tested, full-LOPO-tested, and event-evaluated claims.
7. **Reproducibility:** every experiment has a frozen configuration, code/data signature, seed, runtime, checkpoint policy, and machine-readable result.

## 3. What success means

### Primary reporting

Report the following for every outer patient and across all 23 outer folds:

- Event sensitivity: proportion of annotated seizure events matched by at least one alarm.
- False alarms per hour of non-seizure recording time.
- Window-level AUPRC and AUROC.
- Window-level sensitivity, specificity, precision, F1 and F2 at a threshold selected without outer-test data.
- Confusion counts and class support.
- Patient-macro mean, median, range and fold-wise distribution; pooled metrics are supplementary.

### Operating points

Do not select a threshold by repeatedly maximizing F1/F2 on the same small validation subset. Instead, predefine and report event-sensitivity versus false-alarms/hour operating curves. Compare sensitivity at several clearly labelled false-alarm budgets (for example 0.1, 0.5 and 1.0 false alarms/hour); these are reporting points, not claims that any one budget is clinically acceptable.

Before computing event metrics, freeze and document:
- How consecutive positive windows are merged into an alarm.
- Whether a small gap between alarms is merged.
- The minimum alarm duration, if any.
- The event-to-alarm matching rule and any onset tolerance.
- How non-seizure hours are counted.

These settings may be selected on inner validation only. They must be frozen before outer-test scoring. Report the exact rule so results are reproducible.

## 4. Accuracy improvement workstreams

### Workstream A — evaluation and error-analysis foundation

**Goal:** Ensure that the project optimizes the right outcome and can explain every miss/false alarm.

Tasks:
1. **Implemented on the active branch (local tests pending):** `evaluation/event_metrics.py` converts binary window predictions into merged alarm intervals, matches alarms to seizure annotations, calculates event sensitivity/detection delay, and computes false alarms/hour using non-seizure exposure.
2. **Implemented:** deterministic tests cover alarm merging, short-gap policy, detected/missed events, false alarms, boundary-only contact, multiple events, manifest interval parsing and pooled aggregation.
3. **Implemented:** newly processed NPZ and metadata artifacts persist exact EDF recording duration. Existing processed NPZ files predate this field and must be backfilled from EDF headers or reprocessed before event false-alarm rates are calculated.
4. **Implemented and locally verified:** `scripts/backfill_recording_durations.py` performs restartable EDF-header-only duration backfill, updates aligned metadata CSVs and a central JSON manifest, and preserves existing NPZ bytes. Synthetic tests cover idempotent resume, dry-run, missing EDF handling and NPZ preservation. Focused tests 4/4 and full suite 151 passed, 4 subtests passed, 4 skipped.
5. Next: integrate per-recording event evaluation into the first-fold runner without tuning on outer-test labels.
5. Keep window metrics and event metrics in separate result fields.

**Exit gate:** event metrics pass deterministic synthetic tests; alarm rules are documented and frozen before any model comparison.

### Workstream B — establish a reliable classical baseline

**Goal:** Establish a trustworthy 23-patient performance reference before deep learning.

Candidates, in this order:
1. Logistic Regression using the existing 221-feature representation.
2. Linear SVM and RBF SVM using the same representation.
3. Random Forest using the same representation.
4. Only then compare the 170-feature relative-spectral representation as a controlled challenger.

For each candidate:
- Reuse cached features across models; do not re-extract features for every classifier.
- Fit all transforms on inner-fit patients only.
- Select hyperparameters and operating thresholds using inner patient groups only.
- Store validation and outer-test raw scores, predictions, fold metrics and configuration.
- Do not change multiple factors in one experiment.

**Threshold/refit issue:** the existing first-fold baseline selected a threshold using one fitted model and evaluated a separately refitted model on all outer-training recordings. That can create a score-distribution mismatch. The newer same-model validation/test runner avoids that mismatch but trains on only the 17 inner-fit patients, not all 22 outer-training patients. Neither protocol should be silently assumed optimal. Compare these protocols on a controlled first-fold engineering pilot, quantify the training-data/runtime trade-off, then freeze one protocol before the full 23-fold run. Do not use outer-test outcomes to choose the protocol.

**Exit gate:** one complete, restartable 23-fold classical baseline with per-fold results and event-level metrics. No model is called the winner from fold 1 alone.

### Workstream C — targeted error reduction

Only after Workstreams A and B:

1. Inspect which patients and seizure events are missed; determine whether errors cluster by patient, event duration, recording, or signal quality.
2. Test class-weighting alternatives on training data only.
3. Test patient-robust/relative spectral features as a single-variable comparison.
4. Test temporal smoothing/alarm merging on validation data only.
5. Assess score calibration and threshold stability across validation patients.
6. Consider channel selection only inside each training fold; never use outer-test labels or global feature importance to select channels.
7. Keep a result table with hypothesis, changed variable, compute cost, validation result, outer-test result, and decision (keep/reject/inconclusive).

Do not use repeated threshold sweeps as a substitute for model improvement. Do not use synthetic signals as evidence of real-EEG accuracy.

**Exit gate:** each retained change improves the predeclared event/false-alarm operating objective across patients, not merely one fold or one pooled window metric.

### Workstream D — deep-learning comparisons

Start only after a full classical baseline and event evaluation exist.

Compare, in order:
1. Multichannel 1D CNN.
2. CNN + BiLSTM.
3. CNN + BiLSTM + temporal attention.

Use the same patient-wise outer folds and inner validation protocol. Use training-only class weighting or carefully controlled sampling. Keep model capacity, regularization, early stopping and checkpointing explicit. Include ablations so the contribution of BiLSTM and attention is measurable. Do not assume the most complex architecture is best.

**Exit gate:** a deep model demonstrates repeatable improvement over the classical baseline on patient-level and event-level metrics, with variability reported.

### Workstream E — application integration

Only after model and inference preprocessing are frozen:
- Package preprocessing and inference into a versioned Python API.
- Verify inference preprocessing exactly matches training preprocessing.
- Integrate FastAPI, then React.
- Display model estimates and limitations; never label outputs as a medical diagnosis.

## 5. Compute and execution policy

No multi-hour experiment is launched until a short preflight reports:

- Number of recordings/windows to process.
- Estimated cache size and free disk space.
- A timed sample of representative recordings (including a seizure recording and a long recording).
- Peak memory and elapsed time per stage.
- Whether cached work will survive interruption.
- Exact models to be trained and whether model fitting itself is checkpointable.

Required execution sequence:
1. Static compile/import check.
2. Focused unit tests.
3. Full regression tests.
4. Small synthetic/integration smoke test.
5. Timed real-data pilot on a small, representative subset.
6. Only after reviewing the measured runtime and cache size, run one complete first-fold pilot.
7. Review outputs and approve the 23-fold schedule.

The pilot is an engineering/runtime check, not a performance claim. Every long job must checkpoint at recording/fold/model-stage boundaries, write atomically, and safely resume. Do not rerun completed stages unnecessarily. Do not delete raw EDFs, processed cohort files, or useful checkpoints during cleanup.

## 6. Immediate execution sequence

| Order | Task | Status / decision |
|---|---|---|
| 0 | Freeze this roadmap and reconcile phase numbering/branch status in project docs | This document establishes the proposed roadmap |
| 1 | Stop or leave paused the unverified robust-feature run; do not start it again yet | Required before further experiments |
| 2 | Install pytest and run focused feature tests | **Complete (5 Nov 2026):** focused robust-feature tests 6/6 passed in 1.20 s; full repository suite 136 passed, 4 subtests passed, 4 skipped, 0 failures/errors in 3.95 s |
| 3 | Implement event-level metrics, alarm post-processing, synthetic tests and exact-duration persistence | **Complete:** focused 12/12 and full suite 147 passed; 4 subtests passed, 4 skipped |
| 4 | Implement restartable exact-duration backfill | **Complete:** 4/4 focused tests; full suite 151 passed |
| 5 | Dry-run duration backfill on the real 683-recording cohort | Next; inspect failures before writing |
| 6 | Integrate event metrics into the first-fold runner | After duration coverage is verified |
| 7 | Inspect/optimize scaler and feature-cache runtime using a timed pilot | Pending; no full-cohort run |
| 6 | Finalize the threshold/refit protocol comparison | Controlled pilot only |
| 7 | Run first-fold engineering pilot and inspect per-event errors | After gates 2–6 |
| 8 | Build and run restartable 23-fold classical evaluation | After pilot approval |
| 9 | Run targeted feature/imbalance/model comparisons | After full baseline |
| 10 | Run deep-learning models and ablations | After classical baseline |
| 11 | Freeze model, API, React integration and final report | Final stage |

## 7. Execution focus

Phase labels are not a blocker for this work. Continue the accuracy-improvement pipeline in the execution order above; do not spend further time reconciling conversational phase numbers.

## 8. Decision rule

A result is retained only if:
- The code and tests pass.
- No patient leakage is introduced.
- The change is evaluated on identical outer folds.
- The primary event/false-alarm objective improves consistently or a clearly justified trade-off is documented.
- Runtime and memory are acceptable and reproducible.

If a candidate does not meet these conditions, record it as rejected or inconclusive and move to the next planned hypothesis. Do not keep changing the plan in response to one disappointing fold.

## Validation-only event threshold sweep (implementation added)

A saved-score-only threshold sweep is implemented in
`scripts/sweep_event_thresholds.py`. It uses 121 score quantiles between
0.95 and 1.00 plus the existing validation F1/F2 thresholds. Selection is
based exclusively on macro patient event F2 on seizure-bearing inner-validation
patients. Because exact recording durations are unavailable, precision uses
an explicitly labelled event-F2 proxy that penalizes false-alarm episode
counts; false alarms/hour is not calculated. The outer-test patient is only
evaluated after the validation threshold is frozen and is not used for
selection. User execution and verification are pending. This is exploratory
first-fold evidence only, not a replacement for nested LOPO evaluation.


---

## CTO roadmap update — reusable features and patient-wise operating evaluation (5 Nov 2026)

This section supersedes earlier statements in this document that exact recording durations are unavailable. The primary cohort's durations have now been recovered from the committed inventory file; see the verified audit below.

### A. Verified duration source and false-alarm-rate recovery

A direct audit matched `docs/signal_inventory.csv` to the 683 recordings in the primary cohort derived from `docs/full_annotation_audit.csv`:

- Inventory rows: 686; all 686 report successful header inspection.
- Primary-cohort recordings matched: 683/683.
- Missing, invalid, or internally inconsistent duration rows: 0.
- Every matched duration agrees with `n_samples / sampling_frequency_hz` within 0.01 seconds.
- Sum of primary-cohort recording durations: 979.9286 hours (raw recording exposure, before subtracting annotated seizure time).

**Decision:** use `docs/signal_inventory.csv` as the exact-duration source for existing processed recordings. Join by normalized `relative_path` / `subject + edf_file`, never by row order or filename alone. Validate one-to-one matching and duration consistency before evaluation. For each recording, non-seizure exposure is recording duration minus the union of annotated seizure intervals, bounded to the recording interval. Then calculate false alarms/hour. If any recording lacks a valid exact duration or annotation alignment, fail closed for the affected aggregate rate; do not silently substitute window-end estimates.

The previous NPZ duration audit remains true: the historical NPZ files contain no duration field. The inventory is a separate verified source and resolves this limitation without restoring or reprocessing EDF files.

### B. Fast, reusable feature extraction

**Approved direction:** create a deterministic, fold-independent feature cache from the existing processed EEG windows. Store features with canonical patient ID, recording path, window start, label, feature-set version, channel/montage version, sampling frequency and source signature. Prefer partitioned Parquet or memory-mapped NumPy arrays with a manifest over one giant CSV. Keep cache artifacts out of Git.

- Extract deterministic features once; do not fit a learned scaler during feature extraction.
- Parallelize by recording only after a timed memory/CPU preflight. Bound worker count and chunk size to avoid RAM pressure.
- Keep feature extraction independent of labels; labels and timestamps are stored alongside features for later split selection.
- Any model-level `StandardScaler` remains inside the estimator pipeline and is fitted on the training patients for that CV fold only.
- Cache identity must include source checksums/signatures, feature definitions/version, channel order, window parameters and preprocessing version.

### C. Feature ablations

Run these as controlled, separately versioned experiments on the same patient folds:

1. **Scale-free core:** relative delta/theta/alpha/beta/gamma power, normalized spectral entropy, skewness, excess kurtosis and line length divided by standard deviation.
2. **Amplitude ablation:** add `log_std` to the scale-free core; do not assume amplitude is either useful or harmful.
3. **Channel representation:** compare per-channel features, channel-pooled median + IQR, and their combination. Test channel maximum as a separate ablation because it can be driven by a noisy channel.
4. Preserve the legacy 221-feature representation and its saved baseline as the reference. Do not replace it based on a first-fold result.

The current 170-feature `relative_spectral_shape_v1` implementation includes `log_std` and is currently computed after fold-fitted channel standardization. It is an existing exploratory implementation, not yet the approved final scale-free cache design. Update or create a separate feature version rather than silently changing its semantics or reusing incompatible caches.

### D. Patient-wise model selection

Within each outer LOPO fold, use only its 22 outer-training patients for model selection. Generate inner out-of-fold scores using patient groups; no patient's recordings or overlapping windows may cross a group boundary. `StratifiedGroupKFold` may be used only after verifying that each fold has adequate seizure-bearing patient/event support. Stratifying millions of windows alone does not guarantee patient/event balance; add explicit fold diagnostics and deterministic fallback logic if needed.

Approved, bounded comparisons:

- Logistic Regression regularization `C` on a logarithmic grid (initial grid: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1; optionally 10).
- Compare class weighting, patient-balanced weighting, and their combination as separate configurations. Define patient weights using training data only and document normalization so one patient does not dominate by window count.
- Compare one tree-based challenger only. Random Forest already exists in the codebase and is the first low-cost candidate; consider `HistGradientBoostingClassifier` only if there is a clear reason to replace or supplement it. Do not expand into a broad model zoo.
- Do not use SMOTE or window-level oversampling as the default; retain natural temporal structure and evaluate weighting first.

All hyperparameters, weighting, calibration, feature selection and alarm parameters are selected from inner-training/OOF data only. Store OOF scores with patient ID, recording ID, window start, label and experiment signature.

**Score-transfer safeguard:** thresholds selected on inner OOF scores may not transfer unchanged to a final model refit on all 22 outer-training patients. Explicitly evaluate the calibration/score-scale transfer protocol using inner data only, then freeze it before outer-test scoring. Do not choose the protocol based on outer-test results.

### E. Alarm policy and operating points

Extend the existing event evaluator with validation-only comparisons of:
- Existing consecutive-positive-window merging.
- Rolling score means over 3–6 windows.
- K-of-N rules.
- Short-gap merging.

Because windows are 10 seconds with a 5-second stride, report the added detection delay from smoothing and persistence rules. Freeze all post-processing parameters using inner OOF/validation predictions.

Select/report sensitivity at predeclared false-alarm budgets such as 0.1, 0.5 and 1.0 false alarms per hour. These are research comparison points, not clinically established safety limits. Never optimize on outer-test labels. Report event sensitivity, false alarms/hour, delay and window metrics together.

### F. Final nested evaluation and learning curves

After the pipeline is frozen, run all 23 outer LOPO folds once with durable per-fold results and restart-safe scheduling. Each outer fold independently derives its feature/model/alarm choices from its own 22 training patients. Report one row per held-out patient, then patient-macro median, IQR and mean; also report pooled metrics as supplementary results.

Learning curves at 4, 8, 12 and 17 training patients are optional, lower priority than completing the 23-fold evaluation. If run, keep held-out patients and all preprocessing/model-selection rules patient-independent.

### G. Deferred ideas and application work

Defer trailing-baseline normalization until the scale-free feature and alarm experiments are complete. If later tested, it must use only preceding signal history, have an explicit warm-up policy, use no labels, and be identical in training and inference. Defer external Siena validation until the CHB-MIT pipeline and report are stable.

Prepare the API contract and dashboard against a clearly labelled saved baseline if useful for parallel development, but do not imply that it is the final validated model. Freeze preprocessing/model artifacts and inference parity before presenting real model outputs. The application must state that outputs are research estimates, not diagnosis.

### H. Immediate execution order

1. Integrate the verified inventory durations into event evaluation and add tests for path matching, duration consistency, seizure-exposure subtraction and FA/hour aggregation.
2. Finish or safely pause the currently running first-fold robust-feature job; preserve all checkpoints and caches. Do not launch another multi-hour run until its actual result/runtime is reviewed.
3. Build the reusable, deterministic scale-free feature cache with metadata and bounded parallel extraction.
4. Add per-patient metrics, score/feature distribution diagnostics and a post-hoc patient_01 operating curve (diagnostic only).
5. Run patient-grouped inner OOF model/weighting/feature comparisons.
6. Compare alarm smoothing and K-of-N rules using inner OOF data.
7. Freeze the pipeline, complete 23 outer folds, and then integrate the validated inference pipeline.

Time-box engineering and classical experiments, but do not trade away leakage safeguards or required outer-fold coverage to meet a calendar target. A negative or inconclusive result is a valid research outcome.


## Execution update — 5 November 2026

**Duration-inventory integration is implemented and committed; local verification is pending.**
The saved-score first-fold evaluator now joins exact recording durations from
`docs/signal_inventory.csv` using normalized recording identity, validates
duration arithmetic, clips seizure annotations to recording bounds, and
calculates per-recording, patient-pooled and cohort-pooled FA/hour using
non-seizure exposure. The previous v1 result remains unchanged; v2 is written
to a separate output path. Focused tests were added for joins, integrity
failures, overlapping annotation union, zero exposure and FA/hour aggregation.

Next:
1. Pull the branch and run focused tests plus the v2 saved-score evaluation.
2. Review the completed robust-feature JSON artifact (not committed; keep its
   cache/checkpoints intact).
3. Design the reusable scale-free feature cache only after those two review
   gates are closed.


---

## CTO research target — event sensitivity and false-alarm operating point

For this academic research project, use **at least 90% event sensitivity**
as an aspirational development target, with **no more than 1.0 false alarm
per non-seizure hour** as the primary research operating point. Additionally
report sensitivity at 0.1 and 0.5 false alarms/hour, detection delay, and
patient-level distributions. These are research engineering objectives, not
clinical safety or medical-device acceptance criteria.

Success must be assessed across all 23 outer patient-level folds using both
pooled event counts and patient-macro summaries. A pooled 90% score alone is
insufficient if performance is poor for individual patients. The first-fold
robust result (4/11 events, 36.36%, at 2.281 false alarms/hour) is currently
well below this target and cannot establish generalization.

Next: compare legacy and robust per-event outcomes descriptively, then design
alarm-policy experiments using patient-grouped inner OOF predictions only.
Never tune from the already observed patient_01 outer-test results.
