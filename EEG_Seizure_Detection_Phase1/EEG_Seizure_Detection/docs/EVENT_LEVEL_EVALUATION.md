# Event-level evaluation protocol

## Purpose

Window-level classification metrics do not tell us how many annotated seizure
events the system detects or how often it raises alarms during non-seizure
recording time. This module provides deterministic post-processing and
recording-level event metrics. It is an evaluation utility, not a clinical
alarm system.

## Frozen semantics in version 1

- Window predictions are binary and must be generated using a threshold chosen
  on inner validation data. This module never tunes thresholds.
- Each positive window maps to the half-open interval
  `[window_start, window_start + window_seconds)`.
- Overlapping or touching positive windows are merged into one alarm.
- An optional `merge_gap_seconds` can merge alarms separated by a small gap.
  It defaults to zero and must be selected using inner validation only.
- A seizure event is detected if at least one alarm overlaps its annotated
  interval (half-open interval overlap; boundary-only contact is not overlap).
- Detection delay is `max(0, first_overlapping_alarm.start - seizure.start)`.
  An alarm already active at annotated onset therefore has zero delay.
- An alarm is false only if it overlaps no annotated seizure event.
- False alarms/hour = false alarm count / non-seizure recording hours.
  Annotated seizure duration is subtracted from recording duration.
- Event sensitivity is undefined (`null`) for a recording with no annotated
  seizures. False alarms/hour is undefined if there is no non-seizure exposure.

## Exact recording duration is mandatory

Newly processed NPZ files persist `recording_duration_seconds` from the
actual loaded signal sample count divided by sampling frequency, and the
metadata CSV repeats this value. For existing artifacts, first run the read-only
`scripts/audit_recording_durations.py` to inspect duration values already retained
in NPZ files, aligned metadata CSVs and `processed/recording_durations.json`.
The original EDF dataset is currently unavailable in the working copy, so the
EDF-header backfill utility cannot be used unless the source files are restored.
If no exact duration survives, event sensitivity, event counts, detection delay
and false-alarm counts can still be reported from the annotation manifest and
window timestamps. The evaluator accepts a missing duration for these metrics.
In that case non-seizure exposure and false alarms/hour are `null`; pooled
false alarms/hour is also `null` if any included recording lacks exact exposure.
Never substitute the last complete window end as the recording duration.
The last complete window end is only a lower bound because a trailing partial
window may exist. Do **not** use that lower bound as the false-alarm-rate
denominator.

## Matching and aggregation

Each annotated seizure event is evaluated independently: any overlapping
alarm detects that event. Thus one long alarm can overlap two annotations and
detect both; alarm-merging parameters must be frozen and reported. A false
alarm is an alarm with no overlap with any seizure event.

`aggregate_event_metrics` pools event counts and non-seizure exposure across
recordings. If any recording has unknown exposure, pooled exposure and pooled
false alarms/hour are `null`, while event counts, event sensitivity and false-
alarm counts remain available. Compute patient-macro metrics separately by
first aggregating recordings within each canonical patient; do not treat
recording-level values as independent patients.

## Limitations

The overlap-based match is intentionally transparent and has no extra onset
tolerance. It does not model clinical seizure alarm requirements, intervention
latency, or alarm refractory policies. Any alternative matching rule must be
predeclared, validated on inner patients, and versioned before outer-test use.


## First-fold saved-score integration

`scripts/evaluate_first_fold_events.py` evaluates the existing first-fold
inner-validation and outer-test score checkpoints without fitting or loading a
new model. It reconstructs the deterministic patient-grouped split, verifies
the score checkpoint signatures/checksums, and aligns concatenated score
slices to each recording by comparing saved score labels with the NPZ labels.
Any length or label mismatch fails closed.

The script reports event counts, event sensitivity, detection delay and
false-alarm counts for the validation-selected F1 and F2 thresholds. It writes
per-recording and per-patient summaries to
`results/baselines/first_fold_patient_01_event_metrics_v1.json`. Since exact
recording durations are unavailable in the retained cohort, exposure and
false alarms/hour remain null. This is one outer fold only and is not a
full-cohort performance estimate.

Run from the `EEG_Seizure_Detection` directory:

```powershell
python -m scripts.evaluate_first_fold_events --audit ..\\..\\docs\\full_annotation_audit.csv
```

This is a post-processing/evaluation step over saved scores; it does not
retrain Logistic Regression or regenerate EEG features.


## Per-event error analysis

Each recording result also stores an `events` list with one row per annotated
seizure: chronological event index, annotation start/end, detected flag, first
overlapping alarm start, detection delay, and number of matching alarms.
Missed events have null first-alarm and delay fields. This allows the team to
identify exactly which seizure annotations were missed or detected late,
rather than relying only on one aggregate sensitivity value.


Each event record also includes score diagnostics calculated from saved model
scores: number of windows overlapping the annotated event, maximum and median
overlapping-window scores, start time of the highest-scoring overlapping
window, maximum-score margin relative to the selected threshold, and maximum
score in a 30-second context on each side. These values distinguish a
near-threshold miss from an event for which the model assigned uniformly low
scores. Context scores are descriptive only and are not used to change the
threshold or evaluate/tune the model.

## Validation-only event-threshold sweep

`scripts/sweep_event_thresholds.py` evaluates candidate thresholds against
the existing checksum-validated first-fold inner-validation score artifact.
It loads processed labels/timestamps once, verifies the saved score labels
against each recording, and does not train a model or regenerate features.

The candidate set consists of 121 score quantiles from 0.95 through 1.00,
plus the existing validation F1 and F2 thresholds. The selection objective is
macro-average patient event F2 across seizure-bearing inner-validation
patients. Its precision term is an explicitly exploratory proxy:

`detected seizure events / (detected seizure events + false-alarm episodes)`.

This proxy penalizes false-alarm episodes but is not false alarms/hour. Exact
recording exposure is unavailable, so no rate constraint or clinical alarm
claim can be made. Ties are resolved by higher macro event sensitivity, then
fewer false-alarm episodes, then the higher threshold.

After the validation threshold is frozen, the script evaluates the outer-test
patient once at the selected threshold and at the two existing thresholds for
descriptive comparison only. Outer-test outcomes are never used for selection.
The full validation candidate table and the frozen outer-test comparison are
written to `results/baselines/first_fold_event_threshold_sweep_v1.json`.

Run from the `EEG_Seizure_Detection` directory:

```powershell
python -m scripts.sweep_event_thresholds --audit ..\\..\\docs\\full_annotation_audit.csv
```

This is an exploratory first-fold threshold experiment, not a full nested
23-fold result. The selected threshold must not be treated as a general
deployment threshold until repeated nested patient-wise validation is complete.


---

## Exact-duration source correction — primary cohort inventory

The historical processed NPZ files were audited and contain no `recording_duration_seconds` field. However, a separate committed inventory file is available at repository path `docs/signal_inventory.csv` (repository root). A direct identity-based audit against `docs/full_annotation_audit.csv` verified:

- 683/683 primary-cohort recordings matched to an inventory row.
- No missing, invalid or internally inconsistent duration values.
- Each duration agrees with `n_samples / sampling_frequency_hz` within 0.01 seconds.
- Total raw primary-cohort duration is 979.9286 hours.

Therefore, the earlier statement that FA/hour is unavailable for the existing cohort is superseded. The evaluator should load the inventory as an explicit duration source and join on canonical recording identity (prefer normalized `relative_path`; validate `subject` and `edf_file` too). Never align by CSV row order.

For each recording:
1. Validate finite positive `duration_seconds`, positive `n_samples` and `sampling_frequency_hz`.
2. Verify `duration_seconds ≈ n_samples / sampling_frequency_hz` within a documented tolerance.
3. Parse seizure annotations, clip to the recording interval and calculate the union duration of annotated seizure intervals (avoid double-subtracting overlapping annotations).
4. Calculate non-seizure exposure = recording duration − union seizure duration.
5. Calculate false alarms/hour = false-alarm episode count / (non-seizure exposure / 3600).

Fail closed on missing/duplicate/mismatched inventory identities or invalid exposure. Keep the existing policy of never inferring duration from the final complete window. Add deterministic tests for successful joins, duplicate/missing paths, inconsistent duration arithmetic, overlapping seizure intervals, zero exposure and patient-level aggregation.

The 979.9286-hour figure is raw recording exposure, not non-seizure exposure and not a false-alarm-rate denominator. Patient-macro FA/hour should be computed from each patient's summed false alarms divided by that patient's summed non-seizure exposure; also report per-recording values and pooled exposure-weighted FA/hour. Do not average recording FA/hour values as though recordings were equal duration.

The saved-score first-fold evaluator can be extended to use this inventory without retraining. Recompute the first-fold event metrics as a post-processing step and retain the old result artifact for provenance; write a versioned result that records the duration source and matching audit. This first-fold update does not replace full 23-fold evaluation.


## Exact-duration integration implementation — version 2

The first-fold saved-score evaluator now reads repository-root
`docs/signal_inventory.csv` by default (or accepts `--inventory`). It
validates inventory identity and duration arithmetic, then joins primary-cohort
recordings by normalized `relative_path`, cross-checking `subject` and
`edf_file`. It never uses row order. Inventory rows outside the primary
cohort are counted as unmatched extras and excluded from evaluation.

Seizure annotations are clipped to true recording bounds before subtracting
their union from recording duration. Per-recording, patient-pooled and
cohort-pooled non-seizure exposure and false alarms/hour are reported. Patient
rates use summed false alarms divided by summed non-seizure exposure, not an
average of recording rates. Zero non-seizure exposure yields a null rate.

The updated evaluator writes
`results/baselines/first_fold_patient_01_event_metrics_v2.json`; v1 is
retained unchanged for provenance. This is saved-score post-processing only:
no model fit, feature extraction, threshold selection, or outer-test tuning.

Run from the `EEG_Seizure_Detection` directory:

```powershell
python -m scripts.evaluate_first_fold_events --audit ..\\..\\docs\\full_annotation_audit.csv
```

Override the inventory path with `--inventory PATH` if needed. New tests cover
normalized identity joins, duplicate and missing rows, inconsistent duration
arithmetic, and identity mismatches.


---

## Full 22-patient inner OOF event evaluation — implementation

`scripts/evaluate_inner_oof_events.py` evaluates the saved patient-held-out
OOF score artifacts without retraining or re-extracting features.

The evaluator:
- Reconstructs the canonical 22-patient inner OOF cohort from the annotation
  audit and explicitly excludes outer-test `patient_01`.
- Validates every fold artifact, patient identity, ordered recording offsets,
  contiguous score slices, score/label dimensions, report counts, and exact
  processed-recording identity.
- Aligns each recording's saved OOF scores with its processed labels and
  timestamps; any identity, ordering, or label mismatch stops evaluation.
- Joins the exact duration inventory by normalized recording identity, with
  subject and EDF filename cross-checks.
- Reports event sensitivity, missed events, detection delay, false-alarm
  episodes, non-seizure exposure and false alarms/hour at each requested
  threshold, with recording-, patient- and cohort-level summaries.
- Does not fit a model, extract features, select thresholds, or load
  `patient_01`.

By default it reports both the fixed 0.5 reference threshold and the
provisional pooled OOF window-F2 threshold stored in `report.json`. The
pooled threshold was selected using these same OOF labels; its event metrics
are descriptive and must not be presented as an unbiased estimate. Scores
from the 22 distinct fold models may also differ in calibration. Threshold
selection and cross-fold calibration review remain separate follow-up work.

Run from the `EEG_Seizure_Detection` directory:

```powershell
python -m pytest tests/test_inner_oof_event_evaluation.py -q
python -m scripts.evaluate_inner_oof_events --audit ..\\..\\docs\\full_annotation_audit.csv
```

Default output:
`results/baselines/inner_oof_event_metrics_v1.json`.

The user should run the focused tests and inspect the complete evaluator
output before any threshold or alarm-policy decision is made.
