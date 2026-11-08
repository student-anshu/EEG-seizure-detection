# CHB-MIT Patient Mapping and LOPO Evaluation Protocol

## Patient identity mapping

CHB01 and CHB21 represent the same individual and map to patient_01.
All other subjects retain their numeric identity. This produces 23
independent patient groups.

## Audit manifest and primary cohort

The complete source manifest is docs/full_annotation_audit.csv. It contains
686 successfully audited recordings and 198 annotated seizure events.

The primary experiment uses the fixed 17-channel longitudinal bipolar
montage defined in docs/PRIMARY_COHORT_AND_MONTAGE_POLICY.md. Three CHB12
recordings (chb12_27.edf, chb12_28.edf, chb12_29.edf) are excluded because
their montage is incompatible with the required longitudinal bipolar input.
They contain 13 annotated seizure events.

The full audit remains unchanged. The primary cohort contains 683 recordings
and 185 annotated seizure events across 23 independent patients. The cohort
filter is applied before LOPO folds are built.

## Leave-One-Patient-Out

- 23 outer folds, one per independent patient.
- All eligible recordings from one patient form the test set.
- All eligible recordings from the other 22 patients form the training set.
- Every primary-cohort recording is tested exactly once.
- No patient appears in both training and test sets in a fold.

## Inner patient-grouped validation

Validation is built only from each outer fold's 22 training patients.
The deterministic holdout uses 20% validation fraction, random state 42,
and ceiling rounding (5 validation patients, 17 training patients).
All recordings from a patient stay together. The outer test patient never
participates in model selection.

Both inner training and validation must contain seizure events. The nested
inspection script verifies this for every fold. Do not use outer-test metrics
to select hyperparameters, thresholds, epochs, or preprocessing settings.

## Inspection commands

Run from the project code directory:

python -m unittest discover -s tests -v
python -m scripts.inspect_lopo_splits
python -m scripts.inspect_nested_validation

Both split scripts load the full annotation audit, apply the primary cohort
filter, and then construct patient-wise folds.

## Scope

This protocol defines metadata-level folds only. It does not train models,
load signals, fit normalization, or produce performance metrics.
