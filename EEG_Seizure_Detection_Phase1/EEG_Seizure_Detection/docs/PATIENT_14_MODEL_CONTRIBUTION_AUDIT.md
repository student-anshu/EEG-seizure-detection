# Patient-wise OOF model-contribution audit

## Purpose

The initial diagnostic focused on patient_14 (AUROC about 0.44 and AUPRC about
0.0026). The same audit can now be run for any inner-OOF patient other than
patient_01. We will compare patients 13, 14 and 15 because they showed weak
threshold-independent discrimination. The purpose is to determine whether
feature-family and channel-level contribution patterns recur across patients.

## Method and safeguards

`scripts/audit_patient14_model_contributions.py` deterministically reconstructs
the selected patient's existing inner-OOF fold using the same 21-patient training set,
patient-excluded channel scaler, fold feature matrix, incremental
`StandardScaler.partial_fit` sequence, and balanced liblinear Logistic
Regression configuration.

**The audit will not interpret coefficients unless reconstructed probabilities
match that patient's saved OOF artifact within an absolute tolerance of 2e-6.**
It validates recording order, labels, and score count before writing results.

- Outer-test patient_01 is excluded before processed recording access.
- No threshold is selected or changed.
- Existing OOF score artifacts and event metrics are not overwritten.
- This is a deterministic reconstruction of one already evaluated fold, not a
  new candidate model or performance claim.
- Coefficients and additive logit contributions are associations within this
  fitted model, not causal or physiological effects.

## Run

From `EEG_Seizure_Detection` with the project's `.venv311` activated:

```powershell
git pull origin feature/threshold-strategy-score-artifacts
python -m pytest tests/test_patient14_model_contributions.py -q
python -m scripts.audit_patient14_model_contributions --patient patient_13 --audit ..\\..\\docs\\full_annotation_audit.csv
python -m scripts.audit_patient14_model_contributions --patient patient_14 --audit ..\\..\\docs\\full_annotation_audit.csv
python -m scripts.audit_patient14_model_contributions --patient patient_15 --audit ..\\..\\docs\\full_annotation_audit.csv
```

The diagnostic rebuilds the 608-recording fold feature matrix for the selected patient from existing caches and fits the same Logistic Regression configuration. It may take
some time and needs several hundred MB of temporary memory/disk. It does not
require EDF downloads or reprocessing.

## Outputs

- `results/diagnostics/{patient_id}_model_contribution_audit_v1.json`
  - score reconstruction check;
  - average additive logit contribution by feature family, split by positive
    and negative windows;
  - largest absolute coefficients, with channel and feature names;
  - channel-by-feature contribution summaries to expose channel-level cancellation;
  - interpretation limitations.
- `results/diagnostics/{patient_id}_recording_contributions_v1.csv`
  - per-recording positive/negative mean contribution for each feature family.
- `results/diagnostics/{patient_id}_channel_feature_contributions_v1.csv`
  - per-channel, per-feature contribution means and positive-minus-negative differences.

A negative contribution pushes the model logit downward relative to zero; it
does not mean a feature is biologically protective. Positive-window counts vary by patient; overlapping windows are correlated. Treat
per-recording summaries as descriptive, not independent-sample inference.

## Decision gate

1. If reconstruction fails, stop and investigate the pipeline discrepancy;
   do not interpret the coefficient report.
2. If reconstruction passes, inspect whether feature families consistently
   push positive windows toward lower logits, and whether that differs by
   recording.
3. Do not reverse features, remove channels, or change the model based on one
   patient. Compare patients 13, 14 and 15 first. Any candidate change must be
   justified across the full patient-grouped OOF cohort and evaluated later
   under the sealed outer-test protocol.
