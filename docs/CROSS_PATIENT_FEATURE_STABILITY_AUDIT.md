# Cross-patient robust-feature stability audit

## Purpose

This read-only diagnostic compares the existing 170 robust EEG features across
the 22 inner patients used by the first outer fold. It is intended to help
separate patient-to-patient feature-distribution differences from weak
within-patient seizure/non-seizure ranking.

It does not train a model, re-extract features, tune thresholds, select or
remove features, modify OOF score artifacts, or open the sealed outer-test
patient_01 recordings.

## Inputs

- `cache/robust_feature_first_fold_v1/inner_fit` and
  `inner_validation` feature/label arrays.
- `docs/full_annotation_audit.csv` and the existing processed NPZ
  label arrays for recording identity and row-alignment validation.
- `cache/oof_patient_scores_v1/report.json` for existing patient-wise
  AUROC/AUPRC context only.

The script reconstructs the deterministic first outer fold and refuses to
continue unless patient_01 is the held-out patient, 608 recordings remain in
outer training, the two feature caches are complete, all row labels align, and
the OOF report contains exactly the expected 22 inner patients.

## Method

For every patient, channel, feature, and class (non-seizure/seizure), it
calculates the window-level 25th percentile, median, 75th percentile, IQR, and
number of windows. It then compares each patient median with the distribution
of the other patients' medians for that same channel-feature-class cell:

`(patient median - peer-patient median) / peer-patient IQR`.

Each patient contributes one median to the peer reference, so patients with
longer recordings do not dominate the reference. A zero peer IQR produces a
null standardized shift rather than division by zero.

## Outputs

Written to `results/diagnostics/cross_patient_feature_stability_v1/`:

- `report.json`: cohort integrity, method, OOF ranking context, largest
  descriptive shifts, and limitations.
- `patient_cohort_summary.csv`: patient recording/window/positive
  counts and existing OOF AUROC/AUPRC.
- `patient_feature_distributions.csv`: per-patient class-conditional
  feature distributions and peer-patient shift scores.

## Run

From the EEG project root:

```powershell
python -m pytest tests/test_cross_patient_feature_stability.py -q
python -m scripts.audit_cross_patient_feature_stability --audit ....docsull_annotation_audit.csv
```

Choose a new `--output` directory if the default output already exists;
the script refuses to overwrite a prior audit.

## Interpretation safeguards

- This is descriptive screening, not an independent statistical test.
- The existing first-fold robust cache uses a scaler fitted on inner-fit
  recordings. Inner-fit patients therefore contributed to that scaler; the
  distribution summaries are not unbiased held-out estimates.
- Ten-second windows use a five-second stride and overlap. Window counts are
  not independent sample sizes.
- Seizure windows and seizure-positive recordings are sparse and unevenly
  distributed across patients.
- A large shift does not establish a biological cause or prove that
  covariate shift explains poor model ranking.
- Do not remove features or change the model based on this report alone.
- Patient_01 remains sealed. This is not clinical validation.
