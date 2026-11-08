# Fold-aware feature stability audit

## Purpose

This diagnostic audit extends the original cross-patient feature-distribution
screen. For each inner OOF patient, it reconstructs that patient's fold matrix
using the same patient-excluded channel scaler used by the OOF pipeline.

The held-out patient is excluded from fitting the channel scaler. Its feature
distribution is compared with peer-patient distributions transformed under the
same fold scaler, so the values are on a common fold-specific scale.

## Inputs

- Existing robust inner-fit and inner-validation feature caches.
- Existing 608-recording channel-statistics cache.
- Existing per-window raw channel-standard-deviation cache.
- Existing processed NPZ files and full annotation audit.
- Existing completed 22-patient OOF report.

No EDF files are needed. The audit does not train models, alter OOF scores,
select features, tune thresholds, or access outer-test patient_01.

## Run

From the EEG project root, with the project's Python environment active:

```powershell
python -m pytest tests/test_fold_aware_feature_stability.py -q
python -m scripts.audit_fold_aware_feature_stability --audit ..\..\docs\full_annotation_audit.csv
```

Default output:

`results/diagnostics/fold_aware_feature_stability_v1/`

The script refuses to overwrite an existing output directory. If an interrupted
run leaves a partial feature-work folder, the existing OOF matrix builder can
resume it when its progress signature and dimensions are valid. Do not delete
cache folders while a run is active.

## Outputs

- `report.json`: cohort, fold completion, existing OOF ranking context,
  methodology, safeguards, and summary.
- `fold_aware_patient_feature_distributions.csv`: one row per patient,
  channel, feature, and class (22 × 17 × 10 × 2 = 7,480 rows).

Each patient's row is measured in the matrix reconstructed for that patient's
own held-out OOF fold. Peer medians and peer IQRs are calculated from the other
21 patients in that same fold matrix, giving each peer patient equal weight.

## Resource and runtime notes

Only one 652,077 × 170 float32 fold matrix is reconstructed at a time. This
is approximately 443 MB for X.npy, plus labels and temporary memory. The matrix
is removed after its fold is summarized. A previously retained matching fold
matrix may be reused; stale/incomplete matrices are rebuilt by the resumable
OOF matrix builder. The audit may take substantial time because it reconstructs
22 fold matrices and summarizes distributions across the cohort. Keep the
computer plugged in and do not interrupt Python or remove cache files.

## Interpretation limits

- 10-second windows with 5-second stride overlap and are not independent.
- Seizure-positive windows are sparse and unevenly distributed.
- Patient median shifts are descriptive screens, not hypothesis tests or
  causal evidence.
- A large shift does not prove that feature drift caused weak OOF ranking.
- OOF AUROC/AUPRC are included as descriptive context, not independent
  validation or outer-test performance.
- This is an academic research prototype, not a clinical diagnostic device.
