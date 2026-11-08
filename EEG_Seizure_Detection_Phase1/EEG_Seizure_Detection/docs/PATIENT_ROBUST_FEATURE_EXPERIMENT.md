# Patient-robust feature experiment

## Motivation
The first-fold Logistic Regression baseline has highly variable sensitivity across
validation patients. The current 13-per-channel representation includes several
absolute-amplitude statistics and absolute spectral powers. The new representation
tests whether relative spectral composition and scale-normalized temporal shape
generalize better across patients.

## New representation
For each channel and 10-second window, extract 10 deterministic features:
- log standard deviation
- line length divided by standard deviation
- skewness
- excess kurtosis
- relative delta, theta, alpha, beta, and gamma power
- normalized spectral entropy

This produces 170 features for the configured 17-channel montage, compared with
221 in the legacy representation. Relative band power is invariant to uniform
amplitude scaling (up to floating-point tolerance).

## Scientific safeguards
- Feature extraction is deterministic and does not inspect labels.
- The channel standardizer is fitted on inner-fit recordings only.
- The same fitted model is used to select a threshold on inner validation and
  score the held-out outer-test patient. This deliberately avoids transferring
  a threshold between two separately fitted models in this diagnostic experiment.
- Outer-test labels are used only for final metric calculation, never for model
  selection, threshold selection, or feature design.
- Compare patient-wise and pooled AUPRC, AUROC, F2, sensitivity, and precision.
- This is a first-fold engineering experiment, not final 23-fold evidence.

## Run
From the EEG project root:
```powershell
python -m pytest tests/test_robust_feature_extraction.py -q
```

Run the first-fold experiment from the project root:

```powershell
python -m pytest tests/test_robust_feature_extraction.py -q
python -u .\\scripts\\run_robust_feature_first_fold.py
```

The runner caches completed feature extraction per recording and resumes from
its manifest after interruption. Feature extraction is processed in 256-window
chunks so temporary float64 and Welch PSD arrays remain bounded; a recording is
marked complete only after all its chunks have been written. The default run
uses Logistic Regression only, matching the legacy baseline model and keeping
the first comparison controlled. Random Forest remains available explicitly:
`--models logistic_regression random_forest`. Results and model/score artifacts
are written under `results/baselines/robust_feature_first_fold_v1/`. The original
baseline features, result JSON, and checkpoints are preserved.


---

## Next approved feature-cache design (5 Nov 2026)

The current `relative_spectral_shape_v1` is an exploratory 170-feature first-fold experiment. It includes `log_std` and currently receives fold-standardized channel data. Do not silently redefine this version or reuse its cache after changing feature semantics.

The next reusable representation should be a separate version:

### Scale-free core

- Relative delta/theta/alpha/beta/gamma power.
- Normalized spectral entropy.
- Skewness and excess kurtosis.
- Line length divided by standard deviation.

Test `log_std` as a separate amplitude-sensitive ablation, not as an assumed member of the scale-free core. Positive uniform scaling of a channel should not change the scale-free features (within numerical tolerance); add invariance tests.

### Channel aggregation ablations

Compare:
1. Per-channel feature vector.
2. Median and IQR across the 17 channels for each feature.
3. Per-channel plus pooled median/IQR.
4. Maximum pooling as a separate experiment, with attention to noisy-channel sensitivity.

### Cache contract

Extract deterministic features once from the existing processed windows without fitting a learned channel scaler. Store feature rows aligned with:
- Canonical patient ID.
- Recording relative path and subject/EDF identity.
- Window start time and binary label.
- Sampling frequency and channel/montage order.
- Feature-set/version and preprocessing/window configuration.
- Source artifact signature/checksum.

Use partitioned Parquet or memory-mapped NumPy arrays plus a manifest. Avoid a single giant CSV and keep generated caches/results out of Git. Parallelize by recording only after a measured memory/CPU preflight and with a bounded worker count. A model-level scaler may still be used inside the estimator pipeline, fitted on each training fold only.

### Evaluation design

Use identical patient-grouped inner folds for every feature ablation and model comparison. Keep the legacy 221-feature baseline unchanged as the reference. Do not claim a new feature set wins based on one outer fold; compare across all outer patients after the pipeline is frozen.
