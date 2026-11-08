# Baseline Feature Checkpoints and Resume Behavior

## Purpose

The first-fold Logistic Regression runner stores extracted feature matrices as disk-backed NumPy arrays. It writes a progress manifest after each completed recording so an interrupted feature-extraction stage can continue from the last committed recording.

## Local location

The first-fold runner uses:

`EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/cache/first_fold/`

This directory is ignored by Git. It contains temporary experiment artifacts and must not be committed.

## What is checkpointed

For each feature stage (inner fit, inner validation, outer train, outer test), the cache stores:

- `X.npy`: float32 feature matrix, written through a NumPy memory map.
- `y.npy`: int8 binary labels.
- `manifest.json`: cache signature, matrix shape/dtypes, completed recording count, and completed window cursor.

After each recording, the runner writes that recording's features and labels, flushes both arrays, and atomically updates the manifest. If the process stops while extracting a recording, that recording is recomputed on restart; recordings already committed in the manifest are skipped.

## Cache validity and leakage controls

A cache directory is keyed by:
- exact fitted channel-scaler mean and scale,
- scaler epsilon,
- channel and feature names,
- recording identities and seizure counts,
- processed NPZ source-file sizes and modification timestamps,
- cache format version.

Inner-fit and inner-validation features use the scaler fitted only on inner-fit patients. Outer-train and outer-test features use the separate scaler fitted only on outer-training patients. These stages therefore cannot accidentally share features produced under a different fold scaler.

If the signature or stored array metadata does not match, the cache is rebuilt rather than reused.

## RAM and disk behavior

- Feature arrays are stored on disk as uncompressed `.npy` files and opened as memory maps.
- Feature extraction still uses RAM for the current recording and temporary arrays.
- Scikit-learn may allocate additional working memory during model fitting; memory mapping does not eliminate model-training RAM requirements.
- Approximate feature matrix disk sizes for fold 1 are 368 MiB (inner fit), 182 MiB (inner validation), 550 MiB (outer train), and 45 MiB (outer test). Total temporary cache peak is approximately 1.1 GiB, excluding small manifests and filesystem overhead.
- The runner removes the fold cache only after the final JSON result is saved successfully. On exceptions or Ctrl+C, cache files are preserved for resumption.

## Resume command

From the project directory, with the project's Python 3.11 virtual environment active:

```powershell
python -u .\scripts\run_first_baseline_fold.py
```

Use the same dataset, processed recordings, seed, and code/configuration when resuming. A changed scaler, source NPZ, cohort, feature definition, or cache version creates a different cache signature and will trigger fresh extraction.

## Limitations

- The current runner checkpoints feature extraction, not the internal state of scikit-learn's Logistic Regression solver. If interrupted during model fitting or threshold selection, the model fit/threshold step runs again, but completed feature extraction is reused.
- The outer channel scaler is recomputed when the runner restarts before outer feature extraction. This is a streaming pass over processed recordings and does not require feature re-extraction.
- This implementation is wired into the first-fold runner. The same cache utility can be reused when the full 23-fold runner is implemented, with a unique cache key per fold and stage.
