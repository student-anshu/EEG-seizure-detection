# Fold-wise EEG normalization protocol

## Decision

The recording-preparation stage stores filtered EEG windows without applying
per-recording normalization. Normalization is fitted later, independently
inside each evaluation fold.

This avoids calculating normalization statistics from an entire held-out
validation or test recording and makes the same fitted transform reusable at
training, validation, test, and inference time.

## Transform

`preprocessing.normalization.FoldChannelStandardizer` expects a tensor with
shape `(windows, channels, samples)`. It calculates one mean and standard
deviation per channel over the windows and sample axes:

- `mean[channel] = mean(training_windows[:, channel, :])`
- `scale[channel] = max(std(training_windows[:, channel, :]), 1e-6)`
- `normalized = (window - mean) / scale`

The transform returns float32 arrays and rejects empty, non-finite, or
wrong-shaped inputs. A constant channel remains finite because its scale is
floored at the configured epsilon.

## Leakage-prevention contract

For every outer patient-wise LOPO fold:

1. Build the outer training and held-out test patient groups first.
2. Build the inner fit and validation patient groups using only outer-training
   patients.
3. Collect windows only from the inner fit patients and call `fit()` once.
4. Use that same fitted scaler to transform fit, validation, and outer-test
   windows. Never call `fit()` or `fit_transform()` on validation or test.
5. Refit a fresh scaler for every outer fold. Do not share scaler statistics
   between folds.

The standardizer does not inspect patient IDs or enforce splits itself. The
fold orchestration code must pass only the permitted fit-patient windows.

## Window-overlap note

The current window configuration is 10 seconds with a 5-second stride.
Consequently, samples in overlapping windows contribute more than once to
the fit statistics. This is a deliberate window-weighted fit definition for
the current baseline pipeline and must be kept consistent across experiments.
If a later experiment changes the fitting unit to unique continuous samples
or recordings, document it as a protocol change and compare it separately.

## Preprocessing and inference consistency

The EDF preprocessing function applies the configured bandpass/notch filters,
segments the signal, and saves unnormalized filtered windows. The saved NPZ
and metadata mark normalization as
`none_fold_train_only_scaling_required`.

The old `robust_normalize()` helper remains available for isolated legacy
tests but is no longer used by `process_edf()`. Do not train on previously
generated NPZ files that were created by the older per-recording-normalized
pipeline; regenerate them after the preprocessing protocol is finalized.

The current zero-phase filters remain offline processing. A causal streaming
implementation will require a separately validated filter/inference path.

## Current limitations

- This standardizer is channel-wise global standardization, not a claim that
  it is the uniquely optimal EEG normalization method.
- It must be compared against justified alternatives during model
  development, without using outer-test results to choose the method.
- Full-cohort processing and model training remain blocked until the fold
  orchestration loads the primary cohort and enforces this fit/transform
  contract end-to-end.
