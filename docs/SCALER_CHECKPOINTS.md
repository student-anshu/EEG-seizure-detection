# Resumable Channel-Scaler Checkpoints

## Scope

This document describes the per-recording checkpoint mechanism introduced for fold-specific EEG channel normalization. It is an engineering recovery feature; it does not change the statistical definition of the scaler or the patient-wise split design.

## What is checkpointed

The scaler uses a streaming merge of population statistics. After each fully loaded and validated recording, it persists:

- the number of EEG samples accumulated per channel;
- the running channel mean;
- the running sum of squared deviations (M2);
- the channel count;
- the number of completed recordings and the name of the state file.

The saved state is sufficient to continue the same Welford-style population-statistics calculation without rereading completed recordings.

## Atomic write protocol

For recording N:

1. Load and validate the recording windows.
2. Merge that recording into the in-memory count, mean and M2.
3. Write a new compressed state file to a temporary path.
4. Flush and fsync the state file, then atomically replace its final state filename.
5. Write and fsync a temporary JSON manifest.
6. Atomically replace the manifest. Only this final step advances the committed recording cursor.

If the process stops before the manifest replacement, the previous manifest remains authoritative. The unreferenced state file is ignored and the unfinished recording is recomputed on restart.

## Checkpoint identity

The signature includes:

- checkpoint schema and accumulator algorithm version;
- scaler epsilon;
- caller-provided stage/fold/patient context;
- ordered recording identities, patient IDs and seizure counts;
- processed NPZ absolute paths, file sizes and modification timestamps.

An inner-fit scaler and an outer-training scaler use different contexts and checkpoint directories. The inner-fit signature includes the outer held-out patient and the inner-fit/validation patient groups. The outer-training signature includes the held-out patient and all outer-training patient IDs.

A changed cohort, split, source file fingerprint, epsilon or context rejects the old checkpoint and starts the scaler fit again.

## Current integration

`prepare_lopo_fold(..., scaler_checkpoint_dir=...)` checkpoints the inner-fit scaler. The first-fold runner supplies a fold-specific path under:

`cache/first_fold/scalers/fold_1_inner_fit`

`fit_channel_standardizer(..., checkpoint_dir=..., checkpoint_context=...)` checkpoints the all-outer-training scaler under:

`cache/first_fold/scalers/fold_1_outer_train`

Progress is reported every 25 recordings and at completion. Existing callers that omit a checkpoint directory retain the previous non-checkpointed streaming behavior.

The first-fold runner removes its temporary cache directory only after the final fold result has been atomically saved. If an exception or interruption occurs before that point, scaler and feature checkpoints remain available for a compatible restart.

## Validation requirements

The new unit tests cover:

- interruption after complete recording checkpoints;
- resuming from the last committed recording;
- exact agreement with uninterrupted scaler statistics;
- reusing a completed scaler without loading recordings;
- rejecting a checkpoint with an incompatible fold/context signature;
- rejecting a corrupted state file and rebuilding it.

The tests must be run locally before using the checkpointed runner on the full dataset.

## Explicit limitations

This milestone checkpoints scaler fitting and retains the existing per-recording feature cache. It does **not** checkpoint a partially fitted Logistic Regression estimator, validation scoring, final outer-test scoring, or aggregation across all 23 LOPO folds. Logistic Regression fitting must restart if interrupted during its fit. The first-fold runner is still a first-fold engineering runner, not the final 23-fold runner.

Do not describe the whole experiment as end-to-end resumable until fold/model result checkpoints and the 23-fold aggregation workflow have also been implemented and interruption-tested.