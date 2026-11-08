# Completed Model-Stage Checkpoints

## Scope

This layer adds durable checkpoints around completed model operations in the first-fold Logistic Regression runner. It does not serialize an in-progress scikit-learn solver and does not yet implement the complete 23-fold runner.

## Persisted stages

For fold 1, the runner persists:

1. **Inner model fit** — the fitted Logistic Regression pipeline, saved with joblib only after `fit()` returns successfully.
2. **Inner validation scores** — aligned validation labels and continuous positive-class scores.
3. **Validation threshold and metrics** — the F1-maximizing threshold selected using inner-validation labels only.
4. **Final outer-training model fit** — the refitted pipeline trained on all outer-training patients.
5. **Outer-test scores** — labels and continuous scores for the held-out patient. The frozen validation threshold is applied after scores are recovered.
6. **Fold result** — the complete JSON result, written after evaluation and before the user-facing output file is written.

A restart reuses only checkpoints whose manifest signature and artifact SHA-256 match. Validation/test label arrays are also compared against the current split before cached scores are accepted.

## Identity and integrity

The experiment signature includes fold and held-out patient identity, random seed, model configuration, feature/channel definitions, and ordered recording identities with processed NPZ absolute paths, sizes and modification timestamps for inner-fit, validation, outer-training and outer-test recordings.

Artifacts are written to temporary files and atomically replaced. A JSON manifest is written after the artifact and records the checkpoint schema version, experiment signature, stage, artifact name, kind, completion marker and SHA-256 digest. Missing, corrupt, incompatible or incomplete artifacts are rejected and that stage is recomputed.

Runtime artifacts live under `cache/first_fold/model_stages/fold_1_logistic_regression`; they must not be committed to Git.

## Important estimator limitation

Scikit-learn's configured Logistic Regression (`solver="liblinear"`) does not expose a supported, safe continuation checkpoint for its in-progress optimization. If the process stops during `estimator.fit()`, that individual fit restarts from the beginning. Once fit returns and the fitted estimator is saved, later scoring or evaluation interruptions can reuse it.

This is stage-boundary recovery, not mid-fit recovery. Do not claim that the first-fold or complete 23-fold experiment is fully resumable yet.

## Validation still required

The unit tests for this layer must verify:
- completed estimator save/load;
- validation/test score array round-trip;
- signature mismatch rejection;
- artifact corruption rejection;
- invalid labels/scores rejection.

The runner must also be exercised locally with a small synthetic or controlled interruption test before another expensive CHB-MIT fold run. Full 23-fold fold-result skipping and aggregation remain future work.
