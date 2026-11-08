# Local Artifact Transfer Guide

**Last reviewed:** 6 November 2026  
**Repository:** https://github.com/shivkumar15/EEG  
**Branch:** `feature/cross-patient-feature-stability-audit`

## Purpose

A Git clone is not a complete reproduction environment. EEG data, processed NPZs, runtime caches and some experiment artifacts are intentionally excluded from Git. A missing local file does not mean the research must be rerun.

**Important:** paths below are known from the primary developer's recorded environment. Their current presence, complete contents and hashes must be checked on the source machine before transfer. This document does not claim that a transfer bundle already exists.

## 1. Transfer categories

| Category | Examples | Transfer policy |
|---|---|---|
| Source code and docs | tracked Python files, tests, `docs/*.md` | Obtain through Git |
| Robust model | `results/baselines/robust_feature_first_fold_v1/logistic_regression/model.joblib` | Include in approved artifact bundle if not reliably available in clone |
| Scaler checkpoint | `cache/robust_feature_first_fold_v1/scaler_checkpoint/` | Local/cache artifact; transfer with manifest and all required state files |
| Validation feature cache | `cache/robust_feature_first_fold_v1/inner_validation/` or exact current cache path | Transfer only if required for parity/reference; confirm exact path |
| OOF caches | `cache/oof_patient_scores_v1/` | Research outputs; not required for basic inference if not used by inference contract |
| Processed NPZ recordings | `processed/` (actual location must be inspected) | Required for local processed-recording parity; transfer approved development recordings only |
| Raw EDF dataset | Original CHB-MIT archive | Not required for immediate application work; do not ask for redownload by default |
| Virtual environment | `.venv311/` | Never transfer; recreate from dependency declarations |

Some historical notes contain different cache-root names. Read the actual runner configuration and cache manifests before copying; do not assume a path solely from this guide.

## 2. Known local verification context

The primary developer previously verified:
- Robust model deserializes as a scikit-learn Pipeline with 170 input features.
- Fold scaler checkpoint has a completed manifest for 469 recordings and state file `state_000469.npz`.
- Restored/finalized scaler has 17 channels and 1,116,344,320 accumulated samples.
- Validation feature cache has 216,005 rows and 170 columns.
- Cached-feature-to-model parity passed on 216,005 validation rows; maximum absolute probability difference was approximately 2.33e-15.

These are local verification results. A teammate must reproduce artifact loading and hash verification against transferred files. They do not establish raw-EEG-to-model parity.

## 3. Source-machine inventory

On the source Windows machine:

```powershell
Set-Location "D:\2026-THE ENDGAME\byteBazzar\EEG"
.\.venv311\Scripts\Activate.ps1
Set-Location .\EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection
```

Inspect expected locations:

```powershell
Get-ChildItem .\results\baselines\robust_feature_first_fold_v1 -Recurse -File
Get-ChildItem .\cache\robust_feature_first_fold_v1 -Recurse -File
Get-ChildItem .\cache\oof_patient_scores_v1 -Recurse -File -ErrorAction SilentlyContinue
Get-ChildItem .\processed -Recurse -Filter *.npz -File -ErrorAction SilentlyContinue
```

Do not infer absence from one guessed directory. Check runner configuration and cache manifests for the actual paths.

## 4. Create a checksum manifest

After selecting the exact approved files/directories, create a manifest. Example:

```powershell
$files = @(
  ".\results\baselines\robust_feature_first_fold_v1\logistic_regression\model.joblib",
  ".\cache\robust_feature_first_fold_v1\scaler_checkpoint\manifest.json",
  ".\cache\robust_feature_first_fold_v1\scaler_checkpoint\state_000469.npz"
)
$files | ForEach-Object {
  if (Test-Path $_) {
    Get-FileHash -Algorithm SHA256 $_
  } else {
    Write-Warning "MISSING: $_"
  }
}
```

Correct these example paths to match the actual source-machine inventory. Prefer a manifest containing relative path, byte size and SHA-256 for every file in the approved bundle. Include every checkpoint state file referenced by the manifest, not just the final state.

## 5. Package and transfer

1. Agree which development recording(s) and model/scaler files are needed.
2. Exclude raw EDFs unless there is a separately approved research reason and license/provenance handling is agreed.
3. Exclude `.venv`, `__pycache__`, `.pytest_cache`, logs, temporary reports and unrelated caches.
4. Preserve relative directory structure.
5. Generate SHA-256 hashes after packaging.
6. Transfer through an approved private drive or storage location; do not publish private datasets or credentials in GitHub.
7. Share the bundle file list, total size, hash manifest and transfer location through the team's private channel.

Do not put large model/cache files into Git without reviewing repository size, licensing and artifact policy.

## 6. Receiver verification

On the teammate machine, verify the SHA-256 hash of every received file against the source manifest:

```powershell
Get-FileHash -Algorithm SHA256 ".\path\to\received\model.joblib"
```

Then:
- check checkpoint manifest completeness and referenced state files;
- load the scaler using the project's restore code and verify channel count, sample count, finite mean/scale and feature compatibility;
- load the model and verify expected feature count (170 for the robust candidate);
- verify the approved processed NPZ recording identity, required arrays, channel order and sampling rate;
- run cached-feature parity first, then raw processed-recording inference parity;
- record exact outputs and software versions.

Do not train models or alter thresholds to make parity pass. Investigate mismatches in preprocessing, normalization, feature order, artifact version and numerical tolerance.

## 7. What a teammate needs

- **Read docs / inspect source:** Git clone only.
- **Run unit tests:** Git clone + Python 3.11 dependencies; dataset/cache may not be needed for unit fixtures.
- **Reproduce cached-feature parity:** model + exact validation feature cache + reference scores + required metadata.
- **Reproduce scaler restore:** all scaler checkpoint state files + manifest + source restore code.
- **Complete raw processed EEG parity:** model + scaler + exact preprocessing/feature code + approved processed NPZ development recording + matching reference output.
- **Run OOF diagnostics:** complete relevant OOF caches and metadata; not a prerequisite for application integration.
- **Reprocess raw EDFs:** not part of the immediate handoff plan.

## 8. Transfer acceptance record

Fill this in only after the actual transfer:

| Field | Record |
|---|---|
| Bundle filename/version | Pending |
| Source machine/date | Pending |
| Receiving teammate/date | Pending |
| Included files and directories | Pending |
| Total bundle size | Pending |
| SHA-256 manifest location | Pending |
| Model artifact hash | Pending |
| Scaler manifest/state hashes | Pending |
| Approved development recording identity | Pending |
| Receiver verification result | Pending |
| Missing files / blockers | Pending |

Until this record is completed, artifact transfer is **not confirmed**.
