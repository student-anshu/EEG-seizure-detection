# Upstream Change Log — Mirror to Original Repository

**Working copy:** https://github.com/shivkumar15/EEG  
**Original repository:** https://github.com/student-anshu/EEG-seizure-detection

This ledger tracks changes developed here that the team must also apply to the original repository.

## 2026-11-02 — Documentation baseline (working copy)

### Files added
- `README.md` — repository overview and navigation.
- `docs/PROJECT_MASTER_PLAN.md` — research architecture, roadmap, evaluation protocol and working rules.
- `docs/DEVELOPMENT_LOG.md` — verified repository baseline and next actions.
- `docs/AI_HANDOFF.md` — context for teammates and future AI sessions.
- `docs/CHANGELOG_FOR_UPSTREAM.md` — this upstream mirror ledger.

### Code/data changes
None. No source code, dataset or model artifact was changed in this documentation-only update.

### What to mirror upstream
Copy these five documentation paths/files to the original repository. There is no code patch for this update.

### Pending implementation changes (not yet applied)
- Fix CHB-MIT annotation parsing to support actual summary-file key variants and add tests.
- Fix dataset path resolution for the sibling `dataset` directory.
- Normalize import/package execution conventions.
- Add tests and validate preprocessing, channel handling and normalization.
- Correct `.gitignore` scope and decide raw EDF storage policy.

For each later implementation fix, append exact paths, behavior before/after, test command/results and an upstream-ready patch. Do not mark upstream changes complete until confirmed by the user or verified from an upstream commit.


## 2026-11-02 — Phase 0, Task 1: annotation parser correction

### Files changed in working copy
- Updated `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/preprocessing/annotations.py`.
- Added `EEG_Seizure_Detection_Phase1/EEG_Seizure_Detection/tests/test_annotations.py`.

### Behavior change
- Supports unnumbered CHB-MIT fields (`Seizure Start Time`, `Seizure End Time`) and numbered variants.
- Verifies annotation count against `Number of Seizures in File`.
- Returns `[]` for an explicitly zero-seizure recording.
- Raises clear errors for missing/mismatched annotations, invalid intervals, missing summary files and EDF files not listed in the summary.

### Tests
Six unit tests were added. Full test execution and direct CHB01/CHB03 integration verification remain pending; do not mark the parser fully validated until those commands pass.

### Mirror to original repository
Copy both changed files from this working repository to the exact same relative paths in `student-anshu/EEG-seizure-detection`. No other source files or dataset files need to be copied for this task. After copying, run the same unit tests in the original repository.
