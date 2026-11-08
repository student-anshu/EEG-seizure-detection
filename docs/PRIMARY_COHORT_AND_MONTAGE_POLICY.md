# Primary CHB-MIT Cohort and Montage Policy

The primary cross-patient experiment uses a fixed 17-channel longitudinal
bipolar montage. Every included recording must contain all 17 channels in
the exact order below. No fallback montage, zero filling, duplication, or
silent channel substitution is allowed.

T8-P8 is omitted because the channel coverage audit found that exact
normalized label in only 28 of 686 recordings. Related labels such as
T8-P8-0 and T8-P8-1 are not assumed equivalent without validation.

## Canonical channel order

FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, P8-O2, FZ-CZ, CZ-PZ.

## Montage exclusions

The following recordings remain in the original dataset and full annotation
audit, but are excluded from the primary fixed-montage cohort:

- chb12/chb12_27.edf
- chb12/chb12_28.edf
- chb12/chb12_29.edf

Their channel configurations are not the required longitudinal bipolar
montage. Together they contain 13 annotated seizure events. We do not
reconstruct their channels because that would require additional assumptions
about reference and channel identity; label 01 is not silently renamed to O1.

This conservative exclusion is consistent with a 2024 peer-reviewed
CHB-MIT analysis that excluded these same three recordings because they
did not have longitudinal bipolar montages:
https://doi.org/10.3389/fneur.2024.1389731

The CHB12 summary documents recording-specific montage and seizure
annotations:
https://physionet.org/content/chbmit/1.0.0/chb12/chb12-summary.txt

## Cohort accounting

- Full audit: 686 recordings and 198 seizure events.
- Montage exclusions: 3 recordings and 13 seizure events.
- Primary cohort: 683 recordings and 185 seizure events.
- Independent patients: 23; CHB01 and CHB21 remain grouped.

These are audit counts, not processed-window counts or model performance.
The full annotation audit is retained unchanged.

## Implementation

- config.PREFERRED_CHANNELS defines the fixed channel order.
- main.choose_channels fails closed if any required channel is absent.
- data_management.cohort.filter_primary_cohort excludes only the three
  documented recordings before primary LOPO split generation.
- Original EDF files and the full annotation audit are never modified.
- Any future montage experiment must define and validate a separate cohort.
