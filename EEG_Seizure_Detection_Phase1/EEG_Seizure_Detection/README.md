# Cross-Subject Epileptic Seizure Detection Using EEG Signals

Phase 1 starter project: environment setup, CHB-MIT EDF loading, preprocessing,
window segmentation, channel selection, and processed dataset export.

> Academic prototype only. This software is not a medical device and must not
> be used for diagnosis or treatment decisions.

## 1. Requirements
- Windows 10/11, Python 3.10 or 3.11 (recommended)
- VS Code
- CHB-MIT Scalp EEG Database from PhysioNet:
  https://physionet.org/content/chbmit/1.0.0/

Download the dataset and extract it so the directory resembles:

```text
EEG_Seizure_Detection/
  dataset/
    chbmit/
      chb01/
        chb01_01.edf
        chb01-summary.txt
      chb02/
        ...
```

The CHB-MIT download may contain multiple subject folders. Keep the original
folder and file names.

## 2. Create and activate the virtual environment (PowerShell)

```powershell
cd path\to\EEG_Seizure_Detection
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Configure paths
Open `config.py`. The default dataset path is `dataset/chbmit`.
Change it if you extracted the dataset somewhere else.

## 4. Run the data inspection
```powershell
python main.py --inspect
```

## 5. Process one EDF file first
```powershell
python main.py --edf dataset/chbmit/chb01/chb01_03.edf
```

This writes windowed arrays and metadata under `processed/`. The code reads
the matching `chb01-summary.txt` to identify seizure intervals.

## 6. Process all EDF files (after the single-file test works)
```powershell
python main.py --all
```

For a first run, process a small subset rather than the entire dataset. Large
EEG datasets can consume substantial RAM and disk space.

## 7. Important methodology notes
- Windows are labeled using seizure start/end annotations from the matching
  subject summary file.
- Windows overlapping a seizure interval are labeled `1`; other windows are
  labeled `0`.
- Files with no seizure annotations contribute non-seizure windows.
- This initial version uses a fixed common channel list where available. Phase 2
  will implement training-only automatic channel selection and the model.
- Do not randomly split EEG windows from the same patient between train and
  test. The final evaluation must use patient-wise splits / Leave-One-Subject-Out.
- The first implementation uses 10-second windows and a 5-second stride.
  These are configurable in `config.py`.

## 8. Troubleshooting
- `No module named mne`: activate `.venv`, then run `pip install -r requirements.txt`.
- EDF not found: verify the path and spelling.
- No summary file found: make sure the matching `chbXX-summary.txt` is in the
  subject folder.
- No seizure annotation found: that EDF may be a non-seizure file. The loader
  still creates non-seizure windows if its summary exists.
- Memory error: use one EDF at a time; do not run `--all` initially.

## 9. Team division (suggested for 4+ members)
1. Data and signal processing: dataset, EDF inspection, filtering, segmentation.
2. Feature/channel selection: channel harmonization, Mutual Information selection.
3. Deep learning and evaluation: CNN-BiLSTM-attention, LOSO, metrics.
4. Explainability and interface: SHAP, Streamlit, integration, testing.
Everyone should understand the full pipeline and evaluation design.
