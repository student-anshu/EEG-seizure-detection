# Cross-Subject Epileptic Seizure Detection Using EEG Signals

An academic EEG seizure-detection project using the CHB-MIT Scalp EEG Database. The current implementation supports EDF inspection, signal preprocessing, seizure annotation parsing, window segmentation, channel matching, channel-ranking utilities, and processed dataset export.

> **Disclaimer:** This is an academic research prototype, not a medical device. It must not be used for diagnosis or treatment decisions.

## Repositories

- Active working repository: https://github.com/shivkumar15/EEG
- Original source repository: https://github.com/student-anshu/EEG-seizure-detection

## Project Documentation

1. [Project Master Plan](docs/PROJECT_MASTER_PLAN.md)
2. [AI / Teammate Handoff](docs/AI_HANDOFF.md)
3. [Development Log](docs/DEVELOPMENT_LOG.md)
4. [Upstream Change Log](docs/CHANGELOG_FOR_UPSTREAM.md)


## 1. Project Structure

The repository uses the following directory structure:

```text
EEG/
├── .venv311/                         # Python virtual environment
├── EEG_Seizure_Detection_Phase1/
│   ├── dataset/
│   │   └── chbmit/
│   │       ├── chb01/
│   │       ├── chb02/
│   │       └── ...
│   │
│   └── EEG_Seizure_Detection/
│       ├── config.py
│       ├── main.py
│       ├── requirements.txt
│       ├── preprocessing/
│       ├── feature_selection/
│       ├── tests/
│       └── processed/                # Generated locally
│
├── docs/                            # Project plans and development records
└── .gitignore
```

Keep the original CHB-MIT subject folder names and EDF filenames unchanged.

## 2. Requirements

- Windows 10/11
- Python 3.11 recommended
- VS Code or another Python-compatible IDE
- CHB-MIT Scalp EEG Database from PhysioNet:

https://physionet.org/content/chbmit/1.0.0/

Download and extract the dataset into:

```text
EEG_Seizure_Detection_Phase1/
    dataset/
        chbmit/
            chb01/
            chb02/
            chb03/
```

The dataset directory must be a sibling of the `EEG_Seizure_Detection` code directory.

## 3. Environment Setup (PowerShell)

Run these commands from the repository root (`EEG`).

Create the Python 3.11 environment:

```powershell
py -3.11 -m venv .venv311
```

Activate it:

```powershell
.\.venv311\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv311\Scripts\Activate.ps1
```

Move into the project code directory:

```powershell
cd .\EEG_Seizure_Detection_Phase1\EEG_Seizure_Detection
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Dataset Configuration

The dataset path is configured in `config.py`.

The default configuration expects:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT.parent / "dataset" / "chbmit"
```

If you use the recommended folder structure, no path modification is required.

## 5. Inspect the Dataset

From the `EEG_Seizure_Detection` directory, run:

```powershell
python main.py --inspect
```

This inspects the available EDF recordings and their channel information.

## 6. Process One EDF Recording

Always test one recording before processing the complete dataset.

```powershell
python main.py --edf ..\dataset\chbmit\chb01\chb01_03.edf
```

The program reads the corresponding subject summary file to identify seizure intervals.

Processed arrays and metadata are saved under:

```text
processed/
```

## 7. Process All Available EDF Recordings

After verifying single-recording preprocessing:

```powershell
python main.py --all
```

The program processes available EDF recordings and reports any files it cannot process.

Large datasets can consume substantial memory and disk space. Start with a small subset when working with a new dataset.

## 8. Run Tests

From the project code directory:

```powershell
python -m unittest discover -s tests -v
```

The tests cover annotation parsing, configuration paths, signal processing, segmentation, channel selection, and channel matching.

## 9. Current Preprocessing and Methodology

- Seizure annotations are read from the matching CHB-MIT subject summary file.
- Windows overlapping seizure intervals are assigned label `1`; other windows are assigned label `0`.
- Recordings without seizure annotations contribute non-seizure windows.
- The current implementation uses 10-second windows and a 5-second stride by default.
- Signal preprocessing includes bandpass and optional notch filtering.
- Channel matching and channel-ranking utilities are implemented.
- Channel ranking must be performed using training-fold data only when integrated into model training.
- The current normalization implementation is whole-recording and offline; its limitations must be considered in the final evaluation protocol.

**Important evaluation rule:** Do not randomly split overlapping EEG windows from the same patient between training and testing. Use patient-wise evaluation, such as Leave-One-Subject-Out (LOSO), to reduce data leakage.

## 10. Troubleshooting

**Error: `No module named mne`**

Activate the `.venv311` environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

**Error: EDF file not found**

Check the relative path and filename. The dataset folder must be alongside the code directory under `EEG_Seizure_Detection_Phase1`.

**Error: Subject summary file not found**

Ensure the corresponding `chbXX-summary.txt` file exists in the subject folder.

**Memory error**

Process one EDF at a time instead of running `--all`.

## 11. Planned Project Development

The project will be developed in the following stages:

1. Dataset expansion and preprocessing validation.
2. Patient-wise data splitting and leakage prevention.
3. Baseline machine-learning and deep-learning models.
4. Proposed CNN-BiLSTM-attention model and ablation experiments.
5. Model evaluation using sensitivity, specificity, precision, F1-score, AUROC, and AUPRC.
6. Explainability and interpretation.
7. FastAPI backend integration.
8. React frontend dashboard.
9. Final testing, documentation, and academic demonstration.

Everyone on the team should understand the complete data pipeline and evaluation methodology.