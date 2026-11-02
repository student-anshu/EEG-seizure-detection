
from pathlib import Path
from preprocessing.annotations import read_seizure_intervals

dataset = Path("../dataset/chbmit")

recordings = [
    ("chb01", "chb01_03.edf"),
    ("chb01", "chb01_05.edf"),
    ("chb03", "chb03_01.edf"),
    ("chb03", "chb03_05.edf"),
]

for subject, recording in recordings:
    edf_path = dataset / subject / recording
    intervals = read_seizure_intervals(edf_path)

    print(f"{subject} | {recording} | {intervals}")
