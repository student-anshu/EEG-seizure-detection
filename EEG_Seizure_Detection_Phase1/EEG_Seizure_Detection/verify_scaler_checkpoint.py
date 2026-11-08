import json
from pathlib import Path
import numpy as np

root = Path("cache")
expected_signature = "c07e4189d4c28ed1f97db4531da3dfac46c28fce344e0215d7b05f771224c3aa"

output = Path("scaler_checkpoint_verification.txt")
matches = []

for manifest_path in root.rglob("manifest.json"):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == expected_signature:
            matches.append((manifest_path, manifest))
    except Exception:
        continue

with output.open("w", encoding="utf-8") as f:
    f.write(f"Matching checkpoints: {len(matches)}\n")

    for manifest_path, manifest in matches:
        f.write(f"\nManifest: {manifest_path}\n")
        f.write(json.dumps(manifest, indent=2) + "\n")

        state_path = manifest_path.parent / manifest["state_file"]
        f.write(f"State file exists: {state_path.is_file()}\n")

        if state_path.is_file():
            with np.load(state_path, allow_pickle=False) as data:
                for key in data.files:
                    value = data[key]
                    f.write(
                        f"{key}: shape={value.shape}, dtype={value.dtype}\n"
                    )
                    if value.ndim == 1 and value.size:
                        f.write(
                            f"  finite={bool(np.isfinite(value).all())}, "
                            f"min={float(value.min()):.8g}, "
                            f"max={float(value.max()):.8g}\n"
                        )

print(f"Verification report created: {output}")
