from pathlib import Path
import numpy as np

from preprocessing.normalization import FoldChannelStandardizer

checkpoint = Path(
    "cache/robust_feature_first_fold_v1/scaler_checkpoint"
)

import json

manifest = json.loads(
    (checkpoint / "manifest.json").read_text(encoding="utf-8")
)

assert manifest["complete"] is True
assert manifest["completed_recordings"] == 469

state_path = checkpoint / manifest["state_file"]

with np.load(state_path, allow_pickle=False) as saved:
    state = {
        "count": int(saved["count"]),
        "n_channels": int(saved["n_channels"]),
        "mean": saved["mean"].copy(),
        "m2": saved["m2"].copy(),
    }

scaler = FoldChannelStandardizer()
scaler.load_state_dict(state)
scaler.finalize()

assert scaler.n_channels_ == 17
assert scaler.mean_.shape == (17,)
assert scaler.scale_.shape == (17,)
assert np.isfinite(scaler.mean_).all()
assert np.isfinite(scaler.scale_).all()
assert np.all(scaler.scale_ > 0)

print("SCALER RESTORATION: SUCCESS")
print("Channels:", scaler.n_channels_)
print("Sample count:", scaler.n_values_per_channel_)
print("Mean range:", scaler.mean_.min(), scaler.mean_.max())
print("Scale range:", scaler.scale_.min(), scaler.scale_.max())
print("All statistics finite:", np.isfinite(scaler.scale_).all())
