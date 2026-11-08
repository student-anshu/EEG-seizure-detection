import numpy as np
import pytest

from scripts.audit_fold_aware_feature_stability import _quantiles, _collect


def test_quantiles_returns_count_and_iqr():
    result = _quantiles(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert result["window_count"] == 4
    assert result["median"] == 2.5
    assert result["iqr"] == 1.5


def test_quantiles_rejects_empty_and_nonfinite_values():
    with pytest.raises(ValueError):
        _quantiles(np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        _quantiles(np.array([1.0, np.inf], dtype=np.float32))


def test_collect_respects_patient_record_offsets_and_class_labels():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    records = [
        {"start": 0, "end": 3},
        {"start": 3, "end": 6},
    ]
    values, recording_count = _collect(X, y, records, feature_index=1, label=1)
    np.testing.assert_array_equal(values, np.array([3, 7, 9], dtype=np.float32))
    assert recording_count == 2


def test_collect_returns_empty_for_absent_class():
    X = np.ones((3, 2), dtype=np.float32)
    y = np.zeros(3, dtype=np.int8)
    values, recording_count = _collect(
        X, y, [{"start": 0, "end": 3}], feature_index=0, label=1
    )
    assert values.size == 0
    assert recording_count == 0
