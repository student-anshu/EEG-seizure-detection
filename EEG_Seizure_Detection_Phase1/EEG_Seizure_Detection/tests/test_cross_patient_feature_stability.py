import numpy as np
import pytest

from scripts.audit_cross_patient_feature_stability import (
    robust_patient_effect,
    summarize_vector,
)


def test_summarize_vector_returns_quantiles_and_count():
    result = summarize_vector(np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["window_count"] == 4
    assert result["median"] == 2.5
    assert result["q25"] == 1.75
    assert result["q75"] == 3.25
    assert result["iqr"] == 1.5


def test_summarize_vector_rejects_empty_or_nonfinite_input():
    with pytest.raises(ValueError):
        summarize_vector(np.array([], dtype=float))
    with pytest.raises(ValueError):
        summarize_vector(np.array([1.0, np.nan]))


def test_robust_patient_effect_uses_peer_patient_iqr():
    effect, center, spread = robust_patient_effect(8.0, np.array([1.0, 2.0, 3.0, 4.0]))
    assert center == 2.5
    assert spread == 1.5
    assert effect == pytest.approx((8.0 - 2.5) / 1.5)


def test_robust_patient_effect_does_not_divide_by_zero():
    effect, center, spread = robust_patient_effect(5.0, np.array([2.0, 2.0, 2.0]))
    assert effect is None
    assert center == 2.0
    assert spread == 0.0
