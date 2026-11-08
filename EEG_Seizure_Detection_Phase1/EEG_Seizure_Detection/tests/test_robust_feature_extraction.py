import numpy as np
import pytest

from feature_selection.robust_feature_extraction import (
    ROBUST_FEATURE_NAMES,
    extract_robust_features,
)


def test_robust_feature_shape_dtype_and_finiteness():
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(3, 2, 2560))
    result = extract_robust_features(windows, 256)
    assert result.shape == (3, 2 * len(ROBUST_FEATURE_NAMES))
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_relative_band_powers_are_amplitude_invariant():
    time = np.arange(2560) / 256
    signal = np.sin(2 * np.pi * 3 * time) + 0.4 * np.sin(2 * np.pi * 12 * time)
    original = extract_robust_features(signal[None, None, :], 256)
    scaled = extract_robust_features((7.5 * signal)[None, None, :], 256)
    # Feature positions 4:9 are relative delta/theta/alpha/beta/gamma powers.
    np.testing.assert_allclose(original[:, 4:9], scaled[:, 4:9], rtol=1e-5, atol=1e-6)


def test_constant_signal_has_finite_shape_features():
    result = extract_robust_features(np.zeros((1, 1, 2560)), 256)
    assert np.isfinite(result).all()
    assert result[0, 2] == 0.0
    assert result[0, 3] == 0.0


def test_rejects_invalid_sampling_rate():
    with pytest.raises(ValueError):
        extract_robust_features(np.ones((1, 1, 256)), 40)


def test_rejects_too_short_window_for_spectral_features():
    with pytest.raises(ValueError, match="at least 2 seconds"):
        extract_robust_features(np.ones((1, 1, 128)), 256)


def test_accepts_exactly_two_seconds_for_spectral_features():
    result = extract_robust_features(np.ones((1, 1, 512)), 256)
    assert result.shape == (1, len(ROBUST_FEATURE_NAMES))
    assert np.isfinite(result).all()


def test_chunked_extraction_matches_single_chunk():
    rng = np.random.default_rng(19)
    windows = rng.normal(size=(7, 3, 2560)).astype(np.float32)
    single = extract_robust_features(windows, 256, chunk_size=7)
    chunked = extract_robust_features(windows, 256, chunk_size=2)
    np.testing.assert_allclose(chunked, single, rtol=1e-6, atol=1e-6)


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        extract_robust_features(np.ones((1, 1, 512)), 256, chunk_size=0)


def test_rejects_non_finite_values_in_later_chunk():
    windows = np.ones((3, 1, 512), dtype=np.float32)
    windows[2, 0, 10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        extract_robust_features(windows, 256, chunk_size=2)
