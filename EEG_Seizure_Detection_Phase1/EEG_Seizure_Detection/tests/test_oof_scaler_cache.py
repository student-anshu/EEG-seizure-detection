import numpy as np
import pytest

from evaluation.oof_scaler_cache import (
    channel_sufficient_statistics,
    finalize_channel_statistics,
    merge_channel_statistics,
    recompute_log_std_features,
)


def test_merged_recording_statistics_match_concatenated_population_stats():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(3, 2, 128))
    b = rng.normal(loc=2.0, scale=3.0, size=(5, 2, 128))
    merged = merge_channel_statistics(
        channel_sufficient_statistics(a), channel_sufficient_statistics(b)
    )
    mean, scale = finalize_channel_statistics(merged)
    direct = np.concatenate([a, b], axis=0).transpose(1, 0, 2).reshape(2, -1)
    np.testing.assert_allclose(mean, direct.mean(axis=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scale, direct.std(axis=1), rtol=1e-12, atol=1e-12)



def test_chunked_recording_statistics_match_one_batch():
    rng = np.random.default_rng(44)
    values = rng.normal(size=(9, 3, 128))
    chunked = channel_sufficient_statistics(values, chunk_size=2)
    direct = channel_sufficient_statistics(values, chunk_size=9)
    np.testing.assert_allclose(chunked["mean"], direct["mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chunked["m2"], direct["m2"], rtol=1e-12, atol=1e-10)

def test_statistics_merge_is_order_independent_within_tolerance():
    rng = np.random.default_rng(3)
    batches = [rng.normal(size=(n, 2, 64)) for n in (2, 4, 3)]
    states = [channel_sufficient_statistics(batch) for batch in batches]
    left = merge_channel_statistics(merge_channel_statistics(states[0], states[1]), states[2])
    right = merge_channel_statistics(states[0], merge_channel_statistics(states[1], states[2]))
    np.testing.assert_allclose(left["mean"], right["mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(left["m2"], right["m2"], rtol=1e-12, atol=1e-10)


def test_recomputed_log_std_matches_robust_extractor_and_preserves_clipped_windows():
    from feature_selection.robust_feature_extraction import extract_robust_features
    rng = np.random.default_rng(4)
    windows = rng.normal(size=(4, 2, 512)).astype(np.float32)
    windows[0] = 0.0
    mean = np.array([0.1, -0.2])
    scale = np.array([2.0, 0.5])
    standardized = ((windows.astype(np.float64) - mean[None, :, None])
                    / scale[None, :, None]).astype(np.float32)
    expected = extract_robust_features(standardized, 256)[:, 0::10]
    actual = recompute_log_std_features(windows, mean, scale, chunk_size=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.isfinite(actual).all()
    assert np.all(actual[0] <= np.log(1e-12))


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.ones((0, 2, 8))])
def test_statistics_reject_invalid_shapes(bad):
    with pytest.raises(ValueError):
        channel_sufficient_statistics(bad)


def test_statistics_reject_non_finite_values():
    values = np.ones((2, 1, 8))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        channel_sufficient_statistics(values)


def test_recompute_rejects_invalid_scaler():
    with pytest.raises(ValueError, match="positive"):
        recompute_log_std_features(np.ones((1, 2, 8)), np.zeros(2), np.zeros(2))


def test_cached_raw_std_reconstructs_fold_log_std_with_small_float32_error():
    from evaluation.oof_scaler_cache import (
        log_std_from_raw_standard_deviations,
        window_channel_standard_deviations,
    )
    from feature_selection.robust_feature_extraction import extract_robust_features
    rng = np.random.default_rng(912)
    windows = rng.normal(size=(7, 3, 512)).astype(np.float32)
    windows[0, 0] = 0.0
    mean = np.array([0.2, -0.1, 0.4])
    scale = np.array([0.5, 2.0, 1.5])
    raw_std = window_channel_standard_deviations(windows, chunk_size=2)
    actual = log_std_from_raw_standard_deviations(raw_std, scale)
    standardized = ((windows.astype(np.float64) - mean[None, :, None])
                    / scale[None, :, None]).astype(np.float32)
    expected = extract_robust_features(standardized, 128)[:, 0::10]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert np.isfinite(actual).all()


def test_raw_std_cache_rejects_non_finite_and_invalid_scale():
    from evaluation.oof_scaler_cache import (
        log_std_from_raw_standard_deviations,
        window_channel_standard_deviations,
    )
    windows = np.ones((2, 2, 128), dtype=np.float32)
    windows[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        window_channel_standard_deviations(windows)
    with pytest.raises(ValueError, match="positive"):
        log_std_from_raw_standard_deviations(np.ones((2, 2)), np.array([1.0, 0.0]))


def test_patient_grouped_scaler_omits_validation_patient():
    from evaluation.oof_scaler_cache import patient_grouped_channel_scale
    a = channel_sufficient_statistics(np.zeros((2, 1, 8)))
    b = channel_sufficient_statistics(np.full((3, 1, 8), 10.0))
    c = channel_sufficient_statistics(np.full((5, 1, 8), 1000.0))
    mean, scale = patient_grouped_channel_scale({"a": a, "b": b, "c": c}, "c")
    assert mean[0] == pytest.approx(6.0)
    assert scale[0] == pytest.approx(np.sqrt(24.0))



def test_exact_fallback_mask_flags_only_low_normalized_variation_windows():
    from evaluation.oof_scaler_cache import exact_log_std_fallback_mask

    raw_std = np.array([[1e-15, 2e-15], [1e-3, 2e-3], [1e-8, 2e-3]])
    scale = np.array([1e-4, 1e-4])
    np.testing.assert_array_equal(
        exact_log_std_fallback_mask(raw_std, scale),
        np.array([True, False, True]),
    )


def test_exact_fallback_reproduces_original_float32_extractor_for_near_constant_windows():
    from evaluation.oof_scaler_cache import (
        exact_log_std_fallback_mask,
        recompute_log_std_features,
        window_channel_standard_deviations,
    )
    from feature_selection.robust_feature_extraction import extract_robust_features

    # Tiny raw variation around zero becomes a small nonzero normalized level
    # after applying a fold mean. Float32 rounding changes the resulting std.
    ramp = np.linspace(-2e-15, 2e-15, 256, dtype=np.float64)
    windows = np.stack([ramp, ramp * 0.75])[None, :, :]
    mean = np.array([-5.109097766828098e-10, -2.9123343466072157e-10])
    scale = np.array([5.8258332560013724e-5, 5.46497477406169e-5])

    raw_std = window_channel_standard_deviations(windows)
    mask = exact_log_std_fallback_mask(raw_std, scale)
    assert mask.tolist() == [True]

    exact = recompute_log_std_features(windows[mask], mean, scale)
    standardized = ((windows[mask] - mean[None, :, None])
                    / scale[None, :, None]).astype(np.float32)
    expected = extract_robust_features(standardized, 128)[:, 0::10]
    np.testing.assert_array_equal(exact, expected)


@pytest.mark.parametrize(
    "raw_std,scale",
    [
        (np.array([[np.nan]]), np.array([1.0])),
        (np.array([[-1.0]]), np.array([1.0])),
        (np.array([[1.0]]), np.array([0.0])),
    ],
)
def test_exact_fallback_mask_rejects_invalid_inputs(raw_std, scale):
    from evaluation.oof_scaler_cache import exact_log_std_fallback_mask
    with pytest.raises(ValueError):
        exact_log_std_fallback_mask(raw_std, scale)
