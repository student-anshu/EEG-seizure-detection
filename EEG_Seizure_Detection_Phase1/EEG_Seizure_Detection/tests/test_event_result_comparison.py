import pytest

from scripts.compare_first_fold_event_results import compare_splits


def _recording(path, events, false_alarms):
    return {
        "relative_path": path,
        "patient_id": "patient_01",
        "events": events,
        "false_alarm_count": false_alarms,
        "false_alarms_per_hour": None,
    }


def _event(index, detected, start=100.0, end=120.0, delay=0.0):
    return {
        "event_index": index,
        "seizure_start_seconds": start,
        "seizure_end_seconds": end,
        "detected": detected,
        "detection_delay_seconds": delay if detected else None,
    }


def test_compare_splits_identifies_recovered_regressed_and_false_alarm_delta():
    legacy = {"recordings": [
        _recording("chb01/a.edf", [
            _event(1, False), _event(2, True, 200.0, 220.0)
        ], 2)
    ]}
    robust = {"recordings": [
        _recording("chb01/a.edf", [
            _event(1, True, delay=5.0), _event(2, False, 200.0, 220.0)
        ], 5)
    ]}

    result = compare_splits(legacy, robust, "outer_test")

    assert result["summary"]["recovered_events"] == 1
    assert result["summary"]["regressed_events"] == 1
    assert result["summary"]["legacy_detected_events"] == 1
    assert result["summary"]["robust_detected_events"] == 1
    assert result["summary"]["false_alarm_delta_robust_minus_legacy"] == 3
    assert [row["outcome"] for row in result["events"]] == [
        "recovered_by_robust", "detected_only_by_legacy"
    ]


def test_compare_splits_rejects_different_recording_sets():
    with pytest.raises(ValueError, match="recording sets differ"):
        compare_splits(
            {"recordings": [_recording("chb01/a.edf", [], 0)]},
            {"recordings": [_recording("chb02/b.edf", [], 0)]},
            "outer_test",
        )


def test_compare_splits_rejects_different_seizure_annotations():
    with pytest.raises(ValueError, match="annotations differ"):
        compare_splits(
            {"recordings": [_recording("chb01/a.edf", [_event(1, True)], 0)]},
            {"recordings": [_recording("chb01/a.edf", [_event(1, True, 101.0, 121.0)], 0)]},
            "outer_test",
        )
