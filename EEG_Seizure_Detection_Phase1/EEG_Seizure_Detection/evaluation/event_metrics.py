"""Event-level seizure detection metrics and alarm post-processing.

Exact recording duration is optional for event matching but mandatory for
exposure-based false-alarms/hour. Threshold selection belongs to inner
validation; this module never chooses a threshold from evaluation labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True, order=True)
class TimeInterval:
    """Half-open interval in seconds: [start, end)."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError("Interval boundaries must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Interval must satisfy 0 <= start < end")


def parse_seizure_intervals(values: Iterable[str]) -> tuple[TimeInterval, ...]:
    """Parse audit-manifest entries such as '2996-3036' into seconds."""
    intervals: list[TimeInterval] = []
    for index, value in enumerate(values, start=1):
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*[-,]\s*(\d+(?:\.\d+)?)\s*",
            str(value),
        )
        if not match:
            raise ValueError(f"Invalid seizure interval #{index}: {value!r}")
        try:
            intervals.append(TimeInterval(float(match.group(1)), float(match.group(2))))
        except ValueError as exc:
            raise ValueError(f"Invalid seizure interval #{index}: {value!r}") from exc
    return tuple(sorted(intervals))


def _validate_intervals(
    intervals: Sequence[TimeInterval | tuple[float, float]],
    recording_duration_seconds: float | None,
    name: str,
) -> tuple[TimeInterval, ...]:
    duration = None
    if recording_duration_seconds is not None:
        duration = float(recording_duration_seconds)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Recording duration must be finite and positive")
    result = []
    for index, item in enumerate(intervals, start=1):
        try:
            interval = item if isinstance(item, TimeInterval) else TimeInterval(
                float(item[0]), float(item[1])
            )
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"Invalid {name} interval #{index}: {item!r}") from exc
        if duration is not None and interval.end > duration + 1e-9:
            raise ValueError(
                f"{name} interval #{index} ends after recording duration "
                f"({interval.end} > {duration} seconds)"
            )
        result.append(interval)
    return tuple(sorted(result))


def windows_to_alarms(
    window_starts_seconds: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    *,
    recording_duration_seconds: float | None = None,
    window_seconds: float = 10.0,
    merge_gap_seconds: float = 0.0,
) -> tuple[TimeInterval, ...]:
    """Convert positive windows to merged alarm intervals.

    Overlapping or touching positive windows are merged. A positive
    merge_gap_seconds additionally joins alarms separated by a short gap.
    """
    starts = np.asarray(window_starts_seconds, dtype=np.float64)
    predicted = np.asarray(predictions)
    duration = (
        None if recording_duration_seconds is None
        else float(recording_duration_seconds)
    )
    window = float(window_seconds)
    gap = float(merge_gap_seconds)

    if starts.ndim != 1 or predicted.ndim != 1 or starts.shape != predicted.shape:
        raise ValueError("Window starts and predictions must be aligned 1D arrays")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("Recording duration must be finite and positive")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Window duration must be finite and positive")
    if not np.isfinite(gap) or gap < 0:
        raise ValueError("Alarm merge gap must be finite and non-negative")
    if not np.isfinite(starts).all() or np.any(starts < 0):
        raise ValueError("Window starts must be finite and non-negative")
    if len(starts) > 1 and np.any(np.diff(starts) <= 0):
        raise ValueError("Window starts must be strictly increasing")
    if not np.isin(predicted, (0, 1, False, True)).all():
        raise ValueError("Window predictions must be binary (0 or 1)")
    if duration is not None and len(starts) and np.any(
        starts + window > duration + 1e-9
    ):
        raise ValueError("A window extends beyond the supplied recording duration")

    positive_intervals = [
        TimeInterval(float(start), float(start + window))
        for start, label in zip(starts, predicted)
        if int(label) == 1
    ]
    if not positive_intervals:
        return ()

    merged: list[TimeInterval] = []
    current_start = positive_intervals[0].start
    current_end = positive_intervals[0].end
    for interval in positive_intervals[1:]:
        if interval.start <= current_end + gap + 1e-9:
            current_end = max(current_end, interval.end)
        else:
            merged.append(TimeInterval(current_start, current_end))
            current_start, current_end = interval.start, interval.end
    merged.append(TimeInterval(current_start, current_end))
    return tuple(merged)


def _overlaps(left: TimeInterval, right: TimeInterval) -> bool:
    return left.start < right.end and left.end > right.start


def _union_duration(intervals: Sequence[TimeInterval]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0].start, ordered[0].end
    for item in ordered[1:]:
        if item.start <= end:
            end = max(end, item.end)
        else:
            total += end - start
            start, end = item.start, item.end
    return total + end - start


def evaluate_recording_events(
    window_starts_seconds: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    seizure_intervals: Sequence[TimeInterval | tuple[float, float]],
    *,
    recording_duration_seconds: float | None = None,
    window_seconds: float = 10.0,
    merge_gap_seconds: float = 0.0,
) -> dict:
    """Evaluate seizure-event detection for one recording.

    An annotated event is detected when at least one alarm overlaps it.
    Detection delay is max(0, first_overlapping_alarm.start - event.start).
    An alarm is a false alarm only when it overlaps no annotated seizure.
    Exact recording duration is required only to calculate non-seizure
    exposure and false alarms/hour. Without it those fields are null.
    """
    duration = (
        None if recording_duration_seconds is None
        else float(recording_duration_seconds)
    )
    seizures = _validate_intervals(
        seizure_intervals, duration, "Seizure"
    )
    alarms = windows_to_alarms(
        window_starts_seconds,
        predictions,
        recording_duration_seconds=duration,
        window_seconds=window_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )

    detected_delays: list[float] = []
    missed_events = 0
    event_details: list[dict] = []
    for event_index, event in enumerate(seizures, start=1):
        matching = [alarm for alarm in alarms if _overlaps(alarm, event)]
        if not matching:
            missed_events += 1
            event_details.append({
                "event_index": event_index,
                "seizure_start_seconds": float(event.start),
                "seizure_end_seconds": float(event.end),
                "detected": False,
                "first_alarm_start_seconds": None,
                "detection_delay_seconds": None,
                "matching_alarm_count": 0,
            })
            continue
        first_alarm = min(matching, key=lambda alarm: alarm.start)
        delay = max(0.0, first_alarm.start - event.start)
        detected_delays.append(delay)
        event_details.append({
            "event_index": event_index,
            "seizure_start_seconds": float(event.start),
            "seizure_end_seconds": float(event.end),
            "detected": True,
            "first_alarm_start_seconds": float(first_alarm.start),
            "detection_delay_seconds": float(delay),
            "matching_alarm_count": len(matching),
        })

    false_alarm_count = sum(
        not any(_overlaps(alarm, event) for event in seizures)
        for alarm in alarms
    )
    non_seizure_seconds = (
        max(0.0, duration - _union_duration(seizures))
        if duration is not None else None
    )
    event_count = len(seizures)
    detected_count = event_count - missed_events
    return {
        "event_count": event_count,
        "detected_events": detected_count,
        "missed_events": missed_events,
        "event_sensitivity": (
            float(detected_count / event_count) if event_count else None
        ),
        "mean_detection_delay_seconds": (
            float(np.mean(detected_delays)) if detected_delays else None
        ),
        "median_detection_delay_seconds": (
            float(np.median(detected_delays)) if detected_delays else None
        ),
        "alarm_count": len(alarms),
        "false_alarm_count": int(false_alarm_count),
        "non_seizure_duration_seconds": (
            float(non_seizure_seconds)
            if non_seizure_seconds is not None else None
        ),
        "false_alarms_per_hour": (
            float(false_alarm_count / (non_seizure_seconds / 3600.0))
            if non_seizure_seconds is not None and non_seizure_seconds > 0
            else None
        ),
        "events": event_details,
        "alarms": [asdict(alarm) for alarm in alarms],
    }


def aggregate_event_metrics(recording_results: Sequence[dict]) -> dict:
    """Pool event counts and exposure time across recordings.

    Patient-macro summaries should be computed separately from per-patient
    results; this function intentionally returns pooled cohort metrics only.
    """
    if not recording_results:
        raise ValueError("At least one recording result is required")
    event_count = sum(int(item["event_count"]) for item in recording_results)
    detected = sum(int(item["detected_events"]) for item in recording_results)
    missed = sum(int(item["missed_events"]) for item in recording_results)
    false_alarms = sum(int(item["false_alarm_count"]) for item in recording_results)
    exposure_values = [
        item["non_seizure_duration_seconds"] for item in recording_results
    ]
    exposure_complete = all(value is not None for value in exposure_values)
    non_seizure_seconds = (
        sum(float(value) for value in exposure_values)
        if exposure_complete else None
    )
    delays = [
        float(item["mean_detection_delay_seconds"])
        for item in recording_results
        if item["mean_detection_delay_seconds"] is not None
    ]
    return {
        "recording_count": len(recording_results),
        "event_count": event_count,
        "detected_events": detected,
        "missed_events": missed,
        "event_sensitivity": detected / event_count if event_count else None,
        "mean_detection_delay_seconds": (
            float(np.average(
                delays,
                weights=[
                    int(item["detected_events"])
                    for item in recording_results
                    if item["mean_detection_delay_seconds"] is not None
                ],
            )) if delays else None
        ),
        "false_alarm_count": false_alarms,
        "non_seizure_duration_seconds": non_seizure_seconds,
        "false_alarms_per_hour": (
            false_alarms / (non_seizure_seconds / 3600.0)
            if non_seizure_seconds is not None and non_seizure_seconds > 0
            else None
        ),
    }
