"""Explore duration filtering and short-gap alarm episode grouping on saved robust inner-validation alarms.

This is interval-level post-processing, not model retraining. Event detections
are based only on retained model-positive intervals. Gap grouping reduces
episode counts but never creates seizure overlap or changes detection delay.
Run from the EEG project directory:
    python -m scripts.simulate_validation_alarm_policies
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

DEFAULT_INPUT = Path("results/baselines/robust_feature_first_fold_event_metrics_v1.json")
DEFAULT_OUTPUT = Path("results/baselines/robust_validation_alarm_policy_simulation_v2.json")
MIN_DURATIONS = (0.0, 15.0, 20.0, 30.0)
MERGE_GAPS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)


def _interval(start: object, end: object, context: str) -> tuple[float, float]:
    start_value, end_value = float(start), float(end)
    if (
        not math.isfinite(start_value) or not math.isfinite(end_value)
        or start_value < 0 or end_value <= start_value
    ):
        raise ValueError(f"Invalid interval in {context}: {start_value}, {end_value}")
    return start_value, end_value


def merge_intervals(
    intervals: list[tuple[float, float]], gap_seconds: float
) -> list[tuple[float, float]]:
    """Group intervals separated by <= gap_seconds; returned envelopes are not evidence."""
    if not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("gap_seconds must be finite and non-negative")
    ordered = sorted(intervals)
    if not ordered:
        return []
    groups: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= groups[-1][1] + gap_seconds + 1e-9:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return [(start, end) for start, end in groups]


def _overlaps(alarm: tuple[float, float], event: tuple[float, float]) -> bool:
    return alarm[0] < event[1] and alarm[1] > event[0]


def _f2(precision: float, recall: float) -> float:
    denominator = 4 * precision + recall
    return 5 * precision * recall / denominator if denominator else 0.0


def _evaluate_recording(record: dict, min_duration: float, merge_gap: float) -> dict:
    path = record["relative_path"]
    raw_alarms = [
        _interval(a["start"], a["end"], f"alarm {path}")
        for a in record.get("alarms", [])
    ]
    events = [
        _interval(e["seizure_start_seconds"], e["seizure_end_seconds"], f"event {path}")
        for e in record.get("events", [])
    ]

    # Persistence filter is applied to actual positive runs BEFORE gap grouping.
    retained = [
        alarm for alarm in raw_alarms
        if alarm[1] - alarm[0] + 1e-9 >= min_duration
    ]

    # Event detection and delay use only actual retained model-positive intervals.
    delays: list[float] = []
    for event in events:
        matching = [alarm for alarm in retained if _overlaps(alarm, event)]
        if matching:
            first_start = min(alarm[0] for alarm in matching)
            delays.append(max(0.0, first_start - event[0]))

    # Gap grouping changes alarm episode count, not the evidence intervals above.
    groups = merge_intervals(retained, merge_gap)
    false_alarm_count = 0
    for group in groups:
        member_alarms = [
            alarm for alarm in retained
            if alarm[0] >= group[0] - 1e-9 and alarm[1] <= group[1] + 1e-9
        ]
        # An episode is non-false only if actual retained evidence overlaps an event.
        has_event_evidence = any(
            _overlaps(alarm, event)
            for alarm in member_alarms for event in events
        )
        if not has_event_evidence:
            false_alarm_count += 1

    exposure = record.get("non_seizure_duration_seconds")
    if exposure is None:
        raise ValueError(f"Missing exact non-seizure exposure for {path}")
    exposure = float(exposure)
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"Invalid non-seizure exposure for {path}")

    return {
        "relative_path": path,
        "patient_id": record["patient_id"],
        "event_count": len(events),
        "detected_events": len(delays),
        "missed_events": len(events) - len(delays),
        "detection_delays_seconds": delays,
        "alarm_count": len(groups),
        "false_alarm_count": false_alarm_count,
        "non_seizure_duration_seconds": exposure,
    }


def _summarize(records: list[dict]) -> dict:
    event_count = sum(r["event_count"] for r in records)
    detected = sum(r["detected_events"] for r in records)
    false_alarms = sum(r["false_alarm_count"] for r in records)
    exposure = sum(r["non_seizure_duration_seconds"] for r in records)
    delays = [d for r in records for d in r["detection_delays_seconds"]]
    return {
        "recording_count": len(records),
        "event_count": event_count,
        "detected_events": detected,
        "missed_events": event_count - detected,
        "event_sensitivity": detected / event_count if event_count else None,
        "mean_detection_delay_seconds": sum(delays) / len(delays) if delays else None,
        "false_alarm_count": false_alarms,
        "non_seizure_exposure_hours": exposure / 3600.0,
        "false_alarms_per_non_seizure_hour": false_alarms / (exposure / 3600.0),
    }


def simulate(input_path: Path = DEFAULT_INPUT) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        raise ValueError("Input robust event evaluation is not completed")
    validation = data.get("validation")
    if not isinstance(validation, dict) or not isinstance(validation.get("recordings"), list):
        raise ValueError("Input does not contain validation recordings")

    source_records = validation["recordings"]
    paths = [r.get("relative_path") for r in source_records]
    if not paths or any(not p for p in paths) or len(paths) != len(set(paths)):
        raise ValueError("Validation recording paths are missing or duplicated")
    expected = validation.get("aggregate", {}).get("recording_count")
    if expected is not None and expected != len(source_records):
        raise ValueError("Validation recording count does not match aggregate")

    simulations = []
    for min_duration in MIN_DURATIONS:
        for merge_gap in MERGE_GAPS:
            per_recording = [
                _evaluate_recording(r, min_duration, merge_gap)
                for r in source_records
            ]
            pooled = _summarize(per_recording)
            by_patient: dict[str, list[dict]] = defaultdict(list)
            for row in per_recording:
                by_patient[row["patient_id"]].append(row)
            patient_summaries = {
                patient: _summarize(rows) for patient, rows in sorted(by_patient.items())
            }
            seizure_bearing = [
                item for item in patient_summaries.values() if item["event_count"] > 0
            ]
            macro_sensitivity = (
                sum(x["event_sensitivity"] for x in seizure_bearing) / len(seizure_bearing)
                if seizure_bearing else None
            )
            macro_f2 = []
            for item in seizure_bearing:
                denominator = item["detected_events"] + item["false_alarm_count"]
                precision_proxy = item["detected_events"] / denominator if denominator else 0.0
                macro_f2.append(_f2(precision_proxy, item["event_sensitivity"]))

            simulations.append({
                "min_alarm_duration_seconds": min_duration,
                "merge_gap_seconds": merge_gap,
                "pooled": pooled,
                "seizure_bearing_patient_count": len(seizure_bearing),
                "patient_macro_event_sensitivity": macro_sensitivity,
                "patient_macro_event_f2_proxy": sum(macro_f2) / len(macro_f2) if macro_f2 else None,
                "patients": patient_summaries,
            })

    return {
        "status": "completed",
        "experiment": "robust_validation_alarm_policy_interval_simulation",
        "version": "duration_filter_gap_grouping_v2_evidence_preserving",
        "scope": "inner_validation_only",
        "source_result": str(input_path),
        "model_retrained": False,
        "features_reextracted": False,
        "thresholds_changed": False,
        "outer_test_read": False,
        "policy_grid": {
            "minimum_alarm_duration_seconds": list(MIN_DURATIONS),
            "merge_gap_seconds": list(MERGE_GAPS),
            "combination_count": len(simulations),
            "order": "filter each actual alarm by duration first, then group retained alarms by gap",
        },
        "metric_notes": [
            "Seizure detection and delay use only actual retained model-positive intervals; gap envelopes cannot create seizure detections.",
            "Minimum duration is applied to each actual alarm run before gap grouping; silent gaps do not count as positive evidence.",
            "Gap grouping changes alarm episode counts only.",
            "A grouped episode is non-false only when at least one actual retained alarm in that group overlaps an annotated seizure.",
            "False alarms/hour uses saved exact non-seizure exposure.",
            "Patient macro metrics average seizure-bearing validation patients.",
            "Event F2 proxy is exploratory, not clinical utility.",
            "No policy is automatically selected.",
        ],
        "baseline_source_metrics": validation.get("aggregate"),
        "simulations": simulations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = simulate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "model_retrained": False,
        "outer_test_read": False,
        "combination_count": result["policy_grid"]["combination_count"],
        "output": str(args.output),
    }, indent=2))
    print("\nEvidence-preserving policy results (validation only):")
    for item in result["simulations"]:
        p = item["pooled"]
        print(
            f"min={item['min_alarm_duration_seconds']:>4.0f}s "
            f"gap={item['merge_gap_seconds']:>4.0f}s | "
            f"events={p['detected_events']}/{p['event_count']} "
            f"sens={p['event_sensitivity']:.3f} "
            f"FA={p['false_alarm_count']} "
            f"FA/h={p['false_alarms_per_non_seizure_hour']:.3f} "
            f"delay={p['mean_detection_delay_seconds']}"
        )


if __name__ == "__main__":
    main()
