"""Compare legacy and robust first-fold event-result JSON files.

This is descriptive error analysis only. It does not select thresholds,
change alarm rules, train models, or authorize tuning against outer-test data.
Run from the EEG project directory:
    python -m scripts.compare_first_fold_event_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _recording_map(split: dict, label: str) -> dict[str, dict]:
    rows = split.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} result has no recording-level rows")
    result = {}
    for row in rows:
        key = row.get("relative_path")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} contains a recording without relative_path")
        if key in result:
            raise ValueError(f"{label} contains duplicate recording: {key}")
        result[key] = row
    return result


def compare_splits(legacy_split: dict, robust_split: dict, label: str) -> dict:
    legacy = _recording_map(legacy_split, f"legacy {label}")
    robust = _recording_map(robust_split, f"robust {label}")
    if set(legacy) != set(robust):
        missing_robust = sorted(set(legacy) - set(robust))
        missing_legacy = sorted(set(robust) - set(legacy))
        raise ValueError(
            f"{label} recording sets differ; missing robust={missing_robust[:5]}, "
            f"missing legacy={missing_legacy[:5]}"
        )

    event_rows = []
    recording_rows = []
    totals = {
        "recording_count": len(legacy),
        "legacy_detected_events": 0,
        "robust_detected_events": 0,
        "recovered_events": 0,
        "regressed_events": 0,
        "both_detected_events": 0,
        "both_missed_events": 0,
        "legacy_false_alarm_count": 0,
        "robust_false_alarm_count": 0,
    }
    for path in sorted(legacy):
        old, new = legacy[path], robust[path]
        def event_map(row, model):
            events = row.get("events")
            if not isinstance(events, list):
                raise ValueError(f"{model} event details missing for {path}")
            mapped = {}
            for event in events:
                key = (event.get("event_index"), event.get("seizure_start_seconds"),
                       event.get("seizure_end_seconds"))
                if key in mapped:
                    raise ValueError(f"Duplicate {model} event identity in {path}: {key}")
                mapped[key] = event
            return mapped

        old_events = event_map(old, "legacy")
        new_events = event_map(new, "robust")
        if set(old_events) != set(new_events):
            raise ValueError(f"Legacy/robust seizure annotations differ for {path}")
        old_detected = sum(bool(e.get("detected")) for e in old_events.values())
        new_detected = sum(bool(e.get("detected")) for e in new_events.values())
        old_fa = int(old["false_alarm_count"])
        new_fa = int(new["false_alarm_count"])
        recording_rows.append({
            "relative_path": path,
            "patient_id": old.get("patient_id"),
            "event_count": len(old_events),
            "legacy_detected_events": old_detected,
            "robust_detected_events": new_detected,
            "legacy_false_alarm_count": old_fa,
            "robust_false_alarm_count": new_fa,
            "false_alarm_delta_robust_minus_legacy": new_fa - old_fa,
            "legacy_false_alarms_per_hour": old.get("false_alarms_per_hour"),
            "robust_false_alarms_per_hour": new.get("false_alarms_per_hour"),
        })
        for key in sorted(old_events):
            a, b = old_events[key], new_events[key]
            old_hit, new_hit = bool(a.get("detected")), bool(b.get("detected"))
            if not old_hit and new_hit:
                outcome = "recovered_by_robust"
            elif old_hit and not new_hit:
                outcome = "detected_only_by_legacy"
            elif old_hit:
                outcome = "detected_by_both"
            else:
                outcome = "missed_by_both"
            event_rows.append({
                "relative_path": path,
                "patient_id": old.get("patient_id"),
                "event_index": key[0],
                "seizure_start_seconds": key[1],
                "seizure_end_seconds": key[2],
                "legacy_detected": old_hit,
                "robust_detected": new_hit,
                "outcome": outcome,
                "legacy_detection_delay_seconds": a.get("detection_delay_seconds"),
                "robust_detection_delay_seconds": b.get("detection_delay_seconds"),
            })
            totals[{
                "recovered_by_robust": "recovered_events",
                "detected_only_by_legacy": "regressed_events",
                "detected_by_both": "both_detected_events",
                "missed_by_both": "both_missed_events",
            }[outcome]] += 1
        totals["legacy_detected_events"] += old_detected
        totals["robust_detected_events"] += new_detected
        totals["legacy_false_alarm_count"] += old_fa
        totals["robust_false_alarm_count"] += new_fa

    totals["false_alarm_delta_robust_minus_legacy"] = (
        totals["robust_false_alarm_count"] - totals["legacy_false_alarm_count"]
    )
    return {"summary": totals, "events": event_rows, "recordings": recording_rows}


def compare_results(legacy_path: Path, robust_path: Path) -> dict:
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    robust = json.loads(robust_path.read_text(encoding="utf-8"))
    if legacy.get("status") != "completed" or robust.get("status") != "completed":
        raise ValueError("Both source event evaluations must be completed")
    if (legacy.get("fold_id"), legacy.get("held_out_patient")) != (
        robust.get("fold_id"), robust.get("held_out_patient")
    ):
        raise ValueError("Legacy and robust results refer to different outer folds")
    if legacy.get("held_out_patient") != "patient_01":
        raise ValueError("This comparison expects the completed patient_01 first fold")

    legacy_test = legacy["outer_test"]["f2_threshold"]
    robust_test = robust["outer_test"]
    legacy_val = legacy["validation"]["f2_threshold"]
    robust_val = robust["validation"]
    return {
        "experiment": "legacy_vs_robust_first_fold_event_error_analysis",
        "version": "v1",
        "status": "completed",
        "model_retrained": False,
        "thresholds_or_alarm_policies_changed": False,
        "outer_test_used_for_tuning": False,
        "held_out_patient": "patient_01",
        "interpretation": (
            "Descriptive post-hoc error analysis only. Do not use these outer-test "
            "diagnostics to choose thresholds, features, or alarm policies."
        ),
        "validation": compare_splits(legacy_val, robust_val, "validation"),
        "outer_test": compare_splits(legacy_test, robust_test, "outer_test"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path,
        default=Path("results/baselines/first_fold_patient_01_event_metrics_v2.json"))
    parser.add_argument("--robust", type=Path,
        default=Path("results/baselines/robust_feature_first_fold_event_metrics_v1.json"))
    parser.add_argument("--output", type=Path,
        default=Path("results/baselines/legacy_vs_robust_first_fold_event_comparison_v1.json"))
    args = parser.parse_args()
    result = compare_results(args.legacy, args.robust)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "validation": result["validation"]["summary"],
        "outer_test": result["outer_test"]["summary"],
        "recovered_outer_test_events": [
            e for e in result["outer_test"]["events"]
            if e["outcome"] == "recovered_by_robust"
        ],
        "regressed_outer_test_events": [
            e for e in result["outer_test"]["events"]
            if e["outcome"] == "detected_only_by_legacy"
        ],
        "recordings_with_largest_false_alarm_increase": sorted(
            result["outer_test"]["recordings"],
            key=lambda row: row["false_alarm_delta_robust_minus_legacy"],
            reverse=True,
        )[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
