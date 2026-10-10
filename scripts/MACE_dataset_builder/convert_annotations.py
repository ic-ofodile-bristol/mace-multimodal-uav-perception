#!/usr/bin/env python3
"""
convert_annotations.py

Convert the manually labelled MACE temporal annotation spreadsheet into clean
machine-readable annotation files for dataset building.

Expected input columns, based on MACE temporal labels spreadsheet:
    Flight_ID, Video_File, Video_Camera, Duration, Segment_ID,
    Start_Time, End_Time, Start_sec, End_sec, State, Confidence

Outputs:
    temporal_labels_clean.csv
    temporal_labels_clean.jsonl
    video_manifest.csv
    label_map.json
    conversion_report.txt

Example:
    python scripts/MACE_dataset_builder/convert_annotations.py \
        --input data/annotations/MACE_temporal_labels.xlsx \
        --output_dir data/processed/annotations
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


STATE_MAP = {
    "approaching": "approaching",
    "approach": "approaching",
    "entering": "entering",
    "entry": "entering",
    "inside": "inside",
    "in plume": "inside",
    "exiting": "exiting",
    "exit": "exiting",
    "distant plume": "distant_plume",
    "distant_plume": "distant_plume",
    "distant": "distant_plume",
}

CONFIDENCE_MAP = {
    "high": "high",
    "medium": "medium",
    "med": "medium",
    "low": "low",
}

LABEL_MAP = {
    "approaching": 0,
    "entering": 1,
    "inside": 2,
    "exiting": 3,
    "distant_plume": 4,
}


def normalise_colname(name: str) -> str:
    """Normalise spreadsheet column names while preserving expected names."""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    aliases = {
        "flight_id": "Flight_ID",
        "video_file": "Video_File",
        "video_camera": "Video_Camera",
        "duration": "Duration",
        "segment_id": "Segment_ID",
        "start_time": "Start_Time",
        "end_time": "End_Time",
        "start_sec": "Start_sec",
        "end_sec": "End_sec",
        "state": "State",
        "confidence": "Confidence",
    }
    return aliases.get(name.lower(), name)


'''def time_to_seconds(value: Any) -> Optional[float]:
    """Convert spreadsheet time values such as 0:22, 8:51, 1:02:03 to seconds."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        # Excel sometimes stores time as day fraction; otherwise this is already seconds.
        if 0 < float(value) < 1:
            return round(float(value) * 24 * 60 * 60, 3)
        return float(value)

    text = str(value).strip()
    if text == "":
        return None

    # Remove accidental leading apostrophe from Excel text cells.
    text = text.lstrip("'")

    parts = text.split(":")
    try:
        parts_float = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Cannot parse time value: {value!r}")

    if len(parts_float) == 1:
        return parts_float[0]
    if len(parts_float) == 2:
        minutes, seconds = parts_float
        return minutes * 60 + seconds
    if len(parts_float) == 3:
        hours, minutes, seconds = parts_float
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Cannot parse time value: {value!r}")'''

def time_to_seconds(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None

    # IMPORTANT:
    # Excel reads 8:51 as 8 hours 51 minutes.
    # For this MACE spreadsheet, 8:51 means 8 minutes 51 seconds.
    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return float(value.hour * 60 + value.minute + value.second / 60)

    if isinstance(value, (int, float)):
        if 0 < float(value) < 1:
            # Excel day fraction, but spreadsheet intended MM:SS not HH:MM.
            return round(float(value) * 24 * 60, 3)
        return float(value)

    text = str(value).strip()
    if text == "":
        return None

    text = text.lstrip("'")

    parts = text.split(":")
    try:
        parts_float = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Cannot parse time value: {value!r}")

    if len(parts_float) == 1:
        return parts_float[0]
    if len(parts_float) == 2:
        minutes, seconds = parts_float
        return minutes * 60 + seconds
    if len(parts_float) == 3:
        hours, minutes, seconds = parts_float
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Cannot parse time value: {value!r}")

def seconds_to_mmss(seconds: Optional[float]) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    seconds_i = int(round(float(seconds)))
    m, s = divmod(seconds_i, 60)
    return f"{m}:{s:02d}"


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalise_state(value: Any) -> str:
    raw = clean_text(value)
    key = raw.lower().replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()
    if key not in STATE_MAP:
        raise ValueError(f"Unknown State value: {raw!r}")
    return STATE_MAP[key]


def normalise_confidence(value: Any) -> str:
    raw = clean_text(value)
    key = raw.lower().strip()
    if key == "":
        return "unknown"
    if key not in CONFIDENCE_MAP:
        raise ValueError(f"Unknown Confidence value: {raw!r}")
    return CONFIDENCE_MAP[key]


def read_annotations(input_path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def convert_annotations(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = df.copy()
    df.columns = [normalise_colname(c) for c in df.columns]

    required = ["Flight_ID", "Video_File", "Duration", "Segment_ID", "Start_Time", "End_Time", "State", "Confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Video_Camera" not in df.columns:
        df["Video_Camera"] = ""
    if "Start_sec" not in df.columns:
        df["Start_sec"] = None
    if "End_sec" not in df.columns:
        df["End_sec"] = None

    # Remove fully empty rows.
    df = df.dropna(how="all").reset_index(drop=True)

    clean_rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        flight_id = clean_text(row.get("Flight_ID"))
        video_file = clean_text(row.get("Video_File"))
        segment_id = clean_text(row.get("Segment_ID"))

        if not flight_id and not video_file and not segment_id:
            continue

        duration_sec = time_to_seconds(row.get("Duration"))
        start_sec = time_to_seconds(row.get("Start_sec")) if not pd.isna(row.get("Start_sec")) else None
        end_sec = time_to_seconds(row.get("End_sec")) if not pd.isna(row.get("End_sec")) else None

        # Prefer explicitly typed Start_sec/End_sec if present; otherwise compute from Start_Time/End_Time.
        if start_sec is None:
            start_sec = time_to_seconds(row.get("Start_Time"))
        if end_sec is None:
            end_sec = time_to_seconds(row.get("End_Time"))

        if start_sec is None or end_sec is None:
            raise ValueError(f"Missing start/end time in row {idx + 2}")
        if end_sec <= start_sec:
            raise ValueError(f"End time must be greater than start time in row {idx + 2}: {start_sec} -> {end_sec}")

        if duration_sec is not None and end_sec > duration_sec + 1:
            warnings.append(
                f"Row {idx + 2}: end_sec {end_sec} exceeds duration_sec {duration_sec} for {flight_id}/{video_file}."
            )

        state = normalise_state(row.get("State"))
        confidence = normalise_confidence(row.get("Confidence"))

        clean_rows.append(
            {
                "flight_id": flight_id,
                "video_file": video_file,
                "video_camera": clean_text(row.get("Video_Camera")),
                "duration": seconds_to_mmss(duration_sec),
                "duration_sec": duration_sec,
                "segment_id": segment_id,
                "segment_uid": f"{flight_id}_{segment_id}",
                "start_time": seconds_to_mmss(start_sec),
                "end_time": seconds_to_mmss(end_sec),
                "start_sec": round(float(start_sec), 3),
                "end_sec": round(float(end_sec), 3),
                "segment_duration_sec": round(float(end_sec - start_sec), 3),
                "state": state,
                "state_id": LABEL_MAP[state],
                "confidence": confidence,
            }
        )

    out = pd.DataFrame(clean_rows)
    if out.empty:
        raise ValueError("No valid annotation rows were found.")

    out = out.sort_values(["flight_id", "video_file", "start_sec", "end_sec"]).reset_index(drop=True)

    # Check duplicates and overlaps within each flight/video.
    dupes = out[out.duplicated(["flight_id", "video_file", "segment_id"], keep=False)]
    if not dupes.empty:
        warnings.append("Duplicate flight/video/segment_id entries detected.")

    for (flight_id, video_file), group in out.groupby(["flight_id", "video_file"], dropna=False):
        previous_end: Optional[float] = None
        previous_segment: Optional[str] = None
        for _, row in group.iterrows():
            if previous_end is not None and row["start_sec"] < previous_end:
                warnings.append(
                    f"Overlap in {flight_id}/{video_file}: {previous_segment} overlaps {row['segment_id']}."
                )
            previous_end = row["end_sec"]
            previous_segment = row["segment_id"]

    return out, warnings


def write_outputs(clean: pd.DataFrame, warnings: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean.to_csv(output_dir / "temporal_labels_clean.csv", index=False)
    clean.to_json(output_dir / "temporal_labels_clean.jsonl", orient="records", lines=True)

    manifest_cols = ["flight_id", "video_file", "video_camera", "duration", "duration_sec"]
    manifest = clean[manifest_cols].drop_duplicates().sort_values(["flight_id", "video_file"])
    manifest.to_csv(output_dir / "video_manifest.csv", index=False)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    counts = clean.groupby(["state", "confidence"]).size().reset_index(name="count")
    report_lines = [
        "MACE annotation conversion report",
        "=================================",
        f"Segments converted: {len(clean)}",
        f"Unique flights: {clean['flight_id'].nunique()}",
        f"Unique videos: {clean[['flight_id', 'video_file']].drop_duplicates().shape[0]}",
        "",
        "State/confidence counts:",
        counts.to_string(index=False),
        "",
        "Warnings:",
    ]
    report_lines.extend(warnings if warnings else ["None"])
    (output_dir / "conversion_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MACE temporal annotation spreadsheet to clean dataset files.")
    parser.add_argument("--input", required=True, type=Path, help="Path to annotation .xlsx, .xls, or .csv file.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory where converted files will be written.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name. Defaults to first sheet.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    raw = read_annotations(args.input, args.sheet)
    clean, warnings = convert_annotations(raw)
    write_outputs(clean, warnings, args.output_dir)

    print(f"Converted {len(clean)} segments.")
    print(f"Outputs written to: {args.output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warning(s). See conversion_report.txt")


if __name__ == "__main__":
    main()
