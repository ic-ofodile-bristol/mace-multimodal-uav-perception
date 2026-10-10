#!/usr/bin/env python3
"""
===============================================================================
extract_labelled_flight_data.py
===============================================================================

MACE - Extract Synchronised Labelled ArduPilot Flight Telemetry

Purpose
-------
Extract labelled aircraft telemetry directly from ArduPilot BIN files using:

    1. Video temporal annotations
    2. Previously estimated video <-> ArduPilot synchronization offsets
    3. Original ArduPilot BIN logs

No synchronization is estimated in this script.

Synchronization convention
--------------------------
The final synchronization master uses:

    autopilot_time_sec ~= video_time_sec + offset_sec

Therefore, for an annotated video interval:

    [start_sec, end_sec]

the corresponding ArduPilot interval is:

    [start_sec + offset_sec, end_sec + offset_sec]

Telemetry extracted
-------------------
ATT:
    Roll
    Pitch
    Yaw

GPS:
    Alt
    Spd
    Lat
    Lng

BARO:
    Alt

IMU:
    AccX
    AccY
    AccZ
    GyrX
    GyrY
    GyrZ

Outputs
-------
For each flight:

    Flights/<FLIGHT>/telemetry/labelled/

        FLxxx_labelled_telemetry.csv
        FLxxx_annotation_intervals.csv

Global outputs:

    results/flight_baseline/
        labelled_flight_telemetry.csv
        annotation_intervals.csv
        extraction_summary.csv

Important
---------
This script extracts annotation-level labelled telemetry.

It DOES NOT yet create the final 5-second model windows.
Window generation should be performed afterwards so that train/validation/test
sampling can be controlled independently.

===============================================================================
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FLIGHTS_DIR = Path("Flights")
DEFAULT_SYNC_MASTER = Path(
    "results/synchronization/final_synchronization_master.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/flight_baseline")

# Candidate annotation files, in preferred order.
ANNOTATION_CANDIDATES = [
    Path("temporal_labels_clean.csv"),
    Path("annotations/temporal_labels_clean.csv"),
    Path("data/temporal_labels_clean.csv"),
]

# Four-class baseline.
VALID_STATES = {
    "approaching",
    "entering",
    "inside",
    "exiting",
}

STATE_TO_ID = {
    "approaching": 0,
    "entering": 1,
    "inside": 2,
    "exiting": 3,
}


# =============================================================================
# Utility functions
# =============================================================================

def find_annotation_file(explicit_path: Path | None) -> Path:
    """Locate the temporal annotation CSV."""

    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"Annotation file does not exist: {explicit_path}"
            )
        return explicit_path

    for path in ANNOTATION_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate temporal_labels_clean.csv.\n"
        "Provide it explicitly using:\n\n"
        "  --annotations PATH/TO/temporal_labels_clean.csv"
    )


def find_mavlogdump(explicit_path: str | None) -> str:
    """Locate mavlogdump.py."""

    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"mavlogdump.py not found: {p}"
            )
        return str(p)

    # Search active Python environment first.
    import shutil

    found = shutil.which("mavlogdump.py")

    if found:
        return found

    # Common project venv location.
    candidate = Path("envs/mace/bin/mavlogdump.py")

    if candidate.exists():
        return str(candidate.resolve())

    raise FileNotFoundError(
        "Could not locate mavlogdump.py.\n"
        "Activate the mace environment or provide:\n\n"
        "  --mavlogdump /path/to/mavlogdump.py"
    )


def find_bin_file(flight_dir: Path) -> Path:
    """
    Find the ArduPilot BIN file belonging to a flight.

    Expected:
        Flights/FLxxx/telemetry/*.BIN
    """

    telemetry_dir = flight_dir / "telemetry"

    if not telemetry_dir.exists():
        raise FileNotFoundError(
            f"Telemetry directory missing: {telemetry_dir}"
        )

    bins = sorted(
        list(telemetry_dir.glob("*.BIN")) +
        list(telemetry_dir.glob("*.bin"))
    )

    if len(bins) == 0:
        raise FileNotFoundError(
            f"No BIN file found in {telemetry_dir}"
        )

    if len(bins) > 1:
        raise RuntimeError(
            f"Multiple BIN files found in {telemetry_dir}: "
            f"{[p.name for p in bins]}"
        )

    return bins[0]


def normalise_flight_id(value) -> str:
    """Convert flight ID to canonical FLxxx form."""

    s = str(value).strip()

    if s.upper().startswith("FL"):
        return s.upper()

    return s


# =============================================================================
# mavlogdump extraction
# =============================================================================

def extract_message_type(
    bin_path: Path,
    message_type: str,
    mavlogdump: str,
) -> pd.DataFrame:
    """
    Extract one ArduPilot message type from BIN using mavlogdump.py.

    Returns raw mavlogdump CSV as a DataFrame.
    """

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        cmd = [
            mavlogdump,
            str(bin_path),
            "--types",
            message_type,
            "--format",
            "csv",
        ]

        with temp_path.open("w") as fout:
            result = subprocess.run(
                cmd,
                stdout=fout,
                stderr=subprocess.PIPE,
                text=True,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"mavlogdump failed for {message_type}:\n"
                f"{result.stderr}"
            )

        if temp_path.stat().st_size == 0:
            return pd.DataFrame()

        try:
            df = pd.read_csv(temp_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        return df

    finally:
        temp_path.unlink(missing_ok=True)


# =============================================================================
# Column handling
# =============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Find a column using exact or suffix matching.

    mavlogdump versions may produce:
        TimeUS
        ATT.TimeUS
        ATT_Roll
        etc.
    """

    if df.empty:
        return None

    # Exact match.
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Case-insensitive exact match.
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lower_map:
            return lower_map[key]

    # Suffix match.
    for col in df.columns:
        col_lower = str(col).lower()

        for candidate in candidates:
            cand = candidate.lower()

            if (
                col_lower.endswith("." + cand)
                or col_lower.endswith("_" + cand)
            ):
                return col

    return None


def get_time_sec(df: pd.DataFrame) -> pd.Series:
    """
    Extract ArduPilot TimeUS and convert to seconds.

    The synchronization offsets are expressed in the same TimeUS-relative
    seconds convention.
    """

    time_col = find_column(
        df,
        [
            "TimeUS",
            "time_us",
        ],
    )

    if time_col is None:
        raise KeyError(
            f"Could not locate TimeUS column.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    values = pd.to_numeric(
        df[time_col],
        errors="coerce",
    )

    return values / 1_000_000.0


def numeric_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Return a numeric column or NaNs if unavailable."""

    col = find_column(df, candidates)

    if col is None:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[col],
        errors="coerce",
    )


# =============================================================================
# Standardise telemetry message types
# =============================================================================

def prepare_att(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["time_sec"] = get_time_sec(df)

    out["Roll_deg"] = numeric_column(
        df,
        ["Roll"],
    )

    out["Pitch_deg"] = numeric_column(
        df,
        ["Pitch"],
    )

    out["Yaw_deg"] = numeric_column(
        df,
        ["Yaw"],
    )

    return out.dropna(subset=["time_sec"]).sort_values("time_sec")


def prepare_gps(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["time_sec"] = get_time_sec(df)

    out["GPS_Lat"] = numeric_column(
        df,
        ["Lat"],
    )

    out["GPS_Lon"] = numeric_column(
        df,
        ["Lng", "Lon"],
    )

    out["GPS_Alt_m"] = numeric_column(
        df,
        ["Alt"],
    )

    out["GPS_Speed_mps"] = numeric_column(
        df,
        ["Spd", "Speed"],
    )

    return out.dropna(subset=["time_sec"]).sort_values("time_sec")


def prepare_baro(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["time_sec"] = get_time_sec(df)

    out["BARO_Alt_m"] = numeric_column(
        df,
        ["Alt"],
    )

    return out.dropna(subset=["time_sec"]).sort_values("time_sec")


def prepare_imu(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["time_sec"] = get_time_sec(df)

    out["AccX"] = numeric_column(
        df,
        ["AccX"],
    )

    out["AccY"] = numeric_column(
        df,
        ["AccY"],
    )

    out["AccZ"] = numeric_column(
        df,
        ["AccZ"],
    )

    out["GyrX"] = numeric_column(
        df,
        ["GyrX", "GyroX"],
    )

    out["GyrY"] = numeric_column(
        df,
        ["GyrY", "GyroY"],
    )

    out["GyrZ"] = numeric_column(
        df,
        ["GyrZ", "GyroZ"],
    )

    return out.dropna(subset=["time_sec"]).sort_values("time_sec")


# =============================================================================
# Time slicing
# =============================================================================

def slice_interval(
    df: pd.DataFrame,
    start_sec: float,
    end_sec: float,
) -> pd.DataFrame:
    """Extract telemetry samples inside an ArduPilot time interval."""

    if df.empty:
        return df.copy()

    mask = (
        (df["time_sec"] >= start_sec)
        &
        (df["time_sec"] <= end_sec)
    )

    return df.loc[mask].copy()


# =============================================================================
# Resampling
# =============================================================================

def interpolate_stream(
    df: pd.DataFrame,
    target_times: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """
    Interpolate a telemetry stream onto common target timestamps.

    NaN is retained if a signal is unavailable.
    """

    result = pd.DataFrame(
        {"autopilot_time_sec": target_times}
    )

    if df.empty:
        for col in columns:
            result[col] = np.nan

        return result

    source_time = pd.to_numeric(
        df["time_sec"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_time = np.isfinite(source_time)

    source_time = source_time[valid_time]

    if len(source_time) < 2:
        for col in columns:
            result[col] = np.nan

        return result

    # Remove duplicate timestamps.
    _, unique_idx = np.unique(
        source_time,
        return_index=True,
    )

    unique_idx = np.sort(unique_idx)

    source_time = source_time[unique_idx]

    for col in columns:

        if col not in df.columns:
            result[col] = np.nan
            continue

        values = pd.to_numeric(
            df.loc[valid_time, col],
            errors="coerce",
        ).to_numpy(dtype=float)

        values = values[unique_idx]

        valid = (
            np.isfinite(source_time)
            &
            np.isfinite(values)
        )

        if valid.sum() < 2:
            result[col] = np.nan
            continue

        x = source_time[valid]
        y = values[valid]

        interpolated = np.interp(
            target_times,
            x,
            y,
            left=np.nan,
            right=np.nan,
        )

        result[col] = interpolated

    return result


# =============================================================================
# Flight extraction
# =============================================================================

def process_flight(
    flight_id: str,
    flight_annotations: pd.DataFrame,
    offset_sec: float,
    sync_method: str,
    bin_path: Path,
    mavlogdump: str,
    output_rate_hz: float,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print()
    print("=" * 78)
    print(f"Processing {flight_id}")
    print("=" * 78)

    print(f"BIN         : {bin_path}")
    print(f"Offset      : {offset_sec:.6f} s")
    print(f"Sync method : {sync_method}")
    print(f"Annotations : {len(flight_annotations)}")

    output_dir = bin_path.parent / "labelled"
    output_dir.mkdir(parents=True, exist_ok=True)

    flight_output = (
        output_dir /
        f"{flight_id}_labelled_telemetry.csv"
    )

    interval_output = (
        output_dir /
        f"{flight_id}_annotation_intervals.csv"
    )

    if (
        flight_output.exists()
        and interval_output.exists()
        and not overwrite
    ):
        print("Existing outputs found. Loading.")

        return (
            pd.read_csv(flight_output),
            pd.read_csv(interval_output),
        )

    # -------------------------------------------------------------------------
    # Extract raw telemetry once per BIN
    # -------------------------------------------------------------------------

    print("Extracting ATT...")
    att_raw = extract_message_type(
        bin_path,
        "ATT",
        mavlogdump,
    )

    print(f"  rows: {len(att_raw):,}")

    print("Extracting GPS...")
    gps_raw = extract_message_type(
        bin_path,
        "GPS",
        mavlogdump,
    )

    print(f"  rows: {len(gps_raw):,}")

    print("Extracting BARO...")
    baro_raw = extract_message_type(
        bin_path,
        "BARO",
        mavlogdump,
    )

    print(f"  rows: {len(baro_raw):,}")

    print("Extracting IMU...")
    imu_raw = extract_message_type(
        bin_path,
        "IMU",
        mavlogdump,
    )

    print(f"  rows: {len(imu_raw):,}")

    # Standardise.
    att = prepare_att(att_raw)
    gps = prepare_gps(gps_raw)
    baro = prepare_baro(baro_raw)
    imu = prepare_imu(imu_raw)

    # -------------------------------------------------------------------------
    # Annotation intervals
    # -------------------------------------------------------------------------

    flight_rows = []
    interval_rows = []

    dt = 1.0 / output_rate_hz

    flight_annotations = (
        flight_annotations
        .sort_values("start_sec")
        .reset_index(drop=True)
    )

    for annotation_index, row in flight_annotations.iterrows():

        state = str(row["state"]).strip().lower()

        if state not in VALID_STATES:
            continue

        video_start = float(row["start_sec"])
        video_end = float(row["end_sec"])

        if not np.isfinite(video_start) or not np.isfinite(video_end):
            print(
                f"WARNING: invalid annotation timing "
                f"at row {annotation_index}"
            )
            continue

        if video_end <= video_start:
            print(
                f"WARNING: end <= start for annotation "
                f"{annotation_index}"
            )
            continue

        autopilot_start = video_start + offset_sec
        autopilot_end = video_end + offset_sec

        duration = video_end - video_start

        segment_id = (
            row["segment_id"]
            if "segment_id" in row.index
            else f"A{annotation_index:03d}"
        )

        segment_uid = (
            row["segment_uid"]
            if "segment_uid" in row.index
            else f"{flight_id}_{segment_id}"
        )

        print(
            f"  {segment_uid}: "
            f"{state:<12} "
            f"video={video_start:.3f}-{video_end:.3f} s "
            f"-> BIN={autopilot_start:.3f}-"
            f"{autopilot_end:.3f} s"
        )

        # ---------------------------------------------------------------------
        # Determine target timeline
        # ---------------------------------------------------------------------

        target_times = np.arange(
            autopilot_start,
            autopilot_end,
            dt,
            dtype=float,
        )

        if len(target_times) == 0:
            continue

        # ---------------------------------------------------------------------
        # Slice each source slightly wider than required.
        #
        # This provides enough surrounding samples for interpolation near
        # interval boundaries.
        # ---------------------------------------------------------------------

        margin = max(1.0, 2.0 * dt)

        att_slice = slice_interval(
            att,
            autopilot_start - margin,
            autopilot_end + margin,
        )

        gps_slice = slice_interval(
            gps,
            autopilot_start - margin,
            autopilot_end + margin,
        )

        baro_slice = slice_interval(
            baro,
            autopilot_start - margin,
            autopilot_end + margin,
        )

        imu_slice = slice_interval(
            imu,
            autopilot_start - margin,
            autopilot_end + margin,
        )

        # ---------------------------------------------------------------------
        # Interpolate all sources to common rate
        # ---------------------------------------------------------------------

        att_interp = interpolate_stream(
            att_slice,
            target_times,
            [
                "Roll_deg",
                "Pitch_deg",
                "Yaw_deg",
            ],
        )

        gps_interp = interpolate_stream(
            gps_slice,
            target_times,
            [
                "GPS_Lat",
                "GPS_Lon",
                "GPS_Alt_m",
                "GPS_Speed_mps",
            ],
        )

        baro_interp = interpolate_stream(
            baro_slice,
            target_times,
            [
                "BARO_Alt_m",
            ],
        )

        imu_interp = interpolate_stream(
            imu_slice,
            target_times,
            [
                "AccX",
                "AccY",
                "AccZ",
                "GyrX",
                "GyrY",
                "GyrZ",
            ],
        )

        # ---------------------------------------------------------------------
        # Combine
        # ---------------------------------------------------------------------

        combined = att_interp.copy()

        for source in [
            gps_interp,
            baro_interp,
            imu_interp,
        ]:
            for col in source.columns:
                if col == "autopilot_time_sec":
                    continue

                combined[col] = source[col].to_numpy()

        combined.insert(
            0,
            "flight_id",
            flight_id,
        )

        combined["video_time_sec"] = (
            combined["autopilot_time_sec"]
            - offset_sec
        )

        combined["annotation_start_sec"] = video_start
        combined["annotation_end_sec"] = video_end

        combined["segment_id"] = segment_id
        combined["segment_uid"] = segment_uid

        combined["state"] = state
        combined["state_id"] = STATE_TO_ID[state]

        combined["offset_sec"] = offset_sec
        combined["sync_method"] = sync_method

        # Relative time within this annotation.
        combined["annotation_time_sec"] = (
            combined["video_time_sec"]
            - video_start
        )

        flight_rows.append(combined)

        # ---------------------------------------------------------------------
        # Diagnostic interval row
        # ---------------------------------------------------------------------

        interval_rows.append(
            {
                "flight_id": flight_id,
                "segment_id": segment_id,
                "segment_uid": segment_uid,
                "state": state,
                "state_id": STATE_TO_ID[state],
                "video_start_sec": video_start,
                "video_end_sec": video_end,
                "duration_sec": duration,
                "offset_sec": offset_sec,
                "autopilot_start_sec": autopilot_start,
                "autopilot_end_sec": autopilot_end,
                "sync_method": sync_method,
                "output_rate_hz": output_rate_hz,
                "expected_samples": len(target_times),
                "ATT_samples_available": len(att_slice),
                "GPS_samples_available": len(gps_slice),
                "BARO_samples_available": len(baro_slice),
                "IMU_samples_available": len(imu_slice),
            }
        )

    if not flight_rows:
        raise RuntimeError(
            f"No valid labelled telemetry generated for {flight_id}"
        )

    flight_df = pd.concat(
        flight_rows,
        ignore_index=True,
    )

    interval_df = pd.DataFrame(interval_rows)

    # -------------------------------------------------------------------------
    # Reorder columns
    # -------------------------------------------------------------------------

    preferred_columns = [
        "flight_id",
        "segment_id",
        "segment_uid",
        "state",
        "state_id",
        "video_time_sec",
        "autopilot_time_sec",
        "annotation_time_sec",
        "annotation_start_sec",
        "annotation_end_sec",
        "offset_sec",
        "sync_method",

        # Model-relevant signals
        "Roll_deg",
        "Pitch_deg",
        "Yaw_deg",
        "BARO_Alt_m",
        "GPS_Alt_m",
        "GPS_Speed_mps",
        "AccX",
        "AccY",
        "AccZ",
        "GyrX",
        "GyrY",
        "GyrZ",

        # Retained for diagnostics / future use
        "GPS_Lat",
        "GPS_Lon",
    ]

    existing = [
        c
        for c in preferred_columns
        if c in flight_df.columns
    ]

    remaining = [
        c
        for c in flight_df.columns
        if c not in existing
    ]

    flight_df = flight_df[
        existing + remaining
    ]

    # Save.
    flight_df.to_csv(
        flight_output,
        index=False,
    )

    interval_df.to_csv(
        interval_output,
        index=False,
    )

    print()
    print(f"Saved: {flight_output}")
    print(f"Saved: {interval_output}")
    print(f"Rows : {len(flight_df):,}")

    return flight_df, interval_df


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract synchronised labelled ArduPilot telemetry "
            "for the MACE flight-data baseline."
        )
    )

    parser.add_argument(
        "--flights-dir",
        type=Path,
        default=DEFAULT_FLIGHTS_DIR,
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--sync-master",
        type=Path,
        default=DEFAULT_SYNC_MASTER,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--output-rate-hz",
        type=float,
        default=10.0,
        help="Common telemetry sampling rate. Default: 10 Hz.",
    )

    parser.add_argument(
        "--mavlogdump",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--flight-id",
        type=str,
        default=None,
        help="Optional single flight, e.g. FL019.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("MACE SYNCHRONISED LABELLED FLIGHT TELEMETRY EXTRACTION")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Resolve files
    # -------------------------------------------------------------------------

    annotations_path = find_annotation_file(
        args.annotations
    )

    mavlogdump = find_mavlogdump(
        args.mavlogdump
    )

    if not args.sync_master.exists():
        raise FileNotFoundError(
            f"Synchronization master missing: {args.sync_master}"
        )

    print(f"Flights directory : {args.flights_dir}")
    print(f"Annotations       : {annotations_path}")
    print(f"Sync master       : {args.sync_master}")
    print(f"Output directory  : {args.output_dir}")
    print(f"Output rate       : {args.output_rate_hz} Hz")
    print(f"mavlogdump        : {mavlogdump}")

    # -------------------------------------------------------------------------
    # Load annotations
    # -------------------------------------------------------------------------

    annotations = pd.read_csv(
        annotations_path
    )

    required_annotation_columns = {
        "flight_id",
        "start_sec",
        "end_sec",
        "state",
    }

    missing = (
        required_annotation_columns
        - set(annotations.columns)
    )

    if missing:
        raise KeyError(
            f"Annotation CSV missing required columns: {sorted(missing)}\n"
            f"Available: {annotations.columns.tolist()}"
        )

    annotations["flight_id"] = (
        annotations["flight_id"]
        .map(normalise_flight_id)
    )

    annotations["state"] = (
        annotations["state"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    annotations["start_sec"] = pd.to_numeric(
        annotations["start_sec"],
        errors="coerce",
    )

    annotations["end_sec"] = pd.to_numeric(
        annotations["end_sec"],
        errors="coerce",
    )

    # Keep four target states only.
    annotations = annotations[
        annotations["state"].isin(VALID_STATES)
    ].copy()

    # -------------------------------------------------------------------------
    # Load synchronization master
    # -------------------------------------------------------------------------

    sync = pd.read_csv(
        args.sync_master
    )

    required_sync_columns = {
        "flight_id",
        "offset_sec",
        "sync_method",
        "sync_status",
    }

    missing = (
        required_sync_columns
        - set(sync.columns)
    )

    if missing:
        raise KeyError(
            f"Synchronization master missing columns: {sorted(missing)}\n"
            f"Available: {sync.columns.tolist()}"
        )

    sync["flight_id"] = (
        sync["flight_id"]
        .map(normalise_flight_id)
    )

    sync["offset_sec"] = pd.to_numeric(
        sync["offset_sec"],
        errors="coerce",
    )

    sync = sync[
        sync["sync_status"]
        .astype(str)
        .str.lower()
        .eq("success")
    ].copy()

    sync = sync.dropna(
        subset=["offset_sec"]
    )

    # -------------------------------------------------------------------------
    # Determine flights
    # -------------------------------------------------------------------------

    if args.flight_id:

        requested = normalise_flight_id(
            args.flight_id
        )

        flight_ids = [requested]

    else:

        flight_ids = sorted(
            set(annotations["flight_id"])
            &
            set(sync["flight_id"])
        )

    print()
    print(f"Flights to process : {len(flight_ids)}")
    print(" ".join(flight_ids))

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_telemetry = []
    all_intervals = []
    summary_rows = []

    # -------------------------------------------------------------------------
    # Process flights
    # -------------------------------------------------------------------------

    for flight_id in flight_ids:

        try:

            flight_dir = (
                args.flights_dir /
                flight_id
            )

            if not flight_dir.exists():
                raise FileNotFoundError(
                    f"Flight directory missing: {flight_dir}"
                )

            flight_annotations = annotations[
                annotations["flight_id"] == flight_id
            ].copy()

            if flight_annotations.empty:
                raise RuntimeError(
                    "No annotations found."
                )

            sync_rows = sync[
                sync["flight_id"] == flight_id
            ]

            if len(sync_rows) != 1:
                raise RuntimeError(
                    f"Expected exactly one successful sync row; "
                    f"found {len(sync_rows)}."
                )

            sync_row = sync_rows.iloc[0]

            offset_sec = float(
                sync_row["offset_sec"]
            )

            sync_method = str(
                sync_row["sync_method"]
            )

            bin_path = find_bin_file(
                flight_dir
            )

            flight_df, interval_df = process_flight(
                flight_id=flight_id,
                flight_annotations=flight_annotations,
                offset_sec=offset_sec,
                sync_method=sync_method,
                bin_path=bin_path,
                mavlogdump=mavlogdump,
                output_rate_hz=args.output_rate_hz,
                overwrite=args.overwrite,
            )

            all_telemetry.append(
                flight_df
            )

            all_intervals.append(
                interval_df
            )

            summary_rows.append(
                {
                    "flight_id": flight_id,
                    "status": "success",
                    "bin_file": str(bin_path),
                    "offset_sec": offset_sec,
                    "sync_method": sync_method,
                    "n_annotations": len(interval_df),
                    "n_telemetry_rows": len(flight_df),
                    "error": "",
                }
            )

        except Exception as exc:

            print()
            print(
                f"ERROR {flight_id}: {exc}"
            )

            summary_rows.append(
                {
                    "flight_id": flight_id,
                    "status": "failed",
                    "bin_file": "",
                    "offset_sec": np.nan,
                    "sync_method": "",
                    "n_annotations": 0,
                    "n_telemetry_rows": 0,
                    "error": str(exc),
                }
            )

    # -------------------------------------------------------------------------
    # Save master datasets
    # -------------------------------------------------------------------------

    if all_telemetry:

        master_telemetry = pd.concat(
            all_telemetry,
            ignore_index=True,
        )

        master_path = (
            args.output_dir /
            "labelled_flight_telemetry.csv"
        )

        master_telemetry.to_csv(
            master_path,
            index=False,
        )

    else:
        master_telemetry = pd.DataFrame()
        master_path = None

    if all_intervals:

        master_intervals = pd.concat(
            all_intervals,
            ignore_index=True,
        )

        intervals_path = (
            args.output_dir /
            "annotation_intervals.csv"
        )

        master_intervals.to_csv(
            intervals_path,
            index=False,
        )

    else:
        master_intervals = pd.DataFrame()
        intervals_path = None

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        args.output_dir /
        "extraction_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    successful = (
        summary["status"]
        .eq("success")
        .sum()
    )

    failed = (
        summary["status"]
        .eq("failed")
        .sum()
    )

    print()
    print("=" * 78)
    print("EXTRACTION SUMMARY")
    print("=" * 78)

    print(f"Flights processed : {len(summary)}")
    print(f"Successful        : {successful}")
    print(f"Failed            : {failed}")

    if not master_telemetry.empty:

        print()
        print(
            f"Telemetry rows    : "
            f"{len(master_telemetry):,}"
        )

        print()
        print("Rows by state:")

        print(
            master_telemetry[
                "state"
            ].value_counts()
        )

        print()
        print("Annotation segments by state:")

        print(
            master_intervals[
                "state"
            ].value_counts()
        )

    print()
    print("Outputs:")

    if master_path:
        print(f"  {master_path}")

    if intervals_path:
        print(f"  {intervals_path}")

    print(f"  {summary_path}")

    if failed:

        print()
        print("Failures:")

        for _, row in summary[
            summary["status"] == "failed"
        ].iterrows():

            print(
                f"  {row['flight_id']}: "
                f"{row['error']}"
            )

    print()
    print("=" * 78)
    print("Synchronization convention used:")
    print()
    print(
        "  autopilot_time_sec = "
        "video_time_sec + offset_sec"
    )
    print()
    print(
        "No synchronization was estimated or modified "
        "by this script."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()