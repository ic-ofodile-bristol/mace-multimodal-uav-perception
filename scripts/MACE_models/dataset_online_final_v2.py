#!/usr/bin/env python3
"""
===============================================================================
dataset_online_final.py
===============================================================================

Builds the annotation database required for online temporal sampling.

This first version:
    - reads MACE_Temporal_label_complete.xlsx;
    - standardises column and label names;
    - converts annotation times to seconds;
    - loads the existing flight-level train/validation/test split;
    - constructs the source video path as:

          Flights/<Flight_ID>/<Flight_ID>.MP4

      The Video_File column in the workbook is retained only as metadata and is
      not used to locate the uploaded Isambard video;
    - filters out annotations shorter than the five-second training window;
    - groups annotations by class;
    - prints annotation and duration statistics.

Part 2 additionally:
    - creates an exactly balanced class schedule for every epoch;
    - chooses annotations with duration-weighted probability;
    - samples valid five-second start times uniformly;
    - makes sampling reproducible from seed, epoch and index;
    - can export the temporal sampling plan for an epoch.

Final production implementation for Isambard/HPC use.

It:
    - opens source MP4 videos with OpenCV;
    - extracts 25 uniformly spaced RGB frames from each sampled window;
    - applies sequence-consistent Albumentations transforms;
    - returns ImageNet-normalised tensors with shape [T, C, H, W];
    - constrains OpenCV, OpenMP, BLAS and PyTorch CPU threading;
    - maintains a worker-local LRU video cache;
    - performs mostly sequential frame decoding within each clip;
    - recovers from occasional seek/read failures;
    - keeps validation and test sampling fixed across epochs.
===============================================================================
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
import inspect
import os

# HPC-safe thread limits. These are intentionally set before NumPy, OpenCV and
# PyTorch are imported so that their native libraries do not create oversized
# worker pools on Isambard.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")
os.environ.setdefault("FFMPEG_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError as error:
    raise ImportError(
        "OpenCV is required for online video frame extraction. "
        "Install it inside the MACE environment with: "
        "pip install opencv-python-headless"
    ) from error

try:
    import albumentations as A
except ImportError as error:
    raise ImportError(
        "Albumentations is required for image preprocessing and augmentation. "
        "Install it inside the MACE environment with: "
        "pip install albumentations"
    ) from error

# Prevent the thread-creation failure observed on Isambard.
cv2.setNumThreads(1)

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    cv2.setLogLevel(0)
except Exception:
    pass

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# =============================================================================
# Configuration
# =============================================================================

LABEL_MAP = {
    "approaching": 0,
    "entering": 1,
    "inside": 2,
    "exiting": 3,
}

ID_TO_LABEL = {
    class_id: label
    for label, class_id in LABEL_MAP.items()
}

VALID_SPLITS = {
    "train",
    "val",
    "test",
}

DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_FRAMES_PER_SAMPLE = 25
DEFAULT_SAMPLES_PER_EPOCH = 12_000
DEFAULT_BASE_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MAX_CACHED_VIDEOS = 4
DEFAULT_FRAME_SEARCH_RADIUS = 3
DEFAULT_SEQUENTIAL_READ_TOLERANCE = 2
DEFAULT_MIN_WINDOW_SPACING_SEC = 2.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Helper functions
# =============================================================================

def standardise_column_name(column: Any) -> str:
    """Convert a workbook column name to lowercase snake_case."""

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalise_flight_id(value: Any) -> str:
    """Return a clean, uppercase flight identifier such as FL008."""

    if pd.isna(value):
        raise ValueError("Encountered an empty Flight_ID.")

    flight_id = str(value).strip().upper()

    if not flight_id:
        raise ValueError("Encountered an empty Flight_ID.")

    return flight_id


def project_time_to_seconds(value: Any) -> float:
    """
    Convert a MACE workbook annotation time to seconds.

    The workbook displays values such as:

        00:22:00  -> 22 seconds
        03:07:00  -> 3 minutes 7 seconds -> 187 seconds

    Therefore, for three-part values formatted as ``AA:BB:CC``, this project
    treats:

        AA = minutes
        BB = seconds
        CC = fractional/unused trailing field

    This differs from the usual interpretation of HH:MM:SS, but matches the
    existing MACE annotation convention shown in the workbook.

    Numeric values are also supported:
        - values between 0 and 1 are treated as Excel day fractions;
        - values greater than or equal to 1 are treated as seconds.
    """

    if value is None or pd.isna(value):
        return np.nan

    if isinstance(value, pd.Timestamp):
        value = value.time()

    if isinstance(value, datetime):
        value = value.time()

    if isinstance(value, time):
        return (
            float(value.hour * 60)
            + float(value.minute)
            + float(value.second) / 100.0
            + float(value.microsecond) / 100_000_000.0
        )

    if isinstance(value, timedelta):
        return float(value.total_seconds())

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)

        if 0.0 <= numeric_value < 1.0:
            # Excel stores times as fractions of one day.
            # Convert to displayed clock components first, then apply the
            # project's MM:SS:00 convention.
            total_clock_seconds = numeric_value * 24.0 * 60.0 * 60.0
            hours = int(total_clock_seconds // 3600)
            minutes = int((total_clock_seconds % 3600) // 60)
            seconds = total_clock_seconds % 60

            return (
                float(hours * 60)
                + float(minutes)
                + float(seconds) / 100.0
            )

        return numeric_value

    text = str(value).strip()

    if not text:
        return np.nan

    # Permit values such as "0 days 00:22:00".
    if "day" in text.lower():
        try:
            parsed_delta = pd.to_timedelta(text)
            total_clock_seconds = parsed_delta.total_seconds()
            hours = int(total_clock_seconds // 3600)
            minutes = int((total_clock_seconds % 3600) // 60)
            seconds = total_clock_seconds % 60

            return (
                float(hours * 60)
                + float(minutes)
                + float(seconds) / 100.0
            )
        except (TypeError, ValueError):
            pass

    parts = text.split(":")

    try:
        if len(parts) == 3:
            minutes = float(parts[0])
            seconds = float(parts[1])
            trailing = float(parts[2])

            return minutes * 60.0 + seconds + trailing / 100.0

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])

            return minutes * 60.0 + seconds

        if len(parts) == 1:
            return float(parts[0])

    except ValueError as error:
        raise ValueError(
            f"Could not convert annotation time {value!r} to seconds."
        ) from error

    raise ValueError(
        f"Unsupported annotation time format: {value!r}"
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:
    """Return the first matching standardised column name."""

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not find any of the required columns: "
            f"{list(candidates)}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )

    return None


def load_flight_splits(split_file: Path) -> pd.DataFrame:
    """
    Load and standardise flight_splits.csv.

    Expected logical columns:
        flight_id
        split
    """

    if not split_file.exists():
        raise FileNotFoundError(
            f"Flight split file was not found:\n{split_file}"
        )

    split_dataframe = pd.read_csv(split_file)

    split_dataframe.columns = [
        standardise_column_name(column)
        for column in split_dataframe.columns
    ]

    flight_column = find_column(
        split_dataframe,
        ("flight_id", "flight", "flightid"),
    )

    split_column = find_column(
        split_dataframe,
        ("split", "dataset_split", "set"),
    )

    split_dataframe = split_dataframe[
        [
            flight_column,
            split_column,
        ]
    ].copy()

    split_dataframe = split_dataframe.rename(
        columns={
            flight_column: "flight_id",
            split_column: "split",
        }
    )

    split_dataframe["flight_id"] = (
        split_dataframe["flight_id"]
        .apply(normalise_flight_id)
    )

    split_dataframe["split"] = (
        split_dataframe["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "validation": "val",
                "valid": "val",
            }
        )
    )

    invalid_splits = sorted(
        set(split_dataframe["split"]) - VALID_SPLITS
    )

    if invalid_splits:
        raise ValueError(
            f"Unsupported values in flight_splits.csv: {invalid_splits}. "
            f"Expected only {sorted(VALID_SPLITS)}."
        )

    duplicate_split_counts = (
        split_dataframe
        .groupby("flight_id")["split"]
        .nunique()
    )

    inconsistent_flights = duplicate_split_counts[
        duplicate_split_counts > 1
    ]

    if not inconsistent_flights.empty:
        raise ValueError(
            "Some flights are assigned to more than one split:\n"
            f"{inconsistent_flights.to_string()}"
        )

    return (
        split_dataframe
        .drop_duplicates(subset=["flight_id"])
        .reset_index(drop=True)
    )


# =============================================================================
# Dataset
# =============================================================================

class OnlineTemporalVisionDataset(Dataset):
    """
    Complete online temporal vision dataset for the MACE project.

    It performs split-safe annotation loading, balanced online temporal
    sampling, OpenCV frame extraction, sequence preprocessing and conversion
    to PyTorch tensors.
    """

    def __init__(
        self,
        annotation_file: Path | str,
        flights_root: Path | str,
        split_file: Path | str,
        split: str = "train",
        samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        frames_per_sample: int = DEFAULT_FRAMES_PER_SAMPLE,
        augment: bool = False,
        base_seed: int = DEFAULT_BASE_SEED,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_cached_videos: int = DEFAULT_MAX_CACHED_VIDEOS,
        frame_search_radius: int = DEFAULT_FRAME_SEARCH_RADIUS,
        sequential_read_tolerance: int = DEFAULT_SEQUENTIAL_READ_TOLERANCE,
        min_window_spacing_sec: float = DEFAULT_MIN_WINDOW_SPACING_SEC,
        return_metadata: bool = True,
        sheet_name: str | int = "Sheet1",
        verify_videos: bool = True,
    ) -> None:

        self.annotation_file = Path(annotation_file)
        self.flights_root = Path(flights_root)
        self.split_file = Path(split_file)

        self.split = str(split).strip().lower()
        self.samples_per_epoch = int(samples_per_epoch)
        self.window_seconds = float(window_seconds)
        self.frames_per_sample = int(frames_per_sample)
        self.augment = bool(augment)
        self.base_seed = int(base_seed)
        self.image_size = int(image_size)
        self.max_cached_videos = int(max_cached_videos)
        self.frame_search_radius = int(frame_search_radius)
        self.sequential_read_tolerance = int(sequential_read_tolerance)
        self.min_window_spacing_sec = float(min_window_spacing_sec)
        self.return_metadata = bool(return_metadata)
        self.sheet_name = sheet_name
        self.verify_videos = bool(verify_videos)

        if self.split == "validation":
            self.split = "val"

        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split {split!r}. "
                f"Expected one of {sorted(VALID_SPLITS)}."
            )

        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be greater than zero.")

        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")

        if self.frames_per_sample <= 0:
            raise ValueError("frames_per_sample must be greater than zero.")

        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

        if self.max_cached_videos <= 0:
            raise ValueError("max_cached_videos must be greater than zero.")

        if self.frame_search_radius < 0:
            raise ValueError("frame_search_radius cannot be negative.")

        if self.sequential_read_tolerance < 0:
            raise ValueError(
                "sequential_read_tolerance cannot be negative."
            )

        if self.min_window_spacing_sec <= 0.0:
            raise ValueError(
                "min_window_spacing_sec must be greater than zero."
            )

        if self.split != "train" and self.augment:
            print(
                f"Warning: augmentation was requested for split={self.split!r}. "
                "It has been disabled so validation/test inputs remain fixed."
            )
            self.augment = False

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation workbook was not found:\n"
                f"{self.annotation_file}"
            )

        if not self.flights_root.exists():
            raise FileNotFoundError(
                f"Flights directory was not found:\n"
                f"{self.flights_root}"
            )

        self.annotations = self._load_annotations()

        self.annotations_by_label = self._build_annotations_by_label()

        self.class_statistics = self._build_class_statistics()

        self._validate_class_coverage()

        self.current_epoch = 0
        self.label_schedule = np.empty(
            self.samples_per_epoch,
            dtype=np.int64,
        )
        self.epoch_sampling_plan: list[dict[str, Any]] = []

        # VideoCapture objects must be opened lazily inside each DataLoader
        # worker process. They are not safe to share across forked processes.
        self._video_cache: OrderedDict[
            str,
            dict[str, Any],
        ] = OrderedDict()
        self._video_cache_pid: int | None = None

        self.transform = self._build_transform()

        self.start_new_epoch(epoch=0)

    # -------------------------------------------------------------------------

    def _load_annotations(self) -> list[dict[str, Any]]:
        """Read, clean, split and validate the annotation workbook."""

        dataframe = pd.read_excel(
            self.annotation_file,
            sheet_name=self.sheet_name,
            engine="openpyxl",
        )

        dataframe.columns = [
            standardise_column_name(column)
            for column in dataframe.columns
        ]

        flight_column = find_column(
            dataframe,
            ("flight_id", "flight", "flightid"),
        )

        workbook_video_column = find_column(
            dataframe,
            ("video_file", "video", "filename"),
            required=False,
        )

        segment_column = find_column(
            dataframe,
            ("segment_id", "segment", "segment_uid"),
            required=False,
        )

        state_column = find_column(
            dataframe,
            ("state", "label", "interaction_state"),
        )

        start_seconds_column = find_column(
            dataframe,
            ("start_sec", "start_seconds"),
            required=False,
        )

        end_seconds_column = find_column(
            dataframe,
            ("end_sec", "end_seconds"),
            required=False,
        )

        start_time_column = find_column(
            dataframe,
            ("start_time", "start"),
            required=start_seconds_column is None,
        )

        end_time_column = find_column(
            dataframe,
            ("end_time", "end"),
            required=end_seconds_column is None,
        )

        cleaned = pd.DataFrame()

        cleaned["flight_id"] = (
            dataframe[flight_column]
            .apply(normalise_flight_id)
        )

        if workbook_video_column is not None:
            cleaned["workbook_video_file"] = (
                dataframe[workbook_video_column]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        else:
            cleaned["workbook_video_file"] = ""

        if segment_column is not None:
            cleaned["segment_id"] = (
                dataframe[segment_column]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        else:
            cleaned["segment_id"] = ""

        cleaned["label"] = (
            dataframe[state_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
            .str.replace("-", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )

        # Keep only the four labels used by the current classifier.
        cleaned = cleaned[
            cleaned["label"].isin(LABEL_MAP)
        ].copy()

        if start_seconds_column is not None:
            parsed_start_seconds = pd.to_numeric(
                dataframe.loc[
                    cleaned.index,
                    start_seconds_column,
                ],
                errors="coerce",
            )
        else:
            parsed_start_seconds = pd.Series(
                np.nan,
                index=cleaned.index,
                dtype=float,
            )

        if end_seconds_column is not None:
            parsed_end_seconds = pd.to_numeric(
                dataframe.loc[
                    cleaned.index,
                    end_seconds_column,
                ],
                errors="coerce",
            )
        else:
            parsed_end_seconds = pd.Series(
                np.nan,
                index=cleaned.index,
                dtype=float,
            )

        if start_time_column is not None:
            fallback_start_seconds = (
                dataframe.loc[
                    cleaned.index,
                    start_time_column,
                ]
                .apply(project_time_to_seconds)
            )

            parsed_start_seconds = (
                parsed_start_seconds
                .fillna(fallback_start_seconds)
            )

        if end_time_column is not None:
            fallback_end_seconds = (
                dataframe.loc[
                    cleaned.index,
                    end_time_column,
                ]
                .apply(project_time_to_seconds)
            )

            parsed_end_seconds = (
                parsed_end_seconds
                .fillna(fallback_end_seconds)
            )

        cleaned["annotation_start_sec"] = parsed_start_seconds
        cleaned["annotation_end_sec"] = parsed_end_seconds

        cleaned = cleaned.dropna(
            subset=[
                "annotation_start_sec",
                "annotation_end_sec",
            ]
        ).copy()

        cleaned["annotation_start_sec"] = (
            cleaned["annotation_start_sec"]
            .astype(float)
        )

        cleaned["annotation_end_sec"] = (
            cleaned["annotation_end_sec"]
            .astype(float)
        )

        cleaned["annotation_duration_sec"] = (
            cleaned["annotation_end_sec"]
            - cleaned["annotation_start_sec"]
        )

        invalid_intervals = cleaned[
            (
                cleaned["annotation_start_sec"] < 0
            )
            | (
                cleaned["annotation_end_sec"]
                <= cleaned["annotation_start_sec"]
            )
        ]

        if not invalid_intervals.empty:
            raise ValueError(
                "Invalid annotation intervals were found:\n"
                f"{invalid_intervals.to_string(index=False)}"
            )

        self.excluded_short_annotations = cleaned[
            cleaned["annotation_duration_sec"]
            < self.window_seconds
        ].copy()

        cleaned = cleaned[
            cleaned["annotation_duration_sec"]
            >= self.window_seconds
        ].copy()

        flight_splits = load_flight_splits(
            self.split_file
        )

        cleaned = cleaned.merge(
            flight_splits,
            on="flight_id",
            how="left",
            validate="many_to_one",
        )

        missing_split_rows = cleaned[
            cleaned["split"].isna()
        ]

        if not missing_split_rows.empty:
            missing_flights = sorted(
                missing_split_rows["flight_id"]
                .unique()
                .tolist()
            )

            raise ValueError(
                "No split assignment was found for these flights:\n"
                f"{missing_flights}"
            )

        cleaned = cleaned[
            cleaned["split"] == self.split
        ].copy()

        if cleaned.empty:
            raise ValueError(
                f"No eligible annotations were found for split "
                f"{self.split!r}."
            )

        # The actual uploaded Isambard videos use Flight_ID.MP4, irrespective
        # of the different source filename recorded in the workbook.
        cleaned["video_file"] = (
            cleaned["flight_id"]
            + ".MP4"
        )

        cleaned["video_path"] = cleaned.apply(
            lambda row: (
                self.flights_root
                / row["flight_id"]
                / row["video_file"]
            ),
            axis=1,
        )

        if self.verify_videos:
            missing_video_rows = cleaned[
                ~cleaned["video_path"].apply(
                    lambda path: Path(path).is_file()
                )
            ]

            if not missing_video_rows.empty:
                display_columns = [
                    "flight_id",
                    "video_file",
                    "video_path",
                ]

                raise FileNotFoundError(
                    "Some source videos were not found. "
                    "The expected convention is "
                    "Flights/<Flight_ID>/<Flight_ID>.MP4:\n"
                    f"{missing_video_rows[display_columns].to_string(index=False)}"
                )

        cleaned["class_id"] = (
            cleaned["label"]
            .map(LABEL_MAP)
            .astype(int)
        )

        cleaned = cleaned.sort_values(
            by=[
                "flight_id",
                "annotation_start_sec",
                "annotation_end_sec",
            ]
        ).reset_index(drop=True)

        annotation_records: list[dict[str, Any]] = []

        for row_number, row in cleaned.iterrows():

            segment_id = row["segment_id"]

            if not segment_id:
                segment_id = f"ROW{row_number + 1:04d}"

            annotation_id = (
                f"{row['flight_id']}_{segment_id}"
            )

            annotation_records.append(
                {
                    "annotation_id": annotation_id,
                    "flight_id": row["flight_id"],
                    "segment_id": segment_id,
                    "workbook_video_file": row["workbook_video_file"],
                    "video_file": row["video_file"],
                    "video_path": Path(row["video_path"]),
                    "label": row["label"],
                    "class_id": int(row["class_id"]),
                    "split": row["split"],
                    "start_sec": float(
                        row["annotation_start_sec"]
                    ),
                    "end_sec": float(
                        row["annotation_end_sec"]
                    ),
                    "duration_sec": float(
                        row["annotation_duration_sec"]
                    ),
                }
            )

        return annotation_records

    # -------------------------------------------------------------------------

    def _build_annotations_by_label(
        self,
    ) -> dict[int, list[dict[str, Any]]]:
        """Group annotation dictionaries by integer class ID."""

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for annotation in self.annotations:
            grouped[annotation["class_id"]].append(annotation)

        return {
            class_id: grouped[class_id]
            for class_id in sorted(grouped)
        }

    # -------------------------------------------------------------------------

    def _build_class_statistics(
        self,
    ) -> dict[int, dict[str, float | int | str]]:
        """Compute annotation count and duration for every class."""

        statistics: dict[int, dict[str, float | int | str]] = {}

        for class_id, label in ID_TO_LABEL.items():
            class_annotations = self.annotations_by_label.get(
                class_id,
                [],
            )

            statistics[class_id] = {
                "label": label,
                "annotation_count": len(class_annotations),
                "total_duration_sec": float(
                    sum(
                        annotation["duration_sec"]
                        for annotation in class_annotations
                    )
                ),
            }

        return statistics

    # -------------------------------------------------------------------------

    def _validate_class_coverage(self) -> None:
        """Ensure every classifier label has at least one valid annotation."""

        missing_labels = [
            label
            for class_id, label in ID_TO_LABEL.items()
            if not self.annotations_by_label.get(class_id)
        ]

        if missing_labels:
            raise ValueError(
                f"The {self.split!r} split contains no eligible annotations "
                f"for these labels: {missing_labels}. "
                f"Every annotation must be at least "
                f"{self.window_seconds:.1f} seconds long."
            )

    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the fixed number of generated samples in one epoch."""

        return self.samples_per_epoch

    # -------------------------------------------------------------------------

    def start_new_epoch(self, epoch: int) -> None:
        """
        Build the complete balanced sampling plan for one epoch.

        Candidate windows from the same annotation are separated by at least
        ``min_window_spacing_sec`` within a pool cycle. Longer annotations
        contribute more candidates because they contain more valid positions.
        A new cycle is generated only after the available separated coverage
        has been used once.

        Validation and test always use epoch zero, so their plans remain fixed.
        """

        epoch = int(epoch)

        if epoch < 0:
            raise ValueError("epoch must be zero or greater.")

        if self.split != "train":
            epoch = 0

        number_of_classes = len(LABEL_MAP)

        if self.samples_per_epoch % number_of_classes != 0:
            raise ValueError(
                "samples_per_epoch must be divisible by the number of "
                f"classes ({number_of_classes}) for exact balance."
            )

        samples_per_class = self.samples_per_epoch // number_of_classes

        rng = np.random.default_rng(self.base_seed + epoch)

        combined_plan: list[dict[str, Any]] = []

        for class_id in sorted(ID_TO_LABEL):
            combined_plan.extend(
                self._build_class_sampling_plan(
                    class_id=class_id,
                    required_samples=samples_per_class,
                    rng=rng,
                )
            )

        rng.shuffle(combined_plan)

        self.current_epoch = epoch
        self.epoch_sampling_plan = combined_plan
        self.label_schedule = np.asarray(
            [int(sample["class_id"]) for sample in combined_plan],
            dtype=np.int64,
        )

    # -------------------------------------------------------------------------

    def _candidate_starts_for_annotation(
        self,
        annotation: dict[str, Any],
        rng: np.random.Generator,
    ) -> list[float]:
        """Generate one cycle of valid, temporally separated start times."""

        earliest_start = float(annotation["start_sec"])
        latest_start = float(annotation["end_sec"] - self.window_seconds)

        if latest_start < earliest_start:
            raise RuntimeError(
                f"Annotation {annotation['annotation_id']} is shorter than "
                f"the {self.window_seconds:.1f}-second window."
            )

        if np.isclose(earliest_start, latest_start):
            return [earliest_start]

        available_span = latest_start - earliest_start

        number_of_positions = max(
            1,
            int(np.floor(
                available_span / self.min_window_spacing_sec
            )) + 1,
        )

        if number_of_positions == 1:
            return [
                float(rng.uniform(earliest_start, latest_start))
            ]

        occupied_span = (
            number_of_positions - 1
        ) * self.min_window_spacing_sec

        slack = max(0.0, available_span - occupied_span)

        offset = (
            float(rng.uniform(0.0, slack))
            if slack > 0.0
            else 0.0
        )

        starts = (
            earliest_start
            + offset
            + np.arange(number_of_positions, dtype=np.float64)
            * self.min_window_spacing_sec
        )

        starts = np.clip(
            starts,
            earliest_start,
            latest_start,
        )

        return starts.tolist()

    # -------------------------------------------------------------------------

    def _build_class_candidate_pool(
        self,
        class_id: int,
        cycle: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """
        Build one shuffled pool of separated windows for one class.

        Candidate count per annotation is proportional to its usable duration,
        retaining the intent of duration-weighted sampling.
        """

        candidates: list[dict[str, Any]] = []

        for annotation in self.annotations_by_label[class_id]:
            starts = self._candidate_starts_for_annotation(
                annotation=annotation,
                rng=rng,
            )

            rng.shuffle(starts)

            for clip_start_sec in starts:
                candidates.append(
                    {
                        "annotation": annotation,
                        "class_id": class_id,
                        "clip_start_sec": float(clip_start_sec),
                        "sampling_cycle": int(cycle),
                    }
                )

        if not candidates:
            raise RuntimeError(
                f"No candidate windows were generated for class "
                f"{class_id} ({ID_TO_LABEL[class_id]})."
            )

        rng.shuffle(candidates)
        return candidates

    # -------------------------------------------------------------------------

    def _build_class_sampling_plan(
        self,
        class_id: int,
        required_samples: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """
        Consume complete candidate pools before generating another cycle.
        """

        selected: list[dict[str, Any]] = []
        cycle = 0

        while len(selected) < required_samples:
            candidate_pool = self._build_class_candidate_pool(
                class_id=class_id,
                cycle=cycle,
                rng=rng,
            )

            remaining = required_samples - len(selected)
            selected.extend(candidate_pool[:remaining])
            cycle += 1

        return selected

    # -------------------------------------------------------------------------

    def sample_metadata(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Return one precomputed sample from the current epoch plan.

        Precomputing the plan makes temporal-spacing guarantees deterministic
        and safe with multiple DataLoader workers.
        """

        index = int(index)

        if not 0 <= index < self.samples_per_epoch:
            raise IndexError(
                f"Index {index} is outside 0 to "
                f"{self.samples_per_epoch - 1}."
            )

        planned_sample = self.epoch_sampling_plan[index]
        annotation = planned_sample["annotation"]
        class_id = int(planned_sample["class_id"])

        clip_start_sec = float(planned_sample["clip_start_sec"])
        clip_end_sec = clip_start_sec + self.window_seconds

        sample_id = (
            f"E{self.current_epoch + 1:03d}"
            f"_I{index + 1:05d}"
            f"_{annotation['flight_id']}"
            f"_{annotation['segment_id']}"
        )

        return {
            "sample_id": sample_id,
            "epoch": self.current_epoch + 1,
            "epoch_index": index,
            "sampling_cycle": int(planned_sample["sampling_cycle"]),
            "annotation_id": annotation["annotation_id"],
            "flight_id": annotation["flight_id"],
            "segment_id": annotation["segment_id"],
            "video_file": annotation["video_file"],
            "video_path": str(annotation["video_path"]),
            "label": ID_TO_LABEL[class_id],
            "class_id": class_id,
            "annotation_start_sec": float(annotation["start_sec"]),
            "annotation_end_sec": float(annotation["end_sec"]),
            "annotation_duration_sec": float(annotation["duration_sec"]),
            "clip_start_sec": clip_start_sec,
            "clip_end_sec": clip_end_sec,
            "window_seconds": self.window_seconds,
            "frames_per_sample": self.frames_per_sample,
            "min_window_spacing_sec": self.min_window_spacing_sec,
        }

    # -------------------------------------------------------------------------

    def _build_gauss_noise_transform(self):
        """Create a GaussNoise transform compatible with old/new versions."""

        parameters = inspect.signature(
            A.GaussNoise
        ).parameters

        if "std_range" in parameters:
            return A.GaussNoise(
                std_range=(0.04, 0.12),
                mean_range=(0.0, 0.0),
                p=0.50,
            )

        return A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0.0,
            p=0.50,
        )

    # -------------------------------------------------------------------------

    def _build_transform(self):
        """
        Build sequence-consistent preprocessing.

        Every frame is passed to one Albumentations Compose call through
        ``additional_targets``. Consequently, the selected blur/noise settings
        are consistent across the temporal sequence.

        Intentionally excluded:
            - horizontal/vertical flips;
            - brightness, contrast or hue changes;
            - translation, cropping or occlusion.
        """

        additional_targets = {
            f"image{frame_index}": "image"
            for frame_index in range(
                1,
                self.frames_per_sample,
            )
        }

        transforms: list[Any] = [
            A.Resize(
                height=self.image_size,
                width=self.image_size,
            ),
        ]

        if self.augment:
            transforms.extend(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(
                                blur_limit=(7, 15),
                                sigma_limit=(0.1, 3.0),
                                p=0.70,
                            ),
                            A.MotionBlur(
                                blur_limit=(7, 15),
                                p=0.30,
                            ),
                        ],
                        p=0.80,
                    ),
                    self._build_gauss_noise_transform(),
                ]
            )

        transforms.append(
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            )
        )

        compose_arguments: dict[str, Any] = {
            "transforms": transforms,
            "additional_targets": additional_targets,
        }

        # Newer Albumentations releases accept a Compose seed directly.
        if "seed" in inspect.signature(A.Compose).parameters:
            compose_arguments["seed"] = self.base_seed

        return A.Compose(**compose_arguments)

    # -------------------------------------------------------------------------

    def _seed_transform_for_sample(
        self,
        index: int,
    ) -> None:
        """Seed augmentation deterministically where supported."""

        augmentation_seed = int(
            np.random.SeedSequence(
                [
                    self.base_seed,
                    self.current_epoch,
                    int(index),
                    91_337,
                ]
            ).generate_state(1)[0]
        )

        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(
                augmentation_seed
            )

    # -------------------------------------------------------------------------

    def _release_video_cache(self) -> None:
        """Release all OpenCV VideoCapture objects in this process."""

        for cache_entry in self._video_cache.values():
            capture = cache_entry.get("capture")

            if capture is not None:
                capture.release()

        self._video_cache.clear()

    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Public method for explicitly releasing cached video handles."""

        self._release_video_cache()

    # -------------------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort cleanup when a worker or dataset is destroyed."""

        try:
            self._release_video_cache()
        except Exception:
            pass

    # -------------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """
        Remove process-local OpenCV handles before Dataset pickling.

        This is required when DataLoader workers use the spawn start method.
        """

        state = self.__dict__.copy()
        state["_video_cache"] = OrderedDict()
        state["_video_cache_pid"] = None

        return state

    # -------------------------------------------------------------------------

    def _open_video(
        self,
        video_path: str,
    ) -> dict[str, Any]:
        """Open a video and return its capture object and validated metadata."""

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"OpenCV could not open video:\n{video_path}"
            )

        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        frame_count = int(
            round(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
        )

        if not np.isfinite(fps) or fps <= 0.0:
            capture.release()
            raise RuntimeError(
                f"Invalid FPS ({fps}) reported for:\n{video_path}"
            )

        if frame_count <= 0:
            capture.release()
            raise RuntimeError(
                f"Invalid frame count ({frame_count}) reported for:\n"
                f"{video_path}"
            )

        return {
            "capture": capture,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": frame_count / fps,
            "next_frame_index": 0,
        }

    # -------------------------------------------------------------------------

    def _get_video_entry(
        self,
        video_path: str,
    ) -> dict[str, Any]:
        """
        Return an LRU-cached video entry local to the current worker process.
        """

        current_pid = os.getpid()

        if self._video_cache_pid != current_pid:
            self._release_video_cache()
            self._video_cache_pid = current_pid

        cached = self._video_cache.get(
            video_path
        )

        if cached is not None:
            capture = cached["capture"]

            if capture.isOpened():
                self._video_cache.move_to_end(
                    video_path
                )
                return cached

            capture.release()
            del self._video_cache[video_path]

        entry = self._open_video(video_path)
        self._video_cache[video_path] = entry
        self._video_cache.move_to_end(video_path)

        while (
            len(self._video_cache)
            > self.max_cached_videos
        ):
            _, oldest_entry = (
                self._video_cache.popitem(
                    last=False
                )
            )
            oldest_entry["capture"].release()

        return entry

    # -------------------------------------------------------------------------

    @staticmethod
    def _candidate_frame_indices(
        target_index: int,
        minimum_index: int,
        maximum_index: int,
        search_radius: int,
    ) -> list[int]:
        """Return target and nearby fallback indices without duplicates."""

        offsets = [0]

        for distance in range(1, search_radius + 1):
            offsets.extend([-distance, distance])

        candidates: list[int] = []

        for offset in offsets:
            candidate = min(
                max(target_index + offset, minimum_index),
                maximum_index,
            )

            if candidate not in candidates:
                candidates.append(candidate)

        return candidates

    # -------------------------------------------------------------------------

    def _seek_capture(
        self,
        entry: dict[str, Any],
        frame_index: int,
    ) -> None:
        """Seek a cached capture and record its decoder position."""

        capture = entry["capture"]

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            float(frame_index),
        )

        entry["next_frame_index"] = int(
            round(
                capture.get(cv2.CAP_PROP_POS_FRAMES)
            )
        )

    # -------------------------------------------------------------------------

    def _read_current_frame(
        self,
        entry: dict[str, Any],
    ) -> np.ndarray | None:
        """Read the current decoder frame and advance the cached position."""

        capture = entry["capture"]
        success, frame_bgr = capture.read()

        if not success or frame_bgr is None or frame_bgr.size == 0:
            return None

        entry["next_frame_index"] = int(
            round(
                capture.get(cv2.CAP_PROP_POS_FRAMES)
            )
        )

        return cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

    # -------------------------------------------------------------------------

    def _read_frame_at_index(
        self,
        video_path: str,
        target_index: int,
        minimum_index: int,
        maximum_index: int,
    ) -> tuple[np.ndarray | None, int | None]:
        """
        Read one frame, decoding forward when the target is nearby.

        This avoids performing a full random seek for every one of the 25
        requested frames.
        """

        entry = self._get_video_entry(video_path)

        for candidate_index in self._candidate_frame_indices(
            target_index=target_index,
            minimum_index=minimum_index,
            maximum_index=maximum_index,
            search_radius=self.frame_search_radius,
        ):
            next_index = int(
                entry.get("next_frame_index", 0)
            )
            gap = candidate_index - next_index

            if 0 <= gap <= self.sequential_read_tolerance:
                read_failed = False

                for _ in range(gap):
                    if self._read_current_frame(entry) is None:
                        read_failed = True
                        break

                if read_failed:
                    self._seek_capture(entry, candidate_index)
            else:
                self._seek_capture(entry, candidate_index)

            realised_index = int(
                entry.get("next_frame_index", candidate_index)
            )
            frame_rgb = self._read_current_frame(entry)

            if frame_rgb is not None:
                return frame_rgb, realised_index

        # Reopen once if FFmpeg entered a bad decoder state.
        stale = self._video_cache.pop(video_path, None)

        if stale is not None:
            stale["capture"].release()

        entry = self._get_video_entry(video_path)
        self._seek_capture(entry, target_index)

        realised_index = int(
            entry.get("next_frame_index", target_index)
        )
        frame_rgb = self._read_current_frame(entry)

        if frame_rgb is None:
            return None, None

        return frame_rgb, realised_index

    # -------------------------------------------------------------------------

    def _extract_rgb_frames(
        self,
        metadata: dict[str, Any],
    ) -> tuple[list[np.ndarray], list[float], list[int]]:
        """
        Extract uniformly spaced RGB frames from one sampled temporal window.

        ``endpoint=False`` keeps the final timestamp inside the annotation and
        avoids sampling exactly on a neighbouring state boundary.
        """

        video_path = str(metadata["video_path"])
        entry = self._get_video_entry(video_path)

        fps = float(entry["fps"])
        frame_count = int(entry["frame_count"])
        video_duration_sec = float(entry["duration_sec"])

        clip_start_sec = float(metadata["clip_start_sec"])
        clip_end_sec = float(metadata["clip_end_sec"])

        if clip_start_sec < 0.0:
            raise RuntimeError(
                f"Negative clip start for {metadata['sample_id']}: "
                f"{clip_start_sec:.3f} s"
            )

        if clip_end_sec > video_duration_sec + (1.0 / fps):
            raise RuntimeError(
                f"{metadata['sample_id']} ends at {clip_end_sec:.3f} s, "
                f"but {video_path} lasts {video_duration_sec:.3f} s."
            )

        timestamps = np.linspace(
            clip_start_sec,
            clip_end_sec,
            num=self.frames_per_sample,
            endpoint=False,
            dtype=np.float64,
        )

        target_indices = np.floor(
            timestamps * fps + 1e-9
        ).astype(np.int64)

        minimum_index = max(
            0,
            int(np.floor(clip_start_sec * fps)),
        )
        maximum_index = min(
            frame_count - 1,
            int(np.ceil(clip_end_sec * fps)) - 1,
        )

        target_indices = np.clip(
            target_indices,
            minimum_index,
            maximum_index,
        )

        frames: list[np.ndarray] = []
        realised_indices: list[int] = []
        previous_frame: np.ndarray | None = None
        previous_index: int | None = None

        self._seek_capture(
            entry,
            int(target_indices[0]),
        )

        for target_index in target_indices.tolist():
            frame, realised_index = self._read_frame_at_index(
                video_path=video_path,
                target_index=int(target_index),
                minimum_index=minimum_index,
                maximum_index=maximum_index,
            )

            if frame is None:
                if previous_frame is None:
                    raise RuntimeError(
                        f"Could not decode the first required frame for "
                        f"{metadata['sample_id']} from:\n{video_path}"
                    )

                frame = previous_frame.copy()
                realised_index = previous_index

            frames.append(frame)
            previous_frame = frame
            previous_index = int(
                realised_index
                if realised_index is not None
                else target_index
            )
            realised_indices.append(previous_index)

        if len(frames) != self.frames_per_sample:
            raise RuntimeError(
                f"Expected {self.frames_per_sample} frames but extracted "
                f"{len(frames)} for {metadata['sample_id']}."
            )

        return frames, timestamps.tolist(), realised_indices

    # -------------------------------------------------------------------------

    def _transform_frames(
        self,
        frames: list[np.ndarray],
        index: int,
    ) -> torch.Tensor:
        """Apply preprocessing and return float32 [T, C, H, W]."""

        if len(frames) != self.frames_per_sample:
            raise ValueError(
                f"Expected {self.frames_per_sample} input frames; "
                f"received {len(frames)}."
            )

        self._seed_transform_for_sample(index)

        transform_input: dict[str, np.ndarray] = {
            "image": frames[0],
        }

        for frame_index in range(
            1,
            self.frames_per_sample,
        ):
            transform_input[
                f"image{frame_index}"
            ] = frames[frame_index]

        transformed = self.transform(
            **transform_input
        )

        transformed_frames = [
            transformed["image"]
        ]

        transformed_frames.extend(
            transformed[
                f"image{frame_index}"
            ]
            for frame_index in range(
                1,
                self.frames_per_sample,
            )
        )

        frame_array = np.stack(
            transformed_frames,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        # Albumentations uses [T, H, W, C]; PyTorch models use [T, C, H, W].
        frame_tensor = torch.from_numpy(
            np.ascontiguousarray(
                frame_array.transpose(
                    0,
                    3,
                    1,
                    2,
                )
            )
        )

        expected_shape = (
            self.frames_per_sample,
            3,
            self.image_size,
            self.image_size,
        )

        if tuple(frame_tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected tensor shape {tuple(frame_tensor.shape)}; "
                f"expected {expected_shape}."
            )

        return frame_tensor

    # -------------------------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Generate, decode and preprocess one online temporal vision sample.

        Returns:
            frames:
                Float32 tensor [T, C, H, W], ImageNet-normalised.
            label:
                Scalar int64 PyTorch tensor.
            metadata:
                Sampling and video metadata, when return_metadata=True.
        """

        metadata = self.sample_metadata(
            index
        )

        (
            frames,
            sampled_timestamps_sec,
            sampled_frame_indices,
        ) = self._extract_rgb_frames(
            metadata
        )

        frame_tensor = self._transform_frames(
            frames=frames,
            index=int(index),
        )

        label_tensor = torch.tensor(
            metadata["class_id"],
            dtype=torch.long,
        )

        result: dict[str, Any] = {
            "frames": frame_tensor,
            "label": label_tensor,
        }

        if self.return_metadata:
            metadata = metadata.copy()
            metadata["sampled_timestamps_sec"] = (
                sampled_timestamps_sec
            )
            metadata["sampled_frame_indices"] = (
                sampled_frame_indices
            )
            result["metadata"] = metadata

        return result

    # -------------------------------------------------------------------------

    def epoch_class_counts(self) -> dict[str, int]:
        """Return exact class totals in the current epoch schedule."""

        class_ids, counts = np.unique(
            self.label_schedule,
            return_counts=True,
        )

        return {
            ID_TO_LABEL[int(class_id)]: int(count)
            for class_id, count in zip(
                class_ids,
                counts,
            )
        }

    # -------------------------------------------------------------------------

    def export_epoch_sampling_log(
        self,
        output_file: Path | str,
    ) -> Path:
        """Export all sampled windows for the current epoch to CSV."""

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = [
            self.sample_metadata(index)
            for index in range(
                self.samples_per_epoch
            )
        ]

        pd.DataFrame(records).to_csv(
            output_file,
            index=False,
        )

        return output_file

    # -------------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print diagnostics for the prepared annotation database."""

        print("\n" + "=" * 72)
        print("MACE ONLINE TEMPORAL DATASET — FINAL")
        print("=" * 72)

        print(f"Split               : {self.split}")
        print(f"Annotation workbook : {self.annotation_file}")
        print(f"Flights root        : {self.flights_root}")
        print(f"Samples per epoch   : {self.samples_per_epoch}")
        print(f"Window length       : {self.window_seconds:.1f} s")
        print(f"Frames per sample   : {self.frames_per_sample}")
        print(
            f"Minimum start spacing: "
            f"{self.min_window_spacing_sec:.2f} s"
        )
        print(f"Image size          : {self.image_size} x {self.image_size}")
        print(f"Augmentation        : {self.augment}")
        print(f"Return metadata     : {self.return_metadata}")
        print(f"Video cache limit   : {self.max_cached_videos}")
        print(f"OpenCV threads      : {cv2.getNumThreads()}")
        print(f"PyTorch CPU threads : {torch.get_num_threads()}")
        print(f"Base seed           : {self.base_seed}")
        print(f"Current epoch       : {self.current_epoch + 1}")
        print(f"Eligible annotations: {len(self.annotations)}")

        print("\nBalanced samples in current epoch")
        print("-" * 72)

        for label, count in self.epoch_class_counts().items():
            print(f"{label:<11}: {count}")

        print("\nPer-class annotation statistics")
        print("-" * 72)

        for class_id in sorted(ID_TO_LABEL):
            stats = self.class_statistics[class_id]

            print(
                f"Class {class_id} "
                f"({stats['label']:<11}) | "
                f"annotations: {stats['annotation_count']:>3} | "
                f"duration: {stats['total_duration_sec']:>8.1f} s"
            )

        print("\nSource videos")
        print("-" * 72)

        unique_videos = sorted(
            {
                str(annotation["video_path"])
                for annotation in self.annotations
            }
        )

        for video_path in unique_videos:
            print(video_path)

        if not self.excluded_short_annotations.empty:
            print(
                "\nAnnotations excluded because they are shorter than "
                f"{self.window_seconds:.1f} seconds"
            )
            print("-" * 72)

            display_columns = [
                "flight_id",
                "segment_id",
                "label",
                "annotation_start_sec",
                "annotation_end_sec",
                "annotation_duration_sec",
            ]

            print(
                self.excluded_short_annotations[
                    display_columns
                ].to_string(index=False)
            )

        print("\nFirst five eligible annotations")
        print("-" * 72)

        for annotation in self.annotations[:5]:
            print(
                f"{annotation['annotation_id']}: "
                f"{annotation['label']} | "
                f"{annotation['start_sec']:.2f}–"
                f"{annotation['end_sec']:.2f} s | "
                f"{annotation['video_path']}"
            )

        print("=" * 72)
        print("✓ Final HPC-safe online frame dataset created successfully.")
        print("=" * 72)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    annotation_file = (
        project_root
        / "data"
        / "annotations"
        / "MACE_Temporal_label_complete.xlsx"
    )

    split_file = (
        project_root
        / "data"
        / "annotations"
        / "flight_splits.csv"
    )

    flights_root = (
        project_root
        / "Flights"
    )

    train_dataset = OnlineTemporalVisionDataset(
        annotation_file=annotation_file,
        flights_root=flights_root,
        split_file=split_file,
        split="train",
        samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        frames_per_sample=DEFAULT_FRAMES_PER_SAMPLE,
        augment=True,
        base_seed=DEFAULT_BASE_SEED,
        image_size=DEFAULT_IMAGE_SIZE,
        max_cached_videos=DEFAULT_MAX_CACHED_VIDEOS,
        frame_search_radius=DEFAULT_FRAME_SEARCH_RADIUS,
        sequential_read_tolerance=DEFAULT_SEQUENTIAL_READ_TOLERANCE,
        min_window_spacing_sec=DEFAULT_MIN_WINDOW_SPACING_SEC,
        return_metadata=True,
        sheet_name="Sheet1",
        verify_videos=True,
    )

    train_dataset.print_summary()

    print("\nExtracting two complete online samples")
    print("-" * 72)

    try:
        for index in range(2):
            sample = train_dataset[index]
            metadata = sample["metadata"]

            print(
                f"{metadata['sample_id']} | "
                f"{metadata['label']:<11} | "
                f"{metadata['annotation_id']} | "
                f"{metadata['clip_start_sec']:.3f}-"
                f"{metadata['clip_end_sec']:.3f} s"
            )

            print(
                f"  frames shape : {tuple(sample['frames'].shape)}"
            )
            print(
                f"  frames dtype : {sample['frames'].dtype}"
            )
            print(
                f"  label tensor : {sample['label'].item()}"
            )
            print(
                f"  frame range  : "
                f"{metadata['sampled_frame_indices'][0]}-"
                f"{metadata['sampled_frame_indices'][-1]}"
            )
            print(
                f"  finite values: "
                f"{bool(torch.isfinite(sample['frames']).all())}"
            )

        train_dataset.start_new_epoch(epoch=1)

        print("\nBalanced class counts for epoch 2")
        print("-" * 72)
        print(train_dataset.epoch_class_counts())

        epoch_two_sample = train_dataset[0]

        print(
            "\nEpoch 2 sample 0: "
            f"{epoch_two_sample['metadata']['sample_id']} | "
            f"{epoch_two_sample['metadata']['label']} | "
            f"shape={tuple(epoch_two_sample['frames'].shape)}"
        )

        print(
            "\n✓ HPC-safe OpenCV extraction, augmentation, "
            "normalisation and tensor conversion completed successfully."
        )

    finally:
        train_dataset.close()