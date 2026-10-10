#!/usr/bin/env python3
"""
===============================================================================
generate_vision_dataset.py
===============================================================================

Project:
    MACE - Multimodal Spatiotemporal UAV Environmental Perception

Purpose:
    Generate the vision-only dataset used for the initial ViT baseline.

Pipeline:
    sample_manifest.csv
            │
            ▼
    Original GoPro videos
            │
            ▼
    Extract 25 uniformly sampled frames
            │
            ▼
    vision_dataset/
        ├── frames/
        ├── dataset_manifest.csv
        └── generation_report.txt

Author:
    Ikechukwu Ofodile
    University of Bristol
    Practice-Oriented AI CDT

===============================================================================
"""

from pathlib import Path
import argparse
import logging
import sys
import shutil
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

IMAGE_SIZE = 224
FRAMES_PER_SAMPLE = 25
IMAGE_EXTENSION = ".jpg"

# OpenCV JPEG quality (0-100)
JPEG_QUALITY = 95

# Folder names
FRAME_FOLDER = "frames"
DATASET_MANIFEST = "dataset_manifest.csv"
REPORT_FILE = "generation_report.txt"

SUPPORTED_VIDEO_EXTENSIONS = [
    ".mp4",
    ".MP4",
    ".mov",
    ".MOV"
]


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_arguments():
    """
    Parse command-line arguments.

    Example
    -------
    python generate_vision_dataset.py \
        --sample_manifest sample_manifest.csv \
        --video_metadata video_metadata_verified.csv \
        --flight_splits flight_splits.csv \
        --videos_root Flights \
        --output vision_dataset
    """

    parser = argparse.ArgumentParser(
        description="Generate MACE Vision Dataset"
    )

    parser.add_argument(
        "--sample_manifest",
        required=True,
        type=Path,
        help="Path to sample_manifest.csv"
    )

    parser.add_argument(
        "--video_metadata",
        required=True,
        type=Path,
        help="Path to video_metadata_verified.csv"
    )

    parser.add_argument(
        "--flight_splits",
        required=True,
        type=Path,
        help="Path to flight_splits.csv"
    )

    parser.add_argument(
        "--videos_root",
        required=True,
        type=Path,
        help="Root folder containing all flight videos"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output vision dataset directory"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory"
    )

    return parser.parse_args()


# =============================================================================
# Logging
# =============================================================================

def setup_logger(output_dir: Path):
    """
    Configure console and file logging.
    """

    log_file = output_dir / "generation.log"

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    logfile = logging.FileHandler(log_file)
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

    return logger


# =============================================================================
# Directory utilities
# =============================================================================

def prepare_output_directory(output_dir: Path, overwrite: bool):
    """
    Prepare output directory.

    Creates:

        output/
            frames/

    Parameters
    ----------
    overwrite : bool
        Delete previous output if it already exists.
    """

    if output_dir.exists():

        if overwrite:
            shutil.rmtree(output_dir)

        else:
            raise FileExistsError(
                f"{output_dir} already exists.\n"
                "Use --overwrite if you wish to replace it."
            )

    output_dir.mkdir(parents=True)

    (output_dir / FRAME_FOLDER).mkdir()


# =============================================================================
# CSV loading
# =============================================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV with basic validation.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


# =============================================================================
# Report helper
# =============================================================================

class DatasetReport:

    """
    Stores statistics during dataset generation.
    """

    def __init__(self):

        self.total_samples = 0
        self.total_frames = 0

        self.skipped_samples = 0

        self.missing_videos = 0

        self.failed_extractions = 0

    def summary(self):

        return {
            "samples": self.total_samples,
            "frames": self.total_frames,
            "skipped": self.skipped_samples,
            "missing": self.missing_videos,
            "failed": self.failed_extractions,
        }


# =============================================================================
# End of Module 1
# =============================================================================




# =============================================================================
# Module 2
# Data Loading and Video Discovery
# =============================================================================

def load_video_metadata(video_metadata_path: Path) -> pd.DataFrame:
    """
    Load verified video metadata.

    Returns
    -------
    DataFrame indexed by
        (flight_id, video_file)

    Required columns:
        flight_id
        video_file
        fps
        frame_count
        actual_duration_sec
    """

    df = load_csv(video_metadata_path)

    required_columns = [
        "flight_id",
        "video_file",
        "fps",
        "frame_count",
        "actual_duration_sec"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in video metadata:\n{missing}"
        )

    return df.set_index(["flight_id", "video_file"])


# -----------------------------------------------------------------------------


def load_sample_manifest(sample_manifest_path: Path) -> pd.DataFrame:
    """
    Load sample manifest.

    Expected columns

        sample_id
        flight_id
        video_file
        label
        clip_start_sec
        clip_end_sec
    """

    df = load_csv(sample_manifest_path)

    required_columns = [
        "sample_id",
        "flight_id",
        "video_file",
        "label",
        "clip_start_sec",
        "clip_end_sec"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in sample manifest:\n{missing}"
        )

    return df


# -----------------------------------------------------------------------------


def load_flight_splits(split_path: Path) -> pd.DataFrame:
    """
    Load train / validation / test split.

    Expected columns

        flight_id
        split
    """

    df = load_csv(split_path)

    required_columns = [
        "flight_id",
        "split"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in flight split:\n{missing}"
        )

    return df


# -----------------------------------------------------------------------------


def merge_sample_information(
    samples: pd.DataFrame,
    splits: pd.DataFrame
) -> pd.DataFrame:
    """
    Add train/validation/test split to each sample.

    If sample_manifest already contains a split column,
    retain it and do not merge a duplicate.
    """

    samples = samples.copy()

    if "split" in samples.columns:
        if samples["split"].isna().any():
            raise ValueError(
                "sample_manifest.csv contains missing split values."
            )

        return samples

    merged = samples.merge(
        splits[["flight_id", "split"]],
        on="flight_id",
        how="left",
        validate="many_to_one"
    )

    if merged["split"].isna().any():
        missing = merged.loc[
            merged["split"].isna(),
            "flight_id"
        ].unique()

        raise ValueError(
            f"Flights missing from split file: {missing}"
        )

    return merged

# -----------------------------------------------------------------------------


def find_video_file(
    videos_root: Path,
    flight_id: str,
    video_file: str
) -> Optional[Path]:

    candidate = videos_root / flight_id / video_file

    if candidate.exists():
        return candidate.resolve()

    return None

# -----------------------------------------------------------------------------


def validate_video_dataset(
    samples: pd.DataFrame,
    videos_root: Path,
    report: DatasetReport,
    logger
):
    """
    Ensure every sample has
    a corresponding video.
    """

    logger.info("Checking video availability...")

    missing = []

    for row in samples.itertuples():

        path = find_video_file(
            videos_root,
            row.flight_id,
            row.video_file
        )

        if path is None:

            report.missing_videos += 1

            missing.append(
                (
                    row.flight_id,
                    row.video_file
                )
            )

    if missing:

        logger.warning(
            f"{len(missing)} missing videos found."
        )

        for flight, video in missing:

            logger.warning(
                f"Missing: {flight} / {video}"
            )

    else:

        logger.info(
            "All videos successfully located."
        )


# -----------------------------------------------------------------------------


def attach_video_paths(
    samples: pd.DataFrame,
    videos_root: Path
) -> pd.DataFrame:
    """
    Attach the verified video path to every sample.
    """

    samples = samples.copy()

    paths = []

    for row in samples.itertuples():

        path = find_video_file(
        videos_root,
        row.flight_id,
        row.video_file
        )

        paths.append(path)

    samples["resolved_video_path"] = paths

    return samples


# =============================================================================
# End Module 2
# =============================================================================

# =============================================================================
# Module 3
# Frame Sampling and Extraction
# =============================================================================

def compute_frame_indices(
    clip_start_sec: float,
    clip_end_sec: float,
    fps: float,
    frames_per_sample: int = FRAMES_PER_SAMPLE
):
    """
    Compute uniformly sampled frame indices for one clip.

    Parameters
    ----------
    clip_start_sec : float
    clip_end_sec : float
    fps : float

    Returns
    -------
    numpy.ndarray (int)

    Example
    -------
    5-second clip @ 30 fps

        150 frames

    ↓

        25 sampled frames

        [0,6,12,...,144]
    """

    start_frame = int(round(clip_start_sec * fps))
    end_frame = int(round(clip_end_sec * fps))

    total_frames = end_frame - start_frame

    if total_frames < frames_per_sample:
        raise ValueError(
            f"Clip contains only {total_frames} frames."
        )

    indices = np.linspace(
        start_frame,
        end_frame - 1,
        frames_per_sample,
        dtype=np.int32
    )

    return indices


# -----------------------------------------------------------------------------


def extract_sample_frames(
    video_path,
    frame_indices
):
    """
    Efficient sequential extraction.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    images = []

    target = set(frame_indices.tolist())

    current = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if current in target:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            images.append(frame)

            if len(images) == len(frame_indices):
                break

        current += 1

    cap.release()

    if len(images) != len(frame_indices):

        raise RuntimeError(
            "Unable to extract all requested frames."
        )

    return images


# -----------------------------------------------------------------------------


def save_sample_frames(
    images,
    frame_indices,
    fps,
    output_directory: Path
):
    """
    Save extracted frames.

    Naming convention

        000.jpg
        001.jpg
        ...
        024.jpg
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, image in enumerate(images):

        filename = output_directory / f"{i:03d}.jpg"

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            ),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )
    
    frame_records = []

    clip_start_frame = frame_indices[0]

    for i, frame_number in enumerate(frame_indices):

        timestamp = frame_number / fps

        relative_time = (frame_number - clip_start_frame) / fps

        frame_records.append({

            "frame_idx": i,

            "image_name": f"{i:03d}.jpg",

            "frame_number": int(frame_number),

            "timestamp_sec": round(timestamp, 6),

            "relative_time_sec": round(relative_time, 6)

        })

    frame_df = pd.DataFrame(frame_records)

    frame_df.to_csv(
        output_directory / "frames.csv",
        index=False
    )
    


# -----------------------------------------------------------------------------


def process_single_sample(
    sample,
    video_metadata,
    output_root: Path,
    report: DatasetReport,
    logger
):
    """
    Process one machine-learning sample.

    Workflow

        sample

            ↓

        compute frame indices

            ↓

        extract frames

            ↓

        save images

            ↓

        return dataset manifest entry
    """

    metadata = video_metadata.loc[
        (
            sample.flight_id,
            sample.video_file
        )
    ]

    fps = float(metadata["fps"])

    frame_indices = compute_frame_indices(

        sample.clip_start_sec,

        sample.clip_end_sec,

        fps
    )

    images = extract_sample_frames(

        sample.resolved_video_path,

        frame_indices
    )

    sample_folder = (
        output_root /
        FRAME_FOLDER /
        sample.sample_id
    )

    save_sample_frames(
        images,
        frame_indices,
        fps,
        sample_folder
    )

    report.total_samples += 1
    report.total_frames += len(images)

    return {

        "sample_id":
            sample.sample_id,

        "flight_id":
            sample.flight_id,

        "video_file":
            sample.video_file,

        "label":
            sample.label,

        "split":
            sample.split,

        "frame_dir":
            str(
                Path(FRAME_FOLDER)
                /
                sample.sample_id
            ),

        "clip_start_sec":
            sample.clip_start_sec,

        "clip_end_sec":
            sample.clip_end_sec,

        "fps":
            fps,

        "num_frames":
            len(images)

    }


# -----------------------------------------------------------------------------


def generate_dataset(
    samples,
    video_metadata,
    output_root,
    report,
    logger
):
    """
    Main extraction loop.
    """

    logger.info("Generating vision dataset...")

    manifest_rows = []

    for sample in tqdm(
        samples.itertuples(),
        total=len(samples),
        desc="Extracting"
    ):

        try:

            row = process_single_sample(

                sample,

                video_metadata,

                output_root,

                report,

                logger
            )

            manifest_rows.append(row)

        except Exception as e:

            report.failed_extractions += 1

            logger.warning(

                f"{sample.sample_id} skipped"

                f"\nReason: {e}"

            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    return manifest


# =============================================================================
# End Module 3
# =============================================================================

# =============================================================================
# Module 4
# Main Execution
# =============================================================================

def write_generation_report(
    report: DatasetReport,
    output_dir: Path,
    manifest: pd.DataFrame,
    logger
):
    """
    Write generation statistics to disk.
    """

    report_file = output_dir / REPORT_FILE

    split_counts = (
        manifest["split"]
        .value_counts()
        .to_dict()
    )

    label_counts = (
        manifest["label"]
        .value_counts()
        .to_dict()
    )

    with open(report_file, "w") as f:

        f.write("=" * 65 + "\n")
        f.write("MACE Vision Dataset Generation Report\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"Samples generated      : {report.total_samples}\n")
        f.write(f"Frames extracted       : {report.total_frames}\n")
        f.write(f"Skipped samples        : {report.skipped_samples}\n")
        f.write(f"Missing videos         : {report.missing_videos}\n")
        f.write(f"Failed extractions     : {report.failed_extractions}\n\n")

        f.write("-" * 65 + "\n")
        f.write("Split Distribution\n")
        f.write("-" * 65 + "\n")

        for split, count in split_counts.items():
            f.write(f"{split:<12} : {count}\n")

        f.write("\n")

        f.write("-" * 65 + "\n")
        f.write("Class Distribution\n")
        f.write("-" * 65 + "\n")

        for label, count in label_counts.items():
            f.write(f"{label:<15} : {count}\n")

        f.write("\n")
        f.write("=" * 65 + "\n")

    logger.info(f"Generation report saved to:\n{report_file}")


# -----------------------------------------------------------------------------


def save_dataset_manifest(
    manifest: pd.DataFrame,
    output_dir: Path,
    logger
):
    """
    Save dataset manifest.
    """

    manifest_path = output_dir / DATASET_MANIFEST

    manifest.to_csv(
        manifest_path,
        index=False
    )

    logger.info(f"Dataset manifest saved:\n{manifest_path}")


# -----------------------------------------------------------------------------


def print_summary(
    report: DatasetReport,
    manifest: pd.DataFrame
):
    """
    Print final summary.
    """

    print("\n")
    print("=" * 70)
    print("MACE Vision Dataset Generation Complete")
    print("=" * 70)

    print(f"Samples generated : {report.total_samples}")
    print(f"Frames extracted  : {report.total_frames}")
    print(f"Missing videos    : {report.missing_videos}")
    print(f"Failed samples    : {report.failed_extractions}")

    print("\nSplit Summary")

    print(
        manifest["split"]
        .value_counts()
    )

    print("\nLabel Summary")

    print(
        manifest["label"]
        .value_counts()
    )

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_arguments()

    prepare_output_directory(
        args.output,
        args.overwrite
    )

    logger = setup_logger(
        args.output
    )

    logger.info("Loading CSV files...")

    samples = load_sample_manifest(
        args.sample_manifest
    )

    splits = load_flight_splits(
        args.flight_splits
    )

    video_metadata = load_video_metadata(
        args.video_metadata
    )

    video_paths = (
        video_metadata
        .reset_index()[["flight_id", "video_path"]]
        .drop_duplicates(subset=["flight_id"])
    )
    # Remove the old column if it already exists
    samples = samples.drop(columns=["video_path"], errors="ignore")

    samples = samples.merge(
        video_paths,
        on="flight_id",
        how="left",
        validate="many_to_one"
    )
    '''# ------------------------------------------------------------------
    # Use the standardised dataset filenames (FL008.MP4, FL012.MP4, ...)
    # instead of the original GoPro filenames.
    # ------------------------------------------------------------------

    samples["video_file"] = samples["flight_id"] + ".MP4"

    video_metadata["video_file"] = video_metadata["flight_id"] + ".MP4"
    '''

    report = DatasetReport()

    logger.info("Merging split information...")

    samples = merge_sample_information(
        samples,
        splits
    )

    logger.info("Locating videos...")

    print(samples.filter(regex="video").columns.tolist())

    print(samples.columns.tolist())
    print(samples.filter(regex="video|flight").columns.tolist())

    samples = attach_video_paths(
        samples,
        args.videos_root
    )

    validate_video_dataset(
        samples,
        args.videos_root,
        report,
        logger
    )

    logger.info("Starting frame extraction...")

    dataset_manifest = generate_dataset(
        samples,
        video_metadata,
        args.output,
        report,
        logger
    )

    save_dataset_manifest(
        dataset_manifest,
        args.output,
        logger
    )

    write_generation_report(
        report,
        args.output,
        dataset_manifest,
        logger
    )

    print_summary(
        report,
        dataset_manifest
    )

    logger.info("Finished successfully.")


# =============================================================================

if __name__ == "__main__":

    main()