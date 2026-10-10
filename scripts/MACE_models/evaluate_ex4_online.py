#!/usr/bin/env python3
"""
===============================================================================
evaluate_ex4_online.py
===============================================================================

Deterministic annotation-level evaluation for the MACE vision-only experiment.

Evaluation protocol
-------------------
1. Use only annotations belonging to the selected validation or test flights.
2. For every eligible annotation, enumerate every complete 5-second window at
   a fixed deterministic stride.
3. Decode 25 uniformly spaced RGB frames from each window.
4. Run the trained vision model once per window.
5. Average raw logits across all windows belonging to the same annotation.
6. Produce exactly one final prediction per annotation.

Outputs
-------
<output_dir>/
    run_config.json
    evaluation.log
    window_predictions.csv
    annotation_predictions.csv
    metrics.json
    metrics.csv
    classification_report.txt
    classification_report.csv
    confusion_matrix_counts.csv
    confusion_matrix_normalized.csv
    confusion_matrix_counts.png
    confusion_matrix_normalized.png
    tensorboard/
        events.out.tfevents...

The script first imports VisionClassifier from train_ex4_online.py so that the
evaluation model exactly matches the training code. A ViT-B/16 temporal-mean
fallback is provided for portability, but the imported training architecture is
preferred.

Expected annotation columns
---------------------------
The loader is case-insensitive and accepts common variants of:
    flight_id / Flight_ID
    segment_id / Segment_ID
    segment_uid / annotation_id
    start_sec / Start_sec
    end_sec / End_sec
    label / state / Label
    split / Split
    video_file / Video_File

The annotation file may be CSV, XLSX or XLS.

Example
-------
python evaluate_ex4_online.py \
    --annotations ../../data/annotations/MACE_Temporal_labels_complete.xlsx \
    --video-root ../../Flights \
    --checkpoint ../../experiments/2026-07-31_12-00-00/best_model.pt \
    --split test \
    --stride-sec 1.0 \
    --output-dir ../../experiments/2026-07-31_12-00-00/evaluation_test

To select flights explicitly:
python evaluate_ex4_online.py ... --split test \
    --flights FL008 FL016 FL021

Dependencies
------------
torch, torchvision, opencv-python, pandas, numpy, scikit-learn, matplotlib,
openpyxl, tensorboard
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import v2 as T


# =============================================================================
# Constants
# =============================================================================

LABEL_MAP: dict[str, int] = {
    "approaching": 0,
    "entering": 1,
    "inside": 2,
    "exiting": 3,
}
ID_TO_LABEL: dict[int, str] = {value: key for key, value in LABEL_MAP.items()}
CLASS_IDS = list(range(len(LABEL_MAP)))
CLASS_NAMES = [ID_TO_LABEL[index] for index in CLASS_IDS]

DEFAULT_WINDOW_SEC = 5.0
DEFAULT_STRIDE_SEC = 1.0
DEFAULT_FRAMES_PER_WINDOW = 25
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
EPSILON = 1e-8


# =============================================================================
# Configuration records
# =============================================================================

@dataclass(frozen=True)
class WindowRecord:
    annotation_index: int
    annotation_id: str
    segment_id: str
    flight_id: str
    video_path: str
    label_name: str
    label_id: int
    annotation_start_sec: float
    annotation_end_sec: float
    annotation_duration_sec: float
    window_index: int
    window_start_sec: float
    window_end_sec: float


@dataclass
class EvaluationCounters:
    total_rows_read: int = 0
    rows_after_split_filter: int = 0
    rows_after_flight_filter: int = 0
    eligible_annotations: int = 0
    excluded_short_annotations: int = 0
    excluded_unknown_labels: int = 0
    excluded_missing_videos: int = 0
    generated_windows: int = 0
    evaluated_windows: int = 0
    failed_windows: int = 0


# =============================================================================
# Reproducibility
# =============================================================================

def set_deterministic(seed: int) -> None:
    """Configure deterministic evaluation as far as supported by PyTorch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # TF32 can introduce small hardware-dependent numerical differences.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False


# =============================================================================
# Logging
# =============================================================================

def configure_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mace_evaluation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(output_dir / "evaluation.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# =============================================================================
# Annotation loading and normalization
# =============================================================================

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "flight_id": ("flight_id", "flight", "flightid"),
    "segment_id": ("segment_id", "segment", "segmentid"),
    "annotation_id": (
        "annotation_id",
        "annotation_uid",
        "segment_uid",
        "segmentuid",
        "uid",
    ),
    "start_sec": (
        "start_sec",
        "start_seconds",
        "start_time_sec",
        "start",
        "starttime",
    ),
    "end_sec": (
        "end_sec",
        "end_seconds",
        "end_time_sec",
        "end",
        "endtime",
    ),
    "label": ("label", "state", "class", "interaction_state"),
    "split": ("split", "dataset_split", "set"),
    "video_file": (
        "video_file",
        "video",
        "video_filename",
        "filename",
        "file_name",
    ),
}


def canonical_column_name(name: Any) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    canonical_to_original = {
        canonical_column_name(column): str(column) for column in frame.columns
    }

    resolved: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            canonical_alias = canonical_column_name(alias)
            if canonical_alias in canonical_to_original:
                resolved[target] = canonical_to_original[canonical_alias]
                break

    required = ("flight_id", "start_sec", "end_sec", "label")
    missing = [name for name in required if name not in resolved]
    if missing:
        raise ValueError(
            "Annotation file is missing required columns: "
            f"{missing}. Available columns: {list(frame.columns)}"
        )
    return resolved


def parse_seconds(value: Any) -> float:
    """Parse numeric seconds or H:MM:SS / MM:SS strings."""
    if pd.isna(value):
        raise ValueError("time value is empty")

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("time value is empty")

    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"cannot parse time value {value!r}")

    numbers = [float(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60.0 + seconds

    hours, minutes, seconds = numbers
    return hours * 3600.0 + minutes * 60.0 + seconds


def normalize_label(value: Any) -> str:
    return canonical_column_name(value)


def read_annotations(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        actual_sheet: str | int = 0 if sheet_name is None else sheet_name
        return pd.read_excel(path, sheet_name=actual_sheet)
    raise ValueError(
        f"Unsupported annotation format {suffix!r}; use CSV, XLSX or XLS."
    )


def normalize_annotations(
    raw: pd.DataFrame,
    split: str,
    selected_flights: set[str] | None,
    counters: EvaluationCounters,
    logger: logging.Logger,
) -> pd.DataFrame:
    counters.total_rows_read = len(raw)
    columns = resolve_columns(raw)

    output = pd.DataFrame()
    output["flight_id"] = raw[columns["flight_id"]].astype(str).str.strip()
    output["start_sec"] = raw[columns["start_sec"]].map(parse_seconds)
    output["end_sec"] = raw[columns["end_sec"]].map(parse_seconds)
    output["label_name"] = raw[columns["label"]].map(normalize_label)

    if "segment_id" in columns:
        output["segment_id"] = raw[columns["segment_id"]].astype(str).str.strip()
    else:
        output["segment_id"] = [
            f"S{index + 1:03d}" for index in range(len(output))
        ]

    if "annotation_id" in columns:
        supplied_ids = raw[columns["annotation_id"]].astype(str).str.strip()
        output["annotation_id"] = supplied_ids
    else:
        output["annotation_id"] = (
            output["flight_id"]
            + "_"
            + output["segment_id"]
            + "_"
            + output.index.astype(str)
        )

    if "video_file" in columns:
        output["video_file"] = raw[columns["video_file"]].astype(str).str.strip()
    else:
        output["video_file"] = output["flight_id"] + ".MP4"

    if "split" in columns:
        output["split"] = (
            raw[columns["split"]].astype(str).str.strip().str.lower()
        )
        output = output[output["split"] == split.lower()].copy()
    else:
        logger.warning(
            "No split column found. Flight selection must therefore be supplied "
            "with --flights or --flights-file to guarantee held-out evaluation."
        )
        if selected_flights is None:
            raise ValueError(
                "The annotation file has no split column and no explicit flight "
                "selection was supplied. Refusing to evaluate all flights."
            )

    counters.rows_after_split_filter = len(output)

    if selected_flights is not None:
        output = output[output["flight_id"].isin(selected_flights)].copy()

    counters.rows_after_flight_filter = len(output)

    unknown_mask = ~output["label_name"].isin(LABEL_MAP)
    counters.excluded_unknown_labels = int(unknown_mask.sum())
    if unknown_mask.any():
        unknown = sorted(output.loc[unknown_mask, "label_name"].unique())
        logger.warning("Excluding unsupported labels: %s", unknown)
        output = output.loc[~unknown_mask].copy()

    output["label_id"] = output["label_name"].map(LABEL_MAP).astype(int)
    output["annotation_duration_sec"] = output["end_sec"] - output["start_sec"]

    invalid = (
        ~np.isfinite(output["start_sec"])
        | ~np.isfinite(output["end_sec"])
        | (output["start_sec"] < 0.0)
        | (output["end_sec"] <= output["start_sec"])
    )
    if invalid.any():
        examples = output.loc[
            invalid, ["annotation_id", "start_sec", "end_sec"]
        ].head(10)
        raise ValueError(
            "Invalid annotation times were found:\n"
            + examples.to_string(index=False)
        )

    duplicate_ids = output["annotation_id"].duplicated(keep=False)
    if duplicate_ids.any():
        # Preserve supplied IDs while making grouping unambiguous.
        output.loc[duplicate_ids, "annotation_id"] = (
            output.loc[duplicate_ids, "annotation_id"]
            + "__row"
            + output.loc[duplicate_ids].index.astype(str)
        )

    output = output.sort_values(
        ["flight_id", "start_sec", "end_sec", "annotation_id"]
    ).reset_index(drop=True)
    output["annotation_index"] = np.arange(len(output), dtype=int)
    return output


# =============================================================================
# Video and deterministic window construction
# =============================================================================

VIDEO_EXTENSIONS = (".MP4", ".mp4", ".MOV", ".mov", ".AVI", ".avi", ".MKV", ".mkv")


def resolve_video_path(video_root: Path, flight_id: str, video_file: str) -> Path | None:
    requested = Path(video_file)
    candidates: list[Path] = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend(
            [
                video_root / requested,
                video_root / flight_id / requested.name,
            ]
        )

    for extension in VIDEO_EXTENSIONS:
        candidates.extend(
            [
                video_root / f"{flight_id}{extension}",
                video_root / flight_id / f"{flight_id}{extension}",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate.resolve()

    # Final recursive fallback, restricted to the exact stem.
    matches: list[Path] = []
    for extension in VIDEO_EXTENSIONS:
        matches.extend(video_root.rglob(f"{flight_id}{extension}"))
    return matches[0].resolve() if matches else None


def deterministic_window_starts(
    annotation_start: float,
    annotation_end: float,
    window_sec: float,
    stride_sec: float,
) -> list[float]:
    """
    Return all complete fixed-stride windows fully contained in an annotation.

    No random sampling and no forced final partial/off-grid window are used.
    """
    latest_start = annotation_end - window_sec
    if latest_start < annotation_start - EPSILON:
        return []

    count = int(math.floor((latest_start - annotation_start + EPSILON) / stride_sec)) + 1
    return [annotation_start + index * stride_sec for index in range(count)]


def build_window_records(
    annotations: pd.DataFrame,
    video_root: Path,
    window_sec: float,
    stride_sec: float,
    counters: EvaluationCounters,
    logger: logging.Logger,
) -> tuple[list[WindowRecord], pd.DataFrame]:
    records: list[WindowRecord] = []
    retained_annotation_rows: list[dict[str, Any]] = []
    video_cache: dict[tuple[str, str], Path | None] = {}

    for row in annotations.itertuples(index=False):
        starts = deterministic_window_starts(
            float(row.start_sec),
            float(row.end_sec),
            window_sec,
            stride_sec,
        )
        if not starts:
            counters.excluded_short_annotations += 1
            continue

        cache_key = (str(row.flight_id), str(row.video_file))
        if cache_key not in video_cache:
            video_cache[cache_key] = resolve_video_path(
                video_root, str(row.flight_id), str(row.video_file)
            )
        video_path = video_cache[cache_key]

        if video_path is None:
            counters.excluded_missing_videos += 1
            logger.error(
                "Video not found for annotation %s (flight=%s, video_file=%s)",
                row.annotation_id,
                row.flight_id,
                row.video_file,
            )
            continue

        retained_annotation_rows.append(row._asdict())
        for window_index, window_start in enumerate(starts):
            records.append(
                WindowRecord(
                    annotation_index=int(row.annotation_index),
                    annotation_id=str(row.annotation_id),
                    segment_id=str(row.segment_id),
                    flight_id=str(row.flight_id),
                    video_path=str(video_path),
                    label_name=str(row.label_name),
                    label_id=int(row.label_id),
                    annotation_start_sec=float(row.start_sec),
                    annotation_end_sec=float(row.end_sec),
                    annotation_duration_sec=float(row.annotation_duration_sec),
                    window_index=window_index,
                    window_start_sec=float(window_start),
                    window_end_sec=float(window_start + window_sec),
                )
            )

    retained = pd.DataFrame(retained_annotation_rows)
    counters.eligible_annotations = len(retained)
    counters.generated_windows = len(records)
    return records, retained


# =============================================================================
# Dataset
# =============================================================================

def make_eval_transform(image_size: int) -> nn.Module:
    """ImageNet preprocessing, with no stochastic augmentation."""
    return T.Compose(
        [
            T.ToImage(),
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class DeterministicWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[WindowRecord],
        frames_per_window: int,
        transform: nn.Module,
    ) -> None:
        self.records = list(records)
        self.frames_per_window = int(frames_per_window)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _video_metadata(capture: cv2.VideoCapture, path: str) -> tuple[float, int]:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0.0:
            raise RuntimeError(f"Invalid FPS reported for {path}: {fps}")
        if frame_count <= 0:
            raise RuntimeError(
                f"Invalid frame count reported for {path}: {frame_count}"
            )
        return fps, frame_count

    def _decode_window(self, record: WindowRecord) -> torch.Tensor:
        capture = cv2.VideoCapture(record.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {record.video_path}")

        try:
            fps, frame_count = self._video_metadata(capture, record.video_path)

            # endpoint=False keeps every requested timestamp inside the window.
            timestamps = np.linspace(
                record.window_start_sec,
                record.window_end_sec,
                num=self.frames_per_window,
                endpoint=False,
                dtype=np.float64,
            )

            frames: list[torch.Tensor] = []
            for timestamp in timestamps:
                frame_index = int(round(float(timestamp) * fps))
                frame_index = min(max(frame_index, 0), frame_count - 1)

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(
                        f"Failed to decode {record.video_path} at "
                        f"{timestamp:.6f}s (frame {frame_index})."
                    )

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame_rgb)
                frames.append(frame_tensor)

            return torch.stack(frames, dim=0)
        finally:
            capture.release()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = self._decode_window(record)
        return {
            "frames": frames,
            "label": record.label_id,
            "record_index": index,
        }


# =============================================================================
# Model loading
# =============================================================================

class ViTB16TemporalMeanClassifier(nn.Module):
    """
    Portable fallback matching the expected ImageNet ViT-B/16 baseline:
    frame-wise ViT features -> temporal mean -> dropout -> linear classifier.

    The training module's VisionClassifier is always preferred when available.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.5,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames shaped [B,T,C,H,W], got {tuple(frames.shape)}"
            )
        batch_size, sequence_length, channels, height, width = frames.shape
        flat_frames = frames.reshape(
            batch_size * sequence_length, channels, height, width
        )
        frame_features = self.vit(flat_frames)
        frame_features = frame_features.reshape(
            batch_size, sequence_length, -1
        )
        temporal_features = frame_features.mean(dim=1)
        return self.classifier(self.dropout(temporal_features))


def import_training_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mace_train_ex4_online", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instantiate_with_supported_arguments(
    model_class: type[nn.Module],
    num_classes: int,
    dropout: float,
) -> nn.Module:
    signature = inspect.signature(model_class)
    parameters = signature.parameters

    candidates: dict[str, Any] = {
        "num_classes": num_classes,
        "n_classes": num_classes,
        "classes": num_classes,
        "dropout": dropout,
        "dropout_rate": dropout,
        "pretrained": False,
        "weights": None,
    }
    kwargs = {
        name: value
        for name, value in candidates.items()
        if name in parameters
    }

    try:
        return model_class(**kwargs)
    except TypeError as first_error:
        try:
            return model_class()
        except TypeError:
            raise RuntimeError(
                f"Could not instantiate {model_class.__name__}. "
                f"Constructor signature: {signature}"
            ) from first_error


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a state-dict mapping, a training checkpoint "
            "mapping, or a serialized nn.Module."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "network",
        "net",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, nn.Module):
            return candidate.state_dict()
        if isinstance(candidate, Mapping) and candidate:
            if all(torch.is_tensor(value) for value in candidate.values()):
                return candidate  # type: ignore[return-value]

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]

    raise KeyError(
        "Could not find model weights in checkpoint. Expected model_state_dict, "
        "state_dict, model, network, net, or a raw state dict."
    )


def strip_common_prefixes(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    for prefix in ("module.", "_orig_mod.", "model."):
        keys = list(cleaned)
        if keys and all(key.startswith(prefix) for key in keys):
            cleaned = {
                key[len(prefix) :]: value for key, value in cleaned.items()
            }
    return cleaned


def load_model(
    checkpoint_path: Path,
    train_script: Path,
    device: torch.device,
    dropout: float,
    allow_fallback_architecture: bool,
    logger: logging.Logger,
) -> tuple[nn.Module, dict[str, Any]]:
    architecture_source = ""
    model: nn.Module

    if train_script.is_file():
        try:
            module = import_training_module(train_script)
            if not hasattr(module, "VisionClassifier"):
                raise AttributeError(
                    f"{train_script} does not define VisionClassifier"
                )
            model = instantiate_with_supported_arguments(
                module.VisionClassifier,
                num_classes=len(CLASS_NAMES),
                dropout=dropout,
            )
            architecture_source = str(train_script.resolve())
            logger.info(
                "Using VisionClassifier imported from %s", architecture_source
            )
        except Exception:
            if not allow_fallback_architecture:
                raise
            logger.exception(
                "Could not import/instantiate VisionClassifier; using fallback."
            )
            model = ViTB16TemporalMeanClassifier(
                num_classes=len(CLASS_NAMES),
                dropout=dropout,
                pretrained=False,
            )
            architecture_source = "built-in ViTB16TemporalMeanClassifier fallback"
    else:
        if not allow_fallback_architecture:
            raise FileNotFoundError(
                f"Training script not found: {train_script}. Place this evaluator "
                "beside train_ex4_online.py, pass --train-script, or explicitly "
                "allow the fallback architecture."
            )
        model = ViTB16TemporalMeanClassifier(
            num_classes=len(CLASS_NAMES),
            dropout=dropout,
            pretrained=False,
        )
        architecture_source = "built-in ViTB16TemporalMeanClassifier fallback"
        logger.warning(
            "Training script not found; using the built-in fallback architecture."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = strip_common_prefixes(extract_state_dict(checkpoint))

    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint weights do not exactly match the instantiated model. "
            "This evaluator intentionally uses strict loading to prevent invalid "
            "results. Confirm --train-script points to the exact training script "
            "used for this checkpoint.\n\n"
            f"Original error:\n{error}"
        ) from error

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict loading unexpectedly reported incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    checkpoint_metadata: dict[str, Any] = {
        "architecture_source": architecture_source,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    if isinstance(checkpoint, Mapping):
        for key in (
            "epoch",
            "best_epoch",
            "best_val_macro_f1",
            "val_macro_f1",
            "experiment_name",
            "config",
        ):
            if key in checkpoint:
                value = checkpoint[key]
                try:
                    json.dumps(value)
                    checkpoint_metadata[key] = value
                except TypeError:
                    checkpoint_metadata[key] = str(value)

    return model, checkpoint_metadata


# =============================================================================
# Evaluation
# =============================================================================

def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


@torch.inference_mode()
def evaluate_windows(
    model: nn.Module,
    loader: DataLoader,
    records: Sequence[WindowRecord],
    device: torch.device,
    amp: bool,
    counters: EvaluationCounters,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    use_amp = bool(amp and device.type == "cuda")
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(int)
        record_indices = batch["record_index"].cpu().numpy().astype(int)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits_tensor = model(frames)

        if isinstance(logits_tensor, (tuple, list)):
            logits_tensor = logits_tensor[0]
        if isinstance(logits_tensor, Mapping):
            for key in ("logits", "output", "predictions"):
                if key in logits_tensor:
                    logits_tensor = logits_tensor[key]
                    break

        if not torch.is_tensor(logits_tensor):
            raise TypeError(
                f"Model output must be a tensor, got {type(logits_tensor)}"
            )
        if logits_tensor.ndim != 2 or logits_tensor.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                "Expected model logits shaped "
                f"[batch,{len(CLASS_NAMES)}], got {tuple(logits_tensor.shape)}"
            )

        logits = logits_tensor.float().cpu().numpy()
        probabilities = softmax_numpy(logits)
        predictions = probabilities.argmax(axis=1)

        for local_index, record_index in enumerate(record_indices):
            record = records[int(record_index)]
            if labels[local_index] != record.label_id:
                raise RuntimeError("Dataset label/record label mismatch.")

            row: dict[str, Any] = {
                **asdict(record),
                "predicted_id": int(predictions[local_index]),
                "predicted_label": ID_TO_LABEL[int(predictions[local_index])],
                "confidence": float(probabilities[local_index].max()),
                "correct": bool(predictions[local_index] == record.label_id),
            }
            for class_id, class_name in enumerate(CLASS_NAMES):
                row[f"logit_{class_name}"] = float(logits[local_index, class_id])
                row[f"prob_{class_name}"] = float(
                    probabilities[local_index, class_id]
                )
            rows.append(row)

        counters.evaluated_windows += len(record_indices)
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            logger.info(
                "Evaluated batch %d/%d (%d/%d windows)",
                batch_index,
                total_batches,
                counters.evaluated_windows,
                len(records),
            )

    return pd.DataFrame(rows)


def aggregate_annotations(window_predictions: pd.DataFrame) -> pd.DataFrame:
    if window_predictions.empty:
        raise ValueError("No window predictions are available for aggregation.")

    logit_columns = [f"logit_{name}" for name in CLASS_NAMES]
    grouped_rows: list[dict[str, Any]] = []

    group_columns = [
        "annotation_index",
        "annotation_id",
        "segment_id",
        "flight_id",
        "video_path",
        "label_name",
        "label_id",
        "annotation_start_sec",
        "annotation_end_sec",
        "annotation_duration_sec",
    ]

    for keys, group in window_predictions.groupby(
        group_columns, sort=True, dropna=False
    ):
        metadata = dict(zip(group_columns, keys))
        mean_logits = group[logit_columns].to_numpy(dtype=np.float64).mean(axis=0)
        probabilities = softmax_numpy(mean_logits[None, :])[0]
        prediction = int(np.argmax(probabilities))

        row: dict[str, Any] = {
            **metadata,
            "num_windows": int(len(group)),
            "first_window_start_sec": float(group["window_start_sec"].min()),
            "last_window_start_sec": float(group["window_start_sec"].max()),
            "predicted_id": prediction,
            "predicted_label": ID_TO_LABEL[prediction],
            "confidence": float(probabilities[prediction]),
            "correct": bool(prediction == int(metadata["label_id"])),
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"mean_logit_{class_name}"] = float(mean_logits[class_id])
            row[f"prob_{class_name}"] = float(probabilities[class_id])
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows).sort_values(
        ["flight_id", "annotation_start_sec", "annotation_id"]
    ).reset_index(drop=True)


# =============================================================================
# Metrics and artifacts
# =============================================================================

def compute_metrics(
    annotation_predictions: pd.DataFrame,
) -> tuple[dict[str, Any], str, pd.DataFrame, np.ndarray, np.ndarray]:
    y_true = annotation_predictions["label_id"].to_numpy(dtype=int)
    y_pred = annotation_predictions["predicted_id"].to_numpy(dtype=int)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0,
        )
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()

    matrix_counts = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    matrix_normalized = confusion_matrix(
        y_true, y_pred, labels=CLASS_IDS, normalize="true"
    )
    matrix_normalized = np.nan_to_num(matrix_normalized)

    per_class_support = {
        class_name: int((y_true == class_id).sum())
        for class_id, class_name in enumerate(CLASS_NAMES)
    }

    metrics: dict[str, Any] = {
        "evaluation_unit": "annotation",
        "num_annotations": int(len(annotation_predictions)),
        "num_flights": int(annotation_predictions["flight_id"].nunique()),
        "num_windows": int(annotation_predictions["num_windows"].sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class_support": per_class_support,
        "classification_report": report_dict,
    }
    return (
        metrics,
        report_text,
        report_frame,
        matrix_counts,
        matrix_normalized,
    )


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    output_path: Path,
    normalized: bool,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Ground-truth label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    value_format = ".3f" if normalized else "d"
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            display_value = (
                format(float(value), value_format)
                if normalized
                else format(int(value), value_format)
            )
            axis.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                color="white" if float(value) > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    scalar_rows = []
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar_rows.append({"metric": key, "value": value})
    pd.DataFrame(scalar_rows).to_csv(path, index=False)


def log_tensorboard(
    output_dir: Path,
    metrics: dict[str, Any],
    report_frame: pd.DataFrame,
    matrix_counts: np.ndarray,
    matrix_normalized: np.ndarray,
    annotation_predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    if SummaryWriter is None:
        raise ModuleNotFoundError(
            "TensorBoard is required for evaluation outputs. Install it in the "
            "active MACE environment with: python -m pip install tensorboard"
        )

    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "num_annotations",
        "num_flights",
        "num_windows",
    )
    for key in scalar_keys:
        writer.add_scalar(f"evaluation/{key}", float(metrics[key]), 0)

    for class_name in CLASS_NAMES:
        if class_name in report_frame.index:
            for metric_name in ("precision", "recall", "f1-score", "support"):
                writer.add_scalar(
                    f"per_class/{class_name}_{metric_name.replace('-', '_')}",
                    float(report_frame.loc[class_name, metric_name]),
                    0,
                )

    counts_figure, counts_axis = plt.subplots(figsize=(8, 7))
    counts_axis.imshow(matrix_counts, interpolation="nearest")
    counts_axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Ground-truth label",
        title="Annotation-level confusion matrix (counts)",
    )
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            counts_axis.text(
                column,
                row,
                str(int(matrix_counts[row, column])),
                ha="center",
                va="center",
            )
    counts_figure.tight_layout()
    writer.add_figure("evaluation/confusion_matrix_counts", counts_figure, 0)
    plt.close(counts_figure)

    normalized_figure, normalized_axis = plt.subplots(figsize=(8, 7))
    normalized_axis.imshow(matrix_normalized, interpolation="nearest")
    normalized_axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Ground-truth label",
        title="Annotation-level confusion matrix (row-normalized)",
    )
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            normalized_axis.text(
                column,
                row,
                f"{matrix_normalized[row, column]:.3f}",
                ha="center",
                va="center",
            )
    normalized_figure.tight_layout()
    writer.add_figure(
        "evaluation/confusion_matrix_normalized", normalized_figure, 0
    )
    plt.close(normalized_figure)

    writer.add_text(
        "evaluation/configuration",
        "\n".join(
            [
                f"- split: `{args.split}`",
                f"- window_seconds: `{args.window_sec}`",
                f"- stride_seconds: `{args.stride_sec}`",
                f"- frames_per_window: `{args.frames_per_window}`",
                f"- annotation_aggregation: `mean raw logits`",
                f"- checkpoint: `{args.checkpoint}`",
            ]
        ),
        0,
    )

    prediction_table = annotation_predictions[
        [
            "annotation_id",
            "flight_id",
            "label_name",
            "predicted_label",
            "confidence",
            "num_windows",
            "correct",
        ]
    ].copy()
    writer.add_text(
        "evaluation/annotation_predictions",
        prediction_table.to_markdown(index=False),
        0,
    )

    writer.flush()
    writer.close()


# =============================================================================
# CLI
# =============================================================================

def parse_sheet_name(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def read_flights_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Flights file not found: {path}")

    flights: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            # Accommodate one-column CSV/TXT and comma-separated lines.
            flights.update(
                token.strip()
                for token in next(csv.reader([value]))
                if token.strip()
            )
    return flights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic annotation-level evaluation of the MACE "
            "ImageNet-pretrained ViT-B/16 vision model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path(__file__).resolve().with_name("train_ex4_online.py"),
        help="Exact training script defining VisionClassifier.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        required=True,
        help="Held-out split to evaluate.",
    )
    parser.add_argument(
        "--flights",
        nargs="*",
        default=None,
        help="Optional explicit whitelist of selected held-out flight IDs.",
    )
    parser.add_argument(
        "--flights-file",
        type=Path,
        default=None,
        help="Optional TXT/CSV file containing selected flight IDs.",
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--stride-sec", type=float, default=DEFAULT_STRIDE_SEC)
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=DEFAULT_FRAMES_PER_WINDOW,
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--allow-fallback-architecture",
        action="store_true",
        help=(
            "Permit built-in temporal-mean ViT-B/16 if VisionClassifier cannot "
            "be imported. Strict checkpoint loading still applies."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive.")
    if args.stride_sec <= 0:
        raise ValueError("--stride-sec must be positive.")
    if args.frames_per_window <= 0:
        raise ValueError("--frames-per-window must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    if not args.video_root.is_dir():
        raise NotADirectoryError(f"Video root not found: {args.video_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use --overwrite or choose a new directory."
            )


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = build_parser().parse_args()
    args.sheet_name = parse_sheet_name(args.sheet_name)
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir, args.verbose)
    start_time = time.perf_counter()

    try:
        set_deterministic(args.seed)
        device = select_device(args.device)

        selected_flights: set[str] | None = None
        if args.flights:
            selected_flights = {str(value).strip() for value in args.flights}
        if args.flights_file is not None:
            file_flights = read_flights_file(args.flights_file)
            selected_flights = (
                file_flights
                if selected_flights is None
                else selected_flights | file_flights
            )
        if selected_flights is not None and not selected_flights:
            raise ValueError("Explicit flight selection is empty.")

        logger.info("=" * 78)
        logger.info("MACE deterministic annotation-level evaluation")
        logger.info("=" * 78)
        logger.info("Split              : %s", args.split)
        logger.info("Annotation file    : %s", args.annotations.resolve())
        logger.info("Video root         : %s", args.video_root.resolve())
        logger.info("Checkpoint         : %s", args.checkpoint.resolve())
        logger.info("Window / stride    : %.3fs / %.3fs", args.window_sec, args.stride_sec)
        logger.info("Frames per window  : %d", args.frames_per_window)
        logger.info("Device             : %s", device)
        if device.type == "cuda":
            logger.info("GPU                : %s", torch.cuda.get_device_name(0))
        logger.info(
            "Selected flights   : %s",
            sorted(selected_flights) if selected_flights else "from split column",
        )

        counters = EvaluationCounters()
        raw_annotations = read_annotations(args.annotations, args.sheet_name)
        annotations = normalize_annotations(
            raw_annotations,
            split=args.split,
            selected_flights=selected_flights,
            counters=counters,
            logger=logger,
        )
        if annotations.empty:
            raise ValueError("No annotations remain after split/flight filtering.")

        records, retained_annotations = build_window_records(
            annotations,
            video_root=args.video_root,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            counters=counters,
            logger=logger,
        )
        if not records:
            raise ValueError(
                "No valid evaluation windows were generated. Check flight "
                "selection, video paths, annotation durations, and window length."
            )

        logger.info(
            "Eligible annotations : %d (excluded short=%d, missing video=%d)",
            counters.eligible_annotations,
            counters.excluded_short_annotations,
            counters.excluded_missing_videos,
        )
        logger.info("Generated windows  : %d", counters.generated_windows)

        model, checkpoint_metadata = load_model(
            checkpoint_path=args.checkpoint,
            train_script=args.train_script,
            device=device,
            dropout=args.dropout,
            allow_fallback_architecture=args.allow_fallback_architecture,
            logger=logger,
        )

        dataset = DeterministicWindowDataset(
            records=records,
            frames_per_window=args.frames_per_window,
            transform=make_eval_transform(args.image_size),
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
            generator=generator,
            drop_last=False,
        )

        window_predictions = evaluate_windows(
            model=model,
            loader=loader,
            records=records,
            device=device,
            amp=args.amp,
            counters=counters,
            logger=logger,
        )
        annotation_predictions = aggregate_annotations(window_predictions)

        (
            metrics,
            report_text,
            report_frame,
            matrix_counts,
            matrix_normalized,
        ) = compute_metrics(annotation_predictions)

        # Save tabular outputs.
        window_predictions.to_csv(
            args.output_dir / "window_predictions.csv", index=False
        )
        annotation_predictions.to_csv(
            args.output_dir / "annotation_predictions.csv", index=False
        )
        report_frame.to_csv(
            args.output_dir / "classification_report.csv", index=True
        )
        (args.output_dir / "classification_report.txt").write_text(
            report_text, encoding="utf-8"
        )

        counts_frame = pd.DataFrame(
            matrix_counts, index=CLASS_NAMES, columns=CLASS_NAMES
        )
        normalized_frame = pd.DataFrame(
            matrix_normalized, index=CLASS_NAMES, columns=CLASS_NAMES
        )
        counts_frame.to_csv(args.output_dir / "confusion_matrix_counts.csv")
        normalized_frame.to_csv(
            args.output_dir / "confusion_matrix_normalized.csv"
        )

        save_confusion_matrix_plot(
            matrix_counts,
            args.output_dir / "confusion_matrix_counts.png",
            normalized=False,
            title="Annotation-level confusion matrix (counts)",
        )
        save_confusion_matrix_plot(
            matrix_normalized,
            args.output_dir / "confusion_matrix_normalized.png",
            normalized=True,
            title="Annotation-level confusion matrix (row-normalized)",
        )

        elapsed_seconds = time.perf_counter() - start_time
        metrics["runtime_seconds"] = float(elapsed_seconds)
        metrics["split"] = args.split
        metrics["window_sec"] = float(args.window_sec)
        metrics["stride_sec"] = float(args.stride_sec)
        metrics["frames_per_window"] = int(args.frames_per_window)
        metrics["aggregation"] = "arithmetic mean of raw window logits"
        metrics["checkpoint_metadata"] = checkpoint_metadata
        metrics["counters"] = asdict(counters)

        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        write_metrics_csv(metrics, args.output_dir / "metrics.csv")

        run_config = {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected_flights": (
                sorted(selected_flights) if selected_flights is not None else None
            ),
            "label_map": LABEL_MAP,
            "checkpoint_metadata": checkpoint_metadata,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_protocol": {
                "window_generation": (
                    "all complete fixed-stride windows fully contained in each "
                    "annotation"
                ),
                "frame_sampling": (
                    "25 uniformly spaced timestamps per 5-second window, "
                    "endpoint excluded"
                ),
                "annotation_aggregation": "arithmetic mean of raw window logits",
                "final_prediction": "argmax softmax(mean annotation logits)",
            },
        }
        with (args.output_dir / "run_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(run_config, handle, indent=2, sort_keys=True)

        log_tensorboard(
            output_dir=args.output_dir,
            metrics=metrics,
            report_frame=report_frame,
            matrix_counts=matrix_counts,
            matrix_normalized=matrix_normalized,
            annotation_predictions=annotation_predictions,
            args=args,
        )

        logger.info("-" * 78)
        logger.info("Annotation accuracy : %.6f", metrics["accuracy"])
        logger.info("Annotation macro F1 : %.6f", metrics["macro_f1"])
        logger.info("Balanced accuracy   : %.6f", metrics["balanced_accuracy"])
        logger.info("Annotations         : %d", metrics["num_annotations"])
        logger.info("Windows             : %d", metrics["num_windows"])
        logger.info("Runtime             : %.2fs", elapsed_seconds)
        logger.info("Outputs             : %s", args.output_dir.resolve())
        logger.info("-" * 78)
        logger.info("\n%s", report_text)
        return 0

    except Exception:
        logger.exception("Evaluation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
