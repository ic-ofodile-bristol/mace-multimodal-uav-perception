#!/usr/bin/env python3
"""
===============================================================================
train_flight_ex3_binary.py
===============================================================================

MACE - Aircraft-Only Temporal Baseline
Dynamics-8 Binary Disturbance TimeMIL-inspired Patch Transformer

Experiment purpose
------------------
This experiment tests whether aircraft dynamics generalise better as a
binary disturbance-regime signal than as the original four semantic
UAV-plume interaction states.

Binary targets:
    0 = low_disturbance  = approaching + entering
    1 = high_disturbance = inside + exiting

The original four-class annotation is retained in the dataframe for provenance,
but training, validation, loss, confusion matrices and metrics use the binary
disturbance target.

Changes relative to Ex1
-----------------------
- Uses only 8 disturbance-focused channels:
    Roll, Pitch,
    AccX, AccY, AccZ,
    GyrX, GyrY, GyrZ

- Removes:
    yaw,
    relative altitude,
    vertical speed,
    ground speed

- Re-references Roll and Pitch independently inside each 5-second window.
  The model therefore focuses on attitude fluctuations rather than absolute
  aircraft attitude.

- Uses the compact Ex2 Transformer rather than returning to Ex1 capacity.
  The binary task is simpler and Ex1 already showed stronger cross-flight
  overfitting.

- Transformer:
    d_model     = 64
    layers      = 2
    heads       = 4
    ff_dim      = 128
    dropout     = 0.40

- Uses shorter temporal patches:
    patch_len   = 2 samples = 0.2 s at 10 Hz
    patch_stride= 2
    50-step window -> 25 temporal tokens

- Stronger regularisation:
    weight_decay   = 0.05
    label_smoothing= 0.05

- Balanced online sampling:
    train = 6000 samples/epoch by default
    val   = 1200 deterministic samples by default

- Binary balanced sampling therefore targets:
    train = 3000 low + 3000 high samples/epoch
    val   = 600 low + 600 high samples by default

- Saves TensorBoard scalars and confusion matrices in the same general
  experiment structure used by the vision experiments.

Input
-----
results/flight_baseline/labelled_flight_telemetry.csv
data/annotations/flight_splits.csv

Binary classes
--------------
0 low_disturbance  = approaching + entering
1 high_disturbance = inside + exiting

Experiment outputs
------------------
experiments/<SLURM_JOB_NAME>_<SLURM_JOB_ID>_<TIMESTAMP>/

    config.json
    normalization_stats.json
    training_history.csv

    checkpoints/
        best.pt
        last.pt
        epoch_010.pt
        ...

    tensorboard/

    confusion_matrices/
        epoch_001_counts.png
        epoch_001_normalised.png
        ...
        best_counts.png
        best_normalised.png

    reports/
        classification_report_best.txt
        best_validation_metrics.json

    predictions/
        epoch_001_validation_predictions.csv
        ...
        best_validation_predictions.csv

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# =============================================================================
# SLURM / experiment identification
# =============================================================================

JOB_NAME = os.environ.get(
    "SLURM_JOB_NAME",
    "Flight_Transformer_ex3-binary-6k",
)

JOB_ID = os.environ.get(
    "SLURM_JOB_ID",
    "local",
)

TIMESTAMP = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)


# =============================================================================
# Configuration
# =============================================================================

# Original semantic annotation -> binary aircraft-disturbance target.
# The aircraft branch is intentionally NOT asked to distinguish
# approaching from entering, or inside from exiting.
LABEL_MAP = {
    "approaching": 0,
    "entering": 0,
    "inside": 1,
    "exiting": 1,
}

ID_TO_LABEL = {
    0: "low_disturbance",
    1: "high_disturbance",
}

NUM_CLASSES = 2

DEFAULT_TELEMETRY = Path(
    "results/flight_baseline/labelled_flight_telemetry.csv"
)

DEFAULT_SPLITS = Path(
    "data/annotations/flight_splits.csv"
)

# -------------------------------------------------------------------------
# Ex2 Dynamics-8 features
# -------------------------------------------------------------------------

BASE_FEATURES = [
    "Roll_deg",
    "Pitch_deg",
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
]

MODEL_FEATURE_NAMES = [
    "roll_rel_deg",
    "pitch_rel_deg",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

ROLL_INDEX = 0
PITCH_INDEX = 1


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Flight split loading
# =============================================================================

def load_flight_splits(path: Path) -> dict[str, str]:
    """
    Supports common split CSV layouts.

    Preferred:
        flight_id,split
        FL008,train
        FL012,val

    Also supports:
        flight,split

    or wide columns such as:
        train,val,test
        FL008,FL028,FL021
    """

    df = pd.read_csv(path)

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    # ------------------------------------------------------------------
    # Long format
    # ------------------------------------------------------------------

    flight_col = None

    for candidate in [
        "flight_id",
        "flight",
    ]:
        if candidate in lower:
            flight_col = lower[candidate]
            break

    if (
        flight_col is not None
        and "split" in lower
    ):

        split_col = lower["split"]
        mapping = {}

        for _, row in df.iterrows():

            flight = str(
                row[flight_col]
            ).strip().upper()

            split = str(
                row[split_col]
            ).strip().lower()

            if split in {
                "train",
                "val",
                "validation",
                "test",
            }:

                if split == "validation":
                    split = "val"

                mapping[flight] = split

        return mapping

    # ------------------------------------------------------------------
    # Wide format
    # ------------------------------------------------------------------

    mapping = {}

    aliases = {
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "test",
    }

    for col in df.columns:

        key = str(col).lower()

        if key not in aliases:
            continue

        split = aliases[key]

        for value in df[col].dropna():

            flight = str(
                value
            ).strip().upper()

            if flight:
                mapping[flight] = split

    if not mapping:

        raise RuntimeError(
            "Could not interpret flight_splits.csv.\n"
            f"Columns were: {df.columns.tolist()}\n\n"
            "Expected either:\n"
            "  flight_id,split\n"
            "or columns named train,val,test."
        )

    return mapping


# =============================================================================
# Telemetry loading
# =============================================================================

def load_telemetry(
    path: Path,
    split_lookup: dict[str, str],
) -> pd.DataFrame:

    print(f"Loading telemetry:\n  {path}")

    df = pd.read_csv(path)

    required = {
        "flight_id",
        "segment_uid",
        "state",
        "autopilot_time_sec",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Telemetry CSV missing columns: {sorted(missing)}"
        )

    for feature in BASE_FEATURES:

        if feature not in df.columns:
            raise RuntimeError(
                f"Required telemetry feature missing: {feature}"
            )

    df["flight_id"] = (
        df["flight_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["state"] = (
        df["state"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["state"].isin(LABEL_MAP)
    ].copy()

    # Preserve the original four-class semantic annotation.
    df["semantic_state"] = df["state"]

    # Binary target used by this aircraft-only experiment.
    df["state_id"] = (
        df["state"]
        .map(LABEL_MAP)
        .astype(int)
    )

    df["disturbance_state"] = (
        df["state_id"]
        .map(ID_TO_LABEL)
    )

    df["split"] = (
        df["flight_id"]
        .map(split_lookup)
    )

    unknown = sorted(
        df.loc[
            df["split"].isna(),
            "flight_id"
        ].unique()
    )

    if unknown:

        print(
            "\nWARNING: flights absent from split file "
            "will be excluded:"
        )

        print(
            "  " + " ".join(unknown)
        )

    df = df[
        df["split"].notna()
    ].copy()

    df["autopilot_time_sec"] = pd.to_numeric(
        df["autopilot_time_sec"],
        errors="coerce",
    )

    for feature in BASE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df = df.sort_values(
        [
            "flight_id",
            "segment_uid",
            "autopilot_time_sec",
        ]
    ).reset_index(drop=True)

    return df


# =============================================================================
# Segment construction
# =============================================================================

@dataclass
class Segment:

    segment_uid: str
    flight_id: str
    state: str
    state_id: int
    split: str

    times: np.ndarray
    values: np.ndarray


def build_model_features(
    segment_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the Ex2 Dynamics-8 raw feature representation.

    IMPORTANT:
    Roll/Pitch are NOT re-referenced here.

    They are re-referenced independently inside each sampled 5-second window
    before normalisation, so every window starts with Roll=0 and Pitch=0.
    """

    values = np.column_stack(
        [
            segment_df["Roll_deg"].to_numpy(dtype=np.float32),
            segment_df["Pitch_deg"].to_numpy(dtype=np.float32),
            segment_df["AccX"].to_numpy(dtype=np.float32),
            segment_df["AccY"].to_numpy(dtype=np.float32),
            segment_df["AccZ"].to_numpy(dtype=np.float32),
            segment_df["GyrX"].to_numpy(dtype=np.float32),
            segment_df["GyrY"].to_numpy(dtype=np.float32),
            segment_df["GyrZ"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)

    return values


def build_segments(
    df: pd.DataFrame,
    rate_hz: float,
    window_sec: float,
) -> list[Segment]:

    segments = []

    min_samples = int(
        round(
            rate_hz *
            window_sec
        )
    )

    grouped = df.groupby(
        "segment_uid",
        sort=False,
    )

    skipped_short = 0
    skipped_nan = 0

    for segment_uid, g in grouped:

        g = g.sort_values(
            "autopilot_time_sec"
        ).reset_index(drop=True)

        if len(g) < min_samples:
            skipped_short += 1
            continue

        values = build_model_features(
            g
        )

        # Fill only small pre-existing interpolation gaps locally.
        values_df = pd.DataFrame(
            values
        ).interpolate(
            limit_direction="both"
        )

        values = values_df.to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            skipped_nan += 1
            continue

        segments.append(
            Segment(
                segment_uid=str(
                    segment_uid
                ),
                flight_id=str(
                    g["flight_id"].iloc[0]
                ),
                state=str(
                    g["state"].iloc[0]
                ),
                state_id=int(
                    g["state_id"].iloc[0]
                ),
                split=str(
                    g["split"].iloc[0]
                ),
                times=g[
                    "autopilot_time_sec"
                ].to_numpy(dtype=np.float64),
                values=values,
            )
        )

    print()
    print("Segment construction")
    print("-" * 72)
    print(f"Usable segments        : {len(segments)}")
    print(f"Skipped < window       : {skipped_short}")
    print(f"Skipped invalid values : {skipped_nan}")

    return segments


# =============================================================================
# Window preprocessing
# =============================================================================

def rereference_attitude(
    x: np.ndarray,
) -> np.ndarray:
    """
    Re-reference Roll and Pitch within one 5-second window.

    This suppresses flight-specific absolute attitude while preserving
    temporal attitude changes/oscillations.
    """

    x = x.copy()

    x[:, ROLL_INDEX] = (
        x[:, ROLL_INDEX]
        -
        x[0, ROLL_INDEX]
    )

    x[:, PITCH_INDEX] = (
        x[:, PITCH_INDEX]
        -
        x[0, PITCH_INDEX]
    )

    return x


# =============================================================================
# Normalisation
# =============================================================================

def calculate_train_statistics(
    segments: list[Segment],
    window_steps: int,
    seed: int,
    max_windows: int = 10000,
):
    """
    Calculate TRAIN-ONLY normalisation statistics from valid 5-second windows
    AFTER Roll/Pitch re-referencing.

    This is more consistent than computing statistics from absolute attitude
    values and then subtracting the first point later.
    """

    train_segments = [
        s
        for s in segments
        if (
            s.split == "train"
            and
            len(s.values) >= window_steps
        )
    ]

    if not train_segments:
        raise RuntimeError(
            "No training segments found."
        )

    candidates = []

    for segment in train_segments:

        max_start = (
            len(segment.values)
            -
            window_steps
        )

        # Use non-overlapping candidate windows for stable statistics.
        for start_idx in range(
            0,
            max_start + 1,
            window_steps,
        ):

            candidates.append(
                (
                    segment,
                    start_idx,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No valid training windows available for normalization."
        )

    rng = np.random.default_rng(seed)

    if len(candidates) > max_windows:

        chosen = rng.choice(
            len(candidates),
            size=max_windows,
            replace=False,
        )

        candidates = [
            candidates[i]
            for i in chosen
        ]

    arrays = []

    for segment, start_idx in candidates:

        end_idx = (
            start_idx
            +
            window_steps
        )

        x = segment.values[
            start_idx:end_idx
        ]

        x = rereference_attitude(
            x
        )

        arrays.append(
            x
        )

    all_train = np.concatenate(
        arrays,
        axis=0,
    )

    mean = np.mean(
        all_train,
        axis=0,
    ).astype(np.float32)

    std = np.std(
        all_train,
        axis=0,
    ).astype(np.float32)

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    ).astype(np.float32)

    return mean, std


# =============================================================================
# Telemetry augmentation
# =============================================================================

class TelemetryAugmenter:
    """
    Conservative augmentation applied AFTER normalisation.

    Ex2 intentionally disables channel gain scaling because absolute disturbance
    amplitude appears to be physically informative from the diagnostics.

    Augmentations retained:
        - small Gaussian jitter
        - occasional single-channel masking
        - short temporal masking

    Avoided:
        - temporal reversal
        - permutation
        - large time warping
        - aggressive channel scaling
    """

    def __init__(
        self,
        jitter_std: float = 0.015,
        channel_mask_prob: float = 0.03,
        time_mask_prob: float = 0.20,
        max_time_mask_steps: int = 3,
    ):

        self.jitter_std = jitter_std
        self.channel_mask_prob = channel_mask_prob
        self.time_mask_prob = time_mask_prob
        self.max_time_mask_steps = max_time_mask_steps

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x.clone()

        # -------------------------------------------------------------
        # 1. Mild Gaussian sensor jitter
        # -------------------------------------------------------------

        if self.jitter_std > 0:

            x = (
                x
                +
                torch.randn_like(x)
                *
                self.jitter_std
            )

        # -------------------------------------------------------------
        # 2. Occasional feature masking
        # -------------------------------------------------------------

        if self.channel_mask_prob > 0:

            mask = (
                torch.rand(
                    x.shape[1],
                    device=x.device,
                )
                <
                self.channel_mask_prob
            )

            x[:, mask] = 0.0

        # -------------------------------------------------------------
        # 3. Short temporal masking
        # -------------------------------------------------------------

        if (
            random.random()
            <
            self.time_mask_prob
        ):

            length = random.randint(
                1,
                self.max_time_mask_steps,
            )

            if length < x.shape[0]:

                start = random.randint(
                    0,
                    x.shape[0] - length,
                )

                x[
                    start:start + length
                ] = 0.0

        return x


# =============================================================================
# Online training dataset
# =============================================================================

class OnlineBalancedTelemetryDataset(Dataset):

    def __init__(
        self,
        segments: list[Segment],
        split: str,
        samples_per_epoch: int,
        window_steps: int,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
        seed: int,
    ):

        super().__init__()

        self.segments = [
            s
            for s in segments
            if s.split == split
        ]

        self.samples_per_epoch = (
            samples_per_epoch
        )

        self.window_steps = (
            window_steps
        )

        self.mean = mean
        self.std = std

        self.seed = seed
        self.epoch = 0

        self.augmenter = (
            TelemetryAugmenter()
            if augment
            else None
        )

        self.by_class = defaultdict(
            list
        )

        for segment in self.segments:

            if (
                len(segment.values)
                >= self.window_steps
            ):

                self.by_class[
                    segment.state_id
                ].append(segment)

        missing = [
            ID_TO_LABEL[i]
            for i in range(NUM_CLASSES)
            if len(self.by_class[i]) == 0
        ]

        if missing:

            raise RuntimeError(
                f"{split} split has no valid "
                f"5-second segments for classes: "
                f"{missing}"
            )

        print()
        print(
            f"{split.upper()} online dataset"
        )
        print("-" * 72)

        for class_id in range(
            NUM_CLASSES
        ):

            print(
                f"{ID_TO_LABEL[class_id]:12s}: "
                f"{len(self.by_class[class_id])} "
                f"eligible annotation segments"
            )

    def set_epoch(
        self,
        epoch: int,
    ):

        self.epoch = epoch

    def __len__(self):

        return self.samples_per_epoch

    def __getitem__(
        self,
        index,
    ):

        # -------------------------------------------------------------
        # Exact balanced class allocation
        # -------------------------------------------------------------

        class_id = (
            index %
            NUM_CLASSES
        )

        rng = np.random.default_rng(
            self.seed
            + self.epoch * 1_000_003
            + index
        )

        candidates = self.by_class[
            class_id
        ]

        # -------------------------------------------------------------
        # Duration-weighted segment selection
        # -------------------------------------------------------------

        valid_starts = np.array(
            [
                len(s.values)
                - self.window_steps
                + 1
                for s in candidates
            ],
            dtype=np.float64,
        )

        probabilities = (
            valid_starts
            /
            valid_starts.sum()
        )

        segment_index = int(
            rng.choice(
                len(candidates),
                p=probabilities,
            )
        )

        segment = candidates[
            segment_index
        ]

        max_start = (
            len(segment.values)
            -
            self.window_steps
        )

        if max_start > 0:

            start_idx = int(
                rng.integers(
                    0,
                    max_start + 1,
                )
            )

        else:

            start_idx = 0

        end_idx = (
            start_idx
            +
            self.window_steps
        )

        x = segment.values[
            start_idx:end_idx
        ].copy()

        # -------------------------------------------------------------
        # Re-reference Roll/Pitch BEFORE global normalization.
        # -------------------------------------------------------------

        x = rereference_attitude(
            x
        )

        x = (
            x
            -
            self.mean
        ) / self.std

        x = torch.from_numpy(
            x.astype(np.float32)
        )

        if self.augmenter is not None:

            x = self.augmenter(
                x
            )

        metadata = {
            "segment_uid":
                segment.segment_uid,
            "flight_id":
                segment.flight_id,
            "window_start_sec":
                float(
                    segment.times[
                        start_idx
                    ]
                ),
        }

        return (
            x,
            torch.tensor(
                class_id,
                dtype=torch.long,
            ),
            metadata,
        )


# =============================================================================
# Fixed validation dataset
# =============================================================================

class FixedBalancedTelemetryDataset(Dataset):

    def __init__(
        self,
        segments: list[Segment],
        split: str,
        samples: int,
        window_steps: int,
        mean: np.ndarray,
        std: np.ndarray,
        seed: int,
    ):

        super().__init__()

        self.mean = mean
        self.std = std

        by_class = defaultdict(
            list
        )

        for segment in segments:

            if (
                segment.split == split
                and
                len(segment.values)
                >= window_steps
            ):

                by_class[
                    segment.state_id
                ].append(segment)

        missing = [
            ID_TO_LABEL[i]
            for i in range(NUM_CLASSES)
            if len(by_class[i]) == 0
        ]

        if missing:

            raise RuntimeError(
                f"{split} split has no valid "
                f"segments for: {missing}"
            )

        rng = np.random.default_rng(
            seed
        )

        samples_per_class = (
            samples
            //
            NUM_CLASSES
        )

        records = []

        for class_id in range(
            NUM_CLASSES
        ):

            candidates = by_class[
                class_id
            ]

            valid_starts = np.array(
                [
                    len(s.values)
                    - window_steps
                    + 1
                    for s in candidates
                ],
                dtype=float,
            )

            probabilities = (
                valid_starts
                /
                valid_starts.sum()
            )

            for _ in range(
                samples_per_class
            ):

                seg_idx = int(
                    rng.choice(
                        len(candidates),
                        p=probabilities,
                    )
                )

                segment = candidates[
                    seg_idx
                ]

                max_start = (
                    len(segment.values)
                    -
                    window_steps
                )

                if max_start > 0:

                    start_idx = int(
                        rng.integers(
                            0,
                            max_start + 1,
                        )
                    )

                else:

                    start_idx = 0

                records.append(
                    (
                        segment,
                        start_idx,
                        class_id,
                    )
                )

        self.records = records
        self.window_steps = window_steps

        print()
        print(
            f"{split.upper()} fixed dataset"
        )
        print("-" * 72)
        print(
            f"Total windows         : {len(self.records)}"
        )
        print(
            f"Windows/class         : {samples_per_class}"
        )

    def __len__(self):

        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):

        (
            segment,
            start_idx,
            class_id,
        ) = self.records[
            index
        ]

        end_idx = (
            start_idx
            +
            self.window_steps
        )

        x = segment.values[
            start_idx:end_idx
        ].copy()

        x = rereference_attitude(
            x
        )

        x = (
            x
            -
            self.mean
        ) / self.std

        x = torch.from_numpy(
            x.astype(np.float32)
        )

        metadata = {
            "segment_uid":
                segment.segment_uid,
            "flight_id":
                segment.flight_id,
            "window_start_sec":
                float(
                    segment.times[
                        start_idx
                    ]
                ),
        }

        return (
            x,
            torch.tensor(
                class_id,
                dtype=torch.long,
            ),
            metadata,
        )


# =============================================================================
# TimeMIL-inspired model
# =============================================================================

class GatedAttentionPool(nn.Module):
    """
    MIL-style attention pooling over temporal patches.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):

        super().__init__()

        self.v = nn.Linear(
            dim,
            hidden_dim,
        )

        self.u = nn.Linear(
            dim,
            hidden_dim,
        )

        self.w = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x,
    ):

        a = torch.tanh(
            self.v(x)
        )

        b = torch.sigmoid(
            self.u(x)
        )

        scores = self.w(
            a * b
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


class TimeMILPatchTransformer(nn.Module):

    def __init__(
        self,
        num_features: int,
        window_steps: int,
        patch_len: int = 2,
        patch_stride: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.40,
        num_classes: int = 4,
    ):

        super().__init__()

        if window_steps < patch_len:

            raise ValueError(
                "window_steps must be >= patch_len"
            )

        self.num_features = num_features
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.num_patches = (
            1
            +
            (
                window_steps
                -
                patch_len
            )
            //
            patch_stride
        )

        patch_input_dim = (
            num_features
            *
            patch_len
        )

        self.patch_projection = nn.Sequential(
            nn.Linear(
                patch_input_dim,
                d_model,
            ),
            nn.LayerNorm(
                d_model
            ),
            nn.GELU(),
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches,
                d_model,
            )
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(
                d_model
            ),
        )

        self.pool = GatedAttentionPool(
            dim=d_model,
            hidden_dim=max(
                32,
                d_model // 2,
            ),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                d_model // 2,
                num_classes,
            ),
        )

    def make_patches(
        self,
        x,
    ):

        # x: [B, T, F]

        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.patch_stride,
        )

        # [B, P, F, patch_len] -> [B, P, patch_len, F]
        patches = (
            patches
            .permute(
                0,
                1,
                3,
                2,
            )
            .contiguous()
        )

        patches = patches.view(
            patches.shape[0],
            patches.shape[1],
            -1,
        )

        return patches

    def forward(
        self,
        x,
    ):

        patches = self.make_patches(
            x
        )

        tokens = self.patch_projection(
            patches
        )

        tokens = (
            tokens
            +
            self.position_embedding[
                :,
                :tokens.shape[1],
            ]
        )

        encoded = self.transformer(
            tokens
        )

        pooled, attention = self.pool(
            encoded
        )

        logits = self.classifier(
            pooled
        )

        return logits, attention


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# =============================================================================
# Experiment directory
# =============================================================================

def create_experiment_directory(
    project_root: Path,
) -> Path:

    experiment_name = (
        f"{JOB_NAME}_"
        f"{JOB_ID}_"
        f"{TIMESTAMP}"
    )

    experiment_directory = (
        project_root
        / "experiments"
        / experiment_name
    )

    for child in [
        "checkpoints",
        "tensorboard",
        "confusion_matrices",
        "reports",
        "sampling_logs",
        "predictions",
    ]:

        (
            experiment_directory
            / child
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return experiment_directory


# =============================================================================
# Confusion-matrix plotting / TensorBoard
# =============================================================================

def make_confusion_matrix_figure(
    y_true,
    y_pred,
    normalised: bool,
    title: str,
):

    labels = list(
        range(
            NUM_CLASSES
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    if normalised:

        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        cm_plot = np.divide(
            cm.astype(np.float64),
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=np.float64,
            ),
            where=row_sums != 0,
        )

    else:

        cm_plot = cm

    fig, ax = plt.subplots(
        figsize=(7.5, 6.5)
    )

    image = ax.imshow(
        cm_plot,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    class_names = [
        ID_TO_LABEL[i]
        for i in labels
    ]

    ax.set(
        xticks=np.arange(
            NUM_CLASSES
        ),
        yticks=np.arange(
            NUM_CLASSES
        ),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        cm_plot.max() / 2.0
        if cm_plot.size
        else 0.0
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            value = cm_plot[
                i,
                j,
            ]

            text = (
                f"{value:.2f}"
                if normalised
                else
                f"{int(value)}"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    return fig, cm


def save_epoch_confusion_matrices(
    y_true,
    y_pred,
    epoch: int,
    output_dir: Path,
    writer: SummaryWriter,
):

    counts_fig, _ = make_confusion_matrix_figure(
        y_true,
        y_pred,
        normalised=False,
        title=(
            f"Validation confusion matrix "
            f"(counts) - epoch {epoch}"
        ),
    )

    normalised_fig, _ = make_confusion_matrix_figure(
        y_true,
        y_pred,
        normalised=True,
        title=(
            f"Validation confusion matrix "
            f"(normalised) - epoch {epoch}"
        ),
    )

    counts_path = (
        output_dir
        / "confusion_matrices"
        / f"epoch_{epoch:03d}_counts.png"
    )

    normalised_path = (
        output_dir
        / "confusion_matrices"
        / f"epoch_{epoch:03d}_normalised.png"
    )

    counts_fig.savefig(
        counts_path,
        dpi=160,
        bbox_inches="tight",
    )

    normalised_fig.savefig(
        normalised_path,
        dpi=160,
        bbox_inches="tight",
    )

    writer.add_figure(
        "ConfusionMatrix/val_counts",
        counts_fig,
        global_step=epoch,
        close=False,
    )

    writer.add_figure(
        "ConfusionMatrix/val_normalised",
        normalised_fig,
        global_step=epoch,
        close=False,
    )

    plt.close(
        counts_fig
    )

    plt.close(
        normalised_fig
    )


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp,
):

    model.train()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for (
        x,
        y,
        _,
    ) in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits, _ = model(
                x
            )

            loss = criterion(
                logits,
                y,
            )

        if use_amp:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        running_loss += (
            loss.item()
            *
            x.size(0)
        )

        pred = logits.argmax(
            dim=1
        )

        y_true.extend(
            y.detach()
            .cpu()
            .tolist()
        )

        y_pred.extend(
            pred.detach()
            .cpu()
            .tolist()
        )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = (
        running_loss
        /
        len(loader.dataset)
    )

    return metrics


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    use_amp,
):

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []

    records = []

    for (
        x,
        y,
        metadata,
    ) in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits, attention = model(
                x
            )

            loss = criterion(
                logits,
                y,
            )

        running_loss += (
            loss.item()
            *
            x.size(0)
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        pred = logits.argmax(
            dim=1
        )

        y_cpu = y.cpu().numpy()
        pred_cpu = pred.cpu().numpy()
        prob_cpu = probabilities.cpu().numpy()

        y_true.extend(
            y_cpu.tolist()
        )

        y_pred.extend(
            pred_cpu.tolist()
        )

        batch_size = len(
            y_cpu
        )

        for i in range(
            batch_size
        ):

            record = {
                "flight_id":
                    metadata[
                        "flight_id"
                    ][i],
                "segment_uid":
                    metadata[
                        "segment_uid"
                    ][i],
                "window_start_sec":
                    float(
                        metadata[
                            "window_start_sec"
                        ][i]
                    ),
                "true_id":
                    int(
                        y_cpu[i]
                    ),
                "pred_id":
                    int(
                        pred_cpu[i]
                    ),
                "true_state":
                    ID_TO_LABEL[
                        int(
                            y_cpu[i]
                        )
                    ],
                "pred_state":
                    ID_TO_LABEL[
                        int(
                            pred_cpu[i]
                        )
                    ],
            }

            for class_id in range(
                NUM_CLASSES
            ):

                record[
                    f"prob_{ID_TO_LABEL[class_id]}"
                ] = float(
                    prob_cpu[
                        i,
                        class_id,
                    ]
                )

            records.append(
                record
            )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = (
        running_loss
        /
        len(loader.dataset)
    )

    return (
        metrics,
        np.asarray(
            y_true
        ),
        np.asarray(
            y_pred
        ),
        pd.DataFrame(
            records
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    parser.add_argument(
        "--telemetry-csv",
        type=Path,
        default=DEFAULT_TELEMETRY,
    )

    parser.add_argument(
        "--flight-splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )

    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
    )

    # ------------------------------------------------------------------
    # Online sampling
    # ------------------------------------------------------------------

    parser.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=1500,
    )

    # ------------------------------------------------------------------
    # Model - Ex3 binary defaults
    # ------------------------------------------------------------------

    parser.add_argument(
        "--patch-len",
        type=int,
        default=2,
        help=(
            "Patch length in telemetry timesteps. "
            "At 10 Hz, patch_len=2 means 0.2 s."
        ),
    )

    parser.add_argument(
        "--patch-stride",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ff-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.40,
    )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    # ------------------------------------------------------------------
    # Experiment directory
    # ------------------------------------------------------------------

    project_root = Path(
        __file__
    ).resolve().parents[2]

    output_dir = create_experiment_directory(
        project_root
    )

    print()
    print("=" * 80)
    print("EXPERIMENT")
    print("=" * 80)

    print(
        f"Job name             : "
        f"{JOB_NAME}"
    )

    print(
        f"Job ID               : "
        f"{JOB_ID}"
    )

    print(
        f"Experiment directory : "
        f"{output_dir}"
    )

    # ------------------------------------------------------------------
    # Runtime device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
    )

    print()
    print("=" * 80)
    print("MACE AIRCRAFT-ONLY TEMPORAL BASELINE - EX3 BINARY DYNAMICS-8")
    print("Binary target        : low = approaching/entering; high = inside/exiting")
    print("=" * 80)

    print(
        f"Torch               : "
        f"{torch.__version__}"
    )

    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"CUDA available      : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            "GPU                 : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"AMP                 : "
        f"{use_amp}"
    )

    # ------------------------------------------------------------------
    # TensorBoard / config
    # ------------------------------------------------------------------

    writer = SummaryWriter(
        log_dir=str(
            output_dir
            /
            "tensorboard"
        )
    )

    config = vars(
        args
    ).copy()

    config[
        "model_feature_names"
    ] = MODEL_FEATURE_NAMES

    config[
        "experiment_type"
    ] = "Dynamics-8 TimeMIL-inspired Patch Transformer"

    config[
        "slurm_job_name"
    ] = JOB_NAME

    config[
        "slurm_job_id"
    ] = JOB_ID

    with (
        output_dir
        /
        "config.json"
    ).open("w") as f:

        json.dump(
            config,
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # Load split
    # ------------------------------------------------------------------

    split_lookup = load_flight_splits(
        args.flight_splits
    )

    print()
    print("Flight split")
    print("-" * 80)

    for split in [
        "train",
        "val",
        "test",
    ]:

        flights = sorted(
            [
                flight
                for flight, value
                in split_lookup.items()
                if value == split
            ]
        )

        print(
            f"{split:5s}: "
            +
            " ".join(
                flights
            )
        )

    # ------------------------------------------------------------------
    # Load telemetry
    # ------------------------------------------------------------------

    df = load_telemetry(
        args.telemetry_csv,
        split_lookup,
    )

    print()
    print(
        f"Telemetry rows      : "
        f"{len(df):,}"
    )

    print(
        f"Flights             : "
        f"{df['flight_id'].nunique()}"
    )

    print()
    print("Rows by split/state:")

    print(
        pd.crosstab(
            df["split"],
            df["state"],
        )
    )

    # ------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------

    window_steps = int(
        round(
            args.window_sec
            *
            args.rate_hz
        )
    )

    segments = build_segments(
        df=df,
        rate_hz=args.rate_hz,
        window_sec=args.window_sec,
    )

    print()
    print(
        f"Window              : "
        f"{args.window_sec:.1f}s"
    )

    print(
        f"Window timesteps    : "
        f"{window_steps}"
    )

    print(
        f"Model features      : "
        f"{len(MODEL_FEATURE_NAMES)}"
    )

    print(
        "Feature names       : "
        +
        ", ".join(
            MODEL_FEATURE_NAMES
        )
    )

    print(
        "Attitude handling   : "
        "Roll/Pitch re-referenced within each 5-s window"
    )

    # ------------------------------------------------------------------
    # Train-only normalisation
    # ------------------------------------------------------------------

    mean, std = calculate_train_statistics(
        segments=segments,
        window_steps=window_steps,
        seed=args.seed,
    )

    stats = {
        name: {
            "mean": float(
                mean[i]
            ),
            "std": float(
                std[i]
            ),
        }
        for i, name
        in enumerate(
            MODEL_FEATURE_NAMES
        )
    }

    with (
        output_dir
        /
        "normalization_stats.json"
    ).open("w") as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    train_dataset = OnlineBalancedTelemetryDataset(
        segments=segments,
        split="train",
        samples_per_epoch=
            args.train_samples_per_epoch,
        window_steps=
            window_steps,
        mean=mean,
        std=std,
        augment=True,
        seed=args.seed,
    )

    val_dataset = FixedBalancedTelemetryDataset(
        segments=segments,
        split="val",
        samples=
            args.val_samples,
        window_steps=
            window_steps,
        mean=mean,
        std=std,
        seed=args.seed + 1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=
            args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=
            args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = TimeMILPatchTransformer(
        num_features=
            len(
                MODEL_FEATURE_NAMES
            ),
        window_steps=
            window_steps,
        patch_len=
            args.patch_len,
        patch_stride=
            args.patch_stride,
        d_model=
            args.d_model,
        num_heads=
            args.num_heads,
        num_layers=
            args.num_layers,
        ff_dim=
            args.ff_dim,
        dropout=
            args.dropout,
        num_classes=
            NUM_CLASSES,
    ).to(
        device
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Model")
    print("-" * 80)

    print(
        model
    )

    print()
    print(
        f"Total parameters    : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Patches/window      : "
        f"{model.num_patches}"
    )

    print(
        f"Train samples/epoch : "
        f"{args.train_samples_per_epoch:,}"
    )

    print(
        f"Validation samples  : "
        f"{len(val_dataset):,}"
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=
            args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=
            args.learning_rate,
        weight_decay=
            args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=
            args.epochs,
        eta_min=(
            args.learning_rate
            *
            0.01
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=
            use_amp,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = time.time()

        train_dataset.set_epoch(
            epoch
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        (
            val_metrics,
            val_true,
            val_pred,
            val_predictions,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        elapsed = (
            time.time()
            -
            epoch_start
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} "
            f"bal_acc {train_metrics['balanced_accuracy']:.4f} "
            f"F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"acc {val_metrics['accuracy']:.4f} "
            f"bal_acc {val_metrics['balanced_accuracy']:.4f} "
            f"F1 {val_metrics['macro_f1']:.4f} | "
            f"lr {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # -------------------------------------------------------------
        # TensorBoard scalars
        # -------------------------------------------------------------

        writer.add_scalar(
            "Loss/train",
            train_metrics["loss"],
            epoch,
        )

        writer.add_scalar(
            "Loss/val",
            val_metrics["loss"],
            epoch,
        )

        writer.add_scalar(
            "Accuracy/train",
            train_metrics["accuracy"],
            epoch,
        )

        writer.add_scalar(
            "Accuracy/val",
            val_metrics["accuracy"],
            epoch,
        )

        writer.add_scalar(
            "BalancedAccuracy/train",
            train_metrics[
                "balanced_accuracy"
            ],
            epoch,
        )

        writer.add_scalar(
            "BalancedAccuracy/val",
            val_metrics[
                "balanced_accuracy"
            ],
            epoch,
        )

        writer.add_scalar(
            "MacroF1/train",
            train_metrics["macro_f1"],
            epoch,
        )

        writer.add_scalar(
            "MacroF1/val",
            val_metrics["macro_f1"],
            epoch,
        )

        writer.add_scalar(
            "LearningRate",
            current_lr,
            epoch,
        )

        if torch.cuda.is_available():

            writer.add_scalar(
                "GPU/memory_allocated_gb",
                (
                    torch.cuda.memory_allocated()
                    /
                    1024**3
                ),
                epoch,
            )

            writer.add_scalar(
                "GPU/memory_reserved_gb",
                (
                    torch.cuda.memory_reserved()
                    /
                    1024**3
                ),
                epoch,
            )

        # -------------------------------------------------------------
        # Per-class validation metrics
        # -------------------------------------------------------------

        per_class_report = classification_report(
            val_true,
            val_pred,
            labels=list(
                range(
                    NUM_CLASSES
                )
            ),
            target_names=[
                ID_TO_LABEL[i]
                for i in range(
                    NUM_CLASSES
                )
            ],
            output_dict=True,
            zero_division=0,
        )

        for class_name in [
            ID_TO_LABEL[i]
            for i in range(
                NUM_CLASSES
            )
        ]:

            writer.add_scalar(
                f"PerClassF1/val_{class_name}",
                per_class_report[
                    class_name
                ]["f1-score"],
                epoch,
            )

            writer.add_scalar(
                f"PerClassPrecision/val_{class_name}",
                per_class_report[
                    class_name
                ]["precision"],
                epoch,
            )

            writer.add_scalar(
                f"PerClassRecall/val_{class_name}",
                per_class_report[
                    class_name
                ]["recall"],
                epoch,
            )

        # -------------------------------------------------------------
        # Validation confusion matrices every epoch
        # -------------------------------------------------------------

        save_epoch_confusion_matrices(
            y_true=val_true,
            y_pred=val_pred,
            epoch=epoch,
            output_dir=output_dir,
            writer=writer,
        )

        # -------------------------------------------------------------
        # Validation predictions every epoch
        # -------------------------------------------------------------

        val_predictions.to_csv(
            output_dir
            / "predictions"
            / f"epoch_{epoch:03d}_validation_predictions.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # History
        # -------------------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "learning_rate":
                    current_lr,
                "train_loss":
                    train_metrics[
                        "loss"
                    ],
                "train_accuracy":
                    train_metrics[
                        "accuracy"
                    ],
                "train_balanced_accuracy":
                    train_metrics[
                        "balanced_accuracy"
                    ],
                "train_macro_f1":
                    train_metrics[
                        "macro_f1"
                    ],
                "train_weighted_f1":
                    train_metrics[
                        "weighted_f1"
                    ],
                "val_loss":
                    val_metrics[
                        "loss"
                    ],
                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],
                "val_balanced_accuracy":
                    val_metrics[
                        "balanced_accuracy"
                    ],
                "val_macro_f1":
                    val_metrics[
                        "macro_f1"
                    ],
                "val_weighted_f1":
                    val_metrics[
                        "weighted_f1"
                    ],
                "epoch_seconds":
                    elapsed,
                "cuda_memory_allocated_gb":
                    (
                        torch.cuda.memory_allocated()
                        /
                        1024**3
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                "cuda_memory_reserved_gb":
                    (
                        torch.cuda.memory_reserved()
                        /
                        1024**3
                        if torch.cuda.is_available()
                        else 0.0
                    ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_dir
            /
            "training_history.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------

        checkpoint = {
            "epoch":
                epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],
            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],
            "feature_names":
                MODEL_FEATURE_NAMES,
            "normalization_mean":
                mean,
            "normalization_std":
                std,
            "label_map":
                LABEL_MAP,
            "args":
                vars(
                    args
                ),
        }

        torch.save(
            checkpoint,
            output_dir
            / "checkpoints"
            / "last.pt",
        )

        if epoch % 10 == 0:

            torch.save(
                checkpoint,
                output_dir
                / "checkpoints"
                / f"epoch_{epoch:03d}.pt",
            )

        # -------------------------------------------------------------
        # Best model
        # -------------------------------------------------------------

        if (
            val_metrics[
                "macro_f1"
            ]
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                val_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch
            patience_counter = 0

            torch.save(
                checkpoint,
                output_dir
                / "checkpoints"
                / "best.pt",
            )

            val_predictions.to_csv(
                output_dir
                / "predictions"
                / "best_validation_predictions.csv",
                index=False,
            )

            # ---------------------------------------------------------
            # Best confusion matrices
            # ---------------------------------------------------------

            best_counts_fig, best_cm = make_confusion_matrix_figure(
                val_true,
                val_pred,
                normalised=False,
                title=(
                    f"Best validation confusion matrix "
                    f"(counts) - epoch {epoch}"
                ),
            )

            best_norm_fig, _ = make_confusion_matrix_figure(
                val_true,
                val_pred,
                normalised=True,
                title=(
                    f"Best validation confusion matrix "
                    f"(normalised) - epoch {epoch}"
                ),
            )

            best_counts_fig.savefig(
                output_dir
                / "confusion_matrices"
                / "best_counts.png",
                dpi=180,
                bbox_inches="tight",
            )

            best_norm_fig.savefig(
                output_dir
                / "confusion_matrices"
                / "best_normalised.png",
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(
                best_counts_fig
            )

            plt.close(
                best_norm_fig
            )

            pd.DataFrame(
                best_cm,
                index=[
                    ID_TO_LABEL[i]
                    for i in range(
                        NUM_CLASSES
                    )
                ],
                columns=[
                    ID_TO_LABEL[i]
                    for i in range(
                        NUM_CLASSES
                    )
                ],
            ).to_csv(
                output_dir
                / "confusion_matrices"
                / "confusion_matrix_best.csv"
            )

            # ---------------------------------------------------------
            # Best classification report
            # ---------------------------------------------------------

            report_text = classification_report(
                val_true,
                val_pred,
                labels=list(
                    range(
                        NUM_CLASSES
                    )
                ),
                target_names=[
                    ID_TO_LABEL[i]
                    for i in range(
                        NUM_CLASSES
                    )
                ],
                digits=4,
                zero_division=0,
            )

            with (
                output_dir
                / "reports"
                / "classification_report_best.txt"
            ).open("w") as f:

                f.write(
                    report_text
                )

            best_metrics = {
                "epoch":
                    epoch,
                "val_loss":
                    float(
                        val_metrics[
                            "loss"
                        ]
                    ),
                "val_accuracy":
                    float(
                        val_metrics[
                            "accuracy"
                        ]
                    ),
                "val_balanced_accuracy":
                    float(
                        val_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                "val_macro_f1":
                    float(
                        val_metrics[
                            "macro_f1"
                        ]
                    ),
                "val_weighted_f1":
                    float(
                        val_metrics[
                            "weighted_f1"
                        ]
                    ),
                "per_class":
                    per_class_report,
            }

            with (
                output_dir
                / "reports"
                / "best_validation_metrics.json"
            ).open("w") as f:

                json.dump(
                    best_metrics,
                    f,
                    indent=2,
                )

            print(
                f"  -> NEW BEST: "
                f"epoch {epoch}, "
                f"val Macro F1 "
                f"{best_macro_f1:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"  -> no improvement "
                f"({patience_counter}/"
                f"{args.patience})"
            )

        # Advance cosine scheduler after recording current LR.
        scheduler.step()

        writer.flush()

        if (
            patience_counter
            >=
            args.patience
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    writer.close()

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best val Macro F1 : "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Output directory  : "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
