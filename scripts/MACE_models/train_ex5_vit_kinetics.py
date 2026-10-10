#!/usr/bin/env python3
"""
===============================================================================
train_ex5_vit_kinetics.py
===============================================================================

Production training script for MACE Experiment 5 using:

    MACE_Temporal_label_complete.xlsx
        -> flight-level train/validation split
        -> balanced online temporal sampling
        -> direct OpenCV video decoding
        -> 16 uniformly sampled RGB frames per five-second window
        -> Kinetics-400-pretrained MViTv2-S video transformer
        -> four-class interaction-state classifier

Classes:
    0: approaching
    1: entering
    2: inside
    3: exiting

Experiment 4 used:

    25 frames
        -> ImageNet-pretrained ViT-B/16 applied independently to each frame
        -> temporal mean pooling
        -> four MACE classes

Experiment 5 uses:

    16 frames
        -> Kinetics-400-pretrained MViTv2-S applied jointly to the video clip
        -> spatiotemporal video representation
        -> four MACE classes

The training loop, validation loop, TensorBoard logging, CSV logging,
checkpointing, early stopping and online-sampling policy intentionally follow
train_ex4_online_final.py so the comparison remains controlled.

This script uses dataset_online_final.py, not dataset_online_final_v2.py.
Therefore, minimum-spacing sampling is not introduced as an additional
experimental variable.

Expected project structure:

    MACE_Temporal_Label_Flights/
    ├── Flights/
    ├── data/
    │   └── annotations/
    │       ├── MACE_Temporal_label_complete.xlsx
    │       └── flight_splits.csv
    ├── scripts/
    │   └── MACE_models/
    │       ├── dataset_online_final.py
    │       └── train_ex5_vit_kinetics.py
    └── experiments/

Run:

    python scripts/MACE_models/train_ex5_vit_kinetics.py

Optional overrides:

    python scripts/MACE_models/train_ex5_vit_kinetics.py \
        --batch-size 16 \
        --num-workers 4 \
        --train-samples 12000 \
        --val-samples 2000

Optional resume:

    python scripts/MACE_models/train_ex5_vit_kinetics.py \
        --resume experiments/<experiment_name>/checkpoints/last.pt

===============================================================================
"""

from __future__ import annotations


# =============================================================================
# HPC-safe environment configuration
# =============================================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================
# Standard-library imports
# =============================================================================

import argparse
import csv
import json
import math
import random
import sys
import time

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Third-party imports
# =============================================================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.models.video import (
    MViT_V2_S_Weights,
    mvit_v2_s,
)


# =============================================================================
# Slurm experiment identifiers
# =============================================================================

# Use Slurm values when running inside an sbatch job.
# The fallbacks allow --help and small checks on a login node.

JOB_NAME = os.environ["SLURM_JOB_NAME"]

JOB_ID = os.environ["SLURM_JOB_ID"]

TIMESTAMP = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)


# =============================================================================
# Local imports
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )

from dataset_online_final import (  # noqa: E402
    ID_TO_LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
    OnlineTemporalVisionDataset,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Complete Experiment 5 configuration."""

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    train_samples_per_epoch: int = 12_000
    val_samples: int = 2_000

    window_seconds: float = 5.0

    # MViTv2-S Kinetics-400 uses 16-frame clips.
    frames_per_sample: int = 16

    image_size: int = 224

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Must remain False.
    #
    # start_new_epoch() changes the dataset's current epoch and balanced class
    # schedule in the main process. Recreating workers for each new iterator
    # ensures workers receive the updated dataset state.
    persistent_workers: bool = False

    max_cached_videos: int = 2

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    number_of_classes: int = 4
    dropout: float = 0.5
    freeze_backbone: bool = False

    # dataset_online_final.py currently applies ImageNet normalisation.
    # The model wrapper reverses it and applies the official Kinetics
    # normalisation before passing clips to MViTv2-S.
    kinetics_mean: tuple[float, float, float] = (
        0.45,
        0.45,
        0.45,
    )

    kinetics_std: tuple[float, float, float] = (
        0.225,
        0.225,
        0.225,
    )

    # -------------------------------------------------------------------------
    # Optimisation
    # -------------------------------------------------------------------------

    epochs: int = 150

    # Use a smaller learning rate for the pretrained video backbone and a
    # larger one for the newly initialised four-class classifier.
    backbone_learning_rate: float = 1e-5
    classifier_learning_rate: float = 1e-4

    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # -------------------------------------------------------------------------
    # Scheduling and stopping
    # -------------------------------------------------------------------------

    cosine_eta_min: float = 1e-6
    patience: int = 30
    minimum_delta: float = 1e-4

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    seed: int = 42
    use_amp: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True

    # -------------------------------------------------------------------------
    # Logging and checkpointing
    # -------------------------------------------------------------------------

    experiment_prefix: str = (
        JOB_NAME
        + "_"
        + JOB_ID
        + "_"
        + TIMESTAMP
    )

    log_confusion_every: int = 1
    save_every: int = 10


# =============================================================================
# Reproducibility
# =============================================================================


def seed_everything(
    seed: int,
) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(
    worker_id: int,
) -> None:
    """Seed each DataLoader worker deterministically."""

    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Model
# =============================================================================


class KineticsMViTTemporalClassifier(nn.Module):
    """
    Kinetics-400-pretrained MViTv2-S temporal video classifier.

    Input:
        [B, T, C, H, W]

    For this experiment:
        T = 16
        C = 3
        H = W = 224

    Processing:
        1. Validate the input tensor.
        2. Reverse the ImageNet normalisation applied by
           dataset_online_final.py.
        3. Apply the official Kinetics normalisation.
        4. Rearrange [B,T,C,H,W] into [B,C,T,H,W].
        5. Process all 16 frames jointly with MViTv2-S.
        6. Apply LayerNorm, dropout and a four-class linear classifier.

    Output:
        [B, 4] logits
    """

    def __init__(
        self,
        number_of_classes: int,
        dropout: float,
        freeze_backbone: bool = False,
        kinetics_mean: tuple[float, float, float] = (
            0.45,
            0.45,
            0.45,
        ),
        kinetics_std: tuple[float, float, float] = (
            0.225,
            0.225,
            0.225,
        ),
    ) -> None:
        super().__init__()

        weights = (
            MViT_V2_S_Weights.KINETICS400_V1
        )

        try:
            backbone = mvit_v2_s(
                weights=weights,
            )

        except Exception as error:
            raise RuntimeError(
                "Could not load the Kinetics-400-pretrained "
                "MViTv2-S weights. The compute node may not have "
                "internet access and the checkpoint may not yet be "
                "present in the PyTorch cache. Load the weights once "
                "on a connected node or copy the cached checkpoint "
                "into ~/.cache/torch/hub/checkpoints/."
            ) from error

        # torchvision MViTv2-S normally stores its classification head as:
        #
        #     Sequential(
        #         Dropout(...),
        #         Linear(feature_dimension, 400)
        #     )
        #
        # Find the original final linear layer so we can recover the feature
        # dimension without hard-coding it.

        original_linear_layer: nn.Linear | None = None

        if isinstance(
            backbone.head,
            nn.Sequential,
        ):
            for layer in reversed(
                backbone.head
            ):
                if isinstance(
                    layer,
                    nn.Linear,
                ):
                    original_linear_layer = layer
                    break

        elif isinstance(
            backbone.head,
            nn.Linear,
        ):
            original_linear_layer = (
                backbone.head
            )

        if original_linear_layer is None:
            raise RuntimeError(
                "Could not identify the original MViTv2-S "
                "classification layer."
            )

        embedding_dimension = int(
            original_linear_layer.in_features
        )

        # Remove the original 400-class Kinetics classifier while retaining
        # the pretrained spatiotemporal video representation.
        backbone.head = nn.Identity()

        self.backbone = backbone
        self.embedding_dimension = (
            embedding_dimension
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                embedding_dimension,
            ),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                embedding_dimension,
                number_of_classes,
            ),
        )

        # ---------------------------------------------------------------------
        # Input-normalisation buffers
        # ---------------------------------------------------------------------
        #
        # dataset_online_final.py returns:
        #
        #     x_imagenet = (x_rgb - imagenet_mean) / imagenet_std
        #
        # We reconstruct x_rgb and then calculate:
        #
        #     x_kinetics = (x_rgb - kinetics_mean) / kinetics_std
        #
        # Buffers move automatically with model.to(device) and are stored in
        # checkpoints, but they are not trainable parameters.

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(
                IMAGENET_MEAN,
                dtype=torch.float32,
            ).view(
                1,
                1,
                3,
                1,
                1,
            ),
        )

        self.register_buffer(
            "imagenet_std",
            torch.tensor(
                IMAGENET_STD,
                dtype=torch.float32,
            ).view(
                1,
                1,
                3,
                1,
                1,
            ),
        )

        self.register_buffer(
            "kinetics_mean",
            torch.tensor(
                kinetics_mean,
                dtype=torch.float32,
            ).view(
                1,
                1,
                3,
                1,
                1,
            ),
        )

        self.register_buffer(
            "kinetics_std",
            torch.tensor(
                kinetics_std,
                dtype=torch.float32,
            ).view(
                1,
                1,
                3,
                1,
                1,
            ),
        )

        if freeze_backbone:
            for parameter in (
                self.backbone.parameters()
            ):
                parameter.requires_grad = False

    def _prepare_video(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert dataset output to the format expected by MViTv2-S.

        Dataset output:
            [B, T, C, H, W]
            ImageNet-normalised

        MViTv2-S input:
            [B, C, T, H, W]
            Kinetics-normalised
        """

        if frames.ndim != 5:
            raise ValueError(
                "Expected a five-dimensional tensor "
                f"[B,T,C,H,W], received {tuple(frames.shape)}."
            )

        (
            _,
            sequence_length,
            channels,
            height,
            width,
        ) = frames.shape

        if sequence_length != 16:
            raise ValueError(
                "Experiment 5 requires exactly 16 frames per "
                f"sample, received {sequence_length}."
            )

        if channels != 3:
            raise ValueError(
                "Expected RGB input with three channels, "
                f"received {channels}."
            )

        if (
            height != 224
            or width != 224
        ):
            raise ValueError(
                "MViTv2-S Experiment 5 requires 224x224 "
                f"frames, received {height}x{width}."
            )

        frames = frames.float()

        # Reverse the ImageNet normalisation performed by the dataset.
        rgb_zero_to_one = (
            frames
            * self.imagenet_std
            + self.imagenet_mean
        )

        # Noise or floating-point operations can create very small values
        # outside the expected range. Clamp defensively.
        rgb_zero_to_one = (
            rgb_zero_to_one.clamp(
                min=0.0,
                max=1.0,
            )
        )

        # Apply the normalisation used by the official Kinetics weights.
        kinetics_frames = (
            rgb_zero_to_one
            - self.kinetics_mean
        ) / self.kinetics_std

        # Convert:
        #
        #     [B, T, C, H, W]
        #
        # to:
        #
        #     [B, C, T, H, W]
        #
        # which is the torchvision video-model layout.
        kinetics_video = (
            kinetics_frames.permute(
                0,
                2,
                1,
                3,
                4,
            )
            .contiguous()
        )

        return kinetics_video

    def forward(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Return four-class MACE logits."""

        video = self._prepare_video(
            frames
        )

        clip_embedding = self.backbone(
            video
        )

        logits = self.classifier(
            clip_embedding
        )

        return logits


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class EpochMetrics:
    """Metrics produced by one complete train or validation pass."""

    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    precision_per_class: list[float]
    recall_per_class: list[float]
    f1_per_class: list[float]
    support_per_class: list[int]
    confusion_matrix: list[list[int]]


class RunningAverage:
    """Sample-weighted running mean."""

    def __init__(
        self,
    ) -> None:
        self.total = 0.0
        self.count = 0

    def update(
        self,
        value: float,
        number_of_samples: int,
    ) -> None:
        self.total += (
            float(value)
            * int(number_of_samples)
        )

        self.count += int(
            number_of_samples
        )

    @property
    def average(
        self,
    ) -> float:
        if self.count == 0:
            return 0.0

        return (
            self.total
            / self.count
        )


def calculate_epoch_metrics(
    average_loss: float,
    targets: list[int],
    predictions: list[int],
) -> EpochMetrics:
    """Calculate the complete classification metric set."""

    labels = list(
        range(
            len(LABEL_MAP)
        )
    )

    targets_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions_array = np.asarray(
        predictions,
        dtype=np.int64,
    )

    accuracy = float(
        np.mean(
            targets_array
            == predictions_array
        )
    )

    macro_f1 = float(
        f1_score(
            targets_array,
            predictions_array,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            targets_array,
            predictions_array,
        )
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets_array,
        predictions_array,
        labels=labels,
    )

    return EpochMetrics(
        loss=float(
            average_loss
        ),
        accuracy=accuracy,
        macro_f1=macro_f1,
        balanced_accuracy=(
            balanced_accuracy
        ),
        precision_per_class=(
            precision.tolist()
        ),
        recall_per_class=(
            recall.tolist()
        ),
        f1_per_class=(
            f1.tolist()
        ),
        support_per_class=(
            support.astype(int).tolist()
        ),
        confusion_matrix=(
            matrix.astype(int).tolist()
        ),
    )


# =============================================================================
# END OF PART 1
# Paste Part 2 directly below this line.
# =============================================================================

# =============================================================================
# Plotting
# =============================================================================


def create_confusion_matrix_figure(
    matrix: np.ndarray,
    title: str,
) -> plt.Figure:
    """Create an unnormalised confusion matrix figure."""

    class_names = [
        ID_TO_LABEL[index]
        for index in range(len(ID_TO_LABEL))
    ]

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(matrix)

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Ground truth",
        xlabel="Prediction",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size
        else 0.0
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    return figure


def create_normalised_confusion_matrix_figure(
    matrix: np.ndarray,
    title: str,
) -> plt.Figure:
    """Create a row-normalised confusion matrix figure."""

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )

    class_names = [
        ID_TO_LABEL[index]
        for index in range(len(ID_TO_LABEL))
    ]

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        normalised,
        vmin=0.0,
        vmax=1.0,
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Ground truth",
        xlabel="Prediction",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    for row in range(normalised.shape[0]):
        for column in range(normalised.shape[1]):
            value = normalised[row, column]

            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > 0.5
                    else "black"
                ),
            )

    figure.tight_layout()

    return figure


# =============================================================================
# Device and AMP helpers
# =============================================================================


def configure_device(
    config: TrainingConfig,
) -> torch.device:
    """Configure CUDA runtime and return the selected device."""

    if torch.cuda.is_available():
        device = torch.device("cuda")

        torch.backends.cudnn.benchmark = (
            config.cudnn_benchmark
        )

        if hasattr(
            torch.backends.cuda.matmul,
            "allow_tf32",
        ):
            torch.backends.cuda.matmul.allow_tf32 = (
                config.allow_tf32
            )

        if hasattr(
            torch.backends.cudnn,
            "allow_tf32",
        ):
            torch.backends.cudnn.allow_tf32 = (
                config.allow_tf32
            )

        return device

    return torch.device("cpu")


def amp_context(
    device: torch.device,
    enabled: bool,
):
    """Return the appropriate mixed-precision context manager."""

    if (
        device.type != "cuda"
        or not enabled
    ):
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    )


def create_grad_scaler(
    device: torch.device,
    enabled: bool,
):
    """Create a GradScaler compatible with recent PyTorch versions."""

    scaler_enabled = (
        enabled
        and device.type == "cuda"
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
        )

    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=scaler_enabled,
        )


# =============================================================================
# Train and validation loops
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> EpochMetrics:
    """Run one complete training epoch."""

    model.train()

    loss_meter = RunningAverage()

    targets: list[int] = []
    predictions: list[int] = []

    optimizer.zero_grad(
        set_to_none=True
    )

    number_of_batches = len(loader)

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        frames = batch["frames"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        batch_size = labels.shape[0]

        with amp_context(
            device=device,
            enabled=config.use_amp,
        ):
            logits = model(frames)

            raw_loss = criterion(
                logits,
                labels,
            )

            loss = (
                raw_loss
                / config.gradient_accumulation_steps
            )

        scaler.scale(loss).backward()

        should_step = (
            batch_index
            % config.gradient_accumulation_steps
            == 0
            or batch_index == number_of_batches
        )

        if should_step:
            if config.gradient_clip_norm > 0.0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.gradient_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        batch_predictions = logits.argmax(
            dim=1
        )

        loss_meter.update(
            raw_loss.item(),
            batch_size,
        )

        targets.extend(
            labels.detach().cpu().tolist()
        )

        predictions.extend(
            batch_predictions.detach().cpu().tolist()
        )

        if (
            batch_index == 1
            or batch_index % 50 == 0
            or batch_index == number_of_batches
        ):
            current_accuracy = float(
                (
                    batch_predictions
                    == labels
                )
                .float()
                .mean()
                .item()
            )

            print(
                f"Epoch {epoch + 1:03d} "
                f"| train batch "
                f"{batch_index:04d}/{number_of_batches:04d} "
                f"| loss {raw_loss.item():.4f} "
                f"| batch acc {current_accuracy:.4f}",
                flush=True,
            )

    return calculate_epoch_metrics(
        average_loss=loss_meter.average,
        targets=targets,
        predictions=predictions,
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> EpochMetrics:
    """Run one complete fixed validation pass."""

    model.eval()

    loss_meter = RunningAverage()

    targets: list[int] = []
    predictions: list[int] = []

    number_of_batches = len(loader)

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        frames = batch["frames"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with amp_context(
            device=device,
            enabled=config.use_amp,
        ):
            logits = model(frames)

            loss = criterion(
                logits,
                labels,
            )

        batch_predictions = logits.argmax(
            dim=1
        )

        loss_meter.update(
            loss.item(),
            labels.shape[0],
        )

        targets.extend(
            labels.cpu().tolist()
        )

        predictions.extend(
            batch_predictions.cpu().tolist()
        )

        if (
            batch_index == 1
            or batch_index % 25 == 0
            or batch_index == number_of_batches
        ):
            print(
                f"Epoch {epoch + 1:03d} "
                f"| val batch "
                f"{batch_index:04d}/{number_of_batches:04d}",
                flush=True,
            )

    return calculate_epoch_metrics(
        average_loss=loss_meter.average,
        targets=targets,
        predictions=predictions,
    )


# =============================================================================
# Logging
# =============================================================================


CSV_COLUMNS = [
    "epoch",
    "backbone_learning_rate",
    "classifier_learning_rate",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_balanced_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
    "epoch_seconds",
    "cuda_memory_allocated_gb",
    "cuda_memory_reserved_gb",
]


def initialise_csv(
    csv_path: Path,
) -> None:
    """Create the CSV history file if it does not exist."""

    if csv_path.exists():
        return

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()


def append_csv_row(
    csv_path: Path,
    row: dict[str, Any],
) -> None:
    """Append one epoch to the CSV history."""

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
        )

        writer.writerow(row)


def log_metrics_to_tensorboard(
    tensorboard_writer: SummaryWriter,
    split: str,
    metrics: EpochMetrics,
    epoch_number: int,
) -> None:
    """Log scalar and per-class metrics."""

    tensorboard_writer.add_scalar(
        f"Loss/{split}",
        metrics.loss,
        epoch_number,
    )

    tensorboard_writer.add_scalar(
        f"Accuracy/{split}",
        metrics.accuracy,
        epoch_number,
    )

    tensorboard_writer.add_scalar(
        f"MacroF1/{split}",
        metrics.macro_f1,
        epoch_number,
    )

    tensorboard_writer.add_scalar(
        f"BalancedAccuracy/{split}",
        metrics.balanced_accuracy,
        epoch_number,
    )

    for class_id in range(
        len(ID_TO_LABEL)
    ):
        class_name = ID_TO_LABEL[class_id]

        tensorboard_writer.add_scalar(
            f"PerClassF1/{split}/{class_name}",
            metrics.f1_per_class[class_id],
            epoch_number,
        )

        tensorboard_writer.add_scalar(
            f"PerClassRecall/{split}/{class_name}",
            metrics.recall_per_class[class_id],
            epoch_number,
        )

        tensorboard_writer.add_scalar(
            f"PerClassPrecision/{split}/{class_name}",
            metrics.precision_per_class[class_id],
            epoch_number,
        )


# =============================================================================
# Checkpointing
# =============================================================================


def unwrap_model(
    model: nn.Module,
) -> nn.Module:
    """Return the underlying model when DataParallel is used."""

    if isinstance(
        model,
        nn.DataParallel,
    ):
        return model.module

    return model


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    config: TrainingConfig,
    epoch: int,
    best_val_macro_f1: float,
    epochs_without_improvement: int,
) -> dict[str, Any]:
    """Create a complete resumable checkpoint."""

    return {
        "epoch": int(epoch),

        "model_state_dict": (
            unwrap_model(model)
            .state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
        ),

        "scaler_state_dict": (
            scaler.state_dict()
        ),

        "best_val_macro_f1": float(
            best_val_macro_f1
        ),

        "epochs_without_improvement": int(
            epochs_without_improvement
        ),

        "config": asdict(config),
        "label_map": LABEL_MAP,
        "timestamp": datetime.now().isoformat(),
    }


def save_checkpoint(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Save checkpoint atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    device: torch.device,
) -> tuple[int, float, int]:
    """Restore model and complete training state."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = (
        int(checkpoint["epoch"])
        + 1
    )

    best_val_macro_f1 = float(
        checkpoint.get(
            "best_val_macro_f1",
            -math.inf,
        )
    )

    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            0,
        )
    )

    return (
        start_epoch,
        best_val_macro_f1,
        epochs_without_improvement,
    )


# =============================================================================
# Dataset and DataLoader construction
# =============================================================================


def build_datasets(
    project_root: Path,
    config: TrainingConfig,
) -> tuple[
    OnlineTemporalVisionDataset,
    OnlineTemporalVisionDataset,
]:
    """Construct train and validation datasets."""

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
        samples_per_epoch=(
            config.train_samples_per_epoch
        ),
        window_seconds=config.window_seconds,
        frames_per_sample=config.frames_per_sample,
        augment=True,
        base_seed=config.seed,
        image_size=config.image_size,
        max_cached_videos=config.max_cached_videos,
        return_metadata=False,
        verify_videos=True,
    )

    validation_dataset = OnlineTemporalVisionDataset(
        annotation_file=annotation_file,
        flights_root=flights_root,
        split_file=split_file,
        split="val",
        samples_per_epoch=config.val_samples,
        window_seconds=config.window_seconds,
        frames_per_sample=config.frames_per_sample,
        augment=False,
        base_seed=(
            config.seed
            + 10_000
        ),
        image_size=config.image_size,
        max_cached_videos=config.max_cached_videos,
        return_metadata=False,
        verify_videos=True,
    )

    return (
        train_dataset,
        validation_dataset,
    )


def build_loaders(
    train_dataset: OnlineTemporalVisionDataset,
    validation_dataset: OnlineTemporalVisionDataset,
    config: TrainingConfig,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    """Construct deterministic train and validation DataLoaders."""

    generator = torch.Generator()

    generator.manual_seed(
        config.seed
    )

    common_arguments: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,

        "pin_memory": (
            config.pin_memory
            and torch.cuda.is_available()
        ),

        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if config.num_workers > 0:
        common_arguments["persistent_workers"] = (
            config.persistent_workers
        )

        common_arguments["prefetch_factor"] = (
            config.prefetch_factor
        )

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        drop_last=True,
        **common_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    return (
        train_loader,
        validation_loader,
    )


# =============================================================================
# Experiment setup
# =============================================================================


def create_experiment_directory(
    project_root: Path,
    prefix: str,
) -> Path:
    """Create a timestamped experiment directory."""

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_directory = (
        project_root
        / "experiments"
        / f"{prefix}_{timestamp}"
    )

    for child in [
        "checkpoints",
        "tensorboard",
        "confusion_matrices",
        "reports",
        "sampling_logs",
    ]:
        (
            experiment_directory
            / child
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return experiment_directory


def save_json(
    path: Path,
    payload: Any,
) -> None:
    """Write human-readable JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


def print_runtime_summary(
    config: TrainingConfig,
    device: torch.device,
    experiment_directory: Path,
    model: nn.Module,
) -> None:
    """Print the final resolved runtime configuration."""

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\n" + "=" * 78)
    print("MACE ONLINE VISION TRAINING — KINETICS MViTv2-S")
    print("=" * 78)

    print(
        f"Experiment directory : "
        f"{experiment_directory}"
    )

    print(
        f"Device               : "
        f"{device}"
    )

    print(
        f"CUDA available       : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU                  : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version         : "
            f"{torch.version.cuda}"
        )

    print(
        "Pretraining          : "
        "Kinetics-400"
    )

    print(
        "Architecture         : "
        "MViTv2-S"
    )

    print(
        "Temporal processing  : "
        "joint spatiotemporal transformer"
    )

    print(
        f"Input tensor         : "
        f"[B, {config.frames_per_sample}, 3, "
        f"{config.image_size}, {config.image_size}]"
    )

    print(
        f"Batch size           : "
        f"{config.batch_size}"
    )

    print(
        f"Training samples     : "
        f"{config.train_samples_per_epoch}/epoch"
    )

    print(
        f"Validation samples   : "
        f"{config.val_samples} fixed"
    )

    print(
        f"DataLoader workers   : "
        f"{config.num_workers}"
    )

    print(
        f"Backbone LR          : "
        f"{config.backbone_learning_rate}"
    )

    print(
        f"Classifier LR        : "
        f"{config.classifier_learning_rate}"
    )

    print(
        f"Weight decay         : "
        f"{config.weight_decay}"
    )

    print(
        f"Dropout              : "
        f"{config.dropout}"
    )

    print(
        f"Epochs               : "
        f"{config.epochs}"
    )

    print(
        f"Patience             : "
        f"{config.patience}"
    )

    print(
        f"AMP                  : "
        f"{config.use_amp}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print("=" * 78 + "\n")


# =============================================================================
# END OF PART 2
# Paste Part 3 directly below this line.
# =============================================================================

# =============================================================================
# Main training procedure
# =============================================================================


def train(
    project_root: Path,
    config: TrainingConfig,
    resume_path: Path | None = None,
) -> Path:
    """Run the complete Experiment 5 training procedure."""

    seed_everything(
        config.seed
    )

    device = configure_device(
        config
    )

    experiment_directory = (
        create_experiment_directory(
            project_root=project_root,
            prefix=config.experiment_prefix,
        )
    )

    save_json(
        experiment_directory
        / "config.json",
        asdict(config),
    )

    train_dataset, validation_dataset = (
        build_datasets(
            project_root=project_root,
            config=config,
        )
    )

    train_loader, validation_loader = (
        build_loaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=config,
        )
    )

    model = KineticsMViTTemporalClassifier(
        number_of_classes=(
            config.number_of_classes
        ),
        dropout=config.dropout,
        freeze_backbone=(
            config.freeze_backbone
        ),
        kinetics_mean=(
            config.kinetics_mean
        ),
        kinetics_std=(
            config.kinetics_std
        ),
    )

    model = model.to(
        device
    )

    # DataParallel is enabled only when the job genuinely exposes more than
    # one GPU.
    if (
        device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        print(
            f"Using DataParallel across "
            f"{torch.cuda.device_count()} GPUs."
        )

        model = nn.DataParallel(
            model
        )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=(
            config.label_smoothing
        ),
    )

    underlying_model = unwrap_model(
        model
    )

    backbone_parameters = [
        parameter
        for parameter
        in underlying_model.backbone.parameters()
        if parameter.requires_grad
    ]

    classifier_parameters = [
        parameter
        for parameter
        in underlying_model.classifier.parameters()
        if parameter.requires_grad
    ]

    optimizer_parameter_groups: list[
        dict[str, Any]
    ] = []

    if backbone_parameters:
        optimizer_parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate,
                "group_name": "backbone",
            }
        )

    optimizer_parameter_groups.append(
        {
            "params": classifier_parameters,
            "lr": config.classifier_learning_rate,
            "group_name": "classifier",
        }
    )

    optimizer = AdamW(
        optimizer_parameter_groups,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.cosine_eta_min,
    )

    scaler = create_grad_scaler(
        device=device,
        enabled=config.use_amp,
    )

    tensorboard_writer = SummaryWriter(
        log_dir=(
            experiment_directory
            / "tensorboard"
        )
    )

    csv_path = (
        experiment_directory
        / "training_history.csv"
    )

    initialise_csv(
        csv_path
    )

    start_epoch = 0
    best_val_macro_f1 = -math.inf
    epochs_without_improvement = 0

    if resume_path is not None:
        (
            start_epoch,
            best_val_macro_f1,
            epochs_without_improvement,
        ) = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        print(
            f"Resumed from {resume_path} "
            f"at epoch {start_epoch + 1}."
        )

    print_runtime_summary(
        config=config,
        device=device,
        experiment_directory=(
            experiment_directory
        ),
        model=model,
    )

    try:
        for epoch in range(
            start_epoch,
            config.epochs,
        ):
            epoch_start_time = (
                time.perf_counter()
            )

            # Create a fresh exactly balanced class schedule and new temporal
            # windows for this training epoch.
            train_dataset.start_new_epoch(
                epoch
            )

            # Validation must remain fixed. The dataset internally maps
            # validation to epoch zero.
            validation_dataset.start_new_epoch(
                0
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            print(
                "\n"
                + "=" * 78
            )

            print(
                f"EPOCH "
                f"{epoch + 1}/{config.epochs}"
            )

            print("=" * 78)

            print(
                "Training class counts:",
                train_dataset.epoch_class_counts(),
            )

            print(
                "Validation class counts:",
                validation_dataset.epoch_class_counts(),
            )

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
            )

            validation_metrics = (
                validate_one_epoch(
                    model=model,
                    loader=validation_loader,
                    criterion=criterion,
                    device=device,
                    config=config,
                    epoch=epoch,
                )
            )

            backbone_learning_rate = 0.0
            classifier_learning_rate = 0.0

            for parameter_group in (
                optimizer.param_groups
            ):
                group_name = parameter_group.get(
                    "group_name",
                    "",
                )

                if group_name == "backbone":
                    backbone_learning_rate = float(
                        parameter_group["lr"]
                    )

                elif group_name == "classifier":
                    classifier_learning_rate = float(
                        parameter_group["lr"]
                    )

            scheduler.step()

            epoch_seconds = (
                time.perf_counter()
                - epoch_start_time
            )

            if torch.cuda.is_available():
                allocated_gb = (
                    torch.cuda.max_memory_allocated()
                    / (1024**3)
                )

                reserved_gb = (
                    torch.cuda.max_memory_reserved()
                    / (1024**3)
                )

            else:
                allocated_gb = 0.0
                reserved_gb = 0.0

            epoch_number = (
                epoch + 1
            )

            log_metrics_to_tensorboard(
                tensorboard_writer=(
                    tensorboard_writer
                ),
                split="Train",
                metrics=train_metrics,
                epoch_number=epoch_number,
            )

            log_metrics_to_tensorboard(
                tensorboard_writer=(
                    tensorboard_writer
                ),
                split="Validation",
                metrics=validation_metrics,
                epoch_number=epoch_number,
            )

            tensorboard_writer.add_scalar(
                "LearningRate/Backbone",
                backbone_learning_rate,
                epoch_number,
            )

            tensorboard_writer.add_scalar(
                "LearningRate/Classifier",
                classifier_learning_rate,
                epoch_number,
            )

            tensorboard_writer.add_scalar(
                "Runtime/EpochSeconds",
                epoch_seconds,
                epoch_number,
            )

            tensorboard_writer.add_scalar(
                "GPU/PeakAllocatedGB",
                allocated_gb,
                epoch_number,
            )

            tensorboard_writer.add_scalar(
                "GPU/PeakReservedGB",
                reserved_gb,
                epoch_number,
            )

            if (
                epoch_number
                % config.log_confusion_every
                == 0
            ):
                matrix = np.asarray(
                    validation_metrics.confusion_matrix
                )

                raw_figure = (
                    create_confusion_matrix_figure(
                        matrix=matrix,
                        title=(
                            "Validation confusion matrix — "
                            f"epoch {epoch_number}"
                        ),
                    )
                )

                normalised_figure = (
                    create_normalised_confusion_matrix_figure(
                        matrix=matrix,
                        title=(
                            "Validation normalised confusion "
                            f"matrix — epoch {epoch_number}"
                        ),
                    )
                )

                tensorboard_writer.add_figure(
                    "ConfusionMatrix/Validation",
                    raw_figure,
                    epoch_number,
                )

                tensorboard_writer.add_figure(
                    "ConfusionMatrixNormalised/Validation",
                    normalised_figure,
                    epoch_number,
                )

                raw_figure.savefig(
                    experiment_directory
                    / "confusion_matrices"
                    / (
                        f"epoch_{epoch_number:03d}"
                        "_counts.png"
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )

                normalised_figure.savefig(
                    experiment_directory
                    / "confusion_matrices"
                    / (
                        f"epoch_{epoch_number:03d}"
                        "_normalised.png"
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )

                plt.close(
                    raw_figure
                )

                plt.close(
                    normalised_figure
                )

            tensorboard_writer.flush()

            append_csv_row(
                csv_path,
                {
                    "epoch": epoch_number,

                    "backbone_learning_rate": (
                        backbone_learning_rate
                    ),

                    "classifier_learning_rate": (
                        classifier_learning_rate
                    ),

                    "train_loss": (
                        train_metrics.loss
                    ),

                    "train_accuracy": (
                        train_metrics.accuracy
                    ),

                    "train_macro_f1": (
                        train_metrics.macro_f1
                    ),

                    "train_balanced_accuracy": (
                        train_metrics.balanced_accuracy
                    ),

                    "val_loss": (
                        validation_metrics.loss
                    ),

                    "val_accuracy": (
                        validation_metrics.accuracy
                    ),

                    "val_macro_f1": (
                        validation_metrics.macro_f1
                    ),

                    "val_balanced_accuracy": (
                        validation_metrics.balanced_accuracy
                    ),

                    "epoch_seconds": (
                        epoch_seconds
                    ),

                    "cuda_memory_allocated_gb": (
                        allocated_gb
                    ),

                    "cuda_memory_reserved_gb": (
                        reserved_gb
                    ),
                },
            )

            print("\nEpoch summary")
            print("-" * 78)

            print(
                f"Train loss      : "
                f"{train_metrics.loss:.4f}"
            )

            print(
                f"Train accuracy  : "
                f"{train_metrics.accuracy:.4f}"
            )

            print(
                f"Train Macro F1  : "
                f"{train_metrics.macro_f1:.4f}"
            )

            print(
                f"Val loss        : "
                f"{validation_metrics.loss:.4f}"
            )

            print(
                f"Val accuracy    : "
                f"{validation_metrics.accuracy:.4f}"
            )

            print(
                f"Val Macro F1    : "
                f"{validation_metrics.macro_f1:.4f}"
            )

            print(
                f"Val balanced acc: "
                f"{validation_metrics.balanced_accuracy:.4f}"
            )

            print(
                f"Backbone LR     : "
                f"{backbone_learning_rate:.8f}"
            )

            print(
                f"Classifier LR   : "
                f"{classifier_learning_rate:.8f}"
            )

            print(
                f"Epoch duration  : "
                f"{epoch_seconds / 60.0:.2f} min"
            )

            if torch.cuda.is_available():
                print(
                    f"Peak GPU memory : "
                    f"{allocated_gb:.2f} GB allocated, "
                    f"{reserved_gb:.2f} GB reserved"
                )

            for class_id in range(
                len(ID_TO_LABEL)
            ):
                class_name = (
                    ID_TO_LABEL[class_id]
                )

                print(
                    f"Val {class_name:<11} "
                    f"P="
                    f"{validation_metrics.precision_per_class[class_id]:.4f} "
                    f"R="
                    f"{validation_metrics.recall_per_class[class_id]:.4f} "
                    f"F1="
                    f"{validation_metrics.f1_per_class[class_id]:.4f} "
                    f"N="
                    f"{validation_metrics.support_per_class[class_id]}"
                )

            improved = (
                validation_metrics.macro_f1
                > best_val_macro_f1
                + config.minimum_delta
            )

            if improved:
                best_val_macro_f1 = (
                    validation_metrics.macro_f1
                )

                epochs_without_improvement = 0

                best_payload = (
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        epoch=epoch,
                        best_val_macro_f1=(
                            best_val_macro_f1
                        ),
                        epochs_without_improvement=(
                            epochs_without_improvement
                        ),
                    )
                )

                save_checkpoint(
                    experiment_directory
                    / "checkpoints"
                    / "best.pt",
                    best_payload,
                )

                report = {
                    "epoch": epoch_number,

                    "accuracy": (
                        validation_metrics.accuracy
                    ),

                    "macro_f1": (
                        validation_metrics.macro_f1
                    ),

                    "balanced_accuracy": (
                        validation_metrics.balanced_accuracy
                    ),

                    "loss": (
                        validation_metrics.loss
                    ),

                    "classes": {},
                }

                for class_id in range(
                    len(ID_TO_LABEL)
                ):
                    class_name = (
                        ID_TO_LABEL[class_id]
                    )

                    report["classes"][
                        class_name
                    ] = {
                        "precision": (
                            validation_metrics
                            .precision_per_class[
                                class_id
                            ]
                        ),

                        "recall": (
                            validation_metrics
                            .recall_per_class[
                                class_id
                            ]
                        ),

                        "f1_score": (
                            validation_metrics
                            .f1_per_class[
                                class_id
                            ]
                        ),

                        "support": (
                            validation_metrics
                            .support_per_class[
                                class_id
                            ]
                        ),
                    }

                report["confusion_matrix"] = (
                    validation_metrics.confusion_matrix
                )

                save_json(
                    experiment_directory
                    / "reports"
                    / "best_validation_metrics.json",
                    report,
                )

                print(
                    "✓ New best checkpoint: "
                    "validation Macro F1 "
                    f"{best_val_macro_f1:.4f}"
                )

            else:
                epochs_without_improvement += 1

                print(
                    "No validation Macro F1 "
                    "improvement for "
                    f"{epochs_without_improvement} "
                    "epoch(s)."
                )

            last_payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                best_val_macro_f1=(
                    best_val_macro_f1
                ),
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
            )

            save_checkpoint(
                experiment_directory
                / "checkpoints"
                / "last.pt",
                last_payload,
            )

            if (
                config.save_every > 0
                and epoch_number
                % config.save_every
                == 0
            ):
                save_checkpoint(
                    experiment_directory
                    / "checkpoints"
                    / (
                        f"epoch_"
                        f"{epoch_number:03d}.pt"
                    ),
                    last_payload,
                )

            if (
                epochs_without_improvement
                >= config.patience
            ):
                print(
                    f"\nEarly stopping after "
                    f"{epoch_number} epochs. "
                    "Best validation Macro F1: "
                    f"{best_val_macro_f1:.4f}"
                )

                break

    finally:
        tensorboard_writer.close()
        train_dataset.close()
        validation_dataset.close()

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 78)

    print(
        "Best validation Macro F1: "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Outputs: "
        f"{experiment_directory}"
    )

    print("=" * 78)

    return experiment_directory


# =============================================================================
# CLI
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse optional command-line overrides."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the MACE online temporal vision model "
            "using a Kinetics-400-pretrained MViTv2-S."
        )
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to a last.pt or another "
            "resumable checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override the configured batch size."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Override the configured "
            "DataLoader worker count."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override the configured "
            "maximum epoch count."
        ),
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help=(
            "Override generated training "
            "samples per epoch."
        ),
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help=(
            "Override the fixed validation "
            "sample count."
        ),
    )

    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help=(
            "Override the pretrained "
            "MViTv2-S backbone learning rate."
        ),
    )

    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=None,
        help=(
            "Override the new four-class "
            "classifier learning rate."
        ),
    )

    return parser.parse_args()


def validate_config(
    config: TrainingConfig,
) -> None:
    """Reject invalid configuration values before training."""

    if config.batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if config.num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if config.epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if (
        config.train_samples_per_epoch
        % len(LABEL_MAP)
        != 0
    ):
        raise ValueError(
            "train_samples_per_epoch must "
            "be divisible by four."
        )

    if (
        config.val_samples
        % len(LABEL_MAP)
        != 0
    ):
        raise ValueError(
            "val_samples must be divisible by four."
        )

    if (
        config.gradient_accumulation_steps
        <= 0
    ):
        raise ValueError(
            "gradient_accumulation_steps "
            "must be greater than zero."
        )

    if config.number_of_classes != len(
        LABEL_MAP
    ):
        raise ValueError(
            "number_of_classes must match LABEL_MAP."
        )

    if config.frames_per_sample != 16:
        raise ValueError(
            "Experiment 5 requires exactly "
            "16 frames per sample."
        )

    if config.image_size != 224:
        raise ValueError(
            "Experiment 5 requires "
            "224x224 input frames."
        )

    if config.backbone_learning_rate <= 0.0:
        raise ValueError(
            "backbone_learning_rate must "
            "be greater than zero."
        )

    if config.classifier_learning_rate <= 0.0:
        raise ValueError(
            "classifier_learning_rate must "
            "be greater than zero."
        )

    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(
            "dropout must be in [0, 1)."
        )

    if config.weight_decay < 0.0:
        raise ValueError(
            "weight_decay cannot be negative."
        )


def main() -> None:
    """Resolve paths, apply overrides and launch training."""

    arguments = parse_arguments()

    config = TrainingConfig()

    if arguments.batch_size is not None:
        config.batch_size = (
            arguments.batch_size
        )

    if arguments.num_workers is not None:
        config.num_workers = (
            arguments.num_workers
        )

    if arguments.epochs is not None:
        config.epochs = (
            arguments.epochs
        )

    if arguments.train_samples is not None:
        config.train_samples_per_epoch = (
            arguments.train_samples
        )

    if arguments.val_samples is not None:
        config.val_samples = (
            arguments.val_samples
        )

    if (
        arguments.backbone_learning_rate
        is not None
    ):
        config.backbone_learning_rate = (
            arguments.backbone_learning_rate
        )

    if (
        arguments.classifier_learning_rate
        is not None
    ):
        config.classifier_learning_rate = (
            arguments.classifier_learning_rate
        )

    validate_config(
        config
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    resume_path = (
        arguments.resume.resolve()
        if arguments.resume is not None
        else None
    )

    train(
        project_root=project_root,
        config=config,
        resume_path=resume_path,
    )


if __name__ == "__main__":
    main()