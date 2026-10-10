#!/usr/bin/env python3
"""
MACE synchronized multimodal dataset.

This deliberately reuses dataset_online_final.py for the vision branch so the
multimodal experiment inherits the SAME annotation parsing, flight split,
balanced online sampling, video decoding and augmentation used by MViTv2 Ex5.

For each vision sample, its metadata defines one 5-s video interval. Aircraft
telemetry is then taken from labelled_flight_telemetry.csv using video_time_sec,
which was already generated with:
    autopilot_time_sec = video_time_sec + offset_sec

Thus the modalities cannot independently choose different windows.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MACE_MODELS_DIR = PROJECT_ROOT / "scripts" / "MACE_models"

if not MACE_MODELS_DIR.exists():
    raise RuntimeError(
        f"Could not find MACE_models directory: {MACE_MODELS_DIR}"
    )

if str(MACE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MACE_MODELS_DIR))

from dataset_online_final import (
    OnlineTemporalVisionDataset,
    LABEL_MAP,
    ID_TO_LABEL,
)

RAW_FEATURES = ["Roll_deg","Pitch_deg","AccX","AccY","AccZ","GyrX","GyrY","GyrZ"]
MODEL_FEATURES = [
    "roll_rel_deg","pitch_rel_deg","acc_x","acc_y","acc_z",
    "gyro_x","gyro_y","gyro_z",
]

def _stats_from_json(path):
    obj = json.loads(Path(path).read_text())
    # Supports either {"feature":{"mean":...,"std":...}} or baseline arrays.
    if all(k in obj for k in MODEL_FEATURES):
        mean = np.array([obj[k]["mean"] for k in MODEL_FEATURES], np.float32)
        std = np.array([obj[k]["std"] for k in MODEL_FEATURES], np.float32)
        return mean, std
    if "normalization_mean" in obj and "normalization_std" in obj:
        return (np.asarray(obj["normalization_mean"], np.float32),
                np.asarray(obj["normalization_std"], np.float32))
    if "mean" in obj and "std" in obj:
        return np.asarray(obj["mean"], np.float32), np.asarray(obj["std"], np.float32)
    raise ValueError(f"Unrecognised normalization JSON format: {path}")

def _stats_from_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "normalization_mean" not in ckpt or "normalization_std" not in ckpt:
        raise KeyError("Aircraft checkpoint does not contain normalization_mean/std.")
    return (np.asarray(ckpt["normalization_mean"], np.float32),
            np.asarray(ckpt["normalization_std"], np.float32))

class MACEMultimodalFusionDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        flights_root,
        split_file,
        telemetry_csv,
        split="train",
        samples_per_epoch=6000,
        aircraft_checkpoint=None,
        normalization_json=None,
        window_seconds=5.0,
        frames_per_sample=16,
        telemetry_hz=10.0,
        image_size=224,
        augment=False,
        base_seed=42,
        max_cached_videos=2,
    ):
        self.split = split
        self.window_seconds = float(window_seconds)
        self.telemetry_hz = float(telemetry_hz)
        self.telemetry_steps = int(round(self.window_seconds * self.telemetry_hz))

        self.vision = OnlineTemporalVisionDataset(
            annotation_file=annotation_file,
            flights_root=flights_root,
            split_file=split_file,
            split=split,
            samples_per_epoch=samples_per_epoch,
            window_seconds=window_seconds,
            frames_per_sample=frames_per_sample,
            augment=augment,
            base_seed=base_seed,
            image_size=image_size,
            max_cached_videos=max_cached_videos,
            return_metadata=True,
            verify_videos=True,
        )

        tel = pd.read_csv(telemetry_csv)
        required = {
            "flight_id","segment_uid","state","video_time_sec","autopilot_time_sec",
            "offset_sec","sync_method", *RAW_FEATURES
        }
        missing = required - set(tel.columns)
        if missing:
            raise RuntimeError(f"Telemetry CSV missing columns: {sorted(missing)}")

        tel["flight_id"] = tel["flight_id"].astype(str).str.strip().str.upper()
        tel["segment_uid"] = tel["segment_uid"].astype(str).str.strip()
        tel["state"] = tel["state"].astype(str).str.strip().str.lower()
        for c in ["video_time_sec","autopilot_time_sec","offset_sec",*RAW_FEATURES]:
            tel[c] = pd.to_numeric(tel[c], errors="coerce")
        tel = tel.dropna(subset=["video_time_sec", *RAW_FEATURES]).copy()
        tel = tel.sort_values(["flight_id","segment_uid","video_time_sec"])

        self.telemetry = {}
        for (flight, segment), g in tel.groupby(["flight_id","segment_uid"], sort=False):
            self.telemetry[(flight, segment)] = g.reset_index(drop=True)

        if normalization_json:
            mean, std = _stats_from_json(normalization_json)
        elif aircraft_checkpoint:
            mean, std = _stats_from_checkpoint(aircraft_checkpoint)
        else:
            raise ValueError(
                "Provide --aircraft-checkpoint or --normalization-json so telemetry "
                "uses exactly the binary aircraft model's training statistics."
            )
        if mean.shape != (8,) or std.shape != (8,):
            raise ValueError(f"Expected 8 normalization values, got {mean.shape}/{std.shape}")
        if np.any(std <= 0):
            raise ValueError("All normalization standard deviations must be > 0.")
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.vision)

    def start_new_epoch(self, epoch):
        """Regenerate the balanced vision sampling schedule for this epoch."""
        if hasattr(self.vision, "start_new_epoch"):
            self.vision.start_new_epoch(epoch)

    def _telemetry_for_metadata(self, meta):
        flight = str(meta["flight_id"]).strip().upper()
        segment = str(meta["segment_id"]).strip()
        key = (flight, f"{flight}_{segment}" if not segment.startswith(flight+"_") else segment)

        if key not in self.telemetry:
            # Defensive fallback for workbooks whose segment_id already differs in formatting.
            candidates = [
                k for k in self.telemetry
                if k[0] == flight and (
                    k[1].endswith("_" + segment) or k[1] == segment
                )
            ]
            if len(candidates) != 1:
                raise KeyError(
                    f"Could not uniquely match telemetry for flight={flight}, "
                    f"segment={segment}. Candidates={candidates}"
                )
            key = candidates[0]

        g = self.telemetry[key]
        start = float(meta["clip_start_sec"])
        end = float(meta["clip_end_sec"])
        source_t = g["video_time_sec"].to_numpy(np.float64)

        # Exactly t...t+4.9 at 10 Hz. The video interval remains [t,t+5].
        target_t = start + np.arange(self.telemetry_steps, dtype=np.float64) / self.telemetry_hz
        if target_t[0] < source_t[0] - 1e-5 or target_t[-1] > source_t[-1] + 1e-5:
            raise RuntimeError(
                f"Telemetry coverage failure for {key}: requested "
                f"{target_t[0]:.3f}-{target_t[-1]:.3f}s, available "
                f"{source_t[0]:.3f}-{source_t[-1]:.3f}s."
            )

        raw = np.column_stack([
            np.interp(target_t, source_t, g[c].to_numpy(np.float64))
            for c in RAW_FEATURES
        ]).astype(np.float32)

        # Match Ex3 binary preprocessing: window-relative roll and pitch.
        raw[:, 0] -= raw[0, 0]
        raw[:, 1] -= raw[0, 1]
        x = (raw - self.mean[None, :]) / self.std[None, :]

        # Provenance is constant within a labelled segment.
        return (
            torch.from_numpy(x.astype(np.float32)),
            float(np.interp(start, source_t, g["autopilot_time_sec"].to_numpy(np.float64))),
            float(g["offset_sec"].iloc[0]),
            str(g["sync_method"].iloc[0]),
        )

    def __getitem__(self, index):
        # dataset_online_final returns (frames, label, metadata) with metadata enabled.
        item = self.vision[index]
        if isinstance(item, dict):
            frames = item.get("frames", item.get("video"))
            label = item["label"]
            meta = item.get("metadata", item)
        else:
            frames, label, meta = item

        telemetry, ap_start, offset, sync_method = self._telemetry_for_metadata(meta)

        return {
            "video": frames,                 # [T,C,H,W], ImageNet-normalised
            "telemetry": telemetry,          # [50,8], binary-model normalisation
            "label": label.long() if torch.is_tensor(label) else torch.tensor(label, dtype=torch.long),
            "sample_id": str(meta["sample_id"]),
            "flight_id": str(meta["flight_id"]),
            "segment_id": str(meta["segment_id"]),
            "state": str(meta["label"]),
            "video_start_sec": torch.tensor(float(meta["clip_start_sec"]), dtype=torch.float32),
            "video_end_sec": torch.tensor(float(meta["clip_end_sec"]), dtype=torch.float32),
            "autopilot_start_sec": torch.tensor(ap_start, dtype=torch.float32),
            "offset_sec": torch.tensor(offset, dtype=torch.float32),
            "sync_method": sync_method,
        }
