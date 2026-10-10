#!/usr/bin/env python3
"""
MACE: synchronise a GoPro video with ArduPilot IMU using cross-correlation.

Inputs expected
---------------
GoPro ACCL text:
  ./gpmfdemo VIDEO.MP4 -fACCL > FL008_gopro_accel.txt
GoPro GYRO text:
  ./gpmfdemo VIDEO.MP4 -fGYRO > FL008_gopro_gyro.txt
ArduPilot IMU CSV:
  mavlogdump.py 00000042.BIN --types IMU --format csv > FL008_autopilot_imu.csv

Reported mapping
----------------
  autopilot_time_sec ~= video_time_sec + offset_sec

The script saves correlation plots plus raw/normalised full and zoomed overlays.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

SAMPLERATE_RE = re.compile(
    r"sampling\s+rate\s*=\s*([0-9.]+)\s*Hz.*?"
    r"time\s+([+-]?[0-9.]+)\s+to\s+([+-]?[0-9.]+)",
    re.IGNORECASE,
)


def parse_xyz_line(line: str, sensor: str):
    """
    Parse one scaled gpmfdemo line such as:

        ACCL -8.420m/s², -15.578m/s², -7.074m/s²,
        GYRO -0.193rad/s, -0.184rad/s, 0.673rad/s,

    The unit text is intentionally ignored because the superscript ² can
    appear as a replacement character depending on terminal/file encoding.
    """
    s = line.strip()
    # Metadata lines such as "ACCL sampling rate = ..."
    # are not sensor measurements.
    if "SAMPLING RATE" in s.upper():
        return None
    if not s.upper().startswith(sensor.upper()):
        return None

    vals = NUMBER_RE.findall(s[len(sensor):])
    if len(vals) < 3:
        return None

    return float(vals[0]), float(vals[1]), float(vals[2])

def parse_gopro_text(path: Path, sensor: str) -> pd.DataFrame:
    """
    Parse complete GoPro GPMF ACCL/GYRO output produced with:

        gpmfdemo <video> -a -fACCL
        gpmfdemo <video> -a -fGYRO

    gpmfdemo organises the decoded IMU stream into repeated payloads:

        PAYLOAD TIME:
            0.000 to 1.001 seconds
        SCALED DATA:
            ACCL ...
            ACCL ...
            ...

        PAYLOAD TIME:
            1.001 to 2.002 seconds
        SCALED DATA:
            ACCL ...
            ...

    Each payload therefore provides an explicit video-relative time interval.

    For every payload containing N sensor measurements, timestamps are assigned
    uniformly within that payload interval using sample-bin midpoints:

        t_i = t_start + (i + 0.5) / N * (t_end - t_start)

    This is preferable to reconstructing one global time axis from the final
    COMPUTED SAMPLERATES metadata because it preserves the timing structure
    explicitly reported by GPMF.

    The final COMPUTED SAMPLERATES line, e.g.

        ACCL sampling rate = 201.417080Hz
             (time 0.002042 to 531.530946)

    is retained as global validation/reference metadata.

    Returns
    -------
    pd.DataFrame
        Columns:
            time_sec
            x
            y
            z
            sample_index
            payload_index
            payload_sample_index
            payload_start_sec
            payload_end_sec
            payload_sample_count
            payload_sampling_rate_hz
            reported_sampling_rate_hz

    Notes
    -----
    time_sec is relative to the beginning of the GoPro video and is therefore
    the timebase used later for GoPro <-> autopilot cross-correlation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    sensor = sensor.upper()

    if sensor not in {"ACCL", "GYRO"}:
        raise ValueError(
            f"Unsupported GoPro sensor '{sensor}'. "
            f"Expected 'ACCL' or 'GYRO'."
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    payloads = []

    current_payload_start = None
    current_payload_end = None
    current_samples = []

    reported_rate = None
    reported_start = None
    reported_end = None

    # We need this because gpmfdemo prints:
    #
    # PAYLOAD TIME:
    #     0.000 to 1.001 seconds
    #
    # i.e. the timing values occur on the line AFTER "PAYLOAD TIME:".
    expecting_payload_time = False

    # Accept:
    #
    # 0.000 to 1.001 seconds
    # -0.014040 to 1.001 seconds
    #
    payload_time_re = re.compile(
        r"^\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s+to\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s+seconds?"
        r"\s*$",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Helper: close the current payload
    # ------------------------------------------------------------------

    def finish_payload():
        nonlocal current_payload_start
        nonlocal current_payload_end
        nonlocal current_samples

        if current_payload_start is None:
            return

        if current_payload_end is None:
            raise RuntimeError(
                f"Malformed PAYLOAD TIME block in {path}: "
                f"payload start exists without payload end."
            )

        if current_payload_end <= current_payload_start:
            raise RuntimeError(
                f"Invalid payload interval in {path}: "
                f"{current_payload_start:.6f} -> "
                f"{current_payload_end:.6f} s."
            )

        # A payload may theoretically contain no samples of the requested
        # sensor. Do not fabricate measurements in that case.
        if len(current_samples) > 0:
            payloads.append(
                {
                    "start": float(current_payload_start),
                    "end": float(current_payload_end),
                    "samples": current_samples.copy(),
                }
            )

        current_payload_start = None
        current_payload_end = None
        current_samples = []

    # ------------------------------------------------------------------
    # Parse file
    # ------------------------------------------------------------------

    with path.open("r", errors="replace") as f:

        for line_number, raw in enumerate(f, start=1):

            line = raw.rstrip("\n")
            stripped = line.strip()

            # ----------------------------------------------------------
            # 1. Global COMPUTED SAMPLERATES metadata
            # ----------------------------------------------------------
            #
            # Check this BEFORE parse_xyz_line(), because the line begins
            # with ACCL/GYRO and also contains numeric values.
            # ----------------------------------------------------------

            if stripped.upper().startswith(f"{sensor} SAMPLING RATE"):

                m = SAMPLERATE_RE.search(line)

                if m:
                    reported_rate = float(m.group(1))
                    reported_start = float(m.group(2))
                    reported_end = float(m.group(3))

                continue

            # ----------------------------------------------------------
            # 2. PAYLOAD TIME marker
            # ----------------------------------------------------------

            if stripped.upper() == "PAYLOAD TIME:":

                # We have reached a new payload.
                # Finish the previous one first.
                finish_payload()

                expecting_payload_time = True
                continue

            # ----------------------------------------------------------
            # 3. PAYLOAD TIME values
            # ----------------------------------------------------------

            if expecting_payload_time:

                m = payload_time_re.match(stripped)

                if m is None:
                    raise RuntimeError(
                        f"Could not parse PAYLOAD TIME at "
                        f"{path}:{line_number}\n"
                        f"Expected something like:\n"
                        f"    0.000 to 1.001 seconds\n"
                        f"but received:\n"
                        f"    {line!r}"
                    )

                current_payload_start = float(m.group(1))
                current_payload_end = float(m.group(2))

                if current_payload_end <= current_payload_start:
                    raise RuntimeError(
                        f"Invalid PAYLOAD TIME at "
                        f"{path}:{line_number}: "
                        f"{current_payload_start} -> "
                        f"{current_payload_end} s."
                    )

                expecting_payload_time = False
                continue

            # ----------------------------------------------------------
            # 4. Actual ACCL/GYRO measurement
            # ----------------------------------------------------------

            xyz = parse_xyz_line(line, sensor)

            if xyz is not None:

                if current_payload_start is None:
                    raise RuntimeError(
                        f"Found {sensor} measurement before a valid "
                        f"PAYLOAD TIME block at {path}:{line_number}.\n"
                        f"This parser expects output produced using "
                        f"'gpmfdemo -a -f{sensor}'."
                    )

                current_samples.append(xyz)

    # Close final payload.
    finish_payload()

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------

    if len(payloads) == 0:
        raise RuntimeError(
            f"No complete {sensor} payloads were found in {path}.\n"
            f"Confirm that the file was produced using:\n"
            f"    gpmfdemo <video> -a -f{sensor}"
        )

    total_samples = sum(len(p["samples"]) for p in payloads)

    if total_samples < 100:
        raise RuntimeError(
            f"Only {total_samples} {sensor} samples found in {path}.\n"
            f"Confirm that the file was produced using:\n"
            f"    gpmfdemo <video> -a -f{sensor}"
        )

    if reported_rate is None:
        raise RuntimeError(
            f"Could not find COMPUTED SAMPLERATES metadata "
            f"for {sensor} in {path}."
        )

    # reported_start/end are useful metadata but are no longer used to
    # construct the actual sample timestamps.
    if reported_start is None or reported_end is None:
        raise RuntimeError(
            f"Could not parse global reported timing range "
            f"for {sensor} in {path}."
        )

    # ------------------------------------------------------------------
    # Convert payloads -> timestamped measurements
    # ------------------------------------------------------------------

    rows = []

    global_sample_index = 0

    for payload_index, payload in enumerate(payloads):

        start = payload["start"]
        end = payload["end"]
        samples = payload["samples"]

        n = len(samples)
        duration = end - start

        if n <= 0:
            continue

        # --------------------------------------------------------------
        # Assign timestamps at the centre of each sample interval.
        #
        # Example:
        #
        # payload = 0.000 -> 1.001
        # N = 201
        #
        # dt = 1.001 / 201
        #
        # first timestamp:
        #     0.000 + 0.5*dt
        #
        # last timestamp:
        #     0.000 + 200.5*dt
        #
        # Therefore no sample is artificially placed exactly on either
        # payload boundary.
        # --------------------------------------------------------------

        dt = duration / n

        times = start + (np.arange(n, dtype=float) + 0.5) * dt

        payload_rate = n / duration

        for payload_sample_index, (t, xyz) in enumerate(
            zip(times, samples)
        ):
            x, y, z = xyz

            rows.append(
                {
                    "time_sec": float(t),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),

                    "sample_index": global_sample_index,

                    "payload_index": payload_index,
                    "payload_sample_index": payload_sample_index,

                    "payload_start_sec": start,
                    "payload_end_sec": end,
                    "payload_sample_count": n,
                    "payload_sampling_rate_hz": payload_rate,

                    "reported_sampling_rate_hz": reported_rate,
                }
            )

            global_sample_index += 1

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No timestamped {sensor} measurements were reconstructed."
        )

    # ------------------------------------------------------------------
    # Chronological validation
    # ------------------------------------------------------------------

    times = df["time_sec"].to_numpy(dtype=float)

    dt_all = np.diff(times)

    if np.any(dt_all <= 0):
        bad = np.where(dt_all <= 0)[0]

        raise RuntimeError(
            f"{sensor} timestamps are not strictly increasing.\n"
            f"Found {len(bad)} non-positive timestamp differences.\n"
            f"First problematic sample index: {bad[0]}"
        )

    # ------------------------------------------------------------------
    # Payload-level statistics
    # ------------------------------------------------------------------

    payload_counts = np.array(
        [len(p["samples"]) for p in payloads],
        dtype=int,
    )

    payload_durations = np.array(
        [p["end"] - p["start"] for p in payloads],
        dtype=float,
    )

    payload_rates = payload_counts / payload_durations

    first_payload_start = payloads[0]["start"]
    last_payload_end = payloads[-1]["end"]

    payload_span = last_payload_end - first_payload_start

    # Effective rate across the complete GPMF payload span.
    overall_effective_rate = (
        total_samples / payload_span
        if payload_span > 0
        else float("nan")
    )

    # Median within-payload sampling interval is more meaningful than
    # np.diff() across the entire DataFrame because payload boundaries
    # may introduce small timing discontinuities.
    median_payload_rate = float(np.median(payload_rates))

    # ------------------------------------------------------------------
    # Payload continuity analysis
    # ------------------------------------------------------------------

    payload_gaps = []

    for i in range(1, len(payloads)):

        previous_end = payloads[i - 1]["end"]
        current_start = payloads[i]["start"]

        payload_gaps.append(current_start - previous_end)

    payload_gaps = np.asarray(payload_gaps, dtype=float)

    if len(payload_gaps) > 0:
        max_payload_gap = float(np.max(payload_gaps))
        min_payload_gap = float(np.min(payload_gaps))
        median_payload_gap = float(np.median(payload_gaps))
    else:
        max_payload_gap = float("nan")
        min_payload_gap = float("nan")
        median_payload_gap = float("nan")

    # ------------------------------------------------------------------
    # Global timing metadata comparison
    # ------------------------------------------------------------------

    reported_duration = reported_end - reported_start

    # Do NOT require the first/last sample timestamp to equal the global
    # reported range exactly. The payload intervals and COMPUTED
    # SAMPLERATES range describe related but not necessarily identical
    # timing conventions.
    #
    # Instead compare the overall duration as a sanity check.
    duration_difference = payload_span - reported_duration

    # Allow modest discrepancy due to payload boundary conventions.
    duration_tolerance = max(
        0.10,
        2.0 / reported_rate,
    )

    if abs(duration_difference) > duration_tolerance:
        raise RuntimeError(
            f"{sensor} global timing validation failed.\n"
            f"  payloads                 : {len(payloads)}\n"
            f"  samples                  : {total_samples}\n"
            f"  first payload start      : "
            f"{first_payload_start:.6f} s\n"
            f"  last payload end         : "
            f"{last_payload_end:.6f} s\n"
            f"  payload span             : "
            f"{payload_span:.6f} s\n"
            f"  reported global start    : "
            f"{reported_start:.6f} s\n"
            f"  reported global end      : "
            f"{reported_end:.6f} s\n"
            f"  reported global duration : "
            f"{reported_duration:.6f} s\n"
            f"  duration difference      : "
            f"{duration_difference:.6f} s\n"
            f"  tolerance                : "
            f"{duration_tolerance:.6f} s\n"
        )

    # ------------------------------------------------------------------
    # DataFrame metadata
    # ------------------------------------------------------------------

    df.attrs["sensor"] = sensor

    df.attrs["sample_count"] = total_samples
    df.attrs["payload_count"] = len(payloads)

    df.attrs["reported_sampling_rate_hz"] = reported_rate
    df.attrs["reported_start_sec"] = reported_start
    df.attrs["reported_end_sec"] = reported_end
    df.attrs["reported_duration_sec"] = reported_duration

    df.attrs["first_payload_start_sec"] = first_payload_start
    df.attrs["last_payload_end_sec"] = last_payload_end
    df.attrs["payload_span_sec"] = payload_span

    df.attrs["overall_effective_sampling_rate_hz"] = (
        overall_effective_rate
    )

    df.attrs["median_payload_sampling_rate_hz"] = (
        median_payload_rate
    )

    df.attrs["min_payload_sample_count"] = int(
        np.min(payload_counts)
    )

    df.attrs["max_payload_sample_count"] = int(
        np.max(payload_counts)
    )

    df.attrs["median_payload_sample_count"] = float(
        np.median(payload_counts)
    )

    df.attrs["min_payload_gap_sec"] = min_payload_gap
    df.attrs["max_payload_gap_sec"] = max_payload_gap
    df.attrs["median_payload_gap_sec"] = median_payload_gap

    df.attrs["duration_difference_sec"] = duration_difference

    # ------------------------------------------------------------------
    # Useful console report
    # ------------------------------------------------------------------

    print()
    print(f"GoPro {sensor} parsing summary")
    print("-" * 60)
    print(f"File                     : {path}")
    print(f"Payloads                 : {len(payloads):,}")
    print(f"Samples                  : {total_samples:,}")
    print(
        f"Payload sample count     : "
        f"{payload_counts.min()} - {payload_counts.max()} "
        f"(median {np.median(payload_counts):.1f})"
    )
    print(
        f"Reported sampling rate   : "
        f"{reported_rate:.6f} Hz"
    )
    print(
        f"Median payload rate      : "
        f"{median_payload_rate:.6f} Hz"
    )
    print(
        f"Overall effective rate   : "
        f"{overall_effective_rate:.6f} Hz"
    )
    print(
        f"Payload time span        : "
        f"{first_payload_start:.6f} -> "
        f"{last_payload_end:.6f} s"
    )
    print(
        f"Reported global span     : "
        f"{reported_start:.6f} -> "
        f"{reported_end:.6f} s"
    )
    print(
        f"Duration difference      : "
        f"{duration_difference:+.6f} s"
    )

    if len(payload_gaps) > 0:
        print(
            f"Payload boundary gaps    : "
            f"min={min_payload_gap:+.6f}, "
            f"median={median_payload_gap:+.6f}, "
            f"max={max_payload_gap:+.6f} s"
        )

    print("-" * 60)

    return df

def read_autopilot_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    need = ["TimeUS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing ArduPilot columns {missing}. Found: {list(df.columns)}")
    out = df[need].copy()
    for c in need:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().sort_values("TimeUS").drop_duplicates("TimeUS").reset_index(drop=True)
    out["time_sec"] = out["TimeUS"] / 1_000_000.0
    if len(out) < 100:
        raise RuntimeError("Too few valid ArduPilot IMU rows.")
    return out


def estimate_rate(t):
    t = np.asarray(t, dtype=float)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float("nan") if len(d) == 0 else 1.0 / np.median(d)


def magnitude(x, y, z):
    return np.sqrt(np.asarray(x)**2 + np.asarray(y)**2 + np.asarray(z)**2)


def resample(t, x, fs):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    order = np.argsort(t)
    t, x = t[order], x[order]
    t, idx = np.unique(t, return_index=True)
    x = x[idx]
    start = math.ceil(t[0] * fs) / fs
    end = math.floor(t[-1] * fs) / fs
    if end <= start:
        raise RuntimeError("Signal too short to resample.")
    grid = np.arange(start, end + 0.5/fs, 1.0/fs)
    return grid, np.interp(grid, t, x)


def detrend_moving(x, fs, window_sec):
    x = np.asarray(x, dtype=float)
    n = max(3, int(round(window_sec * fs)))
    if n % 2 == 0:
        n += 1
    if n >= len(x):
        return x - np.mean(x)
    return x - uniform_filter1d(x, size=n, mode="nearest")


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-12:
        raise RuntimeError("Signal has near-zero variance.")
    return (x - med) / scale


def prepare(gp_acc, gp_gyr, ap, fs, detrend_window_sec):
    gp_acc_mag = magnitude(gp_acc.x, gp_acc.y, gp_acc.z)
    gp_gyr_mag = magnitude(gp_gyr.x, gp_gyr.y, gp_gyr.z)
    ap_acc_mag = magnitude(ap.AccX, ap.AccY, ap.AccZ)
    ap_gyr_mag = magnitude(ap.GyrX, ap.GyrY, ap.GyrZ)

    gta, ga = resample(gp_acc.time_sec, gp_acc_mag, fs)
    gtg, gg = resample(gp_gyr.time_sec, gp_gyr_mag, fs)
    ata, aa = resample(ap.time_sec, ap_acc_mag, fs)
    atg, ag = resample(ap.time_sec, ap_gyr_mag, fs)

    gp_start, gp_end = max(gta[0], gtg[0]), min(gta[-1], gtg[-1])
    ap_start, ap_end = max(ata[0], atg[0]), min(ata[-1], atg[-1])
    gp_t = np.arange(math.ceil(gp_start*fs)/fs, math.floor(gp_end*fs)/fs + 0.5/fs, 1/fs)
    ap_t = np.arange(math.ceil(ap_start*fs)/fs, math.floor(ap_end*fs)/fs + 0.5/fs, 1/fs)

    ga = np.interp(gp_t, gta, ga)
    gg = np.interp(gp_t, gtg, gg)
    aa = np.interp(ap_t, ata, aa)
    ag = np.interp(ap_t, atg, ag)

    ga_p = robust_z(detrend_moving(ga, fs, detrend_window_sec))
    gg_p = robust_z(detrend_moving(gg, fs, detrend_window_sec))
    aa_p = robust_z(detrend_moving(aa, fs, detrend_window_sec))
    ag_p = robust_z(detrend_moving(ag, fs, detrend_window_sec))

    return dict(fs=fs, gp_t=gp_t, ap_t=ap_t,
                gp_acc_raw=ga, gp_gyr_raw=gg, ap_acc_raw=aa, ap_gyr_raw=ag,
                gp_acc_proc=ga_p, gp_gyr_proc=gg_p, ap_acc_proc=aa_p, ap_gyr_proc=ag_p)


def sliding_ncc(long_signal, template):
    """Efficient valid sliding Pearson-like NCC using FFT + rolling variance."""
    x = np.asarray(long_signal, dtype=float)
    y = np.asarray(template, dtype=float)
    if len(y) >= len(x):
        raise ValueError("Template must be shorter than the ArduPilot signal.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NCC input contains NaN/Inf.")

    y0 = y - y.mean()
    ey = np.sum(y0*y0)
    if ey <= 1e-15:
        raise ValueError("Template has zero variance.")

    num = correlate(x, y0, mode="valid", method="fft")
    m = len(y)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x*x)))
    sm = cs[m:] - cs[:-m]
    sm2 = cs2[m:] - cs2[:-m]
    ex = np.maximum(sm2 - sm*sm/m, 0.0)
    den = np.sqrt(ex * ey)
    out = np.full_like(num, np.nan, dtype=float)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return np.clip(out, -1.0, 1.0)


def correlations(s):
    gyro = sliding_ncc(s["ap_gyr_proc"], s["gp_gyr_proc"])
    accel = sliding_ncc(s["ap_acc_proc"], s["gp_acc_proc"])
    n = min(len(gyro), len(accel))
    gyro, accel = gyro[:n], accel[:n]
    # Placement at ap_t[i] aligns to GoPro time gp_t[0].
    offsets = s["ap_t"][:n] - s["gp_t"][0]
    combined = 0.70*gyro + 0.30*accel
    return pd.DataFrame({"offset_sec": offsets, "gyro_ncc": gyro,
                         "accel_ncc": accel, "combined_ncc": combined})


def distinct_candidates(df, column, k, separation_sec):
    work = df[["offset_sec", column]].dropna().sort_values(column, ascending=False)
    picked = []
    for r in work.itertuples(index=False):
        off, score = float(r.offset_sec), float(getattr(r, column))
        if all(abs(off - p["offset_sec"]) >= separation_sec for p in picked):
            picked.append({"offset_sec": off, "score": score})
            if len(picked) >= k:
                break
    return pd.DataFrame(picked)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_corr(df, best_offset, path, title):
    plt.figure(figsize=(14, 6))
    plt.plot(df.offset_sec, df.gyro_ncc, label="Gyro magnitude NCC", linewidth=1.0)
    plt.plot(df.offset_sec, df.accel_ncc, label="Acceleration magnitude NCC", linewidth=1.0, alpha=0.8)
    plt.plot(df.offset_sec, df.combined_ncc, label="Combined (0.7 gyro + 0.3 accel)", linewidth=1.5)
    plt.axvline(best_offset, linestyle="--", linewidth=1.2, label=f"Selected offset {best_offset:.3f} s")
    plt.xlabel("Candidate offset: ArduPilot TimeUS seconds at video t=0")
    plt.ylabel("Normalized cross-correlation")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig(path)


def aligned_ap(s, offset, key):
    target = s["gp_t"] + offset
    return np.interp(target, s["ap_t"], s[key], left=np.nan, right=np.nan)


def plot_overlay(gp_t, gp, ap_aligned, path, ylabel, title, xlim=None):
    m = np.isfinite(gp_t) & np.isfinite(gp) & np.isfinite(ap_aligned)
    t, g, a = gp_t[m], gp[m], ap_aligned[m]
    if xlim is not None:
        z = (t >= xlim[0]) & (t <= xlim[1])
        t, g, a = t[z], g[z], a[z]
    plt.figure(figsize=(15, 6))
    plt.plot(t, g, label="GoPro", linewidth=1.0)
    plt.plot(t, a, label="ArduPilot IMU", linewidth=1.0, alpha=0.8)
    plt.xlabel("Video-relative time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig(path)


def choose_zoom(gp_t, signal, duration_sec):
    fs = estimate_rate(gp_t)
    win = max(5, int(round(duration_sec*fs)))
    if win >= len(signal):
        return float(gp_t[0]), float(gp_t[-1])
    energy = np.asarray(signal)**2
    rolling = np.convolve(energy, np.ones(win)/win, mode="valid")
    i = int(np.argmax(rolling))
    j = min(i + win, len(gp_t)-1)
    return float(gp_t[i]), float(gp_t[j])


def parse_utc(text):
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main():
    p = argparse.ArgumentParser(description="Synchronise GoPro and ArduPilot IMU data.")
    p.add_argument("--gopro-accel", required=True, type=Path)
    p.add_argument("--gopro-gyro", required=True, type=Path)
    p.add_argument("--autopilot-csv", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--flight-id", default="unknown")
    p.add_argument("--video-name", default="unknown")
    p.add_argument("--coarse-hz", type=float, default=20.0)
    p.add_argument("--fine-hz", type=float, default=50.0)
    p.add_argument("--fine-radius-sec", type=float, default=5.0)
    p.add_argument("--detrend-window-sec", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-separation-sec", type=float, default=10.0)
    p.add_argument("--zoom-duration-sec", type=float, default=30.0)
    p.add_argument("--gps-anchor-timeus", type=float, default=None)
    p.add_argument("--gps-anchor-utc", type=str, default=None)
    a = p.parse_args()

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MACE VIDEO <-> TELEMETRY SYNCHRONISATION")
    print("="*80)
    print(f"Flight: {a.flight_id} | Video: {a.video_name}\n")

    print("[1/8] Parsing GoPro ACCL")
    gp_acc = parse_gopro_text(a.gopro_accel, "ACCL")
    print(
        f"  parsed {len(gp_acc):,} samples | "
        f"reported={gp_acc.attrs['reported_sampling_rate_hz']:.6f} Hz | "
        f"range={gp_acc.attrs['reported_start_sec']:.6f} to "
        f"{gp_acc.attrs['reported_end_sec']:.6f} s"
    )

    print("[2/8] Parsing GoPro GYRO")
    gp_gyr = parse_gopro_text(a.gopro_gyro, "GYRO")
    print(
        f"  parsed {len(gp_gyr):,} samples | "
        f"reported={gp_gyr.attrs['reported_sampling_rate_hz']:.6f} Hz | "
        f"range={gp_gyr.attrs['reported_start_sec']:.6f} to "
        f"{gp_gyr.attrs['reported_end_sec']:.6f} s"
    )
    print("[3/8] Reading ArduPilot IMU CSV")
    ap = read_autopilot_csv(a.autopilot_csv)

    gp_acc.to_csv(out/"gopro_accel_parsed.csv", index=False)
    gp_gyr.to_csv(out/"gopro_gyro_parsed.csv", index=False)
    ap.to_csv(out/"autopilot_imu_clean.csv", index=False)

    r_acc = estimate_rate(gp_acc.time_sec)
    r_gyr = estimate_rate(gp_gyr.time_sec)
    r_ap = estimate_rate(ap.time_sec)
    print(f"  GoPro ACCL: {len(gp_acc):,} samples ~{r_acc:.3f} Hz")
    print(f"  GoPro GYRO: {len(gp_gyr):,} samples ~{r_gyr:.3f} Hz")
    print(f"  ArduPilot : {len(ap):,} samples ~{r_ap:.3f} Hz")

    print(f"[4/8] Coarse NCC at {a.coarse_hz:.1f} Hz")
    coarse_s = prepare(gp_acc, gp_gyr, ap, a.coarse_hz, a.detrend_window_sec)
    coarse = correlations(coarse_s)
    coarse.to_csv(out/"correlation_coarse.csv", index=False)
    cb = coarse.loc[coarse.combined_ncc.idxmax()]
    coarse_offset = float(cb.offset_sec)
    ccands = distinct_candidates(coarse, "combined_ncc", a.top_k, a.candidate_separation_sec)
    ccands.to_csv(out/"candidate_offsets_coarse.csv", index=False)
    print(f"  Coarse best: {coarse_offset:.3f} s | combined={cb.combined_ncc:.4f} gyro={cb.gyro_ncc:.4f} accel={cb.accel_ncc:.4f}")

    print(f"[5/8] Fine NCC at {a.fine_hz:.1f} Hz within +/-{a.fine_radius_sec:.1f} s")
    fine_s = prepare(gp_acc, gp_gyr, ap, a.fine_hz, a.detrend_window_sec)
    fine_all = correlations(fine_s)
    fine_all.to_csv(out/"correlation_fine_full.csv", index=False)
    local = fine_all[(fine_all.offset_sec >= coarse_offset-a.fine_radius_sec) &
                     (fine_all.offset_sec <= coarse_offset+a.fine_radius_sec)].copy()
    if local.empty:
        raise RuntimeError("Fine search window is empty.")
    fb = local.loc[local.combined_ncc.idxmax()]
    best_offset = float(fb.offset_sec)
    local.to_csv(out/"correlation_fine_local.csv", index=False)
    fcands = distinct_candidates(fine_all, "combined_ncc", a.top_k, a.candidate_separation_sec)
    fcands.to_csv(out/"candidate_offsets_fine.csv", index=False)

    near = fine_all[(fine_all.offset_sec >= best_offset-5.0) & (fine_all.offset_sec <= best_offset+5.0)]
    gb = near.loc[near.gyro_ncc.idxmax()]
    ab = near.loc[near.accel_ncc.idxmax()]
    disagreement = abs(float(gb.offset_sec)-float(ab.offset_sec))
    print(f"  Final offset: {best_offset:.3f} s | combined={fb.combined_ncc:.4f} gyro={fb.gyro_ncc:.4f} accel={fb.accel_ncc:.4f}")
    print(f"  Local gyro peak : {float(gb.offset_sec):.3f} s")
    print(f"  Local accel peak: {float(ab.offset_sec):.3f} s")
    print(f"  Peak disagreement: {disagreement:.3f} s")

    print("[6/8] Correlation plots")
    plot_corr(coarse, best_offset, out/"correlation_coarse.png", f"{a.flight_id}: coarse synchronization search")
    plot_corr(local, best_offset, out/"correlation_fine_local.png", f"{a.flight_id}: fine synchronization search")

    print("[7/8] Overlay plots")
    s = fine_s
    ag_raw = aligned_ap(s, best_offset, "ap_gyr_raw")
    aa_raw = aligned_ap(s, best_offset, "ap_acc_raw")
    ag_proc = aligned_ap(s, best_offset, "ap_gyr_proc")
    aa_proc = aligned_ap(s, best_offset, "ap_acc_proc")

    plot_overlay(s["gp_t"], s["gp_gyr_raw"], ag_raw, out/"overlay_gyro_raw_full.png",
                 "Gyroscope magnitude (rad/s)", f"{a.flight_id}: GoPro vs ArduPilot gyro magnitude - full")
    plot_overlay(s["gp_t"], s["gp_acc_raw"], aa_raw, out/"overlay_accel_raw_full.png",
                 "Acceleration magnitude (m/s^2)", f"{a.flight_id}: GoPro vs ArduPilot acceleration magnitude - full")
    plot_overlay(s["gp_t"], s["gp_gyr_proc"], ag_proc, out/"overlay_gyro_normalized_full.png",
                 "Normalized detrended magnitude", f"{a.flight_id}: normalized gyro overlay - full")
    plot_overlay(s["gp_t"], s["gp_acc_proc"], aa_proc, out/"overlay_accel_normalized_full.png",
                 "Normalized detrended magnitude", f"{a.flight_id}: normalized acceleration overlay - full")

    zoom = choose_zoom(s["gp_t"], s["gp_gyr_proc"], a.zoom_duration_sec)
    plot_overlay(s["gp_t"], s["gp_gyr_raw"], ag_raw, out/"overlay_gyro_raw_zoom.png",
                 "Gyroscope magnitude (rad/s)", f"{a.flight_id}: gyro overlay zoom {zoom[0]:.1f}-{zoom[1]:.1f}s", zoom)
    plot_overlay(s["gp_t"], s["gp_gyr_proc"], ag_proc, out/"overlay_gyro_normalized_zoom.png",
                 "Normalized detrended magnitude", f"{a.flight_id}: normalized gyro zoom {zoom[0]:.1f}-{zoom[1]:.1f}s", zoom)
    plot_overlay(s["gp_t"], s["gp_acc_proc"], aa_proc, out/"overlay_accel_normalized_zoom.png",
                 "Normalized detrended magnitude", f"{a.flight_id}: normalized acceleration zoom {zoom[0]:.1f}-{zoom[1]:.1f}s", zoom)

    aligned = pd.DataFrame({
        "video_time_sec": s["gp_t"],
        "autopilot_time_sec": s["gp_t"] + best_offset,
        "gopro_gyro_mag": s["gp_gyr_raw"],
        "autopilot_gyro_mag_aligned": ag_raw,
        "gopro_accel_mag": s["gp_acc_raw"],
        "autopilot_accel_mag_aligned": aa_raw,
        "gopro_gyro_processed": s["gp_gyr_proc"],
        "autopilot_gyro_processed_aligned": ag_proc,
        "gopro_accel_processed": s["gp_acc_proc"],
        "autopilot_accel_processed_aligned": aa_proc,
    })
    aligned.to_csv(out/"aligned_overlay_signals.csv", index=False)

    video_start_utc = None
    if a.gps_anchor_timeus is not None and a.gps_anchor_utc:
        anchor = parse_utc(a.gps_anchor_utc)
        video_start_utc = anchor + timedelta(seconds=best_offset - a.gps_anchor_timeus/1e6)

    peak_margin = None
    if len(ccands) >= 2:
        peak_margin = float(ccands.iloc[0].score - ccands.iloc[1].score)

    result = {
        "flight_id": a.flight_id,
        "video_name": a.video_name,
        "best_offset_sec": best_offset,
        "mapping": "autopilot_time_sec ~= video_time_sec + best_offset_sec",
        "combined_ncc": float(fb.combined_ncc),
        "gyro_ncc": float(fb.gyro_ncc),
        "accel_ncc": float(fb.accel_ncc),
        "local_gyro_peak_offset_sec": float(gb.offset_sec),
        "local_accel_peak_offset_sec": float(ab.offset_sec),
        "gyro_accel_peak_difference_sec": disagreement,
        "coarse_peak_margin_vs_second_distinct": peak_margin,
        "gopro_accel_rate_hz": r_acc,
        "gopro_gyro_rate_hz": r_gyr,
        "gopro_accel_reported_rate_hz": gp_acc.attrs["reported_sampling_rate_hz"],
        "gopro_gyro_reported_rate_hz": gp_gyr.attrs["reported_sampling_rate_hz"],
        "gopro_accel_start_sec": gp_acc.attrs["reported_start_sec"],
        "gopro_accel_end_sec": gp_acc.attrs["reported_end_sec"],
        "gopro_gyro_start_sec": gp_gyr.attrs["reported_start_sec"],
        "gopro_gyro_end_sec": gp_gyr.attrs["reported_end_sec"],
        "autopilot_rate_hz": r_ap,
        "video_start_utc": video_start_utc.isoformat().replace("+00:00", "Z") if video_start_utc else None,
        "zoom_window_video_sec": list(zoom),
    }
    with (out/"synchronization_result.json").open("w") as f:
        json.dump(result, f, indent=2)

    lines = [
        "="*80,
        "MACE VIDEO <-> TELEMETRY SYNCHRONISATION REPORT",
        "="*80,
        f"Flight                : {a.flight_id}",
        f"Video                 : {a.video_name}",
        "",
        f"GoPro ACCL rate       : {r_acc:.3f} Hz",
        f"GoPro GYRO rate       : {r_gyr:.3f} Hz",
        f"ArduPilot IMU rate    : {r_ap:.3f} Hz",
        "",
        f"Coarse offset         : {coarse_offset:.3f} s",
        f"FINAL OFFSET          : {best_offset:.3f} s",
        f"Combined NCC          : {float(fb.combined_ncc):.6f}",
        f"Gyro NCC              : {float(fb.gyro_ncc):.6f}",
        f"Acceleration NCC      : {float(fb.accel_ncc):.6f}",
        f"Local gyro peak       : {float(gb.offset_sec):.3f} s",
        f"Local accel peak      : {float(ab.offset_sec):.3f} s",
        f"Peak disagreement     : {disagreement:.3f} s",
        "",
        f"Mapping: ArduPilot TimeUS seconds ~= Video time seconds + {best_offset:.3f}",
    ]
    if peak_margin is not None:
        lines.append(f"Distinct peak margin   : {peak_margin:.6f}")
    if video_start_utc:
        lines += ["", f"Estimated video start UTC: {video_start_utc.isoformat().replace('+00:00','Z')}"]
    lines += [
        "",
        "VISUAL CHECK ORDER:",
        "  1. overlay_gyro_normalized_zoom.png",
        "  2. overlay_gyro_normalized_full.png",
        "  3. overlay_accel_normalized_zoom.png",
        "  4. correlation_coarse.png",
        "  5. correlation_fine_local.png",
        "",
        "Do not accept the alignment from correlation score alone. Confirm that the",
        "selected peak is distinct and that corresponding motion peaks align visually.",
        "="*80,
    ]
    report = "\n".join(lines)
    (out/"synchronization_report.txt").write_text(report)
    print("[8/8] Report written\n")
    print(report)
    print(f"\nOutputs: {out.resolve()}")


if __name__ == "__main__":
    main()
