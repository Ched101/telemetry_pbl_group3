from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_signal(x):
    x = np.asarray(x, dtype=float)
    x_min = np.min(x)
    x_max = np.max(x)

    if np.isclose(x_max, x_min):
        return np.zeros_like(x)

    return 2 * (x - x_min) / (x_max - x_min) - 1


def create_demo_signal(duration=5.0, fs=100.0):
    t = np.arange(0, duration, 1 / fs)
    x = (
        0.6 * np.sin(2 * np.pi * 0.5 * t)
        + 0.3 * np.sin(2 * np.pi * 1.2 * t)
        + 0.1 * np.sin(2 * np.pi * 2.5 * t)
    )
    return t, x, fs


def load_signal_csv(csv_path, signal_col="value_sg"):
    """
    Load a single segment CSV exported by Student 2.
    Assumes hourly samples if no explicit time vector is constructed.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if signal_col not in df.columns:
        raise ValueError(f"Column '{signal_col}' not found in {csv_path.name}")

    x = df[signal_col].to_numpy(dtype=float)

    # Student 2 handoff says sampling frequency is 1 sample per hour
    fs = 1.0  # samples per hour
    t = np.arange(len(x)) / fs

    meta = {}
    for col in ["pollutant", "sensor_id", "segment_id"]:
        if col in df.columns:
            meta[col] = df[col].iloc[0]

    return t, x, fs, meta


def load_segment_from_combined(
    csv_path,
    pollutant=None,
    sensor_id=None,
    segment_id=None,
    signal_col="value_sg",
):
    """
    Load one segment from the combined high-priority file.
    If pollutant/sensor_id/segment_id are not given, the first segment is used.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"pollutant", "sensor_id", "segment_id", signal_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Combined file is missing columns: {missing}")

    if pollutant is not None:
        df = df[df["pollutant"] == pollutant]
    if sensor_id is not None:
        df = df[df["sensor_id"] == sensor_id]
    if segment_id is not None:
        df = df[df["segment_id"] == segment_id]

    if df.empty:
        raise ValueError("No matching segment found in combined file.")

    first_key = df[["pollutant", "sensor_id", "segment_id"]].iloc[0].to_dict()
    seg_df = df[
        (df["pollutant"] == first_key["pollutant"]) &
        (df["sensor_id"] == first_key["sensor_id"]) &
        (df["segment_id"] == first_key["segment_id"])
    ].copy()

    x = seg_df[signal_col].to_numpy(dtype=float)
    fs = 1.0  # samples per hour
    t = np.arange(len(x)) / fs

    meta = first_key
    return t, x, fs, meta