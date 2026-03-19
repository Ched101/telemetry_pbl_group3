"""
drift_detection.py — Student 5: Monitoring Lead

Sensor drift detection using linear regression over each telemetry segment.

A segment is flagged as drifting when:
  - |slope| > slope_thresh  (persistent directional trend)
  - R² > 0.3               (the trend explains at least 30% of variance)

This mimics physical sensor drift: a slow, systematic bias that accumulates
over time and is distinct from real pollution variability.

Figure output: results/figures/fig_monitoring_drift_detection.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-segment drift detection
# ---------------------------------------------------------------------------

def detect_drift(
    series: pd.Series,
    min_samples: int = 72,
    slope_thresh: float = 0.05,
    r2_thresh: float = 0.3,
) -> dict:
    """
    Fit a linear regression to series and return a drift summary dict.

    Returns:
        {slope, intercept, r_squared, n_samples, drift_flag, trend_line}
        where trend_line is the array of fitted values (same length as series).
    """
    n = len(series)
    if n < min_samples:
        return {
            "slope": np.nan, "intercept": np.nan,
            "r_squared": np.nan, "n_samples": n,
            "drift_flag": False, "trend_line": np.full(n, np.nan),
        }

    x = np.arange(n, dtype=float)
    y = series.values.astype(float)
    finite_mask = np.isfinite(y)
    if finite_mask.sum() < min_samples:
        return {
            "slope": np.nan, "intercept": np.nan,
            "r_squared": np.nan, "n_samples": n,
            "drift_flag": False, "trend_line": np.full(n, np.nan),
        }

    slope, intercept, r_value, _, _ = linregress(x[finite_mask], y[finite_mask])
    r_squared = r_value ** 2
    drift_flag = (abs(slope) > slope_thresh) and (r_squared > r2_thresh)
    trend_line = intercept + slope * x

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n_samples": n,
        "drift_flag": drift_flag,
        "trend_line": trend_line,
    }


def drift_summary(
    df: pd.DataFrame,
    col: str = "value_sg",
    min_samples: int = 72,
    slope_thresh: float = 0.05,
    r2_thresh: float = 0.3,
) -> pd.DataFrame:
    """
    Apply drift detection to every (pollutant, sensor_id, segment_id) group.

    Returns a summary DataFrame with one row per segment.
    """
    records = []
    for (pollutant, sensor, seg), grp in df.groupby(["pollutant", "sensor_id", "segment_id"]):
        result = detect_drift(grp[col], min_samples, slope_thresh, r2_thresh)
        records.append({
            "pollutant": pollutant,
            "sensor_id": sensor,
            "segment_id": seg,
            "n_samples": result["n_samples"],
            "slope": result["slope"],
            "r_squared": result["r_squared"],
            "drift_flag": result["drift_flag"],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure 22: Drift Detection
# ---------------------------------------------------------------------------

def save_drift_figure(
    df: pd.DataFrame,
    col: str = "value_sg",
    out_path: pathlib.Path = FIGURES_DIR / "fig_monitoring_drift_detection.png",
    min_samples: int = 72,
    slope_thresh: float = 0.05,
    r2_thresh: float = 0.3,
) -> None:
    """
    Figure 22 — Drift Detection.
    One subplot per pollutant showing signal with linear trend overlay.
    Drifting segments are highlighted with a red trend line; non-drifting with green.
    """
    pollutants = sorted(df["pollutant"].unique())
    n = len(pollutants)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    units = {"NO2": "µg/m³", "O3": "µg/m³", "PM10": "µg/m³", "PM2_5": "µg/m³"}

    for ax, pollutant in zip(axes, pollutants):
        sub = df[df["pollutant"] == pollutant]
        preferred = sub[(sub["sensor_id"] == "S10") & (sub["segment_id"] == 2)]
        if preferred.empty:
            preferred = sub
        grp = preferred.copy().reset_index(drop=True)

        unit = units.get(pollutant, "µg/m³")
        result = detect_drift(grp[col], min_samples, slope_thresh, r2_thresh)

        ax.plot(grp.index, grp[col], color="steelblue", linewidth=0.7,
                alpha=0.8, label="Signal (value_sg)")

        trend_color = "firebrick" if result["drift_flag"] else "seagreen"
        drift_label = (
            f"Trend (slope={result['slope']:.4f}, R²={result['r_squared']:.3f}) — "
            + ("DRIFT DETECTED" if result["drift_flag"] else "No drift")
        )
        ax.plot(grp.index, result["trend_line"], color=trend_color, linewidth=2.0,
                linestyle="-", label=drift_label)

        sensor_label = preferred["sensor_id"].iloc[0]
        seg_label = preferred["segment_id"].iloc[0]
        ax.set_title(
            f"{pollutant}  |  Sensor {sensor_label}, Segment {seg_label}",
            fontsize=11, fontweight="bold",
        )
        ax.set_ylabel(f"Concentration ({unit})", fontsize=9)
        ax.set_xlabel("Sample index (1 sample = 1 hour)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 22 — Sensor Drift Detection\nLinear Trend Analysis per Segment "
        f"(slope threshold={slope_thresh}, R² threshold={r2_thresh})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[drift_detection] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "monitoring_lead"))
    from feature_extraction import load_segments

    print("Loading segments...")
    df = load_segments()

    print("Computing drift summary...")
    summary = drift_summary(df)
    drifting = summary[summary["drift_flag"]]
    print(f"  Segments analysed: {len(summary)}")
    print(f"  Drifting segments: {len(drifting)}")
    if not drifting.empty:
        print(drifting[["pollutant", "sensor_id", "segment_id", "slope", "r_squared"]].to_string(index=False))

    print("Saving drift detection figure...")
    save_drift_figure(df)
    print("Done.")
