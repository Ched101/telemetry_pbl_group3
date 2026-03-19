"""
threshold_detection.py — Student 5: Monitoring Lead

WHO/EU air quality threshold definitions and exceedance detection.

Thresholds (conservative, based on EU Directive 2008/50/EC and WHO 2021 guidelines):
  NO2  : warn 40 µg/m³ (EU annual limit), alert 80 µg/m³
  O3   : warn 100 µg/m³ (WHO 8-hr), alert 120 µg/m³ (EU target)
  PM10 : warn 45 µg/m³ (WHO 24-hr), alert 50 µg/m³ (EU 24-hr limit)
  PM2_5: warn 15 µg/m³ (WHO annual), alert 25 µg/m³ (EU annual limit)

Figure output: results/figures/fig_monitoring_threshold_detection.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Threshold table
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, dict] = {
    "NO2":   {"warn": 40.0,  "alert": 80.0,  "unit": "µg/m³"},
    "O3":    {"warn": 100.0, "alert": 120.0, "unit": "µg/m³"},
    "PM10":  {"warn": 45.0,  "alert": 50.0,  "unit": "µg/m³"},
    "PM2_5": {"warn": 15.0,  "alert": 25.0,  "unit": "µg/m³"},
}


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def check_thresholds(df: pd.DataFrame, col: str = "value_sg") -> pd.DataFrame:
    """
    Add a 'threshold_level' column to df.

    Values:
        None   — below warn level
        'warn' — between warn and alert
        'alert'— above alert level
    """
    df = df.copy()
    df["threshold_level"] = None

    for pollutant, limits in THRESHOLDS.items():
        mask = df["pollutant"] == pollutant
        warn_mask = mask & (df[col] >= limits["warn"]) & (df[col] < limits["alert"])
        alert_mask = mask & (df[col] >= limits["alert"])
        df.loc[warn_mask, "threshold_level"] = "warn"
        df.loc[alert_mask, "threshold_level"] = "alert"

    return df


def get_exceedance_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where threshold_level is not None."""
    return df[df["threshold_level"].notna()].copy()


# ---------------------------------------------------------------------------
# Figure 21: Threshold Detection
# ---------------------------------------------------------------------------

def save_threshold_figure(
    df_thresh: pd.DataFrame,
    col: str = "value_sg",
    out_path: pathlib.Path = FIGURES_DIR / "fig_monitoring_threshold_detection.png",
) -> None:
    """
    Figure 21 — Threshold Detection.
    One subplot per pollutant: signal + warn/alert threshold lines + highlighted exceedances.
    """
    pollutants = sorted(df_thresh["pollutant"].unique())
    n = len(pollutants)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, pollutant in zip(axes, pollutants):
        limits = THRESHOLDS.get(pollutant, {})
        warn_val = limits.get("warn")
        alert_val = limits.get("alert")
        unit = limits.get("unit", "µg/m³")

        sub = df_thresh[df_thresh["pollutant"] == pollutant]
        # Prefer S10 segment 2
        preferred = sub[(sub["sensor_id"] == "S10") & (sub["segment_id"] == 2)]
        if preferred.empty:
            preferred = sub
        grp = preferred.copy().reset_index(drop=True)

        x = grp.index
        ax.plot(x, grp[col], color="steelblue", linewidth=0.7, alpha=0.8, label="Signal")

        if warn_val is not None:
            ax.axhline(warn_val, color="goldenrod", linewidth=1.4, linestyle="--",
                       label=f"Warn threshold ({warn_val} {unit})")
        if alert_val is not None:
            ax.axhline(alert_val, color="firebrick", linewidth=1.4, linestyle="--",
                       label=f"Alert threshold ({alert_val} {unit})")

        warn_pts = grp[grp["threshold_level"] == "warn"]
        alert_pts = grp[grp["threshold_level"] == "alert"]

        if not warn_pts.empty:
            ax.scatter(warn_pts.index, warn_pts[col], color="goldenrod", s=14,
                       zorder=5, label=f"Warn events ({len(warn_pts)})")
        if not alert_pts.empty:
            ax.scatter(alert_pts.index, alert_pts[col], color="firebrick", s=18,
                       zorder=6, label=f"Alert events ({len(alert_pts)})")

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

    fig.suptitle("Figure 21 — Threshold Detection\nWHO/EU Air Quality Exceedance Analysis",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[threshold_detection] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from feature_extraction import load_segments
    print("Loading segments...")
    df = load_segments()

    print("Checking thresholds...")
    df_thresh = check_thresholds(df)
    exceedances = get_exceedance_events(df_thresh)

    print(f"  Total exceedance events: {len(exceedances)}")
    print(exceedances.groupby(["pollutant", "threshold_level"]).size().to_string())

    print("Saving threshold detection figure...")
    save_threshold_figure(df_thresh)
    print("Done.")
