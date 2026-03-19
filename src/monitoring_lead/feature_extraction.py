"""
feature_extraction.py — Student 5: Monitoring Lead

Loads the high-priority signal segments produced by Student 2 and computes
monitoring-relevant features:
  - rolling mean (24-sample window, i.e. 24-hour baseline)
  - rolling standard deviation (variability indicator)
  - spike flag (z-score threshold)
  - rate of change (first-order difference)
  - multi-pollutant correlation matrix

Figure output: results/figures/fig_monitoring_feature_trends.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SEGMENTS_CSV = PROJECT_ROOT / "data" / "processed" / "high_priority_segments_combined.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_segments(path: pathlib.Path | str = SEGMENTS_CSV) -> pd.DataFrame:
    """Load the combined high-priority segments CSV."""
    df = pd.read_csv(path, parse_dates=["dt_beg_utc"])
    return df


# ---------------------------------------------------------------------------
# Feature functions (operate on a single pd.Series)
# ---------------------------------------------------------------------------

def rolling_mean(series: pd.Series, window: int = 24) -> pd.Series:
    """24-sample rolling mean (each sample = 1 hour → 24-hour baseline)."""
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int = 24) -> pd.Series:
    """24-sample rolling standard deviation (variability indicator)."""
    return series.rolling(window=window, min_periods=1).std().fillna(0)


def spike_detector(series: pd.Series, z_thresh: float = 2.5) -> pd.Series:
    """Return boolean mask where absolute z-score exceeds z_thresh."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > z_thresh


def rate_of_change(series: pd.Series) -> pd.Series:
    """First-order difference (change per sample)."""
    return series.diff().fillna(0)


def multi_pollutant_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between pollutants using the first shared
    segment of each pollutant (sensor S10, segment 2 where available).

    Returns a DataFrame correlation matrix indexed by pollutant.
    """
    pollutants = df["pollutant"].unique()
    series_dict = {}
    for p in pollutants:
        sub = df[df["pollutant"] == p].copy()
        # Prefer S10 segment 2, fallback to any
        preferred = sub[(sub["sensor_id"] == "S10") & (sub["segment_id"] == 2)]
        if preferred.empty:
            preferred = sub
        series_dict[p] = preferred["value_sg"].reset_index(drop=True)

    # Align on shortest length
    min_len = min(len(s) for s in series_dict.values())
    aligned = pd.DataFrame({p: s.iloc[:min_len].values for p, s in series_dict.items()})
    return aligned.corr()


# ---------------------------------------------------------------------------
# Master feature extractor
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame, window: int = 24, z_thresh: float = 2.5) -> pd.DataFrame:
    """
    Apply all feature functions to each (pollutant, sensor_id, segment_id) group.

    Adds columns: rolling_mean, rolling_std, spike_flag, rate_of_change
    """
    results = []
    groups = df.groupby(["pollutant", "sensor_id", "segment_id"], sort=False)
    for (pollutant, sensor, seg), grp in groups:
        grp = grp.copy().reset_index(drop=True)
        s = grp["value_sg"]
        grp["rolling_mean"] = rolling_mean(s, window)
        grp["rolling_std"] = rolling_std(s, window)
        grp["spike_flag"] = spike_detector(s, z_thresh)
        grp["rate_of_change"] = rate_of_change(s)
        results.append(grp)
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 20: Feature Trends
# ---------------------------------------------------------------------------

def save_feature_trends_figure(
    df_features: pd.DataFrame,
    out_path: pathlib.Path = FIGURES_DIR / "fig_monitoring_feature_trends.png",
    window: int = 24,
) -> None:
    """
    Figure 20 — Extracted Feature Trends.
    One subplot per pollutant showing raw signal, rolling mean, and spike flags.
    """
    pollutants = sorted(df_features["pollutant"].unique())
    n = len(pollutants)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    units = {"NO2": "µg/m³", "O3": "µg/m³", "PM10": "µg/m³", "PM2_5": "µg/m³"}

    for ax, pollutant in zip(axes, pollutants):
        sub = df_features[df_features["pollutant"] == pollutant]
        # Use one representative segment per pollutant (prefer S10 seg 2)
        preferred = sub[(sub["sensor_id"] == "S10") & (sub["segment_id"] == 2)]
        if preferred.empty:
            preferred = sub
        grp = preferred.copy().reset_index(drop=True)

        x = grp.index
        unit = units.get(pollutant, "µg/m³")

        ax.plot(x, grp["value_sg"], color="steelblue", linewidth=0.7,
                alpha=0.8, label="Signal (value_sg)")
        ax.plot(x, grp["rolling_mean"], color="darkorange", linewidth=1.5,
                label=f"{window}-hr rolling mean")

        spikes = grp[grp["spike_flag"]]
        if not spikes.empty:
            ax.scatter(spikes.index, spikes["value_sg"], color="red", s=18,
                       zorder=5, label="Spike flag")

        ax.fill_between(
            x,
            grp["rolling_mean"] - grp["rolling_std"],
            grp["rolling_mean"] + grp["rolling_std"],
            alpha=0.15, color="darkorange", label="±1 std (rolling)",
        )

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

    fig.suptitle("Figure 20 — Monitoring Feature Trends\nRolling Mean, Variability Band, and Spike Detection",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[feature_extraction] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading segments...")
    df = load_segments()
    print(f"  Loaded {len(df):,} rows | pollutants: {sorted(df['pollutant'].unique())}")

    print("Extracting features...")
    df_feat = extract_features(df)
    spikes_total = df_feat["spike_flag"].sum()
    print(f"  Spike events detected: {spikes_total}")

    print("Computing multi-pollutant correlation...")
    corr = multi_pollutant_correlation(df)
    print(corr.to_string())

    print("Saving feature trends figure...")
    save_feature_trends_figure(df_feat)
    print("Done.")
