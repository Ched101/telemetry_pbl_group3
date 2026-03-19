"""
alert_logger.py — Student 5: Monitoring Lead

Centralised event logging for all monitoring detections.

Events logged:
  - spike        : z-score spike detected in signal
  - threshold_warn  : signal value exceeded warn threshold
  - threshold_alert : signal value exceeded alert threshold
  - drift        : linear drift detected in a segment

Log output : results/logs/monitoring_alerts.csv
Figure 23  : results/figures/fig_monitoring_alert_summary.png
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
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = LOGS_DIR / "monitoring_alerts.csv"
LOG_COLUMNS = ["timestamp", "event_type", "pollutant", "sensor_id", "segment_id",
               "value", "threshold_level", "details"]


# ---------------------------------------------------------------------------
# AlertLogger class
# ---------------------------------------------------------------------------

class AlertLogger:
    """Accumulates monitoring events and writes them to a CSV log."""

    def __init__(self, log_path: pathlib.Path = LOG_CSV):
        self.log_path = log_path
        self._records: list[dict] = []

    def log_event(
        self,
        timestamp,
        event_type: str,
        pollutant: str,
        sensor_id: str,
        segment_id,
        value: float,
        threshold_level: str | None = None,
        details: str = "",
    ) -> None:
        self._records.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "pollutant": pollutant,
            "sensor_id": sensor_id,
            "segment_id": segment_id,
            "value": round(float(value), 4) if pd.notna(value) else np.nan,
            "threshold_level": threshold_level or "",
            "details": details,
        })

    def get_log(self) -> pd.DataFrame:
        if self._records:
            return pd.DataFrame(self._records, columns=LOG_COLUMNS)
        return pd.DataFrame(columns=LOG_COLUMNS)

    def save(self) -> None:
        df = self.get_log()
        df.to_csv(self.log_path, index=False)
        print(f"[alert_logger] Alert log saved: {self.log_path}  ({len(df)} events)")

    def print_summary(self) -> None:
        df = self.get_log()
        if df.empty:
            print("[alert_logger] No events logged.")
            return
        print("\n=== MONITORING ALERT SUMMARY ===")
        summary = df.groupby(["pollutant", "event_type"]).size().reset_index(name="count")
        print(summary.to_string(index=False))
        print(f"Total events: {len(df)}")
        print("================================\n")


# ---------------------------------------------------------------------------
# Orchestrator: collect events from all detection modules
# ---------------------------------------------------------------------------

def generate_alert_log(
    features_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    drift_summary_df: pd.DataFrame,
    log_path: pathlib.Path = LOG_CSV,
) -> AlertLogger:
    """
    Build an AlertLogger from the outputs of all three detection modules.

    Parameters
    ----------
    features_df      : output of extract_features() — has spike_flag column
    threshold_df     : output of check_thresholds() — has threshold_level column
    drift_summary_df : output of drift_summary()   — has drift_flag column
    """
    logger = AlertLogger(log_path)

    # 1. Spike events (from feature_extraction)
    spikes = features_df[features_df["spike_flag"]].copy()
    for _, row in spikes.iterrows():
        logger.log_event(
            timestamp=row.get("dt_beg_utc", ""),
            event_type="spike",
            pollutant=row["pollutant"],
            sensor_id=row["sensor_id"],
            segment_id=row["segment_id"],
            value=row["value_sg"],
            details=f"z-score spike | roc={row.get('rate_of_change', np.nan):.3f}",
        )

    # 2. Threshold exceedance events (from threshold_detection)
    exceedances = threshold_df[threshold_df["threshold_level"].notna()].copy()
    for _, row in exceedances.iterrows():
        level = row["threshold_level"]
        logger.log_event(
            timestamp=row.get("dt_beg_utc", ""),
            event_type=f"threshold_{level}",
            pollutant=row["pollutant"],
            sensor_id=row["sensor_id"],
            segment_id=row["segment_id"],
            value=row["value_sg"],
            threshold_level=level,
            details=f"Threshold level: {level}",
        )

    # 3. Drift events (from drift_detection)
    drifting = drift_summary_df[drift_summary_df["drift_flag"]].copy()
    for _, row in drifting.iterrows():
        logger.log_event(
            timestamp="",
            event_type="drift",
            pollutant=row["pollutant"],
            sensor_id=row["sensor_id"],
            segment_id=row["segment_id"],
            value=row["slope"],
            details=f"slope={row['slope']:.5f}, R²={row['r_squared']:.3f}",
        )

    logger.save()
    return logger


# ---------------------------------------------------------------------------
# Figure 23: Alert Summary
# ---------------------------------------------------------------------------

def save_alert_summary_figure(
    logger: AlertLogger,
    out_path: pathlib.Path = FIGURES_DIR / "fig_monitoring_alert_summary.png",
) -> None:
    """
    Figure 23 — Alert Output Summary.
    Grouped bar chart: alert counts by pollutant and event type.
    """
    df = logger.get_log()
    if df.empty:
        print("[alert_logger] No events to plot — skipping Figure 23.")
        return

    pivot = df.groupby(["pollutant", "event_type"]).size().unstack(fill_value=0)
    pollutants = pivot.index.tolist()
    event_types = pivot.columns.tolist()

    x = np.arange(len(pollutants))
    width = 0.8 / max(len(event_types), 1)

    color_map = {
        "spike": "steelblue",
        "threshold_warn": "goldenrod",
        "threshold_alert": "firebrick",
        "drift": "mediumpurple",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, etype in enumerate(event_types):
        counts = pivot[etype].values
        offset = (i - len(event_types) / 2 + 0.5) * width
        color = color_map.get(etype, "gray")
        bars = ax.bar(x + offset, counts, width=width * 0.9, label=etype.replace("_", " ").title(),
                      color=color, alpha=0.85, edgecolor="white")
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pollutants, fontsize=10)
    ax.set_ylabel("Number of events", fontsize=10)
    ax.set_xlabel("Pollutant", fontsize=10)
    ax.set_title("Figure 23 — Monitoring Alert Summary\nEvent Counts by Pollutant and Alert Type",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[alert_logger] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "monitoring_lead"))
    from feature_extraction import load_segments, extract_features
    from threshold_detection import check_thresholds
    from drift_detection import drift_summary

    print("Loading and processing segments...")
    df = load_segments()
    df_feat = extract_features(df)
    df_thresh = check_thresholds(df)
    df_drift = drift_summary(df)

    print("Generating alert log...")
    logger = generate_alert_log(df_feat, df_thresh, df_drift)
    logger.print_summary()

    print("Saving alert summary figure...")
    save_alert_summary_figure(logger)
    print("Done.")
