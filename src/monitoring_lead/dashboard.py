"""
dashboard.py — Student 5: Monitoring Lead

Streamlit interactive dashboard for the environmental telemetry monitoring system.

Launch via:
    python scripts/run_dashboard.py
  or directly:
    streamlit run src/monitoring_lead/dashboard.py

Tabs
----
1. Signal Overview     — raw signal + rolling mean + spike flags
2. Threshold Monitoring— signal + WHO/EU threshold lines + exceedance markers
3. Drift Analysis      — per-segment linear trend analysis
4. Alert Log           — filterable event table with download button
5. Modulation Quality  — BER / SNR / correlation from modulation metrics CSV
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from monitoring_lead.feature_extraction import (
    load_segments, extract_features, multi_pollutant_correlation,
)
from monitoring_lead.threshold_detection import (
    check_thresholds, get_exceedance_events, THRESHOLDS,
)
from monitoring_lead.drift_detection import drift_summary, detect_drift
from monitoring_lead.alert_logger import (
    generate_alert_log, AlertLogger, LOG_CSV,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SEGMENTS_CSV = PROJECT_ROOT / "data" / "processed" / "high_priority_segments_combined.csv"
MODULATION_CSV = PROJECT_ROOT / "results" / "modulation" / "all_modulation_results_summary.csv"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading segments...")
def load_data():
    df = load_segments(SEGMENTS_CSV)
    df_feat = extract_features(df)
    df_thresh = check_thresholds(df_feat)
    df_drift = drift_summary(df)
    logger = generate_alert_log(df_feat, df_thresh, df_drift, LOGS_DIR / "monitoring_alerts.csv")
    return df_feat, df_thresh, df_drift, logger


@st.cache_data(show_spinner="Loading modulation metrics...")
def load_modulation_metrics():
    if MODULATION_CSV.exists():
        return pd.read_csv(MODULATION_CSV)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper: get one segment
# ---------------------------------------------------------------------------

def get_segment(df: pd.DataFrame, pollutant: str, sensor: str, segment: int) -> pd.DataFrame:
    return df[
        (df["pollutant"] == pollutant) &
        (df["sensor_id"] == sensor) &
        (df["segment_id"] == segment)
    ].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_signal_overview(df_feat: pd.DataFrame, pollutant: str, sensor: str,
                           segment: int, window: int) -> None:
    st.subheader(f"Signal Overview — {pollutant} | Sensor {sensor} | Segment {segment}")
    grp = get_segment(df_feat, pollutant, sensor, segment)
    if grp.empty:
        st.warning("No data for this selection.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    unit = THRESHOLDS.get(pollutant, {}).get("unit", "µg/m³")

    ax.plot(grp.index, grp["value_sg"], color="steelblue", linewidth=0.8,
            alpha=0.85, label="Signal (value_sg)")
    ax.plot(grp.index, grp["rolling_mean"], color="darkorange", linewidth=1.8,
            label=f"{window}-hr rolling mean")
    ax.fill_between(grp.index,
                    grp["rolling_mean"] - grp["rolling_std"],
                    grp["rolling_mean"] + grp["rolling_std"],
                    alpha=0.15, color="darkorange", label="±1 std")

    spikes = grp[grp["spike_flag"]]
    if not spikes.empty:
        ax.scatter(spikes.index, spikes["value_sg"], color="red", s=25,
                   zorder=5, label=f"Spikes ({len(spikes)})")

    ax.set_xlabel("Sample index (1 = 1 hour)")
    ax.set_ylabel(f"Concentration ({unit})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", len(grp))
    col2.metric("Spike events", int(grp["spike_flag"].sum()))
    col3.metric("Mean", f"{grp['value_sg'].mean():.2f} {unit}")


def render_threshold_panel(df_thresh: pd.DataFrame, pollutant: str,
                           sensor: str, segment: int) -> None:
    st.subheader(f"Threshold Monitoring — {pollutant} | Sensor {sensor} | Segment {segment}")
    grp = get_segment(df_thresh, pollutant, sensor, segment)
    if grp.empty:
        st.warning("No data for this selection.")
        return

    limits = THRESHOLDS.get(pollutant, {})
    unit = limits.get("unit", "µg/m³")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grp.index, grp["value_sg"], color="steelblue", linewidth=0.8, alpha=0.85, label="Signal")

    if "warn" in limits:
        ax.axhline(limits["warn"], color="goldenrod", linewidth=1.5, linestyle="--",
                   label=f"Warn ({limits['warn']} {unit})")
    if "alert" in limits:
        ax.axhline(limits["alert"], color="firebrick", linewidth=1.5, linestyle="--",
                   label=f"Alert ({limits['alert']} {unit})")

    warn_pts = grp[grp["threshold_level"] == "warn"]
    alert_pts = grp[grp["threshold_level"] == "alert"]
    if not warn_pts.empty:
        ax.scatter(warn_pts.index, warn_pts["value_sg"], color="goldenrod", s=18,
                   zorder=5, label=f"Warn events ({len(warn_pts)})")
    if not alert_pts.empty:
        ax.scatter(alert_pts.index, alert_pts["value_sg"], color="firebrick", s=22,
                   zorder=6, label=f"Alert events ({len(alert_pts)})")

    ax.set_xlabel("Sample index (1 = 1 hour)")
    ax.set_ylabel(f"Concentration ({unit})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Warn events", len(warn_pts))
    col2.metric("Alert events", len(alert_pts))
    col3.metric("Max value", f"{grp['value_sg'].max():.2f} {unit}")


def render_drift_panel(df: pd.DataFrame, df_drift: pd.DataFrame,
                       pollutant: str, sensor: str, segment: int) -> None:
    st.subheader(f"Drift Analysis — {pollutant} | Sensor {sensor} | Segment {segment}")
    grp = get_segment(df, pollutant, sensor, segment)
    if grp.empty:
        st.warning("No data for this selection.")
        return

    result = detect_drift(grp["value_sg"])
    unit = THRESHOLDS.get(pollutant, {}).get("unit", "µg/m³")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grp.index, grp["value_sg"], color="steelblue", linewidth=0.8, alpha=0.85, label="Signal")
    trend_color = "firebrick" if result["drift_flag"] else "seagreen"
    label = ("DRIFT DETECTED" if result["drift_flag"] else "No drift") + \
            f" | slope={result['slope']:.4f}, R²={result['r_squared']:.3f}"
    ax.plot(grp.index, result["trend_line"], color=trend_color, linewidth=2.0, label=label)

    ax.set_xlabel("Sample index (1 = 1 hour)")
    ax.set_ylabel(f"Concentration ({unit})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Drift flag", "YES" if result["drift_flag"] else "No")
    col2.metric("Slope", f"{result['slope']:.5f}" if pd.notna(result["slope"]) else "N/A")
    col3.metric("R²", f"{result['r_squared']:.3f}" if pd.notna(result["r_squared"]) else "N/A")

    st.markdown("**All segments — drift summary:**")
    st.dataframe(
        df_drift[df_drift["pollutant"] == pollutant]
        .style.applymap(lambda v: "background-color: #ffcccc" if v is True else "",
                        subset=["drift_flag"]),
        use_container_width=True,
    )


def render_alert_log(logger: AlertLogger) -> None:
    st.subheader("Alert Log")
    log_df = logger.get_log()
    if log_df.empty:
        st.info("No alert events logged.")
        return

    # Filters
    col1, col2 = st.columns(2)
    pollutants = ["All"] + sorted(log_df["pollutant"].dropna().unique().tolist())
    event_types = ["All"] + sorted(log_df["event_type"].dropna().unique().tolist())
    sel_poll = col1.selectbox("Filter by pollutant", pollutants)
    sel_type = col2.selectbox("Filter by event type", event_types)

    filtered = log_df.copy()
    if sel_poll != "All":
        filtered = filtered[filtered["pollutant"] == sel_poll]
    if sel_type != "All":
        filtered = filtered[filtered["event_type"] == sel_type]

    st.dataframe(filtered, use_container_width=True)
    st.metric("Events shown", len(filtered))

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered log (CSV)", data=csv_data,
                       file_name="monitoring_alerts_filtered.csv", mime="text/csv")


def render_modulation_quality(metrics_df: pd.DataFrame) -> None:
    st.subheader("Modulation Quality — Pipeline Telemetry Metrics")
    if metrics_df.empty:
        st.warning("Modulation results file not found. Run Student 3's modulation pipeline first.")
        return

    st.dataframe(metrics_df.head(50), use_container_width=True)

    # BER chart (digital schemes)
    if "ber" in metrics_df.columns and "scheme" in metrics_df.columns:
        digital = metrics_df[metrics_df["ber"].notna()].copy()
        if not digital.empty:
            ber_mean = digital.groupby("scheme")["ber"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(ber_mean["scheme"], ber_mean["ber"], color="steelblue", alpha=0.85)
            ax.set_ylabel("Mean BER")
            ax.set_title("Mean Bit Error Rate by Modulation Scheme")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    # SNR / correlation chart (analog schemes)
    if "recovered_snr_db" in metrics_df.columns:
        analog = metrics_df[metrics_df["recovered_snr_db"].notna()].copy()
        if not analog.empty:
            snr_mean = analog.groupby("scheme")["recovered_snr_db"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(snr_mean["scheme"], snr_mean["recovered_snr_db"], color="darkorange", alpha=0.85)
            ax.set_ylabel("Mean Recovered SNR (dB)")
            ax.set_title("Mean Recovered SNR by Modulation Scheme")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Tab: Event Scope
# ---------------------------------------------------------------------------

def render_event_scope(df_thresh: pd.DataFrame, df_feat: pd.DataFrame,
                       pollutant: str) -> None:
    st.subheader("Event Scope Analysis — Local vs City-Wide")
    st.markdown(
        "Determines whether a pollution event was seen by a **single sensor** (local) "
        "or by **multiple sensors at once** (city-wide). "
        "Events are grouped by segment — each segment is a continuous time window."
    )

    from matplotlib.patches import Patch

    pollutant_sel = pollutant

    df_p = df_thresh[df_thresh["pollutant"] == pollutant_sel].copy()
    total_sensors = df_p["sensor_id"].nunique()

    # --- Threshold-based scope ---
    exceeded = df_p[df_p["threshold_level"].notna()]

    scope = (
        exceeded.groupby("segment_id")["sensor_id"]
        .nunique()
        .reset_index()
        .rename(columns={"sensor_id": "sensors_exceeding"})
    )
    scope["total_sensors"] = total_sensors
    scope["pct_sensors"] = (scope["sensors_exceeding"] / total_sensors * 100).round(1)

    def classify(n, total):
        if n == 1:
            return "Local"
        elif n / total < 0.5:
            return "Partial"
        else:
            return "City-Wide"

    scope["scope"] = scope.apply(
        lambda r: classify(r["sensors_exceeding"], r["total_sensors"]), axis=1
    )

    # --- Spike-based scope ---
    df_spikes = df_feat[df_feat["pollutant"] == pollutant_sel].copy()
    spike_scope = (
        df_spikes[df_spikes["spike_flag"]]
        .groupby("segment_id")["sensor_id"]
        .nunique()
        .reset_index()
        .rename(columns={"sensor_id": "sensors_spiking"})
    )
    scope = scope.merge(spike_scope, on="segment_id", how="left").fillna({"sensors_spiking": 0})
    scope["sensors_spiking"] = scope["sensors_spiking"].astype(int)

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sensors", total_sensors)
    col2.metric("Segments with exceedances", len(scope))
    col3.metric("Local events", int((scope["scope"] == "Local").sum()))
    col4.metric("City-Wide events", int((scope["scope"] == "City-Wide").sum()))

    # --- Bar chart ---
    color_map = {"Local": "steelblue", "Partial": "darkorange", "City-Wide": "firebrick"}
    bar_colors = [color_map[s] for s in scope["scope"]]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(scope["segment_id"].astype(str), scope["sensors_exceeding"],
           color=bar_colors, alpha=0.85, zorder=3)
    citywide_threshold = total_sensors * 0.5
    ax.axhline(citywide_threshold, color="firebrick", linestyle="--", linewidth=1.3,
               label=f"City-Wide threshold (50% = {citywide_threshold:.0f} sensors)")
    ax.set_xlabel("Segment ID")
    ax.set_ylabel("Sensors with threshold exceedance")
    ax.set_title(f"Event Scope per Segment — {pollutant_sel}")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    legend_patches = [
        Patch(facecolor="steelblue",  alpha=0.85, label="Local (1 sensor)"),
        Patch(facecolor="darkorange", alpha=0.85, label="Partial (<50% of sensors)"),
        Patch(facecolor="firebrick",  alpha=0.85, label="City-Wide (≥50% of sensors)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    # --- Summary table ---
    st.markdown("**Segment scope summary:**")
    display_cols = ["segment_id", "sensors_exceeding", "sensors_spiking",
                    "pct_sensors", "total_sensors", "scope"]
    st.dataframe(
        scope[display_cols].rename(columns={
            "segment_id": "Segment",
            "sensors_exceeding": "Sensors w/ exceedance",
            "sensors_spiking": "Sensors w/ spikes",
            "pct_sensors": "% of network",
            "total_sensors": "Total sensors",
            "scope": "Classification",
        }),
        use_container_width=True,
    )

    # --- Drill-down ---
    st.markdown("---")
    st.markdown("**Drill down — which sensors triggered in a specific segment?**")
    seg_options = sorted(scope["segment_id"].unique())
    if seg_options:
        sel_seg = st.selectbox("Select Segment", seg_options, key="scope_seg")

        thresh_detail = exceeded[exceeded["segment_id"] == sel_seg]
        spike_detail  = df_spikes[
            (df_spikes["segment_id"] == sel_seg) & (df_spikes["spike_flag"])
        ]

        all_sensors = sorted(df_p[df_p["segment_id"] == sel_seg]["sensor_id"].unique())
        rows = []
        for s in all_sensors:
            warn_n  = int((thresh_detail[thresh_detail["sensor_id"] == s]["threshold_level"] == "warn").sum())
            alert_n = int((thresh_detail[thresh_detail["sensor_id"] == s]["threshold_level"] == "alert").sum())
            spike_n = int((spike_detail["sensor_id"] == s).sum())
            triggered = warn_n + alert_n > 0 or spike_n > 0
            rows.append({
                "Sensor": s,
                "Warn events": warn_n,
                "Alert events": alert_n,
                "Spike events": spike_n,
                "Triggered": "Yes" if triggered else "No",
            })
        detail_df = pd.DataFrame(rows)
        st.dataframe(detail_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Env Telemetry Monitor",
        page_icon="🌍",
        layout="wide",
    )

    st.title("🌍 Environmental Telemetry Monitoring System")

    # Load data (cached)
    with st.spinner("Initialising monitoring pipeline..."):
        df_feat, df_thresh, df_drift, logger = load_data()
        metrics_df = load_modulation_metrics()

    # ---- Sidebar ----
    st.sidebar.header("Controls")
    pollutants = sorted(df_feat["pollutant"].unique())
    pollutant = st.sidebar.selectbox("Pollutant", pollutants)

    sensors = sorted(df_feat[df_feat["pollutant"] == pollutant]["sensor_id"].unique())
    sensor = st.sidebar.selectbox("Sensor", sensors)

    segments = sorted(df_feat[
        (df_feat["pollutant"] == pollutant) & (df_feat["sensor_id"] == sensor)
    ]["segment_id"].unique())
    segment = st.sidebar.selectbox("Segment", segments)

    window = st.sidebar.slider("Rolling window (hours)", min_value=6, max_value=168,
                               value=24, step=6)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick stats**")
    total_events = len(logger.get_log())
    st.sidebar.metric("Total alert events", total_events)
    st.sidebar.metric("Drifting segments", int(df_drift["drift_flag"].sum()))

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Signal Overview",
        "Threshold Monitoring",
        "Drift Analysis",
        "Alert Log",
        "Modulation Quality",
        "Event Scope",
    ])

    with tab1:
        render_signal_overview(df_feat, pollutant, sensor, segment, window)

    with tab2:
        render_threshold_panel(df_thresh, pollutant, sensor, segment)

    with tab3:
        render_drift_panel(df_feat, df_drift, pollutant, sensor, segment)

    with tab4:
        render_alert_log(logger)

    with tab5:
        render_modulation_quality(metrics_df)

    with tab6:
        render_event_scope(df_thresh, df_feat, pollutant)


if __name__ == "__main__":
    main()
