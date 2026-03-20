from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROCESSED_DIR / "Preprocessing report results"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DIR / "turdata_master_tidy.csv"
QC_PATH = PROCESSED_DIR / "turdata_qc_summary.csv"
GAP_PATH = PROCESSED_DIR / "turdata_gap_summary.csv"
PSD_READY_PATH = PROCESSED_DIR / "turdata_psd_ready.csv"

# One example sensor per pollutant for report-friendly plots
EXAMPLE_SIGNALS = {
    "NO2": "S10",
    "O3": "S10",
    "PM10": "S10",
    "PM2_5": "S10",
}

MOVING_AVG_WINDOW = 9
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 2


def load_data():
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    qc = pd.read_csv(QC_PATH, low_memory=False)
    gap = pd.read_csv(GAP_PATH, low_memory=False)
    psd_ready = pd.read_csv(PSD_READY_PATH, low_memory=False)

    for df in [master, psd_ready]:
        if "dt_beg_utc" in df.columns:
            df["dt_beg_utc"] = pd.to_datetime(df["dt_beg_utc"], errors="coerce", utc=True)
        if "dt_end_utc" in df.columns:
            df["dt_end_utc"] = pd.to_datetime(df["dt_end_utc"], errors="coerce", utc=True)

    for df in [master, qc, gap, psd_ready]:
        for col in ["pollutant", "sensor_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    if "value" in master.columns:
        master["value"] = pd.to_numeric(master["value"], errors="coerce")
    if "value" in psd_ready.columns:
        psd_ready["value"] = pd.to_numeric(psd_ready["value"], errors="coerce")

    return master, qc, gap, psd_ready


def assign_segments(df):
    g = df.copy().sort_values("dt_beg_utc").reset_index(drop=True)
    time_diff_hours = g["dt_beg_utc"].diff().dt.total_seconds().div(3600)
    g["segment_break"] = (time_diff_hours > 1) | (time_diff_hours.isna())
    g["segment_id"] = g["segment_break"].cumsum()
    return g


def prepare_filter_example(psd_ready, pollutant, sensor):
    df = psd_ready[
        (psd_ready["pollutant"] == pollutant) &
        (psd_ready["sensor_id"] == sensor)
    ].copy()

    df = df.dropna(subset=["dt_beg_utc", "value"]).copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("dt_beg_utc")
    df = assign_segments(df)

    parts = []
    for _, seg in df.groupby("segment_id"):
        s = seg.copy().sort_values("dt_beg_utc")

        s["moving_average"] = s["value"].rolling(
            window=MOVING_AVG_WINDOW, center=True, min_periods=1
        ).mean()

        if len(s) < SAVGOL_WINDOW:
            s["savgol"] = s["value"]
        else:
            temp = s["value"].interpolate(limit_direction="both")
            s["savgol"] = savgol_filter(
                temp,
                window_length=SAVGOL_WINDOW,
                polyorder=SAVGOL_POLYORDER,
                mode="interp"
            )
        parts.append(s)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def figure_3_pollutant_distribution(master):
    counts = master.groupby("pollutant").size().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Pollutant Distribution Summary")
    plt.xlabel("Pollutant")
    plt.ylabel("Row count")
    plt.tight_layout()
    out = FIGURES_DIR / "fig_3_pollutant_distribution_summary.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def figure_4_sensor_coverage(master):
    coverage = (
        master.groupby(["pollutant", "sensor_id"])
        .size()
        .reset_index(name="row_count")
    )

    pivot = coverage.pivot(index="sensor_id", columns="pollutant", values="row_count").fillna(0)

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Row count")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Sensor Coverage by Pollutant")
    plt.xlabel("Pollutant")
    plt.ylabel("Sensor ID")
    plt.tight_layout()
    out = FIGURES_DIR / "fig_4_sensor_coverage_by_pollutant.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def figure_5_raw_time_series(master):
    for pollutant, sensor in EXAMPLE_SIGNALS.items():
        df = master[
            (master["pollutant"] == pollutant) &
            (master["sensor_id"] == sensor)
        ].copy()

        df = df.dropna(subset=["dt_beg_utc", "value"]).copy()

        if df.empty:
            print(f"Skipped Figure 5 for {pollutant}-{sensor}: no valid timestamped data")
            continue

        df = df.sort_values("dt_beg_utc")

        plt.figure(figsize=(12, 5))
        plt.plot(df["dt_beg_utc"], df["value"])
        plt.title(f"Example Raw Time Series: {pollutant} - {sensor}")
        plt.xlabel("Time (UTC)")
        plt.ylabel(f"{pollutant} value")
        plt.tight_layout()
        out = FIGURES_DIR / f"fig_5_raw_time_series_{pollutant}_{sensor}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved {out}")


def figure_6_missing_summary(qc):
    ranked = qc.sort_values("missing_pct", ascending=False).copy()
    labels = ranked["pollutant"] + "_" + ranked["sensor_id"]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, ranked["missing_pct"])
    plt.xticks(rotation=90)
    plt.title("Missing Data Summary")
    plt.xlabel("Pollutant_Sensor")
    plt.ylabel("Missing percentage (%)")
    plt.tight_layout()
    out = FIGURES_DIR / "fig_6_missing_data_summary.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def figure_7_qc_summary(qc):
    # Keep the cleaner sensors for the QC report view
    selected = qc[qc["missing_pct"] <= 10].copy()
    selected = selected.sort_values(["pollutant", "missing_pct"])
    labels = selected["pollutant"] + "_" + selected["sensor_id"]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, selected["missing_pct"])
    plt.xticks(rotation=90)
    plt.title("QC Summary for Selected Sensors")
    plt.xlabel("Pollutant_Sensor")
    plt.ylabel("Missing percentage (%)")
    plt.tight_layout()
    out = FIGURES_DIR / "fig_7_qc_summary_selected_sensors.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

    # Also save the selected QC table
    out_csv = TABLES_DIR / "qc_selected_sensors.csv"
    selected.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


def figure_8_gap_distribution(gap):
    plt.figure(figsize=(8, 5))
    plt.hist(gap["longest_missing_run_hours"].dropna(), bins=20)
    plt.title("Gap Length Distribution")
    plt.xlabel("Longest missing run (hours)")
    plt.ylabel("Number of sensor streams")
    plt.tight_layout()
    out = FIGURES_DIR / "fig_8_gap_length_distribution.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def figure_9_raw_vs_filtered(psd_ready):
    for pollutant, sensor in EXAMPLE_SIGNALS.items():
        df = prepare_filter_example(psd_ready, pollutant, sensor)
        if df.empty:
            print(f"Skipped Figure 9 for {pollutant}-{sensor}: no data")
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        first = True
        for _, seg in df.groupby("segment_id"):
            axes[0].plot(seg["dt_beg_utc"], seg["value"], color="tab:blue",
                         label="Raw" if first else "")
            axes[1].plot(seg["dt_beg_utc"], seg["value"], color="tab:blue", alpha=0.35,
                         label="Raw" if first else "")
            axes[1].plot(seg["dt_beg_utc"], seg["savgol"], color="tab:green",
                         label="Savitzky-Golay" if first else "")
            first = False

        axes[0].set_title(f"Raw Signal: {pollutant} - {sensor}")
        axes[1].set_title("Raw vs Filtered Signal")
        axes[1].set_xlabel("Time (UTC)")
        axes[0].set_ylabel(pollutant)
        axes[1].set_ylabel(pollutant)
        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = FIGURES_DIR / f"fig_9_raw_vs_filtered_{pollutant}_{sensor}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved {out}")


def figure_10_filter_comparison(psd_ready):
    for pollutant, sensor in EXAMPLE_SIGNALS.items():
        df = prepare_filter_example(psd_ready, pollutant, sensor)
        if df.empty:
            print(f"Skipped Figure 10 for {pollutant}-{sensor}: no data")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        first = True
        for _, seg in df.groupby("segment_id"):
            axes[0].plot(seg["dt_beg_utc"], seg["value"], color="tab:blue",
                         label="Raw" if first else "")
            axes[1].plot(seg["dt_beg_utc"], seg["value"], color="tab:blue", alpha=0.35,
                         label="Raw" if first else "")
            axes[1].plot(seg["dt_beg_utc"], seg["moving_average"], color="tab:orange",
                         label="Moving Average" if first else "")
            axes[2].plot(seg["dt_beg_utc"], seg["value"], color="tab:blue", alpha=0.35,
                         label="Raw" if first else "")
            axes[2].plot(seg["dt_beg_utc"], seg["savgol"], color="tab:green",
                         label="Savitzky-Golay" if first else "")
            first = False

        axes[0].set_title(f"Raw Signal: {pollutant} - {sensor}")
        axes[1].set_title("Moving Average Filter")
        axes[2].set_title("Savitzky-Golay Filter")
        axes[2].set_xlabel("Time (UTC)")
        for ax in axes:
            ax.set_ylabel(pollutant)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = FIGURES_DIR / f"fig_10_filter_comparison_{pollutant}_{sensor}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved {out}")


def table_1_dataset_summary(master, qc):
    sensor_counts = master.groupby("pollutant")["sensor_id"].nunique().rename("sensor_count")
    row_counts = master.groupby("pollutant").size().rename("row_count")
    start_times = master.groupby("pollutant")["dt_beg_utc"].min().rename("start_time_utc")
    end_times = master.groupby("pollutant")["dt_beg_utc"].max().rename("end_time_utc")
    avg_missing = qc.groupby("pollutant")["missing_pct"].mean().rename("avg_missing_pct")
    max_missing = qc.groupby("pollutant")["missing_pct"].max().rename("max_missing_pct")

    summary = pd.concat(
        [row_counts, sensor_counts, start_times, end_times, avg_missing, max_missing],
        axis=1
    ).reset_index()

    out = TABLES_DIR / "table_1_dataset_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    master, qc, gap, psd_ready = load_data()

    figure_3_pollutant_distribution(master)
    figure_4_sensor_coverage(master)
    figure_5_raw_time_series(master)
    figure_6_missing_summary(qc)
    figure_7_qc_summary(qc)
    figure_8_gap_distribution(gap)
    figure_9_raw_vs_filtered(psd_ready)
    figure_10_filter_comparison(psd_ready)
    table_1_dataset_summary(master, qc)

    print("\nStudent 2 report outputs generated in:")
    print(REPORT_DIR)