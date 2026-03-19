"""
run_monitoring.py — Student 5: Monitoring Lead

Headless pipeline runner. Generates all four report figures and the alert log
without launching a browser. Run from the project root:

    python scripts/run_monitoring.py

Outputs
-------
  results/figures/fig_monitoring_feature_trends.png    (Figure 20)
  results/figures/fig_monitoring_threshold_detection.png (Figure 21)
  results/figures/fig_monitoring_drift_detection.png   (Figure 22)
  results/figures/fig_monitoring_alert_summary.png     (Figure 23)
  results/logs/monitoring_alerts.csv
"""

import sys
import pathlib

# Ensure src/ is on the path regardless of working directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "monitoring_lead"))

from monitoring_lead.feature_extraction import (
    load_segments, extract_features, save_feature_trends_figure,
    multi_pollutant_correlation,
)
from monitoring_lead.threshold_detection import (
    check_thresholds, get_exceedance_events, save_threshold_figure,
)
from monitoring_lead.drift_detection import drift_summary, save_drift_figure
from monitoring_lead.alert_logger import (
    generate_alert_log, save_alert_summary_figure,
)

SEGMENTS_CSV = PROJECT_ROOT / "data" / "processed" / "high_priority_segments_combined.csv"


def main() -> None:
    print("=" * 60)
    print("  TELE 523 — Environmental Telemetry Monitoring Pipeline")
    print("  Student 5: Monitoring Lead")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading high-priority segments...")
    df = load_segments(SEGMENTS_CSV)
    print(f"      {len(df):,} rows | pollutants: {sorted(df['pollutant'].unique())}")
    print(f"      sensors:   {sorted(df['sensor_id'].unique())}")

    # ------------------------------------------------------------------
    # 2. Feature extraction
    # ------------------------------------------------------------------
    print("\n[2/6] Extracting monitoring features...")
    df_feat = extract_features(df)
    spike_count = int(df_feat["spike_flag"].sum())
    print(f"      Spike events flagged: {spike_count}")

    print("      Multi-pollutant correlation matrix:")
    corr = multi_pollutant_correlation(df)
    for line in corr.to_string().split("\n"):
        print(f"        {line}")

    save_feature_trends_figure(df_feat)

    # ------------------------------------------------------------------
    # 3. Threshold detection
    # ------------------------------------------------------------------
    print("\n[3/6] Running threshold detection (WHO/EU limits)...")
    df_thresh = check_thresholds(df)
    exceedances = get_exceedance_events(df_thresh)
    print(f"      Total exceedance events: {len(exceedances)}")
    if not exceedances.empty:
        counts = exceedances.groupby(["pollutant", "threshold_level"]).size()
        for (p, lvl), n in counts.items():
            print(f"        {p:6s}  {lvl:14s} -> {n} events")

    save_threshold_figure(df_thresh)

    # ------------------------------------------------------------------
    # 4. Drift detection
    # ------------------------------------------------------------------
    print("\n[4/6] Running drift detection (linear regression per segment)...")
    df_drift = drift_summary(df)
    drifting = df_drift[df_drift["drift_flag"]]
    print(f"      Segments analysed: {len(df_drift)}")
    print(f"      Drifting segments: {len(drifting)}")
    if not drifting.empty:
        for _, row in drifting.iterrows():
            print(f"        {row['pollutant']:6s} {row['sensor_id']} seg{row['segment_id']} "
                  f"slope={row['slope']:.5f} R2={row['r_squared']:.3f}")

    save_drift_figure(df)

    # ------------------------------------------------------------------
    # 5. Alert logging
    # ------------------------------------------------------------------
    print("\n[5/6] Generating alert log...")
    logger = generate_alert_log(df_feat, df_thresh, df_drift)
    logger.print_summary()
    save_alert_summary_figure(logger)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n[6/6] Output summary")
    figures_dir = PROJECT_ROOT / "results" / "figures"
    logs_dir = PROJECT_ROOT / "results" / "logs"
    outputs = [
        figures_dir / "fig_monitoring_feature_trends.png",
        figures_dir / "fig_monitoring_threshold_detection.png",
        figures_dir / "fig_monitoring_drift_detection.png",
        figures_dir / "fig_monitoring_alert_summary.png",
        logs_dir / "monitoring_alerts.csv",
    ]
    all_ok = True
    for p in outputs:
        exists = p.exists()
        status = "OK" if exists else "MISSING"
        print(f"      [{status}] {p.relative_to(PROJECT_ROOT)}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("  All outputs generated successfully.")
    else:
        print("  WARNING: some outputs are missing -- check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
