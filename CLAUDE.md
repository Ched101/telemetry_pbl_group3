# CLAUDE.md — TELE 523 PBL Group 3

## Project
TELE 523 Telemetry and Remote Control — Python-Based PBL Laboratory
**Title:** Design and Simulation of an Environmental Telemetry Monitoring System Using Urban Air Quality Data

## Domain
Environmental telemetry sensor network using TURDATA Prague urban air quality dataset.
Pollutants: NO2, O3, PM10, PM2_5

## User Role
Student 5 — Monitoring Lead
Responsibilities: feature extraction, threshold detection, drift detection, alert logging, Streamlit dashboard

## Branch Convention
All Student 5 work is done on branch: `student5_monitoring`
Never push to GitHub without asking first.

## Monitoring Output
Using Streamlit for the interactive dashboard.
Headless figure generation via `scripts/run_monitoring.py`.

## Key Paths
- Input signals: `data/processed/high_priority_segments_combined.csv`
- Modulation metrics: `results/modulation/all_modulation_results_summary.csv`
- Alert log: `results/logs/monitoring_alerts.csv`
- Monitoring figures: `results/figures/fig_monitoring_*.png`
- Module code: `src/monitoring_lead/`
- Runner scripts: `scripts/run_monitoring.py`, `scripts/run_dashboard.py`

## Required Report Figures (Student 5)
- Fig 20: `fig_monitoring_feature_trends.png`
- Fig 21: `fig_monitoring_threshold_detection.png`
- Fig 22: `fig_monitoring_drift_detection.png`
- Fig 23: `fig_monitoring_alert_summary.png`

## Thresholds Used (WHO/EU)
| Pollutant | Warn (µg/m³) | Alert (µg/m³) |
|-----------|-------------|--------------|
| NO2       | 40          | 80           |
| O3        | 100         | 120          |
| PM10      | 45          | 50           |
| PM2_5     | 15          | 25           |

## Rules
- Always update CLAUDE.md after each work session or significant change
- Never push to GitHub without asking the user first
- Use branch `student5_monitoring` for all work
- Streamlit is the monitoring output interface
- Run `python scripts/run_monitoring.py` to regenerate all figures headlessly
- Run `python scripts/run_dashboard.py` to launch the Streamlit dashboard

## Session Log
- 2026-03-16: Initial implementation — created all monitoring modules, Streamlit dashboard, headless runner
- 2026-03-19: Pushed student5_monitoring branch to GitHub repository
