# Student 5 — Monitoring Lead
## Technical Documentation: Environmental Telemetry Monitoring System
### TELE 523 Telemetry and Remote Control | BIUST | Group 3

---

## 1. Role Overview

As the **Monitoring Lead (Student 5)**, my responsibility sits at the final stage of the full telemetry pipeline. Once the signal has been acquired from real sensors, cleaned, transmitted through a simulated communication channel, and digitally processed, it arrives at the monitoring layer. My task is to convert that processed signal into **actionable environmental intelligence**.

In practical telemetry systems, the transmission chain is meaningless without a monitoring center that can interpret the received data and raise alerts when conditions change. That is exactly what this module does — it is the decision and interpretation layer of the system.

### Key responsibilities:
- Extract monitoring-relevant features from the received telemetry signals
- Detect threshold exceedances using WHO and EU air quality guidelines
- Detect long-term sensor drift using statistical trend analysis
- Log all detected events into a structured audit trail
- Present all findings through an interactive Streamlit dashboard
- Produce four publication-quality report figures (Figures 20–23)   

---

## 2. What Was Received From the Other Students

The monitoring module does not operate in isolation. It depends on outputs from the upstream pipeline. Understanding those inputs is essential to understanding how the monitoring module was designed.

### From Student 2 — Signal Processing Lead

The most important input file is:

```
data/processed/high_priority_segments_combined.csv
```

This file contains clean, gap-filled, Savitzky-Golay filtered environmental signals for all four pollutants (NO2, O3, PM10, PM2_5) across multiple sensors and time segments. Each row represents one hourly measurement.

**Columns received:**

| Column | Description |
|--------|-------------|
| `pollutant` | Pollutant name: NO2, O3, PM10, or PM2_5 |
| `sensor_id` | Sensor identifier (S5, S10, S12, S14) |
| `segment_id` | Segment number (consecutive clean segment within a sensor stream) |
| `dt_beg_utc` | Timestamp for the start of the segment (UTC) |
| `value_raw` | Original raw measurement value |
| `value_sg` | Savitzky-Golay filtered signal — the primary input used for monitoring |
| `sample_index` | Sequential index within each segment |

The filtered column `value_sg` was used as the monitoring input because it preserves the true environmental trend while suppressing short-term sensor noise. This was a deliberate choice aligned with the signal processing lead's recommendation.

The dataset contained **59,125 rows** across all pollutants and sensors, providing a rich and statistically meaningful basis for monitoring analysis.

### From Student 3 — Modulation Lead

Student 3 produced modulation quality metrics in:

```
results/modulation/all_modulation_results_summary.csv
```

This file contains performance metrics (BER, SNR, correlation, MSE) for all five modulation schemes (AM, FM, ASK, FSK, PSK) run on the same environmental segments. These metrics are consumed by the Modulation Quality tab of the dashboard, allowing the user to see how faithfully each modulation scheme preserved the environmental signal during transmission.

### From Student 4 — Digital Telemetry Lead

Student 4's quantization, PCM encoding, and line coding modules operate on the same dataset. While their outputs (binary representations, NRZ waveforms, parity results) do not feed directly into the monitoring module as data files, the monitoring module was designed with awareness of the digital representation stage — specifically, the fact that signals have been quantised to 8-bit resolution before arriving at the monitoring layer. This is why the monitoring analysis is applied to `value_sg` (the pre-digital representation) rather than a reconstructed quantised signal, maintaining consistency with the rest of the group's pipeline design.

---

## 3. Approach to the Task

### 3.1 Design Philosophy

The central question was: *what makes a telemetry monitoring system useful?*

Most research on datasets like TURDATA focuses on historical pollution analysis — trend studies, spatial mapping, machine learning prediction. These approaches are valuable but they miss the operational monitoring perspective: the system needs to react in near real-time, flag events as they happen, and support an operator making decisions.

The monitoring module was therefore designed around **event detection** rather than prediction. The goal is to answer three operational questions at any point in time:

1. Is the signal behaving abnormally right now? (spike detection)
2. Has the concentration crossed a dangerous level? (threshold detection)
3. Is the sensor itself drifting — becoming unreliable? (drift detection)

### 3.2 Data Source Selection

The `high_priority_segments_combined.csv` file was the natural choice as primary input. It contains only the highest-quality segments (as assessed by Student 2), which have no residual missing values and are long enough for statistically meaningful analysis. Using this file also ensures that the monitoring module is directly connected to the established pipeline handoff — consistent with the group's integration design.

### 3.3 Threshold Reference Standards

Air quality thresholds were sourced from two authoritative reference frameworks:

- **EU Directive 2008/50/EC** on ambient air quality
- **WHO Air Quality Guidelines (2021 update)**

Where the two standards differ, the more conservative (lower) threshold was used for the warning level, and the EU legal limit was used for the alert level. This reflects a two-tier alerting approach: a warning gives early notice before a legal exceedance occurs.

| Pollutant | Warning (µg/m³) | Alert (µg/m³) | Basis |
|-----------|----------------|---------------|-------|
| NO2 | 40 | 80 | EU annual limit / 2× limit |
| O3 | 100 | 120 | WHO 8-hr / EU target |
| PM10 | 45 | 50 | WHO 24-hr / EU 24-hr limit |
| PM2_5 | 15 | 25 | WHO annual / EU annual limit |

### 3.4 Statistical Approach to Spike Detection

Spikes are detected using a z-score method. For each (pollutant, sensor, segment) group, the mean and standard deviation of `value_sg` are computed. Any sample whose absolute z-score exceeds 2.5 is flagged as a spike. This threshold was chosen because it captures genuine anomalies (values more than 2.5 standard deviations from the group mean) while avoiding excessive false positives from normal variability.

The z-score method is signal-agnostic and does not require prior knowledge of the pollutant's typical range — it adapts to each segment's own distribution.

### 3.5 Drift Detection Using Linear Regression

Sensor drift in environmental monitoring manifests as a slow, systematic upward or downward trend in measured values — independent of the true environmental concentration. This is caused by physical degradation of the sensor element over time.

A linear regression (via `scipy.stats.linregress`) is fitted to each segment. A segment is flagged as drifting when **both** of the following conditions are met:

- The slope magnitude exceeds 0.05 µg/m³ per sample (i.e., 0.05 µg/m³ per hour drift rate)
- The coefficient of determination R² exceeds 0.3 (meaning the linear trend explains at least 30% of the signal variance)

Requiring both conditions prevents false positives: a segment with a genuine pollution trend (steep slope but low R² due to variability) is not flagged as sensor drift.

---

## 4. The Code: Module-by-Module Explanation

All monitoring code lives in `src/monitoring_lead/`. The module was built as five focused Python files, each with a single clear responsibility.

---

### 4.1 `feature_extraction.py`

**Purpose:** Load the segment data and compute monitoring features for every group.

**Key functions:**

`load_segments(path)` — reads the combined segments CSV, parses timestamps, and returns a flat DataFrame. This is the entry point for all downstream processing.

`rolling_mean(series, window=24)` — computes a 24-sample rolling average using pandas `.rolling()`. With 1-sample-per-hour data, this gives a 24-hour smoothed baseline. The rolling mean is used both as a visual overlay in the dashboard and as a reference for variability calculation.

`rolling_std(series, window=24)` — computes the rolling standard deviation over the same window. This forms the ±1σ variability band shown in the Signal Overview, giving the operator an immediate visual sense of how stable or noisy each pollutant stream is.

`spike_detector(series, z_thresh=2.5)` — computes the global z-score for the entire segment and returns a boolean Series. Samples with `|z| > 2.5` are flagged True. These are marked as red scatter points in all visualisations.

`rate_of_change(series)` — first-order difference (`series.diff()`). This captures how fast a pollutant is changing between consecutive hours. It is stored in the features DataFrame and included in spike event log entries.

`multi_pollutant_correlation(df)` — selects a representative segment for each pollutant (preferring sensor S10, segment 2), aligns them to the shortest shared length, and computes a Pearson correlation matrix. This reveals cross-pollutant relationships — for example, the strong negative correlation between O3 and PM10 (−0.43) is consistent with known atmospheric chemistry where particulates suppress ozone formation.

`extract_features(df)` — applies all of the above to every (pollutant, sensor_id, segment_id) group and concatenates the results into an enriched DataFrame. This is the master output of the feature extraction stage.

**Figure 20 output (`save_feature_trends_figure`):** Four subplots, one per pollutant, each showing the raw signal in blue, the rolling mean overlay in orange, the ±1σ band shaded in pale orange, and spike events as red scatter points.

---

### 4.2 `threshold_detection.py`

**Purpose:** Apply the WHO/EU threshold table to flag every sample that exceeds a health limit.

The `THRESHOLDS` dictionary is the single source of truth for all threshold values used across both the headless pipeline and the dashboard. Any future change to a threshold only needs to be made in one place.

`check_thresholds(df, col='value_sg')` — iterates over the threshold table and assigns `threshold_level` to each row:
- `None` — below all thresholds (normal)
- `'warn'` — between warn and alert limits
- `'alert'` — above alert limit

The function uses boolean masking rather than row-by-row iteration, making it efficient even on the full 59,125-row dataset.

`get_exceedance_events(df)` — simple filter to extract only rows where `threshold_level` is not None. This is used by the alert logger to collect threshold events.

**Figure 21 output (`save_threshold_figure`):** Four subplots showing the signal with horizontal dashed lines at warn (golden) and alert (red) levels, and scatter points marking actual exceedance samples.

**What the data revealed:** O3 produced no threshold exceedances (concentrations remained below 100 µg/m³ throughout). PM2_5 produced the most — 2,483 alert-level exceedances and 926 warn-level exceedances — indicating that fine particulate matter is the dominant air quality concern in the Prague dataset.

---

### 4.3 `drift_detection.py`

**Purpose:** Detect systematic linear drift within each telemetry segment using linear regression.

`detect_drift(series, min_samples=72, slope_thresh=0.05, r2_thresh=0.3)` — fits a linear regression to the time index vs signal values. Returns a dictionary containing:
- `slope` and `intercept` of the fitted line
- `r_squared` (coefficient of determination)
- `drift_flag` (True/False)
- `trend_line` array (the fitted values at every sample index, used for plotting)

Segments shorter than 72 samples (3 days) are skipped — too short for reliable drift estimation.

`drift_summary(df)` — applies `detect_drift` to every group and returns a compact summary DataFrame with one row per segment. This summary is displayed in the dashboard's Drift Analysis tab.

**Results from the Prague data:** No segments were flagged as drifting. This is expected — the data was collected from reference-grade corrected sensors (MARS-corrected measurements), so systematic sensor bias has already been removed by the data provider. The drift detection module is therefore validated as "correctly detecting no drift" rather than producing false positives.

**Figure 22 output (`save_drift_figure`):** Four subplots, each with the signal and a trend line coloured green (no drift) or red (drift detected), annotated with slope and R² values.

---

### 4.4 `alert_logger.py`

**Purpose:** Centralise all detected events into a structured log file, and produce a summary figure.

The `AlertLogger` class accumulates events from all three detection modules and writes them to `results/logs/monitoring_alerts.csv`. It does not write to disk incrementally — events are collected in memory and written once at the end via `.save()`. This avoids repeated file I/O during processing.

`generate_alert_log(features_df, threshold_df, drift_summary_df)` — the orchestrator function. It iterates over:
1. Rows in `features_df` where `spike_flag == True` → logs as `event_type='spike'`
2. Rows in `threshold_df` where `threshold_level` is not None → logs as `event_type='threshold_warn'` or `'threshold_alert'`
3. Rows in `drift_summary_df` where `drift_flag == True` → logs as `event_type='drift'`

Each log entry captures: timestamp, event type, pollutant, sensor ID, segment ID, measured value, threshold level, and a detail string with supporting statistics.

**Log format (CSV columns):**

```
timestamp | event_type | pollutant | sensor_id | segment_id | value | threshold_level | details
```

**Figure 23 output (`save_alert_summary_figure`):** A grouped bar chart showing event counts broken down by pollutant and event type. This figure is the visual summary of the entire monitoring module's output — a quick diagnostic of which pollutants are most problematic and what kind of events dominate.

**Session results:** 8,203 total events were logged across all pollutants.

---

### 4.5 `dashboard.py` — The Streamlit Application

**Purpose:** Interactive web-based monitoring interface combining all detection modules.

The dashboard is built with Streamlit and uses `@st.cache_data` to load and process the data once at startup, avoiding repeated computation during user interaction.

---

## 5. The Streamlit Dashboard — How It Works

### Launching the Dashboard

From the project root directory, run:

```
python scripts/run_dashboard.py
```

This opens the dashboard in your browser at `http://localhost:8501`. The page title is **Environmental Telemetry Monitoring System**.

### Sidebar Controls

The left sidebar contains four controls that filter the content of the first three tabs:

| Control | What it does |
|---------|-------------|
| **Pollutant** | Select NO2, O3, PM10, or PM2_5 |
| **Sensor** | Select from sensors available for the chosen pollutant (S5, S10, S12, S14) |
| **Segment** | Select a segment number (typically 1, 2, or 3) |
| **Rolling window (hours)** | Slider from 6 to 168 hours — controls the smoothing window for the rolling mean |

Two headline metrics are also shown in the sidebar:
- **Total alert events** — the total number of events logged across all pollutants
- **Drifting segments** — the count of segments flagged for drift

> **Recommended starting point:** Select `PM2_5`, sensor `S10`, segment `2`, and rolling window `24`. This segment is the longest and most data-rich, and PM2_5 has the highest number of threshold exceedances in the dataset — making it the most visually informative for demonstrating the system.

---

### Tab 1: Signal Overview

This tab shows the raw telemetry signal for the selected pollutant/sensor/segment alongside the computed monitoring features.

**What is displayed:**
- The `value_sg` signal in blue (Savitzky-Golay filtered)
- The rolling mean overlay in orange (smoothness controlled by the sidebar slider)
- A pale orange shaded band representing ±1 rolling standard deviation
- Red scatter points marking spike-flagged samples

**Three metrics below the plot:**
- Sample count
- Number of spike events in this segment
- Mean concentration with units

**What to look for:** In PM2_5 and PM10 segments, the rolling mean and raw signal track closely, indicating a genuine pollution trend rather than noise. Spike events tend to cluster at pollution peaks, which is physically reasonable — the highest-concentration events are also the most abrupt.

---

### Tab 2: Threshold Monitoring

This tab overlays the WHO/EU threshold lines on the signal and highlights exceedance samples.

**What is displayed:**
- Signal in blue
- Dashed golden line at the warning threshold
- Dashed red line at the alert threshold
- Golden scatter points for warn-level samples
- Red scatter points for alert-level samples

**Three metrics below the plot:**
- Count of warn events
- Count of alert events
- Maximum value observed in the segment

**What to look for:** In PM2_5 (sensor S10, segment 2), alert-level exceedances are frequent and persistent — the concentration remains above 25 µg/m³ for extended periods. This is the key finding that the monitoring system is designed to surface. In contrast, NO2 produces warn events but no alerts, and O3 produces no exceedances at all.

---

### Tab 3: Drift Analysis

This tab shows the linear trend fitted to the selected segment and reports the drift statistics.

**What is displayed:**
- Signal in blue
- Trend line in green (no drift) or red (drift detected)
- Slope and R² annotated in the legend
- A summary table of all segments for the selected pollutant at the bottom

**Three metrics below the plot:**
- Drift flag (YES / No)
- Slope value (µg/m³ per hour)
- R² value

**What to look for:** All Prague data segments show green trend lines with low R² values, confirming that the MARS-corrected reference sensors are stable and do not exhibit systematic drift. This is a positive result — it validates both the sensor quality and the drift detection algorithm. The summary table at the bottom allows quick comparison across all segments of the same pollutant.

---

### Tab 4: Alert Log

This tab provides a filterable, downloadable table of all 8,203 logged events.

**Controls:**
- **Filter by pollutant** — narrows the table to one pollutant
- **Filter by event type** — narrows to spike, threshold_warn, threshold_alert, or drift

**Download button:** Exports the currently filtered view as a CSV file (`monitoring_alerts_filtered.csv`). This supports operational use — an environmental monitoring operator could download the alert log for PM2_5 threshold alerts and pass it to a regulatory reporting system.

**What to look for:** Filtering to `PM2_5` + `threshold_alert` shows the highest-value events. The `details` column provides context, including the threshold level classification and any supporting statistics.

---

### Tab 5: Modulation Quality

This tab reads the modulation metrics produced by Student 3 and visualises the communication pipeline performance.

**What is displayed:**
- A preview of the full modulation results table
- A bar chart of mean Bit Error Rate (BER) by scheme (for ASK, FSK, PSK)
- A bar chart of mean recovered SNR in dB (for AM, FM)

**What to look for:** This tab connects the monitoring layer back to the communication pipeline — demonstrating that the system is aware of how reliably the signal was transmitted before it was monitored. If BER is low and SNR is high, the monitoring results are trustworthy. If channel conditions were poor, the monitoring output should be interpreted with caution.

> **Note:** If Student 3's modulation pipeline has not been run, this tab displays a warning message. Run `src/modulation_lead/run_all_modulation.py` first.

---

## 6. Running the Headless Pipeline (Without the Dashboard)

For report figure generation without launching a browser:

```
python scripts/run_monitoring.py
```

This runs the full pipeline in sequence and produces:

```
results/figures/fig_monitoring_feature_trends.png      ← Figure 20
results/figures/fig_monitoring_threshold_detection.png ← Figure 21
results/figures/fig_monitoring_drift_detection.png     ← Figure 22
results/figures/fig_monitoring_alert_summary.png       ← Figure 23
results/logs/monitoring_alerts.csv
```

The script also prints a full summary to the console including spike counts, exceedance counts by pollutant and level, drift analysis results, and an output checklist confirming all files were created.

---

## 7. What Stands Out About This Implementation

### Multi-tier alerting aligned with international standards

Rather than using a single threshold, the system implements a two-tier warn/alert model derived from WHO and EU guidelines. This mirrors how operational environmental monitoring networks function in practice. The distinction matters: a warn event tells an operator to monitor the situation; an alert event requires action.

### Statistically principled spike detection

The z-score method adapts to each segment's own distribution. This means the spike detector works equally well on a low-concentration segment (typical NO2 readings around 40–70 µg/m³) and a high-concentration segment (PM2_5 peaks above 50 µg/m³) without requiring manual threshold tuning per pollutant.

### Drift detection with dual criteria

Requiring both a minimum slope and a minimum R² prevents the two most common types of false positives in drift detection:
- A noisy segment with a large mean slope but low R² (random walk, not drift)
- A segment with a high R² fit to a very shallow trend (statistically significant but physically negligible)

This is a more rigorous approach than simply thresholding the slope alone.

### The multi-pollutant correlation result

The correlation matrix revealed a strong negative correlation between O3 and PM (−0.43 between O3 and PM10, −0.41 between O3 and PM2_5). This is consistent with known atmospheric chemistry: nitrogen oxides and volatile organic compounds that produce particulates also consume ozone through oxidation reactions. This finding was not pre-assumed — it emerged directly from the data and provides a scientifically meaningful result that strengthens the report.

### Full integration with the group pipeline

The monitoring module reads exactly the files that Student 2 produces and Student 3's modulation results. It does not require any data transformation or reformatting. The dashboard's Modulation Quality tab directly visualises Student 3's outputs. This tight integration means the monitoring layer genuinely reflects the end-to-end pipeline rather than operating independently.

### Dashboard as operational tool, not just visualisation

The inclusion of a downloadable filtered alert log turns the dashboard from a visualisation exercise into something that resembles an operational monitoring tool. An environmental officer could use it to extract specific event records for a regulatory report.

---

## 8. File Reference Summary

| File | Role |
|------|------|
| `src/monitoring_lead/feature_extraction.py` | Rolling stats, spike detection, correlation |
| `src/monitoring_lead/threshold_detection.py` | WHO/EU threshold definitions and exceedance detection |
| `src/monitoring_lead/drift_detection.py` | Linear drift analysis per segment |
| `src/monitoring_lead/alert_logger.py` | Centralised event logging and Figure 23 |
| `src/monitoring_lead/dashboard.py` | Streamlit interactive dashboard |
| `scripts/run_monitoring.py` | Headless pipeline runner (all figures + log) |
| `scripts/run_dashboard.py` | Streamlit launcher |
| `data/processed/high_priority_segments_combined.csv` | Primary input (from Student 2) |
| `results/modulation/all_modulation_results_summary.csv` | Modulation metrics input (from Student 3) |
| `results/logs/monitoring_alerts.csv` | Alert log output |
| `results/figures/fig_monitoring_*.png` | Report figures (Figures 20–23) |
| `CLAUDE.md` | Project session log and configuration notes |

---

*Document prepared by Student 5 — Monitoring Lead | TELE 523 Group 3 | 2026-03-16*
