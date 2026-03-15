# Report Figures Plan

## Overview
This document defines the figures, plots, tables, and diagrams expected for the final TELE 523 report titled:

**Design and Simulation of an Environmental Telemetry Monitoring System Using Urban Air Quality Data**

The purpose of this file is to help the team generate report-ready visuals in a structured way throughout development, instead of waiting until the end of the project.

Each figure listed here should eventually have:
- a clear title,
- labeled axes where applicable,
- units where applicable,
- a short interpretation,
- and a saved output path inside the repository.

Recommended output location for all final visuals:

```text
results/figures/
````
---
#### Figure Naming Convention

Suggested file naming format: 
````fig_<section>_<description>.png````

Examples:

- ````fig_dataset_sensor_coverage.png````
- ````fig_preprocessing_missingness_summary.png````
- ````fig_modulation_am_waveform.png````
- ````fig_channel_noisy_signal_example.png````
- ````fig_digital_quantized_signal.png````
- ````fig_monitoring_threshold_alerts.png````
---

### Planned Report Figures by Section
#### 1. System Overview Figures
##### *Figure 1: Overall System Block Diagram*

Purpose: Show the full end-to-end telemetry system architecture.

Content:

- dataset input,
- preprocessing,
- modulation,
- channel,
- demodulation,
- digital telemetry,
- monitoring output.

**Owner**: Student 1 — System Architect\
Suggested filename: ````fig_architecture_system_block_diagram.png````

##### *Figure 2: Role-Based Subsystem Ownership Diagram*

Purpose: Show how each team member maps to the overall system.

Content:

- Student 1: integration and architecture,
- Student 2: preprocessing,
- Student 3: modulation/channel,
- Student 4: digital telemetry,
- Student 5: monitoring.

Owner: Student 1 — System Architect\
Suggested filename: ````fig_architecture_role_mapping.png````

---

#### 2. Dataset Description Figures
##### *Figure 3: Pollutant Distribution Summary*

Purpose: Show which pollutant categories are present and their relative data availability.

Possible format:

- bar chart of row counts by pollutant.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_dataset_pollutant_distribution.png````

##### *Figure 4: Sensor Coverage by Pollutant*

Purpose: Show how many sensors exist for each pollutant and help justify sensor selection.

Possible format:

- grouped bar chart,
- heatmap,
- or count matrix.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_dataset_sensor_coverage.png````

##### *Figure 5: Example Raw Time Series*

Purpose: Show a representative raw environmental signal before preprocessing.

Content:

- one selected pollutant,
- one or more selected sensors,
- visible gaps/noise if present.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_dataset_raw_signal_example.png````

---

#### 3. Preprocessing and Signal Conditioning Figures
##### *Figure 6: Missing Data Summary*

Purpose: Show the extent of missing values across sensors or pollutants.

Possible format:

- bar plot of missing percentage,
- heatmap,
- or ranked sensor quality chart.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_preprocessing_missingness_summary.png````

##### *Figure 7: QC Summary for Selected Sensors*

Purpose: Show which sensors are suitable for downstream telemetry experiments.

Possible format:

- summary table,
- bar chart,
- or sensor ranking plot.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_preprocessing_sensor_qc_summary.png````

##### *Figure 8: Gap Length Distribution*

Purpose: Show the distribution of missing-data gaps and justify interpolation policy.

Possible format:
- histogram of gap lengths,
- boxplot,
- or frequency chart.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_preprocessing_gap_distribution.png````

##### *Figure 9: Raw vs Filtered Signal*

Purpose: Show the effect of preprocessing/filtering on signal quality.

Content:

- same signal before and after filtering,
- ideally for one selected pollutant-sensor pair.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_preprocessing_raw_vs_filtered.png````

##### *Figure 10: Filter Comparison Figure*

Purpose: Compare the effect of different filters used during signal conditioning.

Possible content:

- raw signal,
- moving average output,
- Savitzky–Golay output.

Owner: Student 2 — Signal Processing Lead\
Suggested filename: ````fig_preprocessing_filter_comparison.png````

---

#### 4. Modulation and Channel Figures
##### *Figure 11: AM Modulated Signal*

Purpose: Show the waveform produced by amplitude modulation.

Owner: Student 3 — Modulation Lead\
Suggested filename: ````fig_modulation_am_waveform.png````

##### *Figure 12: FM Modulated Signal*

Purpose: Show the waveform produced by frequency modulation.

Owner: Student 3 — Modulation Lead\
Suggested filename: ````fig_modulation_fm_waveform.png````

##### *Figure 13: ASK / FSK / PSK Examples*

Purpose: Show digital modulation examples used in the project.

Possible format:

- separate figures,
- or one comparative figure with clear labeling.

Owner: Student 3 — Modulation Lead\
Suggested filenames:

- ````fig_modulation_ask_waveform.png````
- ````fig_modulation_fsk_waveform.png````
- ````fig_modulation_psk_waveform.png````

##### *Figure 14: Signal Before and After Channel Noise*

Purpose: Show the effect of the channel on the transmitted signal.

Content:

- transmitted signal,
- noisy received signal.

Owner: Student 3 — Modulation Lead\
Suggested filename: ````fig_channel_transmitted_vs_received.png````

##### *Figure 15: Demodulated Signal Recovery*

Purpose: Show how closely the recovered signal matches the transmitted or original signal.

Content:

- original pre-modulated signal,
- demodulated output,
- brief visual comparison.

Owner: Student 3 — Modulation Lead\
Suggested filename: ````fig_demodulation_signal_recovery.png````

---

#### 5. Digital Telemetry Figures
##### *Figure 16: Original Signal vs Quantized Signal*

Purpose: Show the effect of quantization on the continuous-valued environmental signal.

Owner: Student 4 — Digital Telemetry Lead\
Suggested filename: ````fig_digital_original_vs_quantized.png````

##### *Figure 17: PCM Encoding Example*

Purpose: Show how selected signal samples are converted into PCM words.

Possible format:

- signal sample values with corresponding binary codes,
- or a compact illustrative table.

Owner: Student 4 — Digital Telemetry Lead\
Suggested filename: ````fig_digital_pcm_encoding_example.png````

##### *Figure 18: Line Coding Waveform*

Purpose: Show the selected line coding method applied to the PCM bitstream.

Owner: Student 4 — Digital Telemetry Lead\
Suggested filename: ````fig_digital_line_coding.png````

##### *Figure 19: Bit Integrity / Error Summary*

Purpose: Show whether the digital telemetry representation remains reliable.

Possible format:

- simple accuracy plot,
- bit error comparison,
- or integrity check summary chart.

Owner: Student 4 — Digital Telemetry Lead\
Suggested filename: ````fig_digital_bit_integrity_summary.png````

---

#### 6. Monitoring and Output Figures
##### *Figure 20: Extracted Feature Trends*

Purpose: Show one or more monitoring features computed from the processed signal.

Possible features:

- rolling mean,
- drift indicator,
- threshold proximity,
- variability measures.

Owner: Student 5 — Monitoring Lead\
Suggested filename: ````fig_monitoring_feature_trends.png````

##### *Figure 21: Threshold Detection Example*

Purpose: Show points where the signal exceeds predefined operational thresholds.

Content:

- processed signal,
- threshold line,
- highlighted exceedance points.

Owner: Student 5 — Monitoring Lead\
Suggested filename: ````fig_monitoring_threshold_detection.png````

##### *Figure 22: Drift Detection Example*

Purpose: Show how long-term drift is identified in a selected signal stream.

Owner: Student 5 — Monitoring Lead\
Suggested filename: ````fig_monitoring_drift_detection.png````

##### *Figure 23: Alert Output Summary*

Purpose: Show the final monitoring outcome of the telemetry system.

Possible format:

- alert timeline,
- event table,
- summary chart,
- or dashboard screenshot.

Owner: Student 5 — Monitoring Lead\
Suggested filename: ````fig_monitoring_alert_summary.png````

---

#### 7. Integration and Evaluation Figures
##### *Figure 24: End-to-End Pipeline Example*

Purpose: Show one signal as it moves through the full system.

Content:

- raw signal,
- cleaned signal,
- modulated signal,
- noisy received signal,
- demodulated signal,
- quantized signal,
- final monitoring output.

Owner: Student 1 — System Architect, with inputs from all leads\
Suggested filename: ````fig_integration_end_to_end_pipeline.png````

##### *Figure 25: Comparative Performance Summary*

Purpose: Summarize the performance of different telemetry choices.

Possible comparisons:

- modulation methods,
- filtering choices,
- signal quality preservation,
- monitoring usability.

Owner: Student 1 — System Architect\
Suggested filename: ````fig_integration_comparative_summary.png````

---

#### Planned Tables for the Report

Although this file focuses on figures, the report will also need some tables.

##### Table 1: Dataset Summary Table

Purpose: Summarize pollutants, sensor counts, time range, and data quality indicators.

Owner: Student 2

##### Table 2: Module Responsibility Table

Purpose: Show team roles and subsystem ownership.

Owner: Student 1

##### Table 3: Modulation Technique Comparison Table

Purpose: Compare implemented modulation schemes and their characteristics.

Owner: Student 3

##### Table 4: Digital Telemetry Parameter Table

Purpose: Summarize quantization levels, PCM settings, and line coding choices.

Owner: Student 4

##### Table 5: Monitoring Criteria Table

Purpose: Summarize thresholds, drift logic, and alert rules.

Owner: Student 5

---

#### Minimum Required Figures 

The minimum high-value figures to prioritize are:

1. overall system block diagram,
2. pollutant/sensor coverage summary,
3. missing data summary,
4. raw vs filtered signal,
5. one modulation example,
6. channel effect example,
7. original vs quantized signal,
8. threshold detection example,
9. drift detection example,
10. end-to-end pipeline figure.

