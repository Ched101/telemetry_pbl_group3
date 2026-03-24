# TELE 523 PBL Group 3  
## Design and Simulation of an Environmental Telemetry Monitoring System Using Urban Air Quality Data

## Overview
This repository contains the implementation for a **Python-Based Problem-Based Learning (PBL)** project in **TELE 523**, focused on the **design and simulation of an end-to-end environmental telemetry monitoring system**.
The project uses **urban air quality data** to model a practical telemetry pipeline that starts from real sensor measurements and proceeds through:

- signal preprocessing,
- modulation,
- channel simulation,
- demodulation,
- digital telemetry processing,
- feature extraction,
- and monitoring output generation.

The engineering goal is to demonstrate how telemetry principles can be applied to a realistic environmental sensing problem using a modular and collaborative software architecture.

---

## Project Scope

The system is designed around an **environmental monitoring use case**, with emphasis on pollutant-related signals such as:

- PM
- NO2
- O3
- SO2
- drift behaviour
- threshold exceedance events

The project does **not** treat communication blocks as isolated theory. Instead, it models a complete telemetry chain where the output of one subsystem becomes the input of the next.

### Core objectives

- Prepare real environmental sensor data for telemetry analysis
- Simulate analog and digital telemetry stages
- Study signal behaviour under noise and transmission effects
- Convert signals into digital telemetry representations
- Detect events relevant to environmental monitoring
- Integrate all subsystems into one coherent pipeline
- Produce report-ready results, plots, and system documentation

---

## System-Level Architecture

The overall project pipeline is:

```text
Raw Environmental Dataset
        ↓
Signal Preprocessing and Conditioning
        ↓
Modulation
        ↓
Channel / Noise Simulation
        ↓
Demodulation
        ↓
Digital Telemetry Processing
        ↓
Feature Extraction and Monitoring
        ↓
Alerts / Visual Outputs / Report Figures

````
---
## Repository Structure
The repository is structured as follows: 
```text
telemetry_pbl_group3/
│
├── data/
│   ├── raw/                  # Original dataset files
│   └── processed/            # Final processed datasets used by downstream modules
│
├── docs/                     # Notes, design documents, report assets
│   ├── dataset_description.md
│   ├── report_figures.md
│   └── system_architecture.md
├── results/
│   ├── figures/              # Generated plots and diagrams                
│   └── logs/                 # Execution logs
│
├── scripts/                  # Top-level runnable scripts for pipeline execution
│   ├── run_dashboard.py                             
│   └── run_monitoring.py
│   
├── src/
│   ├── signal_processing_lead/
|   |   ├── data_loader.py
|   |   ├── export_modulation_ready_segments.py
│   │   ├── preprocessing.py
|   |   ├── preprocessing_report_outputs.py
│   │   ├── gap_analysis.py
|   |   ├── psd_analysis.py
|   |   ├── psd_multi_sensor.py
|   |   ├── psd_summary_table.py
│   │   ├── prepare_psd_ready.py
|   |   ├── sampling.py
|   |   ├── segment_selection.py
|   |   ├── sensor_selection.py
│   │   ├── filtering_comparison.py
│   │   └── filter_metrics_comparison.py
│   │
│   ├── modulation_lead/
│   │   ├── am_modulation.py
│   │   ├── fm_modulation.py
│   │   ├── digital_modulation.py
│   │   ├── metrics.py
│   │   ├── run_all_modulation.py
|   |   ├── run_batch_analysis.py
|   |   ├── summarise_results.py
│   │   └── channel.py
│   │
│   ├── digital_telemetry_lead/
│   │   ├── quantization.py
│   │   ├── pcm_encoding.py
│   │   ├── line_coding.py
│   │   └── bit_integrity_check.py
│   │
│   ├── monitoring_lead/
│   │   ├── feature_extraction.py
│   │   ├── threshold_detection.py
│   │   ├── drift_detection.py
│   │   ├── alert_logger.py
│   │   └── dashboard.py                         
├── requirements.txt          # Python dependencies
└── README.md
````
---
## Technology Stack
The project is implemented in Python and uses a scientific computing workflow suitable for signal analysis and telemetry simulation.

Main language

- Python 3.10+

Core libraries

- numpy — numerical operations
- pandas — dataset handling and tabular processing
- matplotlib — plotting and report figures
- scipy — filtering, interpolation, and signal analysis
- openpyxl — Excel dataset handling
- streamlit — dashboard and monitoring interface
- pytest — testing and validation

Development environment(s)

- VS Code
- Git and GitHub for version control

---
## Data Workflow

The repository follows a structured data progression:
1. Raw data is stored unchanged in data/raw/
2. Interim data contains cleaned or reshaped outputs
3. Processed data contains finalized files for downstream subsystem use

This separation is important because it:
- preserves the original data,
- makes preprocessing traceable,
- reduces confusion between intermediate and final outputs,
- and improves reproducibility.

--- 
## Setup Instructions
1. Clone the repository
   ````bash
   git clone https://github.com/sbuda47/telemetry_pbl_group3.git
   cd telemetry_pbl_group3
   ````
2. Create a virtual environment
   
   - Windows
   ````bash
   python -m venv .venv
   .venv\Scripts\activate
   ````
   - Linux/macOs
   ````bash
   python3 -m venv .venv
   source .venv/bin/activate
   ````
4. Install dependencies
   ````bash
   pip install -r requirements.txt
   ````
5. Verify installation
   ````bash
   python --version
   pip list
   ````

## Recommended Dependency List
If ````requirements.txt```` is still being refined, the following baseline stack is recommended:
````text
numpy
pandas
matplotlib
scipy
openpyxl
streamlit
pytest
````
---
## How to Run the Project
Since the project is modular, different subsystems may be run independently during development.

Example: run preprocessing
````bash
python src/signal_processing_lead/preprocessing.py
````

Example: run monitoring.py
````bash
python src/scripts/run_monitoring.py
````

To run the whole telemetry pipeline and view the dashboard
````bash
python scripts/run_dashboard.py
````

