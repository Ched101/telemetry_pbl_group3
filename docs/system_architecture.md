# System Architecture

## Overview
The system has been organized as a functional modular architecture. Each major stage of the telemetry chain is assigned to a team role and implemented in its own repository section. This allows parallel development while preserving a clear path toward full integration

The architecture is based on the following engineering principles:
 - modular subsystem ownership,
 - clean signal handoff between stages,
 - reproducible file-based outputs,
 - separation of raw, interim, and processed data,
 - and final end-to-end integration through a system controller.

---

## High-Level System Flow

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

## Main Architectural Modules

### 1. Signal Source Layer
This layer represents the origin of the telemetry signal. In a physical telemetry deployment, this would be the sensing hardware and field measurement unit. In this project, that role is played by the TURDATA Prague urban air quality dataset.

The dataset provides time-varying environmental measurements that act as the source signals for all downstream telemetry operations.

##### Main role: 
- provide real measurement signals for system simulation.

##### Typical outputs: 
- raw environmental records,
- pollutant-specific time series,
- sensor-specific signal streams.

### 2. Signal Preprocessing and Conditioning Layer
This is the first true processing stage of the system. It prepares the raw environmental measurements for telemetry analysis and downstream communication modelling. This module is foundational. Every downstream subsystem depends on the quality and consistency of its outputs.

##### Responsibilities: 
- clean and restructure the raw data,
- normalize timestamps,
- separate pollutant and sensor streams,
- identify missing values,
- summarize data quality,
- perform gap analysis,
- apply filtering where necessary,
- prepare telemetry-ready signal segments.

##### Current repository ownership
````src/signal_processing_lead/````

##### Current outputs
- ````turdata_master_tidy.csv````
- ````turdata_qc_summary.csv````

### 3. Modulation Layer
This layer represents the communication stage where the conditioned environmental signal is mapped into a transmissible form. This block converts the processed measurement signal into a form suitable for channel simulation and transmission analysis.

##### Responsibilities: 
- implement analog and/or digital modulation techniques,
- prepare waveforms for transmission,
- compare different modulation approaches,
- support realistic communication-system analysis.

##### Current repository ownership
````src/modulation_lead/````

##### Current outputs
```` to be added `````

### 4. Channel / Noise Layer
This layer simulates the communication medium between transmitter and receiver. A telemetry system must be evaluated under realistic conditions, not only ideal ones. This stage helps measure the robustness of the telemetry chain.

##### Responsibilities: 
- introduce noise,
- represent non-ideal transmission conditions,
- evaluate how signal quality changes under channel effects.

##### Current repository ownership

Handled together with modulation-related development.

### 5. Demodulation Layer
This layer attempts to recover the original information-bearing signal after it has passed through the channel. This stage determines whether the transmitted environmental information remains usable after communication effects.

##### Responsibilities: 
- reconstruct the transmitted signal,
- reduce communication-stage distortion where possible,
- prepare the recovered signal for digital telemetry or monitoring analysis.

##### Current repository ownership

Handled together with modulation-related development.


### 6. Digital Telemetry Processing Layer
This stage transforms the reconstructed or selected signal into digital telemetry form for encoding, representation, and integrity analysis. This module bridges continuous-valued environmental signals and digital communication representations.

##### Responsibilities: 
- quantization,
- PCM encoding,
- line coding,
- bit integrity verification.

##### Current repository ownership

````src/digital_telemetry_lead/````

### 7. Monitoring and Interpretation Layer
This is the application-oriented output stage of the architecture. It extracts useful information from the processed signal and turns it into monitoring-relevant outputs. This is where the value of the full telemetry chain becomes visible. The purpose of transmission is not only to move data, but to preserve information needed for practical monitoring.

##### Responsibilities: 
- feature extraction,
- threshold detection,
- drift detection,
- alert generation,
- dashboard support.

##### Current repository ownership

````src/monitoring_lead/````

### 8. System Integration and Control Layer
This layer is owned by the System Architect and is responsible for unifying all other modules into one complete project workflow. Without this layer, the project would remain a set of disconnected scripts. This layer ensures that the repository functions as one telemetry system.

##### Responsibilities: 
- define subsystem interfaces,
- coordinate repository structure,
- standardize input and output expectations,
- create integration tests,
- connect all modules into one runnable pipeline,
- support final report architecture documentation.

##### Current repository ownership

````src/system_architect/````

##### Important files
- ````pipeline_controller.py````
- ````integration_tests.py````
- ````system_diagram.py````