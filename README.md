# Heat Exchanger Digital Twin

**By Grayson Kelly | Chemical Engineering & Data Science @ Western University**

[LinkedIn](www.linkedin.com/in/grayson-d-kelly) | [GitHub](https://github.com/grayson-kelly)

A physics-based digital twin for shell-and-tube heat exchanger fault detection that combines process modeling with automated fault detection and a streamlit dashboard

## Overview

My project simulates a shell-and-tube heat exchanger using the effectiveness-NTU method. I developed and calculated a first-principles model of the heat transfer between a cold and hot stream of water. All of the physics was hand derived and validated against an energy balance and thermodynamic constraints before being implemented into the project.

Synthetic sensor data was generated to represent healthy baseline operation and four realistic fault scenarios including fouling (a decreasing heat transfer coefficient [U]), sensor drift (slowly biased temperature readings), blockage (sudden drop in mass flow rate), and leakage (gradually increasing mass and energy balance difference). Each fault was validated against the expected physical behavior before I used it for detection.

A residual-based fault detection pipeline compares the live sensor readings against the physics model's predictions. An Isolation Forest was trained only on healthy operating data to automatically flag faulty operation conditions.

All of this was made into an interactive dashboard using Streamlit, which allows for a live comparison of predicted vs measured performance and presents fault alerts in real time.

I built this project to tie both Chemical Engineering and Data Science together. I combined process modeling with the kind of data pipelines, fault detection, and visualization tools used in industrial process monitoring. This is a working example of the exact skill combination I'm aiming to bring to co-op and industry roles.

## Features

- **Physics-Based Digital Twin** - An effectiveness-NTU model of a 1 shell pass, 2 tube pass shell-and-tube heat exchanger, hand-derived and checked against energy balance and thermodynamic constraints.

- **Synthetic Sensor Data Generation** - Simulated 4 hours of 1 minute sensor readings representing healthy operation, and realistic sensor noise was layered on top of the model's predictions.

- **Fault Injection** - Four fault scenarios were created: fouling, sensor drift, flow blockage, and leakage. Each of the faults were validated against expected physical behavior.

- **Residual-Based Fault Detection** - Residuals were standardized and fed into an Isolation Forest that was trained exclusively on baseline data, automatically flagging anomalous operation conditions.

- **Interactive Streamlit Dashboard** - Live comparison of measured vs. predicted performance for the selected scenario, with real-time fault alerts and an anomaly percentage metric.

## Screenshots

### Baseline Operation
![Baseline dashboard view showing normal operating conditions](assets/Baseline.png)

### Flow Blockage
![Flow blockage showing a sharp, simultaneous drop in both outlet temperatures](assets/FlowBlockage.png)

### Sensor Drift
![Sensor drift showing Th_o diverging from true values while Tc_o remains normal](assets/SensorDrift.png)

### Leak
![Leak detection via a growing energy balance mismatch](assets/Leak.png)

## Tech Stack

- **Python** - core language used

- **NumPy & pandas** - for data generation, manipulation, and analysis

- **scikit-learn** - StandardScalar and IsolationForest

- **Streamlit** - interactive dashboard

- **Plotly** - dashboard data plots/visualization

- **Matplotlib** - validation plots to check physical model

- **Jupyter Notebook** - model development, data generation, analysis

## Project Structure

```
heat-exchanger-digital-twin/
├── app/
│   └── dashboard.py           # Streamlit dashboard
├── data/                       # Generated CSVs (baseline + 4 fault scenarios)
├── notebooks/
│   ├── 01_model_validation.ipynb
│   ├── 02_data_generation.ipynb
│   ├── 03_fault_injection.ipynb
│   └── 04_fault_detection.ipynb
├── src/
│   └── heatexchanger_model.py  # Validated effectiveness-NTU physics model
├── assets/                     # README screenshots
├── requirements.txt
└── README.md
```

## Setup & How to Run

All commands below are run in Command Prompt (Windows) or Terminal (Mac/Linux).

**1. Clone the repository**

git clone https://github.com/grayson-kelly/heat-exchanger-digital-twin.git

cd heat-exchanger-digital-twin

**2. Create and activate a virtual environment**

python -m venv venv

venv\Scripts\activate

**3. Install dependencies**

pip install -r requirements.txt

**4. Run the dashboard**

streamlit run app/dashboard.py

The notebooks (`01_model_validation.ipynb` through `04_fault_detection.ipynb`) can be run in order via Jupyter to regenerate the datasets and reproduce the full analysis

## Methodology

**Physics Model** - The heat exchange was modeled using the effectiveness-NTU method for a 1 shell pass, 2 tube pass configuration. Effectiveness (ε) is calculated from NTU and the capacity ratio (Cr), then used to determining actual heat transfer (Q) and the outlet temperatures, following ε = Q / Q_max.

**Fault Detection** - For each variable we're interested in, a residual is calculated as the healthy baseline's expected value minus the measured value from the evaluated scenario. Residuals are then standardized (mean 0, standard deviation 1) using StandardScalar fit only on healthy data, then they're passed to an Isolation Forest which was trained only on healthy residuals, the model is never exposed to fault data during training, only during scoring.

**Leak Detection** - The leak does not directly affect outlet temperatures, it is detected separately using an energy balance mismatch. The energy mismatch is the difference between heat transfer calculated from inlet flow versus outlet flow. A genuine leak causes these two calculations to diverge, while healthy operation shows only variation from sensor noise.