#Importing everything needed for the dashboard
import streamlit as st  #streamlit itself, for the app interface
import pandas as pd  #for loading CSV's
import numpy as np  #arrays
import sys  # needed to add our src folder to the path
import os #to work with file path
current_dir = os.path.dirname(os.path.abspath(__file__))  #folder dashboard.py lives in
sys.path.append(os.path.join(current_dir, "..", "src"))  #up one level from there to src
DATA_DIR = os.path.join(current_dir, "..", "data")  #reuse this everywhere
from heatexchanger_model import effectiveness_ntu  #my validated physics model
from sklearn.preprocessing import StandardScaler  #scaling residuals
from sklearn.ensemble import IsolationForest  #anomaly detection
import plotly.express as px  #more customizable plotting library

#Trains the model once and caches it (aka storing the already trained model), so it doesn't retrain every time the dashboard reruns
@st.cache_resource  #tells streamlit to remember this function's result instead of recomputing it every time
def train_model():
    baseline_dataframe = pd.read_csv(os.path.join(DATA_DIR, "baseline_data.csv")) #CSV read

    baseline_dataframe["Th_o_residual"] = baseline_dataframe["Th_o_true"] - baseline_dataframe["Th_o_measured"]  #computes Th_o residual
    baseline_dataframe["Tc_o_residual"] = baseline_dataframe["Tc_o_true"] - baseline_dataframe["Tc_o_measured"]  #computes Tc_o residual

    baseline_residuals = baseline_dataframe[["Th_o_residual", "Tc_o_residual"]] 

    scaler = StandardScaler() #creating scaler
    scaler.fit(baseline_residuals) #learning baseline's mean and std
    baseline_scaled = scaler.transform(baseline_residuals) #scales the baseline residuals

    isolation_forest = IsolationForest(contamination=0.05, random_state=60)  #create the isolationforest model
    isolation_forest.fit(baseline_scaled)  #train it only on baseline residuals

    return scaler, isolation_forest, baseline_dataframe  #returns everything the app will need

st.title("Heat Exchanger Digital Twin Dashboard")  # displays the title at the top of the page

#displays my description of my project
st.write("This dashboard visualizes a physics-based digital twin of a shell and tube "
         "heat exchanger, it compares live sensor readings against model predictions to "
         "detect equipment faults in real time. Select a scenario below to explore the 5 operating "
         "scenarios: healthy operation, fouling, sensor drift, flow blockage, or leakage."
)

scaler, isolation_forest, baseline_dataframe = train_model()  #calls the training function, caches the results

#Dropdown Menu 
with st.sidebar:
    scenario = st.selectbox("Select an operating scenario:", ["Baseline", "Fouling", "Sensor Drift", "Flow Blockage", "Leak"])

#Loading the correct dataset
if scenario == "Baseline":
    display_dataframe = baseline_dataframe  #already loaded from training function
elif scenario == "Fouling":
    display_dataframe = pd.read_csv(os.path.join(DATA_DIR, "fouling_fault_data.csv"))
elif scenario == "Sensor Drift":
    display_dataframe = pd.read_csv(os.path.join(DATA_DIR, "drift_fault_data.csv"))
elif scenario == "Flow Blockage":
    display_dataframe = pd.read_csv(os.path.join(DATA_DIR, "blockage_fault_data.csv"))
elif scenario == "Leak":
    display_dataframe = pd.read_csv(os.path.join(DATA_DIR, "leak_fault_data.csv"))

#Plot the fault signal appropriate to the leak
if scenario == "Leak":
    st.write("**Energy Balance Mismatch**")  #leak's real fault signal
    mismatch_chart_data = display_dataframe[["energy_mismatch"]]
    fig_mismatch = px.line(mismatch_chart_data, labels={"value": "Energy Mismatch (W)", "index": "Time step"}, color_discrete_map={"energy_mismatch": "#E8543A"})
    fig_mismatch.update_layout(legend_title_text="")
    st.plotly_chart(fig_mismatch)
else:
    col1, col2 = st.columns(2)  #splits the page into two equal side-by-side sections

    with col1:
        st.write("**Hot Outlet (Th_o)**")  #header
        th_chart_data = display_dataframe[["Th_o_true", "Th_o_measured"]]  #select Th_o columns
        fig_th = px.line(th_chart_data, labels={"value": "Temperature (°C)", "index": "Time step"}, color_discrete_map={"Th_o_true": "#4A90D9", "Th_o_measured": "#E8543A"})
        fig_th.update_layout(legend_title_text="")
        st.plotly_chart(fig_th)

    with col2:
        st.write("**Cold Outlet (Tc_o)**")  #header
        tc_chart_data = display_dataframe[["Tc_o_true", "Tc_o_measured"]]  #select Tc_o columns
        fig_tc = px.line(tc_chart_data, labels={"value": "Temperature (°C)", "index": "Time step"}, color_discrete_map={"Tc_o_true": "#4A90D9", "Tc_o_measured": "#E8543A"})
        fig_tc.update_layout(legend_title_text="")
        st.plotly_chart(fig_tc)

#Compute residuals for the selected scenario (NOT LEAK) ---> Basically taken right from 04_fault_detection
if scenario != "Leak": #If tthis is not the leak scenario
    display_dataframe["Th_o_residual"] = baseline_dataframe["Th_o_true"] - display_dataframe["Th_o_measured"]  #residual vs baseline
    display_dataframe["Tc_o_residual"] = baseline_dataframe["Tc_o_true"] - display_dataframe["Tc_o_measured"]  #residual vs baseline
    residuals = display_dataframe[["Th_o_residual", "Tc_o_residual"]]  #select both residual columns
    scaled_residuals = scaler.transform(residuals)  #apply same scaling learned from baseline
    scores = isolation_forest.predict(scaled_residuals)  #classify each row: 1 = normal, -1 = anomaly
    percent_flagged = (scores == -1).sum() / len(scores) * 100  #calculate percentage flagged
    if percent_flagged > 10:  #if more than 10% of readings look anomalous
        st.error(f"⚠️ Fault Alert")  #red alert box
    else:
        st.success(f"✅ Operating normally")  #green success box
    delta_color_choice = "inverse" if percent_flagged > 10 else "normal"  #if over 10% of data is faulty turn it red, if under green
    st.metric(label="Percent Flagged as Anomalous", value=f"{percent_flagged:.1f}%", delta=f"{percent_flagged - 5:.1f}% vs. healthy baseline", delta_color=delta_color_choice)
        
#Leak portion ---> Train model and almost exactly taken from 04_fault_detection
@st.cache_resource  #trains once, cached across reruns
def train_leak_model():
    baseline_dataframe = pd.read_csv(os.path.join(DATA_DIR, "baseline_data.csv"))  #load baseline fresh
    sensor_noise_mdot_h_baseline = np.random.normal(loc=0, scale=0.01, size=240)  #baseline sensor noise
    mdot_h_outlet_healthy = baseline_dataframe["mdot_h"].values + sensor_noise_mdot_h_baseline 

    C_h_inlet_baseline = baseline_dataframe["mdot_h"].values * 4180  #heat capacity rate of inlet
    C_h_outlet_baseline = mdot_h_outlet_healthy * 4180  #heat capacity rate of outlet

    Q_inlet_baseline = C_h_inlet_baseline * (baseline_dataframe["Th_i"].values - baseline_dataframe["Th_o_true"].values)
    Q_outlet_baseline = C_h_outlet_baseline * (baseline_dataframe["Th_i"].values - baseline_dataframe["Th_o_true"].values)

    baseline_dataframe["energy_mismatch"] = Q_inlet_baseline - Q_outlet_baseline  

    leak_baseline_residuals = baseline_dataframe[["energy_mismatch"]]  #select the one column

    leak_scaler = StandardScaler()  #new scaler just for leak
    leak_scaler.fit(leak_baseline_residuals)  #learn baseline energy_mismatch stats
    baseline_energy_scaled = leak_scaler.transform(leak_baseline_residuals)  #scale baseline's own values

    leak_isolation_forest = IsolationForest(contamination=0.05, random_state=60)  #new model just for leak
    leak_isolation_forest.fit(baseline_energy_scaled)  #train on baseline energy_mismatch only

    return leak_scaler, leak_isolation_forest  

leak_scaler, leak_isolation_forest = train_leak_model()  #calls the leak training function, caches results

#Computes leak alert separately (energy_mismatch not Th/c_o)
if scenario == "Leak":
    leak_baseline_df = pd.read_csv(os.path.join(DATA_DIR, "baseline_data.csv"))  #reload baseline for the mismatch reference

    #Recreate normal mdot_h_outlet reference 
    sensor_noise_mdot_h_baseline = np.random.normal(loc=0, scale=0.01, size=240)
    mdot_h_outlet_healthy = leak_baseline_df["mdot_h"].values + sensor_noise_mdot_h_baseline

    C_h_inlet_baseline = leak_baseline_df["mdot_h"].values * 4180
    C_h_outlet_baseline = mdot_h_outlet_healthy * 4180

    Q_inlet_baseline = C_h_inlet_baseline * (leak_baseline_df["Th_i"].values - leak_baseline_df["Th_o_true"].values)
    Q_outlet_baseline = C_h_outlet_baseline * (leak_baseline_df["Th_i"].values - leak_baseline_df["Th_o_true"].values)

    leak_baseline_df["energy_mismatch"] = Q_inlet_baseline - Q_outlet_baseline

    #Score leak dataset
    leak_residuals = display_dataframe[["energy_mismatch"]]  #select the mismatch column from the leak data
    leak_scaled = leak_scaler.transform(leak_residuals)  #scale using the leak-specific scaler
    leak_scores = leak_isolation_forest.predict(leak_scaled)  #classify each row

    leak_percent_flagged = (leak_scores == -1).sum() / len(leak_scores) * 100  #calculate percentage flagged

    if leak_percent_flagged > 10:
        st.error(f"⚠️ Fault Alert")
    else:
        st.success(f"✅ Operating normally")
    delta_color_choice = "inverse" if leak_percent_flagged > 10 else "normal"  #if over 10% of data is leak turn it red, if under green
    st.metric(label="Percent Flagged as Anomalous", value=f"{leak_percent_flagged:.1f}%", delta=f"{leak_percent_flagged - 5:.1f}% vs. healthy baseline", delta_color=delta_color_choice)
        