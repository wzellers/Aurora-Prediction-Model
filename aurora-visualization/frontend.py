import streamlit as st
from streamlit_folium import st_folium
import folium
import torch
import numpy as np
from aurora.model import Aurora
from aurora.rollout import rollout
from build_batch import build_batch
from utils.padding import pad_to_patch_size
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Build model and batch
@st.cache_resource
def load_model_and_data():
    model = Aurora()
    batch, _ = build_batch(required_steps=1)
    for k in batch.atmos_vars:
        batch.atmos_vars[k] = batch.atmos_vars[k].unsqueeze(0)[:, :1]

    lat_vals = np.linspace(10, 0, 64)
    lon_vals = np.linspace(70, 76, 32)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    lat_tensor = torch.tensor(lat_grid, dtype=torch.float32)
    lon_tensor = torch.tensor(lon_grid, dtype=torch.float32)

    model.lat = pad_to_patch_size(lat_tensor.unsqueeze(0).unsqueeze(0), 32).squeeze(0).squeeze(0)
    model.lon = pad_to_patch_size(lon_tensor.unsqueeze(0).unsqueeze(0), 32).squeeze(0).squeeze(0)
    model._initial_batch = batch
    return model, lat_vals, lon_vals

# Define locations and labels
locations = {
    "Malé (Capital)": (4.17, 73.51),
    "Haa Dhaalu Atoll": (6.00, 72.00),
    "Addu City (Gan)": (0.75, 73.15),
    "Lhaviyani Atoll": (5.25, 74.75),
    "Thaa Atoll": (2.00, 72.50),
    "Shaviyani Atoll": (7.25, 70.75),
    "Southern Indian Ocean": (3.25, 72.75),
    "Northern Maldives Sea": (9.00, 73.00),
    "Baa Atoll": (5.75, 70.25),
    "Southern Equatorial": (1.25, 75.25),
}

var_labels = {
    "2t": "2m Temperature (2t) [K]",
    "10u": "10m U Wind (10u) [m/s]",
    "10v": "10m V Wind (10v) [m/s]",
    "msl": "Mean Sea Level Pressure (msl) [Pa]",
    "t": "Atmospheric Temperature (t) [K]",
    "u": "Atmospheric U Wind (u) [m/s]",
    "v": "Atmospheric V Wind (v) [m/s]",
    "q": "Specific Humidity (q) [kg/kg]",
    "z": "Geopotential Height (z) [m²/s²]",
}

# Predict for a location
def generate_prediction(model, lat, lon, lat_vals, lon_vals):
    def find_closest_idx(array, value):
        return int(np.abs(array - value).argmin())

    lat_idx = find_closest_idx(lat_vals, lat)
    lon_idx = find_closest_idx(lon_vals, lon)

    with torch.no_grad():
        preds = list(rollout(model, model._initial_batch, steps=2))
        pred = preds[1]

    surf_out = {k: pred.surf_vars[k][0, 0, lat_idx, lon_idx].item() for k in pred.surf_vars}
    atmos_out = {k: pred.atmos_vars[k][0, 0, 0, lat_idx, lon_idx].item() for k in pred.atmos_vars}
    return {**surf_out, **atmos_out}

# Streamlit UI

# Title + subtitle
st.markdown("<h1 style='text-align: center;'>🌍 Aurora Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: gray;'>Click a location on the map and view predicted values</p>", unsafe_allow_html=True)

# Sidebar: variables and refresh
st.sidebar.markdown("### Variable(s) to Predict:")
selected_vars = st.sidebar.multiselect(
    "Choose any number of variables:",
    options=list(var_labels.keys()),
    format_func=lambda x: var_labels[x],
)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_resource.clear()
    st.session_state.predicted = False
    st.sidebar.success("✅ Model and data have been refreshed!")

# Load model/batch
model, lat_vals, lon_vals = load_model_and_data()

# Handle session state for clicked location and predictions
if "clicked_location" not in st.session_state:
    st.session_state.clicked_location = None
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = {}

# Prediction + confirm above map
if st.session_state.clicked_location:
    lat, lon = st.session_state.clicked_location["lat"], st.session_state.clicked_location["lng"]
    selected_name = next(
        (name for name, (lt, ln) in locations.items() if abs(lat - lt) < 0.2 and abs(lon - ln) < 0.2), None
    )

    if selected_name:
        st.markdown(f"### Selected Location: **{selected_name}**")
        if st.button(f"Predict for {selected_name}"):
            results = generate_prediction(model, *locations[selected_name], lat_vals, lon_vals)
            st.session_state.predicted = True
            st.session_state.prediction_results = results
            st.session_state.selected_name = selected_name

    if st.session_state.predicted:
        st.markdown(f"### Predictions for **{st.session_state.selected_name}**")
        for var in selected_vars:
            val = st.session_state.prediction_results.get(var)
            if val is not None:
                st.write(f"**{var_labels[var]}**: {val:.2f}")
    else:
        st.info("Click the button to generate predictions.")
else:
    st.info("Click a location on the map to begin.")

# Map at the bottom
st.markdown("---")
m = folium.Map(location=[4.5, 73], zoom_start=6)
for name, (lat, lon) in locations.items():
    color = "green" if "Malé" in name else "purple"
    folium.Marker(location=[lat, lon], popup=name, icon=folium.Icon(color=color)).add_to(m)

map_data = st_folium(m, height=400, width=700)

# Update clicked location
if map_data.get("last_clicked"):
    st.session_state.clicked_location = map_data["last_clicked"]