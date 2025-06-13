# 🌍 Aurora Air Prediction

A lightweight web app using Microsoft's Aurora model to predict atmospheric and surface-level variables for locations across the Maldives region. Built with Streamlit and Folium, this app provides real-time visual predictions based on reanalysis data.

---

## Features

- Interactive map with 10 clickable ocean/coastal locations  
- Real-time predictions of atmospheric & surface variables  
- Streamlit interface with selectable variables  
- Refresh button to reload model & data dynamically  
- Clean, user-friendly UI with centered content and labeled units

---

## Project Structure

```
aurora-visualization/
│
├── aurora/               # Core model and rollout logic
├── data/                 # Downloaded .nc files (atmos, surf, static)
├── utils/                # Padding, preprocessing utilities
├── build_batch.py        # Builds model input batch from data
├── frontend.py           # Streamlit web interface
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md             # You’re here!
```

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Before running predictions, ensure the following NetCDF files are available in the `data/` directory:

- `data/atmos/atmos_vars.nc` — All 5 atmospheric variables in one file  
- `data/surf/surf_vars.nc` — Surface-level variables (2t, 10u, 10v, msl)  
- `data/static/static_vars.nc` — Static variables (e.g., geopotential, land/sea mask)

Use the provided data scripts or CDS API requests to generate these files. All files are aligned on a 0.25° × 0.25° grid over the Maldives region.

---

## Running the App

To launch the Streamlit web app, run the following command from the project root:

```bash
streamlit run frontend.py
```

This will open the Aurora Air Prediction interface in your browser.

---

## Developer Notes

- Model predictions are made with `torch.no_grad()` and assume a pre-loaded `Aurora()` model  
- Surface and atmospheric tensors are padded and reshaped to match model expectations (e.g., 64×32 spatial grid)  
- `streamlit_folium` is used to provide map interactivity  

---