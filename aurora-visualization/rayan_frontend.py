import streamlit as st
import plotly.graph_objects as go
import numpy as np
from predict import get_prediction_grid
from predict import model

variable_map = {
    "Variable 1": "2t",
    "Variable 2": "10u",
    "Variable 3": "10v",
    "Variable 4": "msl",
    "Variable 5": "t",
    "Variable 6": "u",
    "Variable 7": "v",
    "Variable 8": "q",
    "Variable 9": "z",
    # Add more if you're using more than 9
}

variable_labels = {
    "2t": "2m Temperature (2t)",
    "10u": "10m U Wind (10u)",
    "10v": "10m V Wind (10v)",
    "msl": "Mean Sea Level Pressure (msl)",
    "t": "Atmospheric Temperature (t)",
    "u": "Atmospheric U Wind (u)",
    "v": "Atmospheric V Wind (v)",
    "q": "Specific Humidity (q, g/kg)",
    "z": "Geopotential Height (z)"
}

variable_units = {
    "2t": "°C",
    "10u": "m/s",
    "10v": "m/s",
    "msl": "hPa",
    "t": "K",
    "u": "m/s",
    "v": "m/s",
    "q": "g/kg",
    "z": "m"
}

def create_interactive_map(data, lats, lons, title, unit, colorscale='RdYlBu_r', hover_text=None):
    """Create interactive geographic map with fixed color scaling"""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    if hover_text is None:
        hover_text = [f'{val:.2f}' for val in data.flatten()]

    fig = go.Figure(data=go.Scattermap(
        lat=lat_grid.flatten(),
        lon=lon_grid.flatten(),
        mode='markers',
        marker=dict(
            size=8,
            color=data.flatten(),
            colorscale=colorscale,
            showscale=True,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=unit),
            opacity=0.8
        ),
        text=hover_text,
        hovertemplate='<b>Latitude: %{lat:.2f}°</b><br>' +
                     '<b>Longitude: %{lon:.2f}°</b><br>' +
                     f'<b>{title}: %{{text}} {unit}</b><extra></extra>'
    ))

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=np.mean(lats), lon=np.mean(lons)),
            zoom=4.5
        ),
        title=title,
        height=450,
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    return fig

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aurora Design Mockup",
    layout="wide"
)

# --- LOCATION SELECTOR ---
locations = [
    "Malé (Capital)",
    "Haa Dhaalu Atoll",
    "Addu City (Gan)",
    "Lhaviyani Atoll",
    "Thaa Atoll",
    "Shaviyani Atoll",
    "Southern Indian Ocean",
    "Northern Maldives Sea",
    "Baa Atoll",
    "Southern Equatorial"
]

# --- VARIABLE NAMES ---
all_variables = [f"Variable {i+1}" for i in range(36)]
variables_by_tab = [all_variables[i*9:(i+1)*9] for i in range(4)]

# --- SIDEBAR/HEADER LAYOUT ---
st.markdown("""
    <style>
    .location-selector {float: right; margin-top: -60px;}
    .world-map {float: left;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ROW ---
col_map, col_selector = st.columns([2, 1])

# Coordinates for each location
location_coords = {
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

with col_map:
    st.markdown("#### World Map")
    # World map with markers for each location
    world_lats = [v[0] for v in location_coords.values()]
    world_lons = [v[1] for v in location_coords.values()]
    world_names = list(location_coords.keys())
    world_fig = go.Figure(go.Scattergeo(
        lat=world_lats,
        lon=world_lons,
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=world_names,
        textposition="top right"
    ))
    world_fig.update_layout(
        geo=dict(
            projection_type="natural earth",
            center=dict(lat=4.5, lon=73),
            projection_scale=15,  # try 10–20 for tighter view, tweak as needed
            showcountries=True,
            showland=True,
            landcolor="rgb(217, 217, 217)",
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=350
    )
    st.plotly_chart(world_fig, use_container_width=True)

with col_selector:
    st.markdown("#### Location Selector")
    selected_location = st.selectbox("Select a location:", locations, key="location_selector")

# --- SUBPAGE BUTTONS ---
st.markdown("---")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0

with col_b1:
    if st.button("Tab 1", key="tab1"):
        st.session_state["active_tab"] = 0
with col_b2:
    if st.button("Tab 2", key="tab2"):
        st.session_state["active_tab"] = 1
with col_b3:
    if st.button("Tab 3", key="tab3"):
        st.session_state["active_tab"] = 2
with col_b4:
    if st.button("Tab 4", key="tab4"):
        st.session_state["active_tab"] = 3

# --- 3x3 GRID OF MAPS ---
st.markdown("---")
current_variables = variables_by_tab[st.session_state["active_tab"]]
st.markdown(f"### 3x3 Grid of Maps for {selected_location} (Variables {st.session_state['active_tab']*9+1}-{st.session_state['active_tab']*9+9})")

# Get full latitude and longitude arrays from the model
lats = model.lat[:, 0].cpu().numpy()
lons = model.lon[0, :].cpu().numpy()

center_lat, center_lon = location_coords[selected_location]
lat_idx = int(np.abs(lats - center_lat).argmin())
lon_idx = int(np.abs(lons - center_lon).argmin())

for i in range(3):
    cols = st.columns(3)
    for j in range(3):
        var_idx = i * 3 + j
        with cols[j]:
            variable_label = current_variables[var_idx]

            if st.session_state["active_tab"] == 0:
                variable_key = variable_map[variable_label]
                full_data = get_prediction_grid(variable_key)

                step = 4  # 4 * 0.25° = 1° spacing
                lat_range = np.array([lat_idx - step, lat_idx, lat_idx + step])
                lon_range = np.array([lon_idx - step, lon_idx, lon_idx + step])

                # Clip to valid index range
                lat_range = lat_range[(lat_range >= 0) & (lat_range < len(lats))]
                lon_range = lon_range[(lon_range >= 0) & (lon_range < len(lons))]

                data = full_data[np.ix_(lat_range, lon_range)]
                lat_patch = lats[lat_range]
                lon_patch = lons[lon_range]

                readable_label = variable_labels.get(variable_key, variable_key)
                unit = variable_units.get(variable_key, "")
    
                # Clip + format for humidity
                if variable_key == "q":
                    data = np.clip(data, 0, None)
                    hover_text = [f'{val:.4f}' for val in data.flatten()]
                else:
                    hover_text = [f'{val:.2f}' for val in data.flatten()]
            else:
                # Only for placeholder tabs
                data = np.zeros((3, 3))
                lat_patch = np.linspace(center_lat - 1, center_lat + 1, 3)
                lon_patch = np.linspace(center_lon - 1, center_lon + 1, 3)
                hover_text = ["" for _ in range(9)]
                readable_label = variable_label
                unit = ""

            
            fig = create_interactive_map(
                data, lat_patch, lon_patch,
                readable_label, unit, "RdYlBu_r", hover_text=hover_text
            )
            
            st.plotly_chart(fig, use_container_width=True) 