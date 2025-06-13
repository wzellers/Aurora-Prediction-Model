import cdsapi
import os
import datetime

# Create necessary directories
os.makedirs("../data/atmos", exist_ok=True)
os.makedirs("../data/surf", exist_ok=True)
os.makedirs("../data/static", exist_ok=True)

# Get today's date and calculate up to most recently available data
today = datetime.date.today()
year = today.year
month = today.month
latest_available_day = today.day - 5 if today.day > 5 else 1
days = [f"{d:02d}" for d in range(1, latest_available_day + 1)]
month_str = f"{month:02d}"

# Initialize CDS API client
c = cdsapi.Client()

# Atmospheric variables
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"],
        "pressure_level": ["1000", "925", "850", "700", "500", "300", "200", "100"],
        "year": str(year),
        "month": [month_str],
        "day": days,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [10, 70, 0, 76],  # North, West, South, East
        "grid": [0.25, 0.25],
        "format": "netcdf"
    },
    "../data/atmos/atmos_vars.nc"
)

# Surface variables
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
        "year": str(year),
        "month": [month_str],
        "day": days,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [10, 70, 0, 76],
        "grid": [0.25, 0.25],
        "format": "netcdf"
    },
    "../data/surf/surf_vars.nc"
)

# Static variables (use a fixed historical date)
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "geopotential",
            "land_sea_mask",
            "soil_type"
        ],
        "year": "2023",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "area": [10, 70, 0, 76],
        "grid": [0.25, 0.25],
        "format": "netcdf",
    },
    "../data/static/static_vars.nc"
)