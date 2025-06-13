import cdsapi
import os

os.makedirs("../data/surf", exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
        "year": "2023",
        "month": ["06"],
        "day": [f"{d:02d}" for d in range(1, 31)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [10, 70, 0, 76],
        "grid": [0.25, 0.25],
        "format": "netcdf"
    },
    "../data/surf/surface_vars.nc"
)