import cdsapi
import os

os.makedirs("../data/static", exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "geopotential",
            "land_sea_mask",
            "soil_type",
        ],
        "year": "2023",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "area": [10, 70, 0, 76],  # North, West, South, East
        "grid": [0.25, 0.25],
        "format": "netcdf",
    },
    "../data/static/static_vars.nc"
)