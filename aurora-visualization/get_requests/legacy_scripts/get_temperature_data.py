import cdsapi
import os

os.makedirs("../data/atmos", exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": ["temperature"],
        "pressure_level": ["1000", "925", "850", "700", "500", "300", "200", "100"],
        "year": "2023",
        "month": ["06"],
        "day": [f"{d:02d}" for d in range(1, 31)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [10, 70, 0, 76],
        "grid": [0.25, 0.25],
        "format": "netcdf"
    },
    "../data/atmos/temperature.nc"
)