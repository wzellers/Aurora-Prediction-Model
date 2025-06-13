import cdsapi
import os
import sys
from datetime import datetime, timedelta

# Get today's date, but shift back to last available ERA5 date
max_available_date = datetime.today() - timedelta(days=5)
max_date_str = max_available_date.strftime("%Y-%m-%d")

# Parse date from CLI or use most recent valid
if len(sys.argv) > 1:
    requested_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    if requested_date > max_available_date:
        print(f"Requested date is too recent. Using most recent available date: {max_date_str}")
        requested_date = max_available_date
else:
    requested_date = max_available_date

year = requested_date.strftime("%Y")
month = requested_date.strftime("%m")
day = requested_date.strftime("%d")

# Create directory if it doesn't exist
os.makedirs("data/atmos", exist_ok=True)

c = cdsapi.Client()

output_file = f"data/atmos/geopotential_{requested_date.strftime('%Y-%m-%d')}.nc"

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": ["geopotential"],
        "pressure_level": ["1000", "925", "850", "700", "500", "300", "200", "100"],
        "year": year,
        "month": month,
        "day": day,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [10, 70, 0, 76],
        "grid": [0.25, 0.25],
        "format": "netcdf"
    },
    output_file
)

print("Success!")