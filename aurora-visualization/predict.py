import torch
import numpy as np
from aurora.model import Aurora
from aurora.rollout import rollout
from build_batch import build_batch
from utils.padding import pad_to_patch_size
from aurora import AuroraSmallPretrained

model = AuroraSmallPretrained()
model.load_checkpoint()

# Load model and data batch
checkpoint_path = "checkpoints/aurora_0.25deg.pth"
model.eval()  # set to evaluation mode
print("Checkpoint loaded successfully.")
batch, max_steps = build_batch(required_steps=1)

# Ensure atmospheric variables are in correct shape (add batch dim)
for k in batch.atmos_vars:
    batch.atmos_vars[k] = batch.atmos_vars[k].unsqueeze(0)  # (1, T, C, H, W)

# Only use the first time step for rollout
for k in batch.atmos_vars:
    batch.atmos_vars[k] = batch.atmos_vars[k][:, :1, :, :, :]  # Keep all vertical levels (C)

# Extract raw lat/lon from batch and build sorted coordinate arrays
# Define sorted coordinate arrays (lat: decreasing, lon: increasing)
lat_vals = np.linspace(10, 0, 64)  # shape (64,), top to bottom
lon_vals = np.linspace(70, 76, 32)  # shape (32,), left to right

# Create meshgrid with correct indexing
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")  # shape (64, 32)

# Convert to torch and set correct dtype
lat_tensor = torch.tensor(lat_grid, dtype=torch.float32)
lon_tensor = torch.tensor(lon_grid, dtype=torch.float32)

# Pad to patch size (if needed by model)
lat_tensor = pad_to_patch_size(lat_tensor.unsqueeze(0).unsqueeze(0), 32).squeeze(0).squeeze(0)
lon_tensor = pad_to_patch_size(lon_tensor.unsqueeze(0).unsqueeze(0), 32).squeeze(0).squeeze(0)

# Assign to model
model.lat = lat_tensor
model.lon = lon_tensor

print("Lat dtype:", model.lat.dtype)
print("Lon dtype:", model.lon.dtype)
print("Lat tensor sample (center):", model.lat[16, 31:33])
print("Lon tensor sample (center):", model.lon[16, 31:33])
print("Lat row 0:", lat_tensor[0, :5])
print("Lat col 0:", lat_tensor[:5, 0])

# Define locations
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

print("Lat min/max:", model.lat.min().item(), model.lat.max().item())
print("Lat row 0:", model.lat[0, :5])
print("Lat col 0:", model.lat[:5, 0])

# Run rollout for predictions
steps = 1
preds_by_step = {}

with torch.no_grad():
    preds = list(rollout(model, batch, steps=steps + 1))
    for i, p in enumerate(preds[1:], 1):  # Skip initial input state
        preds_by_step[i] = {
            "surf": p.surf_vars,
            "atmos": p.atmos_vars,
        }

print("Atmospheric keys at step 1:", preds_by_step[1]["atmos"].keys())
print("q shape:", preds_by_step[1]["atmos"]["q"].shape)
print("q max value:", preds_by_step[1]["atmos"]["q"].max())
print("q min value:", preds_by_step[1]["atmos"]["q"].min())

# Helper: find closest index
def find_closest_idx(array, value):
    return int(np.abs(array - value).argmin())

# Print predictions
if preds_by_step:
    chosen_step = max(preds_by_step.keys())
    for name, (lat, lon) in locations.items():
        lat_idx = find_closest_idx(lat_vals, lat)
        lon_idx = find_closest_idx(lon_vals, lon)

        print(f"\n🔹 {name} (lat={lat}, lon={lon}) at step {chosen_step}:")
        print("Surface:")
        for var in ["2t", "10u", "10v", "msl"]:
            val = preds_by_step[chosen_step]["surf"][var][0, 0, lat_idx, lon_idx].item()
            print(f"  {var}: {val:.2f}")
        print("Atmospheric:")
        for var in ["t", "u", "v", "q", "z"]:
            if var == "q":
                val = preds_by_step[chosen_step]["atmos"][var][0, 0, :, lat_idx, lon_idx].mean().item()
                val *= 1000
            else:
                val = preds_by_step[chosen_step]["atmos"][var][0, 0, 0, lat_idx, lon_idx].item()
            print(f"  {var}: {val:.2f}")
else:
    print("No predictions were generated.")

def get_prediction_grid(variable_name):
    """
    Return the full 2D grid for a predicted variable at step 1.
    """
    if not preds_by_step:
        raise RuntimeError("No predictions available.")

    chosen_step = max(preds_by_step.keys())  # usually step 1
    surf_vars = preds_by_step[chosen_step]["surf"]
    atmos_vars = preds_by_step[chosen_step]["atmos"]

    if variable_name in surf_vars:
        grid = surf_vars[variable_name][0, 0]  # shape: (lat, lon)
    elif variable_name == "q":
        grid = atmos_vars["q"][0, 0].mean(dim=0) * 1000 # average across levels
    elif variable_name in atmos_vars:
        grid = atmos_vars[variable_name][0, 0, 0]
    else:
        raise ValueError(f"Variable {variable_name} not found in predictions.")

    return grid.cpu().numpy()