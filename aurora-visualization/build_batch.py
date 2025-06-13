from pathlib import Path
import torch
import xarray as xr
from aurora.batch import Batch, Metadata
from utils.padding import pad_to_patch_size

def build_batch(required_steps=1, patch_size=32):
    data_dir = Path("data")
    atmos_file = data_dir / "atmos" / "atmos_vars.nc"
    surf_file = data_dir / "surf" / "surf_vars.nc"
    static_file = data_dir / "static" / "static_vars.nc"

    # Load datasets
    atmos_ds = xr.open_dataset(atmos_file)
    surf_ds = xr.open_dataset(surf_file)
    static_ds = xr.open_dataset(static_file)

    t_ds = atmos_ds["t"]   # or "temperature" — use correct variable name
    u_ds = atmos_ds["u"]
    v_ds = atmos_ds["v"]
    q_ds = atmos_ds["q"]
    z_ds = atmos_ds["z"]

    # Normalize longitudes
    for ds in [t_ds, u_ds, v_ds, q_ds, z_ds, surf_ds, static_ds]:
        ds["longitude"] = ds["longitude"] % 360

    # Interpolate grids
    static_ds = static_ds.interp(latitude=surf_ds.latitude, longitude=surf_ds.longitude)
    z_ds_interp = z_ds.interp(latitude=t_ds.latitude, longitude=t_ds.longitude)

    # Crop region (Maldives)
    lat_range = (0, 10)
    lon_range = (70, 76)

    def crop(ds):
        lat_slice = slice(lat_range[1], lat_range[0]) if ds.latitude[0] > ds.latitude[-1] else slice(lat_range[0], lat_range[1])
        lon_slice = slice(lon_range[0], lon_range[1])
        return ds.sel(latitude=lat_slice, longitude=lon_slice)

    surf_ds = crop(surf_ds)
    static_ds = crop(static_ds)
    t_ds = crop(t_ds)
    u_ds = crop(u_ds)
    v_ds = crop(v_ds)
    q_ds = crop(q_ds)
    z_ds_interp = crop(z_ds_interp)

    if surf_ds.latitude.size == 0 or surf_ds.longitude.size == 0:
        raise ValueError("Cropping resulted in empty dataset. Check region bounds.")

    lat_vals = surf_ds.latitude.values
    lon_vals = surf_ds.longitude.values
    lat_grid = torch.tensor(lat_vals).unsqueeze(1).repeat(1, len(lon_vals))
    lon_grid = torch.tensor(lon_vals).unsqueeze(0).repeat(len(lat_vals), 1)

    max_available_steps = min(
        t_ds.shape[0],
        u_ds.shape[0],
        v_ds.shape[0],
        q_ds.shape[0],
        z_ds_interp.shape[0],
    )
    if max_available_steps < required_steps:
        raise ValueError("Not enough time steps in data.")
    max_steps = required_steps - 1

    surf_vars = {
        "2t": pad_to_patch_size(torch.from_numpy(surf_ds["t2m"].values[:max_steps+1][None]), patch_size),
        "10u": pad_to_patch_size(torch.from_numpy(surf_ds["u10"].values[:max_steps+1][None]), patch_size),
        "10v": pad_to_patch_size(torch.from_numpy(surf_ds["v10"].values[:max_steps+1][None]), patch_size),
        "msl": pad_to_patch_size(torch.from_numpy(surf_ds["msl"].values[:max_steps+1][None]), patch_size),
    }

    atmos_vars = {
        "t": pad_to_patch_size(torch.from_numpy(t_ds.values[:max_steps+1]), patch_size),
        "u": pad_to_patch_size(torch.from_numpy(u_ds.values[:max_steps+1]), patch_size),
        "v": pad_to_patch_size(torch.from_numpy(v_ds.values[:max_steps+1]), patch_size),
        "q": pad_to_patch_size(torch.from_numpy(q_ds.values[:max_steps+1]), patch_size),
        "z": pad_to_patch_size(torch.from_numpy(z_ds_interp.values[:max_steps+1, :1]), patch_size),
    }

    static_vars = {
        "z": pad_to_patch_size(torch.from_numpy(static_ds["z"].values[0])[None, None, :, :], patch_size),
        "slt": pad_to_patch_size(torch.from_numpy(static_ds["slt"].values[0])[None, None, :, :], patch_size),
        "lsm": pad_to_patch_size(torch.from_numpy(static_ds["lsm"].values[0])[None, None, :, :], patch_size),
    }

    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat_grid,
            lon=lon_grid,
            time=surf_ds.valid_time.values.astype("datetime64[s]").tolist()[:max_steps+1],
            atmos_levels=tuple(int(l) for l in t_ds.pressure_level.values),
        ),
    ), max_steps