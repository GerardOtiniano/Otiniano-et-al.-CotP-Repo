import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString
from .handle_age import _cascade_bin
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection
from shapely.geometry import box
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
dir_products = repo_root / "Data" / "Climate Reconstruction Products"

def load_prod(name, meta_df, df, max_age, rebinned_age_step=None):
    path = dir_products / f"{name}.nc"
    nc_o  = xr.open_dataset(path, engine='netcdf4')
    nc_o['year'] = nc_o['year'] / 1000
    nc_subset= nc_o.sel(season="JJA")
    nc_subset, proxy_bbox, unique_coords = subset_netcdf_to_proxy_bbox(nc_subset,meta_df,lon_col="lon",lat_col="lat",pad_lon=0,pad_lat=0)
    dfda = SUBSET_total_records(nc_subset, meta_df)
    da   = dfda.reset_index(drop=False).rename(columns={'year': 'age'})
    if rebinned_age_step is not None:
        da, original_time_step, used_bin = rebin_product(da, full_step=rebinned_age_step, max_age=max_age)
        print(f'{name} Original age bin: {original_time_step}, {used_bin}')
    else:
        print(name)
    return da, unique_coords, meta_df, df

#### NetCDF subsetting ####
def subset_netcdf_to_proxy_bbox(nc_subset, meta_df, lon_col="lon", lat_col="lat", pad_lon=0.0, pad_lat=0.0, verbose=True):
    """
    Remove NetCDF grid cells that do not overlap the proxy-network bounding box.
    A grid cell is retained if its rectangular footprint overlaps the proxy bbox:
        cell_lon_right >= bbox_lon_min
        cell_lon_left  <= bbox_lon_max
        cell_lat_top   >= bbox_lat_min
        cell_lat_bottom<= bbox_lat_max
    This keeps partially overlapping edge cells."""
    bbox = get_proxy_bounding_box(meta_df, lon_col=lon_col, lat_col=lat_col, pad_lon=pad_lon, pad_lat=pad_lat)
    lat_edges, lon_edges, lat_unique, lon_unique = create_netcdf_grid(nc_subset)
    lat_left = lat_edges[:-1]
    lat_right = lat_edges[1:]
    lon_left = lon_edges[:-1]
    lon_right = lon_edges[1:]
    keep_lat_sorted = (lat_right >= bbox["lat_min"]) & (lat_left <= bbox["lat_max"])
    keep_lon_sorted = (lon_right >= bbox["lon_min"]) & (lon_left <= bbox["lon_max"])
    kept_lats = lat_unique[keep_lat_sorted]
    kept_lons = lon_unique[keep_lon_sorted]
    if len(kept_lats) == 0 or len(kept_lons) == 0:
        raise ValueError(
            "Proxy bounding box does not overlap the NetCDF grid. "
            f"Proxy bbox: {bbox}")
    nc_out = nc_subset.sel(lat=kept_lats, lon=kept_lons)
    lat_edges_out, lon_edges_out, lat_unique_out, lon_unique_out = create_netcdf_grid(nc_out)
    return nc_out, bbox, [lat_unique_out, lon_unique_out, lat_edges_out, lon_edges_out]

def get_proxy_bounding_box(meta_df, lon_col="lon", lat_col="lat", pad_lon=0.0, pad_lat=0.0):
    """Create a lon/lat bounding box around all proxy sites"""
    proxy_lons = pd.to_numeric(meta_df[lon_col], errors="coerce").to_numpy()
    proxy_lats = pd.to_numeric(meta_df[lat_col], errors="coerce").to_numpy()
    valid = np.isfinite(proxy_lons) & np.isfinite(proxy_lats)
    if valid.sum() == 0:
        raise ValueError("No finite proxy lat/lon values found in meta_df.")
    lon_min = np.nanmin(proxy_lons[valid]) - pad_lon
    lon_max = np.nanmax(proxy_lons[valid]) + pad_lon
    lat_min = np.nanmin(proxy_lats[valid]) - pad_lat
    lat_max = np.nanmax(proxy_lats[valid]) + pad_lat
    lon_min = max(lon_min, -180)
    lon_max = min(lon_max, 180)
    lat_min = max(lat_min, -90)
    lat_max = min(lat_max, 90)
    return dict(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

def create_netcdf_grid(nc_subset):
    """Compute latitude and longitude cell edges from 1D NetCDF lat/lon centers"""
    lat_unique = np.sort(np.asarray(nc_subset["lat"].values, dtype=float))
    lon_unique = np.sort(np.asarray(nc_subset["lon"].values, dtype=float))
    if len(lat_unique) < 2:
        raise ValueError("Need at least two latitude values to infer grid-cell edges.")
    if len(lon_unique) < 2:
        raise ValueError("Need at least two longitude values to infer grid-cell edges.")
    lat_diff = np.diff(lat_unique) / 2
    lon_diff = np.diff(lon_unique) / 2
    lat_edges = np.concatenate([[lat_unique[0] - lat_diff[0]],lat_unique[:-1] + lat_diff,[lat_unique[-1] + lat_diff[-1]]])
    lon_edges = np.concatenate([[lon_unique[0] - lon_diff[0]],lon_unique[:-1] + lon_diff,[lon_unique[-1] + lon_diff[-1]]])
    lat_edges = np.clip(lat_edges, -90, 90)
    lon_edges = np.clip(lon_edges, -180, 180)
    return lat_edges, lon_edges, lat_unique, lon_unique

def SUBSET_total_records(nc_subset, meta_df):
    """Subset product to proxy time series"""
    data_var = nc_subset['temperature']
    df = data_var.to_dataframe(name='value').reset_index()
    df = df[~df['value'].isna()]
    required_columns = {'year', 'lat', 'lon'}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"The NetCDF subset must contain the following dimensions: {required_columns}")
    df['lat_lon'] = df.apply(lambda r: f"{r.lat:.6f}_{r.lon:.6f}", axis=1)
    grid_points = df[['lat', 'lon']].drop_duplicates().reset_index(drop=True) # unique grid points from netcdf
    nearest_keys = [] # find nearest grid point for each proxy time series
    unique_proxies = meta_df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    for _, proxy in unique_proxies.iterrows():
        proxy_lat = proxy['lat']
        proxy_lon = proxy['lon']
        distances = np.sqrt((grid_points['lat'] - proxy_lat)**2 + (grid_points['lon'] - proxy_lon)**2)
        idx_min = distances.idxmin()
        nearest_grid = grid_points.loc[idx_min]
        key = f"{nearest_grid['lat']:.6f}_{nearest_grid['lon']:.6f}"
        nearest_keys.append(key)
    nearest_keys = set(nearest_keys)
    df = df[df['lat_lon'].isin(nearest_keys)] # dataframe only includes rows corresponding to nearest grid poi t
    df_pivot = df.pivot(index='year', columns='lat_lon', values='value')
    return df_pivot

def rebin_product(df_prod, full_step=0.2, max_age=10.0, fine_step=0.1):
    """ Re-bin climate product into age bins with 'full step' spacing"""
    ages = df_prod['age'].to_numpy()
    cols = df_prod.columns.drop('age')
    # estimate original age step
    diffs = np.diff(ages)
    diffs = diffs[diffs > 0]  # ignore zero / negative diffs if any
    if len(diffs) > 0:
        original_step = float(np.median(diffs))
    else:
        original_step = np.nan
    use_direct_bin = np.isnan(original_step) or (original_step <= full_step + 1e-3)
    # check if ages need rebinning
    if not np.isnan(original_step) and abs(original_step - full_step) < 1e-3:
        return df_prod.copy(), original_step, "Age bin already matches time step"
    # bin structuers
    full_edges   = np.arange(-full_step/2, max_age + full_step, full_step)
    full_centres = full_edges[:-1] + full_step/2
    binned = {'age': full_centres}
    # fine grid
    if not use_direct_bin:
        fine_ages = np.arange(0, max_age + fine_step/2, fine_step)
    for col in cols:
        vals = df_prod[col].to_numpy()
        if use_direct_bin:
            full_means = _cascade_bin(ages, vals, full_step, max_age)
            used_bin = "Data not interpolated before age binning"
        else:
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                f = interp1d(ages[valid], vals[valid],kind='linear',bounds_error=False,fill_value=np.nan)
                fine_vals = f(fine_ages)
                used_bin = "Rebinned"
            else:
                fine_vals = np.full_like(fine_ages, np.nan)
                used_bin = "Tried to rebin, but not enough data to interpolate"
            full_means = _cascade_bin(fine_ages, fine_vals, full_step, max_age)
        binned[col] = full_means
    rebinned_df = pd.DataFrame(binned)
    return rebinned_df, original_step, used_bin
