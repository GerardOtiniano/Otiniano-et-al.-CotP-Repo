import numpy as np
import pandas as pd
from .import_prod import load_prod
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from pathlib import Path
import xarray as xr

def get_common_footprint(model_names, product_dir, var_name="temperature", require_same_grid=True, require_complete_time=False, map_extent=None):
    """Build a common product footprint for products"""
    product_footprints = []
    for name in model_names:
        path = Path(product_dir) / f"{name}.nc"
        fp = _load_product_footprint(path, var_name=var_name, require_complete_time=require_complete_time)
        fp["name"] = name
        product_footprints.append(fp)
        n_valid = int(fp["mask"].sum().values)
        n_total = int(fp["mask"].size)
        print(f"[{name}] valid native cells = {n_valid} / {n_total}")
    if require_same_grid:
        reference_lat = product_footprints[0]["lat"]
        reference_lon = product_footprints[0]["lon"]
        common_mask = None
        for fp in product_footprints:
            same_lat = np.array_equal(reference_lat, fp["lat"])
            same_lon = np.array_equal(reference_lon, fp["lon"])
            if common_mask is None:
                common_mask = fp["mask"]
            else:
                common_mask = common_mask & fp["mask"]
        n_common = int(common_mask.sum().values)
        n_total = int(common_mask.size)
        return common_mask
    product_geometries = []
    for fp in product_footprints:
        cells = _build_product_valid_cell_geometries(fp)
        product_geom = unary_union(cells)
        product_geometries.append(product_geom)
    common_geom = product_geometries[0]
    for product_geom in product_geometries[1:]:
        common_geom = common_geom.intersection(product_geom)
    footprint = {"mode": "native_gridcell_intersection", "products": product_footprints, "common_geom": common_geom}
    print(f"[common footprint, mixed-grid mode] using native grid-cell boundaries from {len(product_footprints)} products.")
    print(f"[common footprint, mixed-grid mode] common boundary bounds = {common_geom.bounds}")
    return footprint

def _load_product_footprint(path, var_name="temperature", require_complete_time=False):
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ds = ds.assign_coords(lon=_wrap_lon(ds.lon)).sortby("lon").sortby("lat")
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    lat_edges = gridcell_edges(lat, hard_min = -90, hard_max = 90)
    lon_edges = gridcell_edges(lon, hard_min = -180, hard_max = 180)
    mask = _get_spatial_mask(ds, var_name=var_name, require_complete_time=require_complete_time)
    mask_values = np.asarray(mask.values, dtype=bool)
    return {"path": Path(path), "name": Path(path).stem, "lat": lat, "lon": lon, "lat_edges": lat_edges, "lon_edges": lon_edges, "mask": mask, "mask_values": mask_values}

def _build_product_valid_cell_geometries(fp):
    lat = np.asarray(fp["lat"], dtype=float)
    lon = np.asarray(fp["lon"], dtype=float)
    lat_edges = np.asarray(fp["lat_edges"], dtype=float)
    lon_edges = np.asarray(fp["lon_edges"], dtype=float)
    mask_values = np.asarray(fp["mask"].values, dtype=bool)
    cells = []
    for i_lat in range(len(lat)):
        for i_lon in range(len(lon)):
            if not mask_values[i_lat, i_lon]:
                continue
            lat0 = float(lat_edges[i_lat])
            lat1 = float(lat_edges[i_lat + 1])
            lon0 = float(lon_edges[i_lon])
            lon1 = float(lon_edges[i_lon + 1])
            cell = box(lon0, lat0, lon1, lat1)
            cells.append(cell)
    return cells

def _get_spatial_mask(ds, var_name="temperature", require_complete_time=False):
    da = ds[var_name]
    non_spatial_dims = [dim for dim in da.dims if dim not in ["lat", "lon"]]
    if non_spatial_dims and require_complete_time:
        mask = da.notnull().all(dim=non_spatial_dims)
    elif non_spatial_dims:
        mask = da.notnull().any(dim=non_spatial_dims)
    else:
        mask = da.notnull()
    if "lat" in mask.dims and "lon" in mask.dims:
        mask = mask.transpose("lat", "lon")
    return mask.astype(bool)

def subset_proxies_to_common_footprint(meta_df, df, common_footprint, product_dir, id_col="index", lat_col="lat", lon_col="lon",age_col="age"):
    """Drop proxies outside the shared product footprint"""
    meta = meta_df.copy()
    meta[id_col] = meta[id_col].astype(str).str.strip()
    meta[lat_col] = meta[lat_col].astype(float)
    meta[lon_col] = _wrap_lon(meta[lon_col].astype(float))
    keep = []
    failed_products = []
    grid_summary = []
    grid_records = []
    mixed_grid_mode = isinstance(common_footprint, dict) and common_footprint.get("mode") == "native_gridcell_intersection"
    for _, row in meta.iterrows():
        proxy_id = str(row[id_col]).strip()
        proxy_lat = float(row[lat_col])
        proxy_lon = float(row[lon_col])
        if mixed_grid_mode:
            product_results = []
            product_failures = []
            product_grid_hits = []
            for fp in common_footprint["products"]:
                inside, i_lat, i_lon = _point_inside_product_footprint(proxy_lat, proxy_lon, fp)
                product_results.append(inside)
                product_grid_hits.append(f"{fp['name']}:({i_lat},{i_lon})")
                da_lat = np.nan
                da_lon = np.nan
                if np.isfinite(i_lat) and np.isfinite(i_lon):
                    da_lat = float(fp["lat"][int(i_lat)])
                    da_lon = float(fp["lon"][int(i_lon)])
                grid_records.append({
                    id_col: proxy_id,
                    "product": fp["name"],
                    "inside_product_footprint": bool(inside),
                    "proxy_lat": proxy_lat,
                    "proxy_lon": proxy_lon,
                    "da_i_lat": i_lat,
                    "da_i_lon": i_lon,
                    "da_lat": da_lat,
                    "da_lon": da_lon})
                if not inside:
                    product_failures.append(fp["name"])
            inside_footprint = all(product_results)
            failed_products.append(",".join(product_failures))
            grid_summary.append("; ".join(product_grid_hits))
        else:
            inside_footprint, i_lat, i_lon = _point_inside_same_grid_footprint(proxy_lat, proxy_lon, common_footprint)
            failed_products.append("" if inside_footprint else "common_mask")
            grid_summary.append(f"common_mask:({i_lat},{i_lon})")
            lat = np.asarray(common_footprint.lat.values, dtype=float)
            lon = np.asarray(common_footprint.lon.values, dtype=float)
            da_lat = np.nan
            da_lon = np.nan
            if np.isfinite(i_lat) and np.isfinite(i_lon):
                da_lat = float(lat[int(i_lat)])
                da_lon = float(lon[int(i_lon)])
            grid_records.append({
                id_col: proxy_id,
                "product": "common_mask",
                "inside_product_footprint": bool(inside_footprint),
                "proxy_lat": proxy_lat,
                "proxy_lon": proxy_lon,
                "da_i_lat": i_lat,
                "da_i_lon": i_lon,
                "da_lat": da_lat,
                "da_lon": da_lon})
        keep.append(inside_footprint)
    meta["_inside_common_footprint"] = keep
    meta["_failed_products"] = failed_products
    meta["_grid_hits"] = grid_summary
    meta_sub = meta.loc[meta["_inside_common_footprint"]].copy()
    dropped = meta.loc[~meta["_inside_common_footprint"]].copy()
    record_ids = [str(c).strip() for c in df.columns if c != age_col]
    ids_in_df = set(record_ids)
    wanted_ids = set(meta_sub[id_col].astype(str))
    matched_ids = sorted(ids_in_df & wanted_ids)
    missing_ids = sorted(wanted_ids - ids_in_df)
    df_sub = df.loc[:, [age_col] + matched_ids].copy()
    meta_sub = meta_sub.set_index(id_col).reindex(matched_ids).reset_index().rename(columns={"index": id_col})
    drop_cols = ["_inside_common_footprint", "_failed_products", "_grid_hits"]
    meta_sub = meta_sub.drop(columns=[c for c in drop_cols if c in meta_sub.columns])
    dropped_clean = dropped.drop(columns=[c for c in ["_inside_common_footprint", "_grid_hits"] if c in dropped.columns])
    return meta_sub, df_sub, dropped_clean, meta

def gridcell_edges(coord, hard_min=None, hard_max=None):
    """
    Get lat/lon edges of grid cells - assumes coordinates are at center of gridcell
    """
    coord = np.asarray(coord, dtype=float)
    if coord.size == 1:
        delta = 0.5
        edges = np.array([coord[0] - delta, coord[0] + delta])
    else:
        mids = (coord[:-1] + coord[1:]) / 2
        first = coord[0] - (mids[0] - coord[0])
        last = coord[-1] + (coord[-1] - mids[-1])
        edges = np.concatenate([[first], mids, [last]])
    if hard_min is not None:
        edges = np.maximum(edges, hard_min)
    if hard_max is not None:
        edges = np.minimum(edges, hard_max)
    return edges

def _wrap_lon(lon):
    return ((lon + 180) % 360) - 180

def _point_inside_product_footprint(proxy_lat, proxy_lon, fp):
    i_lat = np.searchsorted(fp["lat_edges"], proxy_lat, side="right") - 1
    i_lon = np.searchsorted(fp["lon_edges"], proxy_lon, side="right") - 1
    inside_grid = 0 <= i_lat < len(fp["lat"]) and 0 <= i_lon < len(fp["lon"])
    if not inside_grid:
        return False, np.nan, np.nan
    inside_valid_cell = bool(fp["mask_values"][i_lat, i_lon])
    return inside_valid_cell, i_lat, i_lon
