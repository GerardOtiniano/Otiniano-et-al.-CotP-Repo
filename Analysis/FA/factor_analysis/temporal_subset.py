import numpy as np
import pandas as pd
import cartopy.crs as ccrs

def subset_prod_temporally(da, prod_name, unique_coords, results,proxy_key="n_z_2",age_col='age',decimals_age=6):
    """Mask DA tiles in time so that each tile only keeps ages where its
    overlapping proxy records actually have data."""
    # Get proxy data from results
    proxy_block = results["Proxy"][proxy_key]
    meta_df = proxy_block["meta_df"].reset_index(drop=True)
    proxy_age = np.asarray(proxy_block["age"], float)
    X_proxy = np.asarray(proxy_block["X"], float)   # (N_age, N_proxy)
    if X_proxy.shape[1] != len(meta_df):
        raise ValueError(
            f"X_proxy has {X_proxy.shape[1]} columns but meta_df has {len(meta_df)} rows. "
            "These should match (one column per proxy record).")
    # Get prod tile coordiantes from column names
    tile_cols = [c for c in da.columns if c != age_col]
    lat_list, lon_list = [], []
    for col in tile_cols:
        lat_str, lon_str = col.split("_")
        lat_list.append(float(lat_str))
        lon_list.append(float(lon_str))
    grid_points = pd.DataFrame({"lat": lat_list, "lon": lon_list, "key": tile_cols})
    grid_lat = grid_points["lat"].values
    grid_lon = grid_points["lon"].values
    # map recon. prod to proxy col (closest)
    tile_to_proxy_cols: dict[str, list[int]] = {k: [] for k in tile_cols}
    proxy_lats = meta_df["lat"].values
    proxy_lons = meta_df["lon"].values
    for j in range(len(meta_df)):
        plat = proxy_lats[j]
        plon = proxy_lons[j]
        dlat = grid_lat - plat
        dlon = grid_lon - plon
        distances = np.sqrt(dlat * dlat + dlon * dlon)
        idx_min = int(np.argmin(distances))
        grid_key = grid_points["key"].iloc[idx_min]
        tile_to_proxy_cols[grid_key].append(j)
        grid_proxy_count = None
    if unique_coords is not None:
        lat_unique = np.asarray(unique_coords[0])
        lon_unique = np.asarray(unique_coords[1])
        lat_edges  = np.asarray(unique_coords[2])
        lon_edges  = np.asarray(unique_coords[3])
        grid_proxy_count = np.zeros((len(lat_unique), len(lon_unique))) * np.nan
        for tile_key, proxy_idx_list in tile_to_proxy_cols.items():
            row = grid_points.loc[grid_points["key"] == tile_key].iloc[0]
            tlat = row["lat"]
            tlon = row["lon"]
            lat_idx = np.abs(lat_unique - tlat).argmin()
            lon_idx = np.abs(lon_unique - tlon).argmin()
            count = len(proxy_idx_list) # number of overlap proxies for a tile
            grid_proxy_count[lat_idx, lon_idx] = count
    da_ages = da[age_col].values.astype(float) # map recon prod to proxy age
    age_map = {}
    for i, a in enumerate(proxy_age):
        age_map[round(float(a), decimals_age)] = i
    # determine which proxy row matches each recon. prod row
    age_idx_in_proxy = np.full(len(da_ages), -1, dtype=int)
    for i, a in enumerate(da_ages):
        key = round(float(a), decimals_age)
        if key in age_map:
            age_idx_in_proxy[i] = age_map[key]
    # create masked copy of recon. prod and subset temporally
    da_masked = da.copy()
    for tile in tile_cols:
        proxy_cols = tile_to_proxy_cols.get(tile, [])
        if len(proxy_cols) == 0:
            continue
        col_values = da_masked[tile].to_numpy(copy=True)
        for i_row in range(len(da_ages)):
            ip = age_idx_in_proxy[i_row]
            if ip < 0:
                col_values[i_row] = np.nan
                continue
            vals_here = X_proxy[ip, proxy_cols]
            if np.all(np.isnan(vals_here)):
                col_values[i_row] = np.nan
        da_masked[tile] = col_values
    return da_masked
