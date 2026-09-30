import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from .fa_wrappers import fa_proxy_wrapper

def _bounded_scale(ref, test):
    """fit one amplitude multiplier using paired finite values"""
    valid = np.isfinite(ref) & np.isfinite(test)
    denominator = np.sum(test[valid] ** 2)
    if denominator <= 0 or not np.isfinite(denominator):
        return 1.0
    scale = np.sum(ref[valid] * test[valid]) / denominator
    return float(np.clip(scale, 0.2, 2.0)) # Both scale_Z_bounds and scale_W_bounds are fixed at (0.2, 2.0).

def align_ZW_factors(Z_ref, W_ref, Z_test, W_test):
    """center, sign-align and scale corresponding factors, then measure distance"""
    Z_ref, W_ref, Z_test, W_test = [np.asarray(a, dtype=float).copy() for a in (Z_ref, W_ref, Z_test, W_test)]
    # center Z
    Z_ref -= np.nanmean(Z_ref, axis=0)
    Z_test -= np.nanmean(Z_test, axis=0)
    signs = np.where(np.nansum(Z_ref * Z_test, axis=0) < 0, -1.0, 1.0)
    Z_test *= signs
    # couple sign fliups for Z and W
    W_test *= signs
    # center W
    # W_ref -= np.nanmean(W_ref, axis=0)
    # W_test -= np.nanmean(W_test, axis=0)
    # fit Z and W amplitudes separately
    Z_scales = np.array([_bounded_scale(Z_ref[:, k], Z_test[:, k]) for k in range(2)])
    W_scales = np.array([_bounded_scale(W_ref[:, k], W_test[:, k]) for k in range(2)])
    Z_aligned = Z_test * Z_scales
    W_aligned = W_test * W_scales
    # fivide residual norms by reference norms
    dZ = np.linalg.norm(np.nan_to_num(Z_ref - Z_aligned), axis=0) / (np.linalg.norm(np.nan_to_num(Z_ref), axis=0) + 1e-12)
    dW = np.linalg.norm(np.nan_to_num(W_ref - W_aligned), axis=0) / (np.linalg.norm(np.nan_to_num(W_ref), axis=0) + 1e-12)
    info = {"signs": signs, "Z_scales": Z_scales, "W_scales": W_scales,"Z_ref_for_dist": Z_ref, "W_ref_for_dist": W_ref}
    return Z_aligned, W_aligned, dZ, dW, info

def lonlat_to_xyz(lon, lat):
    lon = np.deg2rad((np.asarray(lon, dtype=float) + 180) % 360 - 180)
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    return np.column_stack([np.cos(lat) * np.cos(lon),np.cos(lat) * np.sin(lon), np.sin(lat)])

def build_aligned_W(results, da_name):
    """pair each proxy loading with its nearest product-grid loading"""
    proxy = results["Proxy"]["n_z_2"]
    W_proxy = np.asarray(proxy["eof_vars"]["W"], dtype=float)[:, :2]
    meta = proxy["meta_df"]
    W_da = results["DA"][da_name]["n_z_2"]["eof_vars"]["W"]
    proxy_xyz = lonlat_to_xyz(meta["lon"], meta["lat"])
    da_xyz = lonlat_to_xyz(W_da["longitude"], W_da["latitude"])
    valid_da = np.all(np.isfinite(da_xyz), axis=1)
    # match locations
    distance, local_idx = cKDTree(da_xyz[valid_da]).query(proxy_xyz)
    tile_idx = np.flatnonzero(valid_da)[local_idx]
    W_matched = W_da.iloc[tile_idx, :2].to_numpy(dtype=float)
    match_info = {
        "row_idx": np.arange(len(meta)),
        "proxy_id": meta["index"].astype(str).str.strip().to_numpy(dtype=object),
        "match_source": "nearest", "da_tile_idx": tile_idx,
        "match_dist_unit_sphere": distance,
        "proxy_lat": meta["lat"].to_numpy(dtype=float),
        "proxy_lon": meta["lon"].to_numpy(dtype=float),
        "da_lat": W_da["latitude"].to_numpy(dtype=float)[tile_idx],
        "da_lon": W_da["longitude"].to_numpy(dtype=float)[tile_idx],}
    return W_proxy, W_matched, match_info

def build_proxy_self_dist_distribution_ZW(df, results, subsample_frac, n_boot=100):
    """refit two-factor FA to proxy subsets and compare with the full network"""
    proxy = results["Proxy"]["n_z_2"]
    Z_ref = np.asarray(proxy["eof_vars"]["Z_mean"], dtype=float)[:, :2]
    W_ref = np.asarray(proxy["eof_vars"]["W"], dtype=float)[:, :2]
    meta = proxy["meta_df"].set_index("index", drop=False)
    proxy_cols = np.asarray(df.columns.drop("age"))
    rng = np.random.default_rng(42)
    n_keep = max(2, int(subsample_frac * len(proxy_cols)))
    dZ_list = np.empty((n_boot, 2))
    dW_list = np.empty((n_boot, 2))
    stored_FA = []
    for i in tqdm(range(n_boot), desc=f"Proxy subsets ({subsample_frac:.0%})"):
        boot_cols = rng.choice(proxy_cols, size=n_keep, replace=False)
        row_idx = meta.index.get_indexer(boot_cols)
        meta_boot = meta.loc[boot_cols].reset_index(drop=True).copy()
        df_boot = df[["age", *boot_cols]].copy()
        best_q, boot_results = fa_proxy_wrapper(meta_boot, df_boot, 2, update_mu=True)
        boot = boot_results["Proxy"]["n_z_2"]
        Z_boot = np.asarray(boot["eof_vars"]["Z_mean"], dtype=float)[:, :2]
        W_boot = np.asarray(boot["eof_vars"]["W"], dtype=float)[:, :2]
        _, _, dZ_list[i], dW_list[i], info = align_ZW_factors(
            Z_ref, W_ref[row_idx], Z_boot, W_boot)
        # Store the unaligned subset solutions
        stored_FA.append({
            "boot_cols": boot_cols.copy(), "meta_boot": meta_boot.copy(),
            "Z_boot": Z_boot.copy(), "W_boot": W_boot.copy(),
            "dZ_k": dZ_list[i].copy(), "dW_k": dW_list[i].copy(),
            "age": boot["age"], "info": info,
            "number_of_components": 2, "best_q": best_q, "target_q": 2})
    overall_list = 0.5 * dZ_list + 0.5 * dW_list # 50/50 split weight for Z and W distances
    return Z_ref, W_ref, dZ_list, dW_list, overall_list, stored_FA

def compute_self_thresholds(dZ_list, dW_list, overall_list):
    """use the 99th percentile of proxy self-distances for each factor."""
    return {"Z": np.quantile(dZ_list, 0.99, axis=0),"W": np.quantile(dW_list, 0.99, axis=0),"overall": np.quantile(overall_list, 0.99, axis=0)}

def compare_da_to_proxies(results, da_names, thresholds):
    """compare two-factor solution from each product with full proxy network"""
    proxy = results["Proxy"]["n_z_2"]
    Z_ref = np.asarray(proxy["eof_vars"]["Z_mean"], dtype=float)[:, :2]
    summary = []
    for name in da_names:
        product = results["DA"][name]["n_z_2"]
        Z_da = np.asarray(product["eof_vars"]["Z_mean"], dtype=float)[:, :2]
        W_proxy, W_da, match_info = build_aligned_W(results, name)
        Z_aligned, W_aligned, dZ, dW, info = align_ZW_factors(Z_ref, W_proxy, Z_da, W_da)
        overall = 0.5 * dZ + 0.5 * dW
        residual = info["W_ref_for_dist"] - W_aligned
        summary.append({
            "product": name, "q": 2,
            "dZ_k": dZ, "dW_k": dW, "overall_k": overall,
            "overall_mean": float(overall.mean()),
            "fit_Z_k": dZ <= thresholds["Z"],
            "fit_W_k": dW <= thresholds["W"],
            "fit_overall_k": overall <= thresholds["overall"],
            "signs": info["signs"],
            "Z_scales": info["Z_scales"], "W_scales": info["W_scales"],
            "Z_ref": info["Z_ref_for_dist"], "Z_ref_raw": Z_ref.copy(),
            "Z_aligned": Z_aligned,
            "W_proxy": info["W_ref_for_dist"], "W_proxy_raw": W_proxy,
            "W_aligned": W_aligned, "W_da_nearest": W_da,
            "W_residual": residual, "W_abs_residual": np.abs(residual),
            "W_sq_residual": residual ** 2, "match_info": match_info})
    return summary
