import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def rebin_product(df_prod, full_step=0.3, max_age=10.0, fine_step=0.1):
    """
    1) Interpolate each lat–lon series onto a 0.1 ka grid (0…max_age).
    2) Re-bin series into full_step bins
    """
    ages = df_prod['age'].to_numpy()
    cols = df_prod.columns.drop('age')
    # build fine grid
    fine_ages = np.arange(0, max_age + fine_step/2, fine_step)
    # reconstruct coarse-bin centers
    full_edges   = np.arange(-full_step/2, max_age + full_step, full_step)
    full_centres = full_edges[:-1] + full_step/2
    binned = {'age': full_centres}
    for col in cols:
        vals = df_prod[col].to_numpy()
        # interpolate onto fine grid
        valid = ~np.isnan(vals)
        if valid.sum() >= 2:
            f = interp1d(ages[valid], vals[valid],kind='linear',bounds_error=False,fill_value=np.nan)
            fine_vals = f(fine_ages)
        else:
            fine_vals = np.full_like(fine_ages, np.nan)
        # re-bin fine-series into the coarse bins
        full_means = _cascade_bin(fine_ages, fine_vals, full_step, max_age)
        binned[col] = full_means
    return pd.DataFrame(binned)

def _cascade_bin(ages, values, full_step, max_year, decimals=3):
    half_step   = full_step / 2
    half_edges  = np.arange(-half_step/2, max_year + half_step, half_step)
    half_centres = half_edges[:-1] + half_step/2
    full_edges  = np.arange(-full_step/2, max_year + full_step, full_step)
    full_centres = full_edges[:-1] + full_step/2
    half_centres = np.round(half_centres, decimals)
    full_centres = np.round(full_centres, decimals)
    half_lists = [[] for _ in range(len(half_centres))]
    for a, v in zip(ages, values):
        if np.isnan(a) or np.isnan(v):
            continue
        idx_half = np.digitize(a, half_edges) - 1
        if 0 <= idx_half < len(half_lists):
            half_lists[idx_half].append(v)
    half_means = [np.nanmean(lst) if lst else np.nan for lst in half_lists]
    full_lists = [[] for _ in range(len(full_centres))]
    for h_idx, h_mean in enumerate(half_means):
        if np.isnan(h_mean):
            continue
        centre = half_centres[h_idx]
        idx_full = np.digitize(centre, full_edges) - 1
        if 0 <= idx_full < len(full_lists):
            full_lists[idx_full].append(h_mean)
    full_means = [np.nanmean(lst) if lst else np.nan for lst in full_lists]
    return full_means
