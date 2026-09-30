import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path
figure_dir = Path(__file__).resolve().parent
repo_root = figure_dir.parents[1]
sys.path.insert(0, str(repo_root / "Analysis" / "FA"))
from factor_analysis.eof import run_fa
from factor_analysis.fa_wrappers import reorder_factors, calc_BIC

###### Factor Analysis on Briner et al., 2016 dataset
# load data
def load_briner_subset(meta, subset_dir,age_key='age_name',val_key='value_name',fname_key='file_name'):
    meta = meta[(meta['variable'] == 'Temperature') &
                meta[fname_key].notna()]
    if meta[fname_key].duplicated().any():
        raise ValueError('Each Briner time series must have a unique file_name.')
    errors = pd.to_numeric(meta['Error_val'], errors='coerce')
    if not (np.isfinite(errors) & (errors > 0)).all():
        raise ValueError('Case C requires a positive, finite Error_val for every record.')
    records = []
    for _, row in meta.iterrows():
        fpath = subset_dir / (row[fname_key] + '.csv')
        df = pd.read_csv(fpath)
        df = df[df[row[age_key]]<8100]
        age   = pd.to_numeric(df[row[age_key]], errors='coerce').values/1000
        value = pd.to_numeric(df[row[val_key]], errors='coerce').values
        if "hjort" in row[fname_key].lower():
            value = value*-1
        keep = ~(np.isnan(age) | np.isnan(value))
        age, value = age[keep], value[keep]
        records.append({
            'id'   : row['Site ID'],
            'record_name': row[fname_key],
            'Error_val': float(row['Error_val']),
            'proxy': row.get('proxy', 'unknown'),
            'lat'  : row['lat'],
            'lon'  : row['lon'],
            'age'  : age,
            'value': value})
    return records


# bin data by age
def cascade_bin(ages, values, full_step, max_age):
    half_step   = full_step / 4
    half_edges  = np.arange(-half_step/4, max_age + half_step, half_step)
    half_cent   = half_edges[:-1] + half_step/4
    full_edges  = np.arange(-full_step/2, max_age + full_step, full_step)
    full_cent   = full_edges[:-1] + full_step/2
    # first stage
    half_lists = [[] for _ in range(len(half_cent))]
    for a, v in zip(ages, values):
        if np.isnan(a) or np.isnan(v): continue
        h = np.digitize(a, half_edges) - 1
        if 0 <= h < len(half_lists):
            half_lists[h].append(v)
    half_means = [np.nanmean(lst) if lst else np.nan for lst in half_lists]
    # second stage
    full_lists = [[] for _ in range(len(full_cent))]
    for h_idx, h_mean in enumerate(half_means):
        if np.isnan(h_mean): continue
        idx_full = np.digitize(half_cent[h_idx], full_edges) - 1
        if 0 <= idx_full < len(full_lists):
            full_lists[idx_full].append(h_mean)
    return np.array([np.nanmean(lst) if lst else np.nan
                     for lst in full_lists]), full_cent

# common age matrix
def build_matrix(records, max_age, full_step):
    _, age_vec = cascade_bin(np.array([0,]), np.array([0,]),full_step, max_age)                      
    n_age = len(age_vec)
    n_rec = len(records)
    X = np.full((n_age, n_rec), np.nan)
    meta_rows = []
    for j, rec in enumerate(records):
        binned, _ = cascade_bin(rec['age'], rec['value'],full_step, max_age)
        X[:, j] = binned
        meta_rows.append(dict(
            index=rec["record_name"],
            record_name=rec["record_name"],
            Error_val=rec["Error_val"],
            ID   =rec['id'],
            lat  =rec['lat'],
            lon  =rec['lon'],
            proxy=rec['proxy']))
    return pd.DataFrame(meta_rows), X, age_vec


# run factor analysis
max_age = 8
step=0.3
Q_MAX = 15

meta = pd.read_csv(figure_dir / "Briner2016Table1.csv", encoding="unicode_escape")
subsetdir = figure_dir / "Briner 2016 subset"
recs = load_briner_subset(meta, subsetdir)
meta_df, X, age_vec = build_matrix(recs, max_age=max_age, full_step=step)
df = pd.DataFrame(X, columns=meta_df["index"])
df.insert(0, "age", age_vec)

def varimax(W):
    rotation = np.eye(W.shape[1])
    previous = 0.0
    for _ in range(20):
        loadings = W @ rotation
        u, values, vh = np.linalg.svd(W.T @ (loadings**3 - (1.0 / len(W)) * loadings @ np.diag(np.sum(loadings**2, axis=0))))
        rotation = u @ vh
        current = np.sum(values)
        if previous != 0 and current - previous < 1e-6:
            break
        previous = current
    return W @ rotation, rotation

results = {"Proxy": {}}
for q in range(2, Q_MAX + 1):
    eof_vars, meta_df_out, _, X_proc = run_fa(meta_df.copy(), df, Q=q, psi_mode="C", ref_min=0, ref_max=8, tol=1e-6, update_mu=True)
    mu, W, sigma2, Z_mean = eof_vars
    W, rotation = varimax(W)
    Z_mean = Z_mean @ rotation
    factor_variance = np.sum(W**2, axis=0)
    mu, W, Z_mean, frac_exp, sigma2 = reorder_factors(factor_variance / factor_variance.sum(), (mu, W, sigma2, Z_mean))
    meta_df_out.loc[:, [f"Loading {k}" for k in range(q)]] = W
    empirical_variance = np.nansum(np.nanvar(X_proc, axis=0, ddof=1))
    ev = {"share_total_empirical_by_factor": np.sum(W**2, axis=0) / empirical_variance}
    bic = calc_BIC(X_proc, mu, W, sigma2, q, "C")
    results["Proxy"][f"n_z_{q}"] = {"mu": mu, "W": W, "sigma2": sigma2, "Z_mean": Z_mean, "frac_explained": frac_exp, "meta_df": meta_df_out, "ev": ev, "BIC": bic}
best_q = min(range(2, Q_MAX + 1), key=lambda q: results["Proxy"][f"n_z_{q}"]["BIC"])
print(f"Best Q: {best_q}")
output_dir = repo_root / "Data" / "Analysis Output"
output_dir.mkdir(parents=True, exist_ok=True)
with (output_dir / "briner_case_C.pkl").open("wb") as file:
    pickle.dump({"results": results, "data": df, "meta_df": meta_df, "age": age_vec, "best_q": best_q, "psi_mode": "C", "rotation": "varimax", "update_mu": True, "ref_min": 0, "ref_max": 8, "tol": 1e-6}, file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# Figure S1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

q          = 5
neg_flip   = [False, True, True, True, True]
lon0, lat0 = -30, 70
extent     = [-102, -20, 56, 80]
cmap_lims  = lambda v: (-np.nanmax(np.abs(v)), np.nanmax(np.abs(v)))
fig = plt.figure(figsize=(10, 15*q/5))
gs  = fig.add_gridspec(q, 2, width_ratios=[1.25, 1], hspace=0.2, wspace=0.05)

ts_axes  = []
map_axes = []
cbar_axes = []
for row, (pc_idx, flip) in enumerate(zip(range(q), neg_flip)):
    ax_map = fig.add_subplot(gs[row, 1],
                             projection=ccrs.LambertConformal(lon0, lat0))
    map_axes.append(ax_map)
    ylocs = [y for y in np.arange(50, 90, 5) if y != 66]
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.3, color='grey', alpha=.8,
                          linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl.ylocator = mticker.FixedLocator(ylocs)

    lon_arc = np.linspace(-180, 180, 720)
    ax_map.plot(lon_arc, np.full_like(lon_arc, 66),
                transform=ccrs.PlateCarree(),
                color='grey', linewidth=1.6, zorder=2)

    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='0.9', zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=.4)

    loadvec              = results['Proxy'][f'n_z_{q}']['W'][:, pc_idx].copy()
    if flip: loadvec *= -1
    vmin, vmax           = cmap_lims(loadvec)
    sc = ax_map.scatter(meta_df['lon'], meta_df['lat'],
                        c=loadvec, cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, s=150, edgecolor='k',
                        transform=ccrs.PlateCarree(), zorder=3,
                        alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax_map, shrink=0.7,
                        pad=0.08,  # slightly wider gap from map
                        location='right')
    cbar_axes.append(cbar.ax)
    cbar.set_label(f'W$_{pc_idx+1}$')

    ####### Time series #######
    ax_ts = fig.add_subplot(gs[row, 0])
    ts_axes.append(ax_ts)
    series = results['Proxy'][f'n_z_{q}']['Z_mean'][:, pc_idx].copy()
    if flip: series *= -1

    # var_pc = results['Proxy'][f'n_z_{q}']['frac_explained'][pc_idx].copy()
    var_pc = results['Proxy'][f'n_z_{q}']['ev']['share_total_empirical_by_factor'][pc_idx].copy()
    ax_ts.plot(age_vec, series, color='k', lw=1.8)
    ax_ts.invert_xaxis()
    # ax_ts.set_xlabel('Age (ka BP)')
    ax_ts.spines['right'].set_visible(False)

    ax_ts.set_ylabel(f'Z$_{pc_idx+1}$  ({var_pc*100:.1f} %)')
    ax_ts.grid(alpha=.7)
    ax_ts.spines['right'].set_visible(True)
    if row !=4:
        plt.setp(ax_ts.get_xticklabels(), visible=False)
    if row==4:
        ax_ts.set_xlabel("Age (cal Ka BP)")

fig.align_ylabels(ts_axes)

# Subplot Labels
import string
axes_for_labels = ts_axes + map_axes
labels = list(string.ascii_uppercase[:len(axes_for_labels)])
for ax, label in zip(axes_for_labels, labels):
    x_pos = 0.02
    y_pos = 0.95
    # Optional: nudge map labels inward slightly
    if ax in map_axes:
        x_pos = 0.02
        y_pos = 0.95
    ax.text(
        x_pos, y_pos,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.5,
            edgecolor="none",
            boxstyle="square,pad=0.25"),
             zorder=20)
    
plt.savefig(figure_dir / 'Fig. S1.png', dpi=300)