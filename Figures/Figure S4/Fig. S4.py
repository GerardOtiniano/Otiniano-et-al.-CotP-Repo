import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]

def max_abs_scale(df):
    abs_v = np.abs(df)
    return df/np.max(abs_v) 
name_map = {"Truax": "Truax\n(This study)", "Badgeley_low": "Badgeley et al.\n(2020) Low", "Badgeley_pref": "Badgeley et al.\n(2020) Preferred", "Buizert": "Buizert et al.\n(2018)", "Osman": "Osman et al.\n(2021)", "TraCE21k": "TraCE21k\nLiu et al. (2009)"}

path = repo_root / "Data" / "Analysis Output" / "fa_proxy_prod.pkl"
with open(path, 'rb') as handle:
    data = pickle.load(handle)
    
repo_root = Path(__file__).resolve().parents[2]
path = repo_root / "Data" / "Analysis Output" / "procrustes_results.pkl"
with open(path, 'rb') as handle:
    null = pickle.load(handle)

plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(
    ncols=3,
    nrows=2,
    figsize=(12, 10),
    subplot_kw={
        "projection": ccrs.Orthographic(
            central_longitude=-40,
            central_latitude=75)})
axs = axs.ravel()
W = 0
products = ["Truax", "Osman", "Badgeley_low", "Badgeley_pref", "Buizert", "TraCE21k"]
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = "RdBu_r"
for i, ax in enumerate(axs):
    product = products[i]
    for prod in null[0.7]['summary_da_full']:
        if prod['product']==product:
            sign = prod['signs'][W]
    ax.coastlines(zorder=3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4)
    ax.set_extent([-62, -22, 54, 83], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False, linestyle='--')
    gl.right_labels = False
    gl.bottom_labels = False
    ev =  data['DA'][product]['n_z_2']['frac_explain'][W]
    ax.set_title(f"{name_map[product]}\n({int(np.round(ev*100))}%)", fontsize=14)
    unique_coords = data['DA'][product]['n_z_2']['unique_coords']
    da_loading = pd.DataFrame(data['DA'][product]['n_z_2']['eof_vars']['W'])
    vals = max_abs_scale(da_loading[W])
    vals = vals * sign
    da_loading['current_loading'] = vals
    loading_matrix = np.full((len(unique_coords[0]), len(unique_coords[1])), np.nan)
    for idx, row in da_loading.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        y = row['current_loading']
        lat_idx = np.abs(unique_coords[0] - lat).argmin()
        lon_idx = np.abs(unique_coords[1] - lon).argmin()
        loading_matrix[lat_idx, lon_idx] = y
    mesh = ax.pcolormesh(unique_coords[3],unique_coords[2],loading_matrix,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),alpha=1,zorder=2)   
fig.subplots_adjust(right=0.85, wspace=0.05, hspace=0.2)
cbar_ax = fig.add_axes([0.9, 0.18, 0.015, 0.64])
cbar = fig.colorbar(mesh, cax=cbar_ax)
cbar.set_label(f"Scaled W$_{W+1}$", rotation=270, labelpad=20)
plt.savefig(repo_root / "Figures" / "Figure S4" / "Fig. S4.png",dpi=300,bbox_inches="tight",pad_inches=0.05)
plt.show()