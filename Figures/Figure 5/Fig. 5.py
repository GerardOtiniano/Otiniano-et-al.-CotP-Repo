import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path

def max_abs_scale(df):
    abs_v = np.abs(df)
    return df/np.max(abs_v) 

width_ratios=[0.8,0.01,0.7,0.7]
wspace=0.2
hspace = 0.05
fx=18
fy=7
height_ratios = [0.4,0.4]
repo_root = Path(__file__).resolve().parents[2]
da_max_w = np.inf
path = repo_root / "Data" / "Analysis Output" / "fa_proxy_prod.pkl"
with open(path, 'rb') as handle:
    data_ol = pickle.load(handle)
path = repo_root / "Data" / "Analysis Output" / "procrustes_results.pkl"
with open(path, 'rb') as handle:
    null = pickle.load(handle)

# Color dictionary
products = ["Osman", "Badgeley_low", "Badgeley_pref", "Buizert", "TraCE21k", "Truax" ]
cb_palette = ["#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#0072B2",]
colors = {prod: cb_palette[i] for i, prod in enumerate(products)}

# Subplots
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(fx, fy))
outer_gs = GridSpec(
    nrows=2, ncols=4, figure=fig, width_ratios=width_ratios,
    height_ratios=[0.76, 0.24], wspace=wspace, hspace=0.28,
    bottom=0.07, top=0.90)
left_gs = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=outer_gs[0, 0], height_ratios=height_ratios, hspace=hspace)
center_gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=outer_gs[:, 2], height_ratios=[1])
right_gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=outer_gs[:, 3], height_ratios=[1])
z1_axis = fig.add_subplot(left_gs[0, 0])
z2_axis = fig.add_subplot(left_gs[1, 0], sharex=z1_axis)
w1_axis = fig.add_subplot(center_gs[0, 0],  projection=ccrs.Orthographic(central_longitude=-40, central_latitude=75))
w2_axis = fig.add_subplot(right_gs[0, 0],  projection=ccrs.Orthographic(central_longitude=-40, central_latitude=75))
legend_axis = fig.add_subplot(outer_gs[1, 0])
legend_axis.set_axis_off()

# Z1
z1_axis.plot(data_ol['Proxy']['n_z_2']['age'], data_ol['Proxy']['n_z_2']['eof_vars']['Z_mean'][:,0], c='k', linewidth=3, label="Proxies")
z1b = z1_axis.twinx()
for name, dat in data_ol['DA'].items():
    if name == "Truax":
        zo=2
    else: zo=0
    for product_data in null[0.6]['summary_da_full']:
        if product_data['product']==name:
            sign = product_data['signs'][0]
    z1b.plot(dat['n_z_2']['age'],dat['n_z_2']['eof_vars']['Z_mean'][:, 0]*sign, linewidth=3,
             label=name,color=colors.get(name, "gray"),zorder=zo)
z1_axis.set_xlim(10, 0)
z1_axis.set_ylabel('Z$_{1}$ Proxy')
z1b.set_ylabel('Z$_{1}$ Reconstructions', rotation=270, labelpad=20)

# Z2
z2_axis.plot(data_ol['Proxy']['n_z_2']['age'], data_ol['Proxy']['n_z_2']['eof_vars']['Z_mean'][:,1], c='k', linewidth=3, label="Proxies")
z2b = z2_axis.twinx()
for name, dat in data_ol['DA'].items():
    if name == "Truax":
        zo=2
    else: zo=0
    for product_data in null[0.6]['summary_da_full']:
        if product_data['product']==name:
            sign = product_data['signs'][1]
    z2 = dat['n_z_2']['eof_vars']['Z_mean'][:, 1]*sign
    z2b.plot(dat['n_z_2']['age'],z2,linewidth=3,label=name,color=colors.get(name, "gray"),zorder=zo)     
z2_axis.set_ylabel('Z$_{2}$ Proxy')
z2b.set_ylabel('Z$_{2}$ Reconstructions', rotation=270, labelpad=20)
z2_axis.set_xlabel("Age (cal ka BP)")

# Maps
## Proxies
w_norm = {}
for ax_map, W in zip([w1_axis, w2_axis], [0,1]):
    ax_map.set_anchor("N")
    ax_map.coastlines(zorder=3)
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4)
    ax_map.set_extent([-61, -24, 55, 82], crs=ccrs.PlateCarree())
    gl = ax_map.gridlines(draw_labels=False, linestyle='--')
    gl.right_labels = False
    gl.left_labels=True
    gl.bottom_labels = True
    meta_df = data_ol['Proxy']['n_z_2']['meta_df']
    loading = data_ol['Proxy']['n_z_2']['eof_vars']['W'][:,W]
    if W==1: loading = loading
    loading = max_abs_scale(loading)
    meta_df['loading'] = loading
    meta_df.loc[meta_df['archive']=="MarineSediment", 'marker'] = 'd'
    meta_df.loc[meta_df['archive']!="MarineSediment", 'marker'] = 'o'
    unique_markers = meta_df['marker'].dropna().unique()
    max_val = 1
    norm = plt.Normalize(vmin=max_val*-1, vmax=max_val)
    for marker in unique_markers:
        subset = meta_df[meta_df['marker'] == marker]
        y = subset['loading']
        scatter = ax_map.scatter(
            subset['lon'],
            subset['lat'],
            c=y,
            cmap='RdBu_r',
            norm=norm,
            s=100,
            marker=marker,
            edgecolor='k',
            transform=ccrs.PlateCarree(),
            label=f"Marker '{marker}'", zorder=5)
    w_norm[W] = norm
    ax_map.set_title(f"W$_{W+1}$ Proxy & Truax")
    if W == 1:
        cax = ax_map.inset_axes([1.025, 0, 0.04, 1])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(f"Normalized W", rotation=270, labelpad=25)

# reconstruction products
for ax_map, W in zip([w1_axis, w2_axis], [0,1]):
    unique_coords = data_ol['DA']['Truax']['n_z_2']['unique_coords']
    da_loading =  pd.DataFrame(data_ol['DA']['Truax']['n_z_2']['eof_vars']['W'])
    vals = max_abs_scale(da_loading[W])
    vals = vals *-1
    da_loading['current_loading'] = vals
    loading_matrix = np.full((len(unique_coords[0]), len(unique_coords[1])), np.nan)
    for idx, row in da_loading.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        if W==0: y=row['current_loading']*-1
        else: y=row['current_loading']
        loading = y
        lat_idx = np.abs(unique_coords[0] - lat).argmin()
        lon_idx = np.abs(unique_coords[1] - lon).argmin()
        loading_matrix[lat_idx, lon_idx] = loading
    mesh = ax_map.pcolormesh(
        unique_coords[3],
        unique_coords[2],
        loading_matrix,
        cmap='RdBu_r',
        norm = w_norm[W],
        transform=ccrs.PlateCarree(),
        alpha = 1,
        zorder=2)
    
# Legend
name_map = {
    "Truax": "Truax\n(This Study)",
    "Badgeley_low": "Badgeley et al. (2020)\nLow",
    "Badgeley_pref": "Badgeley et al. (2020)\nPreferred",
    "Buizert": "Buizert et al. (2018)",
    "Erb": "Erb et al. (2022)",
    "Osman": "Osman et al. (2021)",
    "TraCE21k": "TraCE21k\nLiu et al. (2009)",
    "Proxies": "Proxies"}
legend_order = [
    "Truax", "Badgeley_low", "Buizert", "Osman",
    "Badgeley_pref", "TraCE21k", "Proxies"]
legend_column_order = legend_order[::2] + legend_order[1::2]
legend_handles = [
    Line2D(
        [0], [0],
        color="k" if prod == "Proxies" else colors[prod],
        linewidth=3,
        label=name_map.get(prod, prod))
    for prod in legend_column_order]
legend_axis.legend(handles=legend_handles,loc="upper left",bbox_to_anchor=(0, 1),ncol=2, frameon=False,fontsize=10,borderaxespad=0, columnspacing=1.0,labelspacing=0.5,handlelength=1.5)

# Subplot labels
axes = [z1_axis, z2_axis, w1_axis, w2_axis]
labels = ["(A)", "(B)", "(C)", "(D)"]
for ax, label in zip(axes, labels):
    if label == "(A)":
        x = 0.05
    else:
        x = 0.02
    ax.text(x, 0.98, label,transform=ax.transAxes,ha="left",va="top",fontsize=16,fontweight="bold",zorder=20,bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
plt.setp(z1_axis.get_xticklabels(), visible=False)
fig.align_ylabels([z1_axis, z2_axis])
fig.align_ylabels([z1b, z2b])
plt.savefig(repo_root / "Figures" / "Figure 5"/ "Fig. 5.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()

