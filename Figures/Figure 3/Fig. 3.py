import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator as op
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
plt.rcParams.update({'font.size': 12})
with open(repo_root / "Data" / "Analysis Output" / "fa_proxy_prod.pkl" , 'rb') as handle:
    results = pickle.load(handle)
    
def max_abs_scale(df):
    abs_v = np.abs(df)
    return df/np.max(abs_v)

def plot_quant_ensemble(ax, results, factor, sign_flip=1):
    qual = ["Bennike2002", "Bennike2008", "BennikeWagner2012", "BennikeWeidick2001", "Christiansen2002", "Klug2009", "Schmidt2011_Duck", "Schmidt2011_Hjort", "Wagner2000", "Wagner2008", "WagnerBennike2015"]
    df = pd.read_csv(repo_root / "Figures" / "Figure 3" / "proxy_ts.csv")
    meta_df = results['Proxy']['n_z_2']['meta_df'].copy()
    meta_df[f'Loading {factor}'] = results['Proxy']['n_z_2']['eof_vars']['W'][:, factor]
    meta_df
    age = results['Proxy']['n_z_2']['age']
    truths = [[0, op.ge, "red"], [0, op.le, "blue"]]
    num_ts = []
    for thresh, oprtr, coly in truths:
        centered_records = []
        for rec in meta_df['index']:
            if rec in qual: continue
            loading = meta_df.loc[meta_df["index"] == rec, f"Loading {factor}"].values[0] * sign_flip
            if oprtr(loading, thresh):
                temp = pd.DataFrame()
                temp['age'] = age
                temp['val'] = df.loc[:, rec].values
                temp = temp.dropna()
                temp[rec] = (temp['val'] - temp['val'].mean())/temp['val'].std() # Z scored
                centered_records.append(temp[['age', rec]].set_index('age'))
        centered_df = pd.concat(centered_records, axis=1).sort_index()
        median = centered_df.median(axis=1, skipna=True)
        std = centered_df.std(axis=1, skipna=True)
        lower_std = median - std
        upper_std = median + std
        ax.fill_between(centered_df.index, lower_std, upper_std, alpha=0.25, label='±1 SD', color=coly)
        ax.plot(centered_df.index, median, linewidth=3, c=coly, label='Median')
        num_ts.append(len(centered_records))
    ax.text(0.7, 0.2, f"(n = {num_ts[0]})", transform=ax.transAxes, c="red", ha="left")
    ax.text(0.7, 0.05, f"(n = {num_ts[1]})", transform=ax.transAxes, c="blue", ha="left")

def plot_quals(ax, results, factor, sign_flip=1):
    qual = ["Bennike2002", "Bennike2008", "BennikeWagner2012", "BennikeWeidick2001", "Christiansen2002", "Klug2009", "Schmidt2011_Duck", "Schmidt2011_Hjort", "Wagner2000", "Wagner2008", "WagnerBennike2015"]
    df = pd.read_csv(repo_root / "Figures" / "Figure 3" / "proxy_ts.csv")
    meta_df = results['Proxy']['n_z_2']['meta_df'].copy()
    meta_df[f'Loading {factor}'] = results['Proxy']['n_z_2']['eof_vars']['W'][:, factor]
    age = results["Proxy"]["n_z_2"]["age"]
    truths = [
        {"name": "positive", "threshold": 0, "operator": op.ge, "color": "red",  "text_y": 0.87},
        {"name": "negative", "threshold": 0, "operator": op.le, "color": "blue", "text_y": 0.70},]
    num_ts = {"positive": 0, "negative": 0}
    for truth in truths:
        group = truth["name"]
        thresh = truth["threshold"]
        oprtr = truth["operator"]
        coly = truth["color"]
        centered_records = []
        for rec in meta_df["index"]:
            if rec not in qual:
                continue
            loading = (
                meta_df.loc[meta_df["index"] == rec, f"Loading {factor}"].values[0]
                * sign_flip)
            if oprtr(loading, thresh):
                temp = pd.DataFrame()
                temp["age"] = age
                temp["val"] = df.loc[:, rec].values
                temp = temp.dropna()
                temp[rec] = temp["val"] - temp["val"].mean() # Centered
                temp[rec] = (temp["val"] - temp["val"].mean())/temp["val"].std() # Z scored
                centered_records.append(temp[["age", rec]].set_index("age"))
        num_ts[group] = len(centered_records)
        if len(centered_records) > 0:
            centered_df = pd.concat(centered_records, axis=1).sort_index()
            median = centered_df.median(axis=1, skipna=True)
            std = centered_df.std(axis=1, skipna=True)
            lower_std = median - std
            upper_std = median + std
            ax.fill_between(centered_df.index, lower_std,upper_std,alpha=0.25,label=f"{group} ±1 SD",color=coly)
            ax.plot(centered_df.index,median,linewidth=3,c=coly,label=f"{group} median")
    for truth in truths:
        group = truth["name"]
        count = num_ts[group]
        if count > 0:
            ax.text(0.7,truth["text_y"],f"(n = {count})",transform=ax.transAxes,c=truth["color"], ha="left",)
        else:
            ax.text(0.7,truth["text_y"],f"(n = 0)",transform=ax.transAxes,c=truth["color"], ha="left",)
    return num_ts

def extend_gris(gris, x_min):
    x = gris['age_mean'].to_numpy(dtype=float)
    y = gris['area'].to_numpy(dtype=float)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    y_scale = np.nanmax(y)
    y_scaled = y / y_scale
    def model(x, a, b, c):
        return a * np.exp(b * x) + c
    p0 = [y_scaled.max() - y_scaled.min(), 0.1, y_scaled.min()]
    params, covariance = curve_fit(
        model,
        x,
        y_scaled,
        p0=p0,
        maxfev=10000)
    a, b, c = params
    x_extended = np.linspace(x_min, x.min(), 5)
    y_fit_scaled = model(x_extended, a, b, c)
    y_fit = y_fit_scaled * y_scale
    return x_extended, y_fit


with open(repo_root / "Data" / "Analysis Output" / "fa_proxy_prod.pkl" , 'rb') as handle:
    data_ol = pickle.load(handle)

# Null distribution for signs
path = repo_root / "Data" / "Analysis Output" / "procrustes_results.pkl"
with open(path, 'rb') as handle:
    null = pickle.load(handle)

# Previous work
insolation = pd.read_csv(repo_root / "Figures" / "Figure 3"/ "Ind. Data" / "June21_Insolation.csv")
ice = pd.read_csv(repo_root / "Figures" / "Figure 3"/ "Ind. Data" / "Dalton_2023_Total_Optimal_LIS.csv")
gris = pd.read_csv(repo_root / "Figures" / "Figure 3"/ "Ind. Data" / "GRIS Area Extent Leger et al. 2025.csv")
sand = pd.read_csv(repo_root / "Figures" / "Figure 3"/ "Ind. Data" / "Perner_sand_2013.csv")

# Color dictionary
products = ["Truax", "Osman", "Badgeley_low", "Badgeley_pref", "Buizert", "TraCE21k"]
cb_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
colors = {prod: cb_palette[i] for i, prod in enumerate(products)}

# Figures
fig = plt.figure(figsize=(14, 10))
outer_gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[1, 1, 1.5], wspace=0.35)

# Subplot columns
left_gs = GridSpecFromSubplotSpec(nrows=5, ncols=1, subplot_spec=outer_gs[0, 0], height_ratios=[1.5, 1.5, 1.5, 1.5, 1.5], hspace=0.2)
center_gs = GridSpecFromSubplotSpec(nrows=5, ncols=1, subplot_spec=outer_gs[0, 1], height_ratios=[1.5, 1.5, 1.5, 1.5, 1.5], hspace=0.2)
right_gs = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=outer_gs[0, 2], height_ratios=[1, 1], hspace=0.1)

# Axes
z1_axis = fig.add_subplot(left_gs[0, 0])
proxy_axis1 = fig.add_subplot(left_gs[1, 0], sharex=z1_axis)
qual_axis1 = fig.add_subplot(left_gs[2, 0], sharex=z1_axis)
insolation_axis = fig.add_subplot(left_gs[3, 0], sharex=z1_axis)
ice_axis = fig.add_subplot(left_gs[4, 0], sharex=z1_axis)

z2_axis = fig.add_subplot(center_gs[0, 0])
proxy_axis2 = fig.add_subplot(center_gs[1, 0], sharex=z2_axis)
qual_axis2 = fig.add_subplot(center_gs[2, 0], sharex=z2_axis)
sand_axis = fig.add_subplot(center_gs[3, 0], sharex=z2_axis)

w1_axis = fig.add_subplot(right_gs[0, 0],  projection=ccrs.Orthographic(central_longitude=-40, central_latitude=75))
w2_axis = fig.add_subplot(right_gs[1, 0],  projection=ccrs.Orthographic(central_longitude=-40, central_latitude=75))

# Z1
age = data_ol['Proxy']['n_z_2']['age']
z_data = data_ol['Proxy']['n_z_2']['eof_vars']
z1_axis.plot(age, z_data['Z_mean'][:,0], c='k', linewidth=3)
z1_axis.set_xlim(10, 0)
z1_axis.set_ylabel(f"Temporal\nFactor Score\n({np.round(data_ol['Proxy']['n_z_2']['frac_explain'][0]*100):.0f}%)")
# Quant Z1
plot_quant_ensemble(proxy_axis1, results, 0, sign_flip=1)
# proxy_axis1.set_ylim(-5,3.5)
proxy_axis1.set_ylabel("Quant Summer\nTemperatures\n(°C)")

# Qual Z1
plot_quals(qual_axis1, results, 0, sign_flip=1)
qual_axis1.set_ylabel("Qual Summer\nTemperatures\n(°C)")

# Z2
z2_axis.plot(age, z_data['Z_mean'][:,1], c='k', linewidth=3)
z2_axis.set_ylabel(f"Temporal\nFactor Score\n({np.round(data_ol['Proxy']['n_z_2']['frac_explain'][1]*100):.0f}%)")
z2_axis.set_ylim(z1_axis.get_ylim())

# Quant Z2
plot_quant_ensemble(proxy_axis2, results, 1, sign_flip=1)#sign_flip=-1)
z2_axis.set_xlim(10, 0)
# proxy_axis2.set_ylim(-5,3.5)
proxy_axis2.set_ylim(proxy_axis1.get_ylim())

# Qual Z2
plot_quals(qual_axis2, results, 1, sign_flip=1)
qual_axis2.set_ylim(qual_axis1.get_ylim())

# Sand
sand_axis.plot(sand['Age'], sand['Sand %'], c='k', linewidth =2)
sand_axis.set_ylabel("% Sand, Disko Bugt")


# Insolation
insolation_axis.plot(insolation['Age'] * -1, insolation['Insolation (wm-2)'], c='k', linewidth=3)
insolation_axis.set_ylabel("Peak Boreal\nInsolation\n(W m$^{-2}$)", labelpad=10)

# LIS Ice Sheet
lis_col = '#069494'
lis_axis=ice_axis
lis_axis.plot(ice["Age ka"], ice["Area km2"]/1e6, c=lis_col, linewidth=4)
lis_axis.set_ylabel("LIS Area\n(10$^{5}$ km$^{2}$)", c=lis_col)
lis_axis.tick_params(axis="y")

# GrIS Decay
gris_col = '#BE5103'
gris_axis = lis_axis.twinx()
gris = gris[gris["age_mean"]<=10]
gris_axis.plot(gris['age_mean'], gris['area']/1e6, linewidth=4, linestyle='-', c=gris_col, alpha = 1)

# Extend and plot GRIS decay
gris_x, gris_y = extend_gris(gris, 5)
gris_ext = pd.DataFrame()
gris_ext['age'] = gris_x
gris_ext['area'] = gris_y/1e6
new_rows = pd.DataFrame({
    'age': [5, 3, 0],
    'area': [gris_ext['area'].min(), gris_ext['area'].min(), 1.75]})
gris_ext = pd.concat([gris_ext, new_rows], ignore_index=True)
gris_ext.sort_values('age', inplace=True)
gris_axis.plot(gris_ext['age'], gris_ext['area'], linewidth=4, linestyle='--', c=gris_col, alpha = 1)
gris_axis.set_ylabel("GIS Area\n(10$^{5}$ km$^{2}$)", rotation=270, labelpad=40, color=gris_col)

# Maps
for ax_map, W in zip([w1_axis, w2_axis], [0,1]):
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
    meta_df['loading'] = loading
    max_val = np.max(np.abs(loading))
    norm = plt.Normalize(vmin=-1*max_val, vmax=max_val)
    meta_df.loc[meta_df['archive']=="MarineSediment", 'marker'] = 'd'
    meta_df.loc[meta_df['archive']!="MarineSediment", 'marker'] = 'o'
    unique_markers = meta_df['marker'].dropna().unique()
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
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="4%", pad=0.08, axes_class=plt.Axes)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(f"W$_{W+1}$", rotation=270, labelpad=20)

# Map Legend
map_legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor='k',
        markeredgecolor='k',
        markersize=10,
        linestyle='None',
        label='Terrestrial'),
    Line2D(
        [0], [0],
        marker='d',
        color='none',
        markerfacecolor='k',
        markeredgecolor='k',
        markersize=10,
        linestyle='None',
        label='Marine')]

fig.legend(
    handles=map_legend_handles,
    loc='center',
    bbox_to_anchor=(0.74, 0.05),
    ncol=2,
    frameon=False,
    columnspacing=1.5,
    handletextpad=0.5)

# Remove x tick labels
for ax in [z1_axis, z2_axis, proxy_axis1, proxy_axis2, qual_axis1, qual_axis2, insolation_axis, sand_axis]: # qual_axis2,
    plt.setp(ax.get_xticklabels(), visible=False )
ice_axis.set_xlabel("Age (ka cal BP)")
sand_axis.set_xlabel("Age (ka cal BP)")
for ax in [z2_axis, proxy_axis2, qual_axis2]:
    plt.setp(ax.get_yticklabels(), visible=False )

# Align Axes
fig.align_ylabels([z1_axis, proxy_axis1, qual_axis1, lis_axis, insolation_axis])

# Text
z1_axis.text(0.5, 1.1, "Z$_{1}$", transform=z1_axis.transAxes, ha="center", fontweight="bold")
z2_axis.text(0.5, 1.1, "Z$_{2}$", transform=z2_axis.transAxes, ha="center", fontweight="bold")
gris_axis.text(2, 1.75, "?",  ha="center", c=gris_col, fontweight="bold", fontsize=24)

# Subplot labels
axes_for_labels = [
    z1_axis,
    proxy_axis1,
    qual_axis1,
    insolation_axis,
    ice_axis,
    z2_axis,
    proxy_axis2,
    qual_axis2,
    sand_axis,
    w1_axis,
    w2_axis
]

labels = list("ABCDEFGHIJK")

for ax, label in zip(axes_for_labels, labels):
    x_pos = 0.02
    y_pos = 0.94
    alphy = 0

    if ax in [w1_axis, w2_axis]:
        x_pos = 0.03
        y_pos = 0.97
        alphy = 1.0

    if ax in [insolation_axis]:
        x_pos = 0.02
        y_pos = 0.8
        alphy = 0

    if ax in [ice_axis]:
        x_pos = 0.06
        y_pos = 0.94
        alphy = 0

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
            alpha=alphy,
            edgecolor="none",
            boxstyle="square,pad=0.25"), zorder=20)

# Sand Trend
age_arr = sand["Age"]
wgc_arr = sand["Sand %"]
mask = (age_arr >= 0) & (age_arr <= 7) & np.isfinite(age_arr) & np.isfinite(wgc_arr)
x_fit = age_arr[mask]
y_fit = wgc_arr[mask]
m, b = np.polyfit(x_fit, y_fit, 1)
b=b+10
x0, x1 = 7, 0
y0 = m * x0 + b
y1 = m * x1 + b
yrange = np.nanmax(wgc_arr) - np.nanmin(wgc_arr)
offset = 0.12 * yrange
arrow_y0 = y0 + offset
arrow_y1 = y1 + offset
sand_axis.annotate(
    "",
    xy=(x1, arrow_y1),
    xytext=(x0, arrow_y0),
    arrowprops=dict(
        arrowstyle="->",
        lw=2,
        color="black"))
sand_axis.text(
    0.3,
    (arrow_y0 + arrow_y1) / 2 + 0.1 * yrange,
    "Decreasing WGC\nStrength",
    ha="right",
    va="bottom",
    fontsize=12)
sand_axis.set_ylim(0,40)
plt.savefig(repo_root / "Figures" / "Figure 3"/ "Fig. 3.png", dpi=300, bbox_inches="tight", pad_inches=0.051)
plt.show()