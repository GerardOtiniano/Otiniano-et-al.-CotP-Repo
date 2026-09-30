import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from adjustText import adjust_text
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy.signal import savgol_filter
repo_root = Path(__file__).resolve().parents[2]

# helper function
def plot_currents(ax, currents, color="grey", linewidth=3.5, window_length=21, polyorder=3, n_points=40, arrow_size=12, zorder=1):
    for _, row in currents.iterrows():
        lon, lat = row.geometry.xy
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        xy = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)
        x = xy[:, 0]
        y = xy[:, 1]
        distance = np.insert(np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)), 0, 0)
        distance /= distance[-1]
        distance_new = np.linspace(0, 1, n_points)
        x_interp = np.interp(distance_new, distance, x)
        y_interp = np.interp(distance_new, distance, y)
        x_smooth = savgol_filter(x_interp, window_length=window_length, polyorder=polyorder)
        y_smooth = savgol_filter(y_interp, window_length=window_length, polyorder=polyorder)
        x_smooth[0], y_smooth[0] = x_interp[0], y_interp[0]
        x_smooth[-1], y_smooth[-1] = x_interp[-1], y_interp[-1]
        ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth, zorder=zorder)
        dx = x_smooth[-1] - x_smooth[-5]
        dy = y_smooth[-1] - y_smooth[-5]
        angle = np.degrees(np.arctan2(dy, dx))
        marker = (3, 0, angle - 90)
        ax.scatter(x_smooth[-1], y_smooth[-1], marker=marker, s=arrow_size**2, color=color, edgecolor=color, zorder=zorder+0.1)

plt.rcParams.update({'font.size': 12})
df = pd.read_csv(repo_root / "Figures" / "Figure 2" / "proxy_ts.csv")
meta_df = pd.read_csv(repo_root / "Figures" / "Figure 2" / "proxy_meta.csv",encoding="unicode_escape")
recs = df.columns[1:]
meta_sub = meta_df[meta_df["index"].isin(recs)].copy()
meta_sub = meta_sub.sort_values("lat").reset_index(drop=True)
ordered_recs = meta_sub["index"].tolist()
proxy_types = meta_sub["proxy"].unique()
marker_dict = {proxy: marker for proxy, marker in zip(proxy_types,["o", "s", "^", "D", "v", "P", "X", "*"])}     
color_dict = {
    "MarineSediment": "#30638e",
    "LakeSediment": "#edae49",
    "Qualitative": "#EF3054"}

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(nrows=2,ncols=2,height_ratios=[0.85, 0.15], width_ratios=[1, 1], hspace=0.05,wspace=0.15)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.Orthographic(central_longitude=-40,central_latitude=75))
ax_leg = fig.add_subplot(gs[1, 1])
ax_leg.axis("off")
ages = df["age"]
for i, rec_name in enumerate(ordered_recs):
    temp = df[["age", rec_name]].dropna()
    proxy_type = meta_sub.loc[i, "proxy"]
    archive_type = meta_sub.loc[i, "archive"]
    ax1.scatter(temp["age"],np.full(len(temp), i),marker=marker_dict[proxy_type],color=color_dict[archive_type],edgecolor="k",s=80,alpha=0.8)
ax1.set_xlabel("Age (cal ka BP)")
ax1.set_ylabel("Site Number")
ax1.set_yticks(range(len(ordered_recs)))
ax1.set_yticklabels(range(len(ordered_recs)))
ax1.invert_xaxis()
ax2.coastlines()
ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4, zorder=1)
ax2.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

# currents
labels = [("BC",-69, 72), ("EGC", -12, 73),("WGC", -59, 71),("IC", -30, 62),("LC", -63, 61),]
currents = gpd.read_file(repo_root / "Figures" / "Figure 2" / "currents_shp" / "greenland_currents.shp")
plot_currents(ax2, currents, color="grey")
for label, lon, lat in labels:
    ax2.text(lon, lat, label, transform=ccrs.PlateCarree(), color="w", fontweight="bold", fontsize=14, ha="center", va="center", zorder=2)
    ax2.text(lon, lat, label, transform=ccrs.PlateCarree(), color="k", fontweight="bold", fontsize=12, ha="center", va="center", zorder=2)

ax2.text(-66, 73.6, "Baffin\nBay", ha='center', fontstyle='italic', transform=ccrs.PlateCarree())
ax2.text(-54, 58.5, "Labrador\nSea", ha='center', fontstyle='italic', transform=ccrs.PlateCarree())
ax2.text(-34, 59, "North\nAtlantic\nOcean", ha='center', fontstyle='italic', transform=ccrs.PlateCarree())

texts = []
x_points = []
y_points = []

label_offsets = {
    0: (-2, 0),
    1: (2, 0),
    2: (0, 1),
    3: (1, 1),
    4: (-1, -1),
    5: (-2, 0.8),
    6: (1, -1),
    7: (1, 0.5),
    8: (-1, 1),
    9: (-1, -1),
    10: (1, 1),
    11: (-1.2, -1),
    12: (1.5, -1),
    13: (0, -1.2),
    14: (-1, -1),
    15: (-0.5, 0.8),
    16: (-1, -1),
    17: (3, -0.3),
    18: (-1.5, 0.5),
    19: (1.2, 1),
    20: (0, -1),
    21: (1, 1),
    22: (-1, -1),
    23: (-4, 0),
    24: (4, 0),
    25: (-6, -0.2),
    26: (-6, 0.4),
    27: (-6, -0.2),
    28: (5, -0.5),
    29: (-5, 0.5),
    30: (4.5, 0.6),
    31: (-1, -1),
    32: (5, 0.5),
    33: (-5, 0.),
    34: (-7, -0.2),
    35: (6, -0.3),
    36: (-7, 0),
    37: (7, 0)}
for (i, row), lab_off_key in zip(meta_sub.iterrows(),label_offsets.keys()):
    proxy_type = row["proxy"]
    archive_type = row["archive"]
    ax2.scatter(row["lon"],row["lat"],marker=marker_dict[proxy_type],color=color_dict[archive_type],edgecolor="k",s=300,transform=ccrs.PlateCarree(),zorder=3)
    dx, dy = label_offsets[lab_off_key]
    txt = ax2.text(row["lon"]+dx,row["lat"]+dy,str(i),transform=ccrs.PlateCarree(),fontsize=12,fontweight="bold",ha="center",va="center",zorder=5,path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])        
    texts.append(txt)
    x_points.append(row["lon"])
    y_points.append(row["lat"])
gl = ax2.gridlines(draw_labels=True, linestyle="--",zorder=0)
gl.left_labels = False
gl.bottom_labels = False
proxy_handles = [Line2D([0], [0],marker=marker,color='black',linestyle='None',markersize=8,label=proxy)for proxy, marker in marker_dict.items()]
archive_handles = [Line2D([0], [0],marker='o',color='none',markerfacecolor=color,markeredgecolor='black',linestyle='None',markersize=8,label=archive)for archive, color in color_dict.items()]
leg1 = ax_leg.legend(handles=proxy_handles,title="Proxy Type",loc="center",bbox_to_anchor=(0.5, 0.50),ncol=3,frameon=False)
ax_leg.add_artist(leg1)
ax_leg.legend(handles=archive_handles,title="Archive Type",loc="center",bbox_to_anchor=(0.5, -0.4),ncol=len(archive_handles),frameon=False)
plt.savefig(repo_root / "Figures" / "Figure 2" / "Fig. 2.png", dpi=300,bbox_inches="tight")
plt.show()