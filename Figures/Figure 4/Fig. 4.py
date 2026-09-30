import numpy as np
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from matplotlib.patches import Rectangle, ConnectionPatch

# Functions
plt.rcParams.update({'font.size': 14})
def density_threshold(x,y, densities = [0.34, 0.68, 0.95]):
    """Create KDE densities for null distribution of a given proxy retention amount"""
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    # Map KDE to grid
    x_pad = (x.max() - x.min())
    y_pad = (y.max() - y.min())
    xgrid, ygrid = np.meshgrid(
        np.linspace(x.min() - x_pad,
                    x.max() + x_pad, 300),
        np.linspace(y.min() - y_pad,
                    y.max() + y_pad, 300),)
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    density = kde(positions).reshape(xgrid.shape)
    # Convert density to cumulative probability
    dx = xgrid[0, 1] - xgrid[0, 0]
    dy = ygrid[1, 0] - ygrid[0, 0]
    cell_area = dx * dy
    density_flat = density.ravel()
    idx = np.argsort(density_flat)[::-1]
    density_sorted = density_flat[idx]
    cumulative = np.cumsum(density_sorted * cell_area)
    cumulative /= cumulative[-1]
    # Highest-density thresholds
    levels = {}
    for dense in densities:
        levels[dense] = density_sorted[np.searchsorted(cumulative, dense)]
    return xgrid, ygrid, density, levels

repo_root = Path(__file__).resolve().parents[2]
path = repo_root / "Data" / "Analysis Output" / "procrustes_results.pkl"
with open(path, 'rb') as handle:
    data_full = pickle.load(handle)


# Figure
percentile = 0.7
percentiles = [0.34, 0.68, 0.95]
data = data_full[percentile]
xlim = [-0.1,1.2]
ylim=[-0.1,1.25]
    
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(
    nrows=1,
    ncols=3,
    width_ratios=[1, 1, 0.55],
    left=0.08,
    right=0.98,
    bottom=0.28,
    top=0.9,
    wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")

products = ["Truax", "Osman", "Badgeley_low", "Badgeley_pref", "Buizert", "TraCE21k"]
cb_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
colors = {prod: cb_palette[i] for i, prod in enumerate(products)}

z_xgrid, z_ygrid, z_density, z_density_levels = density_threshold(
    data['dZ_list'][:, 0],
    data['dZ_list'][:, 1],
    percentiles)

for level, color, z_pos in zip(
        z_density_levels.values(),
        ["#495057", "#adb5bd", "#dee2e6"],
        [3, 2, 1]):

    ax1.contourf(
        z_xgrid, z_ygrid, z_density,
        levels=[level, z_density.max()],
        colors=[color],
        zorder=z_pos)

    ax1.contour(
        z_xgrid, z_ygrid, z_density,
        levels=[level],
        colors=["k"],
        zorder=4)

w_xgrid, w_ygrid, w_density, w_density_levels = density_threshold(
    data['dW_list'][:, 0],
    data['dW_list'][:, 1],
    percentiles)

for level, color, z_pos in zip(
        w_density_levels.values(),
        ["#982d40", "#bd767e", "#dcbcc2"],
        [3, 2, 1]):

    ax2.contourf(
        w_xgrid, w_ygrid, w_density,
        levels=[level, w_density.max()],
        colors=[color],
        zorder=z_pos)

    ax2.contour(
        w_xgrid, w_ygrid, w_density,
        levels=[level],
        colors=["k"],
        zorder=4)

for ax, x, y in zip(
        [ax1, ax2],
        ["dZ_k", "dW_k"],
        ["dZ_k", "dW_k"]):

    for r in data['summary_da_full']:

        prod = r['product']
        col = colors[prod]

        x_val = r[x][0]
        y_val = r[y][1]

        ax.scatter(
            x_val, y_val,
            s=280,
            c="w",
            zorder=8,
            alpha=0.5)

        ax.scatter(
            x_val, y_val,
            s=200,
            ec="k",
            c=col,
            zorder=8,
            alpha=1)

ax1.set_xlabel("Z$_{1}$ Distance")
ax1.set_ylabel("Z$_{2}$ Distance")

ax2.set_xlabel("W$_{1}$ Distance")
ax2.set_ylabel("W$_{2}$ Distance")

for ax in [ax1, ax2]:

    ax.grid(True, alpha=0.5, zorder=0)

    ax.set_xlim(xlim)#-0.3, 1.2)
    ax.set_ylim(ylim)#-0.4, 1.2)

    ax.axvline(
        0, c='k',
        linewidth=2,
        zorder=3,
        linestyle="--",
        alpha=0.75)

    ax.axhline(
        0, c='k',
        linewidth=2,
        zorder=3,
        linestyle="--",
        alpha=0.75)
    
    ax.axvline(
        0, c='k',
        linewidth=2,
        linestyle="-",
        alpha=1,
        zorder=0)

    ax.axhline(
        0, c='k',
        linewidth=2,
        linestyle="-",
        alpha=1,
        zorder=0)

zoom_xlim = (0.8, 0.85)
zoom_ylim = (1.0, 1.1)

ax_inset = ax3.inset_axes([-0.2, 0.52, 1.0, 0.45])

for level, color, z_pos in zip(
        w_density_levels.values(),
        ["#982d40", "#bd767e", "#dcbcc2"],
        [3, 2, 1]):

    ax_inset.contourf(
        w_xgrid, w_ygrid, w_density,
        levels=[level, w_density.max()],
        colors=[color],
        zorder=z_pos)

    ax_inset.contour(
        w_xgrid, w_ygrid, w_density,
        levels=[level],
        colors=["k"],
        zorder=4)

for r in data['summary_da_full']:

    prod = r['product']
    col = colors[prod]

    x_val = r["dW_k"][0]
    y_val = r["dW_k"][1]

    ax_inset.scatter(
        x_val, y_val,
        s=280,
        c="w",
        zorder=998,
        alpha=0.5)

    ax_inset.scatter(
        x_val, y_val,
        s=200,
        ec="k",
        c=col,
        zorder=999,
        alpha=1)

ax_inset.set_xlim(zoom_xlim)
ax_inset.set_xticks([0.8, 0.85])
ax_inset.set_xticklabels(["0.8", "0.85"])
ax_inset.set_ylim(zoom_ylim)
ax_inset.set_yticks([1.0, 1.1])
ax_inset.set_yticklabels(["1.0", "1.1"])
ax_inset.yaxis.tick_right()

zoom_rect = Rectangle(
    (zoom_xlim[0], zoom_ylim[0]),
    zoom_xlim[1] - zoom_xlim[0],
    zoom_ylim[1] - zoom_ylim[0],
    fill=False,
    edgecolor="k",
    linewidth=1.5,
    linestyle="--",
    zorder=10)

con1 = ConnectionPatch(
    xyA=(zoom_xlim[1], zoom_ylim[1]),
    coordsA=ax2.transData,
    xyB=(zoom_xlim[0], zoom_ylim[1]),
    coordsB=ax_inset.transData,
    color="k",
    linewidth=1.0,
    linestyle="--",
    zorder=2)

con2 = ConnectionPatch(
    xyA=(zoom_xlim[1], zoom_ylim[0]),
    coordsA=ax2.transData,
    xyB=(zoom_xlim[0], zoom_ylim[0]),
    coordsB=ax_inset.transData,
    color="k",
    linewidth=1.0,
    linestyle="--",
    zorder=2)

ax2.add_artist(con1)
ax2.add_artist(con2)

name_map = {
    "Truax": "Truax (This Study)",
    "Badgeley_low": "Badgeley et al. (2020) Low",
    "Badgeley_pref": "Badgeley et al. (2020) Preferred",
    "Buizert": "Buizert et al. (2018)",
    "Erb": "Erb et al. (2022)",
    "Osman": "Osman et al. (2021)",
    "TraCE21k": "TraCE21k, Liu et al. (2009)"}

product_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=colors[prod],
        markeredgecolor='k',
        markersize=14,
        label=name_map.get(prod, prod))
    for prod in products]

fig.legend(
    handles=product_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0),
    frameon=False,
    ncol=3,
    columnspacing=2.0,
    handletextpad=0.7,
    labelspacing=1.0)

ax1.text(0.1, 0.93,"(A)",transform=ax1.transAxes,fontsize=16,fontweight="bold")
ax2.text(0.1, 0.93,"(B)",transform=ax2.transAxes,fontsize=16,fontweight="bold",zorder=999)
ax_inset.text(0.05, 0.82,"(C)", transform=ax_inset.transAxes,fontsize=16,fontweight="bold")

plt.tight_layout()
plt.savefig(repo_root / "Figures" / "Figure 4" / "Fig. 4.png",dpi=300,bbox_inches="tight",pad_inches=0.05)
plt.show()