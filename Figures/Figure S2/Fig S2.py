import numpy as np
import pickle
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import string
from pathlib import Path

# Functions
plt.rcParams.update({'font.size': 14})
def density_threshold(x,y, densities = [0.34, 0.68, 0.95]):
    """Create KDE densities for null distribution of a given proxy retention amount""" 
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
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
prcy=0.99
percents = [0.5, 0.6, 0.7, 0.8]

fig, axs = plt.subplots(nrows = len(percents), ncols = 2, figsize=(7,10), sharex='col')
fig.subplots_adjust(wspace=0.45)
cluster_colors = ['#8E6C88', '#458A6C']

for ax_row, percentile in enumerate(percents):
    data = data_full[percentile]
    
    # Cluster
    Zdist = data['dZ_list'][:, :2]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(Zdist)
    centroids = kmeans.cluster_centers_
    order = np.argsort(centroids[:, 0])
    relabel = np.zeros_like(labels)
    for new_label, old_label in enumerate(order):
        relabel[labels == old_label] = new_label
    labels = relabel
    centroids = centroids[order]
    xgrid, ygrid, density, density_levels = density_threshold(
        data['dZ_list'][:, 0],
        data['dZ_list'][:, 1],
        [0.34, 0.68, prcy])
    for level, color, z_pos in zip(
        density_levels.values(),
        ["#495057", "#adb5bd", "#dee2e6"],
        [3, 2, 1]):
        axs[ax_row, 0].contourf(
            xgrid, ygrid, density,
            levels=[level, density.max()],
            colors=[color],
            zorder=z_pos)
        axs[ax_row, 0].contour(
            xgrid, ygrid, density,
            levels=[level],
            colors=["k"],
            zorder=4)

    # W Distance
    xgrid, ygrid, density, density_levels = density_threshold(
        data['dW_list'][:, 0],
        data['dW_list'][:, 1],
        [0.34, 0.68, prcy])
    for level, color, z_pos in zip(
        density_levels.values(),
        ["#982d40", "#bd767e", "#dcbcc2"],
        [3, 2, 1]):
        axs[ax_row, 1].contourf(
            xgrid, ygrid, density,
            levels=[level, density.max()],
            colors=[color],
            zorder=z_pos)
        axs[ax_row, 1].contour(
            xgrid, ygrid, density,
            levels=[level],
            colors=["k"],
            zorder=4)  

for ax_row in range(len(percents)):
    for ax_col in [0, 1]:
        ax = axs[ax_row, ax_col]
        ax.grid(True, alpha=0.5, zorder=0)
        ax.set_xlim(-0.15, 1.05)
        ax.set_ylim(-0.35, 1.4)
        ax.axvline(0, linewidth=2, c='k', zorder=0)
        ax.axhline(0, linewidth=2, c='k', zorder=0)
        ax.axvline(0, linewidth=2, c='k', zorder=3, linestyle='--')
        ax.axhline(0, linewidth=2, c='k', zorder=3, linestyle='--')


for ax, percy in zip(axs[:,1], percents):
    # ax.set_ylabel("Mean W$_{1,2}$ Distance")
    ax.text(1.1, 0.5, f"{round(percy*100)}% retention",
            fontweight="bold", ha="center", va='center',
            size=14, transform = ax.transAxes, rotation=270)

for i, ax in enumerate(axs[0,:]):
    ax.text(0.5, 1.1, f"({string.ascii_uppercase[i]})",
            transform=ax.transAxes,ha="center",va="center", fontweight="bold", clip_on=False)
    
# Axis labels
for ax in axs[:,0]:
    ax.set_ylabel("Z$_{2}$ Distance")
axs[len(percents)-1,0].set_xlabel("Z$_{1}$, Distance")

for ax in axs[:,1]:
    ax.set_ylabel("W$_{2}$ Distance")
axs[len(percents)-1,1].set_xlabel("W$_{1}$, Distance")

plt.savefig(repo_root / "Figures" / "Figure S2" / "Fig. S2.png",dpi=300,bbox_inches="tight",pad_inches=0.05)
plt.show()
