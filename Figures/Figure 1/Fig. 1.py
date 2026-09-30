from pathlib import Path
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import box

figure_dir = Path(__file__).resolve().parent
data_path = figure_dir.parents[1] / "Data" / "Climate Reconstruction Products"
shp_path = figure_dir / "greenland_shp" / "gl.shp"
greenland = gpd.read_file(shp_path).geometry.buffer(0).union_all()
da_names = ["Truax", "Badgeley_low", "Buizert", "Osman", "Badgeley_pref", "TraCE21k"]
time_slices = [10.0, 8.0, 6.0, 4.0, 2.0]
product_labels = {"Truax": "Truax\n(This Study)", "Badgeley_low": "Badgeley et al.\n(2020) Low", "Badgeley_pref": "Badgeley et al.\n(2020) Preferred", "Buizert": "Buizert et al.\n(2018)", "Osman": "Osman et al.\n(2021)", "TraCE21k": "TraCE21k\nLiu et al.(2009)"}

def grid_edges(centers):
    centers = np.asarray(centers, dtype=float)
    spacing = np.diff(centers) / 2
    return np.concatenate(([centers[0] - spacing[0]], centers[:-1] + spacing, [centers[-1] + spacing[-1]]))

def greenland_tile_mask(ds):
    lon_edges = grid_edges(ds.lon)
    lat_edges = np.clip(grid_edges(ds.lat), -90, 90)
    mask = np.zeros((ds.sizes["lat"], ds.sizes["lon"]), dtype=bool)
    for j in range(ds.sizes["lat"]):
        for i in range(ds.sizes["lon"]):
            mask[j, i] = box(lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1]).intersects(greenland)
    return xr.DataArray(mask, coords={"lat": ds.lat, "lon": ds.lon}, dims=("lat", "lon"))

def handle_da(name):
    with xr.open_dataset(data_path / f"{name}.nc") as ds:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon").sortby("year", ascending=False)
        da = ds["temperature"].where(greenland_tile_mask(ds))
        if name == "Buizert":
            da = da - 273.15
        return da.sel(season="JJA").squeeze(drop=True).load()

products = {name: handle_da(name) for name in da_names}
ref_da = products["Truax"]
anom_dict = {name: {} for name in da_names}
for name, da in products.items():
    reference = da.sel(year=2000, method="nearest")
    for time in time_slices:
        anomaly = da.sel(year=time * 1000, method="nearest") - reference
        anom_dict[name][time] = anomaly.interp(lat=ref_da.lat, lon=ref_da.lon, method="linear")

values = np.concatenate([anomaly.values.ravel() for times in anom_dict.values() for anomaly in times.values()])
absmax = float(np.max(np.abs(values[np.isfinite(values)]))) or 1.0
norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
plt.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(nrows=len(da_names), ncols=len(time_slices) + 1, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=-0.05, hspace=0.08)
proj = ccrs.LambertConformal(central_longitude=-40)
axs = np.array([[fig.add_subplot(gs[i, j], projection=proj) for j in range(len(time_slices))] for i in range(len(da_names))])
cax = fig.add_subplot(gs[1:4, -1])
shape_feature = ShapelyFeature(Reader(shp_path).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor="k")

for i, name in enumerate(da_names):
    for j, time in enumerate(time_slices):
        deviation = anom_dict[name][time]
        ax = axs[i, j]
        ax.set_extent([-60, -20, 57, 84], crs=ccrs.PlateCarree())
        mappable = ax.pcolormesh(deviation.lon, deviation.lat, deviation, transform=ccrs.PlateCarree(), cmap="bwr", norm=norm, shading="auto")
        ax.add_feature(shape_feature)
        ax.set_frame_on(False)
        if i == 0:
            ax.text(0.5, 1.08, f"{time} ka", transform=ax.transAxes, ha="center", va="center")
    axs[i, 0].text(-0.5, 0.5, product_labels[name], transform=axs[i, 0].transAxes, rotation=90, ha="center", va="center")

cbar = fig.colorbar(mappable, cax=cax)
cbar.set_label("JJA temperature anomaly relative to 2 ka (°C)")
plt.savefig(figure_dir / "Fig. 1.png", bbox_inches="tight", dpi=300)
plt.show()
