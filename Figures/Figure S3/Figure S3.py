import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import operator as op
import string
from pathlib import Path

name_map = {"Truax": "Truax\n(This study)", "Badgeley_low": "Badgeley et al.\n(2020) Low", "Badgeley_pref": "Badgeley et al.\n(2020) Preferred", "Buizert": "Buizert et al.\n(2018)", "Erb": "Erb et al.\n(2022)", "Osman": "Osman et al.\n(2021)", "TraCE21k": "TraCE21k\nLiu et al. (2009)"}
repo_root = Path(__file__).resolve().parents[2]
def ensemble_temperature(pos_loadings, da):
    pos_df = pd.DataFrame()
    for j in range(len(pos_loadings)):
        lat_j = pos_loadings.loc[j, 'latitude']
        lon_j = pos_loadings.loc[j, 'longitude']
        da_sub = da.sel(lat=lat_j, lon=lon_j, method='nearest')
        temp = da_sub['temperature'].values
        pos_df[j] = temp
    age = da.year.values
    pos_df.insert(0, 'age', age)
    temp_matrix = pos_df.drop(columns='age').to_numpy()
    median = np.nanmedian(temp_matrix, axis=1)
    std = np.nanstd(temp_matrix, axis=1)
    lower_std = median - std
    upper_std = median + std
    ensemble_df = pd.DataFrame({'age': age, 'median': median, 'std': std, 'lower_std': lower_std, 'upper_std': upper_std, 'n_cells': temp_matrix.shape[1]})
    return ensemble_df, pos_df


def plot_quant_ensemble(ax, results, factor, sign_flip=1):
    qual = ["Bennike2002", "Bennike2008", "BennikeWagner2012", "BennikeWeidick2001", "Christiansen2002", "Klug2009", "Schmidt2011_Duck", "Schmidt2011_Hjort", "Wagner2000", "Wagner2008", "WagnerBennike2015"]
    df = pd.read_csv(repo_root / "Figures" / "Figure S3" / "proxy_ts.csv")
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
                temp[rec] = temp['val'] - temp['val'].mean()
                centered_records.append(temp[['age', rec]].set_index('age'))
        centered_df = pd.concat(centered_records, axis=1).sort_index()
        median = centered_df.median(axis=1, skipna=True)
        std = centered_df.std(axis=1, skipna=True)
        lower_std = median - std
        upper_std = median + std
        ax.fill_between(centered_df.index, lower_std, upper_std, alpha=0.25, label='±1 SD', color=coly)
        ax.plot(centered_df.index, median, linewidth=3, c=coly, label='Median')
        num_ts.append(len(centered_records))
    ax.text(0.7, 0.13, f"(n = {num_ts[0]})", transform=ax.transAxes, c="red", ha="left")
    ax.text(0.7, 0.05, f"(n = {num_ts[1]})", transform=ax.transAxes, c="blue", ha="left")
    
    

path = repo_root / "Data" / "Analysis Output" / "fa_proxy_prod.pkl"
with open(path, 'rb') as handle:
    results = pickle.load(handle)

path = repo_root / "Data" / "Analysis Output" / "procrustes_results.pkl"
with open(path, 'rb') as handle:
    proc_results = pickle.load(handle)

da_names = ["Truax", "Osman", "Badgeley_low", "Badgeley_pref", "Buizert", "TraCE21k"]
ensemble_results = {}
nrows = len(da_names) + 1
ncols = 2
fig = plt.figure(figsize=(10, 15))
gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, height_ratios=[2] + [1] * len(da_names), hspace=0.35, wspace=0.15)
axs = np.empty((nrows, ncols), dtype=object)

for r in range(nrows - 1, -1, -1):
    for c in range(ncols):
        if r == nrows - 1 and c == 0:
            axs[r, c] = fig.add_subplot(gs[r, c])
        elif r == nrows - 1 and c == 1:
            axs[r, c] = fig.add_subplot(gs[r, c], sharey=axs[r, 0])
        elif c == 0:
            axs[r, c] = fig.add_subplot(gs[r, c], sharex=axs[nrows - 1, c])
        else:
            axs[r, c] = fig.add_subplot(gs[r, c], sharex=axs[nrows - 1, c], sharey=axs[r, 0])

for i, da_name in enumerate(da_names):
    i = i + 1
    da = xr.load_dataset(repo_root / "Data" / "Climate Reconstruction Products" / f"{da_name}.nc")
    da = da.where(da.year < 10000, drop=True)
    da = da.sel(season='JJA')
    da_proc = next(entry for entry in proc_results[0.7]["summary_da_full"] if entry["product"] == da_name)
    ensemble_results[da_name] = {}
    for W in [0, 1]:
        ax = axs[i, W]
        sign = da_proc['signs'][W]
        loadings = results['DA'][da_name]['n_z_2']['eof_vars']['W'].copy()
        loadings['alpha'] = np.abs(loadings[W]) / np.nanmax(np.abs(loadings[W]))
        loadings[W] = loadings[W] * sign
        neg_c = "#247BA0"
        neg_loadings = loadings.loc[loadings[W] <=0].reset_index(drop=True)
        ensemble_results[da_name][f'W{W + 1}'] = {}
        if len(neg_loadings) > 1:
            ensemble_df, neg_df = ensemble_temperature(neg_loadings, da)
            ensemble_results[da_name][f'W{W + 1}']['neg'] = {'negative_loading_cells': neg_loadings, 'temperature_matrix': neg_df, 'ensemble': ensemble_df}
            ax.fill_between(ensemble_df['age'] / 1000, ensemble_df['lower_std'], ensemble_df['upper_std'], alpha=0.3, color=neg_c, zorder=2)
            ax.plot(ensemble_df['age'] / 1000, ensemble_df['median'], c=neg_c, lw=1.5, zorder=3, label=f"- (n={neg_df.shape[1]})")
            neg_num = int(ensemble_df['n_cells'].iloc[0])
        elif len(neg_loadings) == 1:
            lat_j = neg_loadings['latitude'].values[0]
            lon_j = neg_loadings['longitude'].values[0]
            da_sub = da.sel(lat=lat_j, lon=lon_j, method='nearest')
            temp = da_sub['temperature'].values
            age = da_sub['year'].values
            ax.plot(age / 1000, temp, c=neg_c, lw=1.5, zorder=3, label="- (n=1)")
            neg_num = 1
        else:
            neg_num = 0
        pos_c = "#DE541E"
        pos_loadings = loadings.loc[loadings[W] >=0].reset_index(drop=True)
        if len(pos_loadings) > 1:
            ensemble_df, pos_df = ensemble_temperature(pos_loadings, da)
            ensemble_results[da_name][f'W{W + 1}']['pos'] = {'positive_loading_cells': pos_loadings, 'temperature_matrix': pos_df, 'ensemble': ensemble_df}
            ax.fill_between(ensemble_df['age'] / 1000, ensemble_df['lower_std'], ensemble_df['upper_std'], alpha=0.3, color=pos_c, zorder=2)
            ax.plot(ensemble_df['age'] / 1000, ensemble_df['median'], c=pos_c, lw=1.5, zorder=3, label=f"+ (n={pos_df.shape[1]})")
            pos_num = int(ensemble_df['n_cells'].iloc[0])
        elif len(pos_loadings) == 1:
            lat_j = pos_loadings['latitude'].values[0]
            lon_j = pos_loadings['longitude'].values[0]
            da_sub = da.sel(lat=lat_j, lon=lon_j, method='nearest')
            temp = da_sub['temperature'].values
            age = da_sub['year'].values
            ax.plot(age / 1000, temp, c=pos_c, lw=1.5, zorder=3, label="+ (n=1)")
            pos_num = 1
        else:
            pos_num = 0
        ax.text(0.7, 0.25, f"(n = {pos_num})", transform=ax.transAxes, c=pos_c, ha="left")
        ax.text(0.7, 0.1, f"(n = {neg_num})", transform=ax.transAxes, c=neg_c, ha='left')

for row_num, da_name in enumerate(da_names):
    row_num = row_num + 1
    axs[row_num, 1].text(1.1, 0.5, name_map[da_name], rotation=270, va='center', ha='center', transform=axs[row_num, 1].transAxes)

plot_quant_ensemble(axs[0, 0], results, 0, sign_flip=1)
plot_quant_ensemble(axs[0, 1], results, 1, sign_flip=1)

for fac in [0, 1]:
    axs[0, fac].text(5, 6.5, f"Factor {fac + 1}", ha="center", fontweight="bold")

axs[0, 0].set_ylabel("Centered Summer\nTemperatures (°C)")
axs[0, 1].text(1.1, 0.5, "Quantitative Proxy\nTime Series", va="center", ha='center', rotation=270, transform=axs[0, 1].transAxes)
fig.text(0.5, 0.07, 'Age (cal ka BP)', ha='center', fontweight="bold")
fig.text(0.05, 0.5, 'Mean JJA Temperature (°C)', rotation=90, ha='center', va='center', fontweight="bold")

for ax in axs.flat:
    ax.set_xlim(10, 0)

for ax in axs[:-1, :].flat:
    ax.tick_params(labelbottom=False)

for ax in axs[:, 1]:
    ax.tick_params(labelleft=False)
    
for ax in axs[1:, :].flat:
    ax.set_ylim(-7,8)

for ax in axs[:, :].flat:
    ax.axhline(0, c='k', zorder=-10, alpha = 0.5)
    
for i, ax in enumerate(axs[:, -1]):
    ax.text(
        1.18, 0.5, f"({string.ascii_uppercase[i]})",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
        rotation=270)

top_axes = []
for c in range(ncols):
    top_ax = axs[0, c].secondary_xaxis('top')
    top_ax.set_xlabel("")
    top_ax.tick_params(axis='x', which='both', direction='out')
    top_axes.append(top_ax)

fig.savefig(repo_root / "Figures" / "Figure S3" / "Fig S3.png", dpi=300,bbox_inches="tight",pad_inches=0.05)
plt.show()

