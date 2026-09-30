import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from factor_analysis.procrustes import build_proxy_self_dist_distribution_ZW,compare_da_to_proxies,compute_self_thresholds

repo_root = Path(__file__).resolve().parents[2]
output_dir = repo_root / "Data" / "Analysis Output"
with (output_dir / "fa_proxy_prod.pkl").open("rb") as file:
    results = pickle.load(file)

# Subset proxies used in original FA FA
data = pd.read_csv(repo_root / "Data" / "Proxy Time Series" / "proxy_ts.csv")
proxy_ids = results["Proxy"]["n_z_2"]["meta_df"]["index"]
data = data[["age", *proxy_ids]]

da_names = ["Truax", "Osman", "Badgeley_low", "Badgeley_pref", "Buizert", "TraCE21k"] # products for comparison
fractions = [0.5, 0.6, 0.7, 0.8, 0.9] # proxy network fractions to retain
n_boot = 100 # number of bootstrap samples to use
null_results = {}

for fraction in fractions:
    Z_ref, W_ref, dZ, dW, overall, stored_FA = build_proxy_self_dist_distribution_ZW(data, results, fraction, n_boot=n_boot)
    thresholds = compute_self_thresholds(dZ, dW, overall)
    summary = compare_da_to_proxies(results, da_names, thresholds)
    null_results[fraction] = {"Z_ref": Z_ref, "W_ref": W_ref,
        "dZ_list": dZ, "dW_list": dW, "overall_list": overall,
        "stored_FA": stored_FA, "summary_da_full": summary,
        "thresholds": thresholds}
    print(f"\nProxy fraction: {fraction:.0%}; 99th-percentile thresholds:")
    for label, values in thresholds.items():
        print(f"  {label}: {np.array2string(values, precision=3)}")
    for entry in sorted(summary, key=lambda entry: entry["overall_mean"]):
        print(f"  {entry['product']:14s} mean distance = {entry['overall_mean']:.3f}")

output_path = output_dir / "procrustes_results.pkl"
with output_path.open("wb") as file:
    pickle.dump(null_results, file, protocol=pickle.HIGHEST_PROTOCOL)
