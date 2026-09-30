import pandas as pd
from pathlib import Path
from factor_analysis.fa_wrappers import fa_proxy_wrapper, fa_prod_wrapper
from factor_analysis.import_prod import load_prod
from factor_analysis.temporal_subset import subset_prod_temporally
import pickle
repo_root = Path(__file__).resolve().parents[2]
data = pd.read_csv(repo_root / "Data" / "Proxy Time Series" / 'proxy_ts.csv') # proxy time series
meta = pd.read_csv(repo_root / "Data" / "Proxy Time Series" / 'proxy_meta.csv') # proxy time series metdata

q = 10 # number of components
best_q, results = fa_proxy_wrapper(meta, data, q) # proxy time series factor analysis

# factor analysis on reconstrution products
for prod_name in ["Truax", "Osman", "Badgeley_low", "Badgeley_pref" , "TraCE21k", "Buizert"]:
    prod, unique_coords, _, _ = load_prod(prod_name,meta,data,max_age=10,rebinned_age_step=0.3)
    prod = subset_prod_temporally(prod, prod_name, unique_coords, results)
    best_q, results = fa_prod_wrapper(prod, q, results, prod_name, unique_coords)

# Save analysis output
out_path = repo_root / "Data" / "Analysis Output" / "fa_proxy_prod.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)
