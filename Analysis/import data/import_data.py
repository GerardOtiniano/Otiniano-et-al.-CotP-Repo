from pathlib import Path
import sys
from proxy_import import import_proxies

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / 'Analysis' / 'FA'))
from factor_analysis.constrain_boundaries import get_common_footprint, subset_proxies_to_common_footprint

meta, data = import_proxies(repo_root / 'Data' / 'Raw')
product_dir = repo_root / 'Data' / 'Climate Reconstruction Products'
products = ['Truax', 'Osman', 'Badgeley_low', 'Badgeley_pref', 'Buizert', 'TraCE21k']
footprint = get_common_footprint(products, product_dir=product_dir, require_same_grid=False)
meta, data, _, _ = subset_proxies_to_common_footprint(meta, data, footprint, product_dir=product_dir)
output_dir = repo_root / 'Data' / 'Proxy Time Series'
output_dir.mkdir(parents=True, exist_ok=True)
meta.to_csv(output_dir / 'proxy_meta3.csv', index=False)
data.to_csv(output_dir / 'proxy_ts3.csv', index=False)
print(f'Saved {len(meta)} records at {len(data)} ages to {output_dir}')
