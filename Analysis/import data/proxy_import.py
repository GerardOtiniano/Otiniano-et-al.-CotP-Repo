import pickle
import numpy as np
import pandas as pd

AGES = np.arange(0, 10.2, 0.3)
BLOCKED_SITES = {1766, 1769, 1772, 2167, 782, 681, 301, 363, 364, 915, 1506, 1132, 1994, 611, 1108, 1709, 1751, 1918, 2158, 1159, 638, 1011, 744, 1972, 821, 1276, 1677, 1952, 1546, 1228, 1166, 1035, 1128, 1512, 1773} # dupes
BLOCKED_RECORDS = {('iglutalik.davis.1980', 'pollen'), ('hu91-039-008.levac.2001', 'dinocyst'), ('sip29.solignac.2004', 'dinocyst'), ('md99-2227.devernal.2013', 'dinocyst'), ('jm96-1207.solignac.2013', 'dinocyst')} # dupes


def cascade_bin(ages, values):
    # Average observations in 0.15 ka bins, then average those means in 0.3 ka bins.
    half_edges = np.arange(-0.075, 10.15, 0.15)
    full_edges = np.arange(-0.15, 10.3, 0.3)
    half_centres = np.round(half_edges[:-1] + 0.075, 3)
    half_bins = [[] for _ in half_centres]
    full_bins = [[] for _ in AGES]
    for age, value in zip(ages, values):
        if np.isnan(age) or np.isnan(value): 
            continue
        i = np.digitize(age, half_edges) - 1
        if 0 <= i < len(half_bins):
            half_bins[i].append(value)
    for age, values in zip(half_centres, half_bins):
        i = np.digitize(age, full_edges) - 1
        if values and 0 <= i < len(full_bins):
            full_bins[i].append(np.nanmean(values))
    return [np.nanmean(values) if values else np.nan for values in full_bins]


def import_quantitative(raw_dir):
    with open(raw_dir / 'GRate_paleoData.pickle', 'rb') as handle:
        records = pickle.load(handle)
    data = pd.DataFrame({'age': AGES})
    rows = []
    for key, record in records.items():
        m = record['metadata']
        pair = (str(m['dataset']).strip().lower(), str(m['proxy']).strip().lower())
        if key in BLOCKED_SITES or pair in BLOCKED_RECORDS or m['seasonality_general'] != 'summer' or m['proxy'] in {'foraminifera', 'radiolaria', 'd18O'}:
            continue
        ages = np.asarray(record['variables']['age'])
        if len(ages) == 0 or ages.max() < 6 or ages.min() > 4:
            continue
        record_id = str(m['index'])
        data[record_id] = cascade_bin(ages, np.asarray(record['variables']['original_values']))
        rows.append({'index': record_id, 'name': m['dataset'], 'ID': m['dataset'], 'lat': m['latitude'], 'lon': m['longitude'], 'archive': m['archive'], 'proxy': m['proxy'], 'seasonality': m['seasonality_general']})
    return pd.DataFrame(rows), data


def import_qualitative(raw_dir):
    folder = raw_dir / 'Qualitative Data'
    metadata = pd.read_csv(folder / 'qualitative_meta.csv', encoding='latin1')
    metadata.columns = metadata.columns.str.lower()
    data = pd.DataFrame({'age': np.round(AGES, 3)})
    rows = []
    for path in sorted((folder / 'Time Series').glob('*.csv')):
        matches = metadata.loc[metadata['id'] == path.stem]
        if matches.empty or path.stem in {'Bennike1999.Hochstetter', 'Bennike1999.JamesonLand'}:
            continue
        m = matches.iloc[0]
        record = pd.read_csv(path)
        # Qualitative bins start at the labelled age; the original output was not centered.
        values = record['Quant'].replace({-1.0: -1.5, 1.0: 1.5}).astype(float)
        bins = (np.floor(record['Age'] / 1000 / 0.3) * 0.3).round(3)
        data[path.stem] = values.groupby(bins).mean().reindex(data['age']).to_numpy()
        rows.append({'index': path.stem, 'ID': path.stem, 'lat': float(m['latitude']), 'lon': float(m['longitude']), 'archive': str(m['archive']), 'proxy': str(m['proxy']), 'seasonality': str(m.get('seasonality', m.get('seasonality_general', m.get('season', 'all'))))})
    return pd.DataFrame(rows), data


def import_sudip(raw_dir):
    folder = raw_dir / 'Sudip Data'
    metadata = pd.read_csv(folder / 'meta.csv', encoding='utf-8-sig')
    metadata.columns = metadata.columns.str.lower().str.strip()
    data = pd.DataFrame({'age': AGES})
    rows = []
    for path in sorted((folder / 'Records').glob('*.csv')):
        m = metadata.loc[metadata['id'] == path.stem].iloc[0]
        record = pd.read_csv(path)
        data[path.stem] = cascade_bin(pd.to_numeric(record['age'], errors='coerce'), pd.to_numeric(record['temperature'], errors='coerce'))
        rows.append({'index': path.stem, 'ID': path.stem, 'lat': float(m['latitude']), 'lon': float(m['longitude']), 'archive': str(m['archive']), 'proxy': str(m['proxy']), 'seasonality': str(m['seasonality']).strip().lower()})
    return pd.DataFrame(rows), data


def import_proxies(raw_dir):
    meta, data = import_quantitative(raw_dir)
    qual_meta, qual_data = import_qualitative(raw_dir)
    sud_meta, sud_data = import_sudip(raw_dir)
    meta = pd.concat([meta, qual_meta, sud_meta], ignore_index=True)
    meta['proxy'] = meta['proxy'].replace({'GDGT': 'brGDGTs', 'GDGTs': 'brGDGTs', 'nan': 'chironomid'})
    data = pd.merge_asof(qual_data.sort_values('age'), data.sort_values('age'), on='age', direction='nearest', tolerance=0.005)
    data = pd.merge_asof(data.sort_values('age'), sud_data.sort_values('age'), on='age', direction='nearest', tolerance=0.0005)
    record_ids = sorted(meta['index'])
    meta = meta.set_index('index').loc[record_ids].reset_index()
    return meta, data[['age'] + record_ids]
