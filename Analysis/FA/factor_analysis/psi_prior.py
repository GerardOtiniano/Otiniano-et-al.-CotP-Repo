from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.linalg import slogdet, inv
from pathlib import Path
from typing import Mapping

def make_sigma2_from_records(df, meta_df, scale_b, *, error_col='Error_val'):
    """Mode C - ts specific uncertainty"""
    cols = df.columns.drop('age') # remove age column
    # match uncertainties to records.
    if 'index' in meta_df.columns:
        key_series = meta_df['index'].astype(str)
    else: # if no names, assume rows are in same order
        key_series = pd.Index(cols, name='index').astype(str)
    sigma_map = dict(zip(key_series, meta_df[error_col].astype(float))) # map ts name to uncertainty
    sigma2 = []
    # Pair ts with scale used for normalizing
    for col, b in zip(cols, scale_b):
        sigma_native = sigma_map.get(str(col), np.nan)
        if not np.isfinite(b) or b <= 0 or not np.isfinite(sigma_native) or sigma_native <= 0:
            sigma2.append(np.nan)
        else:
            sigma2.append((sigma_native / b)**2) # convert to normalized units and get variance
    return np.asarray(sigma2)

def make_sigma2_from_calibration(df, meta_df, proxy_sigma_C, scale_b):
    """Mode B - proxy specific uncertainty"""
    cols = df.columns.drop('age')
    # map ts name to proxy type
    idx_map = dict(zip(meta_df['index'].astype(str), meta_df['proxy'].astype(str))) if meta_df is not None else {}
    sigma2 = []
    for col, b in zip(cols, scale_b):
        p = idx_map.get(str(col))
        sigma = proxy_sigma_C.get(p, np.nan) if p is not None else np.nan
        if not np.isfinite(b) or b <= 0 or not np.isfinite(sigma):
            sigma2.append(np.nan)
        else:
            sigma2.append((sigma / b)**2) # ts of same proxy type can have different scales
    return np.asarray(sigma2)

def load_proxy_sigma_c(csv_path: str | Path | None = None) -> dict[str, float]:
    """Read the uncertainty for each proxy type from a CSV file"""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "proxy_uncertainty.csv"
    else:
        csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Proxy uncertainty CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # The CSV must identify each proxy type and its uncertainty.
    required = {"proxy", "uncertainty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {', '.join(sorted(missing))}" )
    proxy_sigma = {row["proxy"]: float(row["uncertainty"])
                   for _, row in df.iterrows()}
    return proxy_sigma
