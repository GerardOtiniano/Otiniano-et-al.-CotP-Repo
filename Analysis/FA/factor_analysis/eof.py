import pandas as pd
import numpy as np
from factor_analysis.time_norm import normalize_with_mode
from factor_analysis.fa_em import fa_em
from factor_analysis.psi_prior import make_sigma2_from_calibration, make_sigma2_from_records, load_proxy_sigma_c

def run_fa(meta_df, df, Q, age_step=0.3, psi_mode='A', ref_min=0, ref_max=10,tol=1e-7, update_mu=False):
    """
    psi_mode:
      'A' → no prior (exactly your current default when run_psi_prior=False)
      'B' → proxy-specific prior (exactly your current behavior when run_psi_prior=True)
      'C' → record-specific prior from meta_df['Error_val'] (1σ in native units)
    """
    ages   = df['age'].to_numpy()
    X_orig = df.drop(columns=['age']).to_numpy()
    # Center or z_score
    X, shift_a, scale_b = normalize_with_mode(X_orig, ages, mode="z_score", ref_min=ref_min, ref_max=ref_max)
    col_var_proc = np.nanvar(X, axis=0)
    # Psi prior selection
    if psi_mode == 'A':
        # A: no prior
        psi_prior = None

    elif psi_mode == 'B':
        # B: proxy-specific prior
        proxy_sigma = load_proxy_sigma_c()
        raw_prior = make_sigma2_from_calibration(df, meta_df, proxy_sigma, scale_b)
        median_val = np.nanmedian(raw_prior)
        psi_prior = np.where(np.isnan(raw_prior), median_val, raw_prior)

    elif psi_mode == 'C':
        # C: record-specific prior
        # need "Error_val" in meta_df, containing record-specific uncertainty/error
        raw_prior  = make_sigma2_from_records(df, meta_df, scale_b)
        median_val = np.nanmedian(raw_prior)
        psi_prior  = np.where(np.isnan(raw_prior), median_val, raw_prior)

    # Run FA
    mu, W, sigma2, Z_mean = fa_em(X, q=Q, psi_prior=psi_prior,tol=tol,update_mu=update_mu)
    component_variances = np.sum(W**2, axis=0)
    fraction_explained = component_variances / np.sum(component_variances)  # common variance
    if meta_df is None:
        return [mu, W, sigma2, Z_mean], fraction_explained, X
    else:
        nrows, ncols = W.shape
        for i in range(len(meta_df)):
            for j in range(ncols):
                meta_df.loc[i, f'Loading {j}'] = W[i, j]
        return [mu, W, sigma2, Z_mean], meta_df, fraction_explained, X
