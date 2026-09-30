import pandas as pd
import numpy as np
from factor_analysis.eof import run_fa
from factor_analysis.bic import compute_log_likelihood, bic_score

def fa_proxy_wrapper(meta_df, df, number_of_components, psi_mode='B', update_mu=False):
    """Wrapper function for handling proxies in Otiniano et al."""
    # Setup blank parameters for analysis
    results = {}
    results['Proxy'] = {}
    best_bic_value = np.inf
    best_q = None
    Q = number_of_components+1

    # Run fa over Q components
    for i in range(2,Q):
        eof_vars, meta_df, frac_explain, X_proc  = run_fa(meta_df = meta_df, df=df, Q=i, psi_mode=psi_mode, update_mu=update_mu)
        mu, W, Z_mean, frac_explain, sigma2 = reorder_factors(frac_explain, eof_vars)
        bic_val = calc_BIC(X_proc, mu, W, sigma2, i, psi_mode) # compute BIC
        # Track best model via BIC
        if bic_val < best_bic_value:
            best_bic_value = bic_val
            best_q = i

        results['Proxy'][f'n_z_{i}'] = {
            'meta_df': meta_df.copy(deep=True),
            'eof_vars': {
                'mu': mu,
                'W': W,
                'sigma2': sigma2,
                'Z_mean': Z_mean},
            'frac_explain': frac_explain,
            'X': X_proc,
            'age': df['age'],
            'BIC': bic_val}
    return best_q, results

def fa_prod_wrapper(df, Q, results, prod_name, unique_coords, psi_mode='A'):
    """"Wrapper function for running products in Otiniano et al."""
    if "DA" not in results.keys():
        results["DA"] = {}
    results['DA'][prod_name]={}
    best_bic_value = np.inf
    best_q = None
    best_eof_vars = None
    Q = Q+1
    for i in range(2, Q):
        eof_vars, frac_explain, X_proc = run_fa(df=df, Q=i, meta_df = None, psi_mode = psi_mode,tol=1e-4)
        mu, W, Z_mean, frac_explain, sigma2 = reorder_factors(frac_explain, eof_vars)
        bic_val = calc_BIC(X_proc, mu, W, sigma2, i, psi_mode) # compute BIC
        if bic_val < best_bic_value:
            best_bic_value = bic_val
            best_q = i
        results['DA'][prod_name][f'n_z_{i}'] = {
            'eof_vars': {
                'mu': mu,
                'W': W,
                'sigma2': sigma2,
                'Z_mean': Z_mean},
            'frac_explain': frac_explain,
            'X_proc': X_proc,
            'age': df['age'],
            'BIC': bic_val,
            'unique_coords':unique_coords}
        results['DA'][prod_name]['best_q'] = best_q
        results['DA'][prod_name][f'n_z_{i}']['eof_vars']['W'] = add_LatLon_to_W(df, results['DA'][prod_name][f'n_z_{i}']['eof_vars']['W'])
    return best_q, results

def reorder_factors(frac_explain, eof_vars):
    """Reorders factors based on explained variance"""
    mu, W, sigma2, Z_mean = eof_vars
    ordering = np.argsort(frac_explain)[::-1]
    W = W[:, ordering]
    Z_mean = Z_mean[:, ordering]
    frac_explain = frac_explain[ordering]
    return mu, W, Z_mean, frac_explain, sigma2

def calc_BIC(X_proc, mu, W, sigma2, i, psi_mode):
    "caluclate BIC for fa iteration"
    N, D = X_proc.shape
    logL = compute_log_likelihood(X_proc, mu, W, sigma2)
    if psi_mode == "A":
        num_params = D * i + D + D   # number of parameters from W (D*i), Psi (D) and mu (D)
    elif psi_mode in ["B", "C"]:
        num_params = D * i + D   # number of parameters from W (D*i), and mu (D) (Psi is constant so not added)
    bic_val = bic_score(logL, num_params, N)
    return bic_val

def add_LatLon_to_W(df, W):
    """add latitudes and longitudes to loading matrix - helpful for plotting"""
    col_names = df.columns[1:]  # skip age column
    latitudes, longitudes = zip(*[map(float, col.split('_')) for col in col_names])
    W_df = pd.DataFrame(W)
    W_df['latitude'] = latitudes
    W_df['longitude'] = longitudes
    return W_df
