import pandas as pd
import numpy as np
from numpy.linalg import slogdet, inv

def compute_log_likelihood(X, mu, W, sigma2):
    """
    Compute the log-likelihood for observed data in fa model
    Inputs:
        X: array of observations
        mu: mean of each column
        W: loading matrix
        sigma2: noise variance
    Note:
        x can have missing values (np.nan)
    """
    N, D = X.shape
    C = W @ W.T + sigma2 * np.eye(D) # covariance for all columns
    total_ll = 0.0
    for i in range(N):
        obs_mask = ~np.isnan(X[i, :])
        obs_idx = np.where(obs_mask)[0]
        obs_len = len(obs_idx)
        if obs_len == 0:
            continue # skip row if nothing observed
        x_obs = X[i, obs_idx] - mu[obs_idx]  # extract observed subset of x_i and then centre
        C_obs = C[obs_idx, :][:, obs_idx]    # extract corresponding submatrix of covariance
        sign, logdetC_obs = slogdet(C_obs) # determinant
        C_obs_inv = inv(C_obs) # inverse
        # contribution of row i to log likelohood
        quad_form = x_obs @ C_obs_inv @ x_obs
        row_ll = -0.5 * (obs_len * np.log(2.0 * np.pi) + logdetC_obs + quad_form)
        total_ll += row_ll
    return total_ll

def bic_score(log_likelihood, num_params, N):
    """compute bic"""
    return -2.0 * log_likelihood + num_params * np.log(N)
