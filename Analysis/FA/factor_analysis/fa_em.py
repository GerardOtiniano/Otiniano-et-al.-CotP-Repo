import numpy as np
import pandas as pd
from numpy.linalg import solve, slogdet, norm

def _row_obs_indices(X: np.ndarray):
    return [np.where(np.isfinite(X[i]))[0] for i in range(X.shape[0])]

def e_step(X, W, Psi, mu, row_obs, Iq):
    """expextation - estimate latent factor Z from current W, Psi, mu"""
    N = X.shape[0]
    q = W.shape[1]
    Z_mean = np.zeros((N, q))
    Z_cov = np.zeros((N, q, q))

    # loop through time steps, N
    for i in range(N):
        obs_idx = row_obs[i] # identify time series with valid observations in time bin
        if obs_idx.size == 0:
            Z_mean[i] = 0.0
            Z_cov[i] = Iq
            continue
        # subset model parameters for valid observations
        Psi_obs = Psi[obs_idx]
        W_obs = W[obs_idx, :]
        x_obs = X[i, obs_idx]
        mu_obs = mu[obs_idx]

        A = W_obs.T @ (W_obs / Psi_obs[:, None]) # Information about Z from observed proxies weighted by residual variance
        M = Iq + A # posterior precision matrix - latent factors
        Z_cov[i] = solve(M, Iq) # posterior covariance (uncertainty) of latent factors
        rhs = W_obs.T @ ((x_obs - mu_obs) / Psi_obs) # project centered observatios into latent factor space
        Z_mean[i] = solve(M, rhs) # posterior factor score
    return Z_mean, Z_cov


def m_step(X, Z_mean, Z_cov, W, Psi, mu, col_has_obs, col_var, psi_floor_frac, ridge, update_mu, freeze_psi, Iq):
    """maximization - given estimated Z, improve W, Psi, mu"""
    N, D = X.shape
    q = W.shape[1]
    new_W = np.zeros_like(W)
    new_Psi = Psi.copy()
    new_mu = mu.copy()
    Ezzt = Z_cov + np.einsum("ni,nj->nij", Z_mean, Z_mean)  # outer product (combine factor scores and uncertainty)

    # loop through proxy time series (D)
    for d in range(D):
        if not col_has_obs[d]: # proxies without observations
            new_W[d] = 0.0
            new_Psi[d] = Psi[d]
            continue
        idx = np.where(np.isfinite(X[:, d]))[0] # identify time bins with valid observations for time series
        if idx.size == 0:
            new_W[d] = W[d]
            new_Psi[d] = Psi[d]
            continue
        # prep update of W
        inv_Psi_d = 1.0 / Psi[d]
        num = np.zeros((q,))
        den = np.zeros((q, q))

        # W update
        for i in idx:
            xi_centered = X[i, d] - mu[d]
            num += inv_Psi_d * xi_centered * Z_mean[i] # variance of proxy with  latent factors
            den += inv_Psi_d * Ezzt[i] # variance in latent factors (Ezzt accounts for uncertainty)
        den = den + ridge * Iq # add ridge regularization
        w_d = solve(den, num) # estimate new loadings for proxy d
        new_W[d] = w_d
        # mu update
        if update_mu:
            numer_mu = 0.0
            for i in idx:
                numer_mu += (X[i, d] - np.dot(w_d, Z_mean[i])) # subtract information explained by latent factors
            new_mu[d] = numer_mu / idx.size
        # psi update
        if not freeze_psi:
            sse = 0.0
            for i in idx:
                err = X[i, d] - new_mu[d] - np.dot(w_d, Z_mean[i]) # observed - predicted
                sse += err**2 + w_d @ Z_cov[i] @ w_d
            est = sse / idx.size # residual variance
            est = max(est, psi_floor_frac * col_var[d]) # prevents total decay of psi
            new_Psi[d] = est
    return new_W, new_Psi, new_mu

def _compute_loglik(X, W, Psi, mu, row_obs):
    """Compute log-likelihood of observed data in FA model"""
    N = X.shape[0]
    ll = 0.0

    # loop through time bins
    for i in range(N):
        obs_idx = row_obs[i]
        if obs_idx.size == 0:
            continue
        W_obs = W[obs_idx, :]
        Psi_obs = Psi[obs_idx]
        mu_obs = mu[obs_idx]
        C = W_obs @ W_obs.T + np.diag(Psi_obs) # covariance matrix
        C = C + (1e-9 * np.eye(C.shape[0])) # stabalize
        sign, logdet = slogdet(C) # sign and logarithm of the determinant of C
        if sign <= 0:
            print("unstable determinant")
            logdet = np.log(np.linalg.det(C + 1e-6 * np.eye(C.shape[0])) + 1e-12) # if C doesnt have determinant, try additional stabalization
        x_obs = X[i, obs_idx] # proxy observation
        alpha = solve(C, x_obs - mu_obs)
        ll += -0.5 * (obs_idx.size * np.log(2.0 * np.pi)+logdet+(x_obs-mu_obs) @ alpha) # multivariate normal log-likelihood
    return ll

def fa_em(X, q, psi_prior, update_mu=False, psi_floor_frac=1e-3,ridge=1e-6,compute_loglik=True,tol=1e-6,max_iter=1000000,random_state=42):
    """Factor Analysis EM algorithm. Optional implementations of priors, and mu. update_mu: Update in M-step (True) or hold at fixed initial value (False)"""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    iters = range(max_iter) # maximum number of iterations for convergence
    # Proxy means and variances
    col_means = np.nanmean(X, axis=0)
    mu = np.where(np.isfinite(col_means), col_means, 0.0) # proxy time series means
    col_var = np.nanvar(X - mu, axis=0) # variance of proxy time series
    col_var = np.where(np.isfinite(col_var) & (col_var > 0), col_var, 1.0) # safeguard if variance is undefined
    # Initialize W
    W = 0.01 * rng.standard_normal((D, q))
    col_has_obs = np.isfinite(np.nanmean(X, axis=0)) # check for empty proxy time series
    if psi_prior is not None:
        psi_prior = np.asarray(psi_prior, float).reshape(-1)
        psi_prior = np.where(
            np.isfinite(psi_prior) & (psi_prior > 0), psi_prior, col_var)
        freeze_psi = True
        Psi = np.maximum(psi_floor_frac * col_var, psi_prior).copy()
    else:
        freeze_psi = False
        Psi = np.clip(col_var, psi_floor_frac * col_var, None)
    Psi = np.where(col_has_obs, Psi, 1.0)
    W[~col_has_obs, :] = 0.0
    row_obs = _row_obs_indices(X)
    Iq = np.eye(q)
    ll_history = []
    prev_ll = None
    for it in iters:
        # expectation
        Z_mean, Z_cov = e_step(X, W, Psi, mu, row_obs, Iq)
        # maximization
        new_W, new_Psi, new_mu = m_step(X, Z_mean, Z_cov, W, Psi, mu, col_has_obs, col_var, psi_floor_frac, ridge, update_mu, freeze_psi, Iq)
        # change in W, psi, mu
        rel_W = norm(new_W - W) / (norm(W) + 1e-12)
        rel_Psi = (norm(new_Psi - Psi) / (norm(Psi) + 1e-12) if not freeze_psi else 0.0)
        rel_mu = (norm(new_mu - mu) / (norm(mu) + 1e-12) if update_mu else 0.0)
        # finalize updated values
        W = new_W
        if update_mu:
            mu = new_mu
        if not freeze_psi:
            Psi = new_Psi
        # log-likelihood
        if compute_loglik:
            ll = _compute_loglik(X, W, Psi, mu, row_obs)
            ll_history.append(ll)
            if prev_ll is None or not np.isfinite(prev_ll):
                rel_ll_improve = np.inf # first iteration skip rel_ll_improve vs -inf
            else:
                rel_ll_improve = abs((ll - prev_ll) / (abs(prev_ll) + 1e-12))
            prev_ll = ll
        else:
            rel_ll_improve = 0.0
        if max(rel_W, rel_mu, rel_Psi, rel_ll_improve) < tol and it > 0:
            break
    return mu, W, Psi, Z_mean
