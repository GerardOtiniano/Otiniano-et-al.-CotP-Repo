import numpy as np

def normalize_with_mode(X_orig, ages, mode, ref_min=None, ref_max=None):
    """
    Normalize proxy records for FA.
    center: Subtract nanmean. Mean is computed over ref_min/max interval (full record if None)
    z_score: Subtract mean (over reference interval) and divide by record-specific std
    """

    X = np.asarray(X_orig, float)
    ages = np.asarray(ages, float)
    N, D = X.shape

    # reference period
    if ref_min is not None and ref_max is not None:
        mask = (ages >= ref_min) & (ages <= ref_max)
    else:
        mask = np.isfinite(ages)

    # center
    if mode.lower() == "center":
        # Compute means over reference period
        a = np.nanmean(X[mask, :], axis=0)
        a = np.where(np.isfinite(a), a, 0.0)

        b = np.ones(D)  # no scaling
        Xp = X - a[None, :]
        return Xp, a, b

    # z-score
    elif mode.lower() == "z_score":
        a = np.nanmean(X[mask, :], axis=0)
        a = np.where(np.isfinite(a), a, 0.0)
        std = np.nanstd(X[mask, :] - a[None, :], axis=0)
        std = np.where((std > 0) & np.isfinite(std), std, 1.0)
        Xp = (X - a[None, :]) /std[None, :]
        return Xp, a, std

    else:
        raise ValueError("mode must be 'center' or 'z_score'")
