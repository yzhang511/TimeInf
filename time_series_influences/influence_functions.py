import numpy as np

def empirical_IC_linear_approx(x, y, X, beta, b, inv_hess, remove_cov_idxs=[]):
    """
    Compute empirical influence curves for model parameters with linear approximation.
    No leave-one-out retraining is needed. 
    """
    if len(beta.shape) == 2:
        n_series = beta.shape[0]
        block_length = beta.shape[1] // n_series
        grad = x.reshape(n_series, block_length) * (y - x.T @ beta.T - b).reshape(-1,1)
        eic = - (1/len(X)) * inv_hess @ grad.flatten()
    else:
        eic = - (1/len(X)) * inv_hess @ (x * (y - x.T @ beta - b))

    if len(remove_cov_idxs) != 0:
        mask = np.ones_like(x).astype(bool)
        mask[remove_cov_idxs] = 0
        u, inv_hess = x[mask], inv_hess[mask].T[mask]
        grad = u * (y - x.T @ beta.T - b)
        eic = - (1/len(X)) * inv_hess @ grad
    return eic

def compute_loss_grad(x_val, y_val, beta, b, remove_cov_idxs=[]):
    "Compute gradient of the validation loss."
    if len(beta.shape) == 2:
        n_series = beta.shape[0]
        block_length = beta.shape[1] // n_series
        loss_grad = - x_val.reshape(n_series, block_length) * (y_val - x_val.T @ beta.T - b).reshape(-1,1)
    else:
        loss_grad = - x_val * (y_val - x_val.T @ beta - b)

    if len(remove_cov_idxs) != 0:
        mask = np.ones_like(x_val).astype(bool)
        mask[remove_cov_idxs] = 0
        loss_grad = - x_val[mask] * (y_val - x_val.T @ beta - b)
    return loss_grad.flatten()

def empirical_IF_linear_approx(x, y, x_val, y_val, X, params, remove_cov_idxs=[]):
    """
    Compute empirical influence functions for model prediction with linear approximation.
    No leave-one-out retraining is needed.
    """
    beta, b, inv_hess = params
    eic = empirical_IC_linear_approx(x, y, X, beta, b, inv_hess, remove_cov_idxs)
    loss_grad = compute_loss_grad(x_val, y_val, beta, b, remove_cov_idxs)
    return loss_grad @ eic
    
def compute_loo_linear_approx(train_idx, val_idx, X_train, Y_train, X_val, Y_val, params, remove_cov_idxs=[]):
    """
    Compute empirical LOO with linear approximation.
    No model retraining is needed.
    """
    x, y = X_train[train_idx], Y_train[train_idx]
    x_val, y_val = X_val[val_idx], Y_val[val_idx]
    loo = empirical_IF_linear_approx(x, y, x_val, y_val, X_train, params, remove_cov_idxs)
    return loo
    

