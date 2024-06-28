import numpy as np
from typing import Tuple


def calc_param_influence(x, y, X, beta, b, inv_hess):
    """Calculate empirical influences for model parameters."""
    
    is_univar = True if len(beta.shape) == 1 else False
    
    if is_univar:
        param_inf = - (1/len(X)) * inv_hess @ (x * (y - x.T @ beta - b))
    else:
        n_blk = beta.shape[0]
        blk_len = beta.shape[1]//n_blk
        grad = x.reshape(n_blk, blk_len) * (y - x.T @ beta.T - b).reshape(-1,1)
        param_inf = - (1/len(X)) * inv_hess @ grad.flatten()

    return param_inf


def calc_loss_grad(x_test, y_test, beta, b):
    """Calculate test loss gradient."""

    is_univar = True if len(beta.shape) == 1 else False

    if is_univar:
        loss_grad = - x_test * (y_test - x_test.T@beta - b)
    else:
        n_blk = beta.shape[0]
        blk_len = beta.shape[1]//n_blk
        loss_grad = - x_test.reshape(n_blk, blk_len) * (y_test - x_test.T@beta.T - b).reshape(-1,1)
    
    return loss_grad.flatten()


def calc_block_influence(x, y, x_test, y_test, X, params):
    """Calculate empirical influences for model predictions."""
    
    beta, b, inv_hess = params
    param_inf = calc_param_influence(x, y, X, beta, b, inv_hess)
    loss_grad = calc_loss_grad(x_test, y_test, beta, b)
    blk_inf = loss_grad @ param_inf
    
    return blk_inf


def calc_linear_time_inf(
    train_idx, test_idx, X_train, Y_train, X_test, Y_test, params
):
    """Calculate TimeInf using linear AR models."""
    
    x, y = X_train[train_idx], Y_train[train_idx]
    x_test, y_test = X_test[test_idx], Y_test[test_idx]
    time_inf = calc_block_influence(x, y, x_test, y_test, X_train, params)
    
    return time_inf
    

