import numpy as np

def empirical_IC_linear_approx(x, y, X, beta, b):
    """
    Compute empirical influence curves for model parameters with linear approximation.
    No leave-one-out retraining is needed. 
    """
    eic = (1/len(X)) * np.linalg.inv(X.T @ X) @ (x * (y - x.T @ beta - b))
    return eic

def compute_loss_grad(x_val, y_val, beta, b):
    "Compute gradient of the validation loss."
    loss_grad = x_val * (y_val - x_val.T @ beta - b)
    return loss_grad

def empirical_IF_linear_approx(x, y, x_val, y_val, X, params):
    """
    Compute empirical influence functions for model prediction with linear approximation.
    No leave-one-out retraining is needed.
    """
    beta, b = params
    eic = empirical_IC_linear_approx(x, y, X, beta, b)
    loss_grad = compute_loss_grad(x_val, y_val, beta, b)
    return loss_grad @ eic
    
def compute_loo_linear_approx(train_idx, val_idx, X_train, Y_train, X_val, Y_val, params):
    """
    Compute empirical LOO with linear approximation.
    No model retraining is needed.
    """
    x, y = X_train[train_idx], Y_train[train_idx]
    x_val, y_val = X_val[val_idx], Y_val[val_idx]
    loo = empirical_IF_linear_approx(x, y, x_val, y_val, X_train, params)
    return loo


