import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())

def scale_influence(inf, blk_len):
    """Scale influences to produce anomaly scores."""
    
    scaled_inf = min_max_scaler(inf)
    # Hack: Early time points appear in fewer time blocks and have less reliable influences. 
    scaled_inf[:blk_len] = np.nanmean(scaled_inf)
    return np.abs(scaled_inf - np.nanmean(scaled_inf))

def eval_anomaly_detector(
    gt, pred, anomaly_scores, verbose=True
):
    """Evaluate time series anomaly detector."""

    prec, rec, f1, support = precision_recall_fscore_support(gt, pred, average='binary')
    auc = roc_auc_score(gt, anomaly_scores)
        
    if verbose:
        print(f"Precision: {prec:.2f} Recall: {rec:.2f} F1: {f1:.2f} AUC: {auc:.2f}")
        
    return prec, rec, f1, auc

    