import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())

def scale_influence_functions(influences):
    "Scale computed influence functions for anomaly detection."
    scaled_influences = min_max_scaler(influences)
    return np.abs(scaled_influences - np.nanmean(scaled_influences))

def eval_anomaly_detector(ground_truth, model_pred, verbose=True):
    "Evaluate time series anomaly detectors."
    prec = precision_score(ground_truth, model_pred)
    rec = recall_score(ground_truth, model_pred)
    f1 = f1_score(ground_truth, model_pred)
    if verbose:
        print(f"precision: {prec:.3f} recall: {rec:.3f} F1: {f1:.3f}")
    return prec, rec, f1
    