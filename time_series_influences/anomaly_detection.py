import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())

def scale_influence_functions(influences, block_length):
    "Scale computed influence functions for anomaly detection."
    scaled_influences = min_max_scaler(influences)
    # TO DO: initial time points are contained in fewer time blocks,
    #        so that their influences are unreliable 
    scaled_influences[:block_length] = np.nanmean(scaled_influences)
    return np.abs(scaled_influences - np.nanmean(scaled_influences))

def eval_anomaly_detector(ground_truth, model_pred, verbose=True, adjust_detection=False):
    "Evaluate time series anomaly detectors."
    
    if adjust_detection:
        anomaly_state = False
        for i in range(len(ground_truth)):
            if ground_truth[i] == 1 and model_pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if ground_truth[j] == 0:
                        break
                    else:
                        if model_pred[j] == 0:
                            model_pred[j] = 1
                for j in range(i, len(ground_truth)):
                    if ground_truth[j] == 0:
                        break
                    else:
                        if model_pred[j] == 0:
                            model_pred[j] = 1
            elif ground_truth[i] == 0:
                anomaly_state = False
            if anomaly_state:
                model_pred[i] = 1       
        model_pred = np.array(model_pred)
        ground_truth = np.array(ground_truth)

    prec, rec, f1, support = precision_recall_fscore_support(
        ground_truth, model_pred, average='binary'
    )
        
    if verbose:
        print(f"precision: {prec:.3f} recall: {rec:.3f} F1: {f1:.3f}")
    return prec, rec, f1


def eval_anomaly_detector_all_thresholds(ground_truth, anomaly_scores, verbose=True, adjust_detection=False):

    contam_ratio = np.linspace(0.001, 0.1, 100)

    best_prec, best_rec, best_f1 = 0., 0., 0.
    for ratio in contam_ratio:
        detected_outliers = anomaly_scores > np.quantile(anomaly_scores, 1-ratio)
        prec, rec, f1 = eval_anomaly_detector(
            ground_truth, detected_outliers, verbose=False, adjust_detection=adjust_detection
        )
        if f1 > best_f1:
            best_prec = prec
            best_rec = rec
            best_f1 = f1
    if verbose:
        print(f"precision: {best_prec:.3f} recall: {best_rec:.3f} F1: {best_f1:.3f}")
    return best_prec, best_rec, best_f1

    