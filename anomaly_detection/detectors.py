from time_series_influences.utils import split_time_series, match_train_time_block_index
from time_series_influences.influence_functions import compute_loo_linear_approx
from time_series_influences.anomaly_detection import scale_influence_functions, eval_anomaly_detector, eval_anomaly_detector_all_thresholds
from sklearn.linear_model import LinearRegression, Ridge
# import periodicity_detection as pyd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.backends import cudnn
import os
import numpy as np
from ..Anomaly_Transformer.solver import Solver

def str2bool(v):
    return v.lower() in ('true')

class BaseDetector(object):

    def __init__(self):
        pass

    def calculate_anomaly_scores(self, *args):
        pass

    def evaluation(self, ground_truth, anomaly_scores, anomaly_ratio):
        detected_outliers = anomaly_scores > np.quantile(anomaly_scores, 1-anomaly_ratio)
        
        print("eval w/o point adjustment:")
        prec, rec, f1, auc = eval_anomaly_detector(ground_truth[:len(detected_outliers)], detected_outliers, anomaly_scores)
        print("eval with point adjustment:")
        prec_adj, rec_adj, f1_adj, _ = eval_anomaly_detector(ground_truth[:len(detected_outliers)], detected_outliers, anomaly_scores, adjust_detection=True)

        _, _, best_f1 = eval_anomaly_detector_all_thresholds(ground_truth[:len(detected_outliers)], anomaly_scores, adjust_detection=False, verbose=False)

class LSTMDetector(BaseDetector):
    def __init__(self):
        pass
        
    def calculate_anomaly_scores(self, channel_id):
        pass

class ARIMADetector(BaseDetector):
    def __init__(self):
        pass

    def calculate_anomaly_scores(self, channel_id):
        pass

class InfluenceFunctionDetector(BaseDetector):

    def __init__(self, config):
        super().__init__()
        self.block_length = config.win_size

    def calculate_anomaly_scores(self, channel_id):
        ts = ...
        print(f"block length is {self.block_length} time points.")
        X_train, Y_train = split_time_series(ts, block_length=self.block_length)
        matched_block_idxs = match_train_time_block_index(ts, X_train)
        
        lr = LinearRegression().fit(X_train, Y_train)
        beta = lr.coef_
        b = lr.intercept_
        inv_hess = len(X_train) * np.linalg.inv(X_train.T @ X_train)
        params = (beta, b, inv_hess)
        
        # compute IF for each time block
        time_block_loos = []
        for i in tqdm(range(len(X_train)), total=len(X_train), desc="Compute LOO"):
            time_block_loos.append(compute_loo_linear_approx(i, i, X_train, Y_train, X_train, Y_train, params))
            time_block_loos = np.array(time_block_loos)
        
        # compute IF for each time point
        time_point_loos = []
        for i in range(len(matched_block_idxs)):
            time_point_loos.append((time_block_loos[matched_block_idxs[i]]).mean())
            time_point_loos = np.array(time_point_loos)
        
        anomaly_scores = scale_influence_functions(time_point_loos, self.block_length)
        return anomaly_scores


class AnomalyTransformerDetector(BaseDetector):

    def __init__(self, config):

        super().__init__()
        cudnn.benchmark = True
        if (not os.path.exists(config.model_save_path)):
            os.makedirs(config.model_save_path)
        self.detector = Solver(vars(config))
        if not os.path.exists(os.path.join(str(self.detector.model_save_path), str(self.detector.dataset) + '_checkpoint.pth')):
           print(f'No pretrained model found for {self.detector.dataset}! Start Training ...')
           self.detector.train()


    def calculate_anomaly_scores(self, channel_ids):
        anomaly_scores = self.detector.test(channel_ids)
        return anomaly_scores
        