from time_series_influences.utils import split_time_series, match_train_time_block_index
from time_series_influences.influence_functions import compute_loo_linear_approx
from time_series_influences.anomaly_detection import scale_influence_functions, eval_anomaly_detector, eval_anomaly_detector_all_thresholds
from sklearn.linear_model import LinearRegression, Ridge
# import periodicity_detection as pyd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from torch.backends import cudnn
import os
import numpy as np
from Anomaly_Transformer.solver import Solver as AnomTransSolver
from LSTM.solver import Solver as LSTMSolver
from pmdarima import auto_arima



def str2bool(v):
    return v.lower() in ('true')

class BaseDetector(object):

    def __init__(self):
        pass

    def calculate_anomaly_scores(self, *args, **kwargs):
        pass

    def evaluate(self, ground_truth, anomaly_scores, anomaly_ratio):
        detected_outliers = anomaly_scores > np.quantile(anomaly_scores, 1-anomaly_ratio)
        
        print("eval w/o point adjustment:")
        prec, rec, f1, auc = eval_anomaly_detector(ground_truth[:len(detected_outliers)], detected_outliers, anomaly_scores) # be careful! truncate the data by default
        print("eval with point adjustment:")
        prec_adj, rec_adj, f1_adj, _ = eval_anomaly_detector(ground_truth[:len(detected_outliers)], detected_outliers, anomaly_scores, adjust_detection=True)

        _, _, best_f1 = eval_anomaly_detector_all_thresholds(ground_truth[:len(detected_outliers)], anomaly_scores, adjust_detection=False, verbose=False)

        return prec, rec, f1, auc, prec_adj, rec_adj, f1_adj,  best_f1

class LSTMDetector(BaseDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def calculate_anomaly_scores(self,ts, channel_id, *args, **kwargs):
        self.config.channel_id = channel_id
        self.detector = LSTMSolver(vars(self.config))
        self.detector.train()
        anomaly_scores = self.detector.test()
        assert len(anomaly_scores) == len(ts)
        return anomaly_scores

class ARIMADetector(BaseDetector):
    def __init__(self, config):
        super().__init__()
        self.random_seed = config.seed

    def calculate_anomaly_scores(self, ts, *args, **kwargs):
        # TO DO: check the parameters & multivariate models
        ts = ts.reshape(-1)
        model = auto_arima(ts, max_p=10, max_q=10, 
                   seasonal=True, trace=False,
                   error_action='ignore', suppress_warnings=True,
                   stepwise=True)

        print(model.summary())

        preds = model.predict_in_sample()
        anomaly_scores = (preds - ts)**2
        return anomaly_scores


class IForestDetector(BaseDetector):
    def __init__(self, config):
        super().__init__()
        self.random_seed = config.seed

    def calculate_anomaly_scores(self, ts, contamination='auto', *args, **kwargs):
        clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=self.random_seed)
        clf.fit(ts)   
        decision_scores = clf.decision_function(ts) # the lower, the more abnormal
        anomaly_scores = -decision_scores
        return anomaly_scores
        

class InfluenceFunctionDetector(BaseDetector):

    def __init__(self, config):
        super().__init__()
        self.block_length = config.win_size

    def calculate_anomaly_scores(self, ts, *args, **kwargs):
        print(f"block length is {self.block_length} time points.")
        X_train, Y_train = split_time_series(ts, block_length=self.block_length)
        
        matched_block_idxs = match_train_time_block_index(ts, X_train)

        if len(ts.shape) > 1:
            seq_len, n_dim = ts.shape
            X_train = X_train.reshape((-1, self.block_length*n_dim))
        
        lr = LinearRegression().fit(X_train, Y_train)
        beta = lr.coef_
        b = lr.intercept_
        try:
            inv_hess = len(X_train) * np.linalg.inv(X_train.T @ X_train)
        except:
            inv_hess = len(X_train) * np.linalg.pinv(X_train.T @ X_train)
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
        self.detector = AnomTransSolver(vars(config))
        if not os.path.exists(os.path.join(str(self.detector.model_save_path), str(self.detector.dataset) + '_checkpoint.pth')):
           print(f'No pretrained model found for {self.detector.dataset}! Start Training ...')
           self.detector.train()


    def calculate_anomaly_scores(self, channel_id, *args, **kwargs):
        anomaly_scores = self.detector.test(channel_id)
        return anomaly_scores
        
