import os
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch_influence import BaseObjective, CGInfluenceModule
torch.set_default_dtype(torch.double)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# from pmdarima import auto_arima
from LSTM.solver import Solver as LSTMSolver
from AnomalyTransformer.solver import Solver as AnomTransSolver

from timeinf.utils import block_time_series, sync_time_block_index
from timeinf.linear_influence import calc_linear_time_inf
from timeinf.anomaly_detection import scale_influence, eval_anomaly_detector
from timeinf.nonparametric_influence import calc_nonparametric_timeinf


def str2bool(v):
    return v.lower() in ('true')


class BaseDetector(object):

    def __init__(self):
        pass

    def calculate_anomaly_scores(self, *args, **kwargs):
        pass

    def auto_anomaly_detection(self, anomaly_scores):
    
        kmeans = KMeans(n_clusters=2, random_state=0).fit(anomaly_scores.reshape(-1, 1))
        guess_index=np.where(kmeans.labels_ == np.argmax(kmeans.cluster_centers_))[0]
        anomaly_ratio = len(guess_index) / len(anomaly_scores)

        return anomaly_ratio

    def evaluate(self, ground_truth, anomaly_scores, anomaly_ratio = None):
        
        if anomaly_ratio is None:
            anomaly_ratio = self.auto_anomaly_detection(anomaly_scores)
        thres = np.quantile(anomaly_scores, 1-anomaly_ratio)
        detected_outliers = anomaly_scores > thres
        print('Threshold:{:.2f}'.format(thres))
        
        prec, rec, f1, auc = eval_anomaly_detector(
            ground_truth[:len(detected_outliers)], detected_outliers, anomaly_scores
        )
        return prec, rec, f1, auc


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

        
# class ARIMADetector(BaseDetector):
#     def __init__(self, config):
#         super().__init__()
#         self.random_seed = config.seed

#     def calculate_anomaly_scores(self, ts, *args, **kwargs):
#         ts = ts.reshape(-1)
#         model = auto_arima(ts, max_p=10, max_q=10, 
#                    seasonal=True, trace=False,
#                    error_action='ignore', suppress_warnings=True,
#                    stepwise=True)

#         print(model.summary())

#         preds = model.predict_in_sample()
#         anomaly_scores = (preds - ts)**2
#         return anomaly_scores


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
        print(f"Block length: {self.block_length}")
        X_train, Y_train = block_time_series(ts, self.block_length)
        
        synced_blk_idxs = sync_time_block_index(ts, X_train)

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

        # compute influence for each time block
        block_infs = []
        for i in tqdm(range(len(X_train)), total=len(X_train), desc="Compute block influence"):
            block_infs.append(calc_linear_time_inf(i, i, X_train, Y_train, X_train, Y_train, params))
        block_infs = np.array(block_infs)
        
        # compute influence for each time point
        time_infs = []
        for i in tqdm(range(len(synced_blk_idxs)), total=len(synced_blk_idxs), desc="Compute TimeInf"):
            time_infs.append((block_infs[synced_blk_idxs[i]]).mean())
        time_infs = np.array(time_infs)
        
        anomaly_scores = scale_influence(time_infs, self.block_length)
        return anomaly_scores


class NonparametricInfluenceFunctionDetector(BaseDetector):

    def __init__(self, config):
        super().__init__()
        self.block_length = config.win_size
        self.learner = config.learner
        self.loss_function = config.loss_function
        self.n_subsets = config.n_subsets
        self.subset_frac = config.subset_frac

    def calculate_anomaly_scores(self, ts, *args, **kwargs):
        print(f"Block length: {self.block_length}")
        X_train, Y_train = block_time_series(ts, self.block_length)
        
        synced_blk_idxs = sync_time_block_index(ts, X_train)

        if len(ts.shape) > 1:
            seq_len, n_dim = ts.shape
            X_train = X_train.reshape((-1, self.block_length*n_dim))

        if self.loss_function == "mean_squared_error":
            loss_function = mean_squared_error

        if self.learner == "GradientBoosting":
            learner = GradientBoostingRegressor()
        elif self.learner == "LinearRegression":
            learner = LinearRegression()
        elif self.learner == "RandomForest":
            learner = RandomForestRegressor()
        elif self.learner == "KNN":
            learner = KNeighborsRegressor()
        elif self.learner == "SVR":
            learner = SVR()
        else:
            raise Exception(f"Learner not implemented yet.")

        subset_size = int(self.subset_frac * len(ts))
            
        # compute nonparametric influence for each time block        
        block_infs = calc_nonparametric_timeinf(
            X_train, Y_train.squeeze(), self.n_subsets, subset_size, learner, loss_function
        )
        
        # compute nonparametric influence for each time point
        time_infs = []
        for i in tqdm(range(len(synced_blk_idxs)), total=len(synced_blk_idxs), desc="Compute TimeInf"):
            time_infs.append((block_infs[synced_blk_idxs[i]]).mean())
        time_infs = np.array(time_infs)
        
        anomaly_scores = scale_influence(time_infs, self.block_length)
        
        return anomaly_scores


class BlackBoxInfluenceFunctionDetector(BaseDetector):

    def __init__(self, config):
        super().__init__()
        self.block_length = config.win_size
        self.black_box_model = config.black_box_model
        self.device = config.device
        self.lr = config.lr
        self.n_epochs = config.num_epochs
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        self.n_layers = config.n_layers
        self.hidden_size = config.hidden_size

    def calculate_anomaly_scores(self, ts, *args, **kwargs):
        
        print(f"Block length: {self.block_length}")

        if len(ts.shape) > 1:
            seq_len, n_dim = ts.shape
            if n_dim > 1:
                raise Exception("Black-box TimeInf for multivariate data not implemented yet.")
            ts = ts.squeeze()
            
        if self.device == "gpu":
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            DEVICE = torch.device("cpu")
        print('Device: ', DEVICE)

        X_train, Y_train = create_dataset(ts, block_length=self.block_length, device=DEVICE)
        X_train, Y_train = X_train[:,:,None], Y_train.squeeze()
        
        synced_blk_idxs = sync_time_block_index(ts, X_train)

        model = Forecaster(
            model_type=self.black_box_model, hidden_size=self.hidden_size, n_layers=self.n_layers
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        loader = data.DataLoader(
            data.TensorDataset(X_train, Y_train), shuffle=True, batch_size=self.batch_size
        )

        # model training
        for epoch in range(self.n_epochs):
            model.train()
            for X_batch, Y_batch in loader:
                Y_pred = model(X_batch)
                loss = loss_fn(Y_pred, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                Y_pred = model(X_train)
                train_rmse = np.sqrt(loss_fn(Y_pred, Y_train).cpu().numpy())
            print("Epoch %d: train RMSE %.4f" % (epoch, train_rmse))
        
        # compute black-box influence for each time block
        train_set = data.TensorDataset(X_train, Y_train)
        
        module = CGInfluenceModule(
            model=model,
            objective=TimeSeriesObjective(),
            train_loader=data.DataLoader(train_set, batch_size=self.batch_size),
            test_loader=data.DataLoader(train_set, batch_size=self.batch_size),
            device=DEVICE,
            damp=0.001,
            atol=1e-8,
            maxiter=self.n_epochs,
        )

        all_train_idxs = list(range(X_train.shape[0]))
        block_infs = module.influences(train_idxs=all_train_idxs, test_idxs=all_train_idxs)
        
        # compute black-box influence for each time point
        time_infs = []
        for i in tqdm(range(len(synced_blk_idxs)), total=len(synced_blk_idxs), desc="Compute TimeInf"):
            time_infs.append((block_infs[synced_blk_idxs[i]]).mean())
        time_infs = np.array(time_infs)
        
        anomaly_scores = scale_influence(time_infs, self.block_length)
        return anomaly_scores


class AnomalyTransformerDetector(BaseDetector):

    def __init__(self, config):

        super().__init__()
        if (not os.path.exists(config.model_save_path)):
            os.makedirs(config.model_save_path)
        self.config = config

    def calculate_anomaly_scores(self, channel_id, *args, **kwargs):
        self.config.channel_id = channel_id
        cudnn.benchmark = True
        detector = AnomTransSolver(vars(self.config))
        detector.train()
        anomaly_scores = detector.test(channel_id)
        return anomaly_scores


# ----------------
# helper functions 
# ----------------


def create_dataset(dataset, block_length, device=None):
    X, Y = [], []
    for i in range(len(dataset)-block_length):
        X.append(dataset[i:i+block_length])
        Y.append(dataset[i+1:i+block_length+1])
    if device is not None:
        return torch.tensor(X).to(device), torch.tensor(Y).to(device)
    else:
        return np.array(X), np.array(Y)

        
class Forecaster(nn.Module):
    def __init__(self, model_type, hidden_size, n_layers):
        super().__init__()
        if model_type == "LSTM":
            self.forecaster = nn.LSTM(
                input_size=1, 
                hidden_size=hidden_size, 
                num_layers=n_layers, 
                batch_first=True
            )
        else:
            self.forecaster  = nn.RNN(
                input_size=1, 
                hidden_size=hidden_size, 
                num_layers=n_layers, 
                batch_first=True
            ) 
        self.linear = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten(start_dim=-2)
        
    def forward(self, x):
        x, _ = self.forecaster(x)
        x = self.linear(x)
        x = self.flatten(x)
        return x


class TimeSeriesObjective(BaseObjective):

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.mse_loss(outputs, batch[1])

    def train_regularization(self, params):
        return 1e-4 * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        outputs = model(batch[0])
        return F.mse_loss(outputs, batch[1])
        
