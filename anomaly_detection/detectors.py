from time_series_influences.utils import split_time_series, match_train_time_block_index
from time_series_influences.influence_functions import compute_loo_linear_approx
from time_series_influences.anomaly_detection import scale_influence_functions, eval_anomaly_detector, eval_anomaly_detector_all_thresholds
from time_series_influences.nonparametric_influences import compute_nonparametric_influences
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from tqdm import tqdm
import periodicity_detection as pyd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.backends import cudnn
from Anomaly_Transformer.solver import Solver as AnomTransSolver
from LSTM.solver import Solver as LSTMSolver
from pmdarima import auto_arima

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch_influence import BaseObjective, CGInfluenceModule
torch.set_default_dtype(torch.double)


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

        # Count data points in each cluster
        # cluster_counts = np.bincount(labels)

        # # Find the ratio of the minority cluster
        # anomaly_ratio = np.min(cluster_counts) / len(anomaly_scores)
        anomaly_ratio = len(guess_index) / len(anomaly_scores)

        return anomaly_ratio

    def evaluate(self, ground_truth, anomaly_scores, anomaly_ratio = None):
        if anomaly_ratio is None:
            anomaly_ratio = self.auto_anomaly_detection(anomaly_scores)
        thres = np.quantile(anomaly_scores, 1-anomaly_ratio)
        detected_outliers = anomaly_scores > thres
        print('Threshold:{:.2f}'.format(thres))
        
        print("eval w/o point adjustment:")
        prec, rec, f1, auc = eval_anomaly_detector(ground_truth[:len(detected_outliers)], detected_outliers, anomaly_scores)
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


class NonparametricInfluenceFunctionDetector(BaseDetector):

    def __init__(self, config):
        super().__init__()
        self.block_length = config.win_size
        self.learner = config.learner
        self.loss_function = config.loss_function
        self.n_subsets = config.n_subsets
        self.subset_frac = config.subset_frac

    def calculate_anomaly_scores(self, ts, *args, **kwargs):
        print(f"block length is {self.block_length} time points.")
        X_train, Y_train = split_time_series(ts, block_length=self.block_length)
        
        matched_block_idxs = match_train_time_block_index(ts, X_train)

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

        subset_size = int(self.subset_frac * len(ts))
            
        # compute nonparametric IF for each time block        
        time_block_loos = compute_nonparametric_influences(
            X_train, 
            Y_train.squeeze(), 
            self.n_subsets, 
            subset_size, 
            learner, 
            loss_function
        )
        
        # compute IF for each time point
        time_point_loos = []
        for i in range(len(matched_block_idxs)):
            time_point_loos.append((time_block_loos[matched_block_idxs[i]]).mean())
        time_point_loos = np.array(time_point_loos)
        
        anomaly_scores = scale_influence_functions(time_point_loos, self.block_length)
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
        print(f"block length is {self.block_length} time points.")

        if len(ts.shape) > 1:
            seq_len, n_dim = ts.shape
            if n_dim > 1:
                raise Exception("Sorry, black-box models for multivariate data is still in progress.")
            ts = ts.squeeze()
            
        if self.device == "gpu":
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            DEVICE = torch.device("cpu")
        print(DEVICE)

        X_train, Y_train = create_dataset(ts, block_length=self.block_length, device=DEVICE)
        X_train, Y_train = X_train[:,:,None], Y_train.squeeze()
        
        matched_block_idxs = match_train_time_block_index(ts, X_train)

        model = Forecaster(
            model_type=self.black_box_model, 
            hidden_size=self.hidden_size, 
            n_layers=self.n_layers
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
                
            # if epoch % 10 != 0:
            #     continue
                
            model.eval()
            with torch.no_grad():
                Y_pred = model(X_train)
                train_rmse = np.sqrt(loss_fn(Y_pred, Y_train).cpu().numpy())
            print("Epoch %d: train RMSE %.4f" % (epoch, train_rmse))
        
        # -- influence functions
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
        time_block_loos = module.influences(train_idxs=all_train_idxs, test_idxs=all_train_idxs)
        
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
        if (not os.path.exists(config.model_save_path)):
            os.makedirs(config.model_save_path)
        # self.detector = AnomTransSolver(vars(config))
        # if not os.path.exists(os.path.join(str(self.detector.model_save_path), str(self.detector.dataset) + '_checkpoint.pth')):
        #    print(f'No pretrained model found for {self.detector.dataset}! Start Training ...')
        #    self.detector.train()
        self.config = config


    def calculate_anomaly_scores(self, channel_id, *args, **kwargs):
        self.config.channel_id = channel_id
        cudnn.benchmark = True
        detector = AnomTransSolver(vars(self.config))
        detector.train()
        anomaly_scores = detector.test(channel_id)
        return anomaly_scores


# helper functions for black-box influences

def create_dataset(dataset, block_length, device=None):
    X, Y = [], []
    for i in range(len(dataset)-block_length):
        x = dataset[i:i+block_length]
        y = dataset[i+1:i+block_length+1]
        X.append(x)
        Y.append(y)
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
        
