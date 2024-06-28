import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import time
from LSTM.model.LSTM import LSTMModel
from LSTM.data_factory.data_loader import get_loader_segment
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error as mse_score

result_path = './results/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def aggregate_predictions(predictions, n_predictions):
    aggregated = torch.zeros((len(predictions)+n_predictions - 1,predictions.shape[2]))  # (batch_size, n_predictions, dimension)
    counts = torch.zeros_like(aggregated)
    for i, pred in enumerate(predictions):
        for j in range(n_predictions):
            if i + j < len(aggregated):
                aggregated[i + j] += pred[j, :]
                counts[i + j] += 1
    aggregated /= counts
    return aggregated


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0003):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader, self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               n_predictions = self.lstm_n_predictions,
                                               mode='train',
                                               dataset=self.dataset,
                                               dims = self.dimensions,
                                               channel_id = self.channel_id)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        set_seed(self.seed)
        self.model = LSTMModel(input_dim=self.input_c, n_predictions = self.lstm_n_predictions, hidden_dims=[80,80], dropout=self.dropout) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {num_params}")

    def vali(self, vali_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in vali_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(vali_loader)

    def train(self):

        print("======================TRAIN MODE======================")
        early_stopping = EarlyStopping(patience=10)
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_loss = self.vali(self.vali_loader)
            if self.verbose:
                print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(self.train_loader)}, Validation Loss: {val_loss}")
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        if not os.path.exists(result_path +str(self.dataset)):
            os.mkdir(result_path +str(self.dataset))

    def test(self):
        self.test_loader = get_loader_segment(self.data_path, batch_size=1, win_size=self.win_size,
                                              n_predictions = self.lstm_n_predictions,
                                              mode='test',
                                              dataset=self.dataset,
                                              dims = self.dimensions,
                                              channel_id = self.channel_id)
        self.model.eval()

        print("======================TEST MODE======================")

        all_predictions = []
        all_targets = []
        for i, (input_data, targets) in enumerate(self.test_loader):
            input_data = input_data.float().to(self.device)
            targets = targets.float().to(self.device)
            with torch.no_grad():
                predictions = self.model(input_data)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu()[:,0,:])
        for j in range(1,targets.size(1)):
            all_targets.append(targets.cpu()[-1:,j,:])
        all_predictions = torch.cat(all_predictions,dim=0)
        all_targets = torch.cat(all_targets,dim=0)
        aggregated_predictions = aggregate_predictions(all_predictions, self.lstm_n_predictions)
        test_scores =  torch.zeros(self.win_size + len(aggregated_predictions))
        test_scores[self.win_size:] = torch.mean((aggregated_predictions - all_targets)**2,dim = -1)
        return test_scores

