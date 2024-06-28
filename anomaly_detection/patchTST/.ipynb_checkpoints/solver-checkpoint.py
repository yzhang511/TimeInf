import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from tqdm import tqdm
import logging

from transformers import PatchTSTConfig, PatchTSTForPrediction

path_root = '..'
sys.path.append(str(path_root))

from detectors import create_dataset
from timeinf.utils import sync_time_block_index

torch.set_default_dtype(torch.double)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()   


        self.patch_tst_config = PatchTSTConfig(
            num_input_channels=self.n_dim,         
            context_length=self.win_size,       # context length of the input sequence
            patch_length=5,                     # each patch is a token
            prediction_length=1,                # prediction horizon 
            num_hidden_layers=self.num_hidden_layers,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.ffn_dim,
            attention_dropout=0.,
            dropout=self.dropout,
            random_mask_ratio=0.1,
            num_targets=1,
        )
        print("-- Config --")
        print(self.patch_tst_config)
        self.build_model()

    def prepare_data(self,ts):
        
        seq_len, n_dim = ts.shape

        scaler = StandardScaler().fit(ts)
        ts = scaler.transform(ts)
        ts = ts.squeeze()

        ts_train, ts_test = ts[:int(len(ts)*.9)], ts[int(len(ts)*.9):]
        X_train, Y_train = create_dataset(ts_train, block_length=self.win_size, device=self.device)
        self.synced_block_idxs = sync_time_block_index(ts_train, X_train)
        X_train, Y_train = X_train[:,:,None], Y_train[:,-1,None,None]
        X_test, Y_test = create_dataset(ts_test, block_length=self.win_size, device=self.device)
        X_test, Y_test = X_test[:,:,None], Y_test[:,-1,None,None]

        self.train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, Y_train), shuffle=True, batch_size=1
        )
        self.detect_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, Y_train), shuffle=False, batch_size=1
        )
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train), shuffle=False, batch_size=1
        )
    
    def build_model(self):
        self.model = PatchTSTForPrediction(self.patch_tst_config).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0)
        print('learning_rate: ',self.learning_rate)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {num_params}")


    def train(self):
        self.weights = []
        self.train_loss_history = []
        self.val_loss_history = []
        early_stopping = EarlyStopping(patience=8)

        for epoch in range(1, self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                # inputs: (bs, block_len, n_dim)
                # preds:  (bs, forecast_len, n_dim)
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()

                outputs = self.model(
                    past_values=inputs,
                    future_values=targets,
                )
                preds = outputs.prediction_outputs
                loss = nn.MSELoss()(targets, preds)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
        
            if epoch % 2 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path,f'model_v1_epoch_{epoch}'))
                self.weights.append(f'model_v1_epoch_{epoch}')

            val_loss = self.eval()
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            epoch_loss = running_loss / len(self.train_loader)
            self.train_loss_history.append(epoch_loss)
            self.val_loss_history.append(val_loss)

            log_message = f"Epoch {epoch}, Training Loss: {epoch_loss}, Validation Loss: {val_loss}"
            print(log_message)
            self.logger.info(log_message)
        
        self.plot_and_save_loss_curves()
        print('Finished Training!')

    def eval(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(
                    past_values=inputs,
                    future_values=targets,
                )
                preds = outputs.prediction_outputs
                loss = nn.MSELoss()(targets, preds)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def calculate_self_tracin(self,
        weights_paths: [str or Path],
    ):

        LR = self.learning_rate

        score_matrix = np.zeros((len(self.detect_loader)))

        for train_id, (x_train, y_train) in tqdm(enumerate(self.detect_loader), total=len(self.detect_loader), desc='Train'):
            grad_sum = 0

            counter = 0
            for w in weights_paths:
                counter += 1
                if counter > 2:
                    break
                if w not in os.listdir(self.checkpoint_path):
                    continue
                model = PatchTSTForPrediction(self.patch_tst_config).to(device)
                model.load_state_dict(torch.load(os.path.join(self.checkpoint_path,w), map_location=device)) # checkpoint
                model.eval()
                inputs, targets = x_train.to(device), y_train.to(device)
                outputs = model(
                    past_values=inputs,
                    future_values=targets,
                )
                preds = outputs.prediction_outputs
                loss = nn.MSELoss()(targets, preds)
                loss.backward() # back
                train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters() if param.grad is not None])

                grad_sum += LR * np.dot(train_grad.cpu(), train_grad.cpu()) # scalar mult, TracIn formula

            score_matrix[train_id] = grad_sum

        time_point_loos = []
        for i in range(len(self.synced_block_idxs)):
            time_point_loos.append((score_matrix[self.synced_block_idxs[i]]).mean())
        time_point_loos = np.array(time_point_loos)

        return time_point_loos


    def calculate_prediction_error(self):
        test_targets, test_preds = [], []
        for i, data in enumerate(self.detect_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = self.model(
                    past_values=inputs,
                    future_values=targets,
                )
                preds = outputs.prediction_outputs
            test_targets.append(targets)
            test_preds.append(preds)
        test_targets = torch.concat(test_targets).squeeze().detach().cpu().numpy()
        test_preds = torch.concat(test_preds).squeeze().detach().cpu().numpy()
        test_scores =  np.zeros(self.win_size + len(test_preds))
        test_scores[self.win_size:] = np.mean((test_targets - test_preds)**2,axis = -1)
        return test_scores

    
    def plot_and_save_loss_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.checkpoint_path, 'loss_curves.png')
        plt.savefig(plot_path)
        plt.close()

    def setup_logging(self):
        log_filename = os.path.join(self.checkpoint_path, "training_log.txt")
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        self.logger = logging.getLogger()

