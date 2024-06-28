import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


def shape_data(arr, l_s, n_predictions):
    """
    Shape raw input streams for ingestion into LSTM and create a PyTorch DataLoader.

    Args:
        arr (np array): Array of input streams with dimensions [timesteps, 1, input dimensions].
        l_s (int): Sequence length of prior timesteps fed into the model at each timestep t.
        n_predictions (int): Number of future steps to predict.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data (useful for training).
        train (bool): If shaping training data, this indicates data can be shuffled.
    """

    data = []
    for i in range(len(arr) - l_s - n_predictions + 1): # NOTE
        data.append(arr[i:i + l_s + n_predictions])
    data = np.array(data)

    assert len(data.shape) == 3

    X = data[:, :-n_predictions, :]
    y = data[:, -n_predictions:, :]

    return X, y

def get_loader_segment(data_path, batch_size, win_size=100, n_predictions=10, mode='train', dataset='KDD', channel_id = None, dims = None):
    all_test_data =[]
    # all_test_channel =[]
    for filename in os.listdir(data_path):
        if 'test.pkl' in filename:
            channel_no = filename.split('_')[0]
            if channel_id is None or channel_no == channel_id:
                with open(data_path + '/' + filename,'rb') as f:
                    data = pickle.load(f)
                if dims is not None:
                    all_test_data.append(data[:,dims])
                else:
                    all_test_data.append(data)
                # all_test_channel.extend([channel_no]*len(data))
    all_test_data = np.concatenate(all_test_data,axis = 0)
    
    
    X, y = shape_data(all_test_data, win_size, n_predictions)
    

    shuffle = False
    if mode == 'train':
        shuffle = True
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, shuffle=True)
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        valid_X = torch.tensor(valid_X, dtype=torch.float32)
        valid_y = torch.tensor(valid_y, dtype=torch.float32)
        train_dataset = TensorDataset(train_X, train_y)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_dataset = TensorDataset(valid_X, valid_y)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_data_loader, valid_data_loader
    else:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        test_dataset = TensorDataset(X, y)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_data_loader



