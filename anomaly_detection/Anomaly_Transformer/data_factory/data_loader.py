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



class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        # data = pd.read_csv(data_path + '/train.csv')
        data = pd.read_csv(data_path + '/test.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", channel_id=None, dims = None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        all_test_data =[]
        all_test_channel =[]
        all_test_labels =[]
        # data = np.load("./dataset/SMAP/SMAP_train.npy")
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
                    all_test_channel.extend([channel_no]*len(data))
                    with open(data_path + '/' + channel_no + '_test_label.pkl','rb') as f:
                        labels = pickle.load(f)
                    all_test_labels.append(labels)
        all_test_data = np.concatenate(all_test_data,axis = 0)
        self.scaler.fit(all_test_data)
        self.train = self.scaler.transform(all_test_data)
        self.test = self.scaler.transform(all_test_data)
        self.test_channels = all_test_channel
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.concatenate(all_test_labels,axis = 0)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
             return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), self.test_channels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", channel_id=None, dims = None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        all_test_data =[]
        all_test_channel =[]
        all_test_labels =[]
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
                    all_test_channel.extend([channel_no]*len(data))
                    with open(data_path + '/' + channel_no + '_test_label.pkl','rb') as f:
                        labels = pickle.load(f)
                    all_test_labels.append(labels)
        all_test_data = np.concatenate(all_test_data,axis = 0)
        self.scaler.fit(all_test_data)
        self.train = self.scaler.transform(all_test_data)
        self.test = self.scaler.transform(all_test_data)
        self.test_channels = all_test_channel
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.concatenate(all_test_labels,axis = 0)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
             return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), self.test_channels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", channel_id=None,dims = None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        # data = np.load(data_path + "/SMD_train.npy")
        all_test_data =[]
        all_test_channel =[]
        all_test_labels =[]
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
                    all_test_channel.extend([channel_no]*len(data))
                    with open(data_path + '/' + channel_no + '_test_label.pkl','rb') as f:
                        labels = pickle.load(f)
                    all_test_labels.append(labels)
        all_test_data = np.concatenate(all_test_data,axis = 0)
        self.scaler.fit(all_test_data)
        self.train = self.scaler.transform(all_test_data)
        self.test = self.scaler.transform(all_test_data)
        self.test_channels = all_test_channel
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.concatenate(all_test_labels,axis = 0)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
             return  np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), self.test_channels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]


class UCRSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        file_name = data_path.split('/')[-1]
        file_path = '/'.join(data_path.split('/')[:-1])
        file_name_list = file_name[:-4].split('_')
        train_id, anomaly_start, anomaly_end = int(file_name_list[4]),int(file_name_list[5]),int(file_name_list[6])

        all_data = pd.read_csv(Path(file_path) / file_name, delimiter='\t', header=None).to_numpy()
        data, test_data = all_data[:train_id],all_data[train_id:]
        self.scaler.fit(test_data)
        self.train = self.scaler.transform(test_data)
        self.test = self.scaler.transform(test_data)
        data_len = len(data)
        # self.val = data[(int)(data_len * 0.8):]
        self.val = self.scaler.transform(test_data)
        # self.train = data #data[:(int)(data_len * 0.8)]
        self.test_labels = np.zeros(len(self.test))
        self.test_labels[anomaly_start-train_id:anomaly_end-train_id] = np.ones(anomaly_end-anomaly_start)
        assert anomaly_start >= train_id

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'thre'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1 
        elif (self.mode == 'train_thre'):
            return (self.train.shape[0] - self.win_size) // self.win_size + 1 
        elif (self.mode == 'vali_thre'):
            return (self.val.shape[0] - self.win_size) // self.win_size + 1 

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        elif (self.mode == 'thre'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        elif (self.mode == 'train_thre'):
            return np.float32(self.train[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        elif (self.mode == 'vali_thre'):
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', channel_id = None, dims = None):
    if (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif ('UCR' in dataset):
        dataset = UCRSegLoader(data_path, win_size, 1, mode)
    elif ('SMD' in dataset):
        dataset = SMDSegLoader(data_path, win_size, step, mode, channel_id, dims)
    elif ('SMAP' in dataset):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode, channel_id, dims)
    elif ('MSL' in dataset):
        dataset = MSLSegLoader(data_path, win_size, 1, mode, channel_id, dims)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
