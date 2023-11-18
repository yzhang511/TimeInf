import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from time_series_influences.utils import split_time_series, match_train_time_block_index
from time_series_influences.influence_functions import compute_loo_linear_approx
from time_series_influences.anomaly_detection import scale_influence_functions, eval_anomaly_detector

import os
os.chdir(r'D:\Coursework\Research\TS\time_series_anomaly_benchmark\time-series-influence')

def block_time_series(target, covariates, block_length):
    X, y, index = [], [], []
    all_sequences = np.concatenate([target,covariates],axis = 0) # target, covariates: (1,n), (k,n)
    for i in range(target.shape[1] - block_length):
        block = all_sequences[:,i:i+block_length]
        predicted_value = target[0,i+block_length]
        X.append(block)
        y.append(predicted_value)
        index.append(np.arange(i, i+block_length))
    return np.array(X), np.array(y), np.array(index)

def compute_empirical_influence_curves(x, y, X, beta, b, seq_len):
  EIC = (seq_len - 1) * np.linalg.inv(X.T @ X) @ (x * (y - x.T @ beta + b))
  return EIC

def compute_perturbed_empirical_influence_curves(x, y, X, beta, b, seq_len):
  EIC = (seq_len - 1) * np.linalg.inv(X.T @ X) * (y - 2*x.T @ beta + b)
  return EIC

def compute_loss_grad(X_test, y_test, beta, b):
  loss_grad = X_test * (y_test - X_test.T @ beta + b)
  return loss_grad

def compute_empirical_influences(x, y, x_val, y_val, X, beta, b, seq_len):
  eic = compute_empirical_influence_curves(x, y, X, beta, b, seq_len)
  loss_grad = compute_loss_grad(x_val, y_val, beta, b)
  return loss_grad @ eic

def compute_perturbed_empirical_influences(x, y, x_val, y_val, X, beta, b, seq_len):
  eic = compute_perturbed_empirical_influence_curves(x, y, X, beta, b, seq_len)
  loss_grad = compute_loss_grad(x_val, y_val, beta, b)
  return loss_grad @ eic

def compute_loo(train_idx, test_idx, X_train, y_train, X_test, y_test):

  n = len(X_train)
  x = X_train[train_idx]
  y = y_train[train_idx]
  mask = np.ones(n).astype(bool)
  mask[train_idx] = 0
  X_masked, y_masked = X_train[mask], y_train[mask]

  lr = LinearRegression().fit(X_masked, y_masked)
  beta = lr.coef_
  b = lr.intercept_

  x_val, y_val = X_test[test_idx],y_test[test_idx]

  loo = compute_empirical_influences(x, y, x_val, y_val, X_masked, beta, b, n)
  return loo

def compute_perturbed_loo(train_idx, test_idx, X_train, y_train, X_test, y_test):

  n = len(X_train)
  x = X_train[train_idx]
  y = y_train[train_idx]
  mask = np.ones(n).astype(bool)
  mask[train_idx] = 0
  X_masked, y_masked = X_train[mask], y_train[mask]

  lr = LinearRegression().fit(X_masked, y_masked)
  beta = lr.coef_
  b = lr.intercept_

  x_val, y_val = X_test[test_idx],y_test[test_idx]

  loo = compute_perturbed_empirical_influences(x, y, x_val, y_val, X_masked, beta, b, n)
  return loo


class BlockIFCalculation():
  def __init__(self,ts, ts_train,covar_train, ts_test,covar_test ,block_length, anomaly_idx, mode = 'point'):
    self.ts = ts
    self.X_train, self.y_train, self.block_index = block_time_series(ts_train, covar_train, block_length)
    self.X_test, self.y_test, _ = block_time_series(ts_test, covar_test,block_length)
    self.block_length = block_length
    self.anomaly_idx = anomaly_idx
    self.mode = mode
    print(self.X_train.shape, self.y_train.shape)
    print(self.X_test.shape, self.y_test.shape)

    train_block_index = []
    for i in range(len(self.X_train)):
      train_index = []
      for j in range(len(self.X_train)):
        if i in self.block_index[j]:
          train_index.append(j)
      train_block_index.append(train_index)
    self.train_block_index = train_block_index

  def train(self):
    lr = LinearRegression().fit(self.X_train.reshape(self.X_train.shape[0],-1), self.y_train)
    y_hat = lr.predict(self.X_train.reshape(self.X_train.shape[0],-1))
    y_pred = lr.predict(self.X_test.reshape(self.X_test.shape[0],-1))
    plt.figure(figsize=(16,3))
    train_pred = np.ones_like(self.ts) * np.nan
    test_pred = train_pred.copy()
    train_pred[self.block_length:self.block_length + len(y_hat)] = y_hat
    test_pred[-len(y_pred):] = y_pred
    plt.plot(self.ts, c="gray", linewidth=1)
    plt.plot(train_pred, c="g", linewidth=1, linestyle="--")
    plt.plot(test_pred, c="r", linewidth=1, linestyle="--")
    plt.show()

  def calculate_if(self):
  
    train_loos = []
    for i in tqdm(range(len(self.X_train)), total=len(self.X_train), desc="Compute LOO"):
      test_loos = []
      for j in np.arange(len(self.X_test)):
        test_loos.append(compute_loo(i, j,self.X_train.reshape(self.X_train.shape[0],-1), self.y_train,self.X_test.reshape(self.X_test.shape[0],-1), self.y_test))
      train_loos.append(np.mean(test_loos))
    train_loos = np.array(train_loos)

    mean_loos = []
    for i in range(len(self.train_block_index)):
      mean_loos.append((train_loos[self.train_block_index[i]]).mean())
    mean_loos = np.array(mean_loos)
    plt.figure(figsize=(16,3))
    loos_viz = np.ones_like(self.ts) * np.nan
    loos_viz[:len(self.X_train)] = mean_loos
    plt.plot(loos_viz)
    plt.axhline(y=0, c="r", linestyle="--")
    if self.mode == 'point':
      plt.scatter(self.anomaly_idx, [loos_viz[i] for i in self.anomaly_idx], color='red', label='anomaly', marker="x")
    else:
      plt.axvspan(self.anomaly_idx[0], self.anomaly_idx[-1], facecolor='red', alpha=.2)
    plt.show()
    return mean_loos

  def calculate_if_training(self):
    train_loos = []
    for i in tqdm(range(len(self.X_train)), total=len(self.X_train), desc="Compute LOO"):
      train_loos.append(compute_loo(i, i, self.X_train.reshape(self.X_train.shape[0],-1), self.y_train, self.X_train.reshape(self.X_train.shape[0],-1), self.y_train))
    train_loos = np.array(train_loos)

    mean_loos = []
    for i in range(len(self.train_block_index)):
      mean_loos.append((train_loos[self.train_block_index[i]]).mean())
    mean_loos = np.array(mean_loos)

    plt.figure(figsize=(16,3))
    loos_viz = np.ones_like(self.ts) * np.nan
    loos_viz[:len(self.X_train)] = mean_loos
    plt.plot(loos_viz)
    plt.axhline(y=0, c="r", linestyle="--")
    if self.mode == 'point':
      plt.scatter(self.anomaly_idx, [loos_viz[i] for i in self.anomaly_idx], color='red', label='anomaly', marker="x")
    else:
      plt.axvspan(self.anomaly_idx[0], self.anomaly_idx[-1], facecolor='red', alpha=.2)    
    plt.show()
    return mean_loos
  
  def calculate_perturbed_if(self):
    train_loos = []
    for i in tqdm(range(len(self.X_train)), total=len(self.X_train), desc="Compute LOO"):
      test_loos = []
      for j in np.arange(len(self.X_test)):
        test_loos.append(compute_loo(i, j,self.X_train.reshape(self.X_train.shape[0],-1), self.y_train,self.X_test.reshape(self.X_test.shape[0],-1), self.y_test))
      train_loos.append(np.mean(test_loos))
    train_loos = np.array(train_loos)

    mean_loos = []
    for i in range(len(self.train_block_index)):
      mean_loos.append((train_loos[self.train_block_index[i]]).mean())
    mean_loos = np.array(mean_loos)

    plt.figure(figsize=(16,3))
    loos_viz = np.ones_like(self.ts) * np.nan
    loos_viz[:len(self.X_train)] = mean_loos
    plt.plot(loos_viz)
    plt.axhline(y=0, c="r", linestyle="--")
    if self.mode == 'point':
      plt.scatter(self.anomaly_idx, [loos_viz[i] for i in self.anomaly_idx], color='red', label='anomaly', marker="x")
    else:
      plt.axvspan(self.anomaly_idx[0], self.anomaly_idx[-1], facecolor='red', alpha=.2)    
    plt.show()
    return mean_loos
  
  def calculate_perturbed_if_training(self):
    train_loos = []
    for i in tqdm(range(len(self.X_train)), total=len(self.X_train), desc="Compute LOO"):
      train_loos.append(compute_perturbed_loo(i, i, self.X_train.reshape(self.X_train.shape[0],-1), self.y_train, self.X_train.reshape(self.X_train.shape[0],-1), self.y_train))
    train_loos = np.array(train_loos)

    mean_loos = []
    for i in range(len(self.train_block_index)):
      mean_loos.append((train_loos[self.train_block_index[i]]).mean())
    mean_loos = np.array(mean_loos)

    plt.figure(figsize=(16,3))
    loos_viz = np.ones_like(self.ts) * np.nan
    loos_viz[:len(self.X_train)] = mean_loos
    plt.plot(loos_viz)
    plt.axhline(y=0, c="r", linestyle="--")
    if self.mode == 'point':
      plt.scatter(self.anomaly_idx, [loos_viz[i] for i in self.anomaly_idx], color='red', label='anomaly', marker="x")
    else:
      plt.axvspan(self.anomaly_idx[0], self.anomaly_idx[-1], facecolor='red', alpha=.2)    
    plt.show()
    return mean_loos
  

data_path = r'.\dataset\SMD'
ts = np.load(data_path + "/SMD_train.npy")
ts_test = np.load(data_path + "/SMD_test.npy")
test_labels = np.load(data_path + "/SMD_test_label.npy")
anomaly_idx = np.where(test_labels==1)[0] + len(ts)


block_length = 100
X_train, Y_train = split_time_series(ts, block_length=block_length)
print(X_train.shape, Y_train.shape)

task1 =  BlockIFCalculation(ts[:,0], ts[:,0:1].T,ts[:,1:].T, ts_test[:,0:1].T,ts_test[:,1:].T ,100,anomaly_idx)
task1.train()
# if1 = task1.calculate_if()
if1_train = task1.calculate_if_training()
# if1_perturbed_train = task1.calculate_perturbed_if_training()