import gzip
import pandas as pd
import numpy as np

def load_data(dataset):
    if dataset == 'smap' or dataset == 'msl':
        data_path = f'../data/multivariate/{dataset}/{dataset}.npy'
        data = np.load(data_path)
        return data
    if dataset == 'NAB':
        data_path = f'../data/multivariate/{dataset}/{dataset}.csv'
        data = pd.read_csv(data_path)['value'].to_numpy().reshape(-1,1)
        return data
    data_path = f'../data/multivariate/{dataset}/{dataset}.txt.gz'
    with gzip.open(data_path, 'rt') as f:
        data = pd.read_csv(f, header=None)
    data = data.to_numpy()
    return data