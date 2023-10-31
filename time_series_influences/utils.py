import numpy as np

def split_time_series(series, block_length):
    "Split a univariate time series into consecutive time blocks."
    xs, ys = [], []
    for i in range(len(series) - block_length):
        x = series[i:i+block_length]
        y = series[i+block_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def match_train_time_block_index(train_series, train_block):
    "Search for time blocks that contain the target time point in the series."
    if len(train_block.shape) == 2:
        _, block_length = train_block.shape
    else:
        _, block_length, _ = train_block.shape
    
    train_block_idxs = []
    for i in range(len(train_block)):
        if i < block_length:
            block_idx = list(range(i+1))
        elif i > len(train_series) - block_length:
            delta = len(train_series) - i
            block_idx = list(range(len(train_block) - delta, len(train_block)))   
        else:
            block_idx = list(range(i-block_length+1, i+1))
        train_block_idxs.append(block_idx)
    return train_block_idxs


