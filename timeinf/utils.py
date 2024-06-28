import numpy as np

def block_time_series(ts, blk_len):
    """Split time series into consecutive overlapping time blocks."""
    
    num_blk = len(ts)-blk_len
    xs = np.array([ts[b:b+blk_len] for b in range(num_blk)])
    ys = np.array([ts[b+blk_len] for b in range(num_blk)])
    
    return xs, ys


def sync_time_block_index(train_ts, train_blk):
    """Match time block index to the original time series index."""
    
    num_blk = train_blk.shape[0]
    blk_len = train_blk.shape[1]
    
    train_blk_idxs = []
    for b in range(num_blk):
        if b < blk_len:
            blk_idx = list(range(b+1))
        elif b > len(train_ts)-blk_len:
            delta = len(train_ts)-b
            blk_idx = list(range(len(train_blk)-delta, num_blk))   
        else:
            blk_idx = list(range(b-blk_len+1, b+1))
        train_blk_idxs.append(blk_idx)
        
    return train_blk_idxs


