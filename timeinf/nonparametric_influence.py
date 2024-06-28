import numpy as np
from tqdm import tqdm

def calc_nonparametric_timeinf(
    X, Y, n_subsets, subset_size, learner, metric
):
    """Calculate self-influence nonparametrically.  
    
    Args:
        n_subsets: number of subsets (learners)
        subset_size: number of data points in each subset
        learner: sklearn's regressor, e.g., sklearn.ensemble.GradientBoostingRegressor
        metric: sklearn's regression metric, e.g., sklearn.metrics.mean_squared_error

    Returns:
        inf_lst: list of self-influence
    """
    
    # sample random subsets from the dataset 
    n_blk = len(X)
    subsets = [np.random.choice(np.arange(n_blk), subset_size) for _ in range(n_subsets)]

    # fit learners to the sampled subsets
    learners, preds = [], []
    for subset_idx in tqdm(range(n_subsets)):
        learner.fit(X[subsets[subset_idx]], Y[subsets[subset_idx]])
        learners.append(learner)
        preds.append(learner.predict(X))

    # influence computation
    inf_lst = []
    for blk_idx in tqdm(range(n_blk)):
        is_in = np.array([blk_idx in subsets[subset_idx] for subset_idx in range(n_subsets)])
        arg_in, arg_out = np.where(is_in == 1)[0].astype(int), np.where(is_in == 0)[0].astype(int)
        preds_in = [preds[subset_idx][blk_idx] for subset_idx in arg_in]
        preds_out = [preds[subset_idx][blk_idx] for subset_idx in arg_out]
        metric_in = metric(np.ones_like(preds_in)*Y[blk_idx], preds_in)
        metric_out = metric(np.ones_like(preds_out)*Y[blk_idx], preds_out)
        inf_lst.append(metric_in - metric_out)
    inf_lst = np.array(inf_lst)

    return inf_lst
    