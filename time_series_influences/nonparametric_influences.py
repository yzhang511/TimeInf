import numpy as np
from tqdm import tqdm

def compute_nonparametric_influences(X, Y, t, m, learner, metric):
    """
    t: number of subsets / learners
    m: number of data points in each subset
    learner: sklearn's regressor
    metric: regression metrics
    """
    # sample t random subsets of [n] of size m
    n = len(X)
    subsets = [np.random.choice(np.arange(n), m) for k in range(t)]

    # train t learners
    learners, preds = [], []
    for k in tqdm(range(t)):
        learner.fit(X[subsets[k]], Y[subsets[k]])
        learners.append(learner)
        Yhat = learner.predict(X)
        preds.append(Yhat)

    # compute self-influence
    mem_lst = []
    for i in tqdm(range(n)):
        is_in = np.array([i in subsets[k] for k in range(t)])
        arg_in = np.where(is_in == 1)[0].astype(int)
        arg_out = np.where(is_in == 0)[0].astype(int)
        preds_in = [preds[k][i] for k in arg_in]
        preds_out = [preds[k][i] for k in arg_out]
        metric_in = metric(np.ones_like(preds_in)*Y[i], preds_in)
        metric_out = metric(np.ones_like(preds_out)*Y[i], preds_out)
        mem = metric_in - metric_out
        mem_lst.append(mem)
    mem_lst = np.array(mem_lst)

    return mem_lst
    