import numpy as np


def get_indices_in_held_out_fold(T, pct_to_drop, contiguous=False):
    """
    :param T: length of the sequence
    :param pct_to_drop: % of T in the held out fold
    :param contiguous: if True generate a block of indices to drop else generate indices by iid sampling
    :return: o (the set of indices in the fold)
    """
    if contiguous:
        l = np.floor(pct_to_drop / 100. * T)
        anchor = np.random.choice(np.arange(l + 1, T))
        o = np.arange(anchor - l, anchor).astype(int)
    else:
        # i.i.d LWCV
        o = np.random.choice(T - 2, size=int(pct_to_drop / 100. * T), replace=False) + 1
    return o


def genSyntheticDataset(K, T, N, D, sigma0=None, seed=1234, varainces_of_mean=1.0,
                        diagonal_upweight=False):
    np.random.seed(seed)
    if sigma0 is None:
        sigma0 = np.eye(D)

    A = np.random.dirichlet(alpha=np.ones(K), size=K)
    if diagonal_upweight:
        A = A + 5 * np.eye(K)  # add 5 to the diagonal and renormalize
        A = A / A.sum(axis=1)

    pi0 = np.random.dirichlet(alpha=np.ones(K))
    mus = np.random.normal(size=(K, D), scale=np.sqrt(varainces_of_mean))
    zs = np.empty((N, T), dtype=np.int64)
    X = np.empty((N, T, D))

    for n in range(N):
        zs[n, 0] = int(np.random.choice(np.arange(K), p=pi0))
        X[n, 0] = np.random.multivariate_normal(mean=mus[zs[n, 0]], cov=sigma0)
        for t in range(1, T):
            zs[n, t] = int(np.random.choice(np.arange(K), p=A[zs[n, t - 1], :]))
            X[n, t] = np.random.multivariate_normal(mean=mus[zs[n, t]], cov=sigma0)

    return X, zs, A, pi0, mus

# Potts.py helpers -----------------------------------------------------------------
def save_adjacency_mat(path):
    file = np.load(path)
    W = np.asarray(file['w'], dtype=int)
    np.savetxt("toy_adj_mat.csv", W, fmt="%d",delimiter=",")
    return 

def extract_folds(path):
    """
    Load data from path, make full data and the leave-one-out folds.
    Also return the number of folds.  
    """
    file = np.load(path)

    data = {}
    # full data
    data['full'] = {}
    y = file['counts']
    W = np.asarray(file['w'], dtype=int)
    data['full']['y'] = y
    data['full']['W'] = W

    # leave-one-out folds
    N = len(y)
    for i in range(N):
        data[i] = {}
        newy = np.delete(y, i)
        # indexer[j] = name of vertex j in the leave-out-fold
        # in the original data
        indexer = {}
        for j in range(N - 1):
            if (j < i):
                indexer[j] = j
            else:
                indexer[j] = j + 1
        newW = np.zeros((N - 1, N - 1))
        for j in range(N - 1):
            for t in range(N - 1):
                if W[indexer[j], indexer[t]] == 1:
                    newW[j, t] = 1
        data[i]['y'] = newy
        data[i]['W'] = newW
    return data, N


def genMRFdatafolds(adj_path, lam_hi, lam_lo, seed):
    """
    Inputs:
        adj_path: str, typicall "toy_adj_mat.csv"
        lam_hi: scalar, mean of lower class
        lam_lo: scalar, mean of higher class
        seed: scalar, random seed for replicability
    Outputs:
    """
    data = {}
    
    # load adjacency matrix 
    W = np.genfromtxt(adj_path, delimiter=',')
    N = W.shape[0]
    
    # generate random class assignments 
    np.random.seed(seed)
    z = np.random.binomial(1, 0.5, size=N)
    
    # generate counts
    y = np.empty(N, dtype = np.int64)
    for i in range(N):
        if (z[i] == 1):
            tmp_lam = lam_hi
        if(z[i] == 0): 
            tmp_lam = lam_lo
        y[i] = np.random.poisson(lam = tmp_lam)
    
    # make folds
    data['full'] = {}
    data['full']['y'] = y
    data['full']['W'] = W
    
    # leave-one-out folds
    for i in range(N):
        data[i] = {}
        newy = np.delete(y,i)
        # indexer[j] = name of vertex j in the leave-out-fold
        # in the original data
        indexer = {}
        for j in range(N-1):
            if (j < i):
                indexer[j] = j
            else:
                indexer[j] = j+1
        newW = np.zeros((N-1,N-1))
        for j in range(N-1):
            for t in range(N-1):
                if W[indexer[j],indexer[t]] == 1:
                    newW[j,t] = 1
        data[i]['y'] = newy
        data[i]['W'] = newW
    return data, N

def genSyntheticCounts(z, lam_hi, lam_lo):
    N = z.size
    counts = np.empty(N, dtype = np.int64)
    for i in range(N):
        if (z[i] == 1):
            tmp_lam = lam_hi
        if(z[i] == 0): 
            tmp_lam = lam_lo
        counts[i] = np.random.poisson(lam = tmp_lam)
    data = {}
    data['counts'] = counts
    return data

