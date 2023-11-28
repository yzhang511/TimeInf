import numpy as np
from sklearn.linear_model import LinearRegression
from time_series_influences.utils import split_time_series, match_train_time_block_index

def empirical_IC_prop(x, y, X, beta, b, inv_hess):
    if len(beta.shape) == 2:
        n_series = beta.shape[0]
        block_length = beta.shape[1] // n_series
        grad = x.reshape(n_series, block_length) * (y - x.T @ beta.T - b).reshape(-1,1)
        eic = - (1/len(X)) * grad.flatten()
    else:
        eic = - (1/len(X)) * (x * (y - x.T @ beta - b))
    return eic

def rhat(k, n, IF):
    return sum([IF[t]*IF[t+abs(k)] for t in range(1, n-abs(k))]) / n

w_TH = lambda x: (1 + np.cos(np.pi * x)) / 2 if abs(x) <= 1 else 0

def w_SC(x):
    if abs(x) <= 0.8:
        return 1
    elif (abs(x) <= 1) & (abs(x) > 0.8):
        return (1 + np.cos(5*np.pi * (x-0.8))) / 2
    else:
        return 0

def b_i(i, b, n, IF):
    assert i in [1, 2, 3, 4]
    num = sum([rhat(k, n, IF)**2 for k in range(1-n, n-1)])
    denom = 6 * sum([(w_SC(k*b[i-1]*np.power(n, 4/21))**2)*(k**2)*(rhat(k, n, IF)**2) for k in range(1-n, n-1)])
    # print(num, denom)
    return np.power(n, -1/3) * np.power(num/denom, 1/3)

def calc_b(b4, n, IF):
    num = 2 * sum([w_TH(k*b4*np.power(n, 4/21))*rhat(k, n, IF) for k in range(1-n, n-1)]) ** 2
    denom = 3 * sum([w_SC(k*b4*np.power(n, 4/21))*abs(k)*rhat(k, n, IF) for k in range(1-n, n-1)]) ** 2
    # print(num, denom)
    return np.power(n, -1/3) * np.power(num/denom, 1/3)

def compute_optimal_block_length(ts, start_point, end_point):
    X_train, Y_train = split_time_series(ts, block_length=1)
    matched_block_idxs = match_train_time_block_index(ts, X_train)
    
    lr = LinearRegression().fit(X_train, Y_train)
    beta = lr.coef_
    b = lr.intercept_
    inv_hess = len(X_train) * np.linalg.inv(X_train.T @ X_train)
    params = (beta, b, inv_hess)

    # compute IF for each time block
    time_block_loos = []
    for i in range(len(X_train)):
      time_block_loos.append(empirical_IC_prop(X_train[i], Y_train[i], X_train, beta, b, inv_hess))
    time_block_loos = np.array(time_block_loos).squeeze()

    point_IF = time_block_loos[start_point:end_point]
    n = len(point_IF)
    b0 = 1/n
    b1 = b_i(1, [b0], n, point_IF)
    b2 = b_i(2, [b0, b1], n, point_IF)
    b3 = b_i(3, [b0, b1, b2], n, point_IF)
    b4 = b_i(4, [b0, b1, b2, b3], n, point_IF)
    block_length = int(1/calc_b(b4, n, point_IF))
    print("The optimal block length is: ", block_length)
    return block_length

    






    