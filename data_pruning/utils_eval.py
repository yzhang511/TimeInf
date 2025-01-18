import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from time_series_influences.utils import split_time_series

def moving_average_imputation(ts, ind, window_size = 30):
    start = max(ind-window_size,0)
    if start == ind:
        return ts[start,:]
    imputed_value = np.nanmean(ts[start:ind,:])
    return imputed_value


def block_removal_experiment(value_dict, X, y, X_test, y_test,predictor='lr'):
    removal_ascending_dict, removal_descending_dict=dict(), dict()
    for key in value_dict.keys():
        removal_ascending_dict[key]=block_removal_core(X, y, X_test, y_test, value_dict[key], ascending=True, predictor=predictor)
        removal_descending_dict[key]=block_removal_core(X, y, X_test, y_test, value_dict[key], ascending=False, predictor=predictor)
    return {'ascending':removal_ascending_dict, 'descending':removal_descending_dict}

def block_removal_core(X, y, X_test, y_test, value_list, ascending=True, predictor='lr'):
    n_sample=len(X)
    if ascending is True:
        sorted_value_list=np.argsort(value_list) # ascending order. low to high.
    else:
        sorted_value_list=np.argsort(value_list)[::-1] # descending order. high to low.
    
    accuracy_dict={'r2':[],'mse':[]}
    n_period = min(n_sample//200, 2) 
    for percentile in tqdm(range(0, 100, n_period)):
        '''
        We repeatedly remove 5% of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        '''
        start_index = int(n_sample * percentile / 100)
        sorted_value_list_tmp=sorted_value_list[start_index:]
        if predictor == 'lr':
            try:
                model=LinearRegression()
                model.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred) 
                r2 = r2_score(y_test, y_pred)
            except:
                mse,r2 = 0,0
        
        accuracy_dict['mse'].append(mse)
        accuracy_dict['r2'].append(r2)

    return accuracy_dict


def point_removal_experiment(value_dict, X, X_test, y_test, block_length=30, predictor='lr'):
    removal_ascending_dict, removal_descending_dict=dict(), dict()
    for key in value_dict.keys():
        removal_ascending_dict[key]=point_removal_core(X.copy(), X_test, y_test, block_length, value_dict[key], ascending=True, predictor=predictor)
        removal_descending_dict[key]=point_removal_core(X.copy(), X_test, y_test, block_length, value_dict[key], ascending=False, predictor=predictor)
    return {'ascending':removal_ascending_dict, 'descending':removal_descending_dict}

def point_removal_core(ts_train, X_test, y_test, block_length, value_list, ascending=True, predictor='lr'):
    n_sample=max(len(ts_train) - block_length,0)
    if ascending is True:
        sorted_value_list=np.argsort(value_list) # ascending order. low to high.
    else:
        sorted_value_list=np.argsort(value_list)[::-1] # descending order. high to low.
    
    accuracy_dict={'r2':[],'mse':[]}
    start_index = 0
    n_period = min(n_sample//200, 5) 
    for percentile in tqdm(range(0, 100, n_period)):
        '''
        We repeatedly remove 5% of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        '''
        end_index = int(n_sample * percentile / 100) + 1
        imputed_index_tmp = sorted_value_list[:start_index]
        for ind in imputed_index_tmp:
            ts_train[ind] = moving_average_imputation(ts_train, ind)
        
        X_train, Y_train = split_time_series(ts_train, block_length=block_length)
        n_dim = X_train.shape[-1]
        X_train = X_train.reshape((-1, block_length*n_dim))
        Y_train = Y_train.squeeze()

        if predictor == 'lr':
            try:
                model=LinearRegression()
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred) 
                r2 = r2_score(y_test, y_pred)
            except:
                mse,r2 = 0,0
        
        accuracy_dict['mse'].append(mse)
        accuracy_dict['r2'].append(r2)
        start_index = end_index

    return accuracy_dict
