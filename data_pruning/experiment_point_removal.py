import numpy as np
from time_series_influences.utils import split_time_series, match_train_time_block_index
from time_series_influences.influence_functions import compute_loo_linear_approx
from tqdm import tqdm
from dataset import load_data
from utils_eval import point_removal_experiment
import time
import pickle
import argparse

from opendataval.dataloader import DataFetcher
from opendataval.dataval import (
    DataBanzhaf,
    DataOob,
    DataShapley,
    KNNShapley,
    LavaEvaluator,
    LeaveOneOut,
    RandomEvaluator,
)


from opendataval.dataloader import DataFetcher
from sklearn import tree
from opendataval.model import RegressionSkLearnWrapper
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def main(dataset, block_length, N = 10):
    data = load_data(dataset)
    all_res = []
    all_values = []
    all_time = []
    try:
        random_state = 42
        rng = np.random.default_rng(random_state)
        indices = rng.choice(data.shape[1], N, replace=False) 
    # try:
    #     indices = np.random.choice(data.shape[1], N, replace=False) 
    except:
        indices = np.arange(0,data.shape[1])
    
    # start =  data.shape[0]-5000 #np.random.choice(data.shape[0]-5000,replace = False)
    start =  np.random.choice(data.shape[0]-5000,replace = False)
    for i in tqdm(indices): 

        ts = data[start:start+5000,i:i+1]
        train_size = 3000
        valid_size = 1000
        test_size = 1000
        train_ts = ts[:train_size,:]
        val_ts = ts[train_size:train_size + valid_size,:]
        test_ts = ts[train_size + valid_size:,:]
        print(f'Train size:{train_size:d}, Valid size:{valid_size:d}, Test size:{test_size:d}')

        X_train, Y_train = split_time_series(train_ts, block_length=block_length)
        X_val, Y_val = split_time_series(val_ts, block_length=block_length)
        X_test, Y_test = split_time_series(test_ts, block_length=block_length)

        print(X_train.shape, Y_train.shape)
        print(X_val.shape, Y_val.shape)
        print(X_test.shape, Y_test.shape)

        seq_len, n_dim = train_ts.shape
        X_train = X_train.reshape((-1, block_length*n_dim))
        X_val = X_val.reshape((-1, block_length*n_dim))
        X_test = X_test.reshape((-1, block_length*n_dim))
        Y_train,Y_val,Y_test = Y_train.squeeze(),Y_val.squeeze(),Y_test.squeeze()
        matched_block_idxs =match_train_time_block_index(train_ts, X_train)

        lr = LinearRegression().fit(X_train, Y_train)
        beta = lr.coef_
        b = lr.intercept_
        try:
            inv_hess = len(X_train) * np.linalg.inv(X_train.T @ X_train)
        except:
            inv_hess = len(X_train) * np.linalg.pinv(X_train.T @ X_train)
        params = (beta, b, inv_hess)

        start_time = time.time()
        time_block_loos = []
        for i in tqdm(range(len(X_train)), total=len(X_train), desc="Compute LOO"):
            val_influences = 0
            for j in range(len(X_val)):
                val_influences += compute_loo_linear_approx(i, j, X_train, Y_train, X_val, Y_val, params)
            time_block_loos.append(val_influences / len(X_val))
        time_block_loos = np.array(time_block_loos)

        time_point_loos = []
        for i in range(len(matched_block_idxs)):
            time_point_loos.append((time_block_loos[matched_block_idxs[i]]).mean())
        time_point_loos = np.array(time_point_loos)

        if_evaluator_time = time.time() -  start_time
        print(f"InfluenceFunction computation time: {if_evaluator_time} seconds")
        print(f"InfluenceFunction Computation Done!")

        pred_model = RegressionSkLearnWrapper(LinearRegression)

        fetcher = DataFetcher.from_data_splits(X_train, Y_train, X_val, Y_val, X_test, Y_test, one_hot=False)

        # RandomEvaluator
        start_time = time.time()
        random_evaluator = RandomEvaluator().train(fetcher=fetcher)
        random_values = random_evaluator.data_values
        time_point_random_values = []
        for i in range(len(matched_block_idxs)):
            time_point_random_values.append((random_values[matched_block_idxs[i]]).mean())
        time_point_random_values = np.array(time_point_random_values)
        random_evaluator_time = time.time() - start_time
        print(f"RandomEvaluator computation time: {random_evaluator_time} seconds")
        print(f"Random Computation Done!")

        # KNNShapley
        start_time = time.time()
        knn_shapley = KNNShapley(k_neighbors=0.1 * len(X_train)).train(fetcher=fetcher)
        knn_shapley_values = knn_shapley.data_values
        time_point_knn_shapley_values = []
        for i in range(len(matched_block_idxs)):
            time_point_knn_shapley_values.append((knn_shapley_values[matched_block_idxs[i]]).mean())
        time_point_knn_shapley_values = np.array(time_point_knn_shapley_values)
        
        knn_shapley_time = time.time() - start_time
        print(f"KNNShapley computation time: {knn_shapley_time} seconds")
        print("KNNShapley Computation Done!")

        # DataShapley
        start_time = time.time()
        data_shapley = DataShapley(gr_threshold=1.10, max_mc_epochs=100, min_models=100).train(fetcher=fetcher, pred_model=pred_model)
        data_shapley_values = data_shapley.data_values

        time_point_data_shapley_values = []
        for i in range(len(matched_block_idxs)):
            time_point_data_shapley_values.append((data_shapley_values[matched_block_idxs[i]]).mean())
        time_point_data_shapley_values = np.array(time_point_data_shapley_values)

        data_shapley_time = time.time() - start_time
        print(f"Data Shapley computation time: {data_shapley_time} seconds")
        print("Data Shapley Computation Done!")

        # DataOob
        start_time = time.time()
        data_oob = DataOob(num_models=1000).train(fetcher=fetcher, pred_model=pred_model)
        oob_values = data_oob.data_values

        time_point_oob_values = []
        for i in range(len(matched_block_idxs)):
            time_point_oob_values.append((oob_values[matched_block_idxs[i]]).mean())
        time_point_oob_values = np.array(time_point_oob_values)

        data_oob_time = time.time() - start_time
        print(f"DataOob computation time: {data_oob_time} seconds") 
        print("DataOob Computation Done!")

        value_dict = {"InfluenceFunction":time_point_loos,
        'DataShapley':time_point_data_shapley_values,
        "Random":time_point_random_values,
        # "LAVA":lava_values,
        "OOB":time_point_oob_values,
        "KNNShapley":time_point_knn_shapley_values,
                        #   "LOO Values":loo_values
                        }
        results = point_removal_experiment(value_dict, train_ts.copy(), X_test, Y_test, block_length = block_length)
        all_res.append(results)
        all_values.append(value_dict)
        all_time.append([if_evaluator_time, data_shapley_time, random_evaluator_time,knn_shapley_time,data_oob_time])
    print('Elapsed Time: ', np.mean(np.array(all_time),axis=0))
    return all_res, all_values
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run point removel experiment on a dataset.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset to analyze.')
    parser.add_argument('--block_length', type=int, help='The length of sliding windows.')
    parser.add_argument('--result_path', type=str, help='The file path to save results.')
    parser.add_argument('--max_num_series', type=int, default = 10, help='The max number of time series.')
    args = parser.parse_args()

    final_results, final_values= main(args.dataset, args.block_length, args.max_num_series)

    with open(args.result_path + f'results.pkl','wb') as f:
        pickle.dump(final_results,f)
    with open(args.result_path + f'data_values.pkl','wb') as f:
        pickle.dump(final_values,f)
