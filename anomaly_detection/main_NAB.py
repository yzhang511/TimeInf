import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import detectors
import os
import time
import argparse
from os import listdir
import json

def parse_int_list(s):
    try:
        return [int(dim) for dim in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("dims must be a comma-separated list of integers")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='../data_processed/SMAP')
    parser.add_argument('--model_save_path', type=str, default='../Anomaly_Transformer/checkpoints')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--dimensions', type=parse_int_list, default = None,
                        help='A comma-separated list of dimensions (e.g., "1,2,3")')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--detector_type', type=str, required=True, default='InfluenceFunctionDetector',
                    help='Type of the detector to use')
    parser.add_argument('--use_anomaly_ratio', action='store_true', default=False,
                    help='Whether the true anomaly ratio is used for detection or not.')
    # To Do: model-specific params
    parser.add_argument('--lstm_n_predictions', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    # params for non-parametric influences
    parser.add_argument('--n_subsets', type=int, default=1000)
    parser.add_argument('--subset_frac', type=float, default=0.7)
    parser.add_argument('--learner', type=str, default='GradientBoosting',
                        choices=['GradientBoosting', 'RandomForest', 'LinearRegression',
                                 'KNN', 'SVR'])
    parser.add_argument('--loss_function', type=str, default='mean_squared_error')

    # params for black box influences
    parser.add_argument('--black_box_model', type=str, default='LSTM',
                        choices=['LSTM', 'RNN'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=8)


    config = parser.parse_args()

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    args = vars(config)

    if hasattr(detectors, config.detector_type):
        DetectorClass = getattr(detectors, config.detector_type)
        if callable(DetectorClass):
            detector = DetectorClass(config)
        else:
            raise TypeError(f"{config.detector_type} is not a callable class")
    else:
        raise ValueError(f"Detector type {config.detector_type} not found in detectors module")


    dataset = config.dataset
    data_path = Path("./data_processed/NAB/") / dataset
    test_file = "./data_processed/NAB/labels/combined_labels.json"
    with open(test_file, 'r') as file:
        test_df = json.load(file)
    

    len_test_dict, len_anomaly_dict, len_ratio_dict, len_detected_ratio_dict = {}, {}, {}, {}
    prec_dict, rec_dict, f1_dict, auc_dict, best_f1_dict = {}, {}, {}, {}, {}
    prec_adj_dict, rec_adj_dict, f1_adj_dict = {}, {}, {}
    time_dict = {}
    
    # get file list
    file_names = [Path(f).stem for f in listdir(data_path)]


    
    for file in file_names:

        ts_test = pd.read_csv(data_path/f"{file}.csv")
        seq_len = len(ts_test)
        
        ts_test["ground_truth"] = 0
        ts_test.set_index("timestamp",inplace = True)
        
        
        anomaly_seqs = test_df.get(f"{dataset}/{file}.csv")
        for anom in anomaly_seqs:
            ts_test.loc[anom, "ground_truth"]=1.
        
        ground_truth = ts_test["ground_truth"].to_numpy()
        ts_test.reset_index(inplace = True)
        ts_test = ts_test.drop(columns = ["ground_truth","timestamp"])
        ts_test = ts_test.to_numpy()
        
        
        anomaly_len = sum(ground_truth)
        anomaly_ratio = anomaly_len / seq_len
        print(f"anomaly ratio is {anomaly_ratio * 100.:.3f} %.")

        len_test_dict.update({file: seq_len})
        len_anomaly_dict.update({file: anomaly_len})
        len_ratio_dict.update({file: anomaly_ratio})

        print(f"start detection for file {file} ..")
        start_time = time.time()
        anomaly_scores = detector.calculate_anomaly_scores(ts = ts_test, channel_id = file, contamination = min(anomaly_ratio,0.5)) # contamination has to be inthe range (0.0,0.5]
        detected_anomaly_ratio = detector.auto_anomaly_detection(anomaly_scores)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        if config.use_anomaly_ratio:
            prec, rec, f1, auc, prec_adj, rec_adj, f1_adj,  best_f1 = detector.evaluate(ground_truth, anomaly_scores, anomaly_ratio) 
        else:
            prec, rec, f1, auc, prec_adj, rec_adj, f1_adj,  best_f1 = detector.evaluate(ground_truth, anomaly_scores, detected_anomaly_ratio) 

        len_detected_ratio_dict.update({file:detected_anomaly_ratio})
        prec_dict.update({file: prec})
        rec_dict.update({file: rec})
        f1_dict.update({file: f1})
        auc_dict.update({file: auc})
        best_f1_dict.update({file: best_f1})

        time_dict.update({file: elapsed_time})

        prec_adj_dict.update({file: prec_adj})
        rec_adj_dict.update({file: rec_adj})
        f1_adj_dict.update({file: f1_adj})
        
        anom_df=pd.DataFrame()
        anom_df["anom_socres"]=anomaly_scores
        anom_df["ground_truth"]=ground_truth[-len(anomaly_scores):]
        anom_df.to_csv(os.path.join(config.result_path, f"{file}_"+config.detector_type+"_anom_scores.csv"))


    nab_metrics = pd.DataFrame({
        "Num_of_Test": len_test_dict,
        "Len_of_Anomaly": len_anomaly_dict,
        "True_Anomaly_Ratio": len_ratio_dict,
        "Predicted_Anomaly_Ratio":len_detected_ratio_dict,
        "Precision(w.o. Adjustment)": prec_dict,
        "Recall(w.o. Adjustment)": rec_dict,
        "F1(w.o. Adjustment)": f1_dict,
        "Precision(w. Adjustment)": prec_adj_dict,
        "Recall(w. Adjustment)": rec_adj_dict,
        "F1(w. Adjustment)": f1_adj_dict,
        "Best_F1_Score": best_f1_dict,
        "AUC": auc_dict,
        'Elapsed Time(s)': time_dict
    })
    
    nab_metrics.insert(0, "Dataset", nab_metrics.index)
    nab_metrics.reset_index(drop = True, inplace = True)
    
    nab_metrics.to_csv(os.path.join(config.result_path, config.dataset+"_"+config.detector_type+"_results.csv"))
    
    
