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


    dataset = "SMAP_MSL"
    data_path = Path("../data/multivariate/") / dataset
    test_df = pd.read_csv(data_path/"labeled_anomalies.csv")
    df = test_df.loc[test_df.spacecraft ==  config.dataset]
    df = df .loc[df .chan_id != "P-2"]

    len_test_dict, len_anomaly_dict, len_ratio_dict, len_detected_ratio_dict = {}, {}, {}, {}
    prec_dict, rec_dict, f1_dict, auc_dict, best_f1_dict = {}, {}, {}, {}, {}
    prec_adj_dict, rec_adj_dict, f1_adj_dict = {}, {}, {}
    time_dict = {}


    
    for channel in df.chan_id:

        if config.dimensions is not None:
            ts_test = np.load(data_path/"test"/f"{channel}.npy")[:,config.dimensions]
        else:
            ts_test = np.load(data_path/"test"/f"{channel}.npy")

        # import ipdb
        # ipdb.set_trace()

        seq_len = len(ts_test)
        anomaly_seqs = df.loc[df.chan_id == channel].anomaly_sequences.to_numpy().item()
        anomaly_seqs = re.findall(r'\d+', anomaly_seqs)
        anomaly_intervals = []
        for i in list(range(0, len(anomaly_seqs), 2)):
            anomaly_intervals.append(anomaly_seqs[i:i+2])
        anomaly_intervals = np.array(anomaly_intervals).astype(int)

        ground_truth = np.zeros(ts_test.shape[0])
        for anomaly_points in anomaly_intervals:
            ground_truth[anomaly_points[0]:anomaly_points[-1]] = 1.

        anomaly_len = sum(ground_truth)
        anomaly_ratio = anomaly_len / seq_len
        print(f"anomaly ratio is {anomaly_ratio * 100.:.3f} %.")

        len_test_dict.update({channel: seq_len})
        len_anomaly_dict.update({channel: anomaly_len})
        len_ratio_dict.update({channel: anomaly_ratio})

        print(f"start detection for channel {channel} ..")
        start_time = time.time()
        anomaly_scores = detector.calculate_anomaly_scores(ts = ts_test, channel_id = channel, contamination = min(anomaly_ratio,0.5)) # contamination has to be inthe range (0.0,0.5]
        detected_anomaly_ratio = detector.auto_anomaly_detection(anomaly_scores)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        if config.use_anomaly_ratio:
            prec, rec, f1, auc, prec_adj, rec_adj, f1_adj,  best_f1 = detector.evaluate(ground_truth, anomaly_scores, anomaly_ratio) 
        else:
            prec, rec, f1, auc, prec_adj, rec_adj, f1_adj,  best_f1 = detector.evaluate(ground_truth, anomaly_scores, detected_anomaly_ratio) 

        len_detected_ratio_dict.update({channel:detected_anomaly_ratio})
        prec_dict.update({channel: prec})
        rec_dict.update({channel: rec})
        f1_dict.update({channel: f1})
        auc_dict.update({channel: auc})
        best_f1_dict.update({channel: best_f1})

        time_dict.update({channel: elapsed_time})

        prec_adj_dict.update({channel: prec_adj})
        rec_adj_dict.update({channel: rec_adj})
        f1_adj_dict.update({channel: f1_adj})

    smap_metrics = pd.DataFrame({
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
    
    smap_metrics.insert(0, "Dataset", smap_metrics.index)
    smap_metrics.reset_index(drop = True, inplace = True)
    
    smap_metrics.to_csv(os.path.join(config.result_path, config.dataset+"_"+config.detector_type+"_results.csv"))



