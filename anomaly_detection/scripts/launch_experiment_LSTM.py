import os
import pandas as pd
from pathlib import Path
import time
import subprocess

dataset = 'SMAP'
data_path = f"../data_processed/{dataset}"
anomaly_ratio = 1
num_dims =  1
dims = '0'
if not os.path.exists('./logs_LSTM/'):
    os.mkdir('./logs_LSTM/')

for win_size in [25, 50,75,100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./LSTM/checkpoints/win_size_{win_size}/'
    cmd = f'''export CUDA_VISIBLE_DEVICES=0
    python main_SMAP_MSL.py --anomaly_ratio {anomaly_ratio} --lr 0.001 --num_epochs 100 --batch_size 64 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --dimensions {dims} --detector_type LSTMDetector --verbose
    '''
    print(cmd)
    with open("run_temp_lstm.sh", "w") as f:
        f.write(cmd)
    with open('./logs_LSTM/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_lstm.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)


dataset = 'MSL'
data_path = f"../data_processed/{dataset}"
anomaly_ratio = 1
num_dims =  1
dims = '0'
if not os.path.exists('./logs_LSTM/'):
    os.mkdir('./logs_LSTM/')

for win_size in [25, 50,75,100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./LSTM/checkpoints/win_size_{win_size}/'
    cmd = f'''export CUDA_VISIBLE_DEVICES=0
    python main_SMAP_MSL.py --anomaly_ratio {anomaly_ratio} --lr 0.001 --num_epochs 100 --batch_size 64 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --dimensions {dims} --detector_type LSTMDetector --verbose
    '''
    print(cmd)
    with open("run_temp_lstm.sh", "w") as f:
        f.write(cmd)
    with open('./logs_LSTM/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_lstm.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)


data_path = "../data_processed/SMD"
dataset = 'SMD'
anomaly_ratio = 1
num_dims =  38
if not os.path.exists('./logs_LSTM/'):
    os.mkdir('./logs_LSTM/')

for win_size in [25, 50,75,100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./LSTM/checkpoints/win_size_{win_size}/'
    cmd = f'''export CUDA_VISIBLE_DEVICES=0
    python main_SMD.py --lr 0.001 --num_epochs 100 --batch_size 64 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --detector_type LSTMDetector --verbose
    '''
    print(cmd)
    with open("run_temp_lstm.sh", "w") as f:
        f.write(cmd)
    with open('./logs_LSTM/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_lstm.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)