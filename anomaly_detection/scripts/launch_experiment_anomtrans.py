import os
import pandas as pd
from pathlib import Path
import time
import subprocess

data_path = "../data_processed/SMAP"
dataset = 'SMAP'
anomaly_ratio = 1
num_dims =  1
dims = '0'
if not os.path.exists('./logs_AnomTrans/'):
    os.mkdir('./logs_AnomTrans/')

for win_size in [50,75,100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./Anomaly_Transformer/checkpoints/win_size_{win_size}/'
    cmd = f'''export CUDA_VISIBLE_DEVICES=0
    python main_SMAP_MSL.py --anomaly_ratio {anomaly_ratio} --num_epochs 10 --batch_size 256 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --dimensions {dims} --detector_type AnomalyTransformerDetector
    '''
    print(cmd)
    with open("run_temp_anomtrans.sh", "w") as f:
        f.write(cmd)
    with open('./logs_AnomTrans/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_anomtrans.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)


data_path = "../data_processed/MSL"
dataset = 'MSL'
anomaly_ratio = 1
num_dims =  1
dims = '0'
if not os.path.exists('./logs_AnomTrans/'):
    os.mkdir('./logs_AnomTrans/')

for win_size in [25,50,75,100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./Anomaly_Transformer/checkpoints/win_size_{win_size}/'
    cmd = f'''export CUDA_VISIBLE_DEVICES=0
    python main_SMAP_MSL.py --anomaly_ratio {anomaly_ratio} --num_epochs 10 --batch_size 256 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --dimensions {dims} --detector_type AnomalyTransformerDetector --verbose
    '''
    print(cmd)
    with open("run_temp_anomtrans.sh", "w") as f:
        f.write(cmd)
    with open('./logs_AnomTrans/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_anomtrans.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)

data_path = "../data_processed/SMD"
dataset = 'SMD'
anomaly_ratio = .5
num_dims =  38
if not os.path.exists('./logs_AnomTrans/'):
    os.mkdir('./logs_AnomTrans/')

for win_size in [25,50,75,100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./Anomaly_Transformer/checkpoints/win_size_{win_size}/'
    cmd = f'''export CUDA_VISIBLE_DEVICES=0
    python main_SMD.py --anomaly_ratio {anomaly_ratio} --num_epochs 10 --batch_size 256 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --detector_type AnomalyTransformerDetector    '''
    print(cmd)
    with open("run_temp_anomtrans.sh", "w") as f:
        f.write(cmd)
    with open('./logs_AnomTrans/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_anomtrans.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)
