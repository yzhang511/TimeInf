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

for i,win_size in enumerate([25, 50, 75, 100, 125, 150]):
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./Anomaly_Transformer/checkpoints/win_size_{win_size}/'
    cmd = f'''#!/bin/sh
#
#SBATCH --account=statsaff        # The account name for the job.
#SBATCH --job-name={dataset}_{i}    # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=0-03:00           # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=15gb        # The memory the job will use per cpu core.
#SBATCH --output=/burg/stats/users/js5544/time-series-influence/anomaly_detection/logs_AnomTrans/win_size_{win_size}/slurm_log_{dataset}{str(i)}-%A.out
 
module load cuda11.1/toolkit 
export CUDA_VISIBLE_DEVICES=0
python main_SMAP_MSL.py --use_anomaly_ratio --num_epochs 10 --batch_size 64 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --dimensions {dims} --detector_type AnomalyTransformerDetector --e_layers 1 --n_heads 4 --d_model 128 --d_ff 128 --verbose
'''
    print(cmd)
    with open("run_temp_anomtrans.sh", "w") as f:
        f.write(cmd)
    with open('./logs_AnomTrans/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['sbatch', 'run_temp_anomtrans.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)


data_path = "../data_processed/MSL"
dataset = 'MSL'
anomaly_ratio = 1
num_dims =  1
dims = '0'
if not os.path.exists('./logs_AnomTrans/'):
    os.mkdir('./logs_AnomTrans/')

for i,win_size in enumerate([25, 50, 75, 100, 125, 150]):
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./Anomaly_Transformer/checkpoints/win_size_{win_size}/'
    cmd = f'''#!/bin/sh
#
#SBATCH --account=statsaff        # The account name for the job.
#SBATCH --job-name={dataset}_{i}    # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=0-04:00           # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=15gb        # The memory the job will use per cpu core.
#SBATCH --output=/burg/stats/users/js5544/time-series-influence/anomaly_detection/logs_AnomTrans/win_size_{win_size}/slurm_log_{dataset}{str(i)}-%A.out

 
module load cuda11.1/toolkit 
export CUDA_VISIBLE_DEVICES=0
python main_SMAP_MSL.py --use_anomaly_ratio --num_epochs 10 --batch_size 64 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --dimensions {dims} --detector_type AnomalyTransformerDetector --e_layers 1 --n_heads 4 --d_model 128 --d_ff 128 --verbose
    '''
    print(cmd)
    with open("run_temp_anomtrans.sh", "w") as f:
        f.write(cmd)
    with open('./logs_AnomTrans/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['sbatch', 'run_temp_anomtrans.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)

data_path = "../data_processed/SMD"
dataset = 'SMD'
anomaly_ratio = .5
num_dims =  38
if not os.path.exists('./logs_AnomTrans/'):
    os.mkdir('./logs_AnomTrans/')

for i,win_size in enumerate([25, 50, 75, 100, 125, 150]):
    result_path = f'./results/win_size_{win_size}/'
    model_save_path = f'./Anomaly_Transformer/checkpoints/win_size_{win_size}/'
    cmd = f'''#!/bin/sh
#
#SBATCH --account=statsaff        # The account name for the job.
#SBATCH --job-name={dataset}_{i}    # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=0-05:00           # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=20gb        # The memory the job will use per cpu core.
#SBATCH --output=/burg/stats/users/js5544/time-series-influence/anomaly_detection/logs_AnomTrans/win_size_{win_size}/slurm_log_{dataset}{str(i)}-%A.out

 
module load cuda11.1/toolkit 
export CUDA_VISIBLE_DEVICES=0
python main_SMD.py --use_anomaly_ratio --num_epochs 10 --batch_size 64 --win_size {win_size}   --dataset {dataset}    --data_path {data_path}  --input_c {num_dims}  --result_path {result_path}   --output_c {num_dims} --model_save_path {model_save_path} --detector_type AnomalyTransformerDetector --e_layers 1 --n_heads 4 --d_model 128 --d_ff 128 --verbose  '''
    print(cmd)
    with open("run_temp_anomtrans.sh", "w") as f:
        f.write(cmd)
    with open('./logs_AnomTrans/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['sbatch', 'run_temp_anomtrans.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)
