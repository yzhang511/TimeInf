import os
import pandas as pd
from pathlib import Path
import time
import subprocess


data_path = "../data_processed/MSL"
dataset = 'MSL'
num_dims = 1
dims = '0'
if not os.path.exists('./logs_ARIMA/'):
    os.mkdir('./logs_ARIMA/')

result_path = f'./results/'
cmd =f"""#!/bin/bash
#
#SBATCH --job-name=arima
#SBATCH --account=statsaff
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=FAIL

python main_SMAP_MSL.py --dataset {dataset} --dimensions {dims}  --data_path {data_path}  --result_path {result_path}  --detector_type ARIMADetector    
"""
print(cmd)
with open("run_temp_arima.sh", "w") as f:
    f.write(cmd)
with open('./logs_ARIMA/'+f'{dataset}'+f'_log.log', 'w') as logfile:
    result = subprocess.run(['sbatch', 'run_temp_arima.sh'], text=True, stdout=logfile, stderr=logfile)
time.sleep(1)

'''export CUDA_VISIBLE_DEVICES=0
python main_SMAP_MSL.py --dataset {dataset} --dimensions {dims}  --data_path {data_path}  --result_path {result_path}  --detector_type IForestDetector    
    python main_SMD.py --dataset {dataset} --data_path {data_path}  --result_path {result_path}  --detector_type IForestDetector

'''

