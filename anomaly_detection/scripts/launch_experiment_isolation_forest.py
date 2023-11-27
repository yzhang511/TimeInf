import os
import pandas as pd
from pathlib import Path
import time
import subprocess

dataset = 'SMAP'
data_path = f"../data_processed/{dataset}"
num_dims = 1
dims = '0'
if not os.path.exists('./logs_IForest/'):
    os.mkdir('./logs_IForest/')

result_path = f'./results/'
cmd = f'''export CUDA_VISIBLE_DEVICES=0
python main_SMAP_MSL.py --dataset {dataset} --dimensions {dims}  --data_path {data_path}  --result_path {result_path}  --detector_type IForestDetector
'''
print(cmd)
with open("run_temp_IForest.sh", "w") as f:
    f.write(cmd)
with open('./logs_IForest/'+f'{dataset}'+f'_log.log', 'w') as logfile:
    result = subprocess.run(['bash', 'run_temp_IForest.sh'], text=True, stdout=logfile, stderr=logfile)
time.sleep(1)

########################

dataset = 'MSL'
data_path = f"../data_processed/{dataset}"
num_dims = 1
dims = '0'
if not os.path.exists('./logs_IForest/'):
    os.mkdir('./logs_IForest/')

result_path = f'./results/'
cmd = f'''export CUDA_VISIBLE_DEVICES=0
python main_SMAP_MSL.py --dataset {dataset} --dimensions {dims}  --data_path {data_path}  --result_path {result_path}  --detector_type IForestDetector
'''
print(cmd)
with open("run_temp_IForest.sh", "w") as f:
    f.write(cmd)
with open('./logs_IForest/'+f'{dataset}'+f'_log.log', 'w') as logfile:
    result = subprocess.run(['bash', 'run_temp_IForest.sh'], text=True, stdout=logfile, stderr=logfile)
time.sleep(1)

########################

dataset = 'SMD'
data_path = f"../data_processed/{dataset}"
num_dims = 38
dims = '0'
if not os.path.exists('./logs_IForest/'):
    os.mkdir('./logs_IForest/')

result_path = f'./results/'
cmd = f'''export CUDA_VISIBLE_DEVICES=0
python main_SMD.py --dataset {dataset} --data_path {data_path}  --result_path {result_path}  --detector_type IForestDetector
'''
print(cmd)
with open("run_temp_IForest.sh", "w") as f:
    f.write(cmd)
with open('./logs_IForest/'+f'{dataset}'+f'_log.log', 'w') as logfile:
    result = subprocess.run(['bash', 'run_temp_IForest.sh'], text=True, stdout=logfile, stderr=logfile)
time.sleep(1)

'''export CUDA_VISIBLE_DEVICES=0
python main_SMAP_MSL.py --dataset {dataset} --dimensions {dims}  --data_path {data_path}  --result_path {result_path}  --detector_type IForestDetector    '''

