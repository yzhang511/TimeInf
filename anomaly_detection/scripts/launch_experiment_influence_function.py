import os
import pandas as pd
from pathlib import Path
import time
import subprocess


dataset = 'SMAP'
data_path = f"../data_processed/{dataset}"
num_dims = 1
dims = '0'
if not os.path.exists('./logs_InF/'):
    os.mkdir('./logs_InF/')

for win_size in [25, 50, 75, 100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    cmd = f"""#!/bin/bash
#
#SBATCH --job-name={dataset}{win_size}
#SBATCH --account=statsaff
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=FAIL

python main_SMAP_MSL.py --win_size {win_size}   --dataset {dataset}  --data_path {data_path}  --result_path {result_path} --dimensions {dims} --detector_type InfluenceFunctionDetector
"""
    print(cmd)
    with open("run_temp_inf.sh", "w") as f:
        f.write(cmd)
    with open('./logs_InF/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_inf.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)

#################


dataset = 'MSL'
data_path = f"../data_processed/{dataset}"
num_dims = 1
dims = '0'
if not os.path.exists('./logs_InF/'):
    os.mkdir('./logs_InF/')

for win_size in [25, 50, 75, 100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    cmd = f"""#!/bin/bash
#
#SBATCH --job-name={dataset}{win_size}
#SBATCH --account=statsaff
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=FAIL

python main_SMAP_MSL.py --win_size {win_size}   --dataset {dataset}  --data_path {data_path}  --result_path {result_path} --dimensions {dims} --detector_type InfluenceFunctionDetector
"""
    print(cmd)
    with open("run_temp_inf.sh", "w") as f:
        f.write(cmd)
    with open('./logs_InF/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['sbatch', 'run_temp_inf.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)


################

dataset = 'SMD'
data_path = f"../data_processed/{dataset}"
num_dims = 38
if not os.path.exists('./logs_InF/'):
    os.mkdir('./logs_InF/')

for win_size in [25, 50, 75, 100,125,150]:
    result_path = f'./results/win_size_{win_size}/'
    cmd = f"""#!/bin/bash
#
#SBATCH --job-name={dataset}{win_size}
#SBATCH --account=statsaff
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=FAIL

python main_SMD.py --win_size {win_size}   --dataset {dataset}  --data_path {data_path}  --result_path {result_path} --detector_type InfluenceFunctionDetector"""
    print(cmd)
    with open("run_temp_inf.sh", "w") as f:
        f.write(cmd)
    with open('./logs_InF/'+f'{dataset}'+f'_{win_size}_log.log', 'w') as logfile:
        result = subprocess.run(['bash', 'run_temp_inf.sh'], text=True, stdout=logfile, stderr=logfile)
    time.sleep(1)