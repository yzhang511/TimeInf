import ast
import csv
import os
import sys
from pickle import dump

import numpy as np
# from tfsnippet.utils import makedirs

output_folder = './'
os.makedirs(output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, channel, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    with open(os.path.join(output_folder, dataset, channel + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    if dataset == 'SMD':
        dataset_folder =  '../data/multivariate/' + dataset
        os.makedirs(os.path.join(output_folder, dataset), exist_ok=True)
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, dataset, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename,dataset, filename.strip('.txt'), dataset_folder)
                load_and_save('test_label', filename, dataset, filename.strip('.txt'), dataset_folder)
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = '../data/multivariate/SMAP_MSL/' 
        os.makedirs(os.path.join(output_folder, dataset), exist_ok=True)
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        # labels = []
        for row in data_info:
            channel_no = row[0]
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            with open(os.path.join(output_folder,dataset, channel_no + "_" + 'test_label' + ".pkl"), "wb") as file:
                dump(label, file)

        def save(category):
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data = np.asarray(temp)
                with open(os.path.join(output_folder, dataset, filename  + "_" + category + ".pkl"), "wb") as file:
                    dump(data, file)

        for c in ['train', 'test']:
            save(c)


if __name__ == '__main__':
    datasets = ['SMD', 'SMAP', 'MSL']
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
        """)
