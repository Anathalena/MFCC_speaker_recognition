from feature_extraction import MFCC
from model import kNN
from data_loader import DataLoader

import os
import mlflow
import numpy as np
from itertools import product

# Load data
working_dir = os.getcwd()
data_path_test = os.path.join(working_dir, 'Data', 'old_samples','test')
data_path_train = os.path.join(working_dir, 'Data','old_samples', 'train')


melbands = 15 #number of MFCC
maxmel = 8000 #range for Mel-filterbank [0 - maxmel]Hz
parts_lst = [1,2,3,(30,20)] #how to split a signal, provide either number of parts in which to divide a signal or (window_length, window_overlap)
distance_lst = ['euclidean','manhattan'] #distance metric for kNN
k_lst = np.unique(np.logspace(start=1, stop=6, base=2, num=10, dtype=np.int64)) #number of neighbours for kNN

experiment = mlflow.set_experiment(experiment_name='Hypertuning for old samples (Sample 1)')

def run():
    i = 1
    for parts, distance, k in product(parts_lst, distance_lst, k_lst):
        if (parts==1 or parts==2 or parts==3) and k>7:
            continue
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='run number {}'.format(i)):
            print('Running experiment: {}'.format(i))
            print('Parts: {}'.format(parts))
            print('Neighbours: {}'.format(k))
            feature_extractor = MFCC(melbands, maxmel, parts)
            train = DataLoader(feature_extractor, data_path_train)
            test = DataLoader(feature_extractor, data_path_test) 
            
            model = kNN(k=k, distance=distance)
            model.fit(train.data,train.labels)
            _, f1, precision, recall = model.predict(test.data, test.labels)

            mlflow.log_param('Number of neighbours', k)
            mlflow.log_param('Number of parts of the signal', parts)
            mlflow.log_param('Distance', distance)
            mlflow.log_metric('F1 score', f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)

            i+=1

if __name__ == "__main__":
    run()
    
    