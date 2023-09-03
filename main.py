from feature_extraction import MFCC
from model import kNN
from data_loader import DataLoader
import os
import mlflow
import numpy as np
from itertools import product

working_dir = os.getcwd()
data_path_test = os.path.join(working_dir, 'Data', 'test')
data_path_train = os.path.join(working_dir, 'Data', 'train')

melbands_lst = [15, 20] #number of MFCC
maxmel = 8000 #range for Mel-filterbank [0 - maxmel]Hz
parts_lst = [1,2,3,(30,20)] #how to split a signal, provide either no of parts or (window_length, window_overlap)
distance_lst = ['euclidean', 'manhattan'] #metric for kNN
k_lst = np.arange(10,100,10) #number of neighbours

experiment = mlflow.set_experiment(experiment_name='Hypertuning')

def run():
    i = 1
    for melbands, parts, distance, k in product(melbands_lst, parts_lst, distance_lst, k_lst):
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='run number {}'.format(i)):
            print('Running experiment: {}'.format(i))
            feature_extractor = MFCC(melbands, maxmel, parts)
            train = DataLoader(feature_extractor, data_path_train)
            test = DataLoader(feature_extractor, data_path_test) 

            model = kNN(k=k, distance=distance)
            model.fit(train.data,train.labels)
            _, accuracy, precision, recall = model.predict(test.data, test.labels)
            mlflow.log_param('Number of neighbours', k)
            mlflow.log_param('Melbands', melbands)
            mlflow.log_param('Number of parts of the signal', parts)
            mlflow.log_param('Distance', distance)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)

            i+=1

if __name__ == "__main__":
    run()
    
    