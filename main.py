from feature_extraction import MFCC
from model import kNN
from data_loader import DataLoader
import os
import pandas as pd
from librosa import load

working_dir = os.getcwd()
data_path_test = os.path.join(working_dir, 'Data', 'test')
data_path_train = os.path.join(working_dir, 'Data', 'train')

melbands = 20 #number of MFCC
maxmel = 8000 #range for Mel-filterbank [0 - maxmel]Hz
parts = 2 #how to split a signal, provide either no of parts or (window_length, window_overlap)
distance = 'euclidean' #metric for kNN


feature_extractor = MFCC(melbands, maxmel, parts)

train = DataLoader(feature_extractor, data_path_train)
test = DataLoader(feature_extractor, data_path_test) 

model = kNN(k=10, distance=distance)
model.fit(train.data,train.labels)
pred, acc = model.predict(test.data, test.labels)
print(acc)