import pandas as pd
from librosa import load
import os
from  natsort import natsorted

class DataLoader():
    def __init__(self, feature_extractor, data_path):
        self.extractor = feature_extractor
        self.path = data_path
        self.data, self.labels = self.load_data()

    def load_data(self):
        df = pd.DataFrame(data=[], columns=[i for i in range(self.extractor.melbands-1)])
        df['Speaker'] = None

        for dir in natsorted(os.listdir(self.path)):
            for filename in natsorted(os.listdir(os.path.join(self.path, dir))):
                y, fs = load(os.path.join(self.path, dir, filename), sr=None)
                features = self.extractor.mfcc(y,fs)

                tmp = pd.DataFrame(data=features)
                tmp['Speaker'] = dir
                
                df = pd.concat([df,tmp],axis=0).reset_index(drop=True) 
                
        data = df.drop('Speaker', axis=1).to_numpy()
        labels = df['Speaker'].to_numpy()

        return data, labels