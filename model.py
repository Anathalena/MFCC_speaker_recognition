import numpy as np
from statistics import mode
from sklearn.metrics import accuracy_score

class kNN():
    def __init__(self, k=10, distance='euclidean'):
        self.k = k
        self.distance = distance
    
    # Euclidean distance (l2 norm)
    def euclidean(self, v1, v2):
        return np.sqrt(np.sum((v1-v2)**2, axis=1))
    
    # Manhattan distance (l1 norm)
    def manhattan(self, v1, v2):
        return np.sum(np.abs(v1-v2), axis=1)
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_neighbours(self, test_point):
        if self.distance == 'euclidean':
            distances = self.euclidean(test_point, self.X_train)
        elif self.distance == 'manhattan':
            distances = self.manhattan(test_point, self.X_train)
        indices = np.argpartition(distances, self.k-1)[:self.k]
        nearest_neighbours = self.y_train[indices]
        return nearest_neighbours
    
    def predict(self, X_test, y_test):
        predictions = []
        for x in X_test:
            neighbours = self.get_neighbours(x)
            predictions.append(mode(neighbours))
        
        acc = accuracy_score(y_test, predictions)
        return predictions, acc