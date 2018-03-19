import math
import numpy as np
from scipy import stats

class DistanceData:

    def __init__(self, originalIndex, distance):
        self.originalIndex = originalIndex
        self.distance = distance

    def __eq__(self, other):
        return self.originalIndex == other.originalIndex

class KNearestNeighbor:

    features = []
    labels = []
    distancesInOrder = []
    distances = []
    regression = True

    def __init__(self, features, labels, k):
        self.k = k
        self.features = []
        self.labels = []
        for i in range(features.rows):
            self.features.append(features.row(i))
            self.labels.append(labels.row(i)[0])
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.distancesInOrder = []
        self.distances = []
        self.currRowNumber = 0
        self.weights = []

    def calculateDistancesForDataRow(self, row):
        row = np.array(row)
        row = row[np.newaxis, :]
        differences = self.features - row
        self.distances = np.sum(differences**2, axis = 1)

    def nearestNeighborVote(self, row):
        kNearestDistanceIndexs = np.argpartition(self.distances, self.k )
        kNearestDistanceIndexs = kNearestDistanceIndexs[:self.k]
        votes = self.labels[kNearestDistanceIndexs]
        self.weights = 1 / (kNearestDistanceIndexs[:self.k] ** 2)
        if self.regression:
            prediction = votes * self.weights
            prediction = np.sum(prediction, axis=0)
            prediction = prediction / np.sum(self.weights, axis=0)
            return prediction if not math.isnan(prediction) else 0
            # mean = np.mean(votes)
            # return mean
        else:
            labelCounts = [0,0]
            for x in np.nditer(votes):
                labelCounts[int(x)] += self.weights[int(x)]
            if labelCounts[0] > labelCounts[1]:
                return 0
            else:
                return 1
            # mode = stats.mode(votes)
            # return mode

