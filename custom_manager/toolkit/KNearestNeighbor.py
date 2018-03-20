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

    def __init__(self, features, labels, k, regression):
        self.regression = regression
        self.k = k
        self.features = []
        self.labels = []
        for i in range(features.rows):
            self.features.append(features.row(i))
            self.labels.append(labels.row(i)[0])
        featureMaxes = []
        featureMins = []
        # for i in range(features.cols):  #normalize
        #     bestMax = 0
        #     bestMin = math.inf
        #     for entry in features.col(i):
        #         if entry > bestMax:
        #             bestMax = entry
        #         if entry < bestMin:
        #             bestMin = entry
        #     featureMaxes.append(bestMax)
        #     featureMins.append(bestMin)
        # for i in range(len(self.features)):
        #     for j in range(len(self.features[i])):
        #         self.features[i][j] = (self.features[i][j] - featureMins[j])/(featureMaxes[j] - featureMins[j])
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.distancesInOrder = []
        self.distances = []
        self.currRowNumber = 0
        self.weights = []

    def calculateDistancesForDataRow(self, row):
        #True is cont, False is nominal
        featureTypes = [False, True, True, False, False, False, False, True, False, False, True, False, False, True, True]
        row = np.array(row)
        row = row[np.newaxis, :]
        differences = self.features - row
        differences[differences == 0.0] = .000000001
        differences[differences == math.inf ] = 1  #if x or y is unknown
        differences[differences > 1.0] = 1 # 0 if x=y, 1 otherwise for nominal.
        self.distances = np.sqrt(np.sum(differences ** 2, axis=1)) #HEOM
        # self.distances = np.sum(differences**2, axis = 1)  #Eucilidean

    def nearestNeighborVote(self, row):
        kNearestDistanceIndexs = np.argpartition(self.distances, self.k)
        kNearestDistanceIndexs = kNearestDistanceIndexs[:self.k]
        votes = self.labels[kNearestDistanceIndexs]
        test =  (self.distances[kNearestDistanceIndexs[:self.k]] ** 2)
        self.weights = 1 / (self.distances[kNearestDistanceIndexs[:self.k]] ** 2)
        if self.regression:
            # labelWeightedVotes = [0,0]
            # for voteLabel, weight in zip(np.nditer(votes), np.nditer(self.weights)):
            #     labelWeightedVotes[int(voteLabel)] += weight
            # return np.argmax(labelWeightedVotes)
            # prediction = votes * self.weights  #wrong?
            # prediction = np.sum(prediction, axis=0)
            # prediction = prediction / (np.sum(self.weights, axis=0))
            # # if math.isnan(prediction):
            # #     i=4
            # return prediction #if not math.isnan(prediction) else 0
            mean = np.mean(votes)
            return mean
        else:
            # if self.k == 1:
            #     return votes[0]
            # labelCounts = [0,0]
            # for x in np.nditer(votes):
            #     labelCounts[int(x)] += self.weights[int(x)]
            # if labelCounts[0] > labelCounts[1]:
            #     return 0
            # else:
            #     return 1
            mode = stats.mode(votes)[0][0]
            return mode

