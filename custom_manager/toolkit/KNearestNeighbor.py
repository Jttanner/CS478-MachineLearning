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

        self.features = features
        self.labels = labels

        # if using matrix class
        # for i in range(features.rows):
        #     if i != 0:
        #         self.features.append(features.row(i))
        #         self.labels.append(labels.row(i)[0])

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

        # if using toolkit
        # self.features = np.array(self.features)
        # self.labels = np.array(self.labels)
        self.distancesInOrder = []
        self.distances = []
        self.currRowNumber = 0
        self.weights = []
        for row, i in zip(self.features, range(len(self.features))):
            if i == len(self.features):
                break
            i -= 1
            originalFeatures = np.copy(self.features)
            originalLabels = np.copy(self.labels)
            try:
                self.features = np.delete(self.features, i, axis=0)
                self.labels = np.delete(self.labels, i, axis=0)
                self.calculateDistancesForDataRow(row)
                winningVote = self.nearestNeighborVote(row)
                if winningVote == self.labels[i]:
                    originalFeatures = np.delete(originalFeatures, i, axis=0)
                    originalLabels = np.delete(originalLabels, i, axis=0)
                else:
                    i += 1
                self.features = originalFeatures
                self.labels = originalLabels
            except:
                break



        ######################################
        # BAD REDUCTION TACTIC.  DIDN'T WORK #
        ######################################
        # usedToDeleteIndexes = []
        # deleteIndexes = []
        # for row, i in zip(self.features, range(len(self.features))):
        #     self.calculateDistancesForDataRow(row)
        #     d = 2
        #     try:
        #         smallestIndexs = np.argpartition(self.distances, d)[:d]
        #     except:
        #         i = 4
        #     smallestValues = (self.distances[smallestIndexs])
        #
        #     meanDistance = np.mean(self.distances)
        #     # print(smallestIndexs)
        #     # print(smallestValues)
        #     for j in range(len(smallestValues)):
        #         for k in range(len(smallestValues)):
        #             if j != k:
        #                 if self.labels[j] == self.labels[k]:
        #                     if (smallestIndexs[k] not in deleteIndexes) and (smallestIndexs[k] not in usedToDeleteIndexes):
        #                         if smallestValues[j] - smallestValues[k] < abs(.0000000001):
        #                             deleteIndexes.append(smallestIndexs[k])
        #                             if smallestIndexs[j] not in usedToDeleteIndexes:
        #                                 usedToDeleteIndexes.append(smallestIndexs[j])
        #             # print(self.labels[j])
        # self.features = np.delete(self.features, deleteIndexes, axis=0) if deleteIndexes != [] else self.features
        #     # for index in deleteIndexes:
        #     #     self.features = np.delete(self.features, index, axis=0)


    def calculateDistancesForDataRow(self, row):
        #True is cont, False is nominal
        featureTypes = [False, True, True, False, False, False, False, True, False, False, True, False, False, True, True]
        row = np.array(row)
        row = row[np.newaxis, :]
        differences = self.features - row
        differences[differences == 0.0] = .000000001
        # differences[differences == math.inf ] = 1  #if x or y is unknown
        # differences[differences > 1.0] = 1 # 0 if x=y, 1 otherwise for nominal.
        # self.distances = np.sqrt(np.sum(differences ** 2, axis=1)) #HEOM
        self.distances = np.sum(differences**2, axis = 1)  #Eucilidean

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
            if self.k == 1:
                return votes[0]
            # labelCounts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # for x in np.nditer(votes):
            #     labelCounts[int(x)] += self.weights[int(x)]
            # if labelCounts[0] > labelCounts[1]:
            #     return 0
            # else:
            #     return 1
            mode = stats.mode(votes)[0][0]
            return mode

