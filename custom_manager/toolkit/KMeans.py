from random import randint
import numpy as np

class KMeans:

    centroidIndexes = None
    groups = None

    def __init__(self, features, labels, k):
        self.features = []
        self.labels = []
        for i in range(features.rows):
            self.features.append(features.row(i))
            self.labels.append(labels.row(i))
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.k = k
        self.bestAccuracySoFar = 0
        self.lastAccuarcy = 0
        self.runsWithNoMeaningfulUpdate = 0
        self.centroidIndexes = []
        self.groups = []
        # self.pickInitialRandomCentroids()

    def train(self):
        self.pickInitialRandomCentroids()
        self.calculateGroups()
        while (self.runsWithNoMeaningfulUpdate < 5):
            self.recalculateCentroids()
            self.groups = []
            self.calculateGroups()

    def recalculateCentroids(self):
        pass

    def pickInitialRandomCentroids(self):
        for i in range(self.k):
            randomInitialCentroidIndex = randint()
            while randomInitialCentroidIndex in self.centroidIndexes:
                randomInitialCentroidIndex = randint()
            self.centroidIndexes.append(randomInitialCentroidIndex)

    # group sets for centroids
    def calculateGroups(self):
        for i in np.nditer(self.features):
            self.groups.append(self.calculateDistanceToCentroids(self.features[i]))
            # bestCentroidIndex = self.calculateDistanceToCentroids(self.features[i])
            # self.groups.append(bestCentroidIndex)

    # @return best centroid index
    def calculateDistanceToCentroids(self, feature):
        distances = []
        for centroidIndex in self.centroidIndexs:
            distance = 0
            for i in range(len(feature)):
                distance += (feature[i] - self.features[centroidIndex][i])**2
            distances.append(distance)
        bestIndex = 0
        for i in range(len(distances)):
            if distances[i] < distances[bestIndex]:
                bestIndex = i
        return bestIndex

    # features will not be empty
    def calculateAccuracy(self):
        correct = 0
        total = 0
        for i in np.nditer(self.labels):
            total += 1
            if self.labels(self.groups[i])[0] == self.labels[i][0]:
                correct += 1
        return correct / total