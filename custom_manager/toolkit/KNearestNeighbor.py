import math

class DistanceData:

    def __init__(self, originalIndex, distance):
        self.originalIndex = originalIndex
        self.distance = distance

class KNearestNeighbor:

    features = []
    labels = []
    distancesInOrder = []

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.distances = []

    def insertionSortForDistances(self, newEntry):
        if len(self.distances) == 0:
            self.distances.append(newEntry)

    def calculateDistancesForDataRow(self, row):
        for feature in self.features:  #each entry
            distance = 0
            for i in range(len(row)):  #each attribute
                distance += (feature[i] - row[i])**2
            distance = math.sqrt(distance)
            self.distances.append(distance)

