import math

class DistanceData:

    def __init__(self, originalIndex, distance):
        self.originalIndex = originalIndex
        self.distance = distance

    def __eq__(self, other):  #TODO: confirm it works properly
        return self.originalIndex == other.originalIndex
        # return (self.distance == other.distance) and (self.originalIndex == other.originalIndex)

class KNearestNeighbor:

    features = []
    labels = []
    distancesInOrder = []
    distances = []

    def __init__(self, features, labels):
        self.features = []
        self.labels = []
        for i in range(features.rows):
            self.features.append(features.row(i))
            self.labels.append(labels.row(i))
        self.distancesInOrder = []
        self.distances = []

    def insertionSortForDistances(self, newEntry, index):
        newDistanceData = DistanceData(index, newEntry)
        self.distancesInOrder.append(newDistanceData)
        if len(self.distances) == 0:
            self.distances.append(newDistanceData)
        else:
            for i in range(len(self.distances)):
                if self.distances[i].distance > newDistanceData.distance:
                    self.distances.append(newDistanceData)

    def calculateDistancesForDataRow(self, row):
        self.distances = []
        self.distancesInOrder = []
        for feature, rowNumber in zip(self.features, range(len(self.features))):  #each entry
            distance = 0
            for i in range(len(row)):  #each attribute
                distance += (feature[i] - row[i])**2
            distance = math.sqrt(distance)
            self.insertionSortForDistances(distance, rowNumber)

    def nearestNeighborVote(self, k, row):
        votes = []
        for i in range(len(row)):
            votes.append(0)
        for i in range(k):
            index = self.distances[i].originalIndex
            votes[self.labels[index]] += 1
        bestVoteIndex = 0
        for i in range(len(votes)):
            if votes[i] > votes[bestVoteIndex]:
                bestVoteIndex = i
        return bestVoteIndex