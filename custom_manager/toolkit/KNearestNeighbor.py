import math

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
        self.distancesInOrder = []
        self.distances = []

    def insertForDistances(self, newEntry, index):
        newDistanceData = DistanceData(index, newEntry)
        self.distancesInOrder.append(newDistanceData)
        if len(self.distances) == 0:
            self.distances.append(newDistanceData)
        else:  #we only care about first k entries
            if len(self.distances) < self.k:
                for i in range(len(self.distances)):
                    if self.distances[i].distance > newDistanceData.distance:
                        self.distances.insert(i, newDistanceData)
                    elif i == len(self.distances) - 1:
                        self.distances.append(newDistanceData)
            else:
                for i in range(self.k):
                    if self.distances[i].distance > newDistanceData.distance:
                        self.distances.insert(i, newDistanceData)

    def calculateDistancesForDataRow(self, row):
        self.distances = []
        self.distancesInOrder = []
        for feature, rowNumber in zip(self.features, range(len(self.features))):  #each entry
            distance = 0
            for i in range(len(row)):  #each attribute
                distance += (feature[i] - row[i])**2  #Euclidean
                #distance += abs(feature[i] - row[i])  #Manhattan
            distance = math.sqrt(distance)
            self.insertForDistances(distance, rowNumber)

    def nearestNeighborVote(self, row):
        if self.regression:
            distanceLabelSum = 0
            for i in range(self.k):
                distanceLabelSum += self.labels[int(self.distances[i].distance)]
            distanceLabelMean = distanceLabelSum / self.k
            return  distanceLabelMean
        else:
            votes = []
            for i in range(len(row)):
                votes.append(0)
            for i in range(self.k):
                index = self.distances[i].originalIndex
                votes[int(self.labels[index])] += 1
            bestVoteIndex = 0
            for i in range(len(votes)):
                if votes[i] > votes[bestVoteIndex]:
                    bestVoteIndex = i
            return bestVoteIndex