from random import randint
import numpy as np
import math
from scipy import stats

class KMeans:

    labelTypes = [0,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1]  #0 is real, 1 is nominal
    REAL = 0
    NOMINAL = 1
    centroids = None
    groups = None
    forceFirstFourInitialCentroids = True

    def __init__(self, features, labels, k):
        self.features = []
        self.labels = []
        for i in range(features.rows):
            feature = []
            label = []
            for j in range(len(features.row(i))):
                if j != 0:
                    feature.append(features.row(i)[j])
            label.append(labels.row(i))
            self.features.append(feature)
            self.labels.append(label)
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.k = k
        self.bestAccuracySoFar = 0
        self.lastAccuarcy = 0
        self.runsWithNoMeaningfulUpdate = 0
        self.centroids = []
        self.groups = []

    def print2dList(self,list):
        for i in range(len(list)):
            print(list[i])

    def train(self):
        self.pickInitialRandomCentroids()
        self.calculateGroups()
        lastsse = 0
        iterations = 0
        converged = False
        while not converged:
            print("Iteration: " + str(iterations))
            print("Caluclating Centroids")
            self.recalculateCentroids()
            currCentroid = 0
            for centroid in self.centroids:
                print("Centroid :" + str(currCentroid))  # + centroid)
                self.print2dList(centroid)
                currCentroid += 1
            self.groups = []
            print("Assigning Groups:")
            self.calculateGroups()
            for i in range(len(self.groups)):
                print(str(i) + "=" + str(self.groups[i]))
            sse = self.calculateSSE()
            print(sse)
            if sse == lastsse:
                converged = True
            iterations += 1
        print("SSE has converged.")
        # currAccuracy = self.calculateAccuracy()
        # if abs(currAccuracy - lastAccuarcy) < .0001:
        #     self.runsWithNoMeaningfulUpdate += 1


    def calculateSSE(self):
        sse = 0
        for i in range(len(self.labels)):
            sse += (self.labels[int(self.groups[i])][0] - self.labels[i][0])**2
        return sse
            # total += 1
            # if self.labels[self.groups[i]][0] == self.labels[i][0]:
                # correct += 1

    def recalculateCentroids(self):
        newCentroids = []
        for i in range(len(self.centroids)):
            newCentroid = []
            for j in range(len(self.features[0])):
                newCentroid.append(0)
            for j in range(len(self.features[0])):
                if self.labelTypes[j] == self.REAL:
                    for k in range(len(self.features)):
                        newCentroid[j] += self.features[k][j]
                    newCentroid[j] = newCentroid[j] / len(self.features)
                else:
                    columnInfo = []
                    for k in range(len(self.features)):
                        columnInfo.append(self.features[k][j])
                    columnInfo = np.array(columnInfo)
                    newCentroid[j] = stats.mode(columnInfo)[0][0]
            newCentroids.append(newCentroid)
        self.centroids = newCentroids

    def pickInitialRandomCentroids(self):
        if self.forceFirstFourInitialCentroids:
            self.centroids.append(self.features[0])
            self.centroids.append(self.features[1])
            self.centroids.append(self.features[2])
            self.centroids.append(self.features[3])
        else:
            lastInt = -1
            for i in range(self.k):
                randomInitialCentroidIndex = randint()
                while randomInitialCentroidIndex == lastInt:
                    randomInitialCentroidIndex = randint()
                lastInt = randomInitialCentroidIndex
                centroid = []
                for j in range(len(self.features[i])):
                    centroid.append(self.features[i][j])
                self.centroids.append(centroid)

    # group sets for centroids
    def calculateGroups(self):
        for i in range(len(self.features)):
            self.groups.append(self.calculateDistanceToCentroids(self.features[i]))

    # @return best centroid index
    def calculateDistanceToCentroids(self, feature):
        distances = []
        for centroid in self.centroids:
            distance = 0
            for i in range(len(feature)):
                distanceDelta = 0
                if self.labelTypes[i] == self.REAL:
                    if math.isinf(feature[i]) or math.isinf(centroid[i]):
                        distanceDelta = 1
                    else:
                        distanceDelta  = (feature[i] - centroid[i])**2
                else:
                    print(feature[i])
                    print(centroid[i])
                    if math.isinf(feature[i]) or math.isinf(centroid[i]):
                        distanceDelta = 1
                    elif feature[i] == centroid[i]:
                        distanceDelta = 0
                    else:
                        distanceDelta = 1
                distance += distanceDelta
            distances.append(distance)
        bestIndex = 0
        for i in range(len(distances)):
            if distances[i] < distances[bestIndex]:
                bestIndex = i
        return int(bestIndex)

    # features will not be empty
    def calculateAccuracy(self):
        correct = 0
        total = 0
        for i in np.nditer(self.labels):
            total += 1
            if self.labels[self.groups[i]][0] == self.labels[i][0]:
                correct += 1
        return correct / total