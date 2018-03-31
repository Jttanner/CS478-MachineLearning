from random import randint
import numpy as np

class KMeans:

    centroids = None
    groups = None
    forceFirstFourInitialCentroids = True

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
        self.centroids = []
        self.groups = []

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
            for centroid in self.centroids:
                print("Centroid :" + str(iterations) + centroid)
            self.groups = []
            print("Assigning Groups:")
            self.calculateGroups()
            for i in range(len(self.groups)):
                print(str(i) + "=" + self.groups[i])
            sse = self.calculateSSE()
            if sse == lastsse:
                converged = True
            iterations += 1
        print("SSE has converged.")
        # currAccuracy = self.calculateAccuracy()
        # if abs(currAccuracy - lastAccuarcy) < .0001:
        #     self.runsWithNoMeaningfulUpdate += 1


    def calculateSSE(self):
        sse = 0
        for i in np.nditer(self.labels):
            sse += (self.labels[self.groups[i]][0] - self.labels[i][0])**2
        return sse
            # total += 1
            # if self.labels[self.groups[i]][0] == self.labels[i][0]:
                # correct += 1

    def recalculateCentroids(self):
        newCentroids = []
        for i in range(len(self.groups)):
            centroid = []
            for j in range(len(self.groups[i])):
                centroid.append(0)
            centroid = np.array(centroid)
            for j in range(len(self.features)):
                centroid = centroid + self.features[j]
            centroid = centroid / len(self.features)
            newCentroids.append(newCentroids)
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
        for i in np.nditer(self.features):
            self.groups.append(self.features[self.calculateDistanceToCentroids(self.features[i])])

    # @return best centroid index
    def calculateDistanceToCentroids(self, feature):
        distances = []
        for centroid in self.centroids:
            distance = 0
            for i in range(len(feature)):
                distanceDelta = 0

                distance += (feature[i] - centroid[i])**2
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