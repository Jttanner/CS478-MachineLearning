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
        for i in range(len(self.features[0])):  #for each column
            max = 0
            min = math.inf
            for j in range(len(self.features)):  #for each row:
                if self.features[j][i] != float("nan"):
                    if self.features[j][i] > max:
                        max = self.features[j][i]
                    if self.features[j][i] < min:
                        min = self.features[j][i]
            for j in range(len(self.features)):  #for each row:
                if self.features[j][i] != float("nan"):
                    self.features[j][i] = (self.features[j][i] - min )/ (max - min)
        i = 4


    def print2dList(self,list):
        printMe = "["
        for i in range(len(list)):
            if self.labelTypes[i] == self.NOMINAL:
                printMe += '(nom)'
            printMe += str(list[i]) + ', ' if str(list[i]) != "nan" else '?, '
        print(printMe + ']')

    def train(self):
        print("Iteration: 1")
        print("Caluclating Centroids")
        self.pickInitialRandomCentroids()
        self.calculateGroups()
        currCentroid = 0
        for centroid in self.centroids:
            print("Centroid :" + str(currCentroid))  # + centroid)
            self.print2dList(centroid)
            currCentroid += 1
        print("Assigning Groups:")
        linePrintCount = 0
        printMe = ""
        for i in range(len(self.groups)):
            if linePrintCount < 10:
                linePrintCount += 1
                printMe += str(i) + "=" + str(self.groups[i])
                printMe += " "
            else:
                print(printMe + str(i) + "=" + str(self.groups[i]))
                printMe = ""
                linePrintCount = 0
        # for i in range(len(self.groups)):
        #     print(str(i) + "=" + str(self.groups[i]))
        lastsse = 0
        converged = False
        iterations = 2
        sse = self.calculateSSE()
        print(sse)
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
            linePrintCount = 0
            printMe = ""
            for i in range(len(self.groups)):
                if linePrintCount < 10:
                    linePrintCount += 1
                    printMe += str(i) + "=" + str(self.groups[i])
                    printMe += " "
                else:
                    print(printMe)
                    printMe = ""
                    linePrintCount = 0
            sse = self.calculateSSE()
            print("sse: " + str(sse))
            if abs(sse - lastsse) < .01:
                converged = True
            lastsse = sse
            iterations += 1
        print("SSE has converged.")

    def calculateSSE(self):
        for i in range(self.k):  #for each centroid
            colSSEs = []
            for j in range(len(self.features[0])):  #for each column
                colSSEs.append(0)
                for k in range(len(self.features)):  #for each row
                    if self.labelTypes[j] == self.REAL:
                        if math.isnan(self.features[k][j]) or math.isnan(self.centroids[self.groups[k]][j]):
                            colSSEs[j] += 1
                        else:
                            colSSEs[j] += (self.features[k][j] - self.centroids[self.groups[k]][j])**2
                    else:
                        if math.isnan(self.features[k][j]) or math.isnan(self.centroids[self.groups[k]][j]):
                            colSSEs[j] += 1
                        elif self.features[k][j] == self.centroids[self.groups[k]][j]:
                            colSSEs[j] += 0
                        else:
                            colSSEs[j] += 1
        total = 0
        for sse in colSSEs:
            total += sse
        # total = math.sqrt(total)
        return total

    def recalculateCentroids(self):
        newCentroids = []
        for i in range(self.k):
            newCentroid = []
            for j in range(len(self.features[0])):
                newCentroid.append(0)  #fill
            for j in range(len(self.features[0])):  #for each column
                nanCount = 0
                total = 0
                nominalCol = []
                for k in range(len(self.features)):  #for each row
                    if self.groups[k] == i:  #if its in the cluster
                        if self.labelTypes[j] == self.REAL:
                            if math.isnan(self.features[k][j]):
                                nanCount += 1
                            else:
                                total += 1
                                newCentroid[j] += self.features[k][j]
                        else:
                            total += 1
                            if math.isnan(self.features[k][j]):
                                nanCount += 1
                            else:
                                nominalCol.append(self.features[k][j])
                if self.labelTypes[j] == self.REAL:
                    newCentroid[j] = newCentroid[j] / total if total != 0 else float("nan")
                else:
                    if total == nanCount:
                        newCentroid[j] = float("nan")
                    else:
                        newCentroid[j] = stats.mode(nominalCol, nan_policy='omit')[0][0]
            newCentroids.append(newCentroid)
        self.centroids = newCentroids

    def pickInitialRandomCentroids(self):
        if self.forceFirstFourInitialCentroids:
            self.centroids.append(self.features[0])
            self.centroids.append(self.features[1])
            self.centroids.append(self.features[2])
            self.centroids.append(self.features[3])
            self.centroids.append(self.features[4])
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
        i = 4

    # @return best centroid index
    def calculateDistanceToCentroids(self, feature):
        distances = []
        for centroid in self.centroids:
            distance = 0
            for i in range(len(feature)):
                distanceDelta = 0
                if self.labelTypes[i] == self.REAL:
                    if math.isnan(feature[i]) or math.isnan(centroid[i]):
                        distanceDelta = 1
                    else:
                        distanceDelta  = (feature[i] - centroid[i])**2
                else:
                    if math.isnan(feature[i]) or math.isnan(centroid[i]):
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