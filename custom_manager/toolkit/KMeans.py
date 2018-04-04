from random import randint
from random import seed
import numpy as np
import math
from scipy import stats
import time
from KnnLearner import InstanceBasedLearner

class KMeans:
    # labelTypes for Names
    # labelTypes = [0,0,0,1,1,1,0,1,1,0,0,0,0,0]

    # labelTypes for Labor
    labelTypes = [0,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1]  #0 is real, 1 is nominal

    # labelTypes for Sponge
    # labelTypes = [1,1,1,0,1,1,0,0,1]

    # labelTypes for Iris
    # labelTypes = [0,0,0,0,1]

    # labelTypes for abalone
    # labelTypes = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    REAL = 0
    NOMINAL = 1
    centroids = None
    groups = None
    forceFirstFourInitialCentroids = False
    normalize = True

    def __init__(self, features, labels, k):
        # seed(time.time())
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
        if self.normalize:
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
                    if self.features[j][i] != float("nan") and self.labelTypes[i] != self.NOMINAL:
                        self.features[j][i] = (self.features[j][i] - min )/ (max - min)


    def print2dList(self,list):
        printMe = "["
        for i in range(len(list)):
            if self.labelTypes[i] == self.NOMINAL:
                printMe += '(nom)'
            printMe += str(list[i]) + ', ' if str(list[i]) != "nan" else '?, '
        print(printMe + ']')

    def train(self):
        print("number of clusters: " + str(self.k))
        print("Iteration: 1")
        print("Caluclating Centroids")
        self.pickInitialRandomCentroids()
        self.calculateGroups()
        currCentroid = 0
        for i, centroid in zip(range(self.k),self.centroids):
            print("Centroid :" + str(currCentroid))  # + centroid)
            self.print2dList(centroid)
            currCentroid += 1
            instanceCount = 0
            for j in range(len(self.groups)):
                if self.groups[j] == i:
                    instanceCount +=1
            print("Instances in Cluster " + str(i) + ": " + str(instanceCount))
        print("Assigning Groups:")
        linePrintCount = 0
        printMe = ""
        for i in range(len(self.groups)):
            if linePrintCount < 10:
                linePrintCount += 1
                printMe += str(i) + "=" + str(self.groups[i])
                printMe += " "
            else:
                printMe += str(i) + "=" + str(self.groups[i])
                print(printMe)
                printMe = ""
                linePrintCount = 0
        # for i in range(len(self.groups)):
        #     print(str(i) + "=" + str(self.groups[i]))
        lastsse = 0
        converged = False
        iterations = 2
        sse = self.calculateSSE()
        self.calculateSingleClusterSSE()
        print("total sse: " + str(sse))
        while not converged:
            print("number of clusters: " + str(self.k))
            print("Iteration: " + str(iterations))
            print("Caluclating Centroids")
            self.recalculateCentroids()
            currCentroid = 0
            for i, centroid in zip(range(self.k),self.centroids):
                print("Centroid :" + str(currCentroid))  # + centroid)
                self.print2dList(centroid)
                currCentroid += 1
                instanceCount = 0
                for j in range(len(self.groups)):
                    if self.groups[j] == i:
                        instanceCount += 1
                print("Instances tied to centroid " + str(i) + ": " + str(instanceCount))
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
                    printMe += str(i) + "=" + str(self.groups[i])
                    print(printMe)
                    printMe = ""
                    linePrintCount = 0
            sse = self.calculateSSE()
            self.calculateSingleClusterSSE()
            print("total sse: " + str(sse))
            if abs(sse - lastsse) < .01:
                converged = True
            lastsse = sse
            iterations += 1
        print("SSE has converged.")
        self.calculateSilhouette()
        for i in range(self.k):
            currFeatures = []
            currLabels = []
            for j in range(len(self.features)):
                if self.groups[j] == i:
                    currFeatures.append(self.features[j])
                    currLabels.append(self.labels[j][0])
            knnLearner = InstanceBasedLearner()
            knnLearner.train(currFeatures, currLabels)
            correct = 0
            total = 0
            for j in range((len(currFeatures))):
                prediction = []
                knnLearner.predict(currFeatures[j], prediction)
                total += 1
                if prediction[0] == currLabels[j]:
                    correct += 1
            print("Accuracy using KNN for cluster " + str(i) + ": " + str(correct/total if total !=0 else 0))




    def calculateSSE(self):
        # for i in range(self.k):  #for each centroid
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
            total += sse if not math.isinf(sse) else 0
        # total = math.sqrt(total)
        return total

    def calculateSingleClusterSSE(self):
        for i in range(self.k):  #for each centroid
            colSSEs = []
            for j in range(len(self.features[0])):  #for each column
                colSSEs.append(0)
                for k in range(len(self.features)):  #for each row
                    if self.groups[k] == i:
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
                total += sse if not math.isinf(sse) else 0
            print("sse for cluster " + str(i) + ": " + str(total))

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
                randomInitialCentroidIndex = randint(0,len(self.features) - 1)
                while randomInitialCentroidIndex == lastInt:
                    randomInitialCentroidIndex = randint(0,(len(self.features)))
                lastInt = randomInitialCentroidIndex
                centroid = []
                for j in range(len(self.features[i])):
                    centroid.append(self.features[randomInitialCentroidIndex][j])
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
                        # if math.isinf((feature[i] - centroid[i])**2):
                        #     test = 4
                        distanceDelta  = (feature[i] - centroid[i])**2
                else:
                    if math.isnan(feature[i]) or math.isnan(centroid[i]):
                        distanceDelta = 1
                    elif feature[i] == centroid[i]:
                        distanceDelta = 0
                    else:
                        distanceDelta = 1
                distance += distanceDelta
            distances.append((distance))
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


    def calculateSilhouette(self):
        clusterSilhouetteScores = []
        for i in range(self.k):
            aVector = []
            bVector = []
            instanceScores = []
            aDistances = []
            bDistances = [[],[],[],[],[],[],[]]
            for j in range(len(self.groups)):  #for each instance
                for k in range(len(self.features)):  #for each row
                    distance = 0
                    for l in range(len(self.features[0])):  #for each column
                        if j != k:
                            if self.labelTypes[l] == self.REAL:
                                if math.isnan(self.features[k][l]) or math.isnan(self.features[j][l]):
                                    distanceDelta = 1
                                else:
                                    distanceDelta = (self.features[k][l] - self.features[j][l])**2
                            else:
                                if math.isnan(self.features[k][l]) or math.isnan(self.features[j][l]):
                                    distanceDelta = 1
                                elif self.features[k][l] == self.features[j][l]:
                                    distanceDelta = 0
                                else:
                                    distanceDelta = 1
                            distance += distanceDelta
                    if self.groups[k] == self.groups[j] and j!=k and self.groups[j] == i:
                        aDistances.append(distance)
                    elif j!=k and self.groups[j] == i:
                        bDistances[self.groups[k]].append(distance)
                aEntry = np.sum(aDistances)/(len(aDistances) - 1) if aDistances != [] else 0
                bAverages = []
                for currDistance in bDistances:
                    if currDistance != []:
                        bAverages.append(np.mean(currDistance))
                aVector.append(aEntry)
                bVector.append(np.min(bAverages) if bAverages != [] else 0)
                for k in range(len(aVector)):
                    instanceScores.append((bVector[k] - aVector[k])/max(aVector[k],bVector[k]) if max(aVector[k],bVector[k]) != 0 else 0)
            clusterSilhouetteScores.append(np.mean(instanceScores))
        for score, j in zip(clusterSilhouetteScores, range(len(clusterSilhouetteScores))):
            print("Score for Cluster " + str(j) + ": " + str(score))
        print("Total Score: " +str(np.mean(clusterSilhouetteScores)))

    def old_calculateSilhouette(self):
        clusterSilhouetteScores = []
        for i in range(self.k):  #for each cluster
            aVector = []
            bVector = []
            silhouetteScores = []
            total = 0
            otherTotal = 0
            for j in range(len(self.groups)):  #for each instance
                distances = []
                # otherDistances = []
                otherClustersDistances = [[],[],[],[],[],[],[]]
                for k in range(len(self.features[0])):  #for each row
                    distance = 0
                    for l in range(len(self.features)):  #for each column
                        distanceDelta = 0
                        if j != k:
                            if self.labelTypes[i] == self.REAL:
                                if math.isnan(self.features[l][k]) or math.isnan(self.features[j][k]):
                                    distanceDelta = 1
                                else:
                                    distanceDelta = (self.features[l][k] - self.features[j][k])**2
                            else:
                                if math.isnan(self.features[l][k]) or math.isnan(self.features[j][k]):
                                    distanceDelta = 1
                                elif self.features[l][k] == self.features[j][k]:
                                    distanceDelta = 0
                                else:
                                    distanceDelta = 1
                            distance += distanceDelta
                    if self.groups[k] == i and j!=k:
                        total += 1
                        distances.append((distance))
                    elif j!=k:
                        otherTotal += 1
                        # otherDistances.append(math.sqrt(distance))
                        otherClustersDistances[self.groups[k]].append((distance))
                distances = np.array(distances)
                # otherDistances = np.array(otherDistances)
                aEntry = np.sum(distances)/len(distances) #if len(distances) != 0 else 0
                otherDistanceAverages = []
                for others in otherClustersDistances:
                    if others != []:
                        otherDistanceAverages.append(np.sum(others)/len(others))
                bEntry = np.min(otherDistanceAverages)
                # bEntry = np.min(otherDistances)
                aVector.append(aEntry)
                bVector.append(bEntry)
            for j in range(len(aVector)):
                silhouetteScores.append((bVector[j] - aVector[j])/max(aVector[j],bVector[j]))
            silhouetteScores = np.array(silhouetteScores)
            clusterSilhouetteScores.append(np.mean(silhouetteScores))
        for score, i in zip(clusterSilhouetteScores, range(len(clusterSilhouetteScores))):
            print("Score for Cluster " + str(i) + ": " + str(score))
        print("Total Score: " +str(np.mean(clusterSilhouetteScores)))
