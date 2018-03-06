import math

class DecisionNode:
    decisions = []
    currLayerFeatureInfo = []

    def __init__(self, currLayerFeatureInfo):
        self.currLayerFeatureInfo = currLayerFeatureInfo

class DecisionTree:
    features = None
    labels = None
    numberOfClassificationForAttribute = None
    baseFeatureInfo = []
    def __init__(self, features, labels, numberOfClassificationForAttribute):
        self.numberOfClassificationForAttribute = numberOfClassificationForAttribute
        self.features = features
        self.labels = labels
        self.buildEmptyInfoList(self.baseFeatureInfo)
        self.buildFeatureInfo(self.features, self.baseFeatureInfo)
        self.calculateIndexWithMostInformationGain(self.baseFeatureInfo)


    def buildEmptyInfoList(self, list):
        for i in range(len(self.features[0])):
            currInfoEntry = []
            for j in range(self.numberOfClassificationForAttribute):
                currInfoEntry.append(0)
            list.append(currInfoEntry)

    def buildFeatureInfo(self, features, emptyInfoList):
        for i in range(len(features)): #each row
            for j in range(len(features[i])): #each row entry
                emptyInfoList[j][features[i][j]] += 1

    def calculateIndexWithMostInformationGain(self, featuresInfo):
        information = []
        totalForAllFeatures = 0
        for i in range(len(featuresInfo)):
            for j in range(len(featuresInfo[i])):
                totalForAllFeatures += featuresInfo[i][j]
        for j in range(len(featuresInfo[i])):
            totalForThisFeature = 0
            for i in range(len(featuresInfo)):
                totalForThisFeature += featuresInfo[i][j]
            logSum = 0
            for i in range(len(featuresInfo)):
                logSum += -(featuresInfo[i][j] / totalForThisFeature) * math.log(featuresInfo[i][j] / totalForThisFeature, 2) if featuresInfo[i][j] != 0 else 0
            information.append((totalForThisFeature / totalForAllFeatures) * logSum)
        valueForBestInfoGain = None
        indexWithBestInfoGain = None
        for i in range(len(information)):
            if indexWithBestInfoGain == None or information[i] < valueForBestInfoGain:
                valueForBestInfoGain = information[i]
                indexWithBestInfoGain = i
        # print(information)
        # print(indexWithBestInfoGain)
        return indexWithBestInfoGain
