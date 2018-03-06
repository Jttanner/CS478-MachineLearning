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
        self.buildTree(self.features, self.labels, self.baseFeatureInfo)

    def buildTree(self, features, labels, baseFeatureInfo):
        self.buildEmptyInfoList(baseFeatureInfo)
        self.buildFeatureInfo(features, self.baseFeatureInfo)
        bestInfoIndex = self.calculateIndexWithMostInformationGain(baseFeatureInfo)
        partitionedFeatures, partitionedLabels = self.createPartitonedFeaturesAndLabels(features, labels, bestInfoIndex)

    def createPartitonedFeaturesAndLabels(self, features, labels, bestInfoIndex):
        partitionedFeatures = []
        partitionedLabels = []
        classificationCount = 0
        for i in range(len(features)):
            if features[i][bestInfoIndex] > classificationCount:
                classificationCount = features[i][bestInfoIndex]
        for i in range(classificationCount):
            partitionedFeatures.append([])
            partitionedLabels.append([])
            for j in range(len(features)):
                feature = []
                if features[j][bestInfoIndex] == i:
                    for k in range(len(features[i])):
                        feature.append(features[j][k])
                    partitionedFeatures[i].append(feature)
                    partitionedLabels[i].append(labels[i])
        return partitionedFeatures, partitionedLabels


        # for i in range(len(features)):
        #     partitionedFeature = []
        #     for j in range(len(features[i]) - 1):
        #         if j != bestInfoIndex:
        #             partitionedFeature.append(partitionedFeatures[i][j])
        #     partitionedLabels.append()
        # return partitionedFeatures, partitionedLabels


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
