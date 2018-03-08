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
    currLayerDecisionIndex = None
    nextTreeLayers = []
    baseFeatureInfo = []
    finalDecision = None


    def __init__(self, features, labels, numberOfClassificationForAttribute):
        self.nextTreeLayers = []
        self.currLayerDecisionIndex = None
        self.finalDecision = None
        self.baseFeatureInfo = []
        self.numberOfClassificationForAttribute = numberOfClassificationForAttribute
        self.features = features
        self.labels = labels
        self.buildTree(self.features, self.labels, self.baseFeatureInfo)


    def buildTree(self, features, labels, baseFeatureInfo):
        self.buildEmptyInfoList(baseFeatureInfo)
        self.buildFeatureInfo(features, self.baseFeatureInfo)
        bestInfoIndex = self.calculateIndexWithMostInformationGain(baseFeatureInfo)
        self.currLayerDecisionIndex = bestInfoIndex
        partitionedFeatures, partitionedLabels = self.partitionFeaturesAndLabels(features, labels, bestInfoIndex, max(labels) + 1)
        #  partitionedFeatures, partitionedLabels = self.createPartitonedFeaturesAndLabels(features, labels, bestInfoIndex)
        decisionBranches = len(partitionedFeatures)
        for i in range(decisionBranches):
            if len(partitionedFeatures[i]) > 1:
                nextTree = DecisionTree(partitionedFeatures[i], partitionedLabels[i], len(partitionedFeatures[i]))
                self.nextTreeLayers.append(nextTree)
                self.currLayerDecisionIndex = bestInfoIndex
            elif partitionedFeatures[i] != []:
                bestLabelDecisions = []
                maxDecisions = 0
                for j in range(len(partitionedLabels[i])):
                    if partitionedLabels[i][j] + 1 > maxDecisions:
                        # bestLabelDecisions.append(0)
                        maxDecisions = partitionedLabels[i][j] + 1
                for j in range(maxDecisions):
                    bestLabelDecisions.append(0)
                for j in range(len(partitionedLabels[i])):
                    bestLabelDecisions[partitionedLabels[i][j]] += 1
                bestLabelIndex = 0
                for j in range(len(bestLabelDecisions)):
                    if bestLabelDecisions[j] > bestInfoIndex:
                        bestLabelIndex = j
                self.finalDecision = bestLabelIndex


    def partitionFeaturesAndLabels(self, features, labels, bestInfoIndex, decisionCount):
        partitionedFeatures = []
        partitionedLabels = []
        for i in range(decisionCount):
            partitionedFeatures.append([])
            partitionedLabels.append([])
        for i in range(len(features)):
            partition = []
            valueAtBestInfoIndex = None
            for j in range(len(features[i])):
                if j != bestInfoIndex:
                    partition.append(features[i][j])
                else:
                    valueAtBestInfoIndex = features[i][j]
            partitionedFeatures[valueAtBestInfoIndex].append(partition)
            partitionedLabels[valueAtBestInfoIndex].append(labels[i])
        return  partitionedFeatures, partitionedLabels

    #
    # def createPartitonedFeaturesAndLabels(self, features, labels, bestInfoIndex):
    #     partitionedFeatures = []
    #     partitionedLabels = []
    #     classificationCount = 0
    #     for i in range(len(features)):
    #         if features[i][bestInfoIndex] > classificationCount:
    #             classificationCount = features[i][bestInfoIndex] + 1
    #     for i in range(classificationCount + 1):
    #         partitionedFeatures.append([])
    #         partitionedLabels.append([])
    #         for j in range(len(features)):
    #             feature = []
    #             if features[j][bestInfoIndex] == i:  #TODO: Fix this.  Partition correctly
    #                 for k in range(len(features[j])):
    #                     if k != bestInfoIndex:
    #                         feature.append(features[j][k])
    #                 partitionedFeatures[i].append(feature)
    #                 partitionedLabels[i].append(labels[i])
    #     return partitionedFeatures, partitionedLabels

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
        for j in range(len(featuresInfo[0])):
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
        return indexWithBestInfoGain
