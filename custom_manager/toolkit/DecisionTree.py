import math

class DecisionNode:
    numberOfDecisions = 0
    decisions = []
    def __init__(self, numberOfDecisions):
        self.numberOfDecisions = numberOfDecisions



class DecisionLayer:
    nodes = []
    def __init__(self, nodes):
        self.nodes = nodes

class DecisionTree:

    firstLayer = None

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        maxValues = []
        for i in range(len(features[0])):
            maxValues.append(0)
        for i in range(len(features)):
            for j in range(len(features[i])):
                if maxValues[j] < features[i][j]:
                    maxValues[j] = features[i][j]
        firstNodes = []
        for i in range(len(maxValues)):
            newNode = DecisionNode(maxValues[i])
            firstNodes.append(newNode)
        self.firstLayer = DecisionLayer(firstNodes)
        self.buildTree()


    def buildTree(self):
        pass

#
# class DecisionTree:
#     features = None
#     labels = None
#     currLayerDecisionIndex = None
#     nextTreeLayers = []
#     baseFeatureInfo = []
#     finalDecisions = []
#
#
#     def __init__(self, features, labels):
#         self.nextTreeLayers = []
#         self.currLayerDecisionIndex = 0
#         self.finalDecisions = []
#         self.baseFeatureInfo = []
#         self.features = features
#         self.labels = labels
#         self.buildTree(self.features, self.labels, self.baseFeatureInfo)
#
#
#     def buildTree(self, features, labels, baseFeatureInfo):
#         if self.features == []:
#             return
#         self.buildEmptyInfoList(baseFeatureInfo)
#         self.buildFeatureInfo(features, self.baseFeatureInfo)
#         bestInfoIndex = self.calculateIndexWithMostInformationGain(baseFeatureInfo)
#         self.currLayerDecisionIndex = bestInfoIndex
#         partitionedFeatures, partitionedLabels = self.partitionFeaturesAndLabels(features, labels, bestInfoIndex, max(labels) + 1)
#         #  partitionedFeatures, partitionedLabels = self.createPartitonedFeaturesAndLabels(features, labels, bestInfoIndex)
#         #decisionBranches = len(partitionedFeatures)
#         decisionBranches = 3
#         for i in range(decisionBranches):
#             self.finalDecisions.append([])
#             if len(partitionedFeatures[i]) > 1:
#                 nextTree = DecisionTree(partitionedFeatures[i], partitionedLabels[i])
#                 self.nextTreeLayers.append(nextTree)
#                 self.currLayerDecisionIndex = bestInfoIndex
#             elif partitionedFeatures[i] != []:
#                 self.nextTreeLayers.append([])
#                 bestLabelDecisions = []
#                 maxDecisions = 0
#                 for j in range(len(partitionedLabels[i])):
#                     if partitionedLabels[i][j] + 1 > maxDecisions:
#                         # bestLabelDecisions.append(0)
#                         maxDecisions = partitionedLabels[i][j] + 1
#                 for j in range(int(maxDecisions)):
#                     bestLabelDecisions.append(0)
#                 for j in range(len(partitionedLabels[i])):
#                     bestLabelDecisions[int(partitionedLabels[i][j])] += 1
#                 bestLabelIndex = 0
#                 for j in range(len(bestLabelDecisions)):
#                     if bestLabelDecisions[j] > bestLabelDecisions[bestLabelIndex]:
#                         bestLabelIndex = j
#                 self.finalDecisions[i].append(bestLabelIndex)
#             else:
#                 self.nextTreeLayers.append([])
#
#
#     def partitionFeaturesAndLabels(self, features, labels, bestInfoIndex, decisionCount):
#         if features == []:
#             return
#         partitionedFeatures = []
#         partitionedLabels = []
#         for i in range(int(decisionCount)):
#             partitionedFeatures.append([])
#             partitionedLabels.append([])
#         for i in range(len(features)):
#             partition = []
#             valueAtBestInfoIndex = 0
#             for j in range(len(features[i])):
#                 if j != bestInfoIndex:
#                     partition.append(features[i][j])
#                 else:
#                     test2 = (int(features[i][j]))
#                     valueAtBestInfoIndex = int(features[i][j])
#             test = (int(features[i][j]))
#             try:
#                 partitionedFeatures[int(valueAtBestInfoIndex)].append(partition)
#                 partitionedLabels[int(valueAtBestInfoIndex)].append(labels[i])
#             except:
#                 partitionedFeatures[0].append([0])
#                 partitionedLabels[0].append([0])
#
#         return partitionedFeatures, partitionedLabels
#
#     def buildEmptyInfoList(self, list):
#         maxFeatureValue = 0
#         for feature in self.features:
#             for entry in feature:
#                 if entry > maxFeatureValue:
#                     maxFeatureValue = entry
#         for i in range(len(self.features[0])):
#             currInfoEntry = []
#             for j in range(int(maxFeatureValue + 1)):
#                 currInfoEntry.append(0)
#             list.append(currInfoEntry)
#
#     def buildFeatureInfo(self, features, emptyInfoList):
#         for i in range(len(features)): #each row
#             for j in range(len(features[i])): #each row entry
#                 testIndex = int(features[i][j])
#                 emptyInfoList[j][testIndex] += 1
#
#     def calculateIndexWithMostInformationGain(self, featuresInfo):
#         information = []
#         totalForAllFeatures = 0
#         for i in range(len(featuresInfo)):
#             for j in range(len(featuresInfo[i])):
#                 totalForAllFeatures += featuresInfo[i][j]
#         for j in range(len(featuresInfo[0])):
#             totalForThisFeature = 0
#             for i in range(len(featuresInfo)):
#                 totalForThisFeature += featuresInfo[i][j]
#             logSum = 0
#             for i in range(len(featuresInfo)):
#                 logSum += -(featuresInfo[i][j] / totalForThisFeature) * math.log(featuresInfo[i][j] / totalForThisFeature, 2) if featuresInfo[i][j] != 0 else 0
#             information.append((totalForThisFeature / totalForAllFeatures) * logSum)
#         valueForBestInfoGain = None
#         indexWithBestInfoGain = None
#         for i in range(len(information)):
#             if indexWithBestInfoGain == None or information[i] < valueForBestInfoGain:
#                 valueForBestInfoGain = information[i]
#                 indexWithBestInfoGain = i
#         if len(self.features[0]) <= indexWithBestInfoGain:
#             indexWithBestInfoGain = 0
#         return indexWithBestInfoGain
#
#     def getDecision(self, row):
#         currLayerDecisionFeature = row[self.currLayerDecisionIndex]
#         try:
#             if self.finalDecisions[currLayerDecisionFeature] != []:
#                 value = self.finalDecisions[currLayerDecisionFeature][0]
#                 return value
#             else:
#                 partitionedDecision = []
#                 for i in range(len(row)):
#                     if i != self.currLayerDecisionIndex:
#                         partitionedDecision.append(row[i])
#                 return self.nextTreeLayers[currLayerDecisionFeature].getDecision(partitionedDecision)
#
#         except:
#             print("FAILED TO GET DECISION")
#
#
#     def train(self, features, labels):
#         pass
#
#     def predict(self, row):
#         pass
