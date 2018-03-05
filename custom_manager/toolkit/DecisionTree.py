import math

class DecisionNode:
    decisions = []
    currLayerFeatureInfo = []

    def __init__(self, currLayerFeatureInfo):
        self.currLayerFeatureInfo = currLayerFeatureInfo

class DecisionTree:

    features = None
    labels = None
    maxNumberOfClassificationForAttribute = 10
    baseFeatureInfo = []

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.buildEmptyInfoList(self.baseFeatureInfo)
        self.buildFeatureInfo(self.features, self.baseFeatureInfo)

    def buildEmptyInfoList(self, list):
        for i in range(len(self.features[0])):
            currInfoEntry = []
            for j in range(self.maxNumberOfClassificationForAttribute):
                currInfoEntry.append(0)
            list.append(currInfoEntry)

    def buildFeatureInfo(self, features, emptyInfoList):
        for i in range(len(features)): #each row
            for j in range(len(features[i])): #each row entry
                emptyInfoList[j][features[i][j]] += 1


#         self.features = features
#         self.featureCount = featureCount
#         for i in range(featureCount):
#             self.firstLayer.append(DecisionNode(i))
#             fillFeatureInfo = []
#             for j in range(self.maxNumberOfClassificationForAttribute):
#                 fillFeatureInfo.append(0)
#             self.featureInfo.append(fillFeatureInfo)
#         self.buildTree(self.firstLayer)


# class DecisionNode:
#     children = []
#     index = None
#     currLayerFeatureInfoSize = 0
#     currFeatureInfo = []
#     def __init__(self, index):
#         self.index = index
#         #self.currFeatureInfo = currFeatureInfo
#
#     def calculateNodeInfo(self):
#         total = self.currLayerFeatureInfo
#         currFeatureCount = 0
#         logSum = 0
#         for feature in self.currFeatureInfo:
#             currFeatureCount += feature
#         for feature in self.currFeatureInfo:
#             logSum += (-feature/currFeatureCount) * math.log(feature/currFeatureCount, 2)
#         nodeInfo = (currFeatureCount/total) * logSum
#         return nodeInfo
#
#
#
# class DecisionTree:
#     firstLayerTotal = 0
#     firstLayer = []
#     maxNumberOfClassificationForAttribute = 10
#     featureInfo = []
#     featureCount = 0
#     features = None
#
#     def __init__(self, featureCount, features):
#         self.features = features
#         self.featureCount = featureCount
#         for i in range(featureCount):
#             self.firstLayer.append(DecisionNode(i))
#             fillFeatureInfo = []
#             for j in range(self.maxNumberOfClassificationForAttribute):
#                 fillFeatureInfo.append(0)
#             self.featureInfo.append(fillFeatureInfo)
#         self.buildTree(self.firstLayer)
#
#     def inputTrainingFeatures(self, features):
#         for feature in features:
#             self.featureInfo[feature[0]] += 1
#
#     def buildTree(self, layer):
#         for i in range(len(layer) - 1):  #go through all but the label
#             currNodeInfoList = []
#             for j in range(len(layer) - 1):
#                 if i != j:
#                     nodeInfo = []
#                     for k in range(len(layer[j])):
#                         nodeInfo.append(layer[j][k])
#                     currNodeInfoList.append(nodeInfo)
#             self.buildTree(currNodeInfoList)
#
#     def getIndexWithBestInformationGain(self, layer):
#         infos = []
#         for node in layer:
#             infoGain = node.calculateInfoGain()
#             infos.append(infoGain)
#         bestInfoIndex = None
#         for i in range (layer):
#             bestInfoIndex = i if bestInfoIndex == None or layer[i] < bestInfoIndex else bestInfoIndex
#         return bestInfoIndex
#
