import math

class Decision:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

class DecisionNode:
    numberOfDecisions = 0
    layersForDecisions = []
    countsForDecisions = []
    decisions = []
    def __init__(self, numberOfDecisions):
        self.numberOfDecisions = numberOfDecisions + 1 #total count, not indexed number
        self.layersForDecisions = []
        self.countsForDecisions = []
        self.decisions = []
        for i in range(self.numberOfDecisions):
            self.countsForDecisions.append(0)

    def addLayerForDecision(self, layer):
        self.layersForDecisions.append(layer)



class DecisionLayer:

    nodeCount = [0]

    # treeDepth = [0]

    def __init__(self, nodes, features, labels, nodeIfNoDecisions):
        # self.treeDepth[0] += 1
        self.nodeCount[0] += len(nodes)
        self.finalDecisions = None
        if (len(features) == 0):
            baseFinalDecisions = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
            for i in range(len(nodeIfNoDecisions.decisions)):
                baseFinalDecisions[int(nodeIfNoDecisions.decisions[i].label)][int(nodeIfNoDecisions.decisions[i].feature)] += 1
            self.finalDecisions = baseFinalDecisions
            return
        self.features = features
        self.labels = labels
        self.numberOfLabels = int(max(labels)) + 1 if len(labels) > 0 else 0
        self.nodes = nodes
        if len(features[0]) == 1:
            self.finalDecisions = self.calculateFinalDecision()
            return
        else:
            self.finalDecisions = None
        self.setupFeatureDataForDecisions()
        self.branchIndex = self.calculateBranch()
        self.partitionForBranch()

    def setupFeatureDataForDecisions(self):
        for i in range(len(self.features)):
            for j in range(len(self.features[i])):
                currDecisions = self.nodes[j].decisions
                currDecisions.append(Decision(self.features[i][j], self.labels[i]))
                # self.nodes[j].countsForDecisions[int(self.features[i][j])] += 1

    def calculateBranch(self):
        informationGains = []
        totalEntropy = 0
        labelCounts = []
        for i in range(self.numberOfLabels):
            labelCounts.append(0)
        for i in range(len(self.labels)):
            labelCounts[int(self.labels[i])] += 1
        for i in range(len(labelCounts)):
            p = (labelCounts[i]/len(self.labels)) if len(self.labels) != 0 else 0
            totalEntropy += -p * math.log(p, 2) if p != 0 else 0
        for node in self.nodes:
            featuresForEachLabel = []
            for i in range(self.numberOfLabels):
                featureInfo = []
                for j in range(node.numberOfDecisions):
                    featureInfo.append(0)
                for j in range(len(node.decisions)):
                    if node.decisions[j].label == i:
                        featureInfo[int(node.decisions[j].feature)] += 1
                featuresForEachLabel.append(featureInfo)
            logSum = 0
            for j in range(len(featuresForEachLabel[0])):
                logSumForThisFeature = 0
                totalForFeature = 0
                for i in range(len(featuresForEachLabel)):
                    totalForFeature += featuresForEachLabel[i][j]
                for i in range(len(featuresForEachLabel)):
                    prob = featuresForEachLabel[i][j]/totalForFeature if totalForFeature != 0 else 0
                    logSumForThisFeature += -prob * math.log(prob, 2)  if prob != 0 else 0
                logSumForThisFeature = logSumForThisFeature * totalForFeature/len(node.decisions)
                logSum += logSumForThisFeature
            attributeInformationGain = totalEntropy - logSum
            informationGains.append(attributeInformationGain)
        bestInfoGainIndex = 0
        for i in range(len(informationGains)):
            if informationGains[i] > informationGains[bestInfoGainIndex]:
                bestInfoGainIndex = i
        return bestInfoGainIndex

    def calculateFinalDecision(self):  #TODO: Check functionality
        oneDimFeatureArray = []
        for i in range(len(self.features)):
            oneDimFeatureArray.append(self.features[i][0])
        labelCountsForFeatures = []
        maxFeatureValue = 0
        for i in range(len(self.features)):
            if self.features[i][0] > maxFeatureValue:
                maxFeatureValue = int(self.features[i][0])
        for i in range(0,4):  #TODO: SET TO NUMBER OF FEATURES
            labelsArray = []
            for j in range(0,4):  #TODO: SET TO NUMBER OF LABELS
                labelsArray.append(0)
            labelCountsForFeatures.append(labelsArray)
        for i in range(len(oneDimFeatureArray)):
            labelCountsForFeatures[int(oneDimFeatureArray[i])][int(self.labels[i])] += 1
        return labelCountsForFeatures

    def partitionForBranch(self):
        nodeIfNoDecisions = None
        branchNode = self.nodes[self.branchIndex]
        for decisionIndex in range(branchNode.numberOfDecisions):
            partitionedFeatures = []
            partitionedLabels = []
            for i in range(len(self.features)):
                feature = []
                if self.features[i][self.branchIndex] == decisionIndex:
                    for j in range(len(self.features[i])):
                        if j != self.branchIndex:
                            feature.append(self.features[i][j])
                    partitionedFeatures.append(feature)
                    partitionedLabels.append(self.labels[i])
            newNodes = []
            for i in range(len(self.nodes)):
                if i != self.branchIndex:
                    newNodes.append(DecisionNode(self.nodes[i].numberOfDecisions - 1))
            if partitionedFeatures == []:
                for i in range(len(self.nodes)):
                    if i == self.branchIndex:
                        nodeIfNoDecisions = self.nodes[i]
            branchNode.layersForDecisions.append(DecisionLayer(newNodes, partitionedFeatures, partitionedLabels, nodeIfNoDecisions))

    def getDecision(self, feature):
        if self.finalDecisions != None:
            bestLabel = 0
            for i in range(len(self.finalDecisions[0])):  #for each label
                if self.finalDecisions[int(feature[0])][i] > self.finalDecisions[int(feature[0])][bestLabel]:
                    bestLabel = i
            return bestLabel
        else:
            partitionedFeature = []
            for i in range(len(feature)):
                if i != self.branchIndex:
                    partitionedFeature.append(feature[i])
            return self.nodes[int(self.branchIndex)].layersForDecisions[int(feature[int(self.branchIndex)])].getDecision(partitionedFeature)
            # return self.getDecision(partitionedFeature)

class DecisionTree:

    firstLayer = None

    treeDepth = [0]

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
            newNode = DecisionNode(int(maxValues[i]))
            firstNodes.append(newNode)
        self.firstLayer = DecisionLayer(firstNodes, features, labels, None)  #builds the tree

    def getDecision(self, feature):
        branchIndex = self.firstLayer.branchIndex
        firstSplitNode = self.firstLayer.nodes[branchIndex]
        splitLayerForFeature =  firstSplitNode.layersForDecisions[int(feature[branchIndex])]
        secondBranchIndex = splitLayerForFeature.branchIndex
        secondSplitNode = splitLayerForFeature.nodes[secondBranchIndex]
        secondSplitLayerForFeature = secondSplitNode.layersForDecisions[int(feature[secondBranchIndex])]
        thirdBranchIndex = secondSplitLayerForFeature.branchIndex
        thirdSplitNode = secondSplitLayerForFeature.nodes[thirdBranchIndex]
        thirdSplitLayerForFeature = thirdSplitNode.layersForDecisions[int(feature[thirdBranchIndex])]
        labelCounts = [0,0,0,0]
        for label in splitLayerForFeature.labels:
            labelCounts[int(label)] += 0
        # for label in secondSplitLayerForFeature.labels:
        #     labelCounts[int(label)] += 0
        # if thirdSplitLayerForFeature.finalDecisions == None:
        #     for label in thirdSplitLayerForFeature.labels:
        #         labelCounts[int(label)] += 0
        # else:
        #     for i in range(len(thirdSplitLayerForFeature.finalDecisions)):
        #         for j in range(len(thirdSplitLayerForFeature.finalDecisions[i])):
        #             if thirdSplitLayerForFeature.finalDecisions[i][j] != 0:
        #                 return j
        bestLabel = 0
        for i in range(len(labelCounts)):
            if labelCounts[i] > labelCounts[bestLabel]:
                bestLabel = i
        return bestLabel


        # above is the shallow classification finder
        return self.firstLayer.getDecision(feature)
