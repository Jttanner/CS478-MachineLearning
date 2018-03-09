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

    def __init__(self, nodes, features, labels):
        if (len(features) == 0):
            self.finalDecisions = [0, 0, 0, 0, 0, 0, 0 ,0]
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
        decisionsInfo = []
        for node in self.nodes:
            totalForDecisions = len(node.decisions)  #OKAY
            nodefeaturesForEachLabel = []
            for i in range(self.numberOfLabels):
                nodefeaturesForEachLabel.append([])
            for decision in node.decisions:
                nodefeaturesForEachLabel[int(decision.label)].append(decision.feature)
            logSum = 0
            temp = 0
            for labelDivision, i in zip(nodefeaturesForEachLabel, range(len(nodefeaturesForEachLabel))):
                labelCounts = []
                for j in range(node.numberOfDecisions):
                    labelCounts.append(0)
                for feature in labelDivision:
                    labelCounts[int(feature)] += 1
                instanceOverTotal = len(labelDivision)/totalForDecisions
                for j in range(node.numberOfDecisions):
                    pNum = labelCounts[j]
                    pDenom = len(labelDivision)
                    infoS = (0-1) * (pNum/pDenom) * math.log(pNum/pDenom,2) if (pNum != 0 and pDenom != 0) else 0
                    logSum += infoS
                    temp += infoS
                    i = 4
                    # Si = labelDivision[j]
                    # S =  totalForDecisions
                    # pNum = labelCounts[j]
                    # pDenom = len(labelDivision)
                    # infoS = -pNum/pDenom*math.log(pNum/pDenom,2) if pNum != 0 else 0
                    # tempInfo = (Si/S) * infoS
                    # logSum += tempInfo
                    # labelCountOverInstance = (labelCounts[j]/len(nodefeaturesForEachLabel[i])) if (len(nodefeaturesForEachLabel[i]) != 0) else 0
                    # addMe = instanceOverTotal * (-1) * labelCountOverInstance * math.log(labelCountOverInstance , 2) if labelCountOverInstance != 0 else 0
                    # logSum += addMe
            logSum = logSum * instanceOverTotal
            decisionsInfo.append(logSum)
        minValue = min(decisionsInfo)
        bestInfoIndex = 0
        for i in range(len(decisionsInfo)):
            if minValue == decisionsInfo[i]:
                bestInfoIndex = i
        return bestInfoIndex

    def calculateFinalDecision(self):  #TODO: Check functionality
        labelCountsForEachFeature = []
        for i in range(self.nodes[0].numberOfDecisions):
            labelCountsForEachFeature.append([])
            for j in range(self.numberOfLabels):
                labelCountsForEachFeature[i].append(0)
        for feature, i in zip(self.features, range(len(self.features))):
            for label, j in zip(self.labels, range(len(self.features))):
                labelCountsForEachFeature[int(feature[0])][int(label)] += 1
        finalDecisions = []
        for i in range(len(labelCountsForEachFeature)):
            bestLabel = 0
            for j in range(len(labelCountsForEachFeature[i])):
                if labelCountsForEachFeature[i][j] > labelCountsForEachFeature[i][bestLabel]:
                    bestLabel = j
            finalDecisions.append(bestLabel)
        return finalDecisions

    def partitionForBranch(self):
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
            branchNode.layersForDecisions.append(DecisionLayer(newNodes, partitionedFeatures, partitionedLabels))

    def getDecision(self, partitionedInput, featureAtDecisionIndex):
        nextNode = self.nodes[self.branchIndex]
        nextLayer = nextNode.layersForDecisions[int(featureAtDecisionIndex)]
        if nextLayer.finalDecisions != None:
            return nextLayer.finalDecisions[int(featureAtDecisionIndex)]
        else:
            # nextNode = self.nodes[self.branchIndex]
            # nextLayer = nextNode.layersForDecisions[int(featureAtDecisionIndex)]
            nextPartition = []
            for i in range(len(partitionedInput)):
                if i != self.branchIndex:
                    nextPartition.append(partitionedInput[i])
            return nextLayer.getDecision(nextPartition, partitionedInput[nextLayer.branchIndex])

class DecisionTree:

    firstLayer = None

    def __init__(self, features, labels):
        self.nodeCount = 0
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
            self.nodeCount += 1
        self.firstLayer = DecisionLayer(firstNodes, features, labels)  #builds the tree

    def getDecision(self, feature):
        partitionedFeature = []
        for i in range(len(feature)):
            if i != self.firstLayer.branchIndex:
                partitionedFeature.append(feature[i])
        return self.firstLayer.getDecision(partitionedFeature, feature[self.firstLayer.branchIndex])

