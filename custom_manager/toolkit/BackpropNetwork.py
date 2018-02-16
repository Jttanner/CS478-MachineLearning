from random import uniform
from math import exp

class Node:
    forwardConnections = []
    backConnections = []
    forwardWeights = []
    output = None

    hidden = None

    def __init__(self, hidden):
        self.forwardConnections = []
        self.backConnections = []
        self.forwardWeights = []
        self.hidden = hidden
        self.output = 0



class OutputNode:
    output = None
    backConnections = []

    def __init__(self):
        pass

class BackpropNetwork:

    firstNodes = []
    outputs = []
    learningRate = None


    def __init__(self, numberOfLayers, features, learningRate):
        self.learningRate = learningRate
        self.firstNodes = []
        self.outputs = []
        for i in range(len(features)):
            self.firstNodes.append(Node(False))
            self.firstNodes[i].output = features[i]
            self.outputs.append(OutputNode())
        nextLayer = self.buildNetwork(numberOfLayers)
        for node in self.firstNodes:
            for nextNode in nextLayer:
                node.forwardConnections.append(nextNode)

    def buildNetwork(self, layersRemaining,):
        if layersRemaining > 0:
            newLayer = []
            nextLayer = self.buildNetwork(layersRemaining - 1)
            for i in range(len(nextLayer)):
                newLayer.append(Node(True))
                for node in nextLayer:
                    newLayer[i].forwardConnections.append(node)
                    initialWeight = uniform(-.1, .1)
                    while initialWeight == 0:  # making sure that the weights can never initalize to 0, but stay close
                        initialWeight = uniform(-.1, .1)
                    newLayer[i].forwardWeights.append(initialWeight)
                    node.backConnections.append(newLayer[i])
            return newLayer
        else:
            return self.outputs

    # OUTPUT NODE
    # deltaj = (targj-outputj)*(outputj*(1-outputj)
    # deltawij = learningRate*outputi*deltaj

    # HIDDEN NODE
    # deltaj = sum(deltak*wjk)*f'(netj)

    def updateWeights(self, layer, target):
        nextLayer = layer[0].forwardConnections
        deltas = []
        if type(layer[0]) == type(nextLayer[0]): #next layer is hidden layer
            for jNode in nextLayer:
                for kNode in nextLayer[0].forwardConnections:
                    pass
        else: #next layer is output layer
            for kNode in nextLayer:
                delta = (target - kNode.output)*(kNode.output*(1-kNode.output))
                delta.appends(delta)
            for iNode in layer: #i layer
                for jNode, j in zip(iNode.forwardConnections, range(len(iNode.forwardConnections))):
                    iNode.forwardWeights[j] = self.learningRate * iNode.output * deltas[j]




    def calculateSingleOutput(self, layer, destinationIndex):
        net = 0
        for node in layer:
            net += node.forwardWeights[destinationIndex] * node.output
        node.forwardWeights[destinationIndex].output = 1/(1+exp(-net))
        return node.forwardWeights[destinationIndex].output

    #like perceptron toward each next node with output = 1/(1+exp(-net))
    def processInput(self, features, layer, isTraining):
        nextLayerFeatures = []
        for i in range(len(layer[0].forwardConnections)): #TODO: error handling
            output = self.calculateSingleOutput(layer, i)
            nextLayerFeatures.append(output)
        if type(layer[0]) == type(layer[0].forwardConnections[0]):
            self.processInput(nextLayerFeatures, layer[0].forwardConnections)
        if isTraining:
            self.updateWeights(layer)


    def setInput(self, features):
        for node, feature in zip(self.firstNodes, features):
            node.output = feature


    #predict a single case
    def predict(self, features, label):
        pass

