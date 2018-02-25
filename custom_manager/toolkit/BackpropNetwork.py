from random import uniform
from math import exp


class Node:
    forwardConnections = []
    backConnections = []
    forwardWeights = []
    forwardWeightDeltas = []
    net = None
    output = None
    isBias = None
    delta = None
    fPrimeNet = None
    momentum = .9


    def __init__(self, isBias):
        self.forwardConnections = []
        self.backConnections = []
        self.forwardWeights = []
        self.forwardWeightDeltas = []
        self.output = 0
        self.isBias = isBias
        self.net = 0
        self.delta = 0
        self.fPrimeNet = 0

class OutputNode:
    output = None
    backConnections = []
    net = None
    delta = None
    fPrimeNet = None
    isBias = False

    def __init__(self):
        self.output = 0
        self.net = 0
        self.delta = 0
        self.fPrimeNet = 0
        self.backConnections = []



class BackpropNetwork:

    firstNodes = []
    outputs = []
    learningRate = None
    layerSizesArray = None

    targets = []

    # def copyRec(self, layer, nextLayer):
    #     if type(nextLayer[0]) is OutputNode:
    #         pass
    #     else:
    #         self.copyRec(nextLayer, nextLayer.fowardConnections)
    #     for node in layer:
    #         newNode = Node(node.isBias)
    #         for weight, node in zip(node.forwardWeights, node.forwardConnections):
    #             newNode.forwardWeights.append(weight)
    #             newNode.forwardConnections.append(node)
    #
    # # copy constructor
    # def __init__(self, copyMe):
    #     for outputNode in copyMe.outputNodes:
    #         newNode = OutputNode()
    #         self.outputs.append(newNode)
    #     self.copyRec(copyMe.firstNodes, copyMe.firstNodes.forwardConnections)


    def __init__(self, numberOfLayers, learningRate, layerSizesArray):
        self.targets = []
        self.layerSizesArray = layerSizesArray
        self.learningRate = learningRate
        self.firstNodes = []
        self.outputs = []
        for i in range(layerSizesArray[len(layerSizesArray) - 1]):
            self.outputs.append(OutputNode())
        for i in range(layerSizesArray[0]):
            self.firstNodes.append(Node(False))
            #self.firstNodes[i].output = features[i]
        nextLayer = self.buildNetwork(numberOfLayers - 1, 1)
        for node in self.firstNodes:
            for nextNode in nextLayer:
                node.forwardConnections.append(nextNode)
                node.forwardWeights.append(self.calculateInitialNodeForwardWeight())
                node.forwardWeightDeltas.append(0)
        biasNode = Node(True)
        biasNode.forwardConnections = self.firstNodes[0].forwardConnections
        biasNode.output = 1
        for i in range(len(biasNode.forwardConnections)):
            biasNode.forwardWeights.append(self.calculateInitialNodeForwardWeight())
            biasNode.forwardWeightDeltas.append(0)
        self.firstNodes.append(biasNode)


    def calculateInitialNodeForwardWeight(self):
        initialWeight = uniform(-.1, .1)
        while initialWeight == 0:  # making sure that the weights can never initalize to 0, but stay close
            initialWeight = uniform(-.1, .1)
        #initialWeight = 1
        return initialWeight

    def buildNetwork(self, layersRemaining, layerNumber):
        if layersRemaining > 0:
            newLayer = []
            nextLayer = self.buildNetwork(layersRemaining - 1, layerNumber + 1)
            for i in range(self.layerSizesArray[layerNumber]):
                newLayer.append(Node(False))
                for node in nextLayer:
                    newLayer[i].forwardConnections.append(node)
                    newLayer[i].forwardWeights.append(self.calculateInitialNodeForwardWeight())
                    newLayer[i].forwardWeightDeltas.append(0)
            biasNode = Node(True)
            biasNode.forwardConnections = newLayer[0].forwardConnections
            biasNode.output = 1
            for i in range(len(biasNode.forwardConnections)):
                biasNode.forwardWeights.append(self.calculateInitialNodeForwardWeight())
                biasNode.forwardWeightDeltas.append(0)
            newLayer.append(biasNode)
            return newLayer
        else:
            return self.outputs

    def processInput(self, features, targets, isTraining):
        self.targets = targets
        #reset outputs, nets, etc of entire network
        self.resetNetwork()
        #set input
        for feature, i in zip(features, range(len(features))):
            self.firstNodes[i].output = feature
        #calculate nets
        self.calculateOutput()
        #calculate deltas

        #update weights if its training
        if isTraining:
            self.calculateDeltas()
            self.updateWeightDeltas()
        return self.outputs

    def resetNetwork(self):
        self.resetNetworkRec(self.firstNodes)

    def resetNetworkRec(self, layer):
        for node in layer:
            node.output = 0 if not node.isBias else 1
            node.net = 0
            node.delta = 0
            node.fPrimeNet = 0
            # for i in range(len(node.forwardWeights)):
            # node.forwardWeights[i] = self.calculateInitialNodeForwardWeight()
        if type(layer[0]) is not OutputNode:
            self.resetNetworkRec(layer[0].forwardConnections)


    def calculateOutput(self):
        self.calculateOutputRec(self.firstNodes[0].forwardConnections, self.firstNodes)

    def calculateOutputRec(self, layer, lastLayer):
        for jNode, j in zip(layer, range(len(layer))):
            for iNode in lastLayer:
                jNode.net += iNode.forwardWeights[j] * iNode.output #if not iNode.isBias else iNode.forwardWeights[j]
            # 1/(1+e^-net)
            jNode.output = 1/(1+exp(-jNode.net)) if not jNode.isBias else jNode.output
            jNode.fPrimeNet = jNode.output*(1 - jNode.output) if not jNode.isBias else 0
        if type(layer[0]) is not OutputNode:
            self.calculateOutputRec(layer[0].forwardConnections, layer)

    def calculateDeltas(self):
        #calculate output node deltas
        self.calculateOutputDeltas()
        #calculate hidden node deltas
        self.calculateHiddenDeltas()

    def calculateOutputDeltas(self):
        for outputNode, i in zip(self.outputs, range(len(self.outputs))):
            outputNode.delta = (self.targets[i] - outputNode.output) * outputNode.fPrimeNet

    def calculateHiddenDeltas(self):
        self.calculateHiddenDeltasRec(self.firstNodes[0].forwardConnections)

    def calculateHiddenDeltasRec(self, layer):
        if type(layer[0]) is not OutputNode:
            self.calculateHiddenDeltasRec(layer[0].forwardConnections)
            for j in range(len(layer)):
                for k in range(len(layer[0].forwardConnections)):
                    layer[j].delta += (layer[j].forwardConnections[k].delta * layer[j].forwardWeights[k]) * layer[j].fPrimeNet

    def updateWeightDeltas(self):
        self.updateWeightDeltasRec(self.firstNodes)

    def updateWeightDeltasRec(self, layer):
        if type(layer[0]) is not OutputNode:
            self.updateWeightDeltasRec(layer[0].forwardConnections)
            for iNode in layer:
                for j in range(len(iNode.forwardConnections)):
                    weightDelta = self.learningRate * iNode.output * iNode.forwardConnections[j].delta
                    momentumFactor = (iNode.momentum * iNode.forwardWeightDeltas[j])
                    iNode.forwardWeights[j] += weightDelta + momentumFactor
                    iNode.forwardWeightDeltas[j] = weightDelta


