from random import uniform
from math import exp

class Node:
    forwardConnections = []
    backConnections = []
    forwardWeights = []
    net = None
    output = None
    isBias = None
    delta = None
    fPrimeNet = None


    def __init__(self, isBias):
        self.forwardConnections = []
        self.backConnections = []
        self.forwardWeights = []
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

    #TODO: initialize in appropriate location
    targets = []


    def __init__(self, numberOfLayers, features, learningRate, layerSizesArray, targets):
        self.targets = targets
        self.layerSizesArray = layerSizesArray
        self.learningRate = learningRate
        self.firstNodes = []
        self.outputs = []
        for i in range(layerSizesArray[len(layerSizesArray) - 1]):
            self.outputs.append(OutputNode())
        for i in range(layerSizesArray[0]):
            self.firstNodes.append(Node(False))
            self.firstNodes[i].output = features[i]
        nextLayer = self.buildNetwork(numberOfLayers - 1, 1)
        for node in self.firstNodes:
            for nextNode in nextLayer:
                node.forwardConnections.append(nextNode)
                node.forwardWeights.append(self.calculateInitialNodeForwardWeight())

    # numberOfNodesForLAyer
    # 3
    # 4
    # 3

    # numberoflayers = 3
    # buildNetwork(le)
    #   if number of
    #   nextLayer = buildNetwork(numberOfLayers - 1, numberOfNodesForLayer)
    #   for node in currLayer:
    #       for forwardnode, i in zip(nextLayer, range(len(forwardNodes)):
    #           node.forwardConnections[i] = forwardNode
    #

    #Process input
    #net
    #

    def calculateInitialNodeForwardWeight(self):
        initialWeight = uniform(-.1, .1)
        while initialWeight == 0:  # making sure that the weights can never initalize to 0, but stay close
            initialWeight = uniform(-.1, .1)
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
            biasNode = Node(True)
            biasNode.forwardConnections = newLayer[0].forwardConnections
            biasNode.output = 1
            for i in range(len(biasNode.forwardConnections)):
                biasNode.forwardWeights.append(self.calculateInitialNodeForwardWeight())
            newLayer.append(biasNode)
            return newLayer
        else:
            return self.outputs

    def processInput(self, features):
        #reset outputs, nets, etc of entire network
        self.resetNetwork()
        #set input
        for feature, i in zip(features, range(len(features))):
            self.firstNodes[i].output = feature
        #calculate nets
        self.calculateOutput()

    def resetNetwork(self):
        self.resetNetworkRec(self.firstNodes)

    def resetNetworkRec(self, layer):
        if type(layer[0]) is not OutputNode:
            for node in layer:
                node.output = 0
                node.net = 0
                node.delta = 0
                for i in range(len(node.forwardWeights)):
                    node.forwardWeights[i] = self.calculateInitialNodeForwardWeight()
            self.resetNetworkRec(layer[0].forwardConnections)

    def calculateOutput(self):
        self.calculateOutputRec(self.firstNodes[0].forwardConnections, self.firstNodes)

    def calculateOutputRec(self, layer, lastLayer):
        for jNode, j in zip(layer, range(len(layer))):
            for iNode in lastLayer:
                jNode.net += iNode.forwardWeights[j] * iNode.output
            # 1/(1+e^-net)
            jNode.output = 1/(1+exp(-jNode.net))
            jNode.fPrimeNet = jNode.output*(1 - jNode.output)
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



    #calculate net

    #calculate output

    #calculate deltas

    #calculate delta weights

    # OUTPUT NODE
    # deltaj = (targj-outputj)*(outputj*(1-outputj)
    # deltawij = learningRate*outputi*deltaj

    # HIDDEN NODE
    # deltaj = sum(deltak*wjk)*f'(netj)
    # deltawij = learningRate*outputi*deltaj

    # #TODO: check target logic
    # def computeDelta(self, node, target, nextIsHidden):
    #     nextLayer = node.forwardConnections
    #     if nextIsHidden:
    #         deltaj = 0
    #         for kNode, k in zip(node.forwardConnections, range(len(node.forwardConnections))):
    #             deltak = self.computeDelta(kNode, self.targets, 1 if type(kNode) != type(node) else 0)
    #             deltaj += deltak * node.forwardWeights[k]
    #         return deltaj
    #     else:
    #         return (target - node.output) * (node.output * (1 - node.output))
    #
    # def updateWeights(self, layer, target): #TODO: dont update weights until im done calculating deltas
    #     for iNode in layer:
    #         nextLayer = layer[0].forwardConnections
    #         jDeltas = []
    #         isHidden = False #TODO: check if I can delete this
    #         for jNode, j in zip(nextLayer, range(len(nextLayer))):
    #             if (type(layer[0]) == type(nextLayer[0])):
    #                 isHidden = True
    #             else:
    #                 isHidden = False
    #             jDeltas.append(self.computeDelta(jNode, target, isHidden))
    #             iNode.forwardConnections[j] = jDeltas[j]
    #
    # def calculateSingleOutput(self, layer, destinationIndex):
    #     net = 0
    #     for node in layer:
    #         net += node.forwardWeights[destinationIndex] * node.output
    #     layer[0].forwardWeights[destinationIndex].output = 1/(1+exp(-net))
    #     return layer[0].forwardWeights[destinationIndex].output
    #
    # #like perceptron toward each next node with output = 1/(1+exp(-net))
    # def processInput(self, features, target, layer, isTraining):
    #     self.initInput(features)
    #     self.processInputRec(target, layer)
    #     if isTraining:
    #         self.updateWeights(layer, target)
    #
    # def processInputRec(self, target, layer):
    #     for i in range(len(layer[0].forwardConnections)): #TODO: error handling
    #         self.calculateSingleOutput(layer, i)
    #     if type(layer[0]) == type(layer[0].forwardConnections[0]):
    #         self.processInputRec(target, layer[0].forwardConnections)
    #
    # def initInput(self, features):
    #     self.resetNetwork(self.firstNodes)
    #     for node, feature in zip(self.firstNodes, features):
    #         node.output = feature
    #
    # def resetNetwork(self, layer):
    #     for node in layer:
    #         node.output = None
    #     if layer[0].forwardConnections[0] != None:
    #         self.resetNetwork(layer.forwardConnections)
    #
    # def calculateNetworkOutput(self):
    #     outputIndex = None
    #     greatestOutput = 0
    #     for i in range(len(self.outputs)):
    #         if self.outputs[i].output > greatestOutput:
    #             outputIndex = i
    #     if i != None:
    #         return i
    #     else:
    #         return 0
    #
    #
    #
    # #predict a single case
    # def predict(self, features, target, isTraining):
    #     self.processInput(features, target, self.firstNodes, isTraining)

