from random import uniform


class Node:
    forwardConnections = []
    backConnections = []
    forwardWeights = []

    def __init__(self):
        self.forwardConnections = []
        self.backConnections = []
        self.forwardWeights = []


class OutputNode:
    value = None
    backConnections = []

    def __init__(self):
        pass

class BackpropNetwork:

    firstNodes = []
    outputs = []

    def __init__(self, numberOfLayers, features):
        self.firstNodes = []
        self.outputs = []
        for i in range(len(features)):
            self.firstNodes.append(Node())
            self.outputs.append(OutputNode())
        nextLayer = self.buildNetwork(numberOfLayers)
        for node in self.firstNodes:
            for nextNode in nextLayer:
                node.forwardConnections.append(nextNode)
        i = 4

    def buildNetwork(self, layersRemaining):
        if layersRemaining > 0:
            newLayer = []
            nextLayer = self.buildNetwork(layersRemaining - 1)
            for i in range(len(nextLayer)):
                newLayer.append(Node())
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