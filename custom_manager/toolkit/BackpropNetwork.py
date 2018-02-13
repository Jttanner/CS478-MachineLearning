from random import uniform


class Node:
    forwardConnections = []
    backConnections = []
    forwardWeights = []
    hidden = None

    def __init__(self, hidden):
        self.forwardConnections = []
        self.backConnections = []
        self.forwardWeights = []
        self.hidden = hidden


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
            self.firstNodes.append(Node(False))
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


    def updateWeights(self):
        pass

    #predict a single case
    #@return True/False if correct or not
    def predict(self, features, label):
        pass