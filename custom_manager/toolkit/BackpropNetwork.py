from random import uniform


class Node:
    forwardConnections = []
    backConnections = []
    forwardWeights = []

    def __init__(self):
        self.initWeights()

    def __init__(self, backConnections, forwardConnections):
        self.forwardConnections = forwardConnections
        self.backConnections = backConnections
        self.initWeights()

    def initWeights(self):
        for connection in self.forwardConnections:
            #TODO: small random weights with a mean of 0.  Slack said just from -.1 to .1, but double check
            initialWeight = uniform(-.1, .1)
            while initialWeight == 0: #making sure that the weights can never initalize to 0, but stay close
                initialWeight = uniform(-.1, .1)
            self.forwardWeights.append(initialWeight)

    def getConnectionSize(self):
        return len(self.forwardWeights)

class OutputNode:
    value = None

    def __init__(self):
        pass

class BackpropNetwork:

    firstNodes = []
    outputs = []

    def __init__(self, numberOfLayers, features):
        for i in range(len(features)):
            newNode = Node()
            self.firstNodes.append(newNode)
        print('test')
        #TODO: Build nodes from info's parameters

    def buildNetwork(self, features, layersRemaining, lastNode):
        if layersRemaining > 0:
            for i in range(len(features)):
                newNode = Node()
                lastNode.forwardConnections.append(newNode)
                self.buildNetwork(self,features, layersRemaining - 1, newNode)
        else:
            lastNode.forwardConnections.append(OutputNode())
