from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix

class Perceptron():
    def __init__(self, type):
        self.weights = [0] * 5
        self.type = type

    type = 0
    labels = []
    weights = []
    learningRate = .1

    # deltaw = c(t-z)x
    # c=learning rate
    # t=target
    # z=output
    # x=input
    def updateWeights(self, learningRate, target, output, input):
        for i in range(0, len(self.weights)):
            self.weights[i] += learningRate * (target - output) * input[i]


    def computeNet(self, input):
        #input.append(1)
        sum = 0
        for i in range(0,5):
            sum += input[i] * self.weights[i]
        return sum;

    def computeOutput(self, sum):
        if (sum > 0):
            return 1
        else:
            return 0

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.labels = []

        hasUpdated = True
        while hasUpdated :
            oldweights = self.weights
            hasUpdated = 0
            for i in range(features.rows):
                inputWithBias = features.row(i)
                inputWithBias.append(1)
                net = self.computeNet(inputWithBias)
                output = self.computeOutput(net)
                testA = labels.row(i)[0]
                testB = self.type
                testC = 1 if labels.row(i)[0] else 0
                self.updateWeights(self.learningRate, 1 if labels.row(i)[0] == self.type else 0, output, inputWithBias)
                if oldweights == self.weights:
                    hasUpdate = False

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        net = self.computeNet(features)
        output = self.computeOutput(net)
        if output > 0:
            self.labels.append(0);
        #    return 1
        else:
            self.labels.append(1);
        #    return 0
