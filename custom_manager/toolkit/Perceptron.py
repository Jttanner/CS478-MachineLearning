from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix

class Perceptron():
    def __init__(self, type, inputSize):
        self.weights = [0] * (inputSize + 1)
        self.type = type

    type = 0
    labels = []
    weights = []
    learningRate = .1
    epochs = 0
    epochsWithoutChange = 0
    NUMBER_OF_EPOCHS_WITHOUT_CHANGE_TO_STOP = 5

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
        for i in range(0,len(input)):
            sum += input[i] * self.weights[i]
        return sum;

    def computeOutput(self, sum):
        if (sum > 0):
            return 1
        else:
            return 0

    def copyList(self, original, destination):
        for i in range(0, len(original)):
            destination.append(original[i])



    def checkAccuracyForMeaningfulUpdate(self, accuracy, oldAccuracy, weights, oldWeights):
        if (accuracy < oldAccuracy + .00001 and accuracy > oldAccuracy -.00001):
            self.epochsWithoutChange += 1
            if self.epochsWithoutChange > self.NUMBER_OF_EPOCHS_WITHOUT_CHANGE_TO_STOP:
                return False
            else:
                return True
        else:
            self.epochsWithoutChange = 0
            return True


#    def checkMeaningfulUpdate(self, weights, oldWeights, iterations):
#        if iterations > 200:
#            return False
#        elif weights == oldWeights:
#            return False
#        else:
#            changeAmount = 0
#            for i in range(0, len(weights)):
#                changeAmount += abs(weights[i] - oldWeights[i])
#            if changeAmount < .01:
#                return False
#            else:
#                return True

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.labels = []

        hasUpdated = True
        accuracy = 0
        oldAccuracy = 0
        while hasUpdated :
            oldWeights = []
            total = labels.rows
            correct = 0
            self.copyList(self.weights, oldWeights)
            for i in range(features.rows):
                inputWithBias = []
                self.copyList(features.row(i), inputWithBias)
                inputWithBias.append(1)
                net = self.computeNet(inputWithBias)
                output = self.computeOutput(net)
                self.updateWeights(self.learningRate, 1 if labels.row(i)[0] == self.type else 0, output, inputWithBias)
                if labels.row(i)[0] == self.type:
                    if output == 1:
                        correct += 1
            oldAccuracy = accuracy
            accuracy = correct / total
            hasUpdated = self.checkAccuracyForMeaningfulUpdate(accuracy, oldAccuracy, self.weights, oldWeights)
            #hasUpdated = self.checkMeaningfulUpdate(self.weights, oldWeights, self.epochs)
            self.epochs += 1
        print("number of epochs: " + str(self.epochs))

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        self.labels = []
        inputWithBias = []
        self.copyList(features, inputWithBias)
        inputWithBias.append(1)
        net = self.computeNet(inputWithBias)
        output = self.computeOutput(net)
        if output > 0:
            self.labels.append(1);
        else:
            self.labels.append(0);


