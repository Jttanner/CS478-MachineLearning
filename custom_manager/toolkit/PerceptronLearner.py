from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix
from Perceptron import Perceptron

from graphing import MLGraphing as Graph

class PerceptronLearner(SupervisedLearner):

    perceptronList = []


    def __init__(self):
        #votingPerceptron = Perceptron(1.0, 16)
        #self.perceptronList.append(votingPerceptron)
        #
        # smallPerceptron = Perceptron(1.0, 2)
        # self.perceptronList.append(smallPerceptron)

        setosaPerceptron = Perceptron(0.0, 4)
        versicolorPerceptron = Perceptron(1.0, 4)
        virginicaPerceptron = Perceptron(2.0, 4)
        self.perceptronList.append(setosaPerceptron)
        self.perceptronList.append(versicolorPerceptron)
        self.perceptronList.append(virginicaPerceptron)

    def train(self, features, labels):
        for i in range(0,len(self.perceptronList)):
            self.perceptronList[i].train(features, labels)


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        #labels = []
        for i in range(0,len(self.perceptronList)):
            self.perceptronList[i].predict(features, labels)


        # for i in range(0, len(self.perceptronList)):
        #    if self.perceptronList[i].labels[0] == 1:
        #        labels.append(self.perceptronList[i].type)
        #    else:
        #        if not len(self.perceptronList) > 1:
        #            labels.append(0)

        positiveOutputs = 0
        for i in range (len(self.perceptronList)):
            if self.perceptronList[i].labels[0] == 1:
                positiveOutputs += 1

        if positiveOutputs == 0 or positiveOutputs > 1:
            net1 = self.perceptronList[0].computeNet(features)
            net2 = self.perceptronList[1].computeNet(features)
            net3 = self.perceptronList[2].computeNet(features)
            if net1 > net2 and net1 > net3:
                self.perceptronList[0].labels[0] = 1
                self.perceptronList[1].labels[0] = 0
                self.perceptronList[2].labels[0] = 0
            elif net2 > net1 and net2 > net3:
                self.perceptronList[0].labels[0] = 0
                self.perceptronList[1].labels[0] = 1
                self.perceptronList[2].labels[0] = 0
            elif net3 > net1 and net3 > net2:
                self.perceptronList[0].labels[0] = 0
                self.perceptronList[1].labels[0] = 0
                self.perceptronList[2].labels[0] = 1

        if self.perceptronList[0].labels[0] == 1:
            labels.append(0.0)
        elif self.perceptronList[1].labels[0] == 1:
            labels.append(1.0)
        elif self.perceptronList[2].labels[0] == 1:
            labels.append(2.0)


    def measure_accuracy(self, features, labels, confusion=None):
        accuracy = super(PerceptronLearner, self).measure_accuracy(features, labels)
        #Graph.plotBinaryResults(Graph, features, labels, self.perceptronList[0].weights)
        #print(self.perceptronList[0].weights)
        #Graph.plotMisclassificationRate(self, self.perceptronList[0].accuracyAtEachEpoch, self.perceptronList[0].epochs);
        return accuracy