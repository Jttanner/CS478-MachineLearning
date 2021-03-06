from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix
from Perceptron import Perceptron

class PerceptronLearner(SupervisedLearner):
    #setosaPerceptron = Perceptron(0.0, 4)
    #versicolorPerceptron = Perceptron(1.0, 4)
    #virginicaPerceptron = Perceptron(2.0, 4)

    perceptronList = []

    def __init__(self):
        smallPerceptron = Perceptron(1.0, 2)
        self.perceptronList.append(smallPerceptron)
        #setosaPerceptron = Perceptron(0.0, 4)
        #versicolorPerceptron = Perceptron(1.0, 4)
        #virginicaPerceptron = Perceptron(2.0, 4)
        #self.perceptronList.append(setosaPerceptron)
        #self.perceptronList.append(versicolorPerceptron)
        #self.perceptronList.append(virginicaPerceptron)

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


        for i in range(0, len(self.perceptronList)):
            if self.perceptronList[i].labels[0] == 1:
                labels.append(1)
            else:
                labels.append(0)

        #if self.setosaPerceptron.labels[0] == 1:
        #    labels.append(0.0)
        #elif self.versicolorPerceptron.labels[0] == 1:
        #    labels.append(1.0)
        #elif self.virginicaPerceptron.labels[0] == 1:
        #    labels.append(2.0)

        #for index in range(0, len(labels)):
        #    if(self.setosaPerceptron.labels[index] == 1):
        #        labels.append(0.0);
        #    elif (self.versicolorPerceptron.labels[index] == 1):
        #        labels.append(1.0);
        #    elif (self.virginicaPerceptron.labels[index] == 1):
        #        labels.append(2.0);