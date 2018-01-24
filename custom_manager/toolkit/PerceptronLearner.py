from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix
from Perceptron import Perceptron

class PerceptronLearner(SupervisedLearner):
    setosaPerceptron = Perceptron(0.0)
    versicolorPerceptron = Perceptron(1.0)
    virginicaPerceptron = Perceptron(2.0)

    def train(self, features, labels):

        for i in range(0,1):
            self.setosaPerceptron.train(features, labels)
            self.versicolorPerceptron.train(features, labels)
            self.virginicaPerceptron.train(features,labels)


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        self.setosaPerceptron.predict(features, labels)
        self.versicolorPerceptron.predict(features, labels)
        self.virginicaPerceptron.predict(features, labels)

        for index in range(0, len(self.setosaPerceptron.labels)):
            if(self.setosaPerceptron.labels[index] == 1):
                labels.append(0.0);
            elif (self.versicolorPerceptron.labels[index] == 1):
                labels.append(1.0);
            elif (self.virginicaPerceptron.labels[index] == 1):
                labels.append(2.0);