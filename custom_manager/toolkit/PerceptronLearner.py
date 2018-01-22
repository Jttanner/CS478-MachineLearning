from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix
from Perceptron import Perceptron

class PerceptronLearner(SupervisedLearner):
    setosaPerceptron = Perceptron(0.0)
    versicolorPerceptron = Perceptron(1.0)
    virginicaPerceptron = Perceptron(2.0)

    def train(self, features, labels):

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