from __future__ import (absolute_import, division, print_function, unicode_literals)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matrix import Matrix
import math

# this is an abstract class


class SupervisedLearner:

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Matrix
        :type labels: Matrix
        """
        raise NotImplementedError()

    def predict(self, features, labels):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError

    def plotBinaryResults(self, features, labels):
        xAxisRed = []
        yAxisRed = []
        xAxisBlue = []
        yAxisBlue = []
        for i in range(features.rows):
            if labels.data[i][0] == 0.0:
                xAxisRed.append(features.data[i][0])
                yAxisRed.append(features.data[i][1])
            else:
                xAxisBlue.append(features.data[i][0])
                yAxisBlue.append(features.data[i][0])

        truePlot = plt.plot(xAxisRed, yAxisRed, 'ro')
        plt.setp(truePlot, 'color', 'r')
        falsePlot = plt.plot(xAxisBlue, yAxisBlue, 'ro')
        plt.setp(falsePlot, 'color', 'b')
        falsePlot[0].color = 'b'
        plt.xlabel("Happiness")
        plt.ylabel("Cuteness")
        catLegend = patches.Patch(color='red', label='Cat')
        dogLegend = patches.Patch(color='blue', label='Dog')
        plt.legend(handles=[catLegend, dogLegend])
        plt.show()
        return 0

    def measure_accuracy(self, features, labels, confusion=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: Matrix
        :type labels: Matrix
        :type confusion: Matrix
        :rtype float
        """

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        label_values_count = labels.value_count(0)
        if label_values_count == 0:
            # label is continuous
            pred = []
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                pred[0] = 0.0       # make sure the prediction is not biased by a previous prediction
                self.predict(feat, pred)
                delta = targ[0] - pred[0]
                sse += delta**2
            return math.sqrt(sse / features.rows)

        else:
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            for i in range(features.rows):
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                self.predict(feat, prediction)
                pred = int(prediction[i])
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred)+1)
                if pred == targ:
                    correct_count += 1

            self.plotBinaryResults(features, labels)

            return correct_count / features.rows





