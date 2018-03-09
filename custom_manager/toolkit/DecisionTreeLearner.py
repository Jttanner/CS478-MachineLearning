from supervised_learner import SupervisedLearner
from DecisionTree import DecisionTree

import math

class DecisionTreeLearner(SupervisedLearner):

    tree = None
    maxFeatureValue = None

    def __init__(self):
        tree = None

    def train(self, features, labels):
        arrayFeatures = []
        arrayLabels = []
        maxFeature = 0
        for i in range(features.rows):
            for j in range(len(features.row(i))):
                if features.row(i)[j] > maxFeature and not math.isinf(features.row(i)[j]):
                    maxFeature = features.row(i)[j]
                    self.maxFeatureValue = maxFeature
        for i in range(features.rows):
            currFeature = []
            for j in range(len(features.row(i))):
                currFeature.append(features.row(i)[j] if not math.isinf(features.row(i)[j]) else maxFeature + 1)
            arrayFeatures.append(currFeature)
            arrayLabels.append(labels.row(i)[0])
        self.tree = DecisionTree(arrayFeatures, arrayLabels)

    def predict(self, feature, labels):
        infAlteredInput = []
        for entry in feature:
            if math.isinf(entry):
                infAlteredInput.append(self.maxFeatureValue + 1)
            else:
                infAlteredInput.append(entry)
        labels.append(self.tree.getDecision(infAlteredInput))