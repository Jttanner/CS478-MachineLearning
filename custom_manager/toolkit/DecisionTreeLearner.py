from supervised_learner import SupervisedLearner
from DecisionTree import DecisionTree

class DecisionTreeLearner(SupervisedLearner):

    tree = None

    def __init__(self):
        tree = None

    def train(self, features, labels):
        arrayFeatures = []
        arrayLabels = []
        for i in range(features.rows):
            arrayFeatures.append(features.row(i))
            arrayLabels.append(labels.row(i)[0])
        self.tree = DecisionTree(arrayFeatures, arrayLabels)

    def predict(self, feature, labels):
        labels.append(self.tree.getDecision(feature))