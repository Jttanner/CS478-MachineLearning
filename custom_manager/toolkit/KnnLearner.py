from supervised_learner import SupervisedLearner
from KNearestNeighbor import KNearestNeighbor

class InstanceBasedLearner(SupervisedLearner):

    knn= None
    k = 3
    rowsPredicted = 0
    regression = True

    def __init__(self):
        pass

    def train(self, features, labels):
        self.knn = KNearestNeighbor(features, labels, self.k, self.regression)

    def predict(self, row, labels):
        self.knn.calculateDistancesForDataRow(row)
        prediction = self.knn.nearestNeighborVote(row)
        labels.append(prediction)
        self.rowsPredicted += 1
