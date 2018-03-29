from supervised_learner import SupervisedLearner
from KNearestNeighbor import KNearestNeighbor

class InstanceBasedLearner(SupervisedLearner):

    knn= None
    k = 15
    rowsPredicted = 0
    regression = False

    def __init__(self):
        pass

    def train(self, features, labels):
        self.knn = KNearestNeighbor(features, labels, self.k, self.regression)

    def predict(self, row, labels):
        self.knn.calculateDistancesForDataRow(row)
        prediction = self.knn.nearestNeighborVote(row)
        labels.append(prediction)
        self.rowsPredicted += 1
