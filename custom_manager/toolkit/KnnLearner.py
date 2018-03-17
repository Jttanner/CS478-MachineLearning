from supervised_learner import SupervisedLearner
from KNearestNeighbor import KNearestNeighbor

class InstanceBasedLearner(SupervisedLearner):

    knn= None
    k = 3
    rowsPredicted = 0

    def __init__(self):
        pass

    def train(self, features, labels):
        self.knn = KNearestNeighbor(features, labels, self.k)

    def predict(self, row, labels):
        self.knn.calculateDistancesForDataRow(row)
        prediction = self.knn.nearestNeighborVote(row)
        labels.append(prediction)
        self.rowsPredicted += 1


        # No weights magic telescope
        # Training set accuracy: 1.0
        # Test set accuracy: 0.7856285628562857

