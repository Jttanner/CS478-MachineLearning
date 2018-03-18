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
        # Training set accuracy: 0.8797150720414441
        # Test set accuracy: 0.8082808280828083

        # No weights housing
        # Training Set: 0.8122875182127246
        # Training MSE: 0.07929055816379461
        # Test Set: 0.7770777077707771
        # Test MSE: 0.36331966529985876