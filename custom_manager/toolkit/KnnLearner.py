from supervised_learner import SupervisedLearner
from KNearestNeighbor import KNearestNeighbor

class InstanceBasedLearner(SupervisedLearner):

    knn= None

    def __init__(self):
        pass

    def predict(self, features, labels):
        self.knn = KNearestNeighbor(features, labels)