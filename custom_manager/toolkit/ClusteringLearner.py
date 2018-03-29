from supervised_learner import SupervisedLearner
from KMeans import KMeans

class ClusteringLearner(SupervisedLearner):
    """
    types = ["kMeans", "HAC"]
    """
    k = 3

    def __init__(self):
        pass

    def train(self, features, labels):
        self.kMeans = KMeans(features, labels, self.k)
        self.kMeans.train()
        # self.hac = HAC(features, labels)

    def predict(self, features, labels):
        pass
