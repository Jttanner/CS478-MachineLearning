

class KMeans:

    centroidIndexes = None

    def __init__(self, features, labels, k):
        self.features = features
        self.labels = labels
        self.k = k
        self.centroidIndexes = []
        self.pickInitialRandomCentroids()

    def pickInitialRandomCentroids(self):
        pass

