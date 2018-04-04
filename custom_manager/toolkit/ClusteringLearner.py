from supervised_learner import SupervisedLearner
from KMeans import KMeans
from matrix import Matrix
import numpy as np
import sys

class ClusteringLearner(SupervisedLearner):
    """
    types = ["kMeans", "HAC"]
    """
    k = 2

    def __init__(self):
        pass

    def train(self, features, labels):
        self.kMeans = KMeans(features, labels, self.k)
        self.kMeans.train()
        # self.hac = HAC(features, labels)

    def predict(self, features, labels):
        pass

file_name = sys.argv[1]
learner = ClusteringLearner()
data = Matrix()
data.load_arff(file_name)
# data.normalize()
features = Matrix(data, 0, 0, data.rows, data.cols - 1)
labels = Matrix(data, 0, data.cols - 1, data.rows, 1)
# features = Matrix(data, 0, 0, data.rows, data.cols)

# tempFeatures = []
# tempLabels = []
#
# for i in range(features.rows):
#     frow = []
#     lrow = []
#     for j in range(len(features.row(i))):
#         frow.append(features.row(i)[j])
#     lrow.append(labels.row(i)[0])
#     tempFeatures.append(frow)
#     tempLabels.append(lrow)
# features = np.array(tempFeatures)
# labels = np.array(tempLabels)
learner.train(features, labels)
# labelsForEachGroup = []
# for i in range(learner.k):
#     labelsForEachGroup.append([0,0,0,0])
# for i in range(len(learner.kMeans.groups)):
#     labelsForEachGroup[int(learner.kMeans.groups[i])][int(learner.kMeans.labels[i][0])] += 1
# i = 4