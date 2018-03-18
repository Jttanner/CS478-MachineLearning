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

    # no weights regression on housing
    # Dataset
    # name: knnData / housingTraining.arff
    # Number
    # of
    # instances: 12354
    # Number
    # of
    # attributes: 11
    # Learning
    # algorithm: knn
    # Evaluation
    # method: static
    #
    # Calculating
    # accuracy
    # on
    # separate
    # test
    # set...
    # Test
    # set
    # name: knnData / housingTest.arff
    # Number
    # of
    # test
    # instances: 6666
    # Time
    # to
    # train( in seconds): 0.02707386016845703
    # Backend
    # TkAgg is interactive
    # backend.Turning
    # interactive
    # mode
    # on.
    # Training
    # set
    # accuracy: 0.6452161243322001
    # Test
    # set
    # accuracy: 0.4825982598259826