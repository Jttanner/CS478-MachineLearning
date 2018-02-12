from BackpropNetwork import BackpropNetwork
from supervised_learner import SupervisedLearner

class BackpropagationLearner(SupervisedLearner):
    network = BackpropNetwork()
    features = []
    labels = []
    def __init__(self):
        print('init')

    def train(self, features, labels):
        if len(features) < 1:
            self.network.buildNetwork(features, labels)
        else:
            print('train')

    def predict(self, features, labels):
        print('predict')