from BackpropNetwork import BackpropNetwork
from supervised_learner import SupervisedLearner

class BackpropagationLearner(SupervisedLearner):
    network = None
    currentAccuracy = None
    previousAccuaracy = None
    epochs = None
    epochsWithoutMeaningfulUpdate = None
    numberOfHiddenLayers = 1
    features = []
    labels = []
    def __init__(self):
        network = None
        self.currentAccuracy = 0
        self.previousAccuaracy = 0
        self.epochs = 0
        self.epochsWithoutMeaningfulUpdate = 0

    #TODO: Implement once accuracy is measured during training
    def checkAccuracyForMeaningfulUpdate(self):
        pass

    def train(self, features, labels):
        if self.network == None:
            self.network = BackpropNetwork(self.numberOfHiddenLayers, features)
        else:
            features.shuffle(labels)
            correct = 0
            total = 0
            for feature, label in zip(features, labels):
                result = self.network.predict(feature, label)
                correct = correct + 1 if result == True else correct
                total = total + 1
            self.epochs = self.epochs + 1
            self.epochsWithoutMeaningfulUpdate = \
                self.epochsWithoutMeaningfulUpdate + 1 \
                if self.checkAccuracyForMeaningfulUpdate() \
                else self.epochsWithoutMeaningfulUpdate
            



    def predict(self, features, labels):
        print('predict')