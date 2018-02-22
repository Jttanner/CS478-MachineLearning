from BackpropNetwork import BackpropNetwork
from supervised_learner import SupervisedLearner

class BackpropagationLearner(SupervisedLearner):
    learningRate = .1
    network = None
    currentAccuracy = None
    previousAccuaracy = None
    bestAccuaracy = None
    epochs = None
    epochsWithoutMeaningfulUpdate = None
    numberOfHiddenLayers = 1
    features = []
    labels = []
    accuracyDeltaCutoff = .01

    layerSizesArray = [4, 8, 3]

    def __init__(self):
        network = BackpropNetwork()
        self.bestAccuaracy = 0
        self.currentAccuracy = 0
        self.previousAccuaracy = 0
        self.epochs = 0
        self.epochsWithoutMeaningfulUpdate = 0

    #TODO: Implement once accuracy is measured during training
    def checkAccuracyForMeaningfulUpdate(self):
        oldBestAccuracy = self.bestAccuaracy
        self.bestAccuaracy = self.currentAccuracy if self.currentAccuracy > self.bestAccuaracy else self.bestAccuaracy
        if oldBestAccuracy > self.bestAccuaracy + self.accuracyDeltaCutoff:
            return False
        else:
            return True

    def train(self, features, labels):
        if self.network == None:
            self.network = BackpropNetwork(self.numberOfHiddenLayers, self.learningRate, self.layerSizesArray)
        else:
            while self.epochsWithoutMeaningfulUpdate < 5:
                features.shuffle(labels)
                correct = 0
                total = 0
                for feature, label in zip(features, labels):
                    result = True if self.network.predict(feature, label) == label else False
                    correct = correct + 1 if result == True else correct
                    total = total + 1
                self.previousAccuaracy = self.currentAccuracy
                self.currentAccuracy = correct/total
                self.epochs = self.epochs + 1
                self.epochsWithoutMeaningfulUpdate = \
                    self.epochsWithoutMeaningfulUpdate + 1 \
                    if self.checkAccuracyForMeaningfulUpdate() \
                    else self.epochsWithoutMeaningfulUpdate()




    def predict(self, features, labels):
        print('predict')