from BackpropNetwork import BackpropNetwork
from supervised_learner import SupervisedLearner

class BackpropagationLearner(SupervisedLearner):
    learningRate = .1
    network = None
    bestNetworkSoFar = None
    currentAccuracy = None
    previousAccuaracy = None
    bestAccuaracy = None
    epochs = None
    epochsWithoutMeaningfulUpdate = None
    numberOfHiddenLayers = 1
    features = []
    labels = []
    accuracyDeltaCutoff = .01
    validationSet = None

    layerSizesArray = [4, 8, 3]

    def __init__(self):
        self.network = BackpropNetwork(self.numberOfHiddenLayers + 2, self.learningRate, self.layerSizesArray)
        self.bestNetworkSoFar = None
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
            self.bestNetworkSoFar = BackpropNetwork(self.network)
            return True

    def train(self, features, labels):
        # if self.network == None:
        #     #layers + 2 to account for input and output layers
        #     self.network = BackpropNetwork(self.numberOfHiddenLayers + 2, self.learningRate, self.layerSizesArray)
        # else:
        while self.epochsWithoutMeaningfulUpdate < 5:
            features.shuffle(labels)
            correct = 0
            total = 0
            for i in range(features.rows):
                input = features.row(i)
                correctAnswer = labels.row(i)
                result = self.predict(input, [])
                correct += 1 if result == correctAnswer else 0
                total = 0
            self.previousAccuaracy = self.currentAccuracy
            self.currentAccuracy = correct/total
            self.epochs = self.epochs + 1
            self.epochsWithoutMeaningfulUpdate +=  \
                1 if self.checkAccuracyForMeaningfulUpdate() else 0



    def predict(self, features, labels):
        targets = []
        if labels != []:
            targets.append(labels[0])
        labels = []
        self.network.processInput(features, targets, False)
        output = 0
        for outputNode, i in zip(self.network.outputs, range(len(self.network.outputs))):
            if outputNode.output > outputNode:
                output = i
        labels.append(output)
        return output